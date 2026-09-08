"""Startup-only NPU covariance and symmetric eigensolver, implemented in Triton.

No raw Q/K/V is dumped to the host. Only convergence scalars and final rotation
artifacts leave the device. The covariance objectives follow OSCAR qqt/sst.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _gram(
    X,
    W,
    C,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    ST: tl.constexpr,
    SH: tl.constexpr,
    SD: tl.constexpr,
    WEIGHTED: tl.constexpr,
    BM: tl.constexpr = 16,
    BT: tl.constexpr = 32,
):
    i = tl.program_id(0) * BM + tl.arange(0, BM)
    j = tl.program_id(1) * BM + tl.arange(0, BM)
    t = tl.arange(0, BT)
    total = tl.full((BM, BM), 0, tl.float32)
    for start in range(tl.cdiv(N * H, BT)):
        rows = start * BT + t
        left = tl.load(
            X + rows[:, None] // H * ST + rows[:, None] % H * SH + i[None, :] * SD,
            rows[:, None] < N * H,
            0,
        )
        right = tl.load(
            X + rows[:, None] // H * ST + rows[:, None] % H * SH + j[None, :] * SD,
            rows[:, None] < N * H,
            0,
        )
        if WEIGHTED:
            weights = tl.load(W + rows, rows < N * H, 0)
            total += tl.sum(
                left.to(tl.float32)[:, :, None]
                * right.to(tl.float32)[:, None, :]
                * weights[:, None, None],
                0,
            )
        else:
            total += tl.dot(tl.trans(left.to(tl.bfloat16)), right.to(tl.bfloat16))
    address = i[:, None] * D + j[None, :]
    tl.store(C + address, tl.load(C + address) + total)


def accumulate_gram(x, covariance, weights=None):
    n, heads, dim = x.shape
    _gram[(triton.cdiv(dim, 16), triton.cdiv(dim, 16))](
        x,
        weights if weights is not None else covariance,
        covariance,
        n,
        heads,
        dim,
        *x.stride(),
        weights is not None,
    )


@triton.jit
def _quadratic_project(
    K,
    C,
    KC,
    N: tl.constexpr,
    D: tl.constexpr,
    ST: tl.constexpr,
    SD: tl.constexpr,
    BN: tl.constexpr = 16,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, D)
    k = tl.load(K + row * ST + inner * SD).to(tl.float32)
    covariance = tl.load(C + inner[:, None] * D + cols[None, :])
    result = tl.sum(k[:, None] * covariance, 0)
    tl.store(KC + row * D + cols, result)


@triton.jit
def _quadratic_finish(K, KC, W, D: tl.constexpr, ST: tl.constexpr, SD: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, D)
    k = tl.load(K + row * ST + d * SD).to(tl.float32)
    kc = tl.load(KC + row * D + d)
    tl.store(W + row, tl.maximum(tl.sum(k * kc, 0), 0.0))


@triton.jit
def _sum_weights(W, Total, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    values = tl.load(W + i, i < N, 0)
    tl.store(Total, tl.load(Total) + tl.sum(values, 0))


def sst_weights(key, q_covariance, total_weight):
    n, heads, dim = key.shape
    if heads != 1:
        raise ValueError("Calibration requires TP4 with one KV head per rank")
    projected = torch.empty((n, dim), device=key.device, dtype=torch.float32)
    weights = torch.empty(n, device=key.device, dtype=torch.float32)
    _quadratic_project[(n, triton.cdiv(dim, 16))](
        key,
        q_covariance,
        projected,
        n,
        dim,
        key.stride(0),
        key.stride(2),
    )
    _quadratic_finish[(n,)](key, projected, weights, dim, key.stride(0), key.stride(2))
    _sum_weights[(1,)](weights, total_weight, n, triton.next_power_of_2(n))
    return weights


@triton.jit
def _divide(
    X,
    Denominator,
    Out,
    N: tl.constexpr,
    VALUE: tl.constexpr,
    DEVICE_DENOM: tl.constexpr,
    BLOCK: tl.constexpr = 1024,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    denominator = tl.load(Denominator) if DEVICE_DENOM else VALUE
    value = tl.load(X + i, i < N, 0) / tl.maximum(denominator, 1.0e-30)
    tl.store(Out + i, value, i < N)


def normalize(x, denominator):
    out = torch.empty_like(x)
    device_denominator = isinstance(denominator, torch.Tensor)
    _divide[(triton.cdiv(x.numel(), 1024),)](
        x,
        denominator if device_denominator else x,
        out,
        x.numel(),
        1.0 if device_denominator else float(denominator),
        device_denominator,
    )
    return out


@triton.jit
def _initialize(C, A, U, D: tl.constexpr, BLOCK: tl.constexpr = 1024):
    batch = tl.program_id(0)
    flat = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    row, col = flat // D, flat % D
    value = tl.load(C + batch * D * D + flat, flat < D * D, 0)
    transpose = tl.load(C + batch * D * D + col * D + row, flat < D * D, 0)
    tl.store(A + batch * D * D + flat, (value + transpose) * 0.5, flat < D * D)
    tl.store(U + batch * D * D + flat, (row == col).to(tl.float32), flat < D * D)


@triton.jit
def _pairs(j, round_index, D: tl.constexpr):
    p = tl.where(j == 0, D - 1, (round_index + j) % (D - 1))
    q = tl.where(j == 0, round_index, (round_index - j + D - 1) % (D - 1))
    return p, q


@triton.jit
def _jacobi_columns(
    A, U, B, NextU, CS, ROUND, D: tl.constexpr, BR: tl.constexpr = 32, BP: tl.constexpr = 16
):
    batch = tl.program_id(0)
    rows = tl.program_id(1) * BR + tl.arange(0, BR)
    pair = tl.program_id(2) * BP + tl.arange(0, BP)
    p, q = _pairs(pair, ROUND, D)
    base = batch * D * D
    app = tl.load(A + base + p * D + p)
    aqq = tl.load(A + base + q * D + q)
    apq = tl.load(A + base + p * D + q)
    delta = (aqq - app) * 0.5
    sign = tl.where(delta >= 0, 1.0, -1.0)
    tangent = sign * apq / tl.maximum(tl.abs(delta) + tl.sqrt(delta * delta + apq * apq), 1e-30)
    tangent = tl.where(tl.abs(apq) > 1e-8 * (tl.abs(app) + tl.abs(aqq)) + 1e-20, tangent, 0.0)
    cosine = tl.rsqrt(1.0 + tangent * tangent)
    sine = tangent * cosine
    ap = tl.load(A + base + rows[:, None] * D + p[None, :])
    aq = tl.load(A + base + rows[:, None] * D + q[None, :])
    up = tl.load(U + base + rows[:, None] * D + p[None, :])
    uq = tl.load(U + base + rows[:, None] * D + q[None, :])
    tl.store(B + base + rows[:, None] * D + p[None, :], ap * cosine - aq * sine)
    tl.store(B + base + rows[:, None] * D + q[None, :], ap * sine + aq * cosine)
    tl.store(NextU + base + rows[:, None] * D + p[None, :], up * cosine - uq * sine)
    tl.store(NextU + base + rows[:, None] * D + q[None, :], up * sine + uq * cosine)
    if tl.program_id(1) == 0:
        tl.store(CS + batch * D + pair * 2, cosine)
        tl.store(CS + batch * D + pair * 2 + 1, sine)


@triton.jit
def _jacobi_rows(B, A, CS, ROUND, D: tl.constexpr, BC: tl.constexpr = 32, BP: tl.constexpr = 16):
    batch = tl.program_id(0)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    pair = tl.program_id(2) * BP + tl.arange(0, BP)
    p, q = _pairs(pair, ROUND, D)
    cosine = tl.load(CS + batch * D + pair * 2)
    sine = tl.load(CS + batch * D + pair * 2 + 1)
    bp = tl.load(B + batch * D * D + p[:, None] * D + cols[None, :])
    bq = tl.load(B + batch * D * D + q[:, None] * D + cols[None, :])
    tl.store(
        A + batch * D * D + p[:, None] * D + cols[None, :],
        cosine[:, None] * bp - sine[:, None] * bq,
    )
    tl.store(
        A + batch * D * D + q[:, None] * D + cols[None, :],
        sine[:, None] * bp + cosine[:, None] * bq,
    )


@triton.jit
def _residual(A, Stats, D: tl.constexpr, BR: tl.constexpr = 32):
    batch, tile = tl.program_id(0), tl.program_id(1)
    rows = tile * BR + tl.arange(0, BR)
    cols = tl.arange(0, D)
    values = tl.abs(tl.load(A + batch * D * D + rows[:, None] * D + cols[None, :]))
    off = tl.max(tl.max(tl.where(rows[:, None] == cols[None, :], 0.0, values), 0), 0)
    diag = tl.max(tl.max(tl.where(rows[:, None] == cols[None, :], values, 0.0), 0), 0)
    tl.store(Stats + (batch * (D // BR) + tile) * 2, off)
    tl.store(Stats + (batch * (D // BR) + tile) * 2 + 1, diag)


@triton.jit
def _sort_eigenpairs(A, U, SortedU, Eigenvalues, D: tl.constexpr):
    batch, source = tl.program_id(0), tl.program_id(1)
    offsets = tl.arange(0, D)
    value = tl.load(A + batch * D * D + source * D + source)
    others = tl.load(A + batch * D * D + offsets * D + offsets)
    rank = tl.sum(((others < value) | ((others == value) & (offsets < source))).to(tl.int32), 0)
    column = tl.load(U + batch * D * D + offsets * D + source)
    tl.store(SortedU + batch * D * D + offsets * D + rank, column)
    tl.store(Eigenvalues + batch * D + rank, value)


@triton.jit
def _residual_summary(Stats, Summary, BATCH: tl.constexpr, TILES: tl.constexpr, BB: tl.constexpr):
    batches = tl.arange(0, BB)
    tiles = tl.arange(0, TILES)
    offsets = (batches[:, None] * TILES + tiles[None, :]) * 2
    off = tl.load(Stats + offsets, batches[:, None] < BATCH, 0)
    diag = tl.load(Stats + offsets + 1, batches[:, None] < BATCH, 0)
    invalid = (
        (off != off)
        | (diag != diag)
        | (tl.abs(off) == float("inf"))
        | (tl.abs(diag) == float("inf"))
    )
    bad = tl.sum(tl.sum(invalid.to(tl.int32), 1), 0) > 0
    ratio = tl.max(off, 1) / tl.maximum(tl.max(diag, 1), 1.0e-30)
    tl.store(Summary, tl.where(bad, float("inf"), tl.max(ratio, 0)))


@triton.jit
def _hadamard_compose(U, R, D: tl.constexpr, BITS: tl.constexpr):
    batch, row = tl.program_id(0), tl.program_id(1)
    offsets = tl.arange(0, D)
    values = tl.load(U + batch * D * D + row * D + offsets)
    for bit in tl.static_range(BITS):
        partner = tl.gather(values, offsets ^ (1 << bit), 0)
        values = tl.where((offsets & (1 << bit)) == 0, values + partner, partner - values)
    # Eigenvectors were sorted ascending, so sorted_idx(descending)=D-1-i.
    # Official Pbr has perm[bit_reverse(i)] = sorted_idx[i].
    reversed_index = tl.full((D,), 0, tl.int32)
    source = D - 1 - offsets
    for bit in tl.static_range(BITS):
        reversed_index |= ((source >> bit) & 1) << (BITS - 1 - bit)
    tl.store(R + batch * D * D + row * D + reversed_index, values / tl.sqrt(float(D)))


def calibrated_rotations(covariances, max_sweeps=12, tolerance=1e-6):
    """Batched cyclic Jacobi + U H Pbr, all matrix arithmetic stays on the NPU."""
    if covariances.device.type != "npu" or covariances.dtype != torch.float32:
        raise ValueError("Calibration eigensolver requires FP32 NPU covariances")
    batch, dim, last = covariances.shape
    if dim != 256 or last != dim:
        raise ValueError("Calibration eigensolver is scoped to D256")
    a, b = torch.empty_like(covariances), torch.empty_like(covariances)
    u, next_u = torch.empty_like(a), torch.empty_like(a)
    cs = torch.empty((batch, dim), dtype=torch.float32, device=a.device)
    stats = torch.empty((batch, dim // 32, 2), dtype=torch.float32, device=a.device)
    summary = torch.empty((), dtype=torch.float32, device=a.device)
    _initialize[(batch, triton.cdiv(dim * dim, 1024))](covariances, a, u, dim)
    residual = float("inf")
    for sweep in range(max_sweeps):
        for round_index in range(dim - 1):
            _jacobi_columns[(batch, dim // 32, dim // 32)](a, u, b, next_u, cs, round_index, dim)
            _jacobi_rows[(batch, dim // 32, dim // 32)](b, a, cs, round_index, dim)
            u, next_u = next_u, u
        if (sweep + 1) % 2 == 0 or sweep + 1 == max_sweeps:
            _residual[(batch, dim // 32)](a, stats, dim)
            _residual_summary[(1,)](stats, summary, batch, dim // 32, triton.next_power_of_2(batch))
            # Only the final stop/continue scalar crosses to the host.
            residual = summary.item()
            print(
                f"[OSCAR calibration] Jacobi sweep {sweep + 1}: residual={residual:.3e}", flush=True
            )
            if not math.isfinite(residual):
                raise RuntimeError("Nonfinite calibration covariance; rotations not published")
            if residual < tolerance:
                break
    if not residual < tolerance:
        raise RuntimeError(
            f"Calibration eigensolver did not converge: {residual}; rotations not published"
        )
    sorted_u = torch.empty_like(u)
    eigenvalues = torch.empty((batch, dim), dtype=torch.float32, device=u.device)
    rotations = torch.empty_like(u)
    _sort_eigenpairs[(batch, dim)](a, u, sorted_u, eigenvalues, dim)
    _hadamard_compose[(batch, dim)](sorted_u, rotations, dim, 8)
    return rotations, eigenvalues, {"sweeps": sweep + 1, "relative_off_diagonal": residual}
