"""Triton Ascend kernels. There is deliberately no CPU execution fallback.

Only current-token tensors and split output accumulators are materialized.
History is unpacked tile by tile inside attention, never into a context-size KV.
"""

import torch
import triton
import triton.language as tl

from .config import attention_task_groups, percentile_selection


@triton.jit(do_not_specialize=["N"])
def _rotate_kernel(
    X,
    R,
    Y,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    XT: tl.constexpr,
    XH: tl.constexpr,
    XD: tl.constexpr,
    RT: tl.constexpr,
    RC: tl.constexpr,
    BM: tl.constexpr = 16,
    BN: tl.constexpr = 32,
    BK: tl.constexpr = 32,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(D, BK)):
        ds = start * BK + inner
        x = tl.load(
            X + (rows[:, None] // H) * XT + (rows[:, None] % H) * XH + ds[None, :] * XD,
            (rows[:, None] < N * H) & (ds[None, :] < D),
            0,
        ).to(tl.bfloat16)
        r = tl.load(
            R + ds[:, None] * RT + cols[None, :] * RC, (ds[:, None] < D) & (cols[None, :] < D), 0
        ).to(tl.bfloat16)
        acc += tl.dot(x, r)
    tl.store(
        Y + rows[:, None] * D + cols[None, :], acc, (rows[:, None] < N * H) & (cols[None, :] < D)
    )


def rotate(x, rotation):
    """Dense calibrated rotation; operands/accumulator are BF16/FP32 on Cube."""
    n, h, d = x.shape
    out = torch.empty((n, h, d), device=x.device, dtype=torch.bfloat16)
    if n:
        _rotate_kernel[(triton.cdiv(n * h, 16), triton.cdiv(d, 32))](
            x,
            rotation,
            out,
            n,
            h,
            d,
            *x.stride(),
            *rotation.stride(),
        )
    return out


@triton.jit
def _clip_vec(
    x,
    D: tl.constexpr,
    MODE: tl.constexpr,
    ENABLED: tl.constexpr,
    LOW_RANK: tl.constexpr,
    HIGH_RANK: tl.constexpr,
    WEIGHT: tl.constexpr,
    FACTOR: tl.constexpr,
):
    if ENABLED:
        absolute = tl.abs(x)
        if MODE == "factor":
            threshold = tl.max(absolute, 0) * FACTOR
        else:
            offsets = tl.arange(0, D)
            low_value = tl.full((), 0, tl.float32)
            high_value = tl.full((), 0, tl.float32)
            # Keep a bounded loop instead of cloning each dependent reduction
            # into the Ascend compiler IR (11 K + 22 V iterations by default).
            for rank in range(LOW_RANK + 1):
                maximum = tl.max(absolute, 0)
                if rank == LOW_RANK:
                    low_value = maximum
                if rank == HIGH_RANK:
                    high_value = maximum
                # Remove ONE element, including when the maximum is tied.
                winner = tl.min(tl.where(absolute == maximum, offsets, D), 0)
                absolute = tl.where(offsets == winner, -float("inf"), absolute)
            threshold = low_value + (high_value - low_value) * WEIGHT
        x = tl.minimum(tl.maximum(x, -threshold), threshold)
    return x


@triton.jit
def _store_vec(
    Src,
    Cache,
    row,
    address,
    D: tl.constexpr,
    MODE: tl.constexpr,
    ENABLED: tl.constexpr,
    LOW_RANK: tl.constexpr,
    HIGH_RANK: tl.constexpr,
    WEIGHT: tl.constexpr,
    FACTOR: tl.constexpr,
):
    offsets = tl.arange(0, D)
    x = tl.load(Src + row * D + offsets).to(tl.float32)
    x = _clip_vec(x, D, MODE, ENABLED, LOW_RANK, HIGH_RANK, WEIGHT, FACTOR)
    minimum = tl.min(x, 0)
    maximum = tl.max(x, 0)
    # A normal FP16 lower bound also avoids device-specific subnormal flushing.
    scale = tl.maximum((maximum - minimum) / 3.0, 0.00006103515625).to(tl.float16)
    zero = minimum.to(tl.float16)
    q = tl.minimum(
        tl.maximum(((x - zero.to(tl.float32)) / scale.to(tl.float32) + 0.5).to(tl.int32), 0), 3
    )
    codes = tl.reshape(q, (D // 4, 4))
    packed = tl.sum(codes << (tl.arange(0, 4)[None, :] * 2), 1).to(tl.uint8)
    tl.store(Cache + address + tl.arange(0, D // 4), packed)
    scale_bits = scale.to(tl.uint16, bitcast=True)
    zero_bits = zero.to(tl.uint16, bitcast=True)
    tl.store(Cache + address + D // 4, scale_bits.to(tl.uint8))
    tl.store(Cache + address + D // 4 + 1, (scale_bits >> 8).to(tl.uint8))
    tl.store(Cache + address + D // 4 + 2, zero_bits.to(tl.uint8))
    tl.store(Cache + address + D // 4 + 3, (zero_bits >> 8).to(tl.uint8))


@triton.jit
def _store_kernel(
    K,
    V,
    Cache,
    Slots,
    Counts,
    H: tl.constexpr,
    D: tl.constexpr,
    BP: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    SLOT: tl.constexpr,
    VECTOR: tl.constexpr,
    MODE: tl.constexpr,
    KE: tl.constexpr,
    KL: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    VE: tl.constexpr,
    VL: tl.constexpr,
    VH: tl.constexpr,
    VW: tl.constexpr,
    KF: tl.constexpr,
    VF: tl.constexpr,
):
    row = tl.program_id(0)
    token, head = row // H, row % H
    active = tl.load(Counts + 1)
    slot = tl.load(Slots + token, token < active, -1).to(tl.int64)
    if slot >= 0:
        address = slot // BP * PAGE_BYTES + (slot % BP * H + head) * SLOT
        _store_vec(K, Cache, row, address, D, MODE, KE, KL, KH, KW, KF)
        _store_vec(V, Cache, row, address + VECTOR, D, MODE, VE, VL, VH, VW, VF)


def store_int2(key_rot, value_rot, history, slots, counts, layout):
    cfg = layout.config
    kl, kh, kw = percentile_selection(cfg.k_clip_ratio)
    vl, vh, vw = percentile_selection(cfg.v_clip_ratio)
    _store_kernel[(key_rot.shape[0] * layout.kv_heads,)](
        key_rot,
        value_rot,
        history,
        slots,
        counts,
        layout.kv_heads,
        cfg.head_dim,
        layout.physical_block_tokens,
        layout.stripe_page_bytes,
        cfg.slot_bytes,
        cfg.vector_bytes,
        cfg.clip_mode,
        cfg.k_clip_ratio > 0 if cfg.clip_mode == "percentile" else True,
        kl,
        kh,
        kw,
        cfg.v_clip_ratio > 0 if cfg.clip_mode == "percentile" else True,
        vl,
        vh,
        vw,
        cfg.k_clip_factor,
        cfg.v_clip_factor,
    )


@triton.jit
def _load_vec(Cache, address, valid, D: tl.constexpr, BN: tl.constexpr):
    # Keep the whole unpack/dequant chain 2-D. A [BN,D/4,4] expansion makes
    # Ascend pad its narrow tail axis: the reported [32,64,16] BF16 temporary
    # alone consumes 64 KiB, before transpose and attention workspaces.
    dims = tl.arange(0, D)
    data = tl.load(Cache + address[:, None] + dims[None, :] // 4, valid[:, None], 0).to(tl.int32)
    low = tl.load(Cache + address + D // 4, valid, 0).to(tl.uint16)
    high = tl.load(Cache + address + D // 4 + 1, valid, 0).to(tl.uint16)
    scale = (low | (high << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    low = tl.load(Cache + address + D // 4 + 2, valid, 0).to(tl.uint16)
    high = tl.load(Cache + address + D // 4 + 3, valid, 0).to(tl.uint16)
    zero = (low | (high << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    codes = (data >> (2 * (dims[None, :] % 4))) & 3
    return (codes * scale[:, None] + zero[:, None]).to(tl.bfloat16)


@triton.jit(do_not_specialize=["COPY_COLS"])
def _prepare_tasks_kernel(
    QStarts,
    Counts,
    Tasks,
    TaskCount,
    QT: tl.constexpr,
    MAX_REQS: tl.constexpr,
    BLOCK: tl.constexpr = 32,
    SourceTable=None,
    Table=None,
    COPY_COLS=0,
    SOURCE_STRIDE: tl.constexpr = 0,
    TABLE_STRIDE: tl.constexpr = 0,
    COPY_TABLE: tl.constexpr = False,
):
    req = tl.program_id(0)
    nr = tl.load(Counts)
    active = tl.load(Counts + 1)
    if COPY_TABLE:
        for begin_col in range(0, COPY_COLS, 256):
            cols = begin_col + tl.arange(0, 256)
            valid = (req < nr) & (cols < COPY_COLS)
            blocks = tl.load(SourceTable + req * SOURCE_STRIDE + cols, valid, 0)
            tl.store(Table + req * TABLE_STRIDE + cols, blocks, valid)
    reqs = tl.arange(0, MAX_REQS)
    starts = tl.minimum(tl.load(QStarts + reqs, reqs < nr, 0), active)
    ends = tl.minimum(tl.load(QStarts + reqs + 1, reqs < nr, 0), active)
    tiles = tl.cdiv(tl.maximum(ends - starts, 0), QT)
    base = tl.sum(tl.where(reqs < req, tiles, 0), 0)
    count = tl.sum(tl.where(reqs == req, tiles, 0), 0)
    if req == 0:
        tl.store(TaskCount, tl.sum(tiles, 0))
    for begin in range(0, count, BLOCK):
        offsets = begin + tl.arange(0, BLOCK)
        valid = offsets < count
        tl.store(Tasks + (base + offsets) * 2, req, valid)
        tl.store(Tasks + (base + offsets) * 2 + 1, offsets * QT, valid)


def prepare_tasks(
    qstarts,
    counts,
    tasks,
    task_count,
    max_reqs,
    query_tile,
    num_reqs=None,
    source_table=None,
    table=None,
    copy_columns=0,
):
    _prepare_tasks_kernel[(max(1, max_reqs if num_reqs is None else num_reqs),)](
        qstarts,
        counts,
        tasks,
        task_count,
        query_tile,
        triton.next_power_of_2(max_reqs),
        SourceTable=source_table,
        Table=table,
        COPY_COLS=copy_columns,
        SOURCE_STRIDE=source_table.stride(0) if source_table is not None else 0,
        TABLE_STRIDE=table.stride(0) if table is not None else 0,
        COPY_TABLE=source_table is not None,
    )


@triton.jit
def _attention_kernel(
    Query,
    CurrentK,
    CurrentV,
    History,
    Window,
    QStarts,
    SeqLens,
    Table,
    Counts,
    Tasks,
    TaskCount,
    Partials,
    LSE,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    QT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    TASK_GROUPS: tl.constexpr,
    BK: tl.constexpr,
    BP: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    SLOT: tl.constexpr,
    VECTOR: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    SINK: tl.constexpr,
    RECENT: tl.constexpr,
    RING: tl.constexpr,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    QS2: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VS2: tl.constexpr,
    SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    RAW: tl.constexpr,
):
    total_tasks = tl.load(TaskCount)
    first_task = tl.program_id(0)
    # Bound physical launches and walk the compact task list on device. The
    # one masked iteration for empty lanes retains the balanced CV path.
    for task in range(first_task, tl.maximum(first_task + 1, total_tasks), TASK_GROUPS):
        live = task < total_tasks
        req = tl.load(Tasks + task * 2, live, 0)
        query_offset = tl.load(Tasks + task * 2 + 1, live, 0)
        head = tl.program_id(1)
        split = tl.program_id(2)
        qstart = tl.load(QStarts + req, live, 0)
        qend = tl.minimum(tl.load(QStarts + req + 1, live, 0), tl.load(Counts + 1))
        qlen = qend - qstart
        seq = tl.load(SeqLens + req, live, 0)
        prefix = seq - qlen
        row = tl.arange(0, BM)
        qi = query_offset + row // G
        qhead = head * G + row % G
        token = qstart + qi
        row_valid = live & (row < QT * G) & (qi < qlen)
        dims = tl.arange(0, D)
        q = tl.load(
            Query + token[:, None] * QS0 + qhead[:, None] * QS1 + dims[None, :] * QS2,
            row_valid[:, None],
            0,
        ).to(tl.bfloat16)
        sink_len = tl.minimum(SINK, prefix)
        recent_begin = tl.maximum(sink_len, prefix - RECENT)
        if RAW:
            first_block = tl.load(Table + req * TABLE_STRIDE, live, 0).to(tl.int64)
            owner = first_block // (BP // BK)
            length = sink_len + prefix - recent_begin + qlen
            lo = 0
        else:
            lo = sink_len
            length = recent_begin - lo
        tiles_per_split = tl.cdiv(tl.cdiv(length, BN), SPLITS)
        begin = split * tiles_per_split * BN
        end = tl.minimum((split + 1) * tiles_per_split * BN, length)
        m = tl.full((BM,), -float("inf"), tl.float32)
        denominator = tl.full((BM,), 0, tl.float32)
        acc = tl.full((BM, D), 0, tl.float32)
        for start in range(begin, tl.maximum(begin + BN, end), BN):
            ns = start + tl.arange(0, BN)
            valid = live & (ns < end)
            if RAW:
                high_prefix_len = sink_len + prefix - recent_begin
                pos = tl.where(
                    ns < sink_len,
                    ns,
                    tl.where(
                        ns < high_prefix_len,
                        recent_begin + ns - sink_len,
                        prefix + ns - high_prefix_len,
                    ),
                )
                is_current = ns >= high_prefix_len
                wi = tl.where(pos < SINK, pos, SINK + pos % RING)
                window_address = owner * (PAGE_BYTES // 2) + (wi * 2 * H + head) * D
                k_old = tl.load(
                    Window + window_address[:, None] + dims[None, :],
                    (valid & ~is_current)[:, None],
                    0,
                )
                v_old = tl.load(
                    Window + window_address[:, None] + H * D + dims[None, :],
                    (valid & ~is_current)[:, None],
                    0,
                )
                ci = qstart + pos - prefix
                k_new = tl.load(
                    CurrentK + ci[:, None] * KS0 + head * KS1 + dims[None, :] * KS2,
                    (valid & is_current)[:, None],
                    0,
                )
                v_new = tl.load(
                    CurrentV + ci[:, None] * VS0 + head * VS1 + dims[None, :] * VS2,
                    (valid & is_current)[:, None],
                    0,
                )
                k = tl.where(is_current[:, None], k_new, k_old).to(tl.bfloat16)
                v = tl.where(is_current[:, None], v_new, v_old).to(tl.bfloat16)
            else:
                pos = lo + ns
                block = tl.load(Table + req * TABLE_STRIDE + pos // BK, valid, 0).to(tl.int64)
                slot = block * BK + pos % BK
                address = slot // BP * PAGE_BYTES + (slot % BP * H + head) * SLOT
                k = _load_vec(History, address, valid, D, BN)
                v = _load_vec(History, address + VECTOR, valid, D, BN)
            score = tl.dot(q, tl.trans(k)) * SCALE
            mask = row_valid[:, None] & valid[None, :] & (pos[None, :] <= prefix + qi[:, None])
            score = tl.where(mask, score, -float("inf"))
            new_m = tl.maximum(m, tl.max(score, 1))
            # For an all-masked row both maxima are -inf. Avoid inf-inf NaNs.
            safe_m = tl.where(new_m == -float("inf"), 0.0, new_m)
            alpha = tl.exp(m - safe_m)
            p = tl.exp(score - safe_m[:, None])
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
            denominator = denominator * alpha + tl.sum(p, 1)
            m = new_m
        normalized = acc / tl.maximum(denominator[:, None], 1.0e-30)
        logsum = tl.where(
            denominator > 0, m + tl.log(tl.maximum(denominator, 1.0e-30)), -float("inf")
        )
        outrow = (token.to(tl.int64) * H * G + qhead) * SPLITS + split
        tl.store(Partials + outrow[:, None] * D + dims[None, :], normalized, row_valid[:, None])
        tl.store(LSE + outrow, logsum, row_valid)


def attention_partials(query, key, value, history, window, meta, layout, scale, splits, raw):
    n, hq, d = query.shape
    h = layout.kv_heads
    cfg = layout.config
    partials = torch.empty((n, hq, splits, d), dtype=torch.float32, device=query.device)
    lse = torch.empty((n, hq, splits), dtype=torch.float32, device=query.device)
    groups = attention_task_groups(n, meta.max_num_reqs, h, splits, cfg)
    _attention_kernel[(groups, h, splits)](
        query,
        key,
        value,
        history,
        window,
        meta.query_start_loc,
        meta.seq_lens,
        meta.block_tables,
        meta.counts,
        meta.tasks,
        meta.task_count,
        partials,
        lse,
        h,
        hq // h,
        d,
        cfg.queries_per_tile,
        max(16, triton.next_power_of_2(cfg.queries_per_tile * (hq // h))),
        cfg.block_n,
        groups,
        layout.kernel_block_tokens,
        layout.physical_block_tokens,
        layout.stripe_page_bytes,
        cfg.slot_bytes,
        cfg.vector_bytes,
        meta.block_tables.stride(0),
        cfg.sink_tokens,
        cfg.recent_tokens,
        cfg.ring_tokens,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        splits,
        scale,
        raw,
    )
    return partials, lse


@triton.jit
def _merge_splits_kernel(
    P, LSE, Out, Logsum, Counts, HQ: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr
):
    row = tl.program_id(0)
    valid = row // HQ < tl.load(Counts + 1)
    splits = tl.arange(0, SPLITS)
    dims = tl.arange(0, D)
    ls = tl.load(LSE + row * SPLITS + splits, valid, -float("inf"))
    maximum = tl.max(ls, 0)
    safe_max = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp(ls - safe_max)
    denom = tl.sum(weights, 0)
    values = tl.load(P + (row * SPLITS + splits[:, None]) * D + dims[None, :], valid, 0)
    out = tl.sum(values * weights[:, None], 0) / tl.maximum(denom, 1.0e-30)
    logsum = tl.where(denom > 0, maximum + tl.log(tl.maximum(denom, 1.0e-30)), -float("inf"))
    # Padded outputs must be initialized: the subsequent dense rotation reads them.
    tl.store(Out + row * D + dims, out)
    tl.store(Logsum + row, logsum)


def merge_splits(partials, lse, counts):
    n, hq, splits, d = partials.shape
    out = torch.empty((n, hq, d), dtype=torch.bfloat16, device=partials.device)
    logsum = torch.empty((n, hq), dtype=torch.float32, device=partials.device)
    _merge_splits_kernel[(n * hq,)](partials, lse, out, logsum, counts, hq, d, splits)
    return out, logsum


@triton.jit
def _merge_paths_kernel(A, LA, B, LB, Out, Counts, HQ: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(0)
    valid = row // HQ < tl.load(Counts + 1)
    dims = tl.arange(0, D)
    la = tl.load(LA + row, valid, -float("inf"))
    lb = tl.load(LB + row, valid, -float("inf"))
    maximum = tl.maximum(la, lb)
    maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    wa, wb = tl.exp(la - maximum), tl.exp(lb - maximum)
    a = tl.load(A + row * D + dims, valid, 0).to(tl.float32)
    b = tl.load(B + row * D + dims, valid, 0).to(tl.float32)
    result = (wa * a + wb * b) / tl.maximum(wa + wb, 1.0e-30)
    tl.store(Out + row * D + dims, result)


def merge_paths(history_out, history_lse, raw_partials, raw_lse, counts, output):
    n, hq, d = history_out.shape
    _merge_paths_kernel[(n * hq,)](
        history_out,
        history_lse,
        raw_partials,
        raw_lse,
        output,
        counts,
        hq,
        d,
    )


@triton.jit
def _window_store_kernel(
    K,
    V,
    Window,
    QStarts,
    SeqLens,
    Table,
    Counts,
    TABLE_STRIDE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BP: tl.constexpr,
    BK: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    SINK: tl.constexpr,
    RING: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VS2: tl.constexpr,
    BN: tl.constexpr = 16,
):
    req = tl.program_id(0)
    if req < tl.load(Counts):
        head = tl.program_id(2)
        qstart = tl.load(QStarts + req)
        qend = tl.minimum(tl.load(QStarts + req + 1), tl.load(Counts + 1))
        seq = tl.load(SeqLens + req)
        prefix = seq - (qend - qstart)
        index = tl.program_id(1) * BN + tl.arange(0, BN)
        # Each ring index is assigned exactly its latest current-token writer.
        residue = index - SINK
        latest = seq - 1 - (seq - 1 - residue) % RING
        pos = tl.where(index < SINK, index, latest)
        valid = (index < SINK + RING) & (pos >= prefix) & (pos < seq) & (qend > qstart)
        valid &= (index < SINK) | (pos >= SINK)
        first = tl.load(Table + req * TABLE_STRIDE).to(tl.int64)
        owner = first // (BP // BK)
        address = owner * (PAGE_BYTES // 2) + (index * 2 * H + head) * D
        dims = tl.arange(0, D)
        current = qstart + pos - prefix
        k = tl.load(
            K + current[:, None] * KS0 + head * KS1 + dims[None, :] * KS2, valid[:, None], 0
        )
        v = tl.load(
            V + current[:, None] * VS0 + head * VS1 + dims[None, :] * VS2, valid[:, None], 0
        )
        tl.store(Window + address[:, None] + dims[None, :], k, valid[:, None])
        tl.store(Window + address[:, None] + H * D + dims[None, :], v, valid[:, None])


def store_windows(key, value, window, meta, layout):
    cfg = layout.config
    _window_store_kernel[
        (min(key.shape[0], meta.max_num_reqs), triton.cdiv(cfg.window_tokens, 16), layout.kv_heads)
    ](
        key,
        value,
        window,
        meta.query_start_loc,
        meta.seq_lens,
        meta.block_tables,
        meta.counts,
        meta.block_tables.stride(0),
        layout.kv_heads,
        cfg.head_dim,
        layout.physical_block_tokens,
        layout.kernel_block_tokens,
        layout.stripe_page_bytes,
        cfg.sink_tokens,
        cfg.ring_tokens,
        *key.stride(),
        *value.stride(),
    )


@triton.jit
def _unpack_kernel(
    Cache,
    Slots,
    K,
    V,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BP: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    SLOT: tl.constexpr,
    VECTOR: tl.constexpr,
    BN: tl.constexpr = 16,
):
    tokens = tl.program_id(0) * BN + tl.arange(0, BN)
    head = tl.program_id(1)
    slot = tl.load(Slots + tokens, tokens < N, -1).to(tl.int64)
    valid = (tokens < N) & (slot >= 0)
    address = slot // BP * PAGE_BYTES + (slot % BP * H + head) * SLOT
    k = _load_vec(Cache, address, valid, D, BN)
    v = _load_vec(Cache, address + VECTOR, valid, D, BN)
    dims = tl.arange(0, D)
    out = (tokens[:, None] * H + head) * D + dims[None, :]
    tl.store(K + out, k, (tokens < N)[:, None])
    tl.store(V + out, v, (tokens < N)[:, None])


def dequant_inverse_rotate(history, slots, rk, rv, layout):
    """Diagnostic finite tile only; never called by the serving backend."""
    n, h, d = slots.numel(), layout.kv_heads, layout.config.head_dim
    if n > 4096:
        raise ValueError("Diagnostic dequant is limited to 4096 tokens; use fused attention")
    k = torch.empty((n, h, d), device=history.device, dtype=torch.bfloat16)
    v = torch.empty_like(k)
    cfg = layout.config
    _unpack_kernel[(triton.cdiv(n, 16), h)](
        history,
        slots,
        k,
        v,
        n,
        h,
        d,
        layout.physical_block_tokens,
        layout.stripe_page_bytes,
        cfg.slot_bytes,
        cfg.vector_bytes,
    )
    return rotate(k, rk.T), rotate(v, rv.T)
