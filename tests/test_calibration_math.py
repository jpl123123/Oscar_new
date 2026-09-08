"""Mathematical CPU oracles only. Production calibration never uses CPU EVD."""

import torch


def jacobi_oracle(covariance, sweeps=10):
    a = covariance.double().clone()
    dim = a.shape[0]
    u = torch.eye(dim, dtype=a.dtype)
    j = torch.arange(dim // 2)
    for _ in range(sweeps):
        for round_index in range(dim - 1):
            p = torch.where(j == 0, dim - 1, (round_index + j) % (dim - 1))
            q = torch.where(j == 0, round_index, (round_index - j + dim - 1) % (dim - 1))
            apq, delta = a[p, q], (a[q, q] - a[p, p]) / 2
            sign = torch.where(delta >= 0, 1.0, -1.0)
            tangent = sign * apq / (delta.abs() + (delta**2 + apq**2).sqrt()).clamp_min(1e-30)
            tangent = torch.where(
                apq.abs() > 1e-8 * (a[p, p].abs() + a[q, q].abs()) + 1e-20, tangent, 0
            )
            c, s = (1 + tangent**2).rsqrt(), tangent * (1 + tangent**2).rsqrt()
            b, nu = a.clone(), u.clone()
            b[:, p], b[:, q] = a[:, p] * c - a[:, q] * s, a[:, p] * s + a[:, q] * c
            nu[:, p], nu[:, q] = u[:, p] * c - u[:, q] * s, u[:, p] * s + u[:, q] * c
            a[p], a[q] = (
                c[:, None] * b[p] - s[:, None] * b[q],
                s[:, None] * b[p] + c[:, None] * b[q],
            )
            u = nu
    order = a.diag().argsort()
    return u[:, order], a.diag()[order]


def hadamard(dim):
    matrix = torch.ones((1, 1), dtype=torch.float64)
    while matrix.shape[0] < dim:
        matrix = (
            torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0) / 2**0.5
        )
    return matrix


def pbr(dim):
    bits = dim.bit_length() - 1
    reverse = torch.tensor([int(f"{i:0{bits}b}"[::-1], 2) for i in range(dim)])
    perm = torch.empty(dim, dtype=torch.long)
    perm[reverse] = torch.arange(dim - 1, -1, -1)
    return torch.eye(dim, dtype=torch.float64)[:, perm]


def test_round_robin_covers_each_pair_once():
    for dim in (4, 16, 256):
        visited = []
        for r in range(dim - 1):
            pairs = [(dim - 1, r)] + [
                ((r + j) % (dim - 1), (r - j + dim - 1) % (dim - 1)) for j in range(1, dim // 2)
            ]
            assert len(set(i for pair in pairs for i in pair)) == dim
            visited.extend(tuple(sorted(pair)) for pair in pairs)
        assert len(set(visited)) == dim * (dim - 1) // 2


def test_jacobi_reconstructs_covariance_and_orthogonal_rotation():
    torch.manual_seed(14)
    x = torch.randn(48, 16, dtype=torch.float64)
    covariance = x.T @ x
    u, eigenvalues = jacobi_oracle(covariance)
    torch.testing.assert_close(u @ torch.diag(eigenvalues) @ u.T, covariance, atol=1e-5, rtol=1e-6)
    torch.testing.assert_close(eigenvalues, torch.linalg.eigvalsh(covariance), atol=1e-5, rtol=1e-6)
    rotation = u @ hadamard(16) @ pbr(16)
    torch.testing.assert_close(
        rotation.T @ rotation, torch.eye(16, dtype=torch.float64), atol=1e-10, rtol=1e-10
    )


def test_streaming_sst_matches_official_normalization():
    torch.manual_seed(62)
    q = torch.randn(45, 6, 16, dtype=torch.float64)
    k, v = torch.randn(45, 16, dtype=torch.float64), torch.randn(45, 16, dtype=torch.float64)
    qcov = q.reshape(-1, 16).T @ q.reshape(-1, 16) / (45 * 6)
    weights = (k @ qcov * k).sum(-1)
    normalized = weights / weights.sum() * 45
    vw = v * normalized.sqrt()[:, None]
    official = vw.T @ vw / 45
    total = torch.zeros_like(official)
    denominator = 0.0
    for start in range(0, 45, 7):
        chunk = v[start : start + 7]
        w = weights[start : start + 7]
        total += chunk.T @ (chunk * w[:, None])
        denominator += w.sum()
    torch.testing.assert_close(total / denominator, official)
