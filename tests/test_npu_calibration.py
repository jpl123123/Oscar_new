"""Hardware tests for automatic calibrated rotation generation."""

import pytest
import torch

pytest.importorskip("torch_npu")
import torch_npu  # noqa: E402, F401

if not torch.npu.is_available():
    pytest.skip("Ascend NPU unavailable", allow_module_level=True)

from oscar_ascend import calibration_kernels as ops  # noqa: E402
from tests.test_calibration_math import hadamard, pbr  # noqa: E402

pytestmark = pytest.mark.npu


def test_streamed_npu_covariances_against_fp64_reference():
    torch.manual_seed(88)
    q = torch.randn(137, 6, 256).bfloat16()
    k, v = torch.randn(137, 1, 256).bfloat16(), torch.randn(137, 1, 256).bfloat16()
    qsum = torch.zeros((256, 256), device="npu", dtype=torch.float32)
    for chunk in q.split(32):
        ops.accumulate_gram(chunk.to("npu"), qsum)
    qcov = ops.normalize(qsum, 137 * 6)
    expected_q = q.double().reshape(-1, 256).T @ q.double().reshape(-1, 256) / (137 * 6)
    torch.testing.assert_close(qcov.cpu().double(), expected_q, atol=2e-5, rtol=2e-5)
    vsum = torch.zeros_like(qsum)
    denominator = torch.zeros((), device="npu", dtype=torch.float32)
    for kc, vc in zip(k.split(32), v.split(32)):
        weight = ops.sst_weights(kc.to("npu"), qcov, denominator)
        ops.accumulate_gram(vc.to("npu"), vsum, weight)
    result = ops.normalize(vsum, denominator).cpu().double()
    weights = (k[:, 0].double() @ expected_q * k[:, 0].double()).sum(-1)
    expected_v = v[:, 0].double().T @ (v[:, 0].double() * weights[:, None]) / weights.sum()
    torch.testing.assert_close(result, expected_v, atol=2e-4, rtol=2e-4)


def test_npu_eigen_and_hadamard_composition_reconstruct_input():
    torch.manual_seed(23)
    x = torch.randn(2, 320, 256)
    covariance = x.transpose(-1, -2) @ x / 320
    r, eigenvalues, info = ops.calibrated_rotations(covariance.to("npu"))
    r, eigenvalues = r.cpu().double(), eigenvalues.cpu().double()
    assert info["relative_off_diagonal"] < 1e-6
    u = r @ pbr(256).T @ hadamard(256).T
    reconstruction = (u * eigenvalues[:, None, :]) @ u.transpose(-1, -2)
    torch.testing.assert_close(reconstruction, covariance.double(), atol=2e-4, rtol=2e-3)
    torch.testing.assert_close(
        r.transpose(-1, -2) @ r,
        torch.eye(256, dtype=torch.float64).expand(2, -1, -1),
        atol=1e-4,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        eigenvalues, torch.linalg.eigvalsh(covariance.double()), atol=2e-4, rtol=2e-3
    )
