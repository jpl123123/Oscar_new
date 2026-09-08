"""Actual scalar-summary body checked with CPU primitives, not a device benchmark."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.test_attention_empty_tasks import Pointer


@pytest.mark.parametrize("invalid", [None, float("inf"), float("nan"), -float("inf")])
def test_summary_matches_previous_host_reduction(invalid):
    generator = torch.Generator().manual_seed(40)
    stats = torch.rand(3, 8, 2, generator=generator)
    if invalid is not None:
        stats[1, 3, 0] = invalid
    output = torch.empty(())

    def load(ptr, mask=True, other=0):
        offsets, mask, indices = ptr.indices(mask)
        result = torch.full(offsets.shape, other, dtype=ptr.data.dtype)
        result[mask] = ptr.data[indices]
        return result

    def store(ptr, value):
        ptr.data[0] = value

    tl = SimpleNamespace(
        arange=torch.arange,
        load=load,
        store=store,
        abs=torch.abs,
        int32=torch.int32,
        sum=torch.sum,
        where=torch.where,
        max=lambda x, axis: x.max(dim=axis).values,
        maximum=lambda a, b: torch.maximum(torch.as_tensor(a), torch.as_tensor(b)),
    )
    path = Path(__file__).resolve().parents[1] / "oscar_ascend/calibration_kernels.py"
    tree = ast.parse(path.read_text())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_residual_summary"
    )
    fn.decorator_list = []
    for arg in fn.args.args:
        arg.annotation = None
    ns = {"tl": tl}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    ns[fn.name](Pointer(stats, "stats", []), Pointer(output, "summary", []), 3, 8, 4)
    if invalid is not None:
        assert torch.isposinf(output)
    else:
        old = stats.amax(1)
        expected = (old[:, 0] / old[:, 1].clamp_min(1e-30)).max()
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
