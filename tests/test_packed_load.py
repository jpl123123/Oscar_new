"""Check unpack values and guard against narrow-tail 3-D UB intermediates."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.test_attention_empty_tasks import Pointer


@pytest.mark.parametrize("rows", [4, 16, 32])
def test_packed_values_and_masked_addresses_with_2d_unpack(rows):
    cache = torch.zeros(rows * 160, dtype=torch.uint8)
    addresses = torch.arange(rows, dtype=torch.int64) * 160 + 7
    addresses[1] = 999999
    valid = torch.ones(rows, dtype=torch.bool)
    valid[1] = False
    expected = torch.zeros(rows, 256, dtype=torch.bfloat16)
    for row in range(rows):
        if not valid[row]:
            continue
        codes = (torch.arange(256) + row) % 4
        packed = (codes.reshape(64, 4) << (torch.arange(4) * 2)).sum(1).byte()
        scale = torch.tensor([0.125 * (row + 1)], dtype=torch.float16)
        zero = torch.tensor([-0.25 * row], dtype=torch.float16)
        start = addresses[row].item()
        cache[start : start + 64] = packed
        cache[start + 64 : start + 66] = scale.view(torch.uint8)
        cache[start + 66 : start + 68] = zero.view(torch.uint8)
        expected[row] = (codes.float() * scale.float() + zero.float()).bfloat16()

    reads = []

    def load(ptr, mask=True, other=0):
        offsets, mask, indices = ptr.indices(mask)
        reads.append((offsets.shape, indices.numel()))
        result = torch.full(offsets.shape, other, dtype=ptr.data.dtype)
        result[mask] = ptr.data[indices]
        return result

    tl = SimpleNamespace(
        arange=torch.arange,
        load=load,
        int32=torch.int32,
        uint16=torch.int32,
        float16=torch.float16,
        float32=torch.float32,
        bfloat16=torch.bfloat16,
        reshape=torch.reshape,
    )
    path = Path(__file__).resolve().parents[1] / "oscar_ascend/kernels.py"
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_load_vec")
    fn.decorator_list = []
    for arg in fn.args.args:
        arg.annotation = None

    class Bitcast(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "to"
                and any(kw.arg == "bitcast" and ast.literal_eval(kw.value) for kw in node.keywords)
            ):
                return ast.copy_location(
                    ast.Call(
                        func=ast.Name(id="bitcast", ctx=ast.Load()),
                        args=[node.func.value] + node.args,
                        keywords=[],
                    ),
                    node,
                )
            return node

    fn = Bitcast().visit(fn)
    # Torch lacks uint16 shifts on some CPU builds. Widen those intermediate
    # bit operations, then narrow to the exact 16 bits before reinterpreting.
    ns = {"tl": tl, "bitcast": lambda x, dtype: x.to(torch.uint16).contiguous().view(dtype)}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(path), "exec"
        ),
        ns,
    )
    result = ns["_load_vec"](Pointer(cache, "cache", []), addresses, valid, 256, rows)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    assert reads[0] == (torch.Size([rows, 256]), (rows - 1) * 256)
    # Repeated byte addresses are intentional. This counts source load lanes,
    # not actual global-memory transactions made by the compiled device kernel.
    assert sum(count for _, count in reads) == (rows - 1) * (256 + 4)


def test_unpack_never_expands_a_four_element_tail_axis():
    path = Path(__file__).resolve().parents[1] / "oscar_ascend/kernels.py"
    fn = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_load_vec"
    )
    for node in ast.walk(fn):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Tuple):
            assert len(node.slice.elts) <= 2
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"reshape", "join", "interleave", "expand_dims"}
