"""Check actual unpack source values and number of requested packed-byte elements."""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

from tests.test_attention_empty_tasks import Pointer


def test_packed_byte_loaded_once_for_four_codes():
    cache = torch.zeros(512, dtype=torch.uint8)
    addresses = torch.tensor([7, 999999, 167, 327], dtype=torch.int64)
    valid = torch.tensor([True, False, True, True])
    expected = torch.zeros(4, 256, dtype=torch.bfloat16)
    for row in (0, 2, 3):
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
    result = ns["_load_vec"](Pointer(cache, "cache", []), addresses, valid, 256, 4)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    assert reads[0] == (torch.Size([4, 64]), 3 * 64)
    assert sum(count for _, count in reads) == 3 * 68
