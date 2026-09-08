"""Execute the attention source body with CPU tensor primitives and checked pointers.

These tests verify masks, addresses, and arithmetic; they do not compile Triton
or emulate Ascend's Cube/Vector synchronization.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class Pointer:
    def __init__(self, data, name, accesses, offset=0):
        self.data = data.view(-1)
        self.name, self.accesses, self.offset = name, accesses, offset

    def __add__(self, offset):
        return Pointer(self.data, self.name, self.accesses, self.offset + offset)

    def indices(self, mask):
        offsets, mask = torch.broadcast_tensors(torch.as_tensor(self.offset), torch.as_tensor(mask))
        indices = offsets[mask].long()
        assert not indices.numel() or (indices.min() >= 0 and indices.max() < self.data.numel()), (
            self.name
        )
        if indices.numel():
            self.accesses.append(self.name)
        return offsets, mask, indices


def run_body(raw, active, task):
    accesses, dots = [], []
    generator = torch.Generator().manual_seed(15)
    q = torch.randn(4, 6, 256, generator=generator).bfloat16()
    k = torch.randn(4, 1, 256, generator=generator).bfloat16()
    v = torch.randn(4, 1, 256, generator=generator).bfloat16()
    partials = torch.full((4, 6, 1, 256), 123.0)
    lse = torch.full((4, 6, 1), 123.0)

    def load(ptr, mask=True, other=0):
        offsets, mask, indices = ptr.indices(mask)
        result = torch.full(offsets.shape, other, dtype=ptr.data.dtype)
        result[mask] = ptr.data[indices]
        return result

    def store(ptr, value, mask=True):
        offsets, mask, indices = ptr.indices(mask)
        ptr.data[indices] = torch.broadcast_to(value, offsets.shape)[mask].to(ptr.data.dtype)

    def dot(a, b):
        dots.append((a.shape, b.shape))
        return a.float() @ b.float()

    def minimum(a, b):
        return torch.minimum(torch.as_tensor(a), torch.as_tensor(b))

    def maximum(a, b):
        return torch.maximum(torch.as_tensor(a), torch.as_tensor(b))

    tl = SimpleNamespace(
        load=load,
        store=store,
        dot=dot,
        program_id=lambda axis: torch.tensor(task if axis == 0 else 0, dtype=torch.int32),
        arange=torch.arange,
        minimum=minimum,
        maximum=maximum,
        cdiv=lambda a, b: (a + b - 1) // b,
        cumsum=lambda x, axis=0: torch.cumsum(x, axis),
        sum=torch.sum,
        where=torch.where,
        full=lambda shape, value, dtype: torch.full(shape, value, dtype=dtype),
        max=lambda x, axis: x.max(dim=axis).values,
        trans=lambda x: x.T,
        exp=torch.exp,
        log=torch.log,
        int32=torch.int32,
        int64=torch.int64,
        float32=torch.float32,
        bfloat16=torch.bfloat16,
    )
    source = Path(__file__).resolve().parents[1] / "oscar_ascend/kernels.py"
    tree = ast.parse(source.read_text())
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("_query_tile", "_attention_kernel")
    ]
    for node in body:
        node.decorator_list = []
        for arg in node.args.args:
            arg.annotation = None

    def empty_history(cache, address, valid, dim, bn):
        # These history tests have no prefix, so no packed byte may be read.
        assert not valid.any()
        return torch.zeros(bn, dim, dtype=torch.bfloat16)

    namespace = {"tl": tl, "_load_vec": empty_history}
    exec(compile(ast.Module(body=body, type_ignores=[]), "attention_body", "exec"), namespace)
    tensors = {
        "Query": q,
        "CurrentK": k,
        "CurrentV": v,
        "History": torch.empty(0, dtype=torch.uint8),
        "Window": torch.empty(0, dtype=torch.bfloat16),
        "QStarts": torch.tensor([0, 4] + [0] * 7, dtype=torch.int32),
        "SeqLens": torch.tensor([4] + [0] * 7, dtype=torch.int32),
        "Table": torch.full((8, 6), 2**30, dtype=torch.int32),
        "Counts": torch.tensor([1, 4] if active else [0, 0], dtype=torch.int32),
        "Partials": partials,
        "LSE": lse,
    }
    if not active:
        tensors["QStarts"].fill_(2**30)
        tensors["SeqLens"].fill_(2**30)
    pointers = {name: Pointer(value, name, accesses) for name, value in tensors.items()}
    namespace["_attention_kernel"](
        **pointers,
        H=1,
        G=6,
        D=256,
        QT=4,
        BM=32,
        BN=32,
        MAX_REQS=8,
        BK=128,
        BP=768,
        PAGE_BYTES=768 * 256 * 2,
        SLOT=160,
        VECTOR=68,
        TABLE_STRIDE=6,
        SINK=64,
        RECENT=256,
        RING=260,
        QS0=q.stride(0),
        QS1=q.stride(1),
        QS2=q.stride(2),
        KS0=k.stride(0),
        KS1=k.stride(1),
        KS2=k.stride(2),
        VS0=v.stride(0),
        VS1=v.stride(1),
        VS2=v.stride(2),
        SPLITS=1,
        SCALE=1 / 16,
        RAW=raw,
    )
    return partials, lse, accesses, dots, (q, k, v)


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("active,task", [(False, 0), (False, 3), (True, 3)])
def test_inactive_tiles_execute_masked_dot_pair_without_any_data_access(raw, active, task):
    partials, lse, accesses, dots, _ = run_body(raw, active, task)
    assert len(dots) == 2
    assert not {
        "Query",
        "CurrentK",
        "CurrentV",
        "History",
        "Window",
        "Table",
        "Partials",
        "LSE",
    } & set(accesses)
    assert partials.eq(123).all() and lse.eq(123).all()


def test_live_empty_history_returns_zero_and_negative_infinity():
    partials, lse, accesses, dots, _ = run_body(False, True, 0)
    assert len(dots) == 2
    assert partials.eq(0).all() and torch.isneginf(lse).all()
    assert not {"History", "Window", "Table", "CurrentK", "CurrentV"} & set(accesses)


def test_live_raw_current_attention_remains_causal():
    partials, lse, accesses, dots, (q, k, v) = run_body(True, True, 0)
    scores = torch.einsum("thd,sd->hts", q.float(), k[:, 0].float()) / 16
    scores.masked_fill_(torch.ones(4, 4, dtype=torch.bool).triu(1), -float("inf"))
    expected = torch.einsum("hts,sd->thd", scores.softmax(-1), v[:, 0].float())
    torch.testing.assert_close(partials[:, :, 0], expected, atol=0.035, rtol=0.04)
    torch.testing.assert_close(lse[:, :, 0], scores.logsumexp(-1).T, atol=1e-5, rtol=1e-5)
    assert len(dots) == 2 and not {"History", "Window"} & set(accesses)
