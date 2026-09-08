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


def run_body(raw, active, task, lengths=(4,), order="forward"):
    accesses, dots = [], []
    generator = torch.Generator().manual_seed(15)
    n = sum(lengths)
    pid = [0]
    q = torch.randn(n, 6, 256, generator=generator).bfloat16()
    k = torch.randn(n, 1, 256, generator=generator).bfloat16()
    v = torch.randn(n, 1, 256, generator=generator).bfloat16()
    partials = torch.full((n, 6, 1, 256), 123.0)
    lse = torch.full((n, 6, 1), 123.0)

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
        program_id=lambda axis: torch.tensor(pid[0] if axis == 0 else 0, dtype=torch.int32),
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
        if isinstance(node, ast.FunctionDef) and node.name == "_attention_kernel"
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
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    tensors = {
        "Query": q,
        "CurrentK": k,
        "CurrentV": v,
        "History": torch.empty(0, dtype=torch.uint8),
        "Window": torch.empty(0, dtype=torch.bfloat16),
        "QStarts": torch.tensor(starts + [0] * (9 - len(starts)), dtype=torch.int32),
        "SeqLens": torch.tensor(list(lengths) + [0] * (8 - len(lengths)), dtype=torch.int32),
        "Table": torch.full((8, 6), 2**30, dtype=torch.int32),
        "Counts": torch.tensor([len(lengths), n] if active else [0, 0], dtype=torch.int32),
        "Tasks": torch.empty((max(n, 1), 2), dtype=torch.int32),
        "TaskCount": torch.zeros(1, dtype=torch.int32),
        "Partials": partials,
        "LSE": lse,
    }
    if not active:
        tensors["QStarts"].fill_(2**30)
        tensors["SeqLens"].fill_(2**30)
    from tests.test_tasks import prepare_tasks_cpu

    prepare_tasks_cpu(
        tensors["QStarts"], tensors["Counts"], tensors["Tasks"], tensors["TaskCount"], 8, 4
    )
    pointers = {name: Pointer(value, name, accesses) for name, value in tensors.items()}
    program_ids = list(range(max(n, 1))) if task is None else [task]
    if order == "reverse":
        program_ids.reverse()
    elif order == "interleaved":
        program_ids = program_ids[::2] + program_ids[1::2]
    for pid[0] in program_ids:
        namespace["_attention_kernel"](
            **pointers,
            H=1,
            G=6,
            D=256,
            QT=4,
            BM=32,
            BN=32,
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


@pytest.mark.parametrize("order", ["forward", "reverse", "interleaved"])
def test_single_task_programs_handle_ragged_queries_in_any_order(order):
    lengths = (7, 0, 2, 5)
    partials, lse, accesses, dots, (q, k, v) = run_body(True, True, None, lengths, order)
    start = 0
    for length in lengths:
        if length:
            scores = (
                torch.einsum(
                    "thd,sd->hts",
                    q[start : start + length].float(),
                    k[start : start + length, 0].float(),
                )
                / 16
            )
            scores.masked_fill_(torch.ones(length, length, dtype=torch.bool).triu(1), -float("inf"))
            expected = torch.einsum(
                "hts,sd->thd", scores.softmax(-1), v[start : start + length, 0].float()
            )
            torch.testing.assert_close(
                partials[start : start + length, :, 0], expected, atol=0.035, rtol=0.04
            )
            torch.testing.assert_close(
                lse[start : start + length, :, 0], scores.logsumexp(-1).T, atol=1e-5, rtol=1e-5
            )
        start += length
    assert not {"History", "Window"} & set(accesses)


def test_attention_has_only_the_kv_loop():
    path = Path(__file__).resolve().parents[1] / "oscar_ascend/kernels.py"
    fn = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_attention_kernel"
    )
    loops = [n for n in ast.walk(fn) if isinstance(n, (ast.For, ast.While))]
    assert len(loops) == 1
    assert isinstance(loops[0], ast.For) and loops[0].target.id == "start"
