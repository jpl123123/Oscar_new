"""Execute compact task preparation source on CPU; NPU execution is tested separately."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from oscar_ascend.config import OscarConfig, attention_programs, task_capacity

ROOT = Path(__file__).resolve().parents[1]


def prepare_tasks_cpu(
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
    from tests.test_attention_empty_tasks import Pointer

    pid, accesses = [0], []

    def load(ptr, mask=True, other=0):
        offsets, mask, indices = ptr.indices(mask)
        output = torch.full(offsets.shape, other, dtype=ptr.data.dtype)
        output[mask] = ptr.data[indices]
        return output

    def store(ptr, value, mask=True):
        offsets, mask, indices = ptr.indices(mask)
        ptr.data[indices] = torch.broadcast_to(torch.as_tensor(value), offsets.shape)[mask].to(
            ptr.data.dtype
        )

    tl = SimpleNamespace(
        load=load,
        store=store,
        program_id=lambda axis: torch.tensor(pid[0], dtype=torch.int32),
        arange=torch.arange,
        minimum=lambda a, b: torch.minimum(torch.as_tensor(a), torch.as_tensor(b)),
        maximum=lambda a, b: torch.maximum(torch.as_tensor(a), torch.as_tensor(b)),
        cdiv=lambda a, b: (a + b - 1) // b,
        sum=torch.sum,
        where=torch.where,
    )
    tree = ast.parse((ROOT / "oscar_ascend/kernels.py").read_text())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_prepare_tasks_kernel"
    )
    fn.decorator_list = []
    for arg in fn.args.args:
        arg.annotation = None
    ns = {"tl": tl}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "prepare_tasks_body", "exec"), ns)
    for pid[0] in range(max(1, max_reqs if num_reqs is None else num_reqs)):
        ns[fn.name](
            Pointer(qstarts, "qstarts", accesses),
            Pointer(counts, "counts", accesses),
            Pointer(tasks, "tasks", accesses),
            Pointer(task_count, "task_count", accesses),
            query_tile,
            1 << (max_reqs - 1).bit_length(),
            SourceTable=Pointer(source_table.as_subclass(torch.Tensor), "source_table", accesses)
            if source_table is not None
            else None,
            Table=Pointer(table, "table", accesses) if table is not None else None,
            COPY_COLS=copy_columns,
            SOURCE_STRIDE=source_table.stride(0) if source_table is not None else 0,
            TABLE_STRIDE=table.stride(0) if table is not None else 0,
            COPY_TABLE=source_table is not None,
        )


@pytest.mark.parametrize("query_tile", [1, 2, 4])
@pytest.mark.parametrize(
    "lengths,active",
    [([], 0), ([0, 0], 0), ([4] * 128, 512), ([7, 0, 1, 11], 19), ([7, 0, 1, 11], 10)],
)
def test_compact_map_matches_ragged_query_partition(query_tile, lengths, active):
    max_reqs = 129
    starts = [0]
    for size in lengths:
        starts.append(starts[-1] + size)
    qstarts = torch.zeros(max_reqs + 1, dtype=torch.int32)
    qstarts[: len(starts)] = torch.tensor(starts, dtype=torch.int32)
    counts = torch.tensor([len(lengths), active], dtype=torch.int32)
    capacity = task_capacity(max(active, 1), max_reqs, query_tile)
    tasks = torch.full((capacity, 2), -99, dtype=torch.int32)
    total = torch.full((1,), -99, dtype=torch.int32)
    prepare_tasks_cpu(
        qstarts, counts, tasks, total, max_reqs, query_tile, num_reqs=max(1, len(lengths))
    )
    expected = [
        [req, offset]
        for req in range(len(lengths))
        for offset in range(0, min(starts[req + 1], active) - min(starts[req], active), query_tile)
    ]
    assert total.item() == len(expected)
    assert tasks[: len(expected)].tolist() == expected
    assert tasks[len(expected) :].eq(-99).all()


@pytest.mark.parametrize(
    "tokens,requests,splits",
    [(1, 1, 32), (512, 129, 8), (512, 129, 1), (16000, 129, 1), (32, 8, 16)],
)
def test_grid_covers_each_task_without_persistent_loop(tokens, requests, splits):
    cfg = OscarConfig()
    programs = attention_programs(tokens, requests, cfg.queries_per_tile)
    capacity = task_capacity(tokens, requests, cfg.queries_per_tile)
    assert programs >= capacity
    for total in (0, 1, max(1, capacity // 2), capacity):
        visited = [task for task in range(programs) if task < total]
        assert sorted(visited) == list(range(total))
    if tokens == 512 and splits == 8:
        assert programs * splits == 2048


def test_retired_budget_cannot_reenable_persistent_execution(monkeypatch):
    monkeypatch.setenv("OSCAR_ASCEND_PROGRAM_BUDGET", "32")
    cfg = OscarConfig.from_env()
    assert not hasattr(cfg, "program_budget")
    assert attention_programs(512, 129, cfg.queries_per_tile) == 256
