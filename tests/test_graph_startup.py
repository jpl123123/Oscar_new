"""CPU regressions for native dummy ordering and bounded compilation structure."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from oscar_ascend import startup
from oscar_ascend.config import percentile_selection
from oscar_ascend.layout import CacheLayout

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def metadata_builder(monkeypatch):
    # Execute the real builder on CPU storage. Only native imports/device labels
    # are substituted; the copy/fill ordering and tensor contents are real Torch.
    class Base:
        def __class_getitem__(cls, item):
            return cls

        def __init__(self, *args):
            pass

    class Impl:
        pass

    class NpuLabel(torch.Tensor):
        @property
        def device(self):
            return SimpleNamespace(type="npu")

    def cpu_tasks(
        qstarts, counts, tasks, task_count, max_reqs, query_tile, num_reqs=None, **kwargs
    ):
        from tests.test_tasks import prepare_tasks_cpu

        prepare_tasks_cpu(
            qstarts, counts, tasks, task_count, max_reqs, query_tile, num_reqs, **kwargs
        )

    states = SimpleNamespace(PrefillNoCache=0, DecodeOnly=1, SpecDecoding=2)
    modules = {
        "vllm.v1.attention.backend": {
            "AttentionMetadataBuilder": Base,
            "AttentionCGSupport": SimpleNamespace(ALWAYS=1, UNIFORM_BATCH=2),
        },
        "vllm_ascend.attention.attention_v1": {"AscendAttentionState": states},
        "oscar_ascend.backend": {"OscarAttentionImpl": Impl},
        "oscar_ascend.kernels": {"prepare_tasks": cpu_tasks},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "oscar_ascend._metadata_cpu_test", ROOT / "oscar_ascend/metadata.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    cfg = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={"layer": SimpleNamespace(impl=Impl())}
        ),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=32),
        model_config=SimpleNamespace(max_model_len=4096),
    )
    builder = module.OscarMetadataBuilder(
        SimpleNamespace(block_size=768, num_kv_heads=1), ["layer"], cfg, "cpu"
    )
    cm = SimpleNamespace(
        causal=True,
        num_reqs=1,
        num_actual_tokens=4,
        max_query_len=4,
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32).as_subclass(NpuLabel),
        query_start_loc_cpu=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32).as_subclass(NpuLabel),
        block_table_tensor=torch.zeros(1, 6, dtype=torch.int32).as_subclass(NpuLabel),
        slot_mapping=torch.zeros(4, dtype=torch.int32).as_subclass(NpuLabel),
        attn_state=states.DecodeOnly,
    )
    return builder, cm


def install_fake_runner(monkeypatch, runner_class):
    module = ModuleType("vllm_ascend.worker.model_runner_v1")
    module.NPUModelRunner = runner_class
    module.__spec__ = SimpleNamespace(_initializing=True)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert not startup.install_runner_hooks()
    module.__spec__._initializing = False
    assert startup.install_runner_hooks()
    wrapped = runner_class._dummy_run
    assert startup.install_runner_hooks()
    assert runner_class._dummy_run is wrapped


def test_warmup_cannot_snapshot_slots_before_native_invalidates_them(metadata_builder, monkeypatch):
    builder, cm = metadata_builder
    monkeypatch.setenv("OSCAR_STARTUP_TRACE", "0")

    class Runner:
        def _dummy_run(self, num_tokens, is_graph_capturing=False):
            # Exact native order: build first, invalidate the source afterwards.
            meta = builder.build(0, cm)
            cm.slot_mapping.fill_(-1)
            return meta

        def capture_model(self):
            return self._dummy_run(4, is_graph_capturing=True)

    # Without the scope, the private snapshot retains zero while the native
    # source has become -1: this is the pre-fix write-to-shared-page-zero bug.
    unsafe = Runner()._dummy_run(4)
    assert unsafe.slot_mapping[:4].eq(0).all() and cm.slot_mapping.eq(-1).all()
    cm.slot_mapping.zero_()
    install_fake_runner(monkeypatch, Runner)
    meta = Runner()._dummy_run(4)
    assert meta.is_dummy and not meta.initial_prefill
    assert meta.counts.tolist() == [0, 0]
    assert meta.task_count.item() == 0
    # Existing private buffers can remain stale, but zero counts mask them.
    assert meta.slot_mapping[:4].eq(0).all()
    assert not startup.in_dummy_run()


def test_capture_has_no_live_cache_access_and_real_build_restores_same_buffers(metadata_builder):
    builder, cm = metadata_builder
    # Deliberately poisoned source metadata must not become active during capture.
    cm.slot_mapping.fill_(2**30)
    cm.block_table_tensor.fill_(2**30)
    captured = builder.build_for_cudagraph_capture(cm)
    fields = (
        "query_start_loc",
        "seq_lens",
        "block_tables",
        "slot_mapping",
        "counts",
        "tasks",
        "task_count",
    )
    pointers = {name: getattr(captured, name).data_ptr() for name in fields}
    assert captured.counts.tolist() == [0, 0]
    assert captured.block_tables.eq(0).all()
    assert captured.slot_mapping.eq(-1).all()
    assert captured.is_dummy and not startup.in_dummy_run()
    assert captured.task_count.item() == 0

    cm.block_table_tensor.copy_(torch.arange(6, 12, dtype=torch.int32)[None, :])
    cm.slot_mapping.copy_(torch.arange(768, 772, dtype=torch.int32))
    cm.seq_lens.fill_(12)
    real = builder.build(0, cm)
    assert not real.is_dummy
    assert pointers == {name: getattr(real, name).data_ptr() for name in fields}
    assert real.counts.tolist() == [1, 4]
    assert real.query_start_loc[:2].tolist() == [0, 4]
    assert real.seq_lens[0] == 12
    assert real.slot_mapping[:4].tolist() == [768, 769, 770, 771]
    assert real.block_tables[0, :6].tolist() == list(range(6, 12))
    assert real.task_count.item() == 1 and real.tasks[0].tolist() == [0, 0]


def test_dummy_scope_and_watchdog_are_cleaned_after_failure(monkeypatch, tmp_path):
    events = []
    monkeypatch.setenv("OSCAR_STARTUP_TRACE", "1")
    monkeypatch.setenv("OSCAR_STARTUP_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(
        startup.faulthandler, "dump_traceback_later", lambda *a, **kw: events.append((a, kw))
    )
    monkeypatch.setattr(
        startup.faulthandler, "cancel_dump_traceback_later", lambda: events.append("cancel")
    )

    class Runner:
        device = "npu:0"

        def _dummy_run(self, num_tokens, is_graph_capturing=False):
            assert startup.in_dummy_run()
            assert startup._capturing.get()
            raise RuntimeError("kernel failed")

        def capture_model(self):
            return self._dummy_run(512, is_graph_capturing=True)

    install_fake_runner(monkeypatch, Runner)
    with pytest.raises(RuntimeError, match="kernel failed"):
        Runner().capture_model()
    assert not startup.in_dummy_run()
    assert not startup._capturing.get()
    assert events[0][0] == (120,) and events[0][1]["repeat"]
    stack_file = events[0][1]["file"]
    assert Path(stack_file.name).parent == tmp_path and stack_file.closed
    assert events[1] == "cancel"


def test_metadata_copies_only_columns_reachable_by_host_upper_bound(metadata_builder):
    builder, cm = metadata_builder
    cm.block_table_tensor.copy_(torch.arange(6, 12, dtype=torch.int32)[None, :])
    builder.build(0, cm)
    pointer = builder.table.data_ptr()
    cm.block_table_tensor.fill_(50)
    cm.max_seq_len = 128
    builder.build(0, cm)
    assert builder.table[0, :6].tolist() == [50, 7, 8, 9, 10, 11]
    cm.block_table_tensor.fill_(60)
    cm.max_seq_len = 129
    builder.build(0, cm)
    assert builder.table[0, :6].tolist() == [60, 60, 8, 9, 10, 11]
    cm.max_seq_len = 768
    builder.build(0, cm)
    assert builder.table[0, :6].eq(60).all() and builder.table.data_ptr() == pointer


def test_kernel_trace_preserves_values_and_reports_first_launch_only(monkeypatch, capsys):
    monkeypatch.setenv("OSCAR_STARTUP_TRACE", "1")
    monkeypatch.setattr(startup, "_kernel_proxy", None)
    monkeypatch.setattr(startup, "_traced_ops", set())
    names = (
        "rotate",
        "store_int2",
        "attention_partials",
        "merge_splits",
        "merge_paths",
        "store_windows",
    )
    ops = SimpleNamespace(**{name: lambda *args, **kw: args[0] for name in names})
    proxy = startup.trace_kernels(ops)
    value = torch.zeros(4, 1, 256)
    rotation = torch.eye(256)
    assert proxy.rotate(value, rotation) is value and proxy.rotate(value, rotation) is value
    args = (
        value,
        None,
        None,
        None,
        None,
        SimpleNamespace(max_num_reqs=8),
        CacheLayout(768),
        1 / 16,
        8,
    )
    assert proxy.attention_partials(*args, raw=False) is value
    assert proxy.attention_partials(*args, raw=True) is value
    output = capsys.readouterr().out
    assert output.count("rotate: first launch begin") == 1
    assert "attention_partials.history" in output and "attention_partials.raw" in output
    monkeypatch.setenv("OSCAR_STARTUP_TRACE", "0")
    assert startup.trace_kernels(ops) is ops


def test_first_warmup_variant_waits_for_device_but_capture_never_syncs(monkeypatch):
    events = []
    monkeypatch.setenv("OSCAR_STARTUP_TRACE", "1")
    monkeypatch.setattr(startup, "_kernel_proxy", None)
    monkeypatch.setattr(startup, "_traced_ops", set())
    names = (
        "rotate",
        "store_int2",
        "attention_partials",
        "merge_splits",
        "merge_paths",
        "store_windows",
    )

    def launch(*args, **kwargs):
        events.append("launch")
        return args[0]

    proxy = startup.trace_kernels(
        SimpleNamespace(**{name: launch for name in names}),
        lambda: events.append("sync"),
    )
    k = torch.zeros(4, 1, 256)
    q = torch.zeros(4, 6, 256)
    r = torch.eye(256)
    with startup.dummy_run_scope(capturing=False):
        proxy.rotate(k, r)
        proxy.rotate(k, r)  # Already completed on device.
        proxy.rotate(q, r)  # Different head/stride specialization must complete too.
    assert events == ["sync", "launch", "sync", "launch", "sync", "launch", "sync"]
    events.clear()
    with startup.dummy_run_scope(capturing=True):
        proxy.rotate(q, r.T)  # Even a previously unseen variant cannot synchronize here.
    assert events == ["launch"]
    assert not startup.in_dummy_run() and not startup._capturing.get()


def test_failed_device_completion_does_not_mark_variant_as_ready(monkeypatch):
    monkeypatch.setenv("OSCAR_STARTUP_TRACE", "1")
    monkeypatch.setattr(startup, "_kernel_proxy", None)
    monkeypatch.setattr(startup, "_traced_ops", set())
    names = (
        "rotate",
        "store_int2",
        "attention_partials",
        "merge_splits",
        "merge_paths",
        "store_windows",
    )
    calls = []

    def sync():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("device execution failed")

    proxy = startup.trace_kernels(
        SimpleNamespace(**{name: lambda *a, **k: None for name in names}), sync
    )
    with startup.dummy_run_scope(capturing=False), pytest.raises(RuntimeError, match="device"):
        proxy.rotate(torch.zeros(4, 1, 256), torch.eye(256))
    assert not startup._traced_ops


@pytest.mark.parametrize("ratio", [0, 0.875, 0.92, 0.96, 1.0])
def test_rolled_clip_body_keeps_quantile_semantics(ratio):
    # Run the actual scalar/vector body through Torch equivalents of TL ops.
    # This validates arithmetic and tied-rank removal, not Ascend compilation.
    tree = ast.parse((ROOT / "oscar_ascend/kernels.py").read_text())
    clip = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_clip_vec"
    )
    clip.decorator_list = []
    for arg in clip.args.args:
        arg.annotation = None
    tl = SimpleNamespace(
        abs=torch.abs,
        max=lambda x, axis: x.max(dim=axis).values,
        min=lambda x, axis: x.min(dim=axis).values,
        float32=torch.float32,
        full=lambda shape, value, dtype: torch.full(shape, value, dtype=dtype),
        arange=torch.arange,
        where=torch.where,
        minimum=torch.minimum,
        maximum=torch.maximum,
    )
    ns = {"tl": tl}
    exec(compile(ast.Module(body=[clip], type_ignores=[]), "clip_body", "exec"), ns)
    low, high, weight = percentile_selection(ratio)
    generator = torch.Generator().manual_seed(31)
    for values in (
        torch.randn(256, generator=generator),
        torch.zeros(256),
        torch.arange(256).float().remainder(11) - 5,
    ):
        actual = ns["_clip_vec"](values, 256, "percentile", ratio > 0, low, high, weight, 1.0)
        threshold = torch.quantile(values.abs(), ratio)
        expected = values.clamp(-threshold, threshold) if ratio > 0 else values
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_compile_structure_does_not_unroll_clip_or_specialize_rotation_token_count():
    tree = ast.parse((ROOT / "oscar_ascend/kernels.py").read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    clip_loop = next(node for node in ast.walk(functions["_clip_vec"]) if isinstance(node, ast.For))
    assert isinstance(clip_loop.iter.func, ast.Name) and clip_loop.iter.func.id == "range"
    rotate = functions["_rotate_kernel"]
    assert next(arg for arg in rotate.args.args if arg.arg == "N").annotation is None
    options = {kw.arg: ast.literal_eval(kw.value) for kw in rotate.decorator_list[0].keywords}
    assert options["do_not_specialize"] == ["N"]


def test_reference_dummy_invalidates_source_after_building_metadata():
    path = ROOT / "references/vllm-ascend/vllm_ascend/worker/model_runner_v1.py"
    if not path.exists():
        pytest.skip("Read-only native reference tree is not shipped to deployment")
    tree = ast.parse(path.read_text())
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner"
    )
    dummy = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "_dummy_run"
    )
    build = next(
        node
        for node in ast.walk(dummy)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_build_attention_metadata"
    )
    invalidate = next(
        node
        for node in ast.walk(dummy)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fill_"
        and ast.unparse(node.func.value) == "blk_table.slot_mapping.gpu"
    )
    assert ast.literal_eval(invalidate.args[0]) == -1
    assert build.lineno < invalidate.lineno
