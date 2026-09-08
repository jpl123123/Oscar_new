import ast
import importlib.metadata
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from oscar_ascend import plugin
from oscar_ascend.check import check_model, verify_sources
from tools.compare_benchmarks import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    MATCH_KEYS,
    SCENARIOS,
    compare,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({}, True),
        ({"use_mla": True}, False),
        ({"use_sparse": True}, False),
        ({"use_non_causal": True}, False),
        ({"attn_type": "encoder_only"}, False),
        ({"attn_type": "encoder_decoder"}, False),
    ],
)
def test_routing(changes, expected):
    fields = dict(attn_type="decoder", use_mla=False, use_sparse=False, use_non_causal=False)
    fields.update(changes)
    assert plugin.should_route(SimpleNamespace(**fields)) is expected


def test_plugin_disabled_does_not_import_npu(monkeypatch):
    monkeypatch.delenv("OSCAR_ASCEND_ENABLED", raising=False)
    before = set(sys.modules)
    plugin.register()
    assert not {"vllm", "vllm_ascend", "torch_npu", "triton"} & (set(sys.modules) - before)


def test_hooks_are_idempotent_and_preserve_draft_and_encoder(monkeypatch):
    class FakePlatform:
        @classmethod
        def get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads=None):
            return "native_backend"

    class FakeAttention:
        def __init__(self, num_heads, prefix="", attn_backend=None):
            self.backend = attn_backend
            self.prefix = prefix

        def process_weights_after_loading(self, act_dtype):
            pass

    modules = {
        "vllm_ascend": {"_ensure_global_patch": lambda: None},
        "vllm_ascend.platform": {"NPUPlatform": FakePlatform},
        "vllm.model_executor.layers.attention.attention": {"Attention": FakeAttention},
        "vllm_ascend.attention.attention_v1": {"AscendAttentionBackend": "native-draft"},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.23.0")
    monkeypatch.setattr(plugin, "_registered", False)
    monkeypatch.setenv("OSCAR_ASCEND_ENABLED", "1")
    plugin.register()
    first = FakeAttention.__init__
    plugin.register()
    assert FakeAttention.__init__ is first
    assert FakeAttention(6, "model.mtp.layers.0.attn").backend == "native-draft"
    assert (
        FakeAttention(6, prefix="mtp.layers.0.attn", attn_backend="explicit").backend == "explicit"
    )
    assert FakeAttention(6, "model.language_model.layers.3.attn").backend is None
    selector = SimpleNamespace(
        attn_type="decoder", use_mla=False, use_sparse=False, use_non_causal=False, head_size=256
    )
    assert FakePlatform.get_attn_backend_cls(None, selector).startswith("oscar_ascend.")
    selector.head_size = 0  # Native runner's pre-model backend probe.
    assert FakePlatform.get_attn_backend_cls(None, selector).startswith("oscar_ascend.")
    selector.attn_type = "encoder_only"
    assert FakePlatform.get_attn_backend_cls(None, selector) == "native_backend"


def test_pinned_upstream_contracts():
    roots = {"vllm": ROOT / "references/vllm", "vllm-ascend": ROOT / "references/vllm-ascend"}
    if not all(p.exists() for p in roots.values()):
        pytest.skip("Read-only upstream references absent on deployment machine")
    commits = verify_sources(roots)
    assert commits["vllm-ascend"].startswith("19e436985")
    tree = ast.parse((roots["vllm-ascend"] / "vllm_ascend/platform.py").read_text())
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "get_attn_backend_cls"
    )
    assert [a.arg for a in method.args.args] == [
        "cls",
        "selected_backend",
        "attn_selector_config",
        "num_heads",
    ]
    tree = ast.parse(
        (roots["vllm"] / "vllm/model_executor/layers/attention/attention.py").read_text()
    )
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Attention")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assert {"prefix", "attn_backend"} <= {a.arg for a in init.args.args}
    own = ast.parse((ROOT / "oscar_ascend/metadata.py").read_text())
    assert any(
        isinstance(n, ast.FunctionDef) and n.name == "build_for_cudagraph_capture"
        for n in ast.walk(own)
    )


def test_serving_has_no_full_context_dequant_or_host_tensor_reads():
    backend = (ROOT / "oscar_ascend/backend.py").read_text()
    kernels = (ROOT / "oscar_ascend/kernels.py").read_text()
    for code in (backend, kernels):
        for forbidden in (
            ".cpu(",
            '.to("cpu")',
            ".numpy(",
            ".tolist(",
            ".item(",
            "torch.quantile",
            "argsort(",
            "torch.sort(",
        ):
            assert forbidden not in code
    assert "dequant_inverse_rotate" not in backend
    tree = ast.parse(backend)
    # Protect the crucial write-after-read ordering, not just operator names.
    forward = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )
    statements = forward.body
    last_call = statements[-2].value
    assert isinstance(last_call, ast.Call) and last_call.func.attr == "store_windows"


@pytest.mark.parametrize(
    "mode,disabled", [("native", False), ("native-prefix-off", True), ("oscar", True)]
)
def test_launch_command_preserves_mtp_and_graph(mode, disabled):
    env = dict(os.environ, MODE=mode, DRY_RUN="1", MODEL="/model path/with spaces")
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/serve.sh")],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    cmd = shlex.split(result.stdout.splitlines()[1])
    assert "/model path/with spaces" in cmd
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "4"
    assert ("--no-enable-prefix-caching" in cmd) is disabled
    spec = json.loads(cmd[cmd.index("--speculative-config") + 1])
    graph = json.loads(cmd[cmd.index("--compilation-config") + 1])
    assert spec == {"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": True}
    assert graph["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert "--async-scheduling" in cmd and "--enforce-eager" not in cmd


def test_invalid_model_dimensions_fail_preflight(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"head_dim": 128}}))
    with pytest.raises(ValueError, match="head_dim"):
        check_model(tmp_path)


def write_benchmark_fixture(root, rate=10.0):
    root.mkdir()
    for scenario in SCENARIOS:
        for repeat in range(1, 4):
            data = {key: "same" for key in MATCH_KEYS}
            data.update(
                num_prompts=8,
                completed=8,
                failed=0,
                total_input_tokens=8192,
                total_output_tokens=4096,
            )
            data.update({key: rate for key in HIGHER_IS_BETTER})
            data.update({key: 100.0 for key in LOWER_IS_BETTER})
            (root / f"{scenario}-{repeat}.json").write_text(json.dumps(data))


def test_performance_gate_rejects_missing_and_regressing_measurements(tmp_path):
    baseline, candidate = tmp_path / "native", tmp_path / "oscar"
    assert not compare(baseline, candidate)["performance_passed"]
    write_benchmark_fixture(baseline, 10)
    write_benchmark_fixture(candidate, 9)
    assert not compare(baseline, candidate)["performance_passed"]
    for file in candidate.glob("*.json"):
        data = json.loads(file.read_text())
        data.update({key: 11 for key in HIGHER_IS_BETTER})
        file.write_text(json.dumps(data))
    result = compare(baseline, candidate)
    assert result["performance_passed"]
    assert result["quality_acceptance"] == "not_evaluated"
    file = candidate / "long-1.json"
    data = json.loads(file.read_text())
    data["num_prompts"] = 7
    file.write_text(json.dumps(data))
    assert not compare(baseline, candidate)["performance_passed"]


def test_invalid_benchmark_numbers_never_produce_pass(tmp_path):
    baseline, candidate = tmp_path / "native", tmp_path / "oscar"
    write_benchmark_fixture(baseline)
    write_benchmark_fixture(candidate)
    file = candidate / "long-1.json"
    data = json.loads(file.read_text())
    data["output_throughput"] = float("nan")
    file.write_text(json.dumps(data))
    result = compare(baseline, candidate)
    assert result["performance_passed"] is False
    json.dumps(result, allow_nan=False)
