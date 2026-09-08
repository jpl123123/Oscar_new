"""Regression tests for import-order side effects, without importing the NPU stack."""

import builtins
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from oscar_ascend import calibration_worker, check, plugin
from oscar_ascend.source_interfaces import validate_source_interfaces
from tests.test_compat import mock_runtime

ROOT = Path(__file__).resolve().parents[1]


def reference_roots():
    roots = {"vllm": ROOT / "references/vllm", "vllm-ascend": ROOT / "references/vllm-ascend"}
    if not all(root.exists() for root in roots.values()):
        pytest.skip("Read-only references are not shipped with the deployment")
    return roots


def forbid_native_imports(monkeypatch, allowed=()):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if (
            name == "vllm"
            or name.startswith("vllm.")
            or name == "vllm_ascend"
            or name.startswith("vllm_ascend.")
        ) and name not in allowed:
            pytest.fail(f"Premature native import: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_real_source_interfaces_are_checked_without_imports(monkeypatch):
    roots = reference_roots()
    forbid_native_imports(monkeypatch)
    result = validate_source_interfaces(roots)
    assert result["inspection_mode"] == "source_ast_no_imports"
    assert result["ParallelConfig.worker_extension_cls"] == "declared"
    assert result["AscendCommonAttentionMetadata"] == "required fields declared"


def test_preflight_checks_real_interfaces_without_executing_native_packages(
    monkeypatch, tmp_path, capsys
):
    roots = reference_roots()
    mock_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(
        check.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(
            origin=str(roots[name.replace("_", "-")] / name / "__init__.py")
        ),
    )
    forbid_native_imports(monkeypatch)
    # Crucially, validate_source_interfaces is NOT mocked in this test.
    assert check.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["source_audit"]["exact_reference_match"]
    assert report["interface_checks"]["inspection_mode"] == "source_ast_no_imports"
    assert report["physical_npu_devices"] == "4,5,6,7"


def test_static_preflight_still_rejects_an_incompatible_constructor(tmp_path, monkeypatch):
    roots = reference_roots()
    files = {
        "vllm": [
            "vllm/model_executor/layers/attention/attention.py",
            "vllm/entrypoints/llm.py",
            "vllm/v1/attention/backend.py",
            "vllm/config/parallel.py",
        ],
        "vllm-ascend": [
            "vllm_ascend/platform.py",
            "vllm_ascend/attention/attention_v1.py",
            "vllm_ascend/attention/utils.py",
        ],
    }
    copies = {name: tmp_path / name for name in roots}
    for name, paths in files.items():
        for relative in paths:
            path = copies[name] / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((roots[name] / relative).read_bytes())
    attention = copies["vllm"] / "vllm/model_executor/layers/attention/attention.py"
    attention.write_text(attention.read_text().replace("attn_backend:", "renamed_backend:", 1))
    forbid_native_imports(monkeypatch)
    with pytest.raises(ValueError, match="missing parameters attn_backend"):
        validate_source_interfaces(copies)


def test_calibration_registration_does_not_activate_native_modules(monkeypatch):
    monkeypatch.setenv("OSCAR_ASCEND_CALIBRATING", "1")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.23.0")
    monkeypatch.setattr(
        calibration_worker,
        "install_hook",
        lambda: pytest.fail("Hook installed before model initialization"),
    )
    forbid_native_imports(monkeypatch)
    plugin.register()


def test_serving_registration_defers_attention_until_native_runner_probe(monkeypatch):
    class Platform:
        @classmethod
        def get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads=None):
            return "native"

    module = ModuleType("vllm_ascend.platform")
    module.NPUPlatform = Platform
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.delitem(
        sys.modules, "vllm.model_executor.layers.attention.attention", raising=False
    )
    monkeypatch.setenv("OSCAR_ASCEND_ENABLED", "1")
    monkeypatch.setenv("OSCAR_ASCEND_CALIBRATING", "0")
    monkeypatch.setattr(plugin, "_registered", False)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.23.0")
    forbid_native_imports(monkeypatch, allowed=("vllm_ascend.platform",))
    plugin.register()
    assert not plugin._install_loaded_attention_hooks(Platform)

    class Attention:
        def __init__(self, num_heads, prefix="", attn_backend=None):
            self.prefix = prefix

        def process_weights_after_loading(self, act_dtype):
            pass

    original = Attention.__init__
    attention_module = ModuleType("vllm.model_executor.layers.attention.attention")
    attention_module.Attention = Attention
    attention_module.__spec__ = SimpleNamespace(_initializing=True)
    monkeypatch.setitem(sys.modules, attention_module.__name__, attention_module)
    selector = SimpleNamespace(
        attn_type="decoder", use_mla=False, use_sparse=False, use_non_causal=False, head_size=0
    )
    Platform.get_attn_backend_cls(None, selector)
    assert Attention.__init__ is original  # Do not mutate a partially initialized module.
    attention_module.__spec__._initializing = False
    Platform.get_attn_backend_cls(None, selector)
    wrapped = Attention.__init__
    assert wrapped is not original
    Platform.get_attn_backend_cls(None, selector)
    assert Attention.__init__ is wrapped
    assert Attention(6, prefix="model.layers.3.attn").prefix == "model.layers.3.attn"


def test_calibration_hook_requires_completed_native_model_import(monkeypatch):
    monkeypatch.setattr(calibration_worker, "_installed", False)
    monkeypatch.setattr(calibration_worker, "_phase", 0)
    monkeypatch.delitem(sys.modules, "vllm_ascend.attention.attention_v1", raising=False)
    forbid_native_imports(monkeypatch)
    with pytest.raises(RuntimeError, match="initialized before calibration"):
        calibration_worker.install_hook()

    class NativeImpl:
        def forward(self, layer, query, key, value, kv_cache, attn_metadata):
            return query

    module = ModuleType("vllm_ascend.attention.attention_v1")
    module.AscendAttentionBackendImpl = NativeImpl
    module.__spec__ = SimpleNamespace(_initializing=True)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(RuntimeError, match="initialized before calibration"):
        calibration_worker.install_hook()
    module.__spec__._initializing = False
    calibration_worker.install_hook()
    forward = NativeImpl.forward
    calibration_worker.install_hook()
    assert NativeImpl.forward is forward
    assert NativeImpl().forward(None, "native-result", None, None, None, None) == "native-result"


def test_device_operator_cycle_matches_reported_python_import_failure(tmp_path):
    # The same package-level dependency edges as the pinned Ascend source;
    # use tiny stand-ins so this exercises Python imports without an NPU.
    package = tmp_path / "cycle_fixture"
    files = {
        "__init__.py": "",
        "device_op.py": "from cycle_fixture.ops.triton import kernel\nclass DeviceOperator: pass\n",
        "ops/__init__.py": "from cycle_fixture.ops import fused_moe\n",
        "ops/fused_moe.py": "from cycle_fixture.device_op import DeviceOperator\n",
        "ops/triton/__init__.py": "kernel = None\n",
    }
    for relative, source in files.items():
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    bad = subprocess.run(
        [sys.executable, "-c", "import cycle_fixture.device_op"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert bad.returncode != 0 and "partially initialized module" in bad.stderr
    good = subprocess.run(
        [
            sys.executable,
            "-c",
            "import cycle_fixture.ops; from cycle_fixture.device_op import DeviceOperator",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert good.returncode == 0, good.stderr


def test_native_runner_imports_attention_before_its_backend_probe():
    import ast

    roots = reference_roots()
    tree = ast.parse((roots["vllm-ascend"] / "vllm_ascend/worker/model_runner_v1.py").read_text())
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "vllm.model_executor.layers.attention"
        and any(name.name == "Attention" for name in node.names)
        for node in tree.body
    )
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner"
    )
    init = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "get_attn_backend"
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == 0
        for node in ast.walk(init)
    )
