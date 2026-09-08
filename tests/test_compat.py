"""Version/build-label regressions; mocked preflight never claims real NPU execution."""

import importlib.metadata
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from oscar_ascend import check
from oscar_ascend.compat import (
    require_parameters,
    validate_hook_interfaces,
    validate_triton_target,
    validate_versions,
)

REPORTED = {"vllm": "0.23.0+empty", "vllm-ascend": "0.23.1.dev0+g5cb98caaa.d20260822"}


@pytest.mark.parametrize(
    "ascend",
    ["0.23.0", "0.23.0+custom", "0.23.1", "0.23.1.dev0", "0.23.1.dev0+g5cb98caaa.d20260822"],
)
def test_source_build_versions_normalize(ascend):
    result = validate_versions({**REPORTED, "vllm-ascend": ascend})
    assert result["vllm"] == "0.23.0"
    assert result["vllm-ascend"] in ("0.23.0", "0.23.1")


@pytest.mark.parametrize(
    "package,value",
    [
        ("vllm", "0.24.0"),
        ("vllm", "0.23.1"),
        ("vllm-ascend", "0.23.10"),
        ("vllm-ascend", "0.24.0.dev0"),
        ("vllm-ascend", "unknown"),
        ("vllm-ascend", "1!0.23.1"),
    ],
)
def test_unrelated_or_invalid_versions_still_rejected(package, value):
    with pytest.raises(ValueError):
        validate_versions({**REPORTED, package: value})


def test_missing_hook_api_is_actionable():
    class Platform:
        @classmethod
        def get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads=None):
            pass

    class IncompatibleAttention:
        def __init__(self, prefix=""):
            pass

    with pytest.raises(ValueError, match="attn_backend"):
        validate_hook_interfaces(Platform, IncompatibleAttention)
    with pytest.raises(ValueError, match="Missing required interface"):
        require_parameters("LLM.collective_rpc", None, ("method",))


def test_source_drift_is_reported_but_strict_reference_audit_still_fails(tmp_path):
    roots = {name: tmp_path / name for name in ("vllm", "vllm-ascend")}
    report = check.audit_sources(roots)
    assert not report["exact_reference_match"]
    assert report["mismatches"]
    assert "reference_commits" in report  # Do not label the reference SHA as the installed commit.
    with pytest.raises(ValueError, match="source mismatch"):
        check.verify_sources(roots)


def mock_runtime(monkeypatch, tmp_path, backend="npu"):
    npu = ModuleType("torch_npu")
    npu.__version__ = "2.10.0.post4"
    triton = ModuleType("triton")
    triton.__version__ = "3.2.0"
    triton.runtime = SimpleNamespace(
        driver=SimpleNamespace(
            active=SimpleNamespace(
                get_current_target=lambda: SimpleNamespace(
                    backend=backend, arch="Ascend910B4", warp_size=0
                )
            )
        )
    )
    monkeypatch.setitem(sys.modules, "torch_npu", npu)
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setattr(torch, "__version__", "2.10.0+cpu")
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(is_available=lambda: True, device_count=lambda: 4),
        raising=False,
    )
    monkeypatch.setattr(importlib.metadata, "version", lambda name: REPORTED[name])
    monkeypatch.setattr(
        check.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(tmp_path / name / name / "__init__.py")),
    )
    monkeypatch.setattr(check, "check_model", lambda model: list(range(3, 64, 4)))
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setattr(sys, "argv", ["check", "--model", "/model", "--skip-rotations"])


@pytest.mark.parametrize("backend", ["npu", "ascend"])
def test_reported_stack_reaches_interface_checks_despite_source_build_labels(
    monkeypatch, tmp_path, capsys, backend
):
    mock_runtime(monkeypatch, tmp_path, backend)
    calls = []
    monkeypatch.setattr(
        check, "validate_runtime_interfaces", lambda: calls.append(True) or {"mock": "passed"}
    )
    assert check.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["normalized_releases"] == {"vllm": "0.23.0", "vllm-ascend": "0.23.1"}
    assert result["versions"]["torch"] == "2.10.0+cpu"
    assert not result["source_audit"]["exact_reference_match"]
    assert result["physical_npu_devices"] == "4,5,6,7"
    assert result["npu_acceptance"] == "not_run" and calls == [True]
    assert result["triton_target_info"] == {
        "backend": backend,
        "arch": "Ascend910B4",
        "warp_size": 0,
    }


@pytest.mark.parametrize("backend", ["cuda", "hip", "cpu", None])
def test_other_triton_backends_are_rejected(backend):
    with pytest.raises(ValueError, match="Triton NPU"):
        validate_triton_target(SimpleNamespace(backend=backend, arch="Ascend910B4"))


def test_incompatible_api_stops_preflight_even_with_accepted_version(monkeypatch, tmp_path, capsys):
    mock_runtime(monkeypatch, tmp_path)

    def incompatible():
        raise ValueError("AscendCommonAttentionMetadata missing fields: slot_mapping")

    monkeypatch.setattr(check, "validate_runtime_interfaces", incompatible)
    assert check.main() == 1
    assert "slot_mapping" in json.loads(capsys.readouterr().out)["error"]


def test_reviewed_ascend_base_and_pr_share_adapter_seams():
    import hashlib
    import subprocess

    root = Path(__file__).resolve().parents[1]
    upstream = root / "references/vllm-ascend"
    if not (upstream / ".git").exists():
        pytest.skip("Read-only upstream clone is not shipped with the deployment")
    manifest = json.loads((root / "oscar_ascend/upstream_fingerprints.json").read_text())
    for relative, expected in manifest["vllm-ascend"]["files"].items():
        content = subprocess.run(
            ["git", "-C", str(upstream), "show", f"5cb98caaa:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(content).hexdigest() == expected
