"""Native capture argument equivalence without permitting device-to-host reads."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from oscar_ascend import capture_metadata as capture
from oscar_ascend import startup

ROOT = Path(__file__).resolve().parents[1]


class DeviceValue(torch.Tensor):
    def cpu(self, *args, **kwargs):
        raise AssertionError("Device-to-host copy attempted")

    def item(self, *args, **kwargs):
        raise AssertionError("Device scalar read attempted")

    def tolist(self, *args, **kwargs):
        raise AssertionError("Device list read attempted")

    def numpy(self, *args, **kwargs):
        raise AssertionError("Device NumPy read attempted")


class RecordingBuilder:
    decode_cudagraph_max_bs = 512

    def build(self, prefix, meta, accepted, drafts):
        return prefix, meta, accepted, drafts


def common(starts):
    host = torch.tensor(starts, dtype=torch.int32)
    return SimpleNamespace(
        num_reqs=len(starts) - 1,
        num_actual_tokens=starts[-1],
        query_start_loc=host.clone().as_subclass(DeviceValue),
        query_start_loc_cpu=host,
    )


@pytest.mark.parametrize("starts", [[0, 4, 8, 8], [0, 1, 2], [0, 4], [0, 0]])
def test_capture_arguments_match_native_without_d2h(starts):
    meta = common(starts)
    prefix, same, accepted, drafts = capture.build_gdn_capture_metadata(RecordingBuilder(), meta)
    assert prefix == 0 and same is meta
    expected = torch.diff(meta.query_start_loc_cpu)
    torch.testing.assert_close(accepted.as_subclass(torch.Tensor), expected)
    torch.testing.assert_close(drafts, expected - 1)
    assert drafts.device.type == "cpu"
    assert isinstance(accepted, DeviceValue)


def test_native_reference_reproduces_the_unnecessary_d2h():
    path = ROOT / "references/vllm/vllm/v1/attention/backends/gdn_attn.py"
    if not path.exists():
        pytest.skip("Read-only reference is not distributed with deployment")
    tree = ast.parse(path.read_text())
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "build_for_cudagraph_capture"
    )
    method.decorator_list = []
    for arg in method.args.args:
        arg.annotation = None
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    with pytest.raises(AssertionError, match="Device-to-host"):
        namespace["build_for_cudagraph_capture"](RecordingBuilder(), common([0, 4, 8, 8]))


def test_missing_or_mismatched_host_boundaries_never_fall_back_to_d2h():
    meta = common([0, 4])
    meta.query_start_loc_cpu = torch.empty(2, device="meta", dtype=torch.int32)
    with pytest.raises(ValueError, match="CPU query boundaries"):
        capture.build_gdn_capture_metadata(RecordingBuilder(), meta)
    meta.query_start_loc_cpu = torch.zeros(3, dtype=torch.int32)
    with pytest.raises(ValueError, match="CPU query boundaries"):
        capture.build_gdn_capture_metadata(RecordingBuilder(), meta)
    meta = common([0, 513])
    with pytest.raises(ValueError, match="exceeds"):
        capture.build_gdn_capture_metadata(RecordingBuilder(), meta)


def test_hook_defers_imports_and_changes_only_inherited_capture_method(monkeypatch):
    class NativeBase(RecordingBuilder):
        def build_for_cudagraph_capture(self, common_attn_metadata):
            raise AssertionError("Inherited D2H method must not be called")

    class AscendBuilder(NativeBase):
        pass

    module = ModuleType("vllm_ascend.ops.gdn_attn_builder")
    module.AscendGDNAttentionMetadataBuilder = AscendBuilder
    module.__spec__ = SimpleNamespace(_initializing=True)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    original = NativeBase.build_for_cudagraph_capture
    assert not capture.install_gdn_capture_hook()
    assert AscendBuilder.build_for_cudagraph_capture is original
    module.__spec__._initializing = False
    assert capture.install_gdn_capture_hook()
    wrapped = AscendBuilder.build_for_cudagraph_capture
    assert capture.install_gdn_capture_hook()
    assert AscendBuilder.build_for_cudagraph_capture is wrapped
    assert NativeBase.build_for_cudagraph_capture is original
    assert AscendBuilder.build is RecordingBuilder.build
    assert AscendBuilder().build_for_cudagraph_capture(common([0, 4]))[3].tolist() == [3]

    class OwnCapture(NativeBase):
        def build_for_cudagraph_capture(self, common_attn_metadata):
            return "platform override"

    module.AscendGDNAttentionMetadataBuilder = OwnCapture
    assert not capture.install_gdn_capture_hook()
    assert OwnCapture().build_for_cudagraph_capture(None) == "platform override"


def test_returned_capture_method_does_not_claim_graph_acceptance(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OSCAR_STARTUP_TRACE", "1")
    monkeypatch.setenv("OSCAR_STARTUP_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(startup.faulthandler, "dump_traceback_later", lambda *a, **kw: None)
    monkeypatch.setattr(startup.faulthandler, "cancel_dump_traceback_later", lambda: None)

    class Runner:
        device = "npu:0"

        def _dummy_run(self, num_tokens, is_graph_capturing=False):
            return None

        def capture_model(self):
            return 0  # For example, graph mode was disabled; no graph was recorded.

    module = ModuleType("vllm_ascend.worker.model_runner_v1")
    module.NPUModelRunner = Runner
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert startup.install_runner_hooks()
    assert Runner().capture_model() == 0
    output = capsys.readouterr().out
    assert "capture_model returned" in output
    assert "acceptance are not established" in output
    assert "graph capture complete" not in output
