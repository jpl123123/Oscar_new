"""Config and native declaration regressions; these do not execute ACLGraph."""

import ast
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from oscar_ascend import startup
from tests.test_graph_startup import install_fake_runner

ROOT = Path(__file__).resolve().parents[1]


class Compile(IntEnum):
    NONE = 0
    VLLM_COMPILE = 3


class Graph(IntEnum):
    NONE = 0
    FULL_DECODE_ONLY = 1

    def has_full_cudagraphs(self):
        return self == self.FULL_DECODE_ONLY


def native_predicate():
    path = ROOT / "references/vllm-ascend/vllm_ascend/worker/model_runner_v1.py"
    if not path.exists():
        pytest.skip("Read-only references are not distributed with deployment")
    tree = ast.parse(path.read_text())
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_use_aclgraph"
    )
    ns = {"CompilationMode": Compile, "CUDAGraphMode": Graph}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    return ns["_use_aclgraph"]


def test_direct_full_mode_initializes_native_graph_parameters(monkeypatch):
    class Runner:
        _use_aclgraph = native_predicate()
        compilation_config = SimpleNamespace(
            mode=Compile.NONE, cudagraph_mode=Graph.FULL_DECODE_ONLY
        )
        model_config = SimpleNamespace(enforce_eager=False)

        def _dummy_run(self, num_tokens, is_graph_capturing=False):
            return None

        def capture_model(self):
            return 0

    runner = Runner()
    assert not runner._use_aclgraph()  # The reference incorrectly ties this to FX.
    install_fake_runner(monkeypatch, Runner)
    assert runner._use_aclgraph()
    runner.model_config.enforce_eager = True
    assert not runner._use_aclgraph()
    runner.model_config.enforce_eager = False
    runner.compilation_config.cudagraph_mode = Graph.NONE
    assert not runner._use_aclgraph()
    runner.compilation_config.cudagraph_mode = Graph.FULL_DECODE_ONLY
    runner.compilation_config.mode = Compile.VLLM_COMPILE
    assert runner._use_aclgraph()


def test_operator_compile_setup_runs_once_and_never_forces_eager_to_graph(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(set_compile_mode=lambda **kw: calls.append(kw)), raising=False
    )
    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(
            mode=Compile.NONE, cudagraph_mode=Graph.FULL_DECODE_ONLY
        ),
        model_config=SimpleNamespace(enforce_eager=True),
        use_aclgraph=False,
    )
    startup.configure_graph_runtime(runner)
    assert not calls and not runner.use_aclgraph
    runner.model_config.enforce_eager = False
    startup.configure_graph_runtime(runner)
    startup.configure_graph_runtime(runner)
    assert calls == [{"jit_compile": False}] and runner.use_aclgraph


def test_native_full_wrapper_and_capture_are_independent_of_fx_mode():
    path = ROOT / "references/vllm-ascend/vllm_ascend/worker/model_runner_v1.py"
    if not path.exists():
        pytest.skip("Read-only references are not distributed with deployment")
    tree = ast.parse(path.read_text())
    wrapping = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "self.compilation_config.cudagraph_mode.has_full_cudagraphs()"
        and any(
            isinstance(x, ast.Call)
            and isinstance(x.func, ast.Name)
            and x.func.id == "ACLGraphWrapper"
            for x in ast.walk(n)
        )
    )
    assert "compilation_config.mode" not in ast.unparse(wrapping.test)
    tree = ast.parse((ROOT / "references/vllm/vllm/v1/worker/gpu_model_runner.py").read_text())
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "capture_model"
    )
    first_if = next(n for n in fn.body if isinstance(n, ast.If))
    assert (
        ast.unparse(first_if.test) == "self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE"
    )
