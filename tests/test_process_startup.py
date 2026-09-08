import ast
import builtins
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from oscar_ascend.runtime_env import configure_process_environment

ROOT = Path(__file__).resolve().parents[1]


def test_policy_overrides_fork_before_any_torch_import(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in ("torch", "torch_npu", "vllm", "vllm_ascend"):
            pytest.fail(f"Process policy imported runtime: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    env = {
        "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3",
        "VLLM_WORKER_MULTIPROC_METHOD": "fork",
        "OMP_NUM_THREADS": "8",
        "LD_PRELOAD": "/user/existing/lib.so",
    }
    configure_process_environment(env)
    assert env == {
        "ASCEND_RT_VISIBLE_DEVICES": "4,5,6,7",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "OMP_NUM_THREADS": "8",
        "LD_PRELOAD": "/user/existing/lib.so",
    }


def test_calibrate_main_guard_prevents_recursive_loading_in_spawn(monkeypatch):
    monkeypatch.delitem(sys.modules, "oscar_ascend.calibrate", raising=False)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "vllm" or name.startswith("vllm."):
            pytest.fail("Spawn bootstrap must not import/create LLM from module top level")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    namespace = runpy.run_module("oscar_ascend.calibrate", run_name="__mp_main__")
    assert callable(namespace["main"])


def test_spawn_tree_is_fresh_after_parent_torch_and_background_threads():
    env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES="0,1,2,3", VLLM_WORKER_MULTIPROC_METHOD="fork")
    result = subprocess.run(
        [sys.executable, "-m", "tests.spawn_probe"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["fresh_engine"]
    assert report["engine_method"] == "spawn"
    assert report["worker"]["fresh"]
    assert report["worker"]["method"] == "spawn"
    assert report["worker"]["value"] == 240.0
    assert report["worker"]["devices"] == "4,5,6,7"
    assert report["worker"]["pid"] != report["engine_pid"]


def test_pinned_vllm_uses_shared_context_policy_for_both_levels():
    upstream = ROOT / "references/vllm/vllm"
    if not upstream.exists():
        pytest.skip("Read-only upstream sources are not shipped to deployment")
    util = ast.parse((upstream / "utils/system_utils.py").read_text())
    getter = next(
        n for n in util.body if isinstance(n, ast.FunctionDef) and n.name == "get_mp_context"
    )
    assert any(
        isinstance(n, ast.Attribute) and n.attr == "VLLM_WORKER_MULTIPROC_METHOD"
        for n in ast.walk(getter)
    )
    for relative, class_name, method_name in (
        ("v1/engine/utils.py", "CoreEngineProcManager", "__init__"),
        ("v1/executor/multiproc_executor.py", "MultiprocExecutor", "_init_executor"),
    ):
        tree = ast.parse((upstream / relative).read_text())
        owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        method = next(
            n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == method_name
        )
        assert any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "get_mp_context"
            for n in ast.walk(method)
        )
