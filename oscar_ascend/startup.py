"""Scope native dummy runs without importing or initializing the NPU stack."""

import faulthandler
import functools
import inspect
import os
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace

_dummy_run = ContextVar("oscar_dummy_run", default=False)
_capturing = ContextVar("oscar_capturing", default=False)
_traced_ops = set()
_kernel_proxy = None


def in_dummy_run():
    return _dummy_run.get()


@contextmanager
def dummy_run_scope(capturing=None):
    token = _dummy_run.set(True)
    capture_token = _capturing.set(capturing) if capturing is not None else None
    try:
        yield
    finally:
        if capture_token is not None:
            _capturing.reset(capture_token)
        _dummy_run.reset(token)


def trace_enabled():
    return os.getenv("OSCAR_STARTUP_TRACE", "1") != "0"


def _log(message):
    print(f"[OSCAR startup pid={os.getpid()}] {message}", flush=True)


def trace_kernels(kernels, synchronize=None):
    """Check first warmup launches, never synchronize during capture or serving."""
    global _kernel_proxy
    if not trace_enabled():
        return kernels
    if _kernel_proxy is None:

        def wrap(name):
            function = getattr(kernels, name)

            @functools.wraps(function)
            def invoke(*args, **kwargs):
                label = name
                variant = ()
                if name == "rotate":
                    x, rotation = args
                    variant = (tuple(x.shape[1:]), x.stride(), rotation.stride())
                if name == "attention_partials":
                    label += ".raw" if kwargs["raw"] else ".history"
                    variant = (args[8],)
                key = (label, variant)
                if key in _traced_ops:
                    return function(*args, **kwargs)
                verify_device = synchronize is not None and not _capturing.get()
                started = time.monotonic()
                if verify_device:
                    _log(f"{label} {variant}: drain prior work begin")
                    synchronize()
                _log(f"{label}: first launch begin (may JIT compile)")
                result = function(*args, **kwargs)
                _log(f"{label}: launch returned in {time.monotonic() - started:.2f}s")
                if verify_device:
                    _log(f"{label}: waiting for device completion")
                    synchronize()
                    _log(f"{label}: device complete in {time.monotonic() - started:.2f}s")
                _traced_ops.add(key)
                return result

            return invoke

        _kernel_proxy = SimpleNamespace(
            **{
                name: wrap(name)
                for name in (
                    "rotate",
                    "store_int2",
                    "attention_partials",
                    "merge_splits",
                    "merge_paths",
                    "store_windows",
                )
            }
        )
    return _kernel_proxy


def install_runner_hooks():
    """Attach after the native runner module has finished its normal imports."""
    module = sys.modules.get("vllm_ascend.worker.model_runner_v1")
    if module is None or getattr(getattr(module, "__spec__", None), "_initializing", False):
        return False
    runner_class = getattr(module, "NPUModelRunner", None)
    if runner_class is None:
        return False
    if getattr(runner_class, "_oscar_startup_hooks", False):
        return True
    original_dummy = runner_class._dummy_run
    signature = inspect.signature(original_dummy)
    if not {"num_tokens", "is_graph_capturing"} <= signature.parameters.keys():
        raise RuntimeError("Unsupported NPUModelRunner._dummy_run signature")
    original_capture = runner_class.capture_model

    @functools.wraps(original_dummy)
    def dummy(runner, *args, **kwargs):
        bound = signature.bind(runner, *args, **kwargs)
        bound.apply_defaults()
        capturing = bound.arguments["is_graph_capturing"]
        trace = trace_enabled()
        if trace:
            phase = "capture" if capturing else "warmup/profile"
            label = f"{phase} tokens={bound.arguments['num_tokens']} device={runner.device}"
            started = time.monotonic()
            _log(f"{label}: begin")
        with dummy_run_scope(capturing=capturing):
            result = original_dummy(runner, *args, **kwargs)
        if trace:
            _log(f"{label}: returned in {time.monotonic() - started:.2f}s")
        return result

    @functools.wraps(original_capture)
    def capture(runner, *args, **kwargs):
        trace = trace_enabled()
        stack_file = None
        if trace:
            directory = Path(os.getenv("OSCAR_STARTUP_LOG_DIR", "artifacts/startup"))
            path = directory / f"worker-{os.getpid()}-stacks.log"
            try:
                directory.mkdir(parents=True, exist_ok=True)
                stack_file = path.open("a")
            except OSError as exc:
                _log(f"Cannot open stack log {path}: {exc}; using stderr")
                path = "stderr"
            _log(f"graph capture begin device={runner.device}; stalled stacks every 120s: {path}")
            # Also covers the native synchronize AFTER _dummy_run returns.
            faulthandler.dump_traceback_later(120, repeat=True, file=stack_file)
        try:
            result = original_capture(runner, *args, **kwargs)
        finally:
            if trace:
                faulthandler.cancel_dump_traceback_later()
                if stack_file is not None:
                    stack_file.close()
        if trace:
            _log(f"graph capture complete device={runner.device}")
        return result

    runner_class._dummy_run = dummy
    runner_class.capture_model = capture
    runner_class._oscar_startup_hooks = True
    return True
