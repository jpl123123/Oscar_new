import importlib.util
import os

os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "4,5,6,7"

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--require-npu", action="store_true", help="Fail instead of skipping NPU tests"
    )


def pytest_sessionstart(session):
    if session.config.getoption("--require-npu"):
        if importlib.util.find_spec("torch_npu") is None:
            raise pytest.UsageError(
                "--require-npu: torch_npu is missing; NPU acceptance did not run"
            )
        import torch
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise pytest.UsageError("--require-npu: Ascend NPU is unavailable")
