#!/usr/bin/env bash
set -euo pipefail
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p artifacts
"$PYTHON_BIN" -m pytest tests/test_npu_kernels.py tests/test_npu_calibration.py --require-npu -v \
  --junitxml=artifacts/npu-tests.xml "$@"
