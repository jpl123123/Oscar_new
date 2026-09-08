#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p artifacts
"$PYTHON_BIN" -m pytest tests/test_npu_kernels.py --require-npu -v \
  --junitxml=artifacts/npu-tests.xml "$@"
