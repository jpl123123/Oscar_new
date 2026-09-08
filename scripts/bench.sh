#!/usr/bin/env bash
# Benchmark an already-running server. Does not stop/restart anyone's service.
set -euo pipefail
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-/softwarePlatform/c00879303/Qwen3.5-27B-w8a8-mtp}"
LABEL="${LABEL:?Set LABEL=native, native-prefix-off, or oscar}"
BASE_URL="${BASE_URL:-http://127.0.0.1:5656}"
RESULT_DIR="${RESULT_DIR:-$PROJECT_DIR/artifacts/$LABEL}"
REPEATS="${REPEATS:-3}"
mkdir -p "$RESULT_DIR"
# A 16384 token batching limit ensures continuation on the long-prompt cases.
for scenario in 'short 1024 512 1 8' 'long 32768 512 1 8' 'mixed 16384 512 16 64' 'capacity 65536 256 4 16'; do
  read -r name input output concurrency prompts <<< "$scenario"
  for ((repeat=1; repeat<=REPEATS; repeat++)); do
    "$PYTHON_BIN" -m vllm.entrypoints.cli.main bench serve \
      --backend openai-chat --endpoint /v1/chat/completions --base-url "$BASE_URL" \
      --model qwen3.5 --tokenizer "$MODEL" --dataset-name random \
      --random-input-len "$input" --random-output-len "$output" --random-range-ratio 1 \
      --num-prompts "$prompts" --max-concurrency "$concurrency" --seed 42 \
      --ignore-eos --temperature 0 --request-rate inf \
      --percentile-metrics ttft,tpot,itl --metric-percentiles 50,99 \
      --save-result --save-detailed --result-dir "$RESULT_DIR" \
      --result-filename "$name-$repeat.json" --label "$LABEL" \
      --metadata "scenario=$name" "repeat=$repeat" "seed=42" "input=$input" "output=$output" \
      "mtp=3" "max_batched=16384" "model_path=$MODEL"
  done
done
