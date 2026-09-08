#!/usr/bin/env bash
# Run inside the existing vLLM 0.23.0 + Ascend 0.23.0/0.23.1 environment.
set -euo pipefail
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OSCAR_ASCEND_CALIBRATING=0
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export OSCAR_STARTUP_LOG_DIR="${OSCAR_STARTUP_LOG_DIR:-$PROJECT_DIR/artifacts/startup}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-/softwarePlatform/c00879303/Qwen3.5-27B-w8a8-mtp}"
MODE="${MODE:-oscar}"
PORT="${PORT:-5656}"

case "$MODE" in
  oscar) export OSCAR_ASCEND_ENABLED=1 ;;
  native|native-prefix-off) export OSCAR_ASCEND_ENABLED=0 ;;
  *) echo "MODE must be oscar, native or native-prefix-off" >&2; exit 2 ;;
esac

if [[ "$MODE" == oscar && -n "${VLLM_PLUGINS:-}" ]]; then
  case ",$VLLM_PLUGINS," in
    *,oscar_ascend,*) ;;
    *) export VLLM_PLUGINS="$VLLM_PLUGINS,oscar_ascend" ;;
  esac
fi

export VLLM_OSCAR_GROUP_SIZE="${VLLM_OSCAR_GROUP_SIZE:-256}"
export VLLM_OSCAR_SINK_TOKENS="${VLLM_OSCAR_SINK_TOKENS:-64}"
export VLLM_OSCAR_RECENT_TOKENS="${VLLM_OSCAR_RECENT_TOKENS:-256}"
export VLLM_OSCAR_K_CLIP_RATIO="${VLLM_OSCAR_K_CLIP_RATIO:-0.96}"
export VLLM_OSCAR_V_CLIP_RATIO="${VLLM_OSCAR_V_CLIP_RATIO:-0.92}"

cmd=("$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "$MODEL"
  --served-model-name qwen3.5 --host 0.0.0.0 --port "$PORT"
  --data-parallel-size 1 --tensor-parallel-size 4
  --max-model-len "${MAX_MODEL_LEN:-262144}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}"
  --max-num-seqs "${MAX_NUM_SEQS:-128}" --gpu-memory-utilization 0.9
  --compilation-config '{"cudagraph_capture_sizes":[1,4,8,12,16,24,32,48,56,64,72,84,96,108,112,128,160,172,196,200,212,232,272,288,312,328,344,360,384,400,416,432,448,480,512],"cudagraph_mode":"FULL_DECODE_ONLY"}'
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}'
  --trust-remote-code --async-scheduling --allowed-local-media-path /
  --quantization ascend --mm-processor-cache-gb 0
  --additional-config '{"enable_cpu_binding":true}'
  --mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype bfloat16
  --hf-overrides '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}')

if [[ "$MODE" != native ]]; then
  cmd+=(--no-enable-prefix-caching)
fi
if [[ "$MODE" == oscar ]]; then
  cmd+=(--kv-cache-dtype auto --dtype bfloat16)
fi
if [[ "${OSCAR_ENFORCE_EAGER:-0}" == 1 ]]; then
  cmd+=(--enforce-eager)
fi
cmd+=("$@")

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'MODE=%s OSCAR_ASCEND_ENABLED=%s ASCEND_RT_VISIBLE_DEVICES=%s VLLM_WORKER_MULTIPROC_METHOD=%s\n' "$MODE" "$OSCAR_ASCEND_ENABLED" "$ASCEND_RT_VISIBLE_DEVICES" "$VLLM_WORKER_MULTIPROC_METHOD"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

if [[ "$MODE" == oscar ]]; then
  # Install this small external package only; do not replace torch/vllm/Ascend.
  "$PYTHON_BIN" -m pip install --no-deps --no-build-isolation -e "$PROJECT_DIR"
  rotation_state_file="$(mktemp "${TMPDIR:-/tmp}/oscar-rotation-paths.XXXXXX")"
  trap 'rm -f "$rotation_state_file"' EXIT
  "$PYTHON_BIN" -m oscar_ascend.prepare_rotations --model "$MODEL" \
    --cache-root "${OSCAR_ROTATION_DIR:-$PROJECT_DIR/artifacts/rotations}" \
    --output-json "$rotation_state_file"
  VLLM_OSCAR_K_ROTATION_PATH="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["k"])' "$rotation_state_file")"
  VLLM_OSCAR_V_ROTATION_PATH="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["v"])' "$rotation_state_file")"
  export VLLM_OSCAR_K_ROTATION_PATH VLLM_OSCAR_V_ROTATION_PATH
  rm -f "$rotation_state_file"
  trap - EXIT
fi
exec "${cmd[@]}"
