"""Fresh process: load native TP4 model, calibrate twice, publish rotation pair, exit."""

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path


def main():
    from .runtime_env import configure_process_environment

    configure_process_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument("--profile-fingerprint", required=True)
    parser.add_argument("--data", default=os.getenv("OSCAR_CALIBRATION_DATA"))
    parser.add_argument(
        "--tokens", type=int, default=int(os.getenv("OSCAR_CALIBRATION_TOKENS", "1024"))
    )
    parser.add_argument(
        "--max-sweeps", type=int, default=int(os.getenv("OSCAR_CALIBRATION_SWEEPS", "12"))
    )
    args = parser.parse_args()
    if not 256 <= args.tokens <= 8192 or not 2 <= args.max_sweeps <= 32:
        raise ValueError("Calibration tokens must be 256..8192; Jacobi sweeps must be 2..32")
    os.environ["OSCAR_ASCEND_ENABLED"] = "0"
    os.environ["OSCAR_ASCEND_CALIBRATING"] = "1"
    # vLLM is imported only AFTER the calibration plugin mode is selected.
    from vllm import LLM, SamplingParams

    from .calibration_data import load_texts, token_prompts
    from .check import check_model

    layers = check_model(args.model)
    texts, source = load_texts(args.data)
    print(
        f"[OSCAR calibration] Loading native BF16 KV model; data={source}, prompts={len(texts)}",
        flush=True,
    )
    print("[OSCAR calibration] EngineCore/TP start method=spawn; physical NPUs=4,5,6,7", flush=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=4,
        worker_extension_cls="oscar_ascend.calibration_worker.CalibrationWorkerExtension",
        dtype="bfloat16",
        quantization="ascend",
        trust_remote_code=True,
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=max(4096, args.tokens + 1),
        max_num_batched_tokens=min(2048, args.tokens),
        max_num_seqs=1,
        gpu_memory_utilization=float(os.getenv("OSCAR_CALIBRATION_MEMORY", "0.75")),
        async_scheduling=False,
        mamba_cache_dtype="bfloat16",
        mamba_ssm_cache_dtype="bfloat16",
        compilation_config={"mode": 0, "cudagraph_mode": "NONE"},
        additional_config={"enable_cpu_binding": True},
        hf_overrides={
            "text_config": {
                "rope_parameters": {
                    "mrope_interleaved": True,
                    "mrope_section": [11, 11, 10],
                    "rope_type": "yarn",
                    "rope_theta": 10000000,
                    "partial_rotary_factor": 0.25,
                    "factor": 4.0,
                    "original_max_position_embeddings": 262144,
                }
            }
        },
    )
    try:
        prompts = token_prompts(
            llm.get_tokenizer(), texts, args.tokens, builtin=source.startswith("builtin")
        )
        budget = sum(len(p["prompt_token_ids"]) for p in prompts)
        sampling = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, seed=0)
        provenance = {
            "model_fingerprint": args.model_fingerprint,
            "data_source": source,
            "profile_fingerprint": args.profile_fingerprint,
            "run_id": uuid.uuid4().hex,
            "prompt_sha256": hashlib.sha256(
                json.dumps(prompts, sort_keys=True).encode()
            ).hexdigest(),
            "prompt_tokens": budget,
            "generator_version": 1,
        }
        llm.collective_rpc("oscar_calibration_begin", args=(layers, budget))
        print("[OSCAR calibration] Pass 1/2: Q covariance", flush=True)
        llm.generate(prompts, sampling, use_tqdm=True)
        llm.collective_rpc("oscar_calibration_second_pass")
        print("[OSCAR calibration] Pass 2/2: SST weighted V covariance", flush=True)
        llm.generate(prompts, sampling, use_tqdm=True)
        print("[OSCAR calibration] Computing rotations on NPU", flush=True)
        results = llm.collective_rpc(
            "oscar_calibration_finish", args=(str(args.output_dir), provenance, args.max_sweeps)
        )
        if sum(bool(x.get("published")) for x in results) != 1:
            raise RuntimeError("Calibration did not publish exactly one K/V pair")
        print(f"[OSCAR calibration] Saved K/V rotations to {args.output_dir}", flush=True)
    finally:
        # Release the first model and workers before serve.sh starts the server.
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
