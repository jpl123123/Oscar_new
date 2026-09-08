"""Temporary native-attention hook used only by a separate calibration process."""

import functools
import os
from pathlib import Path

_installed = False
_phase = 0
_states = {}
_budget = 0


class CalibrationWorkerExtension:
    """Named RPCs transported as strings by vLLM's default message serializer."""

    def oscar_calibration_begin(self, layers, token_budget):
        return begin(self, layers, token_budget)

    def oscar_calibration_second_pass(self):
        return second_pass(self)

    def oscar_calibration_finish(self, output_dir, provenance, max_sweeps):
        return finish(self, output_dir, provenance, max_sweeps)


def install_hook():
    global _installed
    if _installed:
        return
    from vllm_ascend import _ensure_global_patch

    _ensure_global_patch()
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

    original = AscendAttentionBackendImpl.forward

    @functools.wraps(original)
    def forward(impl, layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs):
        result = original(impl, layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs)
        if _phase and attn_metadata is not None and key is not None and value is not None:
            observe(impl, layer, query, key, value, attn_metadata.num_actual_tokens)
        return result

    AscendAttentionBackendImpl.forward = forward
    _installed = True


def observe(impl, layer, query, key, value, num_tokens):
    from . import calibration_kernels as ops
    from .plugin import is_draft_layer
    from .rotations import layer_index

    if is_draft_layer(layer.layer_name) or impl.head_size != 256 or impl.attn_type != "decoder":
        return
    index = layer_index(layer.layer_name)
    if index not in _states:
        return
    if impl.num_heads != 6 or impl.num_kv_heads != 1:
        raise ValueError("Calibration requires target TP4 geometry: Hq=6, Hkv=1")
    state = _states[index]
    counter = "q_tokens" if _phase == 1 else "v_tokens"
    limit = _budget if _phase == 1 else state["q_tokens"]
    # RPC result metadata must contain Python integers, even if a native
    # scheduler supplies NumPy integer counts.
    take = int(min(num_tokens, query.shape[0], limit - state[counter]))
    if take <= 0:
        return
    if query.device.type != "npu":
        raise ValueError("Calibration Q/K/V must stay on NPU")
    if _phase == 1:
        q = query[:take].view(take, 6, 256)
        ops.accumulate_gram(q, state["q_sum"])
    else:
        k = key[:take].view(take, 1, 256)
        v = value[:take].view(take, 1, 256)
        weights = ops.sst_weights(k, state["q_cov"], state["weight_sum"])
        ops.accumulate_gram(v, state["v_sum"], weights)
    state[counter] += take


def begin(worker, layers, token_budget):
    """collective_rpc control message; model tensors are never returned by RPC."""
    global _phase, _states, _budget
    import torch

    if not _installed:
        raise RuntimeError("Calibration plugin was not loaded in this worker")
    _budget = token_budget
    device = torch.device("npu", torch.npu.current_device())
    _states = {
        layer: {
            "q_sum": torch.zeros((256, 256), dtype=torch.float32, device=device),
            "v_sum": torch.zeros((256, 256), dtype=torch.float32, device=device),
            "weight_sum": torch.zeros((), dtype=torch.float32, device=device),
            "q_tokens": 0,
            "v_tokens": 0,
        }
        for layer in layers
    }
    _phase = 1
    return {"phase": 1, "layers": len(layers)}


def second_pass(worker):
    global _phase
    from . import calibration_kernels as ops

    counts = {index: state["q_tokens"] for index, state in _states.items()}
    if not counts or min(counts.values()) < 256:
        raise RuntimeError(f"Insufficient real-model Q samples; no rotations generated: {counts}")
    for state in _states.values():
        state["q_cov"] = ops.normalize(state["q_sum"], state["q_tokens"] * 6)
    _phase = 2
    return {"phase": 2, "tokens_per_layer": counts}


def finish(worker, output_dir, provenance, max_sweeps):
    global _phase
    import torch
    from vllm.distributed import get_tensor_model_parallel_rank, tensor_model_parallel_all_reduce

    from . import calibration_kernels as ops

    _phase = 0
    rank = get_tensor_model_parallel_rank()
    matrices = []
    for index, state in sorted(_states.items()):
        if state["v_tokens"] != state["q_tokens"] or state["q_tokens"] < 256:
            raise RuntimeError(f"Calibration pass sample mismatch at layer {index}")
        # Startup-only scalar checks. No raw activation transfer or CPU eigensolver.
        if not state["weight_sum"].isfinite().item() or state["weight_sum"].item() <= 0:
            raise RuntimeError(f"Degenerate SST weights at layer {index}")
        matrices.extend((state["q_cov"], ops.normalize(state["v_sum"], state["weight_sum"])))
    covariances = tensor_model_parallel_all_reduce(torch.stack(matrices))
    covariances = ops.normalize(covariances, 4)
    if rank != 0:
        return {"rank": rank, "published": False}
    rotations, eigenvalues, convergence = ops.calibrated_rotations(
        covariances, max_sweeps=max_sweeps
    )
    # These are final small artifacts (~8 MiB), not Q/K/V dumps.
    rotations_cpu, eigenvalues_cpu = rotations.cpu(), eigenvalues.cpu()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for kind, offset, objective in (("k", 0, "qqt_r_h_pbr"), ("v", 1, "sst_r_h_pbr")):
        result = {
            "format_version": 1,
            "source_grouping": "layer",
            "objective": objective,
            "model_fingerprint": provenance["model_fingerprint"],
            "calibration": {
                **provenance,
                **convergence,
                "arithmetic": "triton_ascend",
                "tp_size": 4,
                "tokens_per_layer": {i: s["q_tokens"] for i, s in _states.items()},
            },
            "layers": {},
        }
        for number, index in enumerate(sorted(_states)):
            matrix = rotations_cpu[number * 2 + offset]
            if not torch.isfinite(matrix).all() or not torch.allclose(
                matrix.T @ matrix, torch.eye(256), atol=0.005, rtol=0.005
            ):
                raise RuntimeError(f"Generated {kind} rotation for layer {index} failed validation")
            result["layers"][index] = {
                "layer_id": index,
                "rotation": matrix,
                "eigenvalues": eigenvalues_cpu[number * 2 + offset],
            }
        path = output / f"{kind}_rotation.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save(result, temporary)
        os.replace(temporary, path)
    return {"rank": 0, "published": True, **convergence}
