"""Shared version policy and the interfaces actually consumed by the adapter."""

import inspect

from packaging.version import InvalidVersion, Version

SUPPORTED_RELEASES = {
    "vllm": {(0, 23, 0)},
    "vllm-ascend": {(0, 23, 0), (0, 23, 1)},
}


def validate_triton_target(target):
    """Triton Ascend uses 'npu' in 3.2 builds; some distributions use 'ascend'.

    Keep the driver's actual target intact. SIMD targets legitimately report
    warp_size=0, which must not be judged using CUDA warp assumptions.
    """
    backend = getattr(target, "backend", None)
    if backend not in ("npu", "ascend"):
        raise ValueError(f"Expected Triton NPU backend ('npu' or 'ascend'), got {backend!r}")
    return {
        "backend": backend,
        "arch": str(getattr(target, "arch", "")),
        "warp_size": getattr(target, "warp_size", None),
    }


def validate_versions(versions):
    """Accept dev/local build labels without confusing 0.23.1 with 0.23.10."""
    normalized = {}
    for package, releases in SUPPORTED_RELEASES.items():
        installed = versions.get(package, "")
        try:
            parsed = Version(installed)
        except InvalidVersion as exc:
            raise ValueError(f"Unrecognized {package} version: {installed!r}") from exc
        if parsed.epoch != 0 or parsed.release not in releases:
            expected = ", ".join(".".join(map(str, r)) for r in sorted(releases))
            raise ValueError(
                f"Unsupported {package} release: {installed}; expected {expected} including dev/local builds"
            )
        normalized[package] = parsed.base_version
    return normalized


def require_parameters(label, method, required):
    if not callable(method):
        raise ValueError(f"Missing required interface: {label}")
    signature = inspect.signature(method)
    missing = set(required) - set(signature.parameters)
    if missing:
        raise ValueError(
            f"Incompatible interface {label}{signature}; missing parameters: {', '.join(sorted(missing))}"
        )
    return str(signature)


def validate_hook_interfaces(platform, attention):
    return {
        "NPUPlatform.get_attn_backend_cls": require_parameters(
            "NPUPlatform.get_attn_backend_cls",
            getattr(platform, "get_attn_backend_cls", None),
            ("selected_backend", "attn_selector_config", "num_heads"),
        ),
        "Attention.__init__": require_parameters(
            "Attention.__init__",
            attention.__init__,
            ("prefix", "attn_backend"),
        ),
        "Attention.process_weights_after_loading": require_parameters(
            "Attention.process_weights_after_loading",
            getattr(attention, "process_weights_after_loading", None),
            ("act_dtype",),
        ),
    }


def validate_runtime_interfaces():
    """Inspect installed APIs without loading a model or allocating a KV cache.

    Source identity is diagnostic. These checks plus runtime tensor geometry
    checks determine whether a source-build variant can proceed to execution.
    They are not a claim of numerical or performance validation.
    """
    from vllm_ascend import _ensure_global_patch

    _ensure_global_patch()
    from vllm import LLM
    from vllm.config.parallel import ParallelConfig
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.v1.attention.backend import AttentionMetadataBuilder
    from vllm_ascend.attention.attention_v1 import (
        AscendAttentionBackend,
        AscendAttentionBackendImpl,
    )
    from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
    from vllm_ascend.platform import NPUPlatform

    checks = validate_hook_interfaces(NPUPlatform, Attention)
    if not hasattr(ParallelConfig, "worker_extension_cls"):
        raise ValueError("ParallelConfig.worker_extension_cls is required for calibration RPC")
    checks["ParallelConfig.worker_extension_cls"] = "available"
    for label, method, required in (
        ("LLM.collective_rpc", getattr(LLM, "collective_rpc", None), ("method", "args", "kwargs")),
        (
            "AttentionMetadataBuilder.build_for_cudagraph_capture",
            getattr(AttentionMetadataBuilder, "build_for_cudagraph_capture", None),
            ("common_attn_metadata",),
        ),
        (
            "AscendAttentionBackendImpl.forward",
            getattr(AscendAttentionBackendImpl, "forward", None),
            ("layer", "query", "key", "value", "kv_cache", "attn_metadata", "output"),
        ),
    ):
        checks[label] = require_parameters(label, method, required)
    fields = set()
    for cls in AscendCommonAttentionMetadata.__mro__:
        fields.update(getattr(cls, "__annotations__", {}))
    required_fields = {
        "query_start_loc",
        "query_start_loc_cpu",
        "seq_lens",
        "block_table_tensor",
        "slot_mapping",
        "num_reqs",
        "num_actual_tokens",
        "max_query_len",
        "attn_state",
        "causal",
    }
    if missing := required_fields - fields:
        raise ValueError(
            f"AscendCommonAttentionMetadata missing fields: {', '.join(sorted(missing))}"
        )
    checks["AscendCommonAttentionMetadata"] = "required fields present"
    shape = AscendAttentionBackend.get_kv_cache_shape(3, 128, 1, 256)
    if tuple(shape) != (2, 3, 128, 1, 256):
        raise ValueError(f"Unsupported native K/V shape contract: {shape}")
    if 128 not in AscendAttentionBackend.get_supported_kernel_block_sizes():
        raise ValueError("Native attention does not support the required 128-token kernel blocks")
    checks["native_cache_shape"] = list(shape)
    return checks
