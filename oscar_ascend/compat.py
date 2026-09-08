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
