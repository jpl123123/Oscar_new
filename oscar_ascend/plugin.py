"""Idempotent general plugin. All hooks are in-memory and construction-only."""

import functools
import inspect
import logging
import os

_registered = False
LOG = logging.getLogger(__name__)


def should_route(selector):
    """Keep vision/encoder, MLA, sparse and noncausal attention native."""
    return (
        selector.attn_type == "decoder"
        and not selector.use_mla
        and not selector.use_sparse
        and not selector.use_non_causal
    )


def is_draft_layer(prefix):
    return "mtp" in prefix.split(".")


def register():
    global _registered
    calibrating = os.getenv("OSCAR_ASCEND_CALIBRATING", "0") == "1"
    if not calibrating and (_registered or os.getenv("OSCAR_ASCEND_ENABLED", "0") != "1"):
        return
    from importlib.metadata import version

    from .compat import validate_hook_interfaces, validate_versions

    validate_versions({name: version(name) for name in ("vllm", "vllm-ascend")})
    if calibrating:
        from .calibration_worker import install_hook

        install_hook()
        return
    # Apply Ascend's own startup patches before resolving its platform class.
    from vllm_ascend import _ensure_global_patch

    _ensure_global_patch()
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm_ascend.platform import NPUPlatform

    validate_hook_interfaces(NPUPlatform, Attention)
    original_select = NPUPlatform.get_attn_backend_cls.__func__

    @classmethod
    @functools.wraps(original_select)
    def select(cls, selected_backend, attn_selector_config, num_heads=None):
        if should_route(attn_selector_config):
            for name in (
                "has_sink",
                "use_mm_prefix",
                "use_per_head_quant_scales",
                "use_kv_connector",
            ):
                if getattr(attn_selector_config, name, False):
                    raise ValueError(f"OSCAR does not support selector option {name}")
            # NPUModelRunner queries a backend with sentinel head_size=0 before
            # model construction, also using it to dispatch graph updates.
            # Real layer implementations still validate D=256 in their constructor.
            if attn_selector_config.head_size not in (0, 256):
                raise ValueError("This OSCAR backend is restricted to Qwen3.5 FULL D256")
            return "oscar_ascend.backend.OscarAttentionBackend"
        return original_select(cls, selected_backend, attn_selector_config, num_heads)

    original_init = Attention.__init__
    signature = inspect.signature(original_init)

    @functools.wraps(original_init)
    def initialize(layer, *args, **kwargs):
        bound = signature.bind(layer, *args, **kwargs)
        prefix = bound.arguments.get("prefix", "")
        # MTP draft has separate weights/rotations. Keep draft cache native;
        # target verification's multiple candidate rows use fused OSCAR attention.
        if is_draft_layer(prefix) and bound.arguments.get("attn_backend") is None:
            from vllm_ascend.attention.attention_v1 import AscendAttentionBackend

            bound.arguments["attn_backend"] = AscendAttentionBackend
        return original_init(*bound.args, **bound.kwargs)

    original_process = Attention.process_weights_after_loading

    @functools.wraps(original_process)
    def process(layer, act_dtype):
        result = original_process(layer, act_dtype)
        from .backend import OscarAttentionImpl

        if isinstance(layer.impl, OscarAttentionImpl):
            layer.impl.initialize_layer(layer)
        return result

    NPUPlatform.get_attn_backend_cls = select
    Attention.__init__ = initialize
    Attention.process_weights_after_loading = process
    _registered = True
    LOG.warning(
        "OSCAR external attention enabled; native allocation and GDN retained; "
        "prefix sharing disabled; target verification quantized, MTP draft native. "
        "NPU correctness/performance acceptance is required."
    )
