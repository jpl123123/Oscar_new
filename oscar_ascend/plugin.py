"""Idempotent general plugin. All hooks are in-memory and construction-only."""

import functools
import inspect
import logging
import os
import sys

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

    from .compat import require_parameters, validate_versions
    from .startup import install_runner_hooks

    validate_versions({name: version(name) for name in ("vllm", "vllm-ascend")})
    if calibrating:
        # Native initialization must run first. The named begin RPC attaches
        # the observation hook after LLM has finished loading the model.
        return
    # The platform module is lightweight; leave native global/worker patches
    # to vLLM's normal initialization. Never import attention/ops here.
    from vllm_ascend.platform import NPUPlatform

    require_parameters(
        "NPUPlatform.get_attn_backend_cls",
        NPUPlatform.get_attn_backend_cls,
        ("selected_backend", "attn_selector_config", "num_heads"),
    )
    original_select = NPUPlatform.get_attn_backend_cls.__func__

    @classmethod
    @functools.wraps(original_select)
    def select(cls, selected_backend, attn_selector_config, num_heads=None):
        # NPUModelRunner probes the backend after its native module imports
        # complete and before model construction. Attach hooks at that point.
        _install_loaded_attention_hooks(cls)
        install_runner_hooks()
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

    NPUPlatform.get_attn_backend_cls = select
    _registered = True
    _install_loaded_attention_hooks(NPUPlatform)
    install_runner_hooks()
    LOG.warning(
        "OSCAR external attention enabled; native allocation and GDN retained; "
        "prefix sharing disabled; target verification quantized, MTP draft native. "
        "NPU correctness/performance acceptance is required."
    )


def _install_loaded_attention_hooks(platform):
    """Wrap only a completed, naturally imported Attention class."""
    from .compat import validate_hook_interfaces

    module = sys.modules.get("vllm.model_executor.layers.attention.attention")
    if module is None or getattr(getattr(module, "__spec__", None), "_initializing", False):
        return False
    Attention = getattr(module, "Attention", None)
    if Attention is None:
        return False
    if getattr(Attention, "_oscar_ascend_hooks_installed", False):
        return True
    validate_hook_interfaces(platform, Attention)
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

    Attention.__init__ = initialize
    Attention.process_weights_after_loading = process
    Attention._oscar_ascend_hooks_installed = True
    return True
