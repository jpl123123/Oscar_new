"""Full attention backend over native Ascend cache views."""

import os

import torch
from vllm.config import get_current_vllm_config
from vllm.v1.attention.backend import AttentionImpl
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend

from .config import OscarConfig, choose_splits


class OscarAttentionBackend(AscendAttentionBackend):
    forward_includes_kv_cache_update = True

    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return OscarAttentionImpl

    @staticmethod
    def get_builder_cls():
        from .metadata import OscarMetadataBuilder

        return OscarMetadataBuilder

    @staticmethod
    def copy_blocks(*args, **kwargs):
        raise NotImplementedError(
            "OSCAR ring ownership does not support shared/copied request pages"
        )

    @staticmethod
    def swap_blocks(*args, **kwargs):
        raise NotImplementedError("OSCAR supports native recompute preemption, not KV swapping")


def validate_vllm_config(config):
    if config.cache_config.enable_prefix_caching:
        raise ValueError("OSCAR requires --no-enable-prefix-caching (BF16 ring owner isolation)")
    if config.cache_config.cache_dtype not in ("auto", "bfloat16"):
        raise ValueError("Use --kv-cache-dtype auto with OSCAR_ASCEND_ENABLED=1")
    if config.model_config.dtype != torch.bfloat16:
        raise ValueError("OSCAR target model must use BF16 activation/cache dtype")
    if config.kv_transfer_config is not None:
        raise ValueError("KV transfer is not supported by this OSCAR layout")
    pc = config.parallel_config
    if (
        getattr(pc, "decode_context_parallel_size", 1) != 1
        or getattr(pc, "prefill_context_parallel_size", 1) != 1
    ):
        raise ValueError("OSCAR currently requires PCP=DCP=1")
    if getattr(pc, "enable_dbo", False):
        raise ValueError("DBO requires separate metadata buffer sets and is not supported")
    if os.getenv("VLLM_USE_V2_MODEL_RUNNER", "0") == "1":
        raise ValueError("This OSCAR adapter targets NPUModelRunner v1")
    if getattr(config.quant_config, "enable_c8_quant", False):
        raise ValueError("Do not combine C8 KV quantization with OSCAR; W8A8 weights are supported")
    hf = config.model_config.hf_text_config
    if getattr(hf, "head_dim", None) != 256 or getattr(hf, "model_type", "") != "qwen3_5_text":
        raise ValueError("This implementation targets the Qwen3.5 D256 hybrid decoder")


class OscarAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads,
        head_size,
        scale,
        num_kv_heads,
        alibi_slopes,
        sliding_window,
        kv_cache_dtype,
        logits_soft_cap,
        attn_type,
        kv_sharing_target_layer_name,
        **kwargs,
    ):
        self.vllm_config = get_current_vllm_config()
        validate_vllm_config(self.vllm_config)
        if alibi_slopes is not None or sliding_window is not None or logits_soft_cap:
            raise ValueError("OSCAR does not implement ALiBi, SWA or logits soft cap")
        if kv_sharing_target_layer_name is not None or kwargs.get("sinks") is not None:
            raise ValueError(
                "OSCAR does not support cross-layer KV sharing or learnable attention sinks"
            )
        if attn_type != "decoder" or head_size != 256:
            raise ValueError("OSCAR requires FULL decoder attention with D256")
        if num_kv_heads != 1 or num_heads != 6:
            raise ValueError("This deployment is scoped to Qwen3.5-27B TP4 (Hq=6, Hkv=1)")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        speculative = self.vllm_config.speculative_config
        self.cfg = OscarConfig.from_env(speculative.num_speculative_tokens if speculative else 0)
        self.rk = self.rv = None
        self.history = self.window = None
        self.cache_identity = None
        self.mask = None

    def initialize_layer(self, layer):
        from vllm_ascend.attention.attention_mask import AttentionMaskBuilder

        from .rotations import get_rotation

        device = torch.device("npu", torch.npu.current_device())
        self.rk = get_rotation(self.cfg.k_rotation_path, layer.layer_name, device)
        self.rv = get_rotation(self.cfg.v_rotation_path, layer.layer_name, device)
        self.mask = AttentionMaskBuilder(device).get_attention_mask(
            True, self.vllm_config.model_config
        )

    @staticmethod
    def update_graph_params(*args, **kwargs):
        # Triton captures fixed pointer inputs. The builder updates device buffers
        # before replay; no native FIA task-group arguments exist for this backend.
        return None

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        if output is None:
            raise ValueError("Ascend must provide an attention output buffer")
        if output_scale is not None or output_block_scale is not None:
            raise ValueError("Fused output quantization is unsupported")
        if attn_metadata is None:
            return output.zero_()
        if self.rk is None or self.rv is None:
            raise RuntimeError("OSCAR rotations were not initialized after model weight loading")
        if key is None or value is None:
            raise ValueError("OSCAR requires the current K/V to maintain exact BF16 windows")
        if query.device.type != "npu":
            raise RuntimeError("OSCAR serving kernels must run on an Ascend NPU")
        from . import kernels

        meta = attn_metadata
        if meta.is_dummy:
            from .startup import trace_kernels

            kernels = trace_kernels(kernels, lambda: torch.npu.current_stream().synchronize())
        n = query.shape[0]
        query = query.view(n, self.num_heads, self.head_size)
        key = key.view(key.shape[0], self.num_kv_heads, self.head_size)
        value = value.view(value.shape[0], self.num_kv_heads, self.head_size)
        if key.shape[0] != n or value.shape[0] != n:
            raise ValueError("Q/K/V must have matching current-token dimensions")
        identity = (kv_cache[0].data_ptr(), kv_cache[1].data_ptr(), meta.layout)
        if self.cache_identity != identity:
            self.history, self.window = meta.layout.tensor_views(kv_cache)
            self.cache_identity = identity
        if not output.is_contiguous():
            raise ValueError("Attention output must be contiguous")
        if n == 0:
            return output

        krot = kernels.rotate(key, self.rk)
        vrot = kernels.rotate(value, self.rv)
        kernels.store_int2(krot, vrot, self.history, meta.slot_mapping, meta.counts, meta.layout)

        if meta.initial_prefill:
            import torch_npu

            nt = meta.num_actual_tokens
            result, _ = torch_npu.npu_fused_infer_attention_score(
                query=query[:nt],
                key=key[:nt].contiguous(),
                value=value[:nt].contiguous(),
                input_layout="TND",
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                scale=self.scale,
                atten_mask=self.mask,
                sparse_mode=3,
                actual_seq_lengths=meta.actual_seq_lengths_q,
                actual_seq_lengths_kv=meta.actual_seq_lengths_q,
            )
            output.view(n, self.num_heads, self.head_size)[:nt].copy_(result)
            output.view(n, self.num_heads, self.head_size)[nt:].zero_()
        else:
            qrot = kernels.rotate(query, self.rk)
            splits = choose_splits(meta.num_reqs, self.num_kv_heads, meta.max_query_len)
            hp, hl = kernels.attention_partials(
                qrot,
                key,
                value,
                self.history,
                self.window,
                meta,
                meta.layout,
                self.scale,
                splits,
                raw=False,
            )
            hrot, hlogsum = kernels.merge_splits(hp, hl, meta.counts)
            hout = kernels.rotate(hrot, self.rv.T)
            rp, rl = kernels.attention_partials(
                query,
                key,
                value,
                self.history,
                self.window,
                meta,
                meta.layout,
                self.scale,
                1,
                raw=True,
            )
            kernels.merge_paths(hout, hlogsum, rp, rl, meta.counts, output)

        # Must follow BOTH attention branches: long current chunks wrap the ring.
        kernels.store_windows(key, value, self.window, meta, meta.layout)
        return output
