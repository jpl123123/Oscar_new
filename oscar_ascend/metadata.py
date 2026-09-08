"""Graph-stable device metadata, without NPU-to-host sequence length reads."""

import math
from dataclasses import dataclass

import torch
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState

from .config import OscarConfig
from .layout import CacheLayout
from .startup import dummy_run_scope, in_dummy_run


@dataclass
class OscarMetadata:
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor
    slot_mapping: torch.Tensor
    counts: torch.Tensor
    num_reqs: int
    num_actual_tokens: int
    max_query_len: int
    max_num_reqs: int
    layout: CacheLayout
    attn_state: object
    initial_prefill: bool = False
    actual_seq_lengths_q: list | None = None
    causal: bool = True
    is_dummy: bool = False


class OscarMetadataBuilder(AttentionMetadataBuilder[OscarMetadata]):
    supports_update_block_table = False

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        from .backend import OscarAttentionImpl

        for name in layer_names:
            impl = vllm_config.compilation_config.static_forward_context[name].impl
            if not isinstance(impl, OscarAttentionImpl):
                raise TypeError(
                    f"OSCAR implementation was replaced (e.g. by C8 KV surgery): {name}"
                )
        speculative = vllm_config.speculative_config
        cfg = OscarConfig.from_env(speculative.num_speculative_tokens if speculative else 0)
        self.layout = CacheLayout(kv_cache_spec.block_size, kv_cache_spec.num_kv_heads, config=cfg)
        self.reorder_batch_threshold = cfg.speculative_tokens + 1
        scheduler = vllm_config.scheduler_config
        self.max_reqs = scheduler.max_num_seqs + 1  # Native graph padding may add a dummy row.
        # Include padding slots and the MTP tail allowed by the native block table.
        self.max_tokens = (
            scheduler.max_num_batched_tokens + (cfg.speculative_tokens + 4) * self.max_reqs
        )
        max_blocks = math.ceil(vllm_config.model_config.max_model_len / 128)
        max_blocks += self.layout.blocks_per_page + 1
        self.qstarts = torch.zeros(self.max_reqs + 1, dtype=torch.int32, device=device)
        self.seqs = torch.zeros(self.max_reqs, dtype=torch.int32, device=device)
        self.table = torch.zeros((self.max_reqs, max_blocks), dtype=torch.int32, device=device)
        self.slots = torch.full((self.max_tokens,), -1, dtype=torch.int64, device=device)
        self.counts = torch.zeros(2, dtype=torch.int32, device=device)

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        return AttentionCGSupport.ALWAYS

    def reorder_batch(self, input_batch, scheduler_output):
        return False

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        cm = common_attn_metadata
        if common_prefix_len:
            raise ValueError("Prefix sharing is unsupported by the request-owned BF16 ring")
        if not cm.causal:
            raise ValueError("OSCAR supports causal decoder attention only")
        nr, nt = cm.num_reqs, cm.num_actual_tokens
        if nr > self.max_reqs or nt > self.max_tokens:
            raise ValueError("OSCAR metadata capacity exceeded")
        columns = cm.block_table_tensor.shape[1]
        if columns > self.table.shape[1]:
            raise ValueError("Native block table exceeds the configured OSCAR buffer")
        for name in ("query_start_loc", "seq_lens", "block_table_tensor", "slot_mapping"):
            if getattr(cm, name).device.type != "npu":
                raise ValueError(f"Expected NPU metadata for {name}")
        dummy = in_dummy_run()
        if dummy:
            # Native warmup invalidates its slot mapping AFTER building metadata.
            # A private snapshot would miss that invalidation and write page zero
            # or stale slots. Dummy runs own no KV pages. Device-side counts mask
            # all cache accesses while still recording every kernel for replay.
            self.qstarts.zero_()
            self.seqs.zero_()
            self.table.zero_()
            self.slots.fill_(-1)
            self.counts.zero_()
        else:
            self.qstarts[: nr + 1].copy_(cm.query_start_loc[: nr + 1], non_blocking=True)
            self.seqs[:nr].copy_(cm.seq_lens[:nr], non_blocking=True)
            self.table[:nr, :columns].copy_(cm.block_table_tensor[:nr], non_blocking=True)
            self.slots[:nt].copy_(cm.slot_mapping[:nt], non_blocking=True)
            self.counts[0].fill_(nr)
            self.counts[1].fill_(nt)

        # Only scheduler CPU metadata is inspected for the optional initial FIA path.
        # PrefillNoCache is a host enum from the native runner; never infer this
        # from a .tolist()/.item() read of device seq_lens.
        initial = not dummy and cm.attn_state == AscendAttentionState.PrefillNoCache
        query_ends = None
        if initial:
            if cm.query_start_loc_cpu.device.type != "cpu":
                raise ValueError("query_start_loc_cpu must really be on CPU")
            query_ends = cm.query_start_loc_cpu[1 : nr + 1].tolist()
        return OscarMetadata(
            self.qstarts,
            self.seqs,
            self.table,
            self.slots,
            self.counts,
            nr,
            nt,
            cm.max_query_len,
            self.max_reqs,
            self.layout,
            cm.attn_state,
            initial,
            query_ends,
            is_dummy=dummy,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata, attn_state=AscendAttentionState.DecodeOnly
    ):
        with dummy_run_scope():
            result = self.build(0, common_attn_metadata)
        result.initial_prefill = False
        result.attn_state = attn_state
        return result

    # Compatibility with the separate Ascend-facing helper name. The runner in
    # the pinned tree calls build_for_cudagraph_capture, which MUST be overridden.
    build_for_graph_capture = build_for_cudagraph_capture
