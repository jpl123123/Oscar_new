"""OSCAR PR environment names, with explicit Ascend-specific constraints."""

import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class OscarConfig:
    head_dim: int = 256
    group_size: int = 256
    sink_tokens: int = 64
    recent_tokens: int = 256
    speculative_tokens: int = 3
    k_clip_ratio: float = 0.96
    v_clip_ratio: float = 0.92
    clip_mode: str = "percentile"
    k_clip_factor: float = 1.0
    v_clip_factor: float = 1.0
    k_rotation_path: str = ""
    v_rotation_path: str = ""
    block_n: int = 32
    queries_per_tile: int = 4

    def __post_init__(self):
        if self.head_dim != 256 or self.group_size < self.head_dim:
            raise ValueError("This adapter requires D=256 and one quantization group per vector")
        if min(self.sink_tokens, self.recent_tokens, self.speculative_tokens) < 0:
            raise ValueError("Window and speculative token counts must be nonnegative")
        if self.speculative_tokens > 15:
            raise ValueError("At most 15 speculative tokens are supported")
        if self.clip_mode not in ("percentile", "factor"):
            raise ValueError("OSCAR_ASCEND_CLIP_MODE must be percentile or factor")
        for ratio in (self.k_clip_ratio, self.v_clip_ratio):
            if not math.isfinite(ratio) or not (ratio == 0 or 0.875 <= ratio <= 1):
                raise ValueError("Clip percentile must be 0 (disabled) or in [0.875, 1]")
        for factor in (self.k_clip_factor, self.v_clip_factor):
            if not math.isfinite(factor) or not 0 < factor <= 1:
                raise ValueError("Fixed clipping factors must be in (0, 1]")
        if self.block_n not in (16, 32):
            raise ValueError("OSCAR_ASCEND_BLOCK_N must be 16 or 32")
        if self.queries_per_tile not in (1, 2, 4):
            raise ValueError("OSCAR_ASCEND_QUERY_TILE must be 1, 2 or 4")

    @property
    def data_bytes(self):
        return self.head_dim // 4

    @property
    def vector_bytes(self):
        return self.data_bytes + 4

    @property
    def slot_bytes(self):
        return ((2 * self.vector_bytes + 31) // 32) * 32

    @property
    def ring_tokens(self):
        return self.recent_tokens + self.speculative_tokens + 1

    @property
    def window_tokens(self):
        return self.sink_tokens + self.ring_tokens

    @classmethod
    def from_env(cls, speculative_tokens=3):
        return cls(
            group_size=int(os.getenv("VLLM_OSCAR_GROUP_SIZE", "256")),
            sink_tokens=int(os.getenv("VLLM_OSCAR_SINK_TOKENS", "64")),
            recent_tokens=int(os.getenv("VLLM_OSCAR_RECENT_TOKENS", "256")),
            speculative_tokens=speculative_tokens,
            k_clip_ratio=float(os.getenv("VLLM_OSCAR_K_CLIP_RATIO", "0.96")),
            v_clip_ratio=float(os.getenv("VLLM_OSCAR_V_CLIP_RATIO", "0.92")),
            clip_mode=os.getenv("OSCAR_ASCEND_CLIP_MODE", "percentile"),
            k_clip_factor=float(os.getenv("OSCAR_ASCEND_K_CLIP_FACTOR", "1")),
            v_clip_factor=float(os.getenv("OSCAR_ASCEND_V_CLIP_FACTOR", "1")),
            k_rotation_path=os.getenv("VLLM_OSCAR_K_ROTATION_PATH", ""),
            v_rotation_path=os.getenv("VLLM_OSCAR_V_ROTATION_PATH", ""),
            block_n=int(os.getenv("OSCAR_ASCEND_BLOCK_N", "32")),
            queries_per_tile=int(os.getenv("OSCAR_ASCEND_QUERY_TILE", "4")),
        )


def choose_splits(num_requests, num_kv_heads, max_query_len):
    """Bound scratch by current tokens. Never specialize on device context length."""
    if max_query_len > 16:
        return 1
    parallel_heads = num_requests * num_kv_heads
    return 32 if parallel_heads <= 8 else 16 if parallel_heads <= 32 else 8


def percentile_selection(ratio, dim=256):
    """Return descending ranks and interpolation weight for torch.quantile semantics."""
    index = (dim - 1) * ratio
    lower = math.floor(index)
    upper = math.ceil(index)
    return dim - 1 - lower, dim - 1 - upper, index - lower


def task_capacity(tokens, requests, query_tile):
    return min(tokens, (tokens + query_tile - 1) // query_tile + requests - 1)


def attention_programs(tokens, requests, query_tile):
    """Cover the compact task list without a loop across tasks inside the kernel."""
    return max(1, task_capacity(tokens, requests, query_tile))
