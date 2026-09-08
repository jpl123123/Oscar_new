"""Byte geometry over the native three-stripe allocation; no tensor allocation."""

from dataclasses import dataclass, field

from .config import OscarConfig


@dataclass(frozen=True)
class CacheLayout:
    physical_block_tokens: int
    kv_heads: int = 1
    kernel_block_tokens: int = 128
    config: OscarConfig = field(default_factory=OscarConfig)

    def __post_init__(self):
        if self.kv_heads <= 0 or self.physical_block_tokens <= 0:
            raise ValueError("Invalid physical cache geometry")
        if self.physical_block_tokens % self.kernel_block_tokens:
            raise ValueError("Physical block must be divisible by the kernel block size")
        if self.history_page_bytes > self.stripe_page_bytes:
            raise ValueError("INT2 slots do not fit native K stripe")
        if self.window_page_bytes > self.stripe_page_bytes:
            raise ValueError(
                f"BF16 window requires {self.window_page_bytes} bytes per owner; "
                f"native V stripe has {self.stripe_page_bytes}. "
                f"Need physical block size >= {2 * self.config.window_tokens}; "
                "do not shrink or reinterpret a Mamba page to bypass this check."
            )

    @property
    def stripe_page_bytes(self):
        return self.physical_block_tokens * self.kv_heads * self.config.head_dim * 2

    @property
    def history_page_bytes(self):
        return self.physical_block_tokens * self.kv_heads * self.config.slot_bytes

    @property
    def window_page_bytes(self):
        return self.config.window_tokens * 2 * self.kv_heads * self.config.head_dim * 2

    @property
    def blocks_per_page(self):
        return self.physical_block_tokens // self.kernel_block_tokens

    def slot_offset(self, slot, head=0):
        if slot < 0 or not 0 <= head < self.kv_heads:
            raise ValueError("Invalid slot/head")
        page, position = divmod(slot, self.physical_block_tokens)
        return (
            page * self.stripe_page_bytes
            + (position * self.kv_heads + head) * self.config.slot_bytes
        )

    def logical_slot(self, block_table, position):
        block_index, token = divmod(position, self.kernel_block_tokens)
        return block_table[block_index] * self.kernel_block_tokens + token

    def window_offset(self, first_kernel_block, position, kv=0, head=0):
        if position < 0 or kv not in (0, 1) or not 0 <= head < self.kv_heads:
            raise ValueError("Invalid window position/head")
        if first_kernel_block < 0 or first_kernel_block % self.blocks_per_page:
            raise ValueError("The first kernel block must start a physical page")
        owner = first_kernel_block // self.blocks_per_page
        index = (
            position
            if position < self.config.sink_tokens
            else (self.config.sink_tokens + position % self.config.ring_tokens)
        )
        return (
            owner * self.stripe_page_bytes
            + ((index * 2 + kv) * self.kv_heads + head) * self.config.head_dim * 2
        )

    def tensor_views(self, kv_cache):
        """Return byte history and BF16 windows over existing B/C views."""
        import torch

        if not isinstance(kv_cache, (tuple, list)) or len(kv_cache) != 2:
            raise ValueError("Expected native Ascend (K, V) cache views")
        k, v = kv_cache
        for tensor in (k, v):
            if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
                raise ValueError("OSCAR requires contiguous native BF16 K/V views")
            if tensor.ndim != 4 or tuple(tensor.shape[1:]) != (
                self.kernel_block_tokens,
                self.kv_heads,
                self.config.head_dim,
            ):
                raise ValueError(f"Unexpected native cache shape {tuple(tensor.shape)}")
            if tensor.numel() * tensor.element_size() % self.stripe_page_bytes:
                raise ValueError("Native stripe is not an integer number of physical pages")
        if k.shape != v.shape:
            raise ValueError("K/V stripes must have identical geometry")
        return k.view(torch.uint8).view(-1), v.view(-1)
