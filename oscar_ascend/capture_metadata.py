"""Eliminate the redundant D2H copy in native GDN capture metadata only."""

import functools
import inspect
import sys


def build_gdn_capture_metadata(builder, common_attn_metadata):
    import torch

    m = common_attn_metadata
    if (
        m.num_reqs > builder.decode_cudagraph_max_bs
        or m.num_actual_tokens > builder.decode_cudagraph_max_bs
    ):
        raise ValueError("GDN capture batch exceeds decode_cudagraph_max_bs")
    host_starts = m.query_start_loc_cpu
    if host_starts.device.type != "cpu" or host_starts.shape != m.query_start_loc.shape:
        raise ValueError("GDN capture requires matching scheduler CPU query boundaries")
    # The native dummy runner constructs these boundaries on CPU and copies
    # them to NPU before calling this method. Keep accepted counts on device;
    # compute the host-only draft classification from the existing CPU mirror.
    # Do not do a device diff followed by .cpu(), or inspect device values here.
    accepted = torch.diff(m.query_start_loc)
    draft_counts_cpu = torch.diff(host_starts) - 1
    return builder.build(0, m, accepted, draft_counts_cpu)


def install_gdn_capture_hook():
    module = sys.modules.get("vllm_ascend.ops.gdn_attn_builder")
    if module is None or getattr(getattr(module, "__spec__", None), "_initializing", False):
        return False
    builder = getattr(module, "AscendGDNAttentionMetadataBuilder", None)
    if builder is None:
        return False
    if getattr(builder, "_oscar_capture_metadata_hook", False):
        return True
    # The audited reference inherits this D2H method from upstream vLLM.
    # Do not overwrite a separate platform's own capture implementation.
    if "build_for_cudagraph_capture" in builder.__dict__:
        return False
    original = builder.build_for_cudagraph_capture
    if "common_attn_metadata" not in inspect.signature(original).parameters:
        raise RuntimeError("Unsupported GDN capture metadata signature")

    @functools.wraps(original)
    def capture(self, common_attn_metadata):
        return build_gdn_capture_metadata(self, common_attn_metadata)

    builder.build_for_cudagraph_capture = capture
    builder._oscar_capture_metadata_hook = True
    return True
