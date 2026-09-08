"""Dense independent Torch oracle for numerical comparisons on CPU or NPU."""

import torch


def quantize(x, ratio=0.0, mode="percentile", factor=1.0):
    x = x.float()
    if mode == "factor":
        threshold = x.abs().amax(-1, keepdim=True) * factor
        x = x.clamp(-threshold, threshold)
    elif ratio:
        # quantile is ONLY in this test oracle, never in serving.
        threshold = torch.quantile(x.abs(), ratio, dim=-1, keepdim=True)
        x = x.clamp(-threshold, threshold)
    zero = x.amin(-1, keepdim=True).half()
    scale = ((x.amax(-1, keepdim=True) - x.amin(-1, keepdim=True)) / 3).clamp_min(2**-14).half()
    code = ((x - zero.float()) / scale.float() + 0.5).to(torch.int32).clamp(0, 3)
    codes = code.reshape(*code.shape[:-1], -1, 4)
    packed = (codes[..., 0] | codes[..., 1] << 2 | codes[..., 2] << 4 | codes[..., 3] << 6).byte()
    decoded = (code.float() * scale.float() + zero.float()).bfloat16()
    return packed, scale, zero, decoded


def packed_slots(k, v, cfg):
    shape = (*k.shape[:-1], cfg.slot_bytes)
    out = torch.zeros(shape, dtype=torch.uint8, device=k.device)
    for value, offset, ratio, factor in (
        (k, 0, cfg.k_clip_ratio, cfg.k_clip_factor),
        (v, cfg.vector_bytes, cfg.v_clip_ratio, cfg.v_clip_factor),
    ):
        codes, scale, zero, _ = quantize(value, ratio, cfg.clip_mode, factor)
        out[..., offset : offset + cfg.data_bytes] = codes
        out[..., offset + cfg.data_bytes : offset + cfg.data_bytes + 2] = scale.contiguous().view(
            torch.uint8
        )
        out[..., offset + cfg.data_bytes + 2 : offset + cfg.data_bytes + 4] = (
            zero.contiguous().view(torch.uint8)
        )
    return out


def dense_attention(q, k, v, prefix):
    """All scores/softmax/AV evaluated independently in FP32."""
    hq, hk = q.shape[1], k.shape[1]
    keys = k.repeat_interleave(hq // hk, 1).float()
    values = v.repeat_interleave(hq // hk, 1).float()
    scores = torch.einsum("qhd,khd->hqk", q.float(), keys) / q.shape[-1] ** 0.5
    qi = torch.arange(q.shape[0], device=q.device) + prefix
    ki = torch.arange(k.shape[0], device=k.device)
    scores.masked_fill_(ki[None, None, :] > qi[None, :, None], -float("inf"))
    prob = scores.softmax(-1)
    return torch.einsum("hqk,khd->qhd", prob, values)


def hybrid_reference(q, k, v, prefix, rk, rv, cfg):
    """Reconstruct quantized history for tests, preserving raw sink/current/recent."""
    end = max(min(cfg.sink_tokens, prefix), prefix - cfg.recent_tokens)
    kr = (k.float() @ rk.float()).bfloat16()
    vr = (v.float() @ rv.float()).bfloat16()
    _, _, _, kd = quantize(kr, cfg.k_clip_ratio, cfg.clip_mode, cfg.k_clip_factor)
    _, _, _, vd = quantize(vr, cfg.v_clip_ratio, cfg.clip_mode, cfg.v_clip_factor)
    keys, values = k.float().clone(), v.float().clone()
    keys[cfg.sink_tokens : end] = kd[cfg.sink_tokens : end].float() @ rk.float().T
    values[cfg.sink_tokens : end] = vd[cfg.sink_tokens : end].float() @ rv.float().T
    return dense_attention(q, keys, values, prefix)
