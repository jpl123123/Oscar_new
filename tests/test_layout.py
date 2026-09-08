from dataclasses import replace

import pytest
import torch

from oscar_ascend.config import OscarConfig, choose_splits, percentile_selection
from oscar_ascend.layout import CacheLayout
from tests.oracle import quantize


def test_exact_target_geometry():
    layout = CacheLayout(768)
    assert layout.config.vector_bytes == 68
    assert layout.config.slot_bytes == 160
    assert layout.stripe_page_bytes == 393216
    assert layout.history_page_bytes == 122880
    assert layout.window_page_bytes == 331776
    assert layout.blocks_per_page == 6
    assert 15360 + layout.stripe_page_bytes * 2 == 801792


@pytest.mark.parametrize("bad", [0, 128, 256, 384, 512, 640, 769])
def test_reject_small_or_nondivisible_page(bad):
    with pytest.raises(ValueError):
        CacheLayout(bad)


def test_virtual_blocks_resolve_to_physical_page_and_do_not_overlap_gdn():
    layout = CacheLayout(768)
    pages = 9
    conv_bytes = 15360 * pages
    stripe_bytes = layout.stripe_page_bytes * pages
    raw = torch.full((conv_bytes + 2 * stripe_bytes,), 0x5A, dtype=torch.uint8)
    k = raw[conv_bytes : conv_bytes + stripe_bytes].view(torch.bfloat16).view(-1, 128, 1, 256)
    v = raw[conv_bytes + stripe_bytes :].view(torch.bfloat16).view(-1, 128, 1, 256)
    hist, ring = layout.tensor_views((k, v))
    assert hist.data_ptr() == k.data_ptr()
    assert ring.data_ptr() == v.data_ptr()
    table = [p * 6 + j for p in (5, 2, 7) for j in range(6)]
    for t in range(2304):
        slot = layout.logical_slot(table, t)
        begin = layout.slot_offset(slot)
        hist[begin : begin + 136] = 7
        offset = layout.window_offset(table[0], t)
        raw[conv_bytes + stripe_bytes + offset : conv_bytes + stripe_bytes + offset + 1024] = 3
    # Other scheduler-owned page numbers, including GDN ssm pages, are unchanged.
    for p in (0, 1, 3, 4, 6, 8):
        assert (
            hist[p * layout.stripe_page_bytes : (p + 1) * layout.stripe_page_bytes] == 0x5A
        ).all()
    assert (raw[:conv_bytes] == 0x5A).all()
    # Only request first page owns the ring; later FULL and all GDN pages are untouched.
    vbytes = v.view(torch.uint8).view(-1)
    for p in range(pages):
        if p != 5:
            assert (
                vbytes[p * layout.stripe_page_bytes : (p + 1) * layout.stripe_page_bytes] == 0x5A
            ).all()


def test_ring_preserves_rejection_rollback_and_sink():
    cfg = OscarConfig()
    layout = CacheLayout(768, config=cfg)
    memory = {}
    end = 0
    for chunk, rejected in ((1024, 0), (4, 3), (4, 2), (4, 0), (300, 0), (4, 3)):
        prefix = end
        # Every prefix recent token needed by this step survived the preceding rejection.
        for t in range(max(cfg.sink_tokens, prefix - cfg.recent_tokens), prefix):
            assert memory[layout.window_offset(12, t)] == t
        for t in range(min(prefix, cfg.sink_tokens)):
            assert memory[layout.window_offset(12, t)] == t
        for t in range(prefix, prefix + chunk):
            memory[layout.window_offset(12, t)] = t
        end = prefix + chunk - rejected


def test_reorder_and_reuse_do_not_use_batch_indices():
    layout = CacheLayout(768)
    owners = {"a": 12, "b": 30}
    before = {req: layout.window_offset(owner, 810) for req, owner in owners.items()}
    after = {req: layout.window_offset(owners[req], 810) for req in reversed(owners)}
    assert before == after and before["a"] != before["b"]
    with pytest.raises(ValueError):
        layout.window_offset(13, 0)


@pytest.mark.parametrize("ratio", [0.875, 0.92, 0.96, 1.0])
def test_top_selection_matches_linear_quantile_with_ties(ratio):
    torch.manual_seed(9)
    x = torch.cat((torch.randn(3, 256), torch.zeros(1, 256), torch.ones(1, 256))).abs()
    lo, hi, weight = percentile_selection(ratio)
    remaining = x.clone()
    selected = []
    for _ in range(lo + 1):
        maxima, index = remaining.max(-1)
        selected.append(maxima)
        remaining.scatter_(-1, index[:, None], -float("inf"))
    threshold = selected[lo] + (selected[hi] - selected[lo]) * weight
    torch.testing.assert_close(threshold, torch.quantile(x, ratio, -1))


@pytest.mark.parametrize("value", [0, 1e-20, 0.5, -3])
def test_constant_quantizer_is_finite(value):
    _, scale, _, decoded = quantize(torch.full((2, 1, 256), value))
    assert torch.isfinite(decoded).all()
    assert (scale > 0).all()
    torch.testing.assert_close(
        decoded.float(), torch.full_like(decoded.float(), value), atol=1e-7, rtol=0
    )


def test_split_policy_bounds_current_scratch():
    assert [choose_splits(b, 1, 4) for b in (1, 16, 128)] == [32, 16, 8]
    assert choose_splits(1, 1, 16384) == 1


def test_bad_config_rejected():
    for kwargs in (
        {"group_size": 128},
        {"head_dim": 128},
        {"k_clip_ratio": 0.5},
        {"k_clip_factor": float("nan")},
        {"sink_tokens": -1},
    ):
        with pytest.raises(ValueError):
            replace(OscarConfig(), **kwargs)
