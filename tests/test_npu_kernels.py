"""Real device tests, including paging, rollback and graph replay.

Run: python -m pytest tests/test_npu_kernels.py --require-npu -v
The FP32 oracle runs on CPU and is intentionally outside the serving path.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")
import torch_npu  # noqa: E402, F401

if not torch.npu.is_available():
    pytest.skip("Ascend NPU unavailable", allow_module_level=True)

from oscar_ascend import kernels  # noqa: E402
from oscar_ascend.config import OscarConfig  # noqa: E402
from oscar_ascend.layout import CacheLayout  # noqa: E402
from tests.oracle import hybrid_reference, packed_slots  # noqa: E402

pytestmark = pytest.mark.npu


def make_state(lengths, prefix_lengths, cfg=None, pages=20, max_reqs=8):
    layout = CacheLayout(768, config=cfg or OscarConfig())
    # Exact native three-stripe views, with sentinels in unused/GDN-owned pages.
    conv = pages * 15360
    stripe = pages * layout.stripe_page_bytes
    raw = torch.full((conv + 2 * stripe,), 0x5A, device="npu", dtype=torch.uint8)
    k = raw[conv : conv + stripe].view(torch.bfloat16).view(-1, 128, 1, 256)
    v = raw[conv + stripe :].view(torch.bfloat16).view(-1, 128, 1, 256)
    history, window = layout.tensor_views((k, v))
    # Disjoint request ownership, non-monotonic physical pages, page zero untouched.
    available = list(range(1, pages))
    available = available[::2] + available[1::2]
    tables = torch.zeros((max_reqs, 6 * pages), dtype=torch.int32)
    qstarts = [0]
    slots = []
    for i, (length, prefix) in enumerate(zip(lengths, prefix_lengths)):
        needed = (length + 767) // 768
        owned, available = available[:needed], available[needed:]
        blocks = [p * 6 + j for p in owned for j in range(6)]
        tables[i, : len(blocks)] = torch.tensor(blocks)
        slots.extend(blocks[t // 128] * 128 + t % 128 for t in range(prefix, length))
        qstarts.append(qstarts[-1] + length - prefix)
    qbuffer = torch.zeros(max_reqs + 1, dtype=torch.int32)
    qbuffer[: len(qstarts)] = torch.tensor(qstarts)
    sbuffer = torch.zeros(max_reqs, dtype=torch.int32)
    sbuffer[: len(lengths)] = torch.tensor(lengths)
    meta = SimpleNamespace(
        query_start_loc=qbuffer.to("npu"),
        seq_lens=sbuffer.to("npu"),
        block_tables=tables.to("npu"),
        counts=torch.tensor([len(lengths), len(slots)], device="npu", dtype=torch.int32),
        slot_mapping=torch.tensor(slots, device="npu", dtype=torch.int64),
        num_reqs=len(lengths),
        max_num_reqs=max_reqs,
    )
    return layout, raw, history, window, meta, tables


@pytest.mark.parametrize("ratio", [0, 0.92, 0.96, 1.0])
def test_store_bytes_and_inverse_with_noncontiguous_rotation(ratio):
    cfg = replace(OscarConfig(), k_clip_ratio=ratio, v_clip_ratio=ratio)
    layout, raw, history, _, meta, _ = make_state([800], [760], cfg)
    torch.manual_seed(3)
    k = torch.randn(40, 1, 256).bfloat16()
    v = torch.randn_like(k)
    k[0].zero_()
    k[1].fill_(0.5)
    k[2].fill_(1e-12)
    kernels.store_int2(k.to("npu"), v.to("npu"), history, meta.slot_mapping, meta.counts, layout)
    expected = packed_slots(k, v, cfg)
    history_cpu = history.cpu()
    for index, slot in enumerate(meta.slot_mapping.cpu().tolist()):
        offset = layout.slot_offset(slot)
        torch.testing.assert_close(
            history_cpu[offset : offset + 136], expected[index, 0, :136], atol=0, rtol=0
        )
        assert (history_cpu[offset + 136 : offset + 160] == 0x5A).all()
    assert (raw[: 20 * 15360].cpu() == 0x5A).all()
    assert (history_cpu[: layout.stripe_page_bytes] == 0x5A).all()
    identity = torch.eye(256, device="npu", dtype=torch.bfloat16)
    kd, vd = kernels.dequant_inverse_rotate(
        history, meta.slot_mapping, identity.T, identity.T, layout
    )
    assert torch.isfinite(kd).all() and torch.isfinite(vd).all()


def test_rotation_strides_match_torch():
    torch.manual_seed(7)
    x = torch.randn(23, 12, 256, device="npu", dtype=torch.bfloat16)[:, ::2]
    r = torch.linalg.qr(torch.randn(256, 256))[0].bfloat16().to("npu").T
    out = kernels.rotate(x, r)
    torch.testing.assert_close(
        out.float().cpu(),
        (x.float() @ r.float()).bfloat16().float().cpu(),
        atol=0.03125,
        rtol=0.015,
    )


def prepare_prefix(k, v, rk, rv, history, window, meta, tables, layout, prefixes):
    """Populate prior steps with the actual NPU store kernels."""
    for req, prefix in enumerate(prefixes):
        if prefix == 0:
            continue
        old = SimpleNamespace(
            query_start_loc=torch.tensor([0, prefix], device="npu", dtype=torch.int32),
            seq_lens=torch.tensor([prefix], device="npu", dtype=torch.int32),
            block_tables=meta.block_tables[req : req + 1],
            counts=torch.tensor([1, prefix], device="npu", dtype=torch.int32),
            slot_mapping=torch.tensor(
                [int(tables[req, t // 128]) * 128 + t % 128 for t in range(prefix)],
                device="npu",
                dtype=torch.int64,
            ),
            num_reqs=1,
            max_num_reqs=1,
        )
        pk, pv = k[req][:prefix].to("npu"), v[req][:prefix].to("npu")
        kernels.store_int2(
            kernels.rotate(pk, rk),
            kernels.rotate(pv, rv),
            history,
            old.slot_mapping,
            old.counts,
            layout,
        )
        kernels.store_windows(pk, pv, window, old, layout)


def run_attention(q, k, v, rk, rv, history, window, meta, layout, splits):
    kr, vr = kernels.rotate(k, rk), kernels.rotate(v, rv)
    kernels.store_int2(kr, vr, history, meta.slot_mapping, meta.counts, layout)
    qp = kernels.rotate(q, rk)
    hp, hl = kernels.attention_partials(
        qp, k, v, history, window, meta, layout, 1 / 16, splits, False
    )
    hr, hls = kernels.merge_splits(hp, hl, meta.counts)
    ho = kernels.rotate(hr, rv.T)
    rp, rl = kernels.attention_partials(q, k, v, history, window, meta, layout, 1 / 16, 1, True)
    out = torch.empty_like(q)
    kernels.merge_paths(ho, hls, rp, rl, meta.counts, out)
    kernels.store_windows(k, v, window, meta, layout)
    return out


@pytest.mark.parametrize(
    "lengths,prefixes,splits",
    [
        ([4], [0], 32),  # empty history / all-masked splits
        ([69, 263], [65, 260], 16),  # ragged MTP, sink/recent overlap, no history
        ([1804, 2601], [1800, 2600], 8),  # nonmonotonic pages, GQA+MTP
        ([1700], [800], 1),  # continuation chunk wraps ring multiple times
    ],
)
def test_fused_attention_against_dense_reference(lengths, prefixes, splits):
    torch.manual_seed(42)
    layout, _, hist, window, meta, tables = make_state(lengths, prefixes)
    rk = torch.linalg.qr(torch.randn(256, 256))[0].bfloat16().to("npu")
    rv = torch.linalg.qr(torch.randn(256, 256))[0].bfloat16().to("npu")
    ks = [torch.randn(n, 1, 256).bfloat16() for n in lengths]
    vs = [torch.randn(n, 1, 256).bfloat16() for n in lengths]
    qs = [torch.randn(n - p, 6, 256).bfloat16() for n, p in zip(lengths, prefixes)]
    prepare_prefix(ks, vs, rk, rv, hist, window, meta, tables, layout, prefixes)
    q = torch.cat(qs).to("npu")
    k = torch.cat([x[p:] for x, p in zip(ks, prefixes)]).to("npu")
    v = torch.cat([x[p:] for x, p in zip(vs, prefixes)]).to("npu")
    actual = run_attention(q, k, v, rk, rv, hist, window, meta, layout, splits).float().cpu()
    expected = torch.cat(
        [
            hybrid_reference(qi, ki, vi, prefix, rk.cpu(), rv.cpu(), layout.config)
            for qi, ki, vi, prefix in zip(qs, ks, vs, prefixes)
        ]
    )
    torch.testing.assert_close(actual, expected, atol=0.035, rtol=0.04)


def test_graph_replay_changes_metadata_and_page_owners():
    # Replay changes request count 1 -> 2, same total tokens and input pointers.
    torch.manual_seed(10)
    layout, _, hist, window, meta, _ = make_state([4], [0], max_reqs=4)
    q = torch.randn(4, 6, 256, device="npu", dtype=torch.bfloat16)
    k = torch.randn(4, 1, 256, device="npu", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    rk = rv = torch.eye(256, device="npu", dtype=torch.bfloat16)
    for _ in range(3):
        run_attention(q, k, v, rk, rv, hist, window, meta, layout, 8)
    torch.npu.synchronize()
    # Startup capture owns no cache pages; all accesses are masked by device
    # counts. Replay must still activate the same recorded kernels afterwards.
    meta.counts.zero_()
    before_capture = hist.clone(), window.clone()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = run_attention(q, k, v, rk, rv, hist, window, meta, layout, 8)
    torch.npu.synchronize()
    assert torch.equal(hist, before_capture[0]) and torch.equal(window, before_capture[1])
    meta.query_start_loc.copy_(torch.tensor([0, 2, 4, 0, 0], device="npu", dtype=torch.int32))
    meta.seq_lens[:2].fill_(2)
    meta.counts[0].fill_(2)
    meta.counts[1].fill_(4)
    meta.block_tables[0, :6].copy_(torch.arange(18, 24, device="npu", dtype=torch.int32))
    meta.block_tables[1, :6].copy_(torch.arange(36, 42, device="npu", dtype=torch.int32))
    meta.slot_mapping.copy_(
        torch.tensor([18 * 128, 18 * 128 + 1, 36 * 128, 36 * 128 + 1], device="npu")
    )
    q.normal_()
    k.normal_()
    v.normal_()
    graph.replay()
    torch.npu.synchronize()
    expected = torch.cat(
        [
            hybrid_reference(
                q[i : i + 2].cpu(),
                k[i : i + 2].cpu(),
                v[i : i + 2].cpu(),
                0,
                rk.cpu(),
                rv.cpu(),
                layout.config,
            )
            for i in (0, 2)
        ]
    )
    torch.testing.assert_close(actual.float().cpu(), expected, atol=0.035, rtol=0.04)


def test_device_ring_survives_mtp_rejection_and_rewrites():
    torch.manual_seed(81)
    layout, _, hist, window, meta, tables = make_state([1500], [0], max_reqs=2)
    rk = rv = torch.eye(256, device="npu", dtype=torch.bfloat16)
    keys = torch.randn(1500, 1, 256).bfloat16()
    values = torch.randn_like(keys)
    prepare_prefix([keys], [values], rk, rv, hist, window, meta, tables, layout, [900])
    prefix = 900
    for rejected in (3, 2, 0, 3, 1):
        keys[prefix : prefix + 4].normal_()
        values[prefix : prefix + 4].normal_()
        query = torch.randn(4, 6, 256, device="npu", dtype=torch.bfloat16)
        meta.query_start_loc[:2].copy_(torch.tensor([0, 4], device="npu", dtype=torch.int32))
        meta.seq_lens[0].fill_(prefix + 4)
        meta.counts.copy_(torch.tensor([1, 4], device="npu", dtype=torch.int32))
        meta.slot_mapping = torch.tensor(
            [int(tables[0, t // 128]) * 128 + t % 128 for t in range(prefix, prefix + 4)],
            device="npu",
            dtype=torch.int64,
        )
        actual = run_attention(
            query,
            keys[prefix : prefix + 4].to("npu"),
            values[prefix : prefix + 4].to("npu"),
            rk,
            rv,
            hist,
            window,
            meta,
            layout,
            16,
        )
        expected = hybrid_reference(
            query.cpu(),
            keys[: prefix + 4],
            values[: prefix + 4],
            prefix,
            rk.cpu(),
            rv.cpu(),
            layout.config,
        )
        torch.testing.assert_close(actual.float().cpu(), expected, atol=0.035, rtol=0.04)
        prefix += 4 - rejected


def test_metadata_builder_capture_entry_and_fixed_pointers():
    from vllm_ascend.attention.attention_v1 import AscendAttentionState

    from oscar_ascend.backend import OscarAttentionImpl
    from oscar_ascend.metadata import OscarMetadataBuilder

    name = "model.language_model.layers.3.self_attn.attn"
    # The builder only needs a layer's implementation type; no model is loaded
    # in this isolated device-metadata test.
    impl = object.__new__(OscarAttentionImpl)
    cfg = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={name: SimpleNamespace(impl=impl)}
        ),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=32),
        model_config=SimpleNamespace(max_model_len=4096),
    )
    spec = SimpleNamespace(block_size=768, num_kv_heads=1)
    builder = OscarMetadataBuilder(spec, [name], cfg, torch.device("npu"))
    cm = SimpleNamespace(
        causal=True,
        num_reqs=1,
        num_actual_tokens=4,
        max_query_len=4,
        query_start_loc=torch.tensor([0, 4], device="npu", dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=torch.tensor([4], device="npu", dtype=torch.int32),
        block_table_tensor=torch.arange(6, 12, device="npu", dtype=torch.int32)[None, :],
        slot_mapping=torch.arange(768, 772, device="npu", dtype=torch.int64),
        attn_state=AscendAttentionState.PrefillNoCache,
    )
    captured = builder.build_for_cudagraph_capture(cm)
    assert captured.initial_prefill is False
    assert captured.is_dummy and captured.counts.cpu().tolist() == [0, 0]
    pointers = [
        getattr(captured, attr).data_ptr()
        for attr in ("query_start_loc", "seq_lens", "slot_mapping", "block_tables", "counts")
    ]
    cm.attn_state = AscendAttentionState.SpecDecoding
    cm.seq_lens.fill_(1200)
    replay = builder.build(0, cm)
    assert pointers == [
        getattr(replay, attr).data_ptr()
        for attr in ("query_start_loc", "seq_lens", "slot_mapping", "block_tables", "counts")
    ]
    assert replay.seq_lens[0].item() == 1200  # Test-only assertion, outside serving.
    assert not replay.is_dummy and replay.counts.cpu().tolist() == [1, 4]
