"""Tests for the MVPFormer reimplementation.

The end-to-end equivalence test runs the whole pipeline (wavelet encoder, MVPA
in both attention modes, blocks, pooling, head) against the upstream
transcription in ``_mvpformer_reference.py`` with shared weights, so a break in
any component fails it. The rest cover logic that path does not: the LoRA merge
and the real released checkpoints. See
``docs/superpowers/specs/2026-06-18-mvpformer-braindecode-design.md``.
"""

import os

import pytest
import torch

from braindecode.models._mvpformer_convert import (
    convert_mvpformer_state_dict,
    merge_swec_checkpoint,
)
from braindecode.models.mvpformer import MVPFormer
from test.unit_tests.models._mvpformer_reference import (
    RefMVPFormerModel,
    make_ref_config,
)

# Small CI config; MVPFormer-S dims (configs/mvpformer_s_*.yaml) for real ckpts.
DIM = dict(d_model=16, n_heads=4, n_head_kv=2, d_inner=32, n_layers=2)
SEG_LEN, N_SEG, N_CHANS, N_OUT, MAX_SEG, MAX_CHAN, SFREQ = 64, 4, 3, 2, 16, 16, 100.0
S_DIM = dict(d_model=768, n_layers=12, n_heads=12, n_head_kv=4, d_inner=1728)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def make_ref(global_att=True):
    cfg = make_ref_config(
        hidden_size=DIM["d_model"],
        num_attention_heads=DIM["n_heads"],
        n_head_kv=DIM["n_head_kv"],
        intermediate_size=DIM["d_inner"],
        n_inner=DIM["d_inner"],
        n_layer=DIM["n_layers"],
        max_position_embeddings=MAX_SEG,
        n_channels=MAX_CHAN,
        global_att=global_att,
    )
    return RefMVPFormerModel(cfg, segment_len=SEG_LEN, n_outputs=N_OUT).eval()


def make_model(**overrides):
    kw = dict(
        n_chans=N_CHANS,
        n_outputs=N_OUT,
        n_times=N_SEG * SEG_LEN,
        sfreq=SFREQ,
        segment_seconds=SEG_LEN / SFREQ,
        max_segments=MAX_SEG,
        max_channels=MAX_CHAN,
        drop_prob=0.0,
        **DIM,
    )
    return MVPFormer(**{**kw, **overrides}).eval()


def _ref_to_upstream(ref_sd):
    """RefMVPFormerModel names -> the real upstream ('genie.') checkpoint scheme,
    adding the vestigial buffers/heads the converter must drop."""
    out, layers = {}, set()
    for k, v in ref_sd.items():
        if k.startswith("encoder."):
            out[k] = v
        elif k.startswith("head."):
            out["head.head." + k[len("head.") :]] = v  # generative head -> dropped
        else:
            out["genie." + k] = v
        if k.startswith("h."):
            layers.add(int(k.split(".")[1]))
    out["seizure_embeddings"] = torch.zeros(3, 4)  # generative-only -> dropped
    for i in layers:  # vestigial attention buffers -> dropped
        out[f"genie.h.{i}.attn.time_bias"] = torch.zeros(1, 1, 8, 8)
        out[f"genie.h.{i}.attn.masked_bias"] = torch.tensor(-1e4)
    return out


def _upstream_to_reference(raw):
    """Real upstream checkpoint -> reference oracle: strip 'genie.', drop the
    generative head + vestigial buffers (the oracle keeps ln_2)."""
    out = {}
    for k, v in raw.items():
        if k == "seizure_embeddings" or k.startswith("head."):
            continue
        if k.endswith(".time_bias") or k.endswith(".masked_bias"):
            continue
        for p in ("genie.", "mvpformer."):
            if k.startswith(p):
                k = k[len(p) :]
                break
        out[k] = v
    return out


@pytest.mark.parametrize("global_att", [True, False])
def test_mvpformer_matches_reference(global_att):
    """Full model (raw signal -> logits) equals the upstream classification path,
    with weights routed through the temporary base-checkpoint converter."""
    ref, mine = make_ref(global_att), make_model(global_att=global_att)
    converted = convert_mvpformer_state_dict(_ref_to_upstream(ref.state_dict()), "base")
    missing, unexpected = mine.load_state_dict(converted, strict=False)
    assert not unexpected, unexpected
    assert all("final_layer" in k for k in missing), missing  # fresh head only
    mine.final_layer.weight.data.copy_(ref.head.weight.data)

    raw = torch.randn(2, N_CHANS, N_SEG * SEG_LEN)
    ref_input = raw.reshape(2, N_CHANS, N_SEG, SEG_LEN).permute(0, 2, 1, 3).contiguous()
    out = mine(raw)
    assert out.shape == (2, N_OUT)
    assert torch.allclose(out, ref(ref_input), atol=1e-5)


def test_merge_swec_checkpoint():
    """merge_swec_checkpoint merges LoRA (q_attn, c_attn) with scaling=alpha/rank
    and installs the swec head, giving a complete classifier."""
    r = 4
    kv_dim = (DIM["d_model"] // DIM["n_heads"]) * DIM["n_head_kv"]
    base_sd = _ref_to_upstream(make_ref().state_dict())
    swec_sd = {"head.head.weight": torch.randn(N_OUT, DIM["d_model"])}
    for i in range(DIM["n_layers"]):
        for name, out_dim in (("q_attn", DIM["d_model"]), ("c_attn", 2 * kv_dim)):
            swec_sd[f"genie.h.{i}.attn.{name}.lora_A"] = torch.randn(r, DIM["d_model"])
            swec_sd[f"genie.h.{i}.attn.{name}.lora_B"] = torch.randn(out_dim, r)

    merged = merge_swec_checkpoint(base_sd, swec_sd, lora_alpha=16, lora_rank=8)
    for name in ("q_attn", "c_attn"):  # scaling = 16/8 = 2
        base_w = base_sd[f"genie.h.0.attn.{name}.weight"]
        delta = (
            swec_sd[f"genie.h.0.attn.{name}.lora_B"]
            @ swec_sd[f"genie.h.0.attn.{name}.lora_A"]
        )
        assert torch.allclose(
            merged[f"blocks.0.attn.{name}.weight"], base_w + 2.0 * delta, atol=1e-6
        )
    assert torch.allclose(merged["final_layer.weight"], swec_sd["head.head.weight"])
    missing, unexpected = make_model().load_state_dict(merged, strict=False)
    assert not missing and not unexpected, (missing, unexpected)


# Real released-checkpoint validation; skipped unless the .pt files are present.
_S_BASE = os.environ.get("MVPFORMER_S_BASE_CKPT")
_S_SWEC = os.environ.get("MVPFORMER_S_SWEC_CKPT")


def _make_s_model(n_chans, n_seg):
    return MVPFormer(
        n_chans=n_chans, n_outputs=2, n_times=2560 * n_seg, sfreq=512.0,
        segment_seconds=5.0, max_segments=110, max_channels=128, drop_prob=0.0, **S_DIM,
    ).eval()


@pytest.mark.skipif(not _S_BASE, reason="MVPFORMER_S_BASE_CKPT not set")
def test_real_s_base_checkpoint_cpu():
    """The real MVPFormer-S base checkpoint converts, loads cleanly, and matches
    the CPU-shimmed original."""
    raw = torch.load(_S_BASE, map_location="cpu", weights_only=True)
    n_chans, n_seg = 32, 4
    mine = _make_s_model(n_chans, n_seg)
    missing, unexpected = mine.load_state_dict(
        convert_mvpformer_state_dict(raw, "base"), strict=False
    )
    assert not unexpected, unexpected[:10]
    assert all("final_layer" in k for k in missing), missing  # fresh head only

    cfg = make_ref_config(
        hidden_size=768, num_attention_heads=12, n_head_kv=4, intermediate_size=1728,
        n_inner=1728, n_layer=12, max_position_embeddings=110, n_channels=128,
    )
    ref = RefMVPFormerModel(cfg, segment_len=2560, n_outputs=2).eval()
    _, unexpected_r = ref.load_state_dict(_upstream_to_reference(raw), strict=False)
    assert not unexpected_r, unexpected_r[:10]
    ref.head.weight.data.copy_(mine.final_layer.weight.data)

    x = torch.randn(1, n_chans, 2560 * n_seg)
    ref_input = x.reshape(1, n_chans, n_seg, 2560).permute(0, 2, 1, 3).contiguous()
    with torch.no_grad():
        out = mine(x)
        assert out.shape == (1, 2) and torch.isfinite(out).all()
        assert torch.allclose(out, ref(ref_input), atol=1e-4)


@pytest.mark.skipif(
    not (_S_BASE and _S_SWEC), reason="MVPFORMER_S_BASE/SWEC_CKPT not set"
)
def test_real_s_swec_classifier_cpu():
    """The real base + swec merge into a complete seizure classifier on CPU."""
    base = torch.load(_S_BASE, map_location="cpu", weights_only=True)
    swec = torch.load(_S_SWEC, map_location="cpu", weights_only=True)
    n_chans, n_seg = 32, 4
    mine = _make_s_model(n_chans, n_seg)
    missing, unexpected = mine.load_state_dict(merge_swec_checkpoint(base, swec), strict=False)
    assert not missing and not unexpected, (missing[:10], unexpected[:10])
    with torch.no_grad():
        out = mine(torch.randn(1, n_chans, 2560 * n_seg))
    assert out.shape == (1, 2) and torch.isfinite(out).all()
