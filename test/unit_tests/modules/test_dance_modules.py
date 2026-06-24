# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: MIT
from __future__ import annotations

import importlib.util

import pytest
import torch

from braindecode.modules import ChannelMerger, FourierEmb, Perceiver, SimpleConv

_HAS_NEURALTRAIN = importlib.util.find_spec("neuraltrain") is not None
_HAS_PERCEIVER = importlib.util.find_spec("perceiver_pytorch") is not None


def test_fourier_emb_shape_and_freqs():
    emb = FourierEmb(dimension=2048)
    assert emb.pos.shape == (32,)  # (2048/2) ** (1/2) == 32
    positions = torch.rand(2, 19, 2)  # (B, n_chans, 2) in [0, 1]
    out = emb(positions)
    assert out.shape == (2, 19, 2048)
    assert torch.isfinite(out).all()


def test_channel_merger_output_channels():
    merger = ChannelMerger(out_channels=270, pos_dim=2048, dropout=0.2)
    assert merger.heads.shape == (270, 2048)
    x = torch.randn(2, 19, 500)
    positions = torch.rand(2, 19, 2)
    out = merger(x, positions)
    assert out.shape == (2, 270, 500)  # 19 -> 270 virtual channels, T preserved


@pytest.mark.skipif(not _HAS_NEURALTRAIN, reason="upstream neuraltrain not installed")
def test_fourier_emb_parity():
    """Gate: FourierEmb must equal neuraltrain FourierEmbModel."""
    from neuraltrain.models.common import FourierEmb as UpFourierEmbConfig

    up = UpFourierEmbConfig(n_freqs=None, total_dim=2048, n_dims=2).build().eval()
    mine = FourierEmb(dimension=2048).eval()
    # pos buffer must match exactly (2*pi*arange / (1+2*margin)).
    torch.testing.assert_close(mine.pos, up.pos, rtol=0, atol=1e-6)
    positions = torch.rand(2, 19, 2)
    torch.testing.assert_close(mine(positions), up(positions), rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not _HAS_NEURALTRAIN, reason="upstream neuraltrain not installed")
def test_channel_merger_parity():
    """Gate: edit FourierEmb/ChannelMerger until scores+output match upstream."""
    from neuraltrain.models.common import ChannelMerger as UpChannelMergerConfig

    torch.manual_seed(0)
    positions = torch.rand(2, 19, 2)
    x = torch.randn(2, 19, 64)
    mine = ChannelMerger(out_channels=270, pos_dim=2048, dropout=0.0).eval()
    up = UpChannelMergerConfig(
        n_virtual_channels=270,
        fourier_emb_config=UpChannelMergerConfig.model_fields[
            "fourier_emb_config"
        ].default.__class__(n_freqs=None, total_dim=2048, n_dims=2),
        dropout=0.0,
        per_subject=False,
        embed_ref=False,
    ).build().eval()
    # Copy heads + pos so only the math is under test (heads init differs).
    up.heads.data.copy_(mine.heads.detach())
    up.embedding.pos.copy_(mine.embedding.pos)
    subject_ids = torch.zeros(2, dtype=torch.long)  # unused (per_subject=False)
    mine_out = mine(x, positions)
    up_out = up(x, subject_ids, positions)  # upstream forward(meg, subject_ids, positions)
    torch.testing.assert_close(mine_out, up_out, rtol=1e-4, atol=1e-5)


def test_simpleconv_shape_and_length_preserved():
    conv = SimpleConv(in_channels=270, out_channels=128, depth=10)
    x = torch.randn(2, 270, 2000)
    out = conv(x)
    assert out.shape == (2, 128, 2000)  # T preserved


def test_simpleconv_dilation_growth():
    conv = SimpleConv(
        in_channels=270, out_channels=128, depth=10,
        kernel_size=9, dilation_growth=2.5,
    )
    dilations = [
        m.dilation[0] for m in conv.modules()
        if isinstance(m, torch.nn.Conv1d) and m.kernel_size[0] == 9
    ]
    # Upstream ConvSequence accumulates `dilation = dilation * dilation_growth`
    # and applies `int(dilation)` at each block (NOT round(growth**i)):
    # block0=int(1)=1, block1=int(2.5)=2, block2=int(6.25)=6, block3=int(15.625)=15,
    # ... block9=int(2.5**9)=int(3814.69..)=3814.
    assert len(dilations) == 10
    assert dilations[:4] == [1, 2, 6, 15]
    assert dilations[-1] == 3814
    # Same value exposed for the model's min-length guard.
    assert conv.max_dilation == dilations[-1]


@pytest.mark.skipif(not _HAS_NEURALTRAIN, reason="upstream neuraltrain not installed")
def test_simpleconv_parity():
    """Gate: SimpleConv conv stack must match upstream SimpleConvModel."""
    from neuraltrain.models.simpleconv import SimpleConv as UpSimpleConvConfig

    torch.manual_seed(0)
    # merger=None on both sides isolates the conv stack (initial_linear + blocks).
    up = UpSimpleConvConfig(
        hidden=512, depth=10, kernel_size=9, dilation_growth=2.5,
        initial_linear=256, initial_depth=1, merger_config=None,
    ).build(n_in_channels=256, n_outputs=128).eval()
    mine = SimpleConv(
        in_channels=256, out_channels=128, hidden=512, depth=10,
        kernel_size=9, dilation_growth=2.5, initial_linear=256, initial_depth=1,
        merger=None,
    ).eval()
    # Copy conv weights: upstream initial_linear.0 -> mine.initial_linear.0;
    # upstream encoder.sequence.{k}.0 (Conv1d) -> mine.sequence.{k}.conv.
    sd = mine.state_dict()
    up_sd = up.state_dict()
    sd["initial_linear.0.weight"] = up_sd["initial_linear.0.weight"].clone()
    sd["initial_linear.0.bias"] = up_sd["initial_linear.0.bias"].clone()
    for k in range(10):
        sd[f"sequence.{k}.conv.weight"] = up_sd[f"encoder.sequence.{k}.0.weight"].clone()
        sd[f"sequence.{k}.conv.bias"] = up_sd[f"encoder.sequence.{k}.0.bias"].clone()
    mine.load_state_dict(sd)
    x = torch.randn(2, 256, 2000)
    torch.testing.assert_close(mine(x), up(x), rtol=1e-4, atol=1e-5)


def test_perceiver_fixed_output_length_agnostic():
    perc = Perceiver(input_dim=128, num_latents=256, latent_dim=128)
    for t in (500, 2000):
        out = perc(torch.randn(2, t, 128))
        assert out.shape == (2, 256, 128)  # fixed regardless of T


def test_perceiver_latents_param_shape():
    perc = Perceiver(input_dim=128, num_latents=256, latent_dim=128)
    assert perc.latents.shape == (256, 128)
    assert torch.isfinite(perc.latents).all()


@pytest.mark.skipif(not _HAS_PERCEIVER, reason="perceiver_pytorch not installed")
def test_perceiver_fourier_encode_parity():
    """Gate: _fourier_encode must match perceiver_pytorch.fourier_encode."""
    from perceiver_pytorch.perceiver_pytorch import fourier_encode

    from braindecode.modules.dance_modules import _fourier_encode

    pos = torch.linspace(-1, 1, 7).reshape(7, 1)
    up = fourier_encode(pos[..., 0], max_freq=10.0, num_bands=6)
    mine = _fourier_encode(pos[..., 0], 10.0, 6)
    assert mine.shape[-1] == 13
    torch.testing.assert_close(mine, up, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not _HAS_PERCEIVER, reason="perceiver_pytorch not installed")
def test_perceiver_full_block_parity():
    """Gate: full Perceiver latents must match perceiver_pytorch.Perceiver
    built with DANCE's exact upstream kwargs (block layout + weights)."""
    from perceiver_pytorch import Perceiver as PtPerceiver

    torch.manual_seed(0)
    up = PtPerceiver(
        input_channels=128,
        latent_dim=128,
        depth=6,
        num_latents=256,
        cross_heads=2,
        latent_heads=2,
        cross_dim_head=64,
        latent_dim_head=64,
        max_freq=10,
        num_freq_bands=6,
        input_axis=1,
        num_classes=1000,
        final_classifier_head=False,
        attn_dropout=0.0,
        ff_dropout=0.0,
        weight_tie_layers=False,
        fourier_encode_data=True,
        self_per_cross_attn=1,
    ).eval()
    mine = Perceiver(
        input_dim=128, num_latents=256, latent_dim=128, depth=6,
        cross_heads=2, latent_heads=2, cross_dim_head=64, latent_dim_head=64,
        max_freq=10.0, num_freq_bands=6, self_per_cross_attn=1,
    ).eval()
    # Our submodule names mirror perceiver_pytorch's layers.* keys exactly;
    # only `latents` (sinusoidal vs random) and `to_logits` (Identity) differ.
    up_sd = up.state_dict()
    mine_sd = mine.state_dict()
    assert set(k for k in up_sd if k.startswith("layers.")) == set(
        k for k in mine_sd if k.startswith("layers.")
    ), "layer key sets must match exactly (no per-layer rename needed)"
    for k in list(mine_sd):
        if k.startswith("layers."):
            mine_sd[k] = up_sd[k].clone()
    mine_sd["latents"] = up_sd["latents"].clone()  # equalise latent init too
    mine.load_state_dict(mine_sd, strict=False)
    data = torch.randn(2, 500, 128)
    torch.testing.assert_close(mine(data), up(data), rtol=1e-4, atol=1e-5)
