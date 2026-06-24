# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: MIT
from __future__ import annotations

import importlib.util

import pytest
import torch

from braindecode.modules import ChannelMerger, FourierEmb

_HAS_NEURALTRAIN = importlib.util.find_spec("neuraltrain") is not None


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
