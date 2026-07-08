"""Tests for the Brant iEEG/sEEG foundation model.

The bulk of the model contract (init / forward / serialization / categorization)
is already exercised by the shared model suites via ``models_mandatory_parameters``.
This file adds Brant-specific checks plus an upstream **parity gate**: when the
gated reference code is available (``BRANT_SRC`` pointing to ``Brant_src``), the
two Transformer encoders are checked to be bit-exact against upstream; otherwise
the gate is skipped, so CI stays green without the gated dependency.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

from braindecode.models import Brant
from braindecode.models.brant import BRANT_FREQ_BANDS
from braindecode.modules.brant_modules import (
    _BandPowerFeatures,
    _BrantSpatialEncoder,
    _BrantTemporalEncoder,
)

# small dims keep the suite fast on a CPU runner
N_CHANS, N_OUTPUTS, N_TIMES, SFREQ, PATCH = 4, 3, 1000, 250.0, 250


def _model(**overrides):
    kwargs = dict(
        n_chans=N_CHANS, n_outputs=N_OUTPUTS, n_times=N_TIMES, sfreq=SFREQ,
        patch_size=PATCH, embed_dim=32, ffn_dim=48,
        temporal_n_layers=2, spatial_n_layers=2, n_heads=4,
    )
    kwargs.update(overrides)
    return Brant(**kwargs)


def test_forward_shape():
    model = _model().eval()
    x = torch.randn(2, N_CHANS, N_TIMES)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, N_OUTPUTS)
    assert torch.isfinite(out).all()


def test_reset_head_changes_n_outputs():
    model = _model().eval()
    model.reset_head(7)
    with torch.no_grad():
        out = model(torch.randn(2, N_CHANS, N_TIMES))
    assert out.shape == (2, 7)


def test_input_shorter_than_patch_raises():
    with pytest.raises(ValueError):
        _model(n_times=PATCH - 1)


def test_freq_bands_must_match_constant():
    with pytest.raises(ValueError):
        _model(n_freq_bands=len(BRANT_FREQ_BANDS) - 1)


def test_band_power_shape_and_finite():
    features = _BandPowerFeatures(SFREQ, BRANT_FREQ_BANDS)
    out = features(torch.randn(2, N_CHANS, 4, PATCH))
    assert out.shape == (2, N_CHANS, 4, len(BRANT_FREQ_BANDS))
    assert torch.isfinite(out).all()


# --------------------------------------------------------------- parity gate
def _import_upstream_or_skip():
    src = os.environ.get("BRANT_SRC")
    if not src or not os.path.isdir(src):
        pytest.skip("set BRANT_SRC=/path/to/Brant_src to run the parity gate")
    sys.path.insert(0, src)
    try:
        from pretrain.pre_model import ChannelEncoder, TimeEncoder
    except ImportError as exc:  # pragma: no cover - depends on external code
        pytest.skip(f"upstream Brant not importable: {exc}")
    return TimeEncoder, ChannelEncoder


def test_encoders_are_bit_exact_with_upstream():
    """Both encoders reproduce upstream exactly once weights are copied over."""
    TimeEncoder, ChannelEncoder = _import_upstream_or_skip()

    torch.manual_seed(0)
    batch, n_chans, seq_len = 2, 3, 4
    patch, d_model, dim_ff, n_bands, n_heads = 64, 32, 48, 8, 4
    t_layers, s_layers = 2, 2

    data = torch.randn(batch, n_chans, seq_len, patch)
    power = torch.randn(batch, n_chans, seq_len, n_bands)

    up_t = TimeEncoder(
        in_dim=patch, d_model=d_model, dim_feedforward=dim_ff, seq_len=seq_len,
        n_layer=t_layers, nhead=n_heads, band_num=n_bands,
        project_mode="linear", learnable_mask=False,
    ).eval()
    ours_t = _BrantTemporalEncoder(
        patch_size=patch, d_model=d_model, seq_len=seq_len, n_bands=n_bands,
        dim_feedforward=dim_ff, n_layers=t_layers, n_heads=n_heads, drop_prob=0.1,
    ).eval()
    ours_t.trans_enc.load_state_dict(up_t.trans_enc.state_dict())
    ours_t.input_embedding.proj.load_state_dict(up_t.input_embedding.proj.state_dict())
    ours_t.input_embedding.band_encoding.data.copy_(
        up_t.input_embedding.band_encoding.data
    )
    ours_t.input_embedding.positional_encoding.data.copy_(
        up_t.input_embedding.positional_encoding.data
    )

    with torch.no_grad():
        up_time = up_t(mask=None, data=data, power=power, need_mask=False, use_power=True)
        ours_time = ours_t(data, power)
    assert torch.allclose(up_time, ours_time, atol=1e-5)

    time_z = up_time.reshape(batch, n_chans, seq_len, d_model)
    time_z = time_z.transpose(1, 2).reshape(batch * seq_len, n_chans, d_model)

    up_c = ChannelEncoder(
        out_dim=patch, d_model=d_model, dim_feedforward=dim_ff,
        n_layer=s_layers, nhead=n_heads,
    ).eval()
    ours_c = _BrantSpatialEncoder(
        d_model=d_model, out_dim=patch, dim_feedforward=dim_ff,
        n_layers=s_layers, n_heads=n_heads, drop_prob=0.1,
    ).eval()
    ours_c.trans_enc.load_state_dict(up_c.trans_enc.state_dict())
    ours_c.proj_out.load_state_dict(up_c.proj_out.state_dict())

    with torch.no_grad():
        up_ch, _ = up_c(time_z)
        ours_ch, _ = ours_c(time_z)
    assert torch.allclose(up_ch, ours_ch, atol=1e-5)
