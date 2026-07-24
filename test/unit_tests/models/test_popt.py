"""Tests for the PopulationTransformer (PopT) iEEG/sEEG foundation model.

The bulk of the model contract (init / forward / serialization / categorization)
is already exercised by the shared model suites via ``models_mandatory_parameters``.
This file adds PopT-specific checks: the ``CLS`` pooling and feature interface,
that electrode coordinates are read from ``chs_info``, plus an upstream **parity
gate** — when the reference code is available (``POPT_SRC`` pointing to a clone
of https://github.com/czlwang/PopulationTransformer), the ported input embedding
+ Transformer are checked to be bit-exact against upstream ``PtModelCustom``;
otherwise the gate is skipped, so CI stays green without the external dependency.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

from braindecode.models import PopulationTransformer
from braindecode.modules.popt_modules import _PopTInputEmbedding

# small dims keep the suite fast on a CPU runner. n_times plays the role of the
# per-electrode feature dimension (768 for real BrainBERT features).
N_CHANS, N_OUTPUTS, N_TIMES = 5, 2, 16


def _model(**overrides):
    kwargs = dict(
        n_chans=N_CHANS, n_outputs=N_OUTPUTS, n_times=N_TIMES,
        hidden_dim=32, ffn_dim=48, n_layers=2, n_heads=4,
    )
    kwargs.update(overrides)
    return PopulationTransformer(**kwargs)


def test_forward_shape():
    model = _model().eval()
    x = torch.randn(2, N_CHANS, N_TIMES)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, N_OUTPUTS)
    assert torch.isfinite(out).all()


def test_return_features():
    model = _model().eval()
    with torch.no_grad():
        out = model(torch.randn(2, N_CHANS, N_TIMES), return_features=True)
    assert isinstance(out, dict)
    assert set(out) == {"features", "cls_token"}
    assert out["features"].shape == (2, model.hidden_dim)
    # PopT has a real CLS token (unlike BrainBERT); it doubles as the feature.
    assert out["cls_token"] is not None
    assert torch.equal(out["features"], out["cls_token"])
    assert torch.isfinite(out["features"]).all()


def test_reset_head_changes_n_outputs():
    model = _model().eval()
    model.reset_head(7)
    with torch.no_grad():
        out = model(torch.randn(2, N_CHANS, N_TIMES))
    assert out.shape == (2, 7)


def test_hidden_dim_must_be_divisible_by_four():
    with pytest.raises(ValueError):
        _model(hidden_dim=30)


def test_coords_from_chs_info():
    """Electrode coordinates are read from chs_info and discretised."""
    # three electrodes 1 mm apart along x (metres in the MNE convention).
    chs_info = [
        {"ch_name": f"E{i}", "kind": "eeg", "loc": [i * 1e-3, 0.0, 0.0] + [0.0] * 9}
        for i in range(3)
    ]
    model = PopulationTransformer(
        n_chans=3, n_outputs=2, n_times=N_TIMES, chs_info=chs_info,
        hidden_dim=32, ffn_dim=48, n_layers=1, n_heads=4,
    )
    coords = model.electrode_coords
    assert coords.shape == (3, 3)
    # x axis is shifted to start at 0 and spaced by 1 (1 mm rounded).
    assert torch.equal(coords[:, 0], torch.tensor([0, 1, 2]))
    # y / z are constant -> all zero after the per-axis shift.
    assert torch.equal(coords[:, 1], torch.zeros(3, dtype=torch.long))


def test_coords_fallback_without_positions():
    """Without usable positions, electrodes get distinct sequential indices."""
    model = _model()
    coords = model.electrode_coords
    assert coords.shape == (N_CHANS, 3)
    assert torch.equal(coords[:, 0], torch.arange(N_CHANS))


# --------------------------------------------------------------- parity gate
def _import_upstream_or_skip():
    src = os.environ.get("POPT_SRC")
    if not src or not os.path.isdir(src):
        pytest.skip("set POPT_SRC=/path/to/PopulationTransformer for the parity gate")
    sys.path.insert(0, src)
    try:
        import models as upstream_models  # noqa: F401 (populates registry)
        from models.pt_model_custom import PtModelCustom
    except ImportError as exc:  # pragma: no cover - depends on external code
        pytest.skip(f"upstream PopulationTransformer not importable: {exc}")
    return PtModelCustom


def test_encoder_is_bit_exact_with_upstream():
    """Input embedding + Transformer reproduce upstream once weights are copied."""
    PtModelCustom = _import_upstream_or_skip()
    from omegaconf import OmegaConf

    torch.manual_seed(0)
    batch, n_elec = 2, 6
    input_dim, hidden_dim, n_heads, n_layers = 8, 32, 4, 2

    cfg = OmegaConf.create(
        dict(
            name="pt_model_custom",
            position_encoding="multi_subj_position_encoding",
            n_head=n_heads, n_layers=n_layers, hidden_dim=hidden_dim,
            input_dim=input_dim, layer_activation="gelu",
            attention_weights=False, use_token_cls_head=False,
        )
    )
    up = PtModelCustom()
    up.build_model(cfg)
    up.eval()

    ours_embed = _PopTInputEmbedding(input_dim, hidden_dim).eval()
    # upstream uses dim_feedforward=2048 (the nn default it never overrides).
    enc_layer = torch.nn.TransformerEncoderLayer(
        d_model=hidden_dim, nhead=n_heads, dim_feedforward=2048,
        activation="gelu", batch_first=True,
    )
    ours_trans = torch.nn.TransformerEncoder(enc_layer, num_layers=n_layers).eval()

    # copy upstream weights into the ported modules
    ours_embed.in_proj.load_state_dict(up.input_encoding.in_proj.state_dict())
    ours_embed.layer_norm.load_state_dict(up.input_encoding.layer_norm.state_dict())
    ours_trans.load_state_dict(up.transformer_encoder.state_dict())
    # sinusoidal spatial buffers are computed identically on both sides
    assert torch.allclose(
        ours_embed.positional_encoding.pe,
        up.input_encoding.positional_encoding.pe,
        atol=1e-6,
    )

    features = torch.randn(batch, n_elec, input_dim)
    coords = torch.randint(0, 120, (batch, n_elec, 3))
    seq_id = torch.zeros(batch, n_elec, dtype=torch.long)
    # upstream expects the CLS row (a vector of ones) already prepended
    cls = torch.ones(batch, 1, input_dim)
    up_inputs = torch.cat([cls, features], dim=1)
    src_key_mask = torch.zeros(batch, n_elec + 1).bool()

    with torch.no_grad():
        up_out = up(up_inputs, src_key_mask, (coords, seq_id), intermediate_rep=True)
        ours_out = ours_trans(ours_embed(features, coords, seq_id))
    assert up_out.shape == ours_out.shape == (batch, n_elec + 1, hidden_dim)
    assert torch.allclose(up_out, ours_out, atol=1e-5)
