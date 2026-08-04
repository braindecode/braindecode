"""Tests for the BrainBERT iEEG/sEEG foundation model.

The bulk of the model contract (init / forward / serialization / categorization)
is already exercised by the shared model suites via ``models_mandatory_parameters``.
This file adds BrainBERT-specific checks: that the in-model STFT front-end
reproduces the upstream scipy spectrogram, plus an upstream **parity gate** —
when the reference code is available (``BRAINBERT_SRC`` pointing to a clone of
https://github.com/czlwang/BrainBERT), the ported input encoding + Transformer
are checked to be bit-exact against upstream ``MaskedTFModel``; otherwise the
gate is skipped, so CI stays green without the external dependency.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest
import torch

from braindecode.models import BrainBERT
from braindecode.modules.brainbert_modules import (
    _BrainBERTInputEmbedding,
    _STFTSpectrogram,
)

# small dims keep the suite fast on a CPU runner
N_CHANS, N_OUTPUTS, N_TIMES, SFREQ = 3, 2, 1000, 2048.0


def _model(**overrides):
    kwargs = dict(
        n_chans=N_CHANS, n_outputs=N_OUTPUTS, n_times=N_TIMES, sfreq=SFREQ,
        hidden_dim=32, ffn_dim=48, n_layers=2, n_heads=4,
    )
    kwargs.update(overrides)
    return BrainBERT(**kwargs)


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
    assert out["cls_token"] is None
    assert torch.isfinite(out["features"]).all()


def test_reset_head_changes_n_outputs():
    model = _model().eval()
    model.reset_head(7)
    with torch.no_grad():
        out = model(torch.randn(2, N_CHANS, N_TIMES))
    assert out.shape == (2, 7)


def test_input_too_short_raises():
    # 100 samples yield no frames after boundary trimming with nperseg=400.
    with pytest.raises(ValueError):
        _model(n_times=100)


# ------------------------------------------------------- STFT front-end vs scipy
def test_stft_front_end_matches_scipy():
    """The in-model STFT reproduces the upstream scipy magnitude spectrogram."""
    scipy_signal = pytest.importorskip("scipy.signal")
    scipy_stats = pytest.importorskip("scipy.stats")

    nperseg, noverlap, cutoff, clip = 400, 350, 40, 5
    rng = np.random.RandomState(0)
    wav = rng.randn(6000).astype(np.float32)

    # upstream demo path (notebooks/demo.ipynb get_stft)
    _, _, zxx = scipy_signal.stft(
        wav, 2048, nperseg=nperseg, noverlap=noverlap, return_onesided=True
    )
    zxx = np.abs(zxx[:cutoff])[:, clip:-clip]
    ref = scipy_stats.zscore(zxx, axis=-1).T  # (n_frames, cutoff)

    stft = _STFTSpectrogram(sfreq=2048, nperseg=nperseg, noverlap=noverlap,
                            freq_cutoff=cutoff, clip=clip)
    ours = stft(torch.from_numpy(wav).view(1, 1, -1))[0, 0].numpy()

    assert ours.shape == ref.shape == (stft.n_frames(6000), cutoff)
    assert np.allclose(ref, ours, atol=1e-4)


# --------------------------------------------------------------- parity gate
def _import_upstream_or_skip():
    src = os.environ.get("BRAINBERT_SRC")
    if not src or not os.path.isdir(src):
        pytest.skip("set BRAINBERT_SRC=/path/to/BrainBERT to run the parity gate")
    sys.path.insert(0, src)
    try:
        import models as upstream_models  # noqa: F401 (populates registry)
    except ImportError as exc:  # pragma: no cover - depends on external code
        pytest.skip(f"upstream BrainBERT not importable: {exc}")
    return upstream_models.MODEL_REGISTRY["masked_tf_model"]


def test_encoder_is_bit_exact_with_upstream():
    """Input encoding + Transformer reproduce upstream once weights are copied."""
    MaskedTFModel = _import_upstream_or_skip()

    torch.manual_seed(0)
    batch, seq_len = 2, 6
    input_dim, hidden_dim, dim_ff, n_heads, n_layers = 8, 32, 48, 4, 2

    cfg = types.SimpleNamespace(
        name="masked_tf_model", input_dim=input_dim, hidden_dim=hidden_dim,
        layer_dim_feedforward=dim_ff, layer_activation="gelu",
        nhead=n_heads, encoder_num_layers=n_layers,
    )
    up = MaskedTFModel()
    up.build_model(cfg)
    up.eval()

    ours_embed = _BrainBERTInputEmbedding(input_dim, hidden_dim).eval()
    enc_layer = torch.nn.TransformerEncoderLayer(
        d_model=hidden_dim, nhead=n_heads, dim_feedforward=dim_ff,
        activation="gelu", batch_first=True,
    )
    ours_trans = torch.nn.TransformerEncoder(enc_layer, num_layers=n_layers).eval()

    # copy upstream weights into the ported modules
    ours_embed.in_proj.load_state_dict(up.input_encoding.in_proj.state_dict())
    ours_embed.layer_norm.load_state_dict(up.input_encoding.layer_norm.state_dict())
    ours_trans.load_state_dict(up.transformer.state_dict())
    # sinusoidal positional buffers are computed identically on both sides
    assert torch.allclose(
        ours_embed.positional_encoding.pe,
        up.input_encoding.positional_encoding.pe,
        atol=1e-6,
    )

    spec = torch.randn(batch, seq_len, input_dim)
    mask = torch.zeros(batch, seq_len).bool()
    with torch.no_grad():
        up_out = up(spec, mask, intermediate_rep=True)
        ours_out = ours_trans(ours_embed(spec))
    assert up_out.shape == ours_out.shape == (batch, seq_len, hidden_dim)
    assert torch.allclose(up_out, ours_out, atol=1e-5)
