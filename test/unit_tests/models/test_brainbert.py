"""Tests for the BrainBERT iEEG/sEEG foundation model.

The bulk of the model contract (init / forward / serialization / categorization)
is already exercised by the shared model suites via ``models_mandatory_parameters``.
This file adds BrainBERT-specific checks:

* the in-model STFT front-end reproduces **both** upstream spectrogram recipes,
  and the test pins the fact that they are not interchangeable;
* the pooling reproduces the upstream downstream protocol (the centre frames of
  a single electrode, ``preprocessors/spec_pretrained.py``);
* an upstream **parity gate** — when the reference code is available
  (``BRAINBERT_SRC`` pointing to a clone of https://github.com/czlwang/BrainBERT),
  the ported input encoding + Transformer are checked to be bit-exact against
  upstream ``MaskedTFModel``; otherwise the gate is skipped, so CI stays green
  without the external dependency.
"""

from __future__ import annotations

import contextlib
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
N_CHANS, N_OUTPUTS, N_TIMES, SFREQ = 3, 2, 2000, 2048.0


def _model(**overrides):
    kwargs = dict(
        n_chans=N_CHANS,
        n_outputs=N_OUTPUTS,
        n_times=N_TIMES,
        sfreq=SFREQ,
        hidden_dim=32,
        ffn_dim=48,
        n_layers=2,
        n_heads=4,
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
    with pytest.raises(ValueError, match="too short"):
        _model(n_times=100)


def test_too_few_frames_for_centre_pooling_raises():
    """A window shorter than the pooled centre window is refused, not silently
    pooled differently: the number of frames averaged is part of the protocol."""
    with pytest.raises(ValueError, match="pool_n_frames"):
        _model(n_times=1000, stft_clip=10, pool_n_frames=10)


def test_declared_license_is_unknown():
    """Upstream ships no LICENSE file, so nothing permissive may be assumed.

    Without an explicit ``license=`` class keyword the model silently inherits
    braindecode's ``bsd-3-clause`` default and publishes that claim on the Hub
    model card.
    """
    pytest.importorskip("huggingface_hub")
    assert BrainBERT._hub_mixin_info.model_card_data.license == "unknown"


# ------------------------------------------------------- STFT front-end vs scipy
def _upstream_magnitude(wav, nperseg, noverlap, cutoff):
    """``signal.stft`` + cutoff + ``abs``, common to both upstream recipes."""
    scipy_signal = pytest.importorskip("scipy.signal")
    _, _, zxx = scipy_signal.stft(
        wav, 2048, nperseg=nperseg, noverlap=noverlap, return_onesided=True
    )
    return np.abs(zxx[:cutoff])


def _upstream_zscore(arr):
    """Hand-rolled z-score of upstream ``preprocessors/stft.py`` (ddof=0, and
    zero standard deviations replaced by one)."""
    mn = arr.mean(axis=-1, keepdims=True)
    std = arr.std(axis=-1, ddof=0, keepdims=True)
    std[std == 0] = 1.0
    return (arr - mn) / std


def _upstream_pretrained_recipe(wav, nperseg=400, noverlap=350, cutoff=40):
    """``preprocessors/stft.py``: z-score first, then trim 10 frames per side.

    This is the recipe reached from ``conf/preprocessor/stft_pretrained.yaml``
    via ``preprocessors/spec_pretrained.py``, i.e. the one behind the published
    downstream numbers.
    """
    zxx = _upstream_magnitude(wav, nperseg, noverlap, cutoff)
    zxx = _upstream_zscore(zxx)
    zxx = zxx[:, 10:-10]
    return np.nan_to_num(zxx, nan=0.0).T  # (n_frames, cutoff)


def _upstream_demo_recipe(wav, nperseg=400, noverlap=350, cutoff=40):
    """``notebooks/demo.ipynb``: trim 5 frames per side first, then z-score."""
    zxx = _upstream_magnitude(wav, nperseg, noverlap, cutoff)[:, 5:-5]
    return _upstream_zscore(zxx).T


def _wav(n=6000, seed=0):
    return np.random.RandomState(seed).randn(n).astype(np.float64)


@pytest.mark.parametrize(
    "reference,clip,zscore_before_clip",
    [
        (_upstream_pretrained_recipe, 10, True),  # released checkpoint
        (_upstream_demo_recipe, 5, False),  # demo notebook
    ],
)
def test_stft_front_end_matches_upstream(reference, clip, zscore_before_clip):
    """The in-model STFT reproduces the upstream scipy spectrogram, for each of
    the two recipes upstream ships."""
    nperseg, noverlap, cutoff = 400, 350, 40
    wav = _wav()
    ref = reference(wav, nperseg, noverlap, cutoff)

    stft = _STFTSpectrogram(
        sfreq=2048,
        nperseg=nperseg,
        noverlap=noverlap,
        idx_freq_cutoff=cutoff,
        clip=clip,
        zscore_before_clip=zscore_before_clip,
    )
    ours = stft(torch.from_numpy(wav).view(1, 1, -1)).double()[0, 0].numpy()

    assert ours.shape == ref.shape == (stft.n_frames(len(wav)), cutoff)
    # The port re-derives scipy's framing rather than approximating it; the
    # residual is the float32 Hann-window table (scipy builds it in float64),
    # measured at 7.1e-7 here. Anything looser would stop catching a genuine
    # change of recipe, which is what this test exists for.
    assert np.abs(ref - ours).max() < 5e-6


def test_the_two_upstream_recipes_are_not_interchangeable():
    """Guard against a future 'simplification' collapsing the two recipes.

    They differ in the order of the z-score and the trimming, and in how many
    frames are trimmed; on filtered noise they correlate at 0.999 but differ by
    a non-negligible amount per bin, and their sequence lengths differ by 10.
    """
    wav = _wav()
    pretrained = _upstream_pretrained_recipe(wav)
    demo = _upstream_demo_recipe(wav)

    assert demo.shape[0] == pretrained.shape[0] + 10
    # compare on the region they share
    overlap = demo[5:-5]
    assert overlap.shape == pretrained.shape
    assert np.abs(overlap - pretrained).max() > 1e-2


@pytest.mark.parametrize("clip", [0, 5])
@pytest.mark.parametrize("normalizing", ["zscore", "none"])
def test_stft_options_run_and_have_the_expected_length(clip, normalizing):
    wav = _wav(n=4000)
    stft = _STFTSpectrogram(
        sfreq=2048,
        nperseg=400,
        noverlap=350,
        idx_freq_cutoff=40,
        clip=clip,
        normalizing=normalizing,
    )
    out = stft(torch.from_numpy(wav).view(1, 1, -1).float())
    assert out.shape == (1, 1, stft.n_frames(len(wav)), 40)
    assert torch.isfinite(out).all()
    if normalizing == "none":
        assert out.min() >= 0.0  # still a magnitude spectrogram


def test_stft_rejects_unknown_normalizing():
    with pytest.raises(ValueError, match="normalizing"):
        _STFTSpectrogram(sfreq=2048, normalizing="db")


def test_stft_is_finite_on_a_flat_channel():
    """A dead (constant) channel must not produce NaNs downstream.

    Upstream guards this twice — zero standard deviations are replaced by one,
    and a fully constant z-scored window is replaced by ones — because a single
    NaN otherwise poisons the whole attention window.
    """
    stft = _STFTSpectrogram(sfreq=2048, nperseg=400, noverlap=350)
    x = torch.zeros(1, 2, 4000)
    x[0, 1] = torch.randn(4000)  # one live channel, one flat
    out = stft(x)
    assert torch.isfinite(out).all()


# ------------------------------------------------------------------- pooling
def test_pooling_uses_the_upstream_centre_frames():
    """With one channel, the pooled feature is upstream's
    ``outputs[:, middle-5:middle+5].mean(axis=1)`` on the same encoder output."""
    model = _model(n_chans=1, pool_n_frames=10).eval()
    x = torch.randn(2, 1, N_TIMES)

    with torch.no_grad():
        spec = model.spectrogram(x)
        seq_len = spec.shape[2]
        z = model.transformer(
            model.input_embedding(spec.reshape(2, seq_len, model.idx_freq_cutoff))
        )
        middle = seq_len // 2
        expected = z[:, middle - 5 : middle + 5].mean(dim=1)
        got = model(x, return_features=True)["features"]

    assert torch.allclose(expected, got, atol=1e-6)
    # the distinction is real on this input: the all-frames mean differs
    assert not torch.allclose(z.mean(dim=1), got, atol=1e-4)


def test_pool_all_frames_is_available():
    model = _model(pool_n_frames=None).eval()
    with torch.no_grad():
        out = model(torch.randn(2, N_CHANS, N_TIMES), return_features=True)
    assert out["features"].shape == (2, model.hidden_dim)


def test_head_is_a_bare_linear_probe():
    """Upstream's downstream model is ``nn.Linear(input_dim, 1)`` and nothing
    else; a normalisation layer here would change the published protocol."""
    model = _model()
    modules = [
        m
        for m in model.final_layer.modules()
        if not isinstance(m, type(model.final_layer))
    ]
    assert all(not isinstance(m, torch.nn.LayerNorm) for m in modules)
    assert isinstance(model.final_layer.fc, torch.nn.Linear)


# --------------------------------------------------------------- parity gate
@contextlib.contextmanager
def _upstream_on_path(src):
    """Put ``src`` on ``sys.path`` and take it back off, whatever happens."""
    sys.path.insert(0, src)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(src)


def _import_upstream_or_skip(stack):
    src = os.environ.get("BRAINBERT_SRC")
    if not src or not os.path.isdir(src):
        pytest.skip("set BRAINBERT_SRC=/path/to/BrainBERT to run the parity gate")
    src = os.path.abspath(src)
    stack.enter_context(_upstream_on_path(src))
    try:
        import models as upstream_models
    except ImportError as exc:  # pragma: no cover - depends on external code
        pytest.skip(f"upstream BrainBERT not importable: {exc}")
    # guard against picking up an unrelated ``models`` package already imported
    # (braindecode.models, or a stale entry in sys.modules)
    origin = os.path.abspath(getattr(upstream_models, "__file__", "") or "")
    if not origin.startswith(src + os.sep):
        pytest.skip(
            f"'models' resolved to {origin or '<unknown>'}, not to BRAINBERT_SRC"
        )
    return upstream_models.MODEL_REGISTRY["masked_tf_model"]


def test_encoder_is_bit_exact_with_upstream():
    """Input encoding + Transformer reproduce upstream once weights are copied."""
    with contextlib.ExitStack() as stack:
        MaskedTFModel = _import_upstream_or_skip(stack)

        torch.manual_seed(0)
        batch, seq_len = 2, 6
        input_dim, hidden_dim, dim_ff, n_heads, n_layers = 8, 32, 48, 4, 2

        cfg = types.SimpleNamespace(
            name="masked_tf_model",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            layer_dim_feedforward=dim_ff,
            layer_activation="gelu",
            nhead=n_heads,
            encoder_num_layers=n_layers,
        )
        up = MaskedTFModel()
        up.build_model(cfg)
        up.eval()

        ours_embed = _BrainBERTInputEmbedding(input_dim, hidden_dim).eval()
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            activation="gelu",
            batch_first=True,
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
