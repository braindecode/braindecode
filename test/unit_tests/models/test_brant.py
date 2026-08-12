"""Tests for the Brant iEEG/sEEG foundation model.

The bulk of the model contract (init / forward / serialization / categorization)
is already exercised by the shared model suites via ``models_mandatory_parameters``.
This file adds Brant-specific checks plus an upstream **parity gate**: when the
gated reference code is available (``BRANT_SRC`` pointing to ``Brant_src``), the
two Transformer encoders are checked for numerical parity within tolerance;
otherwise the gate is skipped, so CI stays green without the gated dependency.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.signal import periodogram

from braindecode.models import Brant
from braindecode.models.brant import (
    BRANT_FREQ_BANDS,
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


def test_return_features():
    model = _model().eval()
    with torch.no_grad():
        out = model(torch.randn(2, N_CHANS, N_TIMES), return_features=True)
    assert isinstance(out, dict)
    assert set(out) == {"features", "cls_token"}
    assert out["features"].shape == (2, model.embed_dim)
    assert out["cls_token"] is None
    assert torch.isfinite(out["features"]).all()


def test_runtime_channel_count_is_parameter_agnostic():
    model = _model().eval()
    with torch.no_grad():
        out = model(torch.randn(2, N_CHANS + 3, N_TIMES))
    assert out.shape == (2, N_OUTPUTS)


def test_reset_head_changes_n_outputs():
    model = _model().double().eval()
    model.reset_head(7)
    assert model.n_outputs == model.get_config()["n_outputs"] == 7
    assert next(model.final_layer.parameters()).dtype == torch.float64
    restored = Brant.from_config(model.get_config())
    assert restored.n_outputs == 7
    with torch.no_grad():
        out = model(torch.randn(2, N_CHANS, N_TIMES, dtype=torch.float64))
    assert out.shape == (2, 7)


def test_local_checkpoint_resets_requested_head(tmp_path):
    pytest.importorskip("huggingface_hub")
    model = _model().eval()
    model._save_pretrained(tmp_path)

    restored = Brant.from_pretrained(tmp_path, n_outputs=7).eval()

    torch.testing.assert_close(
        restored.temporal_encoder.input_embedding.band_encoding,
        model.temporal_encoder.input_embedding.band_encoding,
    )
    assert restored.n_outputs == restored.get_config()["n_outputs"] == 7
    assert restored(torch.randn(2, N_CHANS, N_TIMES)).shape == (2, 7)


def test_input_shorter_than_patch_raises():
    with pytest.raises(ValueError):
        _model(n_times=PATCH - 1)


@pytest.mark.parametrize("n_times", [N_TIMES - 1, N_TIMES + 1])
def test_runtime_input_length_must_match(n_times):
    model = _model()
    with pytest.raises(
        ValueError, match=f"configured for {N_TIMES}.*received {n_times}"
    ):
        model(torch.randn(2, N_CHANS, n_times))


def test_freq_bands_must_match_constant():
    with pytest.raises(ValueError):
        _model(n_freq_bands=len(BRANT_FREQ_BANDS) - 1)


def test_band_power_shape_and_finite():
    features = _BandPowerFeatures(SFREQ, BRANT_FREQ_BANDS)
    out = features(torch.randn(2, N_CHANS, 4, PATCH))
    assert out.shape == (2, N_CHANS, 4, len(BRANT_FREQ_BANDS))
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "dtype, rtol, atol",
    [
        (torch.float32, 1e-6, 1e-7),
        (torch.float64, 1e-12, 1e-12),
    ],
)
def test_band_power_matches_scipy_periodogram(dtype, rtol, atol):
    patches = torch.randn(2, 3, 4, PATCH, dtype=dtype)
    actual = _BandPowerFeatures(SFREQ, BRANT_FREQ_BANDS)(patches)

    freqs, psd = periodogram(patches.numpy(), fs=SFREQ, axis=-1)
    expected = np.stack(
        [
            np.log10(psd[..., (freqs > low) & (freqs <= high)].sum(axis=-1) + 1)
            for low, high in BRANT_FREQ_BANDS
        ],
        axis=-1,
    )

    torch.testing.assert_close(
        actual, torch.from_numpy(expected), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_band_power_dtype_device_and_gradient(dtype):
    patches = torch.randn(
        2, 3, 4, PATCH, dtype=dtype, requires_grad=True
    )

    output = _BandPowerFeatures(SFREQ, BRANT_FREQ_BANDS)(patches)
    output.float().sum().backward()

    assert output.dtype == dtype
    assert output.device == patches.device
    assert torch.isfinite(output).all()
    assert patches.grad is not None
    assert torch.isfinite(patches.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reduced_precision_forward_backward(dtype):
    model = _model().to(dtype=dtype).train()
    x = torch.randn(2, N_CHANS, N_TIMES, dtype=dtype, requires_grad=True)

    output = model(x)
    output.float().square().mean().backward()

    assert output.dtype == dtype
    assert torch.isfinite(output).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_cpu_autocast_forward_backward():
    model = _model().train()
    x = torch.randn(2, N_CHANS, N_TIMES, requires_grad=True)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(x)
        loss = output.float().square().mean()
    loss.backward()

    assert output.dtype == torch.bfloat16
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("loader", ["test-gate", "developer-script"])
def test_upstream_import_is_isolated(tmp_path, monkeypatch, loader):
    package = tmp_path / "pretrain"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "pre_model.py").write_text(
        "class TimeEncoder: pass\nclass ChannelEncoder: pass\n"
    )
    monkeypatch.setenv("BRANT_SRC", str(tmp_path))

    old_path = sys.path.copy()
    old_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "pretrain" or name.startswith("pretrain.")
    }
    for name in old_modules:
        sys.modules.pop(name)

    try:
        if loader == "test-gate":
            time_encoder, channel_encoder = _import_upstream_or_skip()
        else:
            from scripts import brant_parity_check

            time_encoder, channel_encoder = brant_parity_check._import_upstream(
                Path(tmp_path)
            )
        assert time_encoder.__name__ == "TimeEncoder"
        assert channel_encoder.__name__ == "ChannelEncoder"
        assert sys.path == old_path
        assert {
            name: module
            for name, module in sys.modules.items()
            if name == "pretrain" or name.startswith("pretrain.")
        } == old_modules
    finally:
        sys.path[:] = old_path
        for name in list(sys.modules):
            if name == "pretrain" or name.startswith("pretrain."):
                sys.modules.pop(name)
        sys.modules.update(old_modules)


# --------------------------------------------------------------- parity gate
def _import_upstream_or_skip():
    src = os.environ.get("BRANT_SRC")
    if not src or not os.path.isdir(src):
        pytest.skip("set BRANT_SRC=/path/to/Brant_src to run the parity gate")
    old_path = sys.path.copy()
    old_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "pretrain" or name.startswith("pretrain.")
    }
    for name in old_modules:
        sys.modules.pop(name)
    sys.path.insert(0, src)
    try:
        from pretrain.pre_model import ChannelEncoder, TimeEncoder
    except ImportError as exc:  # pragma: no cover - depends on external code
        pytest.skip(f"upstream Brant not importable: {exc}")
    finally:
        sys.path[:] = old_path
        for name in list(sys.modules):
            if name == "pretrain" or name.startswith("pretrain."):
                sys.modules.pop(name)
        sys.modules.update(old_modules)
    return TimeEncoder, ChannelEncoder


def test_encoders_numerically_match_upstream():
    """Both encoders numerically match upstream once weights are copied over."""
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
    torch.testing.assert_close(up_time, ours_time, rtol=0, atol=1e-5)

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
    torch.testing.assert_close(up_ch, ours_ch, rtol=0, atol=1e-5)
