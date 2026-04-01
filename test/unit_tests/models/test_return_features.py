"""Tests for the unified ``return_features`` API on foundation models."""

import mne
import pytest
import torch

from braindecode.models import (
    BENDR,
    BIOT,
    EEGPT,
    REVE,
    CBraMod,
    Labram,
    SignalJEPA,
    SignalJEPA_Contextual,
    SignalJEPA_PostLocal,
    SignalJEPA_PreLocal,
)

N_CHANS = 22
N_TIMES = 1000
N_OUTPUTS = 4
BATCH = 2

_REVE_CH_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8",
]
N_REVE_CHANS = len(_REVE_CH_NAMES)


def _make_chs_info(n_chans=N_CHANS):
    """Create minimal chs_info with 3D locations."""
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_names = montage.ch_names[:n_chans]
    info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types="eeg")
    info.set_montage(montage)
    return info["chs"]


def _make_reve_chs_info():
    """Create chs_info with channels recognized by REVE's position bank."""
    info = mne.create_info(ch_names=_REVE_CH_NAMES, sfreq=256, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"))
    return info["chs"]


# (model_cls, n_chans, extra_kwargs, has_cls_token)
_MODELS = [
    pytest.param(EEGPT, N_CHANS, {}, False, id="EEGPT"),
    pytest.param(
        Labram, N_CHANS, {"patch_size": 200, "chs_info": _make_chs_info()}, True,
        id="Labram",
    ),
    pytest.param(
        REVE, N_REVE_CHANS,
        {"chs_info": _make_reve_chs_info(), "patch_size": 200, "patch_overlap": 0},
        False, id="REVE",
    ),
    pytest.param(BENDR, N_CHANS, {}, False, id="BENDR"),
    pytest.param(BIOT, N_CHANS, {}, False, id="BIOT"),
    pytest.param(CBraMod, N_CHANS, {}, False, id="CBraMod"),
    pytest.param(
        SignalJEPA, N_CHANS, {"chs_info": _make_chs_info()}, False,
        id="SignalJEPA",
    ),
    pytest.param(
        SignalJEPA_Contextual, N_CHANS, {"chs_info": _make_chs_info()}, False,
        id="SignalJEPA_Contextual",
    ),
    pytest.param(
        SignalJEPA_PostLocal, N_CHANS, {"chs_info": _make_chs_info()}, False,
        id="SignalJEPA_PostLocal",
    ),
    pytest.param(
        SignalJEPA_PreLocal, N_CHANS, {"chs_info": _make_chs_info()}, False,
        id="SignalJEPA_PreLocal",
    ),
]


@pytest.mark.parametrize("model_cls, n_chans, extra_kwargs, has_cls", _MODELS)
def test_return_features_is_dict(model_cls, n_chans, extra_kwargs, has_cls):
    model = model_cls(n_chans=n_chans, n_times=N_TIMES, n_outputs=N_OUTPUTS, **extra_kwargs)
    model.eval()
    x = torch.randn(BATCH, n_chans, N_TIMES)
    with torch.no_grad():
        out = model(x, return_features=True)
    assert isinstance(out, dict)
    assert "features" in out
    assert "cls_token" in out
    assert isinstance(out["features"], torch.Tensor)
    assert out["features"].shape[0] == BATCH


@pytest.mark.parametrize("model_cls, n_chans, extra_kwargs, has_cls", _MODELS)
def test_cls_token(model_cls, n_chans, extra_kwargs, has_cls):
    model = model_cls(n_chans=n_chans, n_times=N_TIMES, n_outputs=N_OUTPUTS, **extra_kwargs)
    model.eval()
    x = torch.randn(BATCH, n_chans, N_TIMES)
    with torch.no_grad():
        out = model(x, return_features=True)
    if has_cls:
        assert isinstance(out["cls_token"], torch.Tensor)
        assert out["cls_token"].shape[0] == BATCH
    else:
        assert out["cls_token"] is None


@pytest.mark.parametrize("model_cls, n_chans, extra_kwargs, has_cls", _MODELS)
def test_default_forward_returns_tensor(model_cls, n_chans, extra_kwargs, has_cls):
    """Default forward (return_features=False) still returns a plain Tensor."""
    model = model_cls(n_chans=n_chans, n_times=N_TIMES, n_outputs=N_OUTPUTS, **extra_kwargs)
    model.eval()
    x = torch.randn(BATCH, n_chans, N_TIMES)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, torch.Tensor)


# ---------- Legacy backward-compat tests ----------


def test_eegpt_legacy_return_encoder_output():
    model = EEGPT(n_chans=N_CHANS, n_times=N_TIMES, n_outputs=N_OUTPUTS,
                   return_encoder_output=True)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(BATCH, N_CHANS, N_TIMES))
    assert isinstance(out, torch.Tensor)
    assert out.ndim == 4


def test_biot_legacy_return_feature():
    model = BIOT(n_chans=N_CHANS, n_times=N_TIMES, n_outputs=N_OUTPUTS,
                  return_feature=True)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(BATCH, N_CHANS, N_TIMES))
    assert isinstance(out, tuple)
    assert len(out) == 2


def test_labram_legacy_return_all_tokens():
    model = Labram(n_chans=N_CHANS, n_times=N_TIMES, n_outputs=N_OUTPUTS,
                    patch_size=200, chs_info=_make_chs_info())
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(BATCH, N_CHANS, N_TIMES), return_all_tokens=True)
    assert isinstance(out, torch.Tensor)
