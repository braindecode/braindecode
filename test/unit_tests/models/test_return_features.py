"""Tests for the unified ``return_features`` and ``reset_head`` API."""

import mne
import pytest
import torch

from braindecode.models import (
    BENDR,
    BIOT,
    EEGPT,
    REVE,
    CBraMod,
    InterpolatedLaBraM,
    Labram,
    SignalJEPA,
    SignalJEPA_Contextual,
    SignalJEPA_PostLocal,
    SignalJEPA_PreLocal,
)
from braindecode.models.labram import _LABRAM_TARGET_CHS_INFO

N_CHANS, N_TIMES, N_OUTPUTS, BATCH = 22, 1000, 4, 2

_REVE_CHS = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T7",
    "T8",
    "P7",
    "P8",
]


def _chs(names=None, n=N_CHANS):
    montage = mne.channels.make_standard_montage("standard_1020")
    ch = names or montage.ch_names[:n]
    info = mne.create_info(ch_names=ch, sfreq=256, ch_types="eeg")
    info.set_montage(montage)
    return info["chs"]


# (cls, n_chans, kwargs, has_cls)
_MODELS = [
    pytest.param(EEGPT, N_CHANS, {}, False, id="EEGPT"),
    pytest.param(
        InterpolatedLaBraM,
        N_CHANS,
        {"patch_size": 200, "chs_info": _chs()},
        True,
        id="InterpolatedLaBraM",
    ),
    pytest.param(
        REVE,
        16,
        {"chs_info": _chs(_REVE_CHS), "patch_size": 200, "patch_overlap": 0},
        False,
        id="REVE",
    ),
    pytest.param(BENDR, N_CHANS, {}, False, id="BENDR"),
    pytest.param(BIOT, N_CHANS, {}, False, id="BIOT"),
    pytest.param(CBraMod, N_CHANS, {}, False, id="CBraMod"),
    pytest.param(SignalJEPA, N_CHANS, {"chs_info": _chs()}, False, id="SignalJEPA"),
    pytest.param(
        SignalJEPA_Contextual,
        N_CHANS,
        {"chs_info": _chs()},
        False,
        id="SignalJEPA_Contextual",
    ),
    pytest.param(
        SignalJEPA_PostLocal,
        N_CHANS,
        {"chs_info": _chs()},
        False,
        id="SignalJEPA_PostLocal",
    ),
    pytest.param(
        SignalJEPA_PreLocal,
        N_CHANS,
        {"chs_info": _chs()},
        False,
        id="SignalJEPA_PreLocal",
    ),
]


@pytest.mark.parametrize("cls, nc, kw, has_cls", _MODELS)
def test_return_features(cls, nc, kw, has_cls):
    model = cls(n_chans=nc, n_times=N_TIMES, n_outputs=N_OUTPUTS, **kw)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(BATCH, nc, N_TIMES), return_features=True)
    assert isinstance(out, dict) and "features" in out and "cls_token" in out
    assert out["features"].shape[0] == BATCH
    if has_cls:
        assert isinstance(out["cls_token"], torch.Tensor)
    else:
        assert out["cls_token"] is None


@pytest.mark.parametrize("cls, nc, kw, has_cls", _MODELS)
def test_default_forward_returns_tensor(cls, nc, kw, has_cls):
    model = cls(n_chans=nc, n_times=N_TIMES, n_outputs=N_OUTPUTS, **kw)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(BATCH, nc, N_TIMES))
    assert isinstance(out, torch.Tensor)


# -- Legacy backward compat --

_LEGACY = [
    pytest.param(
        EEGPT,
        {"return_encoder_output": True},
        {},
        lambda out: isinstance(out, torch.Tensor) and out.ndim == 4,
        id="EEGPT-encoder_output",
    ),
    pytest.param(
        BIOT,
        {"return_feature": True},
        {},
        lambda out: isinstance(out, tuple) and len(out) == 2,
        id="BIOT-return_feature",
    ),
    pytest.param(
        Labram,
        {"patch_size": 200, "chs_info": _LABRAM_TARGET_CHS_INFO},
        {"return_all_tokens": True},
        lambda out: isinstance(out, torch.Tensor),
        id="Labram-all_tokens",
    ),
]


@pytest.mark.parametrize("cls, init_kw, fwd_kw, check", _LEGACY)
def test_legacy_params(cls, init_kw, fwd_kw, check):
    # When init_kw supplies chs_info, derive n_chans from it (some backbones
    # like refactored Labram require chs_info to match a canonical order and
    # will reject a user-provided n_chans that does not equal len(chs_info)).
    nc = len(init_kw["chs_info"]) if "chs_info" in init_kw else N_CHANS
    extra = {} if "chs_info" in init_kw else {"n_chans": nc}
    model = cls(n_times=N_TIMES, n_outputs=N_OUTPUTS, **extra, **init_kw)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(BATCH, nc, N_TIMES), **fwd_kw)
    assert check(out)


# -- reset_head --

# Exclude SignalJEPA (SSL model, Identity head)
_RESET_MODELS = [p for p in _MODELS if p.id != "SignalJEPA"]


@pytest.mark.parametrize("cls, nc, kw, has_cls", _RESET_MODELS)
def test_reset_head(cls, nc, kw, has_cls):
    model = cls(n_chans=nc, n_times=N_TIMES, n_outputs=N_OUTPUTS, **kw)
    model.reset_head(10)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(BATCH, nc, N_TIMES))
    assert model.n_outputs == 10
    assert out.shape[-1] == 10
