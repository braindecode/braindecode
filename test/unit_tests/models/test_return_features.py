"""Tests for the unified ``return_features`` and ``reset_head`` API."""

import inspect

import mne
import pytest
import torch
from torch import nn

from braindecode.models import (
    BENDR,
    BIOT,
    EEGPT,
    Labram,
)
from braindecode.models.base import HAS_HF_HUB, EEGModuleMixin
from braindecode.models.labram import _LABRAM_TARGET_CHS_INFO
from braindecode.models.util import (
    models_dict,
    models_mandatory_parameters,
)
from test.unit_tests.models._fixtures import (
    _TOKENIZERS,
    PARAMS,
    _architecture_kwargs,
    _build_case,
    _get_signal_params,
    _materialize,
    _tensor_leaves,
)

N_CHANS, N_TIMES, N_OUTPUTS, BATCH = 22, 1000, 4, 2

_HAS_CLS_TOKEN = {
    "CORTEG",
    "EEGDINO",
    "EpiNT",
    "FNIRSPreT",
    "FNIRST",
    "Labram",
    "PopulationTransformer",
    "STEEGFormer",
}


def _chs(names=None, n=N_CHANS):
    montage = mne.channels.make_standard_montage("colin27_1020")
    ch = names or montage.ch_names[:n]
    info = mne.create_info(ch_names=ch, sfreq=256, ch_types="eeg")
    info.set_montage(montage)
    return info["chs"]


@PARAMS
def test_return_features_on_every_model(model_name, required_params, signal_params):
    if model_name in _TOKENIZERS:
        pytest.skip("tokenizer: use tokenize()")
    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    with torch.no_grad():
        default_out = model(x)
        feats = model(x, return_features=True)
    assert not isinstance(default_out, dict)
    assert list(_tensor_leaves(default_out))
    assert isinstance(feats, dict)
    assert {"features", "embedding", "cls_token"} <= set(feats)
    assert feats["features"].shape[0] == x.shape[0]
    assert feats["embedding"].ndim == 2 and feats["embedding"].shape[0] == x.shape[0]
    assert torch.isfinite(feats["embedding"]).all()
    if model_name in _HAS_CLS_TOKEN:
        assert isinstance(feats["cls_token"], torch.Tensor)
        assert feats["cls_token"].shape[0] == x.shape[0]
    else:
        assert feats["cls_token"] is None


@PARAMS
def test_embedding_dim_invariant_across_lengths_and_channels(
    model_name, required_params, signal_params
):
    if model_name in _TOKENIZERS:
        pytest.skip("tokenizer")
    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    C, T = x.shape[1:]
    with torch.no_grad():
        emb = model(x, return_features=True)["embedding"]
        if model._time_axis in {"free", "dense"}:
            emb_2t = model(torch.randn(2, C, 2 * T), return_features=True)["embedding"]
            assert emb_2t.shape == emb.shape
        if (
            model._channel_axis == "free"
            and "ch_mask" in inspect.signature(type(model).forward).parameters
        ):
            emb_c = model(torch.randn(2, C + 1, T), return_features=True)["embedding"]
            assert emb_c.shape == emb.shape


def test_return_features_under_channel_strategy():
    kw = _architecture_kwargs("BENDR")
    model = BENDR(
        n_chans=N_CHANS,
        n_times=N_TIMES,
        n_outputs=N_OUTPUTS,
        chs_info=_chs(),
        channel_strategy="spline",
        **kw,
    ).eval()
    with torch.no_grad():
        out = model(torch.randn(BATCH, N_CHANS, N_TIMES), return_features=True)
    assert {"features", "embedding", "cls_token"} <= set(out)
    assert out["embedding"].ndim == 2 and out["embedding"].shape[0] == BATCH


def test_feature_fallback_refuses_repeated_heads():
    class _RepeatedHeadModel(EEGModuleMixin, nn.Module):
        def __init__(
            self,
            n_outputs=2,
            n_chans=2,
            chs_info=None,
            n_times=10,
            input_window_seconds=None,
            sfreq=None,
        ):
            super().__init__(
                n_outputs=n_outputs,
                n_chans=n_chans,
                chs_info=chs_info,
                n_times=n_times,
                input_window_seconds=input_window_seconds,
                sfreq=sfreq,
            )
            self.final_layer = nn.Linear(n_times, n_outputs)

        def forward(self, x):
            return self.final_layer(x[:, 0]) + self.final_layer(x[:, 1])

    class _SkippedHeadModel(EEGModuleMixin, nn.Module):
        def __init__(
            self,
            n_outputs=2,
            n_chans=2,
            chs_info=None,
            n_times=10,
            input_window_seconds=None,
            sfreq=None,
        ):
            super().__init__(
                n_outputs=n_outputs,
                n_chans=n_chans,
                chs_info=chs_info,
                n_times=n_times,
                input_window_seconds=input_window_seconds,
                sfreq=sfreq,
            )
            self.final_layer = nn.Linear(n_times, n_outputs)

        def forward(self, x):
            return x.mean(dim=-1)

    class _TimeFirstHeadModel(EEGModuleMixin, nn.Module):
        def __init__(
            self,
            n_outputs=2,
            n_chans=2,
            chs_info=None,
            n_times=10,
            input_window_seconds=None,
            sfreq=None,
        ):
            super().__init__(
                n_outputs=n_outputs,
                n_chans=n_chans,
                chs_info=chs_info,
                n_times=n_times,
                input_window_seconds=input_window_seconds,
                sfreq=sfreq,
            )
            self.final_layer = nn.Linear(n_chans, n_outputs)

        def forward(self, x):
            # (time, batch, chans) into final_layer
            return self.final_layer(x.permute(2, 0, 1)).mean(dim=0)

    x = torch.randn(2, 2, 10)
    with pytest.raises(ValueError, match="final_layer ran 2 times"):
        _RepeatedHeadModel().eval()(x, return_features=True)
    with pytest.raises(ValueError, match="final_layer ran 0 times"):
        _SkippedHeadModel().eval()(x, return_features=True)
    with pytest.raises(ValueError, match="does not read one batched tensor"):
        _TimeFirstHeadModel().eval()(x, return_features=True)


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


# -- reset_head must leave a model that can be saved and loaded back --

_ALL_MODELS = dict(models_dict)
_REGISTRY = {name: sp for name, _, sp in models_mandatory_parameters}
# Every registered model that overrides reset_head.
_RESET_HEAD_CASES = [
    name
    for name in _REGISTRY
    if _ALL_MODELS[name].reset_head is not EEGModuleMixin.reset_head
]


def _build_registered(name, **kwargs):
    """Build a model from its full registry test parameters, in eval mode.

    All signal parameters are passed so that no layer is built lazily: the
    Hugging Face loader cannot load weights into uninitialized parameters.
    """
    signal_params = _get_signal_params(_REGISTRY[name])
    return _ALL_MODELS[name](**{**signal_params, **kwargs}).eval()


@pytest.mark.parametrize("name", _RESET_HEAD_CASES)
def test_reset_head_updates_config(name):
    model = _build_registered(name)
    n_outputs = model.n_outputs + 3
    model.reset_head(n_outputs)
    assert model.n_outputs == n_outputs
    assert model.get_config()["n_outputs"] == n_outputs


@pytest.mark.skipif(not HAS_HF_HUB, reason="huggingface_hub is not installed")
@pytest.mark.parametrize("name", _RESET_HEAD_CASES)
def test_reset_head_model_reloads_after_saving(name, tmp_path):
    model = _build_registered(name)
    model.reset_head(model.n_outputs + 3)
    model.save_pretrained(tmp_path)

    reloaded = _ALL_MODELS[name].from_pretrained(tmp_path)
    torch.testing.assert_close(reloaded.state_dict(), model.state_dict())


@pytest.mark.skipif(not HAS_HF_HUB, reason="huggingface_hub is not installed")
@pytest.mark.parametrize("name", ["EEGDINO", "EEGPT", "LUNA"])  # heads with dropout
def test_from_pretrained_new_n_outputs_is_deterministic_in_eval(name, tmp_path):
    saved = _build_registered(name)
    saved.save_pretrained(tmp_path)
    n_outputs = saved.n_outputs + 3

    model = _ALL_MODELS[name].from_pretrained(tmp_path, n_outputs=n_outputs).eval()
    assert model.get_config()["n_outputs"] == n_outputs
    assert not any(module.training for module in model.modules())
    signal_params = _get_signal_params(_REGISTRY[name])
    x = torch.randn(BATCH, signal_params["n_chans"], signal_params["n_times"])
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


@pytest.mark.skipif(not HAS_HF_HUB, reason="huggingface_hub is not installed")
@pytest.mark.parametrize(
    "name, kwargs",
    [
        pytest.param("BENDR", {"final_layer": False}, id="BENDR"),
        pytest.param("CBraMod", {"return_encoder_output": True}, id="CBraMod"),
        pytest.param("EEGDINO", {"return_encoder_output": True}, id="EEGDINO"),
        pytest.param("LUNA", {"n_outputs": 0}, id="LUNA"),
    ],
)
def test_reset_head_on_feature_extractor_reloads_as_classifier(name, kwargs, tmp_path):
    model = _build_registered(name, **kwargs)
    model.reset_head(N_OUTPUTS)
    model.save_pretrained(tmp_path)

    reloaded = _ALL_MODELS[name].from_pretrained(tmp_path)
    torch.testing.assert_close(reloaded.state_dict(), model.state_dict())
    signal_params = _get_signal_params(_REGISTRY[name])
    x = torch.randn(BATCH, signal_params["n_chans"], signal_params["n_times"])
    with torch.no_grad():
        assert reloaded(x).shape == (BATCH, N_OUTPUTS)


@pytest.mark.skipif(not HAS_HF_HUB, reason="huggingface_hub is not installed")
@pytest.mark.parametrize("via_pretrained", [False, True])
def test_labram_reset_to_zero_reloads_as_identity(tmp_path, via_pretrained):
    model = Labram(n_chans=2, n_times=200, n_outputs=2, num_layers=1, num_heads=10)
    if via_pretrained:
        model.save_pretrained(tmp_path / "positive")
        model = Labram.from_pretrained(tmp_path / "positive", n_outputs=0)
    else:
        model.reset_head(0)

    assert isinstance(model.final_layer, torch.nn.Identity)
    assert model.n_outputs == model.get_config()["n_outputs"] == 0
    from_config = Labram.from_config(model.get_config())
    assert isinstance(from_config.final_layer, torch.nn.Identity)
    model.save_pretrained(tmp_path / "encoder")
    reloaded = Labram.from_pretrained(tmp_path / "encoder")
    assert reloaded.n_outputs == 0
    assert isinstance(reloaded.final_layer, torch.nn.Identity)
    torch.testing.assert_close(reloaded.state_dict(), model.state_dict())
