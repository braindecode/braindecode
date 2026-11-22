"""Tests for the unified ``return_features`` and ``reset_head`` API."""

import inspect

import mne
import pytest
import torch
from torch import nn

from braindecode.models import (
    AXON,
    BBL,
    BENDR,
    BIOT,
    CORTEG,
    EEGDINO,
    EEGPT,
    FNIRS4D,
    FNIRSCNN1D,
    FNIRST,
    LUNA,
    MAPA,
    MEGGPT,
    MEGXL,
    NICE,
    REVE,
    SAMBA,
    BaRISTA,
    BrainBERT,
    Brant,
    CBraMod,
    DuIN,
    EpiNT,
    FNIRSNet,
    FNIRSPreT,
    GeoNIRS,
    Guetschel2026,
    JungMAE,
    Labram,
    MIRepNet,
    MtReNIRS,
    PopulationTransformer,
    Pratibha2,
    SignalJEPA,
    SignalJEPA_Contextual,
    SignalJEPA_PostLocal,
    SignalJEPA_PreLocal,
    SleepFM,
    STEEGFormer,
    TinyMyo,
)
from braindecode.models.base import HAS_HF_HUB, EEGModuleMixin
from braindecode.models.labram import _LABRAM_TARGET_CHS_INFO
from braindecode.models.util import (
    _fnirs_chs_info,
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
)

N_CHANS, N_TIMES, N_OUTPUTS, BATCH = 22, 1000, 4, 2
# Window length per model, from the registry.
_N_TIMES = {
    name: _get_signal_params(sp)["n_times"]
    for name, _, sp in models_mandatory_parameters
}

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
    montage = mne.channels.make_standard_montage("colin27_1020")
    ch = names or montage.ch_names[:n]
    info = mne.create_info(ch_names=ch, sfreq=256, ch_types="eeg")
    info.set_montage(montage)
    return info["chs"]


# (cls, n_chans, kwargs, has_cls)
_MODELS = [
    pytest.param(EEGPT, N_CHANS, {}, False, id="EEGPT"),
    pytest.param(BBL, N_CHANS, {}, False, id="BBL"),
    pytest.param(
        AXON,
        N_CHANS,
        {"chs_info": _chs(), "embed_dim": 64, "depth": 2, "num_heads": 4},
        False,
        id="AXON",
    ),
    pytest.param(
        Pratibha2,
        N_CHANS,
        {"chs_info": _chs(), "embed_dim": 32, "n_freqs": 2, "depth": 2, "num_heads": 4},
        False,
        id="Pratibha2",
    ),
    pytest.param(
        REVE,
        16,
        {"chs_info": _chs(_REVE_CHS), "patch_size": 200, "patch_overlap": 0},
        False,
        id="REVE",
    ),
    pytest.param(SAMBA, N_CHANS, {"chs_info": _chs()}, False, id="SAMBA"),
    pytest.param(BENDR, N_CHANS, {}, False, id="BENDR"),
    pytest.param(NICE, N_CHANS, {}, False, id="NICE"),
    pytest.param(
        BENDR,
        N_CHANS,
        {"chs_info": _chs(), "channel_strategy": "spline"},
        False,
        id="BENDR-spline",
    ),
    pytest.param(
        BaRISTA,
        N_CHANS,
        {"chs_info": _chs(), "patch_size": 200},
        False,
        id="BaRISTA",
    ),
    pytest.param(BIOT, N_CHANS, {}, False, id="BIOT"),
    pytest.param(
        MAPA,
        N_CHANS,
        {
            "contact_labels": [f"L{'AB'[i % 2]}{i + 1}" for i in range(N_CHANS)],
            "d_model": 64,
        },
        False,
        id="MAPA",
    ),
    pytest.param(
        Guetschel2026,
        N_CHANS,
        # a small projection exercises that head without a ~1 GB buffer
        {"chs_info": _chs(), "random_projection": 64},
        False,
        id="Guetschel2026",
    ),
    pytest.param(Brant, N_CHANS, {"sfreq": 250.0}, False, id="Brant"),
    pytest.param(BrainBERT, N_CHANS, {}, False, id="BrainBERT"),
    pytest.param(DuIN, N_CHANS, {"sfreq": 1000.0}, False, id="DuIN"),
    pytest.param(CBraMod, N_CHANS, {}, False, id="CBraMod"),
    pytest.param(PopulationTransformer, N_CHANS, {}, True, id="PopulationTransformer"),
    pytest.param(
        EpiNT,
        N_CHANS,
        {"embed_dim": 32, "n_layers": 1, "n_heads": 4, "ffn_dim": 64},
        True,
        id="EpiNT",
    ),
    pytest.param(EEGDINO, 16, {}, True, id="EEGDINO"),
    pytest.param(
        STEEGFormer,
        N_CHANS,
        {
            "embed_dim": 64,
            "depth": 2,
            "num_heads": 2,
            "chan_pos_idx": list(range(N_CHANS)),
        },
        True,
        id="STEEGFormer",
    ),
    pytest.param(
        CORTEG,
        N_CHANS,
        {
            "chs_info": _chs(),
            "embed_dim": 64,
            "depth": 2,
            "num_heads": 2,
            "n_merge_blocks": 1,
        },
        True,
        id="CORTEG",
    ),
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
    pytest.param(
        SleepFM,
        N_CHANS,
        {
            "sfreq": 128.0,
            "patch_size": 64,
            "embed_dim": 16,
            "num_heads": 4,
            "num_layers": 1,
            "pooling_heads": 4,
            "max_seq_length": 16,
        },
        False,
        id="SleepFM",
    ),
    pytest.param(
        MIRepNet,
        N_CHANS,
        {
            "embed_dim": 16,
            "num_layers": 2,
            "num_heads": 4,
            "feedforward_expansion": 2,
        },
        False,
        id="MIRepNet",
    ),
    pytest.param(
        GeoNIRS,
        4,
        {
            "depth": 1,
            "num_heads": 4,
            "embed_dim": 64,
            "projection_dims": (),
            "chs_info": _fnirs_chs_info(2),
        },
        False,
        id="GeoNIRS",
    ),
    pytest.param(
        MtReNIRS,
        4,
        {"depth": 1, "num_heads": 2, "embed_dim": 32, "chs_info": _fnirs_chs_info(2)},
        False,
        id="MtReNIRS",
    ),
    pytest.param(LUNA, N_CHANS, {}, False, id="LUNA"),
    pytest.param(MEGGPT, N_CHANS, {}, False, id="MEGGPT"),
    pytest.param(
        MEGXL,
        N_CHANS,
        {"chs_info": _chs(), "sfreq": 50.0, "num_layers": 1},
        False,
        id="MEGXL",
    ),
    pytest.param(FNIRSCNN1D, 4, {}, False, id="FNIRSCNN1D"),
    pytest.param(FNIRSNet, 4, {}, False, id="FNIRSNet"),
    pytest.param(FNIRS4D, 4, {"n_layers": 1, "d_state": 16}, False, id="FNIRS4D"),
    pytest.param(FNIRST, 10, {"depth": 1, "num_heads": 2}, True, id="FNIRST"),
    pytest.param(FNIRSPreT, 10, {"depth": 1, "num_heads": 2}, True, id="FNIRSPreT"),
    pytest.param(
        JungMAE,
        4,
        {"depth": 1, "num_heads": 4, "embed_dim": 64},
        False,
        id="JungMAE",
    ),
    pytest.param(TinyMyo, 8, {"depth": 1}, False, id="TinyMyo"),
]


_CLS_TOKEN_MODELS = frozenset(
    {
        "CORTEG",
        "EEGDINO",
        "EpiNT",
        "FNIRSPreT",
        "FNIRST",
        "Labram",
        "PopulationTransformer",
        "STEEGFormer",
    }
)


@PARAMS
def test_return_features_on_every_model(model_name, required_params, signal_params):
    if model_name in _TOKENIZERS:
        pytest.skip("tokenizer: use tokenize()")
    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    with torch.no_grad():
        default_out = model(x)
        assert isinstance(default_out, (torch.Tensor, tuple))
        feats = model(x, return_features=True)
    assert isinstance(feats, dict)
    assert {"features", "embedding", "cls_token"} <= set(feats)
    assert feats["features"].shape[0] == x.shape[0]
    assert feats["embedding"].ndim == 2 and feats["embedding"].shape[0] == x.shape[0]
    assert torch.isfinite(feats["embedding"]).all()
    if model_name in _CLS_TOKEN_MODELS:
        assert isinstance(feats["cls_token"], torch.Tensor)
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
    emb_shape = tuple(emb.shape)
    del model, emb
    if model_name != "SignalJEPA":
        sp_alt = _get_signal_params(signal_params, required_params)
        alt_n_outputs = int(sp_alt.get("n_outputs", 2)) + 3
        sp_alt["n_outputs"] = alt_n_outputs
        model_alt = models_dict[model_name](**sp_alt).eval()
        _materialize(model_alt, x)
        with torch.no_grad():
            emb_alt = model_alt(x, return_features=True)["embedding"]
        assert tuple(emb_alt.shape) == emb_shape
        assert emb_alt.shape[1] != alt_n_outputs


class _ToyHead(EEGModuleMixin, nn.Module):
    """Model whose forward calls ``final_layer`` ``n_calls`` times on ``x``."""

    _time_axis = "free"
    _channel_axis = "fixed"

    def __init__(self, n_chans=2, n_outputs=2, n_times=10, n_calls=1, permute=False):
        super().__init__(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times)
        self.n_calls = n_calls
        self.permute = permute
        self.final_layer = nn.Linear(n_chans if permute else n_times, n_outputs)

    def forward(self, x):
        if self.permute:  # time first: (n_times, batch, n_chans)
            return self.final_layer(x.permute(2, 0, 1)).mean(dim=0)
        out = x.new_zeros(x.shape[0], self.n_outputs)
        for _ in range(self.n_calls):
            out = out + self.final_layer(x.mean(dim=1))
        return out


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n_calls": 2}, "final_layer ran 2 times"),  # Review Focus 4
        ({"n_calls": 0}, "final_layer ran 0 times"),
        ({"permute": True}, "first axis is not the batch"),
    ],
)
def test_feature_fallback_refuses_heads_it_cannot_read(kwargs, match):
    model = _ToyHead(**kwargs).eval()
    with pytest.raises(ValueError, match=match):
        model(torch.randn(1, 2, 10), return_features=True)
    assert not model.final_layer._forward_pre_hooks  # hook removed after a refusal


def test_head_linear_finds_the_last_linear_or_conv():
    model = _ToyHead()
    lin, conv = nn.Linear(4, 2), nn.Conv1d(4, 2, 1)
    model.final_layer = nn.Sequential(nn.Flatten(), lin)
    assert model._head_linear() is lin
    model.final_layer = nn.Sequential(conv, nn.Identity())
    assert model._head_linear() is conv
    model.final_layer = nn.Identity()
    assert model._head_linear() is None


def test_return_features_with_channel_strategy():
    bendr = BENDR(
        n_chans=N_CHANS,
        n_times=N_TIMES,
        n_outputs=N_OUTPUTS,
        chs_info=_chs(),
        channel_strategy="spline",
    ).eval()
    with torch.no_grad():
        out = bendr(torch.randn(BATCH, N_CHANS, N_TIMES), return_features=True)
    assert out["embedding"].shape[0] == BATCH and out["embedding"].ndim == 2
    assert isinstance(bendr._head_linear(), nn.Linear)


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
    n_times = _N_TIMES[cls.__name__]
    kw = {**_architecture_kwargs(cls.__name__), **kw}
    model = cls(n_chans=nc, n_times=n_times, n_outputs=N_OUTPUTS, **kw)
    model.reset_head(10)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(BATCH, nc, n_times))
    assert model.n_outputs == 10
    assert out.shape[-1] == 10


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
