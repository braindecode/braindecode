# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Models that look channels up by id, on the channel layer.

LaBraM, EEGPT, STEEGFormer and SignalJEPA read a channel vocabulary (one
embedding per name); under a ``channel_strategy`` other than ``"native"`` the
embedding lookup uses ``ChannelEncoding.channel_ids``. MVPFormer has no
vocabulary (slots in input order): its contract is ``free``.
"""

import warnings

import mne
import numpy as np
import pytest
import torch

from braindecode.models import (
    EEGPT,
    Labram,
    MVPFormer,
    SignalJEPA,
    SignalJEPA_Contextual,
    STEEGFormer,
)
from braindecode.models.eegpt import EEGPT_19_CHANNELS, EEGPT_CHANNELS
from braindecode.models.labram import LABRAM_CHANNEL_ORDER
from braindecode.models.signal_jepa import _PRETRAIN_CHS_INFO
from braindecode.modules.channels import ChannelTokenizer
from braindecode.modules.channels.head import get_sphere_head

from .test_pretrained_compat import COMPAT, TEN_TWENTY, chs_from_montage, geometries

STRATEGIES = (
    "exact",
    "zero",
    "nearest",
    "idw",
    "spline",
    "field",
    "source",
    "wiener",
    "region",
    "latent",
)
GEOMETRIES = ("G1", "G2", "G3", "G3b", "G4")

# Small backbones: the grid checks the channel plumbing, not the depth.
MODELS = {
    "Labram": dict(cls=Labram, compat="Labram", kwargs=dict(num_layers=2)),
    "EEGPT": dict(
        cls=EEGPT,
        compat="EEGPT",
        kwargs=dict(depth=2, embed_dim=64, chan_proj_type="none"),
    ),
    "EEGPT-proj": dict(cls=EEGPT, compat="EEGPT", kwargs=dict(depth=2, embed_dim=64)),
    "STEEGFormer": dict(
        cls=STEEGFormer, compat="STEEGFormer", kwargs=dict(depth=1, embed_dim=64)
    ),
    "SignalJEPA": dict(
        cls=SignalJEPA,
        compat="SignalJEPA",
        kwargs=dict(transformer__num_encoder_layers=1),
    ),
    "SignalJEPA_Contextual": dict(
        cls=SignalJEPA_Contextual,
        compat="SignalJEPA",
        kwargs=dict(transformer__num_encoder_layers=1),
    ),
    "MVPFormer": dict(cls=MVPFormer, compat="MVPFormer", kwargs=dict(n_layers=1)),
}

_UNKNOWN = "not in the model vocabulary"
_SPHERE = "sphere head model"
_DENSE = "fitted dense montage"
# Cells where a strategy cannot serve the montage: a declared ValueError.
DECLARED_ERRORS = {
    # coordinates-only E1..E8: no name, no vocabulary site within 15 mm
    **{(m, "exact", "G3"): _UNKNOWN for m in MODELS if m != "MVPFormer"},
    **{(m, "wiener", "G3"): _DENSE for m in MODELS if m != "MVPFormer"},
    # biosemi64 sits on its own 95 mm sphere: most sites are > 15 mm from
    # every standard_1005 site of the fitted dense montage.
    **{(m, "wiener", "G2"): _DENSE for m in MODELS if m != "MVPFormer"},
    # biosemi64 names outside the vocabulary (EEGPT: P9, P10, Iz...;
    # SignalJEPA: F1, F2, ...); LaBraM and STEEGFormer cover all 64.
    ("EEGPT", "exact", "G2"): _UNKNOWN,
    ("EEGPT-proj", "exact", "G2"): _UNKNOWN,
    ("SignalJEPA", "exact", "G2"): _UNKNOWN,
    ("SignalJEPA_Contextual", "exact", "G2"): _UNKNOWN,
    # MVPFormer is an iEEG model (kinds=ELECTRODE_KINDS): the compat
    # geometries are sEEG channels, which the scalp-EEG ``source`` refuses.
    **{("MVPFormer", "source", g): _SPHERE for g in ("G1", "G2", "G3")},
}

_POS = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]


def _dense_fit_set(n_samples=400, seed=0):
    """Smooth synthetic fields on all standard_1005 sites (for ``wiener``)."""
    dense = [
        {"ch_name": n, "kind": "eeg", "loc": np.r_[p, np.zeros(9)]}
        for n, p in _POS.items()
    ]
    P = np.array([ch["loc"][:3] for ch in dense])
    basis = np.c_[np.ones(len(P)), P, P**2, P[:, [0]] * P[:, [1]]]
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, basis.shape[1])) @ basis.T * 1e2
    X += 0.1 * rng.standard_normal(X.shape)
    return X, dense


def _signal(spec):
    c = COMPAT[spec["compat"]]
    return dict(sfreq=c["sfreq"], n_times=c["n_times"])


def _build(name, strategy="native", chs_info=None, seed=0, **extra):
    spec = MODELS[name]
    kw = dict(n_outputs=2, **_signal(spec), **spec["kwargs"], **extra)
    if chs_info is not None:
        kw["chs_info"] = chs_info
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return spec["cls"](channel_strategy=strategy, **kw).eval()


def _out(y):
    return y["features"] if isinstance(y, dict) else y


# -- contract -----------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,vocab",
    [
        (Labram, LABRAM_CHANNEL_ORDER),
        (EEGPT, EEGPT_CHANNELS),
        (SignalJEPA, [ch["ch_name"] for ch in _PRETRAIN_CHS_INFO]),
        (SignalJEPA_Contextual, [ch["ch_name"] for ch in _PRETRAIN_CHS_INFO]),
    ],
)
def test_ids_models_declare_their_vocabulary(cls, vocab):
    target = cls._channel_target
    assert target.interface == "ids"
    assert list(target.vocabulary) == list(vocab)


def test_steegformer_vocabulary_is_its_montage_vocabulary():
    from braindecode.models.steegformer import _channel_order

    target = STEEGFormer._channel_target
    assert target.interface == "ids"
    assert list(target.vocabulary) == list(_channel_order())


def test_native_steegformer_does_not_fetch_the_vocabulary(monkeypatch):
    import braindecode.models.steegformer as steeg

    def no_download():
        raise AssertionError("the vocabulary was fetched")

    monkeypatch.setattr(steeg, "_channel_order", no_download)
    model = STEEGFormer(
        n_chans=4, n_times=400, sfreq=100, n_outputs=2, chan_pos_idx=[0, 1, 2, 3]
    )
    assert model.channel_tokenizer.strategy is None


def test_mvpformer_is_free_and_local_sjepa_heads_have_no_contract():
    from braindecode.models import SignalJEPA_PostLocal, SignalJEPA_PreLocal

    assert MVPFormer._channel_target.interface == "free"
    for cls in (SignalJEPA_PostLocal, SignalJEPA_PreLocal):
        assert cls._channel_target is None


# -- native is untouched; exact on the canonical montage equals native ----------


_CANON = {
    "Labram": chs_from_montage(LABRAM_CHANNEL_ORDER),
    "EEGPT": chs_from_montage(EEGPT_CHANNELS),
    "EEGPT-proj": chs_from_montage(EEGPT_19_CHANNELS),
    "STEEGFormer": chs_from_montage(TEN_TWENTY),
    "SignalJEPA": chs_from_montage(TEN_TWENTY),
    "SignalJEPA_Contextual": chs_from_montage(TEN_TWENTY),
    "MVPFormer": chs_from_montage(TEN_TWENTY),
}
# native SignalJEPA reads the pre-training table only when asked to.
_NATIVE_EXTRA = {
    "SignalJEPA": dict(channel_embedding="pretrain_aligned"),
    "SignalJEPA_Contextual": dict(channel_embedding="pretrain_aligned"),
}


@pytest.mark.parametrize("name", list(MODELS))
def test_exact_on_canonical_montage_equals_native(name):
    chs = _CANON[name]
    native = _build(name, "native", chs, **_NATIVE_EXTRA.get(name, {}))
    exact = _build(name, "exact", chs, **_NATIVE_EXTRA.get(name, {}))
    assert native.channel_tokenizer.strategy is None
    assert native.state_dict().keys() == exact.state_dict().keys()
    # Same seed, same backbone: the channel layer draws no random numbers.
    for k, v in native.state_dict().items():
        assert torch.equal(v, exact.state_dict()[k]), k
    exact.load_state_dict(native.state_dict(), strict=True)
    x = torch.randn(2, len(chs), _signal(MODELS[name])["n_times"])
    with torch.no_grad():
        y_native = native(x)
        y_exact = exact(x)
    assert torch.equal(y_native, y_exact)


@pytest.mark.parametrize("name", ["Labram", "EEGPT", "STEEGFormer", "MVPFormer"])
def test_strategy_model_loads_the_native_checkpoint_strictly(name):
    chs = _CANON[name]
    native = _build(name, "native", chs)
    spline = _build(name, "spline", chs_from_montage(["Fz", "Cz", "Pz", "Oz", "C3"]))
    assert native.state_dict().keys() == spline.state_dict().keys()
    spline.load_state_dict(native.state_dict(), strict=True)


@pytest.mark.parametrize("name", ["Labram", "EEGPT", "STEEGFormer"])
def test_from_pretrained_with_a_strategy(name, tmp_path):
    native = _build(name, "native", _CANON[name])
    native.save_pretrained(tmp_path)
    four = chs_from_montage(["Fz", "Cz", "Pz", "Oz"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = MODELS[name]["cls"].from_pretrained(
            tmp_path, chs_info=four, channel_strategy="source"
        )
    # Map-quality warnings (4 inputs) are expected; "freshly initialised"
    # is not, since source adds no weights.
    assert not [w for w in caught if "freshly initialised" in str(w.message)]
    assert model.channel_tokenizer.strategy_name == "source"
    for k, v in native.state_dict().items():
        assert torch.equal(v, model.state_dict()[k]), k
    with torch.no_grad():
        y = model.eval()(torch.randn(1, 4, _signal(MODELS[name])["n_times"]))
    assert torch.isfinite(_out(y)).all()


@pytest.mark.parametrize("strategy", ["spline", "source", "latent"])
@pytest.mark.parametrize("name", list(MODELS))
def test_module_apply_reaches_the_channel_strategy(name, strategy):
    # ``ChannelStrategy`` must not shadow ``nn.Module.apply``: user init code
    # such as ``model.apply(fn)`` walks every submodule, the strategy included.
    model = _build(name, strategy, chs_from_montage(["Fz", "Cz", "Pz", "Oz", "C3"]))
    seen = []
    assert model.apply(seen.append) is model
    assert model.channel_tokenizer.strategy in seen
    assert model in seen


@pytest.mark.parametrize("name", list(MODELS))
def test_channel_strategy_round_trips_through_config(name):
    model = _build(
        name,
        "spline",
        chs_from_montage(["Fz", "Cz", "Pz", "Oz", "C3"]),
        channel_strategy_kwargs={"reg": 1e-2},
    )
    cfg = model.get_config()
    assert cfg["channel_strategy"] == "spline"
    assert cfg["channel_strategy_kwargs"] == {"reg": 1e-2}
    assert model.channel_tokenizer.strategy.reg == 1e-2


# -- every strategy x the compat geometries -------------------------------------


def _cells():
    for name, spec in MODELS.items():
        geos = geometries(COMPAT[spec["compat"]])
        for s in STRATEGIES:
            for g in GEOMETRIES:
                if g in geos:
                    yield pytest.param(name, s, g, geos[g], id=f"{name}-{s}-{g}")


@pytest.mark.parametrize("name,strategy,gname,gkw", list(_cells()))
def test_strategy_by_geometry(name, strategy, gname, gkw):
    spec = MODELS[name]
    n_in = len(gkw["chs_info"])

    def build_and_forward():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.manual_seed(0)
            model = spec["cls"](
                n_outputs=2, channel_strategy=strategy, **gkw, **spec["kwargs"]
            ).eval()
            if strategy == "wiener":
                model.channel_tokenizer.fit(*_dense_fit_set())
            with torch.no_grad():
                return model(torch.randn(1, n_in, gkw["n_times"]))

    key = (name, strategy, gname)
    if key in DECLARED_ERRORS:
        with pytest.raises(ValueError, match=DECLARED_ERRORS[key]):
            build_and_forward()
    else:
        y = build_and_forward()
        assert y.shape[0] == 1 and torch.isfinite(y).all()


# -- LaBraM: #1227's names migration ------------------------------------------


@pytest.mark.parametrize("gname", ["G2", "G3", "G3b"])
def test_labram_not_yet_compat_cells_pass_with_a_strategy(gname):
    # The three strict-xfail cells of test_pretrained_compat (NOT_YET), with
    # the default backbone and channel_strategy set.
    gkw = geometries(COMPAT["Labram"])[gname]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Labram(n_outputs=2, channel_strategy="spline", **gkw).eval()
        with torch.no_grad():
            y = model(torch.randn(1, len(gkw["chs_info"]), gkw["n_times"]))
    assert y.shape == (1, 2) and torch.isfinite(y).all()


def test_labram_ids_replace_its_name_lookup():
    names = ["Cz", "T3", "FP1", "O2"]
    model = _build("Labram", "exact", chs_from_montage(names))
    native = _build("Labram", "native", chs_from_montage(names))
    native.load_state_dict(model.state_dict(), strict=True)
    x = torch.randn(1, 4, 800)
    with torch.no_grad():
        y_ids = model(x)
        y_names = native(x, ch_names=names)  # the native name lookup
        y_shortcut = model(x, ch_names=names)  # names through the layer
    assert torch.equal(y_ids, y_names)
    assert torch.equal(y_ids, y_shortcut)
    with pytest.raises(ValueError, match="either ch_names or chs_info"):
        model(x, ch_names=names, chs_info=chs_from_montage(names))


def test_labram_per_call_montage():
    model = _build("Labram", "spline", chs_from_montage(TEN_TWENTY))
    other = chs_from_montage(["Fz", "Cz", "Pz", "Oz", "C3", "C4"])
    with torch.no_grad():
        y = model(torch.randn(1, 6, 800), chs_info=other)
    assert y.shape == (1, 2) and torch.isfinite(y).all()


# -- observed reaches the key-padding mask ------------------------------------


def _perturbed_outputs(model, emb, ids, x):
    """Features before and after perturbing embedding rows ``ids``."""
    gen = torch.Generator().manual_seed(1)
    delta = torch.randn(len(ids), emb.shape[1], generator=gen)
    with torch.no_grad():
        before = model(x, return_features=True)
        emb[ids] += delta
        after = model(x, return_features=True)
        emb[ids] -= delta
    return before, after


def test_labram_masks_reconstructed_channels_as_keys():
    names = ["Fz", "Cz", "Pz", "Oz"]
    model = _build("Labram", "zero", chs_from_montage(names))
    enc = model.channel_tokenizer(torch.zeros(1, 4, 1))
    assert enc.observed.sum() == 4 and len(enc.observed) == 128
    hidden = enc.channel_ids[~enc.observed]
    shown = enc.channel_ids[enc.observed]
    emb = model.position_embedding.data[0]
    x = torch.randn(1, 4, 800)
    n_p = model.patch_embed[0].n_patchs
    obs_tokens = enc.observed.repeat_interleave(n_p)
    # Unobserved channels are keys nobody attends to: perturbing their
    # embedding leaves the CLS and every observed token unchanged.
    before, after = _perturbed_outputs(model, emb, hidden + 1, x)
    torch.testing.assert_close(before["cls_token"], after["cls_token"])
    torch.testing.assert_close(
        before["features"][:, obs_tokens], after["features"][:, obs_tokens]
    )
    before, after = _perturbed_outputs(model, emb, shown[:1] + 1, x)
    assert not torch.allclose(before["cls_token"], after["cls_token"])


def test_eegpt_masks_reconstructed_channels_as_keys():
    names = ["FZ", "CZ", "PZ", "OZ"]
    model = _build("EEGPT", "zero", chs_from_montage(names))
    enc = model.channel_tokenizer(torch.zeros(1, 4, 1))
    assert enc.observed.sum() == 4 and len(enc.observed) == len(EEGPT_CHANNELS)
    emb = model.target_encoder.chan_embed.weight.data
    x = torch.randn(1, 4, 1024)
    # The summary tokens only read observed channels.
    before, after = _perturbed_outputs(model, emb, enc.channel_ids[~enc.observed], x)
    torch.testing.assert_close(before["features"], after["features"])
    before, after = _perturbed_outputs(model, emb, enc.channel_ids[enc.observed][:1], x)
    assert not torch.allclose(before["features"], after["features"])


def test_eegpt_mask_and_pretraining_mask_are_exclusive():
    model = _build("EEGPT", "zero", chs_from_montage(["FZ", "CZ", "PZ", "OZ"]))
    x = torch.randn(1, len(EEGPT_CHANNELS), 1024)
    keep = torch.ones(len(EEGPT_CHANNELS), dtype=torch.bool)
    with pytest.raises(ValueError, match="mask_x"):
        model.target_encoder(
            x, mask_x=torch.zeros(1, 1, dtype=torch.long), key_padding_mask=keep
        )


# -- model-specific declared errors -------------------------------------------


def test_steegformer_strategy_errors():
    chs = chs_from_montage(TEN_TWENTY)
    with pytest.raises(ValueError, match="chan_pos_idx cannot be combined"):
        _build("STEEGFormer", "spline", chs, chan_pos_idx=list(range(19)))
    with pytest.raises(ValueError, match="published 145-slot vocabulary"):
        _build("STEEGFormer", "spline", chs, n_chans_pos=256)


def test_signal_jepa_exact_with_another_channel_count_is_declared():
    model = _build("SignalJEPA", "exact", chs_from_montage(TEN_TWENTY))
    other = chs_from_montage(["Fz", "Cz", "Pz", "Oz"])
    with pytest.raises(ValueError, match="built for 19"):
        model(torch.randn(1, 4, 2048), chs_info=other)


def test_signal_jepa_transfer_needs_native():
    pre = _build("SignalJEPA", "spline", chs_from_montage(TEN_TWENTY))
    with pytest.raises(ValueError, match="channel_strategy='native'"):
        SignalJEPA_Contextual.from_pretrained(pre, n_outputs=2)


@pytest.mark.parametrize(
    "strategy,expected",
    [("exact", 6), ("spline", 6), ("source", 64), ("latent", 64)],
)
def test_mvpformer_slots_on_an_eeg_montage(strategy, expected):
    chs = chs_from_montage(["Fz", "Cz", "Pz", "Oz", "C3", "C4"])
    model = _build("MVPFormer", strategy, chs, pooling="concat")
    assert model.n_backbone_chans == expected
    assert model.final_layer.in_features == expected * model.d_model
    with torch.no_grad():
        y = model(torch.randn(1, 6, 2560))
    assert y.shape == (1, 2) and torch.isfinite(y).all()


# -- core fix found here ------------------------------------------------------


def test_source_lead_field_finite_for_an_electrode_on_the_sphere_axis():
    # biosemi64 Cz sits at (0, 0, 0.095), on the line through the sphere
    # centre (0, 0, 0.04) and a grid dipole: MNE's formula gave NaN rows.
    pos = np.array([[0.0, 0.0, 0.095], [0.05, 0.0, 0.07], [-0.05, 0.0, 0.07]])
    assert np.isfinite(get_sphere_head().leadfield(pos)).all()
    bio = chs_from_montage(
        mne.channels.make_standard_montage("biosemi64").ch_names, "biosemi64"
    )
    tok = ChannelTokenizer(Labram._channel_target, "source", src_chs_info=bio)
    assert torch.isfinite(tok(torch.randn(1, 64, 10)).x).all()
