# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""``positions``-interface models on the channel layer.

LUNA, REVE, ZUNA, BaRISTA, DIVER-1 and PopT read electrode coordinates. Under a
channel strategy they take ``x`` and the coordinates from the channel layer
instead of parsing ``chs_info`` themselves. Each model is built on its first
geometry (G1, its "training" montage) and fed every geometry G1-G4 per call.
"""

import math
import warnings

import mne
import numpy as np
import pytest
import torch

from braindecode.models import DIVER1, LUNA, REVE, ZUNA, BaRISTA
from braindecode.models.diver1 import channel_metadata_from_chs_info
from braindecode.modules.channels import ChannelEncoding

from .test_integration import convert_model_to_plain
from .test_pretrained_compat import TEN_TWENTY, geometries

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


# name -> (class, small kwargs, geometry spec, declared errors {(strategy,
# geometry or "*"): message}); the model is built on the spec's G1 montage.
MODELS = {
    "LUNA": dict(
        cls=LUNA,
        kwargs=dict(n_outputs=2, embed_dim=32, depth=2, num_queries=4),
        spec=dict(sfreq=200, n_times=800, canon=TEN_TWENTY),
        errors={},
    ),
    # Flattened head: every montage is mapped onto the constructor's 19.
    "REVE": dict(
        cls=REVE,
        kwargs=dict(n_outputs=2, embed_dim=64, depth=2, heads=2, head_dim=16),
        spec=dict(sfreq=200, n_times=800, canon=TEN_TWENTY),
        errors={
            ("exact", "G3"): "not in the input montage",
            ("exact", "G3b"): "not in the input montage",
        },
    ),
    # Rotary buffers and head fixed at construction: mapped onto its 19.
    "ZUNA": dict(
        cls=ZUNA,
        kwargs=dict(
            n_outputs=2, dim=64, n_layers=2, n_heads=2, head_dim=16, latent_dim=8
        ),
        spec=dict(sfreq=256, n_times=1024, canon=TEN_TWENTY),
        errors={
            ("exact", "G3"): "not in the input montage",
            ("exact", "G3b"): "not in the input montage",
        },
    ),
}
SOURCE_IS_EEG_ONLY = "sphere head model"
MODELS.update(
    # sEEG (coords-only G1, no G4); learned pooling: mapped onto its 16 contacts.
    BaRISTA=dict(
        cls=BaRISTA,
        kwargs=dict(n_outputs=2, d_model=16, n_layers=2, num_heads=2),
        spec=dict(sfreq=2048, n_times=1024, canon=None, kind="seeg"),
        errors={
            ("source", "*"): SOURCE_IS_EEG_ONLY,
            ("exact", "G2"): "not in the input montage",
            ("exact", "G3"): "not in the input montage",  # E9..E16 missing
            ("exact", "G3b"): "not in the input montage",
            # biosemi64 sites are not standard_1005 sites (> 15 mm off)
            ("wiener", "G2"): "fitted dense montage",
        },
    ),
    # Flattened head: mapped onto its 19 (pooling="mean" is a pass-through).
    DIVER1=dict(
        cls=DIVER1,
        kwargs=dict(n_outputs=2, d_model=64, n_layers=2, patch_size=500),
        spec=dict(sfreq=500, n_times=1000, canon=TEN_TWENTY),
        errors={
            ("exact", "G3"): "not in the input montage",
            ("exact", "G3b"): "not in the input montage",
        },
    ),
)

_POS = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]


def _circle(n, prefix):
    out = []
    for i in range(n):
        th = 2 * math.pi * i / n
        out.append((f"{prefix}{i + 1}", [0.08 * math.cos(th), 0.08 * math.sin(th), 0.03]))
    return out


def _dense_fit_set(n_samples=300, seed=0):
    """Smooth synthetic fields on standard_1005 + the coords-only sites."""
    sites = list(_POS.items()) + _circle(16, "D16_") + _circle(8, "D8_")
    dense = [
        {"ch_name": n, "kind": "eeg", "loc": np.r_[p, np.zeros(9)]} for n, p in sites
    ]
    P = np.array([ch["loc"][:3] for ch in dense])
    basis = np.c_[np.ones(len(P)), P, P**2, P[:, [0]] * P[:, [1]]]
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, basis.shape[1])) @ basis.T * 1e2
    X += 0.1 * rng.standard_normal(X.shape)
    return X, dense


def _geos(name):
    return geometries(MODELS[name]["spec"])


def _build(name, strategy="native", chs_info=None, **extra):
    entry = MODELS[name]
    geo = _geos(name)["G1"]
    kw = dict(
        chs_info=geo["chs_info"] if chs_info is None else chs_info,
        n_times=geo["n_times"],
        **entry["kwargs"],
    )
    if geo["sfreq"]:
        kw["sfreq"] = geo["sfreq"]
    kw.update(extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.manual_seed(0)
        model = entry["cls"](channel_strategy=strategy, **kw).eval()
        if strategy == "wiener":
            model.channel_tokenizer.fit(*_dense_fit_set())
    return model


def _x(n_chans, n_times, seed=1):
    return torch.randn(2, n_chans, n_times, generator=torch.Generator().manual_seed(seed))


def _logits(out):
    return out["features"] if isinstance(out, dict) else out


def _cells():
    for name in MODELS:
        geos = _geos(name)
        for s in STRATEGIES:
            for g in GEOMETRIES:
                if g in geos:
                    yield pytest.param(name, s, g, id=f"{name}-{s}-{g}")


@pytest.mark.parametrize("name,strategy,gname", list(_cells()))
def test_strategy_by_geometry(name, strategy, gname):
    """Every strategy x G1-G4 forwards finite or raises its declared error."""
    gkw = _geos(name)[gname]
    errors = MODELS[name]["errors"]
    msg = errors.get((strategy, gname), errors.get((strategy, "*")))

    def build_and_forward():
        model = _build(name, strategy)
        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            x = _x(len(gkw["chs_info"]), gkw["n_times"])
            return model(x, chs_info=gkw["chs_info"])

    if msg is not None:
        with pytest.raises(ValueError, match=msg):
            build_and_forward()
    else:
        y = build_and_forward()
        assert y.shape == (2, 2) and torch.isfinite(y).all()


@pytest.mark.parametrize("name", list(MODELS))
def test_native_has_no_layer(name):
    model = _build(name)
    assert model.channel_tokenizer is None
    assert not any(k.startswith("channel_tokenizer") for k in model.state_dict())
    assert type(model)._channel_target.interface == "positions"
    with pytest.raises(ValueError, match="native"):
        _build(name, channel_strategy_kwargs={"reg": 0.1})


@pytest.mark.parametrize("name", list(MODELS))
def test_strategy_model_loads_the_native_checkpoint_strictly(name):
    native = _build(name)
    model = _build(name, "spline")
    assert list(model.state_dict()) == list(native.state_dict())
    model.load_state_dict(native.state_dict(), strict=True)
    trainable = _build(name, "latent")
    with pytest.warns(UserWarning, match="freshly initialised"):
        trainable.load_state_dict(native.state_dict(), strict=True)


@pytest.mark.parametrize("name", list(MODELS))
def test_channel_strategy_round_trips_through_config(name):
    model = _build(name, "spline", channel_strategy_kwargs={"reg": 0.01})
    cfg = model.get_config()
    assert cfg["channel_strategy"] == "spline"
    assert cfg["channel_strategy_kwargs"] == {"reg": 0.01}
    assert model.channel_tokenizer.strategy.reg == 0.01


def _swap_observed(model, observed):
    """Make the layer report ``observed`` (other fields unchanged)."""
    tok = model.channel_tokenizer
    forward = tok.forward

    def patched(x, chs_info=None):
        enc = forward(x, chs_info)
        return ChannelEncoding(
            enc.x, enc.channel_ids, enc.positions, observed, enc.support, enc.weights
        )

    tok.forward = patched


# --------------------------------------------------------------------------- LUNA


def test_luna_positions_replace_its_own_parsing():
    """Under a strategy LUNA reads the layer's positions: same output as
    native with the montage's coordinates passed explicitly."""
    geos = _geos("LUNA")
    native = _build("LUNA")
    model = _build("LUNA", "exact")
    model.load_state_dict(native.state_dict())
    for g in ("G2", "G3", "G3b"):
        chs = geos[g]["chs_info"]
        x = _x(len(chs), 800)
        pos = torch.as_tensor(
            np.stack(
                [
                    ch["loc"][:3] if np.any(ch["loc"][:3]) else _POS[ch["ch_name"]]
                    for ch in chs
                ]
            ),
            dtype=torch.float32,
        )
        with torch.no_grad():
            ref = native(x, channel_locations=pos.expand(2, -1, -1))
            out = model(x, chs_info=chs)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_luna_unobserved_channels_are_masked_in_cross_attention():
    """``observed`` reaches the channel-unification key-padding mask."""
    model = _build("LUNA", "spline")
    observed = torch.ones(19, dtype=torch.bool)
    observed[[0, 5]] = False
    seen = []
    mha = model.cross_attn.cross_attention
    mha_forward = mha.forward

    def spy(*args, **kwargs):
        seen.append(kwargs.get("key_padding_mask"))
        return mha_forward(*args, **kwargs)

    mha.forward = spy
    x = _x(19, 800)
    with torch.no_grad():
        y_all = model(x)
        _swap_observed(model, observed)
        y_masked = model(x)
    assert seen[0] is None  # everything observed: no mask
    mask = seen[1]
    assert mask.shape == (2 * 20, 19)  # (batch * patches, channels)
    assert (mask == ~observed).all()
    assert torch.isfinite(y_masked).all() and not torch.allclose(y_masked, y_all)


# --------------------------------------------------------------------------- REVE


def _std_positions(names):
    return torch.as_tensor(np.stack([_POS[n] for n in names]), dtype=torch.float32)


def test_reve_positions_replace_the_position_bank():
    """A permuted montage under ``exact`` is the native model on the
    constructor's order, with the layer's (standard_1005) positions."""
    geos = _geos("REVE")
    native = _build("REVE")
    model = _build("REVE", "exact")
    model.load_state_dict(native.state_dict())
    x = _x(19, 800)
    with torch.no_grad():
        ref = native(x, pos=_std_positions(TEN_TWENTY).expand(2, -1, -1))
        out = model(x.flip(1), chs_info=geos["G4"]["chs_info"])
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    with torch.no_grad():  # the bank holds the same positions (to 4 nm)
        torch.testing.assert_close(native(x), ref, rtol=0, atol=1e-4)


def test_reve_attention_pooling_is_a_pass_through():
    geos = _geos("REVE")
    native = _build("REVE", attention_pooling=True)
    model = _build("REVE", "spline", attention_pooling=True)
    model.load_state_dict(native.state_dict())
    chs = geos["G2"]["chs_info"]  # 64 biosemi channels, positions from loc
    x = _x(len(chs), 800)
    pos = torch.as_tensor(np.stack([ch["loc"][:3] for ch in chs]), dtype=torch.float32)
    with torch.no_grad():
        ref = native(x, pos=pos.expand(2, -1, -1))
        out = model(x, chs_info=chs)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_reve_unobserved_channels_are_masked_in_self_attention(monkeypatch):
    """Channels the layer reconstructs get an SDPA key-padding mask."""
    import braindecode.models.reve as reve

    seen = []
    sdpa = reve.F.scaled_dot_product_attention

    def spy(q, k, v, attn_mask=None, **kw):
        seen.append(attn_mask)
        return sdpa(q, k, v, attn_mask=attn_mask, **kw)

    monkeypatch.setattr(reve.F, "scaled_dot_product_attention", spy)
    model = _build("REVE", "spline")
    chs = _geos("REVE")["G3b"]["chs_info"]  # Fp1..T7: 8 of the 19 channels
    with torch.no_grad():
        y = model(_x(8, 800), chs_info=chs)
    assert torch.isfinite(y).all() and len(seen) == 2  # one per layer
    n_patches = (800 - 200) // 180 + 1
    observed = torch.tensor([n in TEN_TWENTY[:8] for n in TEN_TWENTY])
    expected = observed.repeat_interleave(n_patches)
    for mask in seen:
        assert mask.shape == (1, 1, 1, 19 * n_patches)
        assert (mask.flatten() == expected).all()
    seen.clear()
    with torch.no_grad():
        model(_x(19, 800))  # constructor montage: all observed, no mask
    assert seen == [None, None]


# --------------------------------------------------------------------------- ZUNA


def test_zuna_maps_any_montage_onto_its_rotary_montage():
    geos = _geos("ZUNA")
    native = _build("ZUNA")
    model = _build("ZUNA", "exact")
    model.load_state_dict(native.state_dict())
    x = _x(19, 1024)
    with torch.no_grad():
        ref = native(x)
        out = model(x.flip(1), chs_info=geos["G4"]["chs_info"])
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    enc = model.channel_tokenizer(x.flip(1), geos["G4"]["chs_info"])
    torch.testing.assert_close(enc.positions, _std_positions(TEN_TWENTY))
    spline = _build("ZUNA", "spline")
    with torch.no_grad():
        y = spline(_x(8, 1024), chs_info=geos["G3b"]["chs_info"])
    assert y.shape == (2, 2) and torch.isfinite(y).all()


def test_zuna_native_stays_scriptable():
    model = _build("ZUNA")
    scripted = torch.jit.script(convert_model_to_plain(model).eval())
    x = _x(19, 1024)
    with torch.no_grad():
        torch.testing.assert_close(scripted(x), model(x), rtol=0, atol=0)


# ------------------------------------------------------------------------ BaRISTA


def test_barista_seeg_contacts_pass_the_resolve_stage():
    """sEEG contacts are electrodes for the sensor strategies; ``source``
    (scalp sphere head) is a declared error, at construction and per call."""
    geos = _geos("BaRISTA")
    native = _build("BaRISTA")
    model = _build("BaRISTA", "nearest")
    model.load_state_dict(native.state_dict())
    x = _x(16, 1024)
    with torch.no_grad():
        torch.testing.assert_close(
            model(x, chs_info=geos["G1"]["chs_info"]), native(x), rtol=0, atol=0
        )
    with pytest.raises(ValueError, match=SOURCE_IS_EEG_ONLY):
        _build("BaRISTA", "source")
    with pytest.raises(ValueError, match=SOURCE_IS_EEG_ONLY):
        _build("BaRISTA", "source", pooling="mean")


def test_barista_mean_pooling_bins_the_layer_positions():
    geos = _geos("BaRISTA")
    native = _build("BaRISTA", pooling="mean")
    model = _build("BaRISTA", "idw", pooling="mean")
    model.load_state_dict(native.state_dict())
    for g in ("G2", "G3"):
        chs = geos[g]["chs_info"]
        x = _x(len(chs), 1024)
        pos = torch.as_tensor(
            np.stack([ch["loc"][:3] for ch in chs]), dtype=torch.float32
        )
        with torch.no_grad():
            ref = native(x, spatial_indices=native._coord_indices(pos).long())
            out = model(x, chs_info=chs)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_barista_native_stays_scriptable():
    model = _build("BaRISTA")
    scripted = torch.jit.script(convert_model_to_plain(model).eval())
    x = _x(16, 1024)
    with torch.no_grad():
        torch.testing.assert_close(scripted(x), model(x), rtol=0, atol=0)


# ------------------------------------------------------------------------- DIVER1


def test_diver1_flatten_maps_any_montage_onto_its_own():
    geos = _geos("DIVER1")
    native = _build("DIVER1")
    model = _build("DIVER1", "exact")
    model.load_state_dict(native.state_dict())
    x = _x(19, 1000)
    with torch.no_grad():
        ref = native(x)
        out = model(x.flip(1), chs_info=geos["G4"]["chs_info"])
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_diver1_mean_pooling_reads_the_layer_positions():
    geos = _geos("DIVER1")
    native = _build("DIVER1", pooling="mean")
    model = _build("DIVER1", "spline", pooling="mean")
    model.load_state_dict(native.state_dict())
    chs = geos["G2"]["chs_info"]
    x = _x(len(chs), 1000)
    with torch.no_grad():
        ref = native(x, chan_metadata=channel_metadata_from_chs_info(chs))
        out = model(x, chs_info=chs)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    # names without loc: the layer supplies standard_1005 positions
    chs = geos["G3b"]["chs_info"]
    meta = channel_metadata_from_chs_info(chs)
    assert meta[:, :3].isnan().all()
    meta[:, :3] = 1e3 * _std_positions([ch["ch_name"] for ch in chs])
    x = _x(8, 1000)
    with torch.no_grad():
        ref = native(x, chan_metadata=meta)
        out = model(x, chs_info=chs)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_diver1_intracranial_channels():
    """sEEG contacts serve the sensor strategies; ``source`` is declared."""
    seeg = geometries(dict(sfreq=500, n_times=1000, canon=None, kind="seeg"))
    chs = seeg["G1"]["chs_info"]
    native = _build("DIVER1", chs_info=chs, pooling="mean")
    model = _build("DIVER1", "nearest", chs_info=chs, pooling="mean")
    model.load_state_dict(native.state_dict())
    x = _x(len(chs), 1000)
    with torch.no_grad():
        torch.testing.assert_close(model(x), native(x), rtol=0, atol=0)
    with pytest.raises(ValueError, match=SOURCE_IS_EEG_ONLY):
        _build("DIVER1", "source", chs_info=chs, pooling="mean")
    eeg_source = _build("DIVER1", "source", pooling="mean")
    with pytest.raises(ValueError, match=SOURCE_IS_EEG_ONLY):
        eeg_source(_x(8, 1000), chs_info=seeg["G3"]["chs_info"])


def test_diver1_native_stays_scriptable():
    model = _build("DIVER1")
    scripted = torch.jit.script(convert_model_to_plain(model).eval())
    x = _x(19, 1000)
    with torch.no_grad():
        torch.testing.assert_close(scripted(x), model(x), rtol=0, atol=0)
