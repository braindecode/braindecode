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

from braindecode.models import LUNA
from braindecode.modules.channels import ChannelEncoding

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
}

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
