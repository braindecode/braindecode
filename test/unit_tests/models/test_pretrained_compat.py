"""Geometry-compatibility contract for models with released weights.

Every model class that ships pretrained weights is built (random weights, no
download) on a grid of input geometries -- canonical montage, permuted order,
a 64-channel montage outside the 10-20 vocabulary, coordinates-only channels,
names without coordinates, short / long / non-divisible windows, other
sampling rates -- and must either forward or raise the *declared* error.

The expected outcome of each cell is derived from the model's channel
contract (the interface of its ``_channel_target``; ``COMPAT`` below holds
the rest), not hard-coded per cell, so adding a model means adding one entry.

``test_channel_strategy_contract`` crosses every model with every channel
strategy on the channel geometries G1-G4: each cell forwards a finite output
or raises one of the layer's declared ``ValueError`` (``DECLARED_ERRORS``).
Strategies other than ``native``, ``exact`` and ``source`` are marked slow.
"""

from __future__ import annotations

import math
import re
import warnings

import mne
import numpy as np
import pytest
import torch

from braindecode.models import (
    BENDR,
    BIOT,
    DIVER1,
    EEGDINO,
    EEGPT,
    LUNA,
    REVE,
    ZUNA,
    BaRISTA,
    BrainBERT,
    Brant,
    CBraMod,
    CodeBrain,
    Labram,
    MIRepNet,
    MVPFormer,
    PopulationTransformer,
    SignalJEPA,
    STEEGFormer,
)
from braindecode.models.bendr import BENDR_CHANNEL_ORDER
from braindecode.models.biot import BIOT_CHANNEL_ORDER
from braindecode.models.eegpt import EEGPT_19_CHANNELS
from braindecode.models.labram import LABRAM_CHANNEL_ORDER
from braindecode.models.mirepnet import MIREPNET_CHANNEL_ORDER

TEN_TWENTY = "Fp1 Fp2 F7 F3 Fz F4 F8 T7 C3 Cz C4 T8 P7 P3 Pz P4 P8 O1 O2".split()


def _montage(name):
    try:
        return mne.channels.make_standard_montage(name)
    except ValueError:  # MNE >= 1.13 renamed the standard montages
        return mne.channels.make_standard_montage(name.replace("standard", "colin27"))


def chs_from_montage(names, montage="standard_1005", kind="eeg"):
    pos = _montage(montage).get_positions()["ch_pos"]
    upper = {k.upper(): k for k in pos}
    chs = []
    for n in names:
        loc = np.zeros(12)
        key = upper.get(n.upper())
        if key is not None:
            loc[:3] = pos[key]
        chs.append({"ch_name": n, "kind": kind, "loc": loc})
    return chs


def chs_coords_only(n, kind="eeg"):
    chs = []
    for i in range(n):
        th = 2 * math.pi * i / n
        loc = np.zeros(12)
        loc[:3] = [0.08 * math.cos(th), 0.08 * math.sin(th), 0.03]
        chs.append({"ch_name": f"E{i + 1}", "kind": kind, "loc": loc})
    return chs


def chs_names_no_loc(names):
    return [{"ch_name": n, "kind": "eeg", "loc": np.zeros(12)} for n in names]


# Declared behaviour per pretrained class. How the checkpoint identifies
# channels comes from the class's ``_channel_target.interface`` (montage / ids
# / positions / slots / free, see ``interface``); a ``channels`` entry is only
# needed for a model without a channel contract. ``min_n_times`` = smallest
# accepted window in samples.
COMPAT = {
    "Labram": dict(
        cls=Labram,
        sfreq=200,
        n_times=800,
        canon=LABRAM_CHANNEL_ORDER,
        names_required_at_forward=True,
    ),
    "EEGPT": dict(
        cls=EEGPT,
        sfreq=256,
        n_times=1024,
        canon=EEGPT_19_CHANNELS,
    ),
    "BENDR": dict(
        cls=BENDR,
        sfreq=256,
        n_times=1024,
        canon=BENDR_CHANNEL_ORDER,
    ),
    "BIOT": dict(
        cls=BIOT,
        sfreq=200,
        n_times=800,
        canon=BIOT_CHANNEL_ORDER,
    ),
    "CBraMod": dict(
        cls=CBraMod,
        sfreq=200,
        n_times=800,
        canon=TEN_TWENTY,
    ),
    "CodeBrain": dict(
        cls=CodeBrain,
        sfreq=200,
        n_times=800,
        canon=TEN_TWENTY,
    ),
    "EEGDINO": dict(
        cls=EEGDINO,
        sfreq=200,
        n_times=800,
        canon=TEN_TWENTY,
        max_n_chans=19,
    ),
    "LUNA": dict(
        cls=LUNA,
        sfreq=200,
        n_times=800,
        canon=TEN_TWENTY,
        coords_checked=False,
    ),
    "REVE": dict(
        cls=REVE,
        sfreq=200,
        n_times=800,
        canon=TEN_TWENTY,
        coords_checked=False,
    ),
    "SignalJEPA": dict(
        cls=SignalJEPA,
        sfreq=128,
        n_times=2048,
        canon=TEN_TWENTY,
        min_n_times=160,
    ),
    "STEEGFormer": dict(
        cls=STEEGFormer,
        sfreq=100,
        n_times=400,
        canon=TEN_TWENTY,
    ),
    "ZUNA": dict(
        cls=ZUNA,
        sfreq=256,
        n_times=1024,
        canon=TEN_TWENTY,
        coords_checked=True,
    ),
    "Brant": dict(
        cls=Brant,
        sfreq=250,
        n_times=1500,
        canon=TEN_TWENTY,
        kwargs=dict(patch_size=250),
    ),
    "BrainBERT": dict(
        cls=BrainBERT,
        sfreq=2048,
        n_times=4096,
        canon=["E1"],
        min_n_times=1401,
    ),
    "BaRISTA": dict(
        cls=BaRISTA,
        sfreq=2048,
        n_times=2048,
        canon=None,
        kind="seeg",
        coords_checked=True,
        min_n_times=512,
    ),
    "DIVER1": dict(
        cls=DIVER1,
        sfreq=500,
        n_times=1000,
        canon=TEN_TWENTY,
        coords_checked=False,
    ),
    "MIRepNet": dict(
        cls=MIRepNet,
        sfreq=250,
        n_times=1000,
        canon=MIREPNET_CHANNEL_ORDER,
    ),
    "MVPFormer": dict(
        cls=MVPFormer,
        sfreq=512,
        n_times=2560,
        canon=None,
        kind="seeg",
    ),
    "PopT": dict(
        cls=PopulationTransformer,
        sfreq=None,
        n_times=768,
        canon=None,
        kind="seeg",
        coords_checked=False,
    ),
}

# Cells the native path cannot serve, run with a channel strategy instead
# (LaBraM looks names up at forward; biosemi64 / coordinates-only / names
# without positions go through the layer).
STRATEGY_CELLS = {
    ("Labram", "G2"): "spline",
    ("Labram", "G3"): "spline",
    ("Labram", "G3b"): "spline",
}

STRATEGIES = (
    "native",
    "exact",
    "source",
    "zero",
    "nearest",
    "idw",
    "spline",
    "field",
    "wiener",
    "region",
    "latent",
)
FAST_STRATEGIES = ("native", "exact", "source")
CHANNEL_GEOMETRIES = ("G1", "G2", "G3", "G3b", "G4")

# What a strategy may refuse, with the layer's declared message: missing
# targets under ``exact``, names outside an id vocabulary, too few positioned
# channels, sites outside the fitted dense montage (``wiener``), non-EEG
# channels for the scalp-EEG ``source``; and the constructor checks of models
# built on a montage without positions (ZUNA, BaRISTA).
DECLARED_ERRORS = "|".join(
    [
        r"not in the input",
        r"not in the model vocabulary",
        r"needs at least \d+ channels with a position",
        r"no electrode of the fitted dense montage",
        r"sphere head model",
        r"requires channel locations",
        r"No spatial indices available",
    ]
)


def interface(spec):
    """Channel interface of a COMPAT entry (from its ``_channel_target``)."""
    target = spec["cls"]._channel_target
    return target.interface if target is not None else spec["channels"]


def geometries(spec):
    sf, nt, kind = spec["sfreq"], spec["n_times"], spec.get("kind", "eeg")
    canon = spec["canon"]
    geos = {}
    if canon is not None:
        geos["G1"] = dict(
            chs_info=chs_from_montage(canon, kind=kind), sfreq=sf, n_times=nt
        )
        geos["G4"] = dict(
            chs_info=chs_from_montage(list(canon[::-1]), kind=kind),
            sfreq=sf,
            n_times=nt,
        )
    else:
        n = spec.get("n_chans_fixed", 16)
        geos["G1"] = dict(chs_info=chs_coords_only(n, kind), sfreq=sf, n_times=nt)
    bio = _montage("biosemi64").ch_names
    geos["G2"] = dict(
        chs_info=chs_from_montage(bio, "biosemi64", kind=kind), sfreq=sf, n_times=nt
    )
    geos["G3"] = dict(chs_info=chs_coords_only(8, kind), sfreq=sf, n_times=nt)
    geos["G3b"] = dict(chs_info=chs_names_no_loc(TEN_TWENTY[:8]), sfreq=sf, n_times=nt)
    base = geos["G1"]["chs_info"]
    if sf:
        geos["G5a"] = dict(chs_info=base, sfreq=sf, n_times=int(sf))
        geos["G5b"] = dict(chs_info=base, sfreq=sf, n_times=int(sf * 30))
        geos["G5c"] = dict(chs_info=base, sfreq=sf, n_times=nt + 37)
    return geos


def expected(spec, gname, gkw, strategy="native"):
    """Return 'ok' or 'raise' from the declared contract (native path)."""
    n_ch = len(gkw["chs_info"])
    has_loc = any(np.any(ch["loc"][:3]) for ch in gkw["chs_info"])
    # length
    min_nt = spec.get("min_n_times")
    if min_nt is not None and gkw["n_times"] < min_nt:
        return "raise"
    if strategy != "native":  # the layer serves the montage
        return "ok"
    # channels
    iface = interface(spec)
    if iface == "slots" and n_ch > spec["max_n_chans"]:
        return "raise"
    if iface == "positions" and spec.get("coords_checked") and not has_loc:
        return "raise"
    if (
        iface == "ids"
        and spec.get("names_required_at_forward")
        and gname in ("G2", "G3", "G3b")
    ):
        return "raise"
    return "ok"


def _cases():
    for name, spec in COMPAT.items():
        for gname, gkw in geometries(spec).items():
            yield pytest.param(name, gname, gkw, id=f"{name}-{gname}")


@pytest.mark.parametrize("name,gname,gkw", list(_cases()))
def test_geometry_contract(name, gname, gkw):
    spec = COMPAT[name]
    strategy = STRATEGY_CELLS.get((name, gname), "native")
    kw = dict(n_outputs=2, **gkw, **spec.get("kwargs", {}))
    if strategy != "native":
        kw["channel_strategy"] = strategy

    def build_and_forward():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = spec["cls"](**kw).eval()
            with torch.no_grad():
                return model(torch.randn(1, len(gkw["chs_info"]), gkw["n_times"]))

    if expected(spec, gname, gkw, strategy) == "raise":
        with pytest.raises((ValueError, RuntimeError)):
            build_and_forward()
    else:
        y = build_and_forward()
        assert torch.is_tensor(y) and y.shape[0] == 1 and torch.isfinite(y).all()


def test_interfaces_come_from_the_channel_contract():
    got = {name: interface(spec) for name, spec in COMPAT.items()}
    assert got == {
        "Labram": "ids",
        "EEGPT": "ids",
        "BENDR": "montage",
        "BIOT": "montage",
        "CBraMod": "free",
        "CodeBrain": "montage",
        "EEGDINO": "slots",
        "LUNA": "positions",
        "REVE": "positions",
        "SignalJEPA": "ids",
        "STEEGFormer": "ids",
        "ZUNA": "positions",
        "Brant": "free",
        "BrainBERT": "free",
        "BaRISTA": "positions",
        "DIVER1": "positions",
        "MIRepNet": "montage",
        "MVPFormer": "free",
        "PopT": "positions",
    }


_POS_1005 = _montage("standard_1005").get_positions()["ch_pos"]


def _dense_fit_set(n_samples=300, seed=0):
    """Smooth synthetic fields on every standard_1005 site (fits ``wiener``)."""
    dense = [
        {"ch_name": n, "kind": "eeg", "loc": np.r_[p, np.zeros(9)]}
        for n, p in _POS_1005.items()
    ]
    P = np.array([ch["loc"][:3] for ch in dense])
    basis = np.c_[np.ones(len(P)), P, P**2, P[:, [0]] * P[:, [1]]]
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, basis.shape[1])) @ basis.T * 1e2
    return X + 0.1 * rng.standard_normal(X.shape), dense


def _strategy_cases():
    for name, spec in COMPAT.items():
        geos = geometries(spec)
        for strategy in STRATEGIES:
            marks = () if strategy in FAST_STRATEGIES else pytest.mark.slow
            for gname in CHANNEL_GEOMETRIES:
                if gname in geos:
                    yield pytest.param(
                        name,
                        strategy,
                        gname,
                        geos[gname],
                        id=f"{name}-{strategy}-{gname}",
                        marks=marks,
                    )


@pytest.mark.parametrize("name,strategy,gname,gkw", list(_strategy_cases()))
def test_channel_strategy_contract(name, strategy, gname, gkw):
    spec = COMPAT[name]
    kw = dict(n_outputs=2, channel_strategy=strategy, **gkw, **spec.get("kwargs", {}))

    def build_and_forward():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.manual_seed(0)
            model = spec["cls"](**kw).eval()
            if strategy == "wiener":
                model.channel_tokenizer.fit(*_dense_fit_set())
            with torch.no_grad():
                y = model(torch.randn(1, len(gkw["chs_info"]), gkw["n_times"]))
        return model, y["features"] if isinstance(y, dict) else y

    if strategy == "native":
        if expected(spec, gname, gkw) == "raise":
            with pytest.raises((ValueError, RuntimeError)):
                build_and_forward()
            return
        model, y = build_and_forward()
        # native keeps the released state dict: no channel-layer keys.
        assert not [k for k in model.state_dict() if k.startswith("channel_tokenizer")]
    else:
        try:
            model, y = build_and_forward()
        except ValueError as exc:
            assert re.search(DECLARED_ERRORS, str(exc)), (
                f"undeclared error: {exc}"
            )
            return
        assert model.get_config()["channel_strategy"] == strategy
    assert torch.is_tensor(y) and y.shape[0] == 1 and torch.isfinite(y).all()
