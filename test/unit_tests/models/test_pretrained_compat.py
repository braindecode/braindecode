"""Geometry-compatibility contract for models with released weights.

Every model class that ships pretrained weights is built (random weights, no
download) on a grid of input geometries -- canonical montage, permuted order,
a 64-channel montage outside the 10-20 vocabulary, coordinates-only channels,
names without coordinates, short / long / non-divisible windows, other
sampling rates -- and must either forward or raise the *declared* error.

The expected outcome of each cell is derived from the model's declared
channel strategy (``COMPAT`` below), not hard-coded per cell, so adding a
model means adding one entry. Cells the native path cannot serve run with a
``channel_strategy`` instead (``STRATEGY_CELLS``).
"""

from __future__ import annotations

import math
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


# Declared behaviour per pretrained class: ``channels`` = how the checkpoint
# identifies channels (names / coords / fixed_order / index_slots / agnostic),
# ``min_n_times`` = smallest accepted window in samples.
COMPAT = {
    "Labram": dict(
        cls=Labram,
        sfreq=200,
        n_times=800,
        canon=LABRAM_CHANNEL_ORDER,
        channels="names",
        names_required_at_forward=True,
    ),
    "EEGPT": dict(
        cls=EEGPT,
        sfreq=256,
        n_times=1024,
        canon=EEGPT_19_CHANNELS,
        channels="names",
    ),
    "BENDR": dict(
        cls=BENDR,
        sfreq=256,
        n_times=1024,
        canon=BENDR_CHANNEL_ORDER,
        channels="fixed_order",
    ),
    "BIOT": dict(
        cls=BIOT,
        sfreq=200,
        n_times=800,
        canon=BIOT_CHANNEL_ORDER,
        channels="fixed_order_unchecked",
    ),
    "CBraMod": dict(
        cls=CBraMod,
        sfreq=200,
        n_times=800,
        canon=TEN_TWENTY,
        channels="agnostic",
    ),
    "CodeBrain": dict(
        cls=CodeBrain,
        sfreq=200,
        n_times=800,
        canon=TEN_TWENTY,
        channels="fixed_order_unchecked",
    ),
    "EEGDINO": dict(
        cls=EEGDINO,
        sfreq=200,
        n_times=800,
        canon=TEN_TWENTY,
        channels="index_slots",
        max_n_chans=19,
    ),
    "LUNA": dict(
        cls=LUNA,
        sfreq=200,
        n_times=800,
        canon=TEN_TWENTY,
        channels="coords",
        coords_checked=False,
    ),
    "REVE": dict(
        cls=REVE,
        sfreq=200,
        n_times=800,
        canon=TEN_TWENTY,
        channels="coords",
        coords_checked=False,
    ),
    "SignalJEPA": dict(
        cls=SignalJEPA,
        sfreq=128,
        n_times=2048,
        canon=TEN_TWENTY,
        channels="names",
        min_n_times=160,
    ),
    "STEEGFormer": dict(
        cls=STEEGFormer,
        sfreq=100,
        n_times=400,
        canon=TEN_TWENTY,
        channels="names",
    ),
    "ZUNA": dict(
        cls=ZUNA,
        sfreq=256,
        n_times=1024,
        canon=TEN_TWENTY,
        channels="coords",
        coords_checked=True,
    ),
    "Brant": dict(
        cls=Brant,
        sfreq=250,
        n_times=1500,
        canon=TEN_TWENTY,
        channels="agnostic",
        kwargs=dict(patch_size=250),
    ),
    "BrainBERT": dict(
        cls=BrainBERT,
        sfreq=2048,
        n_times=4096,
        canon=["E1"],
        channels="agnostic",
        min_n_times=1401,
    ),
    "BaRISTA": dict(
        cls=BaRISTA,
        sfreq=2048,
        n_times=2048,
        canon=None,
        kind="seeg",
        channels="coords",
        coords_checked=True,
        min_n_times=512,
    ),
    "DIVER1": dict(
        cls=DIVER1,
        sfreq=500,
        n_times=1000,
        canon=TEN_TWENTY,
        channels="coords",
        coords_checked=False,
    ),
    "MIRepNet": dict(
        cls=MIRepNet,
        sfreq=250,
        n_times=1000,
        canon=MIREPNET_CHANNEL_ORDER,
        channels="fixed_order_unchecked",
    ),
    "MVPFormer": dict(
        cls=MVPFormer,
        sfreq=512,
        n_times=2560,
        canon=None,
        kind="seeg",
        channels="agnostic",
    ),
    "PopT": dict(
        cls=PopulationTransformer,
        sfreq=None,
        n_times=768,
        canon=None,
        kind="seeg",
        channels="coords",
        coords_checked=False,
    ),
}

# LaBraM looks names up at forward: these montages go through the channel layer.
STRATEGY_CELLS = {("Labram", "G2"), ("Labram", "G3"), ("Labram", "G3b")}


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


def expected(spec, gname, gkw):
    """Return 'ok' or 'raise' from the declared strategy."""
    n_ch = len(gkw["chs_info"])
    has_loc = any(np.any(ch["loc"][:3]) for ch in gkw["chs_info"])
    # length
    min_nt = spec.get("min_n_times")
    if min_nt is not None and gkw["n_times"] < min_nt:
        return "raise"
    # channels
    strategy = spec["channels"]
    if strategy == "index_slots" and n_ch > spec["max_n_chans"]:
        return "raise"
    if strategy == "coords" and spec.get("coords_checked") and not has_loc:
        return "raise"
    if (
        strategy == "names"
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
    kw = dict(n_outputs=2, **gkw, **spec.get("kwargs", {}))
    if (name, gname) in STRATEGY_CELLS:
        kw["channel_strategy"] = "spline"

    def build_and_forward():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = spec["cls"](**kw).eval()
            with torch.no_grad():
                return model(torch.randn(1, len(gkw["chs_info"]), gkw["n_times"]))

    if expected(spec, gname, gkw) == "raise" and (name, gname) not in STRATEGY_CELLS:
        with pytest.raises((ValueError, RuntimeError)):
            build_and_forward()
    else:
        y = build_and_forward()
        assert torch.is_tensor(y) and y.shape[0] == 1 and torch.isfinite(y).all()


# BIOT's canonical input is bipolar; under a strategy it takes electrodes.
@pytest.mark.parametrize("name", [n for n in COMPAT if n != "BIOT"])
def test_native_checkpoint_loads_under_a_strategy(name):
    """The native state dict loads strictly into the same model with a layer."""
    spec = COMPAT[name]
    kw = dict(n_outputs=2, **geometries(spec)["G1"], **spec.get("kwargs", {}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        native = spec["cls"](**kw).state_dict()
        spec["cls"](**kw, channel_strategy="exact").load_state_dict(native)


# One model per channel contract: canonical montage, ids, positions, slots, free.
@pytest.mark.parametrize(
    "name,strategy",
    [(n, "spline") for n in ["BENDR", "Labram", "LUNA", "EEGDINO", "CBraMod"]]
    + [("Labram", s) for s in ["wiener", "region", "latent"]],
)
def test_channel_strategy_smoke(name, strategy):
    """Eight 10-20 names without positions forward through a strategy."""
    spec = COMPAT[name]
    chs = chs_names_no_loc(TEN_TWENTY[:8])
    kw = dict(n_outputs=2, chs_info=chs, sfreq=spec["sfreq"], n_times=spec["n_times"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = spec["cls"](**kw, channel_strategy=strategy).eval()
        if strategy == "wiener":  # fitted on the backbone's own montage
            dense = model.chs_info
            model.channel_layer.fit(torch.randn(500, len(dense)), dense)
        with torch.no_grad():
            y = model(torch.randn(1, len(chs), spec["n_times"]))
    assert model.get_config()["channel_strategy"] == strategy
    assert y.shape[0] == 1 and torch.isfinite(y).all()
