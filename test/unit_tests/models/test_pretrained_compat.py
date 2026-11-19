"""Geometry-compatibility contract for models with released weights.

Every model class that ships pretrained weights is built (random weights, no
download) on a grid of input geometries -- canonical montage, permuted order,
a 64-channel montage outside the 10-20 vocabulary, coordinates-only channels,
names without coordinates, short / long / non-divisible windows, all at the
checkpoint's sampling rate -- and must either forward or raise the *declared*
error.

The expected outcome of each cell is derived from the model's declared
channel strategy (``COMPAT`` below), not hard-coded per cell, so adding a
model means adding one entry. Cells the native path cannot serve run with a
``channel_strategy`` instead (``STRATEGY_CELLS``).
"""

from __future__ import annotations

import csv
import inspect
import json
import math
import re
import warnings
from pathlib import Path

import mne
import numpy as np
import pytest
import torch

from braindecode.models import (
    AXON,
    BBL,
    BENDR,
    BIOT,
    CORTEG,
    DIVER1,
    EEGDINO,
    EEGPT,
    LUNA,
    MAPA,
    MEGGPT,
    MEGXL,
    NICE,
    REVE,
    ZUNA,
    BaRISTA,
    BrainBERT,
    BrainOmni,
    BrainTokenizer,
    Brant,
    CBraMod,
    CodeBrain,
    DuIN,
    EpiNT,
    Guetschel2026,
    Labram,
    MIRepNet,
    MVPFormer,
    NeuroRVQ,
    PopulationTransformer,
    Pratibha2,
    SignalJEPA,
    SignalJEPA_Contextual,
    SignalJEPA_PostLocal,
    SignalJEPA_PreLocal,
    SleepFM,
    SleepFMStager,
    STEEGFormer,
    TFMTokenizer,
)
from braindecode.models.bendr import BENDR_CHANNEL_ORDER
from braindecode.models.biot import BIOT_CHANNEL_ORDER
from braindecode.models.eegpt import EEGPT_19_CHANNELS
from braindecode.models.labram import LABRAM_CHANNEL_ORDER
from braindecode.models.mirepnet import MIREPNET_CHANNEL_ORDER
from braindecode.models.neurorvq import NEURORVQ_CHANNELS
from braindecode.models.util import models_dict, models_mandatory_parameters
from braindecode.util import resolve_montage_name
from test.unit_tests.models._fixtures import _get_signal_params

TEN_TWENTY = "Fp1 Fp2 F7 F3 Fz F4 F8 T7 C3 Cz C4 T8 P7 P3 Pz P4 P8 O1 O2".split()

# Known, harmless warnings from the grid itself; anything else surfaces.
pytestmark = [
    pytest.mark.filterwarnings(f"ignore:{msg}")
    for msg in (
        r"Time dimension \(\d+\) is not divisible by patch_size",
        "Montage name .* is deprecated",
        "A window was not provided",
        "enable_nested_tensor is True",
        r"`torch.nn.utils.weight_norm` is deprecated",
    )
]


def _montage(name):
    return mne.channels.make_standard_montage(resolve_montage_name(name))


def chs_from_montage(names, montage="standard_1005", kind="eeg"):
    montage = _montage(montage)  # in the head frame, as raw.set_montage gives
    montage.apply_trans(mne.channels.compute_native_head_t(montage))
    pos = montage.get_positions()["ch_pos"]
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
# identifies channels (names / coords / fixed_order / index_slots / agnostic /
# contact_labels = sEEG labels ending in a contact number),
# ``min_n_times`` = smallest accepted window in samples, ``n_times_multiple`` =
# required divisor of the window, ``vocab`` = the only accepted channel names.
# ``windows`` = (short, long) window lengths, default (1 s, 30 s); ``skip`` =
# geometries not built. Large-token models (sEEG/MEG scale) use a short
# ``n_times``, ``windows`` and ``skip=("G2",)`` to keep the cells small.
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
    # Not keyed "REVE": test/conftest.py marks that literal as network-only.
    # The position bank comes from a local file (``_reve_position_bank``).
    "REVE-local-bank": dict(
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
        short_match="Kernel size",
    ),
    # Downstream heads that load the SignalJEPA checkpoint (strict=False).
    **{
        cls.__name__: dict(
            cls=cls,
            sfreq=128,
            n_times=2048,
            canon=TEN_TWENTY,
            channels="names",
            min_n_times=160,
            short_match="Kernel size",
        )
        for cls in (SignalJEPA_Contextual, SignalJEPA_PostLocal, SignalJEPA_PreLocal)
    },
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
        channels="coords",  # names without a loc take their 10-05 position
    ),
    "Brant": dict(
        cls=Brant,
        sfreq=250,
        n_times=1500,
        canon=TEN_TWENTY,
        channels="agnostic",
        kwargs=dict(patch_size=250),
    ),
    # Weights on GitHub (RunKZhang/EpiNT), not the Hub; pretrained at 256 and 1024 Hz.
    "EpiNT": dict(
        cls=EpiNT,
        sfreq=256,
        n_times=3072,
        canon=TEN_TWENTY,
        channels="agnostic",
    ),
    "BrainBERT": dict(
        cls=BrainBERT,
        sfreq=2048,
        n_times=4096,
        canon=["E1"],
        channels="agnostic",
    ),
    "DuIN": dict(
        cls=DuIN,
        sfreq=1000,
        n_times=3000,
        canon=None,
        kind="seeg",
        channels="fixed_order_unchecked",
        # on_non_divisible="pad": any window (the last patch zero-padded)
        windows=(1000, 10000),
    ),
    "BaRISTA": dict(
        cls=BaRISTA,
        sfreq=2048,
        n_times=2048,
        canon=None,
        kind="seeg",
        channels="coords",
        coords_checked=True,
    ),
    "DIVER1": dict(
        cls=DIVER1,
        sfreq=500,
        n_times=1000,
        canon=TEN_TWENTY,
        channels="coords",
        coords_checked=False,
    ),
    "AXON": dict(
        cls=AXON,
        sfreq=200,
        n_times=800,
        canon=TEN_TWENTY,
        channels="coords",
        coords_checked=False,
    ),
    # Positions by channel name; loc is not read.
    "Pratibha2": dict(
        cls=Pratibha2,
        sfreq=200,
        n_times=1000,
        canon=TEN_TWENTY,
        channels="names",
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
    # Windows shorter than one patch are zero-padded to one patch ("pad").
    "SleepFM": dict(
        cls=SleepFM,
        sfreq=128,
        n_times=3840,
        canon=None,
        channels="agnostic",
    ),
    "SleepFMStager": dict(
        cls=SleepFMStager,
        sfreq=128,
        n_times=3840,
        canon=None,
        channels="agnostic",
    ),
    "TFMTokenizer": dict(
        cls=TFMTokenizer,
        sfreq=200,
        n_times=1000,
        canon=TEN_TWENTY,
        channels="agnostic",
    ),
    "NeuroRVQ": dict(
        cls=NeuroRVQ,
        sfreq=200,
        n_times=200,
        canon=TEN_TWENTY,
        channels="names",
        vocab=NEURORVQ_CHANNELS,
        windows=(100, 400),
    ),
    "MAPA": dict(
        cls=MAPA,
        sfreq=2048,
        n_times=512,
        canon=None,
        kind="seeg",
        channels="contact_labels",
        min_n_times=448,
        windows=(256, 1024),
        skip=("G2",),
    ),
    "CORTEG": dict(
        cls=CORTEG,
        sfreq=128,
        n_times=128,
        canon=None,
        kind="ecog",
        channels="coords",
        coords_checked=True,
        windows=(128, 1024),
    ),
    "BrainOmni": dict(
        cls=BrainOmni,
        sfreq=256,
        n_times=512,
        canon=TEN_TWENTY,
        channels="coords",
        coords_checked=True,
        windows=(256, 1024),
        skip=("G2",),
    ),
    "BrainTokenizer": dict(
        cls=BrainTokenizer,
        sfreq=256,
        n_times=512,
        canon=TEN_TWENTY,
        channels="coords",
        coords_checked=True,
        windows=(256, 1024),
        skip=("G2",),
    ),
    # random_projection=64 exercises the projection head with a light buffer:
    # 5000 components in float32 take 3.3 GB on the 64-channel cell (G2, 5
    # patches) and 6.4 GB on the 6000-sample one (G5b, 19 channels, 33 patches).
    "Guetschel2026": dict(
        cls=Guetschel2026,
        sfreq=200,
        n_times=1000,
        canon=TEN_TWENTY,
        channels="coords",
        coords_checked=True,
        min_n_times=200,
        short_match="n_times",
        kwargs=dict(random_projection=64),
    ),
    # 52 source-space parcels, the channel embedding indexed by position; any
    # count builds. Windows are padded to 100-sample segments (250 Hz).
    "MEGGPT": dict(
        cls=MEGGPT,
        sfreq=250,
        n_times=200,
        canon=None,
        n_chans_fixed=52,
        channels="fixed_order_unchecked",
        windows=(100, 1000),
    ),
    # Positions and coil orientations from chs_info, as BrainOmni; magnetometers
    # (kind "mag") because the checkpoint is MEG-only.
    "MEGXL": dict(
        cls=MEGXL,
        sfreq=50,
        n_times=600,
        canon=TEN_TWENTY,
        kind="mag",
        channels="coords",
        coords_checked=True,
        skip=("G2",),
    ),
    # MEG sensors in FIF order, zero-padded to the 306 of the release; no
    # geometry is read, so any count up to 306 builds (0.5 s windows at 250 Hz).
    "BBL": dict(
        cls=BBL,
        sfreq=250,
        n_times=125,
        canon=None,
        kind="mag",
        n_chans_fixed=306,
        channels="fixed_order_unchecked",
        windows=(125, 1000),
    ),
    # A spatial convolution over n_chans and a flatten sized by n_times: any
    # montage and window builds (the released weights: 63 channels, 250 samples).
    "NICE": dict(
        cls=NICE,
        sfreq=250,
        n_times=250,
        canon=TEN_TWENTY,
        channels="fixed_order_unchecked",
    ),
}

# LaBraM looks names up at forward: these montages go through the channel layer.
STRATEGY_CELLS = {("Labram", "G2"), ("Labram", "G3"), ("Labram", "G3b")}


@pytest.fixture(autouse=True)
def _reve_position_bank(tmp_path, monkeypatch):
    """Serve REVE's and Pratibha2's position banks from local files instead of the Hub."""
    pos = _montage("standard_1005").get_positions()["ch_pos"]
    bank = {name: [float(v) for v in xyz] for name, xyz in pos.items()}
    for ch in chs_coords_only(128):  # EGI-style E1..E128, as in the released banks
        bank[ch["ch_name"]] = [float(v) for v in ch["loc"][:3]]
    for fname in ("reve_positions.json", "pratibha2_positions.json"):
        (tmp_path / fname).write_text(json.dumps(bank))
    monkeypatch.setenv("REVE_POSITIONS_PATH", str(tmp_path))


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
        short, long = spec.get("windows", (int(sf), int(sf * 30)))
        geos["G5a"] = dict(chs_info=base, sfreq=sf, n_times=short)
        geos["G5b"] = dict(chs_info=base, sfreq=sf, n_times=long)
        geos["G5c"] = dict(chs_info=base, sfreq=sf, n_times=nt + 37)
    for gname in spec.get("skip", ()):
        del geos[gname]
    return geos


def expected(spec, gname, gkw):
    """Return ``None`` (forwards) or the expected error message pattern."""
    n_ch, n_times = len(gkw["chs_info"]), gkw["n_times"]
    names = [ch["ch_name"].lower() for ch in gkw["chs_info"]]
    has_loc = any(np.any(ch["loc"][:3]) for ch in gkw["chs_info"])
    # length
    if n_times < spec.get("min_n_times", 0):
        return spec.get("short_match", "n_times|window")
    if n_times % spec.get("n_times_multiple", 1):
        return "divisible"
    # channels
    strategy = spec["channels"]
    if strategy == "index_slots" and n_ch > spec["max_n_chans"]:
        return "n_channel_embeddings"
    if strategy == "coords" and spec.get("coords_checked") and not has_loc:
        return "locations|positions"
    if strategy == "contact_labels" and not all(re.search(r"\d$", n) for n in names):
        return "trailing number"
    if "vocab" in spec and not set(names) <= set(spec["vocab"]):
        return "channel name"
    if (
        strategy == "names"
        and spec.get("names_required_at_forward")
        and gname in ("G2", "G3", "G3b")
    ):
        return "channel"
    return None


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
        model = spec["cls"](**kw).eval()
        with torch.no_grad():
            return model(torch.randn(1, len(gkw["chs_info"]), gkw["n_times"]))

    match = expected(spec, gname, gkw)
    if match is not None and (name, gname) not in STRATEGY_CELLS:
        with pytest.raises((ValueError, RuntimeError), match=match):
            build_and_forward()
    else:
        y = build_and_forward()
        assert torch.is_tensor(y) and y.shape[0] == 1 and torch.isfinite(y).all()


# ``native`` only, refused with a ValueError naming why: polysomnography
# grouped by modality (SleepFM), MEG sensors or source parcels (BBL, MEGGPT,
# MEGXL), or a second per-channel input the layer would not map (CORTEG's x_hi).
NATIVE_ONLY = {"SleepFM", "SleepFMStager", "BBL", "MEGGPT", "MEGXL", "CORTEG"}

# No ``channel_strategy`` argument (not a foundation model).
NO_STRATEGY = {"NICE"}


# BIOT's canonical input is bipolar; under a strategy it takes electrodes.
# SignalJEPA (and its heads) target its 62 pre-training channels (test_channels.py).
@pytest.mark.parametrize(
    "name",
    [
        n
        for n in COMPAT
        if n != "BIOT"
        and not n.startswith("SignalJEPA")
        and n not in NATIVE_ONLY | NO_STRATEGY
    ],
)
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


@pytest.mark.parametrize(
    "name", [n for n in COMPAT if n not in NATIVE_ONLY | NO_STRATEGY]
)
def test_channel_strategy_round_trip(name):
    """Under a strategy, ``model(x)`` equals ``model.forward(x)`` and the
    config keeps the input montage, so ``from_config`` rebuilds the same model."""
    spec = COMPAT[name]
    names = TEN_TWENTY[:8]
    if spec["channels"] == "contact_labels":  # MAPA: names ending in a number
        names = [n for n in TEN_TWENTY if n[-1].isdigit()][:8]
    chs = chs_from_montage(names, kind=spec.get("kind", "eeg"))
    kw = dict(n_outputs=2, chs_info=chs, sfreq=spec["sfreq"], n_times=spec["n_times"])
    x = torch.randn(1, len(chs), spec["n_times"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = spec["cls"](**kw, **spec.get("kwargs", {}), channel_strategy="spline")
        model.eval()
        config = json.loads(json.dumps(model.get_config()))
        rebuilt = spec["cls"].from_config(config).eval()
        rebuilt.load_state_dict(model.state_dict())
        with torch.no_grad():
            y = model(x)
            torch.testing.assert_close(model.forward(x), y)
            torch.testing.assert_close(rebuilt(x), y)
    assert [c["ch_name"] for c in rebuilt.channel_layer.chs_info] == names
    assert rebuilt.get_config()["chs_info"] == config["chs_info"]


@pytest.mark.parametrize("name", sorted(NATIVE_ONLY))
def test_native_only_models_refuse_a_strategy(name):
    spec = COMPAT[name]
    kw = dict(n_outputs=2, **geometries(spec)["G1"])
    with pytest.raises(ValueError, match="native"):
        spec["cls"](**kw, channel_strategy="spline")


# Models that map EEG montages but not every input they take.
@pytest.mark.parametrize(
    "cls,kw",
    [
        (NeuroRVQ, dict(modality="ecg", chs_info=chs_names_no_loc(["i", "ii"]))),
        (MAPA, dict(normalization="session", chs_info=chs_coords_only(4, "seeg"))),
        (BrainOmni, dict(chs_info=chs_coords_only(4) + chs_coords_only(2, "mag"))),
        (BrainTokenizer, dict(chs_info=chs_coords_only(2, "grad"))),
    ],
    ids=["NeuroRVQ-ecg", "MAPA-session", "BrainOmni-meg", "BrainTokenizer-meg"],
)
def test_strategy_refused_for_inputs_the_layer_cannot_map(cls, kw):
    with pytest.raises(ValueError, match="native"):
        cls(n_outputs=2, n_times=512, **kw, channel_strategy="spline")


def test_mapa_refuses_sensor_indices_under_a_strategy():
    chs = chs_coords_only(4, "seeg")
    model = MAPA(n_outputs=2, n_times=512, chs_info=chs, channel_strategy="spline")
    x = torch.randn(1, 4, 512)
    with pytest.raises(ValueError, match="sensor_indices"):
        model(x, sensor_indices=model.default_sensor_indices)


_SUMMARY = Path(__file__).parents[3] / "braindecode" / "models" / "summary.csv"
FOUNDATION = [
    row["Model"]
    for row in csv.DictReader(_SUMMARY.read_text(encoding="utf-8-sig").splitlines())
    if "Foundation" in row["Categorization"]
]
FIXTURES = {name: sp for name, _, sp in models_mandatory_parameters}
# Foundation models outside COMPAT that refuse a strategy: fNIRS HbO/HbR.
STRATEGY_REFUSED = NATIVE_ONLY | {"GeoNIRS", "JungMAE", "MtReNIRS"}


def _output_tensors(y):
    if isinstance(y, dict):
        return [v for v in y.values() if torch.is_tensor(v)]
    return list(y) if isinstance(y, (tuple, list)) else [y]


@pytest.mark.parametrize("name", FOUNDATION)
def test_foundation_model_maps_each_sample_under_a_strategy(name):
    """Every foundation model takes ``channel_strategy``: it refuses with a
    ValueError naming ``"native"``, or a zero-padded batch of three recordings
    with different montages (``ch_mask`` and one ``chs_info`` per sample) gives
    each recording its output alone."""
    cls = models_dict[name]
    sp = _get_signal_params(FIXTURES[name])
    kw = {k: v for k, v in sp.items() if k not in ("chs_info", "n_chans")}
    chs = sp["chs_info"]
    if name in STRATEGY_REFUSED:
        with pytest.raises(ValueError, match="native"):
            cls(**kw, chs_info=chs, channel_strategy="idw")
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zero-filled targets, sfreq checks
        torch.manual_seed(0)
        model = cls(**kw, chs_info=chs, channel_strategy="idw").eval()
        n = len(chs)  # all channels, all but the first, every other one
        subsets = [list(range(n)), list(range(1, n)), list(range(0, n, 2))]
        mask = torch.zeros(len(subsets), n, dtype=torch.bool)
        for i, s in enumerate(subsets):
            mask[i, s] = True
        x = torch.randn(len(subsets), n, kw.get("n_times") or model.n_times)
        montages = [[chs[j] for j in s] for s in subsets]
        with torch.no_grad():
            padded = x.masked_fill(~mask[..., None], 0.0)
            got = _output_tensors(model(padded, ch_mask=mask, chs_info=montages))
            alone = [
                _output_tensors(model(x[i : i + 1, s], chs_info=montages[i]))
                for i, s in enumerate(subsets)
            ]
    for g, w in zip(got, [torch.cat(t) for t in zip(*alone)], strict=True):
        # batch-size rounding grows with the output scale (CSBrain: ~200)
        tol = 1e-4 * max(1.0, w.abs().max().item())
        torch.testing.assert_close(g, w, rtol=1e-4, atol=tol)


# Classes with released weights that are outside this EEG/iEEG contract.
EXCLUDED = {
    "NeuroPose": "sEMG hand-pose regression, no EEG geometry",
    "VEMG2Pose": "sEMG hand-pose regression, no EEG geometry",
    "TinyMyo": "sEMG channel slots, no EEG geometry",
    "NeuroRVQTokenizer": "returns (target, reconstruction); NeuroRVQ's channel contract",
    "MtReNIRS": "fNIRS HbO/HbR pairs with optode positions, no EEG geometry",
}
_SHIPS_WEIGHTS = re.compile(
    r"hf_hub_download|from_pretrained\(|huggingface\.co/|Hugging Face Hub"
)


def test_compat_covers_every_pretrained_model():
    """A model class whose source points to released weights has a COMPAT entry."""
    covered = {spec["cls"].__name__ for spec in COMPAT.values()}
    shipped = {
        name
        for name, cls in models_dict.items()
        if _SHIPS_WEIGHTS.search(inspect.getsource(cls))
    }
    assert set(EXCLUDED) <= shipped
    assert shipped - covered - set(EXCLUDED) == set()


@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.bfloat16], ids=["float64", "bfloat16"]
)
@pytest.mark.parametrize("name", list(COMPAT))
def test_forward_in_dtype(name, dtype):
    """``model.to(dtype)`` forwards on CPU and returns finite ``dtype`` outputs."""
    spec = COMPAT[name]
    gkw = geometries(spec)["G1"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = spec["cls"](n_outputs=2, **gkw, **spec.get("kwargs", {}))
        model = model.to(dtype).eval()
        x = torch.randn(2, len(gkw["chs_info"]), gkw["n_times"], dtype=dtype)
        with torch.no_grad():
            y = model(x)
    assert y.dtype == dtype and torch.isfinite(y).all()
