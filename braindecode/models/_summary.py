# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3

"""Deterministic generator and drift check for ``braindecode/models/summary.csv``."""

from __future__ import annotations

import argparse
import csv
import difflib
import functools
import inspect
import io
import re
import sys
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from braindecode.models.base import EEGModuleMixin

SUMMARY_CSV_PATH = Path(__file__).resolve().with_name("summary.csv")

SUMMARY_COLUMNS: tuple[str, ...] = (
    "Model",
    "Application",
    "Type",
    "Sampling Frequency (Hz)",
    "Hyperparameters",
    "#Parameters",
    "get_#Parameters",
    "Categorization",
    "Modality",
    "Time axis",
    "Channel axis",
    "Output",
    "Cropped decoding",
    "adapt_input",
    "reset_head",
    "return_features",
)

EDITORIAL_COLUMNS: tuple[str, ...] = (
    "Application",
    "Type",
    "Sampling Frequency (Hz)",
    "Modality",
)

_BADGE_TO_CATEGORY: dict[str, str] = {
    ":bdg-success:`Convolution`": "Convolution",
    ":bdg-secondary:`Recurrent`": "Recurrent",
    ":bdg-info:`Attention/Transformer`": "Attention/Transformer",
    ":bdg-primary:`Filterbank`": "Filterbank",
    ":bdg-warning:`Interpretability`": "Interpretability",
    ":bdg-danger:`Foundation Model`": "Foundation Model",
    ":bdg-dark-line:`Channel`": "Channel",
    ":bdg-light:`Graph Neural Network`": "Graph Neural Network",
}
_BADGE_RE = re.compile(r":bdg-[\w-]+:`[^`]+`")

_TOKENIZERS: frozenset[str] = frozenset(
    {"BrainTokenizer", "NeuroRVQTokenizer", "TFMTokenizer"}
)
_OUTPUT_B_OUT_TIME: frozenset[str] = frozenset(
    {"CareSleepNet", "DeepSOZ", "SeizureTransformer", "SleepFMStager"}
)
_OUTPUT_B_TIME_OUT: frozenset[str] = frozenset(
    {
        "DANCE",
        "EMG2QwertyNet",
        "MetaNeuromotorHand",
        "NeuroPose",
        "SensingDynamics",
        "VEMG2Pose",
    }
)

# Models whose __init__ (or lazy-head materialization) executes data-dependent
# host ops (.item(), _unique2, CPU-only tensor copies) that cannot run on meta.
_CPU_SUMMARY_MODELS: frozenset[str] = frozenset(
    {
        "BaRISTA",
        "CORTEG",
        "DGCNN",
        "EEGPT",
        "GCNsNet",
        "GeoNIRS",
        "Labram",
        "MAPA",
        "MetaNeuromotorHand",
        "NeuroRVQ",
        "NeuroRVQTokenizer",
        "SAMBA",
        "STEEGFormer",
        "TCFormer",
        "ZUNA",
    }
)

_ALL_MODELS_FILES: frozenset[str] = frozenset({"__init__", "_summary", "base", "util"})
_TEN_TWENTY_19: tuple[str, ...] = (
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "O2",
)

# Canonical constructor kwargs for summary.csv parameter counts (using released
# model depths rather than CI-shrunk test depths). A sentinel tuple
# ("_chs", kind, n_chans) builds deterministic channel metadata of length n_chans
# and renders cleanly as ``chs_info=chs_info`` in ``get_#Parameters``.
_SUMMARY_KWARGS: dict[str, dict[str, Any]] = {
    "AXON": {
        "n_chans": 22,
        "chs_info": ("_chs", "eeg", 22),
        "n_outputs": 4,
        "n_times": 1000,
    },
    "BaRISTA": {
        "n_chans": 22,
        "chs_info": ("_chs", "seeg", 22),
        "n_outputs": 4,
        "n_times": 6144,
        "sfreq": 2048,
    },
    "BDTCN": {
        "n_chans": 21,
        "n_outputs": 2,
        "n_times": 6000,
        "n_blocks": 5,
        "n_filters": 55,
        "kernel_size": 16,
    },
    "BIOT": {"n_chans": 2, "n_outputs": 5, "n_times": 6000, "sfreq": 200},
    "BrainBERT": {"n_chans": 2, "n_outputs": 5, "n_times": 2048},
    "CBraMod": {"n_chans": 19, "n_outputs": 2, "n_times": 1000},
    "CodeBrain": {"n_chans": 19, "n_outputs": 2, "n_times": 6000},
    "CSBrain": {"n_chans": 19, "n_outputs": 2, "n_times": 1000},
    "ContraWR": {
        "n_chans": 2,
        "n_outputs": 5,
        "n_times": 3750,
        "emb_size": 256,
        "sfreq": 125,
    },
    "CTNet": {
        "n_chans": 22,
        "n_outputs": 4,
        "n_times": 1000,
        "n_filters_time": 8,
        "kernel_size": 16,
        "num_heads": 2,
        "embed_dim": 16,
    },
    "DeepSleepNet": {"n_chans": 1, "n_outputs": 5, "n_times": 7680, "sfreq": 256},
    "EEGCLIP": {"n_chans": 21, "n_outputs": 64, "n_times": 1000},
    "EEGPT": {
        "n_chans": 58,
        "chs_info": ("_chs", "eeg", 58),
        "n_outputs": 4,
        "n_times": 1024,
    },
    "EEGInceptionERP": {"n_chans": 8, "n_outputs": 2, "n_times": 128, "sfreq": 128},
    "EEGInceptionMI": {
        "n_chans": 22,
        "n_outputs": 4,
        "n_times": 1000,
        "sfreq": 250,
        "n_convs": 5,
        "n_filters": 12,
    },
    "EEGITNet": {"n_chans": 22, "n_outputs": 4, "n_times": 500},
    "EEGNet": {"n_chans": 22, "n_outputs": 4, "n_times": 512},
    "EEGNeX": {"n_chans": 22, "n_outputs": 4, "n_times": 500},
    "EEGMiner": {"n_chans": 62, "n_outputs": 2, "n_times": 2560, "sfreq": 128},
    "EEGSimpleConv": {"n_chans": 22, "n_outputs": 4, "n_times": 320, "sfreq": 80},
    "EEGTCNet": {"n_chans": 22, "n_outputs": 4, "n_times": 1000, "kern_length": 32},
    "Labram": {
        "n_chans": 128,
        "chs_info": ("_chs", "labram", 128),
        "n_outputs": 4,
        "n_times": 1000,
    },
    "SCCNet": {"n_chans": 22, "n_outputs": 4, "n_times": 1000, "sfreq": 125},
    "SignalJEPA": {
        "n_chans": 62,
        "chs_info": ("_chs", "eeg", 62),
        "n_times": 512,
    },
    "SignalJEPA_Contextual": {
        "n_chans": 62,
        "chs_info": ("_chs", "eeg", 62),
        "n_outputs": 2,
        "n_times": 536,
        "sfreq": 128,
    },
    "SignalJEPA_PostLocal": {
        "n_chans": 62,
        "n_outputs": 2,
        "n_times": 536,
        "sfreq": 128,
    },
    "SignalJEPA_PreLocal": {
        "n_chans": 62,
        "n_outputs": 2,
        "n_times": 536,
        "sfreq": 128,
    },
    "SincShallowNet": {"n_chans": 22, "n_outputs": 4, "n_times": 1000, "sfreq": 250},
    "ShallowFBCSPNet": {"n_chans": 22, "n_outputs": 4, "n_times": 1000, "sfreq": 250},
    "SleepFM": {"n_chans": 4, "n_outputs": 5, "n_times": 3840, "sfreq": 128},
    "SleepFMStager": {"n_chans": 4, "n_outputs": 5, "n_times": 3840, "sfreq": 128},
    "SleepStagerBlanco2020": {
        "n_chans": 2,
        "n_outputs": 5,
        "n_times": 3000,
        "sfreq": 100,
    },
    "SleepStagerChambon2018": {
        "n_chans": 2,
        "n_outputs": 5,
        "n_times": 3840,
        "sfreq": 128,
    },
    "AttnSleep": {"n_chans": 1, "n_outputs": 5, "n_times": 3000, "sfreq": 100},
    "SPARCNet": {"n_chans": 16, "n_outputs": 6, "n_times": 2000, "sfreq": 200},
    "SyncNet": {"n_chans": 62, "n_outputs": 3, "n_times": 5120, "sfreq": 256},
    "TSception": {"n_chans": 62, "n_outputs": 3, "n_times": 5120, "sfreq": 256},
    "USleep": {"n_chans": 2, "n_outputs": 5, "n_times": 3000, "sfreq": 100},
    "CareSleepNet": {"n_chans": 8, "n_outputs": 5, "n_times": 60000, "sfreq": 100},
    "FBCNet": {"n_chans": 22, "n_outputs": 4, "n_times": 1000, "sfreq": 250},
    "FBMSNet": {"n_chans": 22, "n_outputs": 4, "n_times": 1000, "sfreq": 250},
    "MetaNeuromotorHand": {
        "n_chans": 16,
        "n_outputs": 100,
        "n_times": 32000,
        "sfreq": 2000,
    },
    "EMG2QwertyNet": {"n_chans": 32, "n_outputs": 99, "n_times": 8000, "sfreq": 2000},
    "TCCNet": {"n_chans": 10, "n_outputs": 53},
    "TinyMyo": {"n_chans": 16, "n_outputs": 53, "n_times": 1000},
    "VEMG2Pose": {"n_chans": 16, "n_outputs": 20, "n_times": 11790, "sfreq": 2000},
    "NeuroPose": {"n_chans": 16, "n_outputs": 20, "n_times": 10000, "sfreq": 2000},
    "SensingDynamics": {"n_chans": 16, "n_outputs": 20, "n_times": 480, "sfreq": 2000},
    "FBLightConvNet": {"n_chans": 22, "n_outputs": 4, "n_times": 1000, "sfreq": 250},
    "IFNet": {"n_chans": 22, "n_outputs": 4, "n_times": 1000, "sfreq": 250},
    "BrainModule": {"n_chans": 64, "n_outputs": 29, "n_times": 160, "sfreq": 1000},
    "Brant": {"n_chans": 16, "n_outputs": 2, "n_times": 3000, "sfreq": 250},
    "EpiNT": {"n_chans": 17, "n_outputs": 2, "n_times": 3072, "sfreq": 256},
    "PBT": {"n_chans": 22, "n_outputs": 4, "n_times": 1000, "sfreq": 250},
    "PopulationTransformer": {"n_chans": 8, "n_outputs": 2, "n_times": 768},
    "Pratibha2": {
        "n_chans": 22,
        "chs_info": ("_chs", "eeg", 22),
        "n_outputs": 4,
        "n_times": 1000,
    },
    "BENDR": {"n_chans": 22, "n_outputs": 4, "n_times": 1000},
    "LUNA": {"n_chans": 22, "n_outputs": 4, "n_times": 512, "sfreq": 128},
    "MAPA": {
        "n_chans": 22,
        "chs_info": ("_chs", "seeg", 22),
        "n_outputs": 4,
        "n_times": 2048,
        "sfreq": 2048,
    },
    "MEGGPT": {"n_chans": 52, "n_outputs": 4, "n_times": 1000},
    "REVE": {
        "n_chans": 19,
        "chs_info": ("_chs", "eeg", 19),
        "n_outputs": 4,
        "n_times": 1000,
    },
    "DGCNN": {
        "n_chans": 62,
        "chs_info": ("_chs", "eeg", 62),
        "n_outputs": 4,
        "n_times": 128,
    },
    "DIVER1": {
        "n_chans": 22,
        "chs_info": ("_chs", "seeg", 22),
        "n_outputs": 4,
        "n_times": 1000,
        "sfreq": 500,
    },
    "DuIN": {"n_chans": 10, "n_outputs": 61, "n_times": 3000, "sfreq": 1000},
    "BrainOmni": {
        "n_chans": 22,
        "chs_info": ("_chs", "eeg", 22),
        "n_outputs": 4,
        "n_times": 1024,
        "sfreq": 256.0,
    },
    "MEGXL": {
        "n_chans": 306,
        "chs_info": ("_chs", "meg", 306),
        "n_outputs": 4,
        "n_times": 7500,
        "sfreq": 50.0,
    },
    "BrainTokenizer": {
        "n_chans": 22,
        "chs_info": ("_chs", "eeg", 22),
        "n_times": 1024,
        "sfreq": 256.0,
    },
    "TFMTokenizer": {"n_chans": 3, "n_outputs": 2, "n_times": 1000, "sfreq": 200},
    "EEGDINO": {"n_chans": 19, "n_outputs": 4, "n_times": 1000},
    "DANCE": {
        "n_chans": 19,
        "chs_info": ("_chs", "eeg", 19),
        "n_outputs": 4,
        "n_times": 6400,
        "sfreq": 200,
    },
    "CORTEG": {
        "n_chans": 16,
        "chs_info": ("_chs", "ecog", 16),
        "n_outputs": 4,
        "n_times": 128,
    },
    "MVPFormer": {"n_chans": 22, "n_outputs": 4, "n_times": 2000, "sfreq": 100},
    "ZUNA": {
        "n_chans": 19,
        "chs_info": ("_chs", "eeg", 19),
        "n_outputs": 4,
        "n_times": 1280,
    },
    "MIRepNet": {"n_chans": 45, "n_outputs": 4, "n_times": 1000},
    "NeuroRVQ": {"n_chans": 64, "n_outputs": 4, "n_times": 800, "sfreq": 200},
    "SeizureTransformer": {"n_chans": 19, "n_outputs": 1, "n_times": 15360},
    "DeepSOZ": {"n_chans": 19, "n_outputs": 2, "n_times": 153600},
    "NeuroRVQTokenizer": {
        "n_chans": 3,
        "chs_info": ("_chs", "eeg", 3),
        "n_outputs": 2,
        "n_times": 200,
        "sfreq": 200,
    },
    "Guetschel2026": {
        "n_chans": 19,
        "chs_info": ("_chs", "eeg", 19),
        "n_outputs": 4,
        "n_times": 1000,
        "sfreq": 200,
    },
    "GCNsNet": {"n_chans": 64, "n_outputs": 4, "n_times": 640},
    "NICE": {"n_chans": 63, "n_outputs": 200, "n_times": 250},
    "FNIRSNet": {"n_chans": 72, "n_outputs": 2, "n_times": 30},
    "FNIRS4D": {"n_chans": 72, "n_outputs": 2, "n_times": 30},
    "FNIRST": {"n_chans": 72, "n_outputs": 2, "n_times": 200},
    "FNIRSPreT": {"n_chans": 72, "n_outputs": 2, "n_times": 200},
    "GeoNIRS": {
        "n_chans": 40,
        "chs_info": ("_chs", "fnirs", 40),
        "n_outputs": 2,
        "n_times": 150,
        "sfreq": 5.0,
    },
    "MtReNIRS": {
        "n_chans": 20,
        "chs_info": ("_chs", "fnirs", 20),
        "n_outputs": 2,
        "n_times": 100,
        "sfreq": 10.0,
    },
    "JungMAE": {"n_chans": 40, "n_outputs": 2, "n_times": 150, "sfreq": 5.0},
    "FNIRSCNN1D": {"n_chans": 40, "n_outputs": 2, "n_times": 60, "sfreq": 5.0},
    "SAMBA": {
        "n_chans": 94,
        "chs_info": ("_chs", "eeg", 94),
        "n_outputs": 4,
        "n_times": 256,
    },
}


@functools.lru_cache(maxsize=None)
def _build_chs_info(kind: str, n_chans: int) -> list[dict[str, Any]]:
    import numpy as np
    from mne.io.constants import FIFF

    from braindecode.models.util import (
        _draw_chs_info,
        _fixture_montage_chs,
        _fnirs_chs_info,
        _get_labram_chs_info,
    )

    if kind == "labram":
        return _get_labram_chs_info()
    if kind == "fnirs":
        return _fnirs_chs_info(n_chans // 2)
    if kind == "seeg":
        return [
            {
                "ch_name": f"LA{i + 1}",
                "kind": "seeg",
                "loc": np.array([0.004 * (i + 1), 0.02, 0.01, *([0.0] * 9)]),
            }
            for i in range(n_chans)
        ]
    if kind == "ecog":
        return [
            {
                "ch_name": f"G{i + 1}",
                "kind": "ecog",
                "loc": np.array([0.005 * (i + 1), 0.02, 0.03, *([0.0] * 9)]),
            }
            for i in range(n_chans)
        ]
    if kind == "meg":
        return [
            {
                "ch_name": f"MEG{i + 1:04d}",
                "kind": FIFF.FIFFV_MEG_CH,
                "coil_type": FIFF.FIFFV_COIL_VV_MAG_T1,
                "loc": np.array(
                    [0.08 * np.cos(i), 0.08 * np.sin(i), 0.05, *([0.0] * 8), 1.0]
                ),
            }
            for i in range(n_chans)
        ]
    if n_chans == 3:
        montage_map = dict(_fixture_montage_chs())
        return [
            {"ch_name": name, "kind": "eeg", "loc": np.array(montage_map[name])}
            for name in ("F3", "F4", "Cz")
        ]
    if n_chans <= len(_TEN_TWENTY_19):
        montage_map = dict(_fixture_montage_chs())
        return [
            {"ch_name": name, "kind": "eeg", "loc": np.array(montage_map[name])}
            for name in _TEN_TWENTY_19[:n_chans]
        ]
    return _draw_chs_info(n_chans, np.random.default_rng(n_chans))


def _fallback_kwargs(
    required_params: Sequence[str], signal_params: Any
) -> dict[str, Any]:
    from braindecode.models.util import _EEG_PARAMS

    raw = signal_params() if callable(signal_params) else (signal_params or {})
    defaults: dict[str, Any] = {
        "n_chans": 22,
        "n_outputs": 4,
        "n_times": 1000,
        "sfreq": 250,
    }
    for key in ("n_chans", "n_outputs", "n_times", "sfreq"):
        if key in raw:
            defaults[key] = raw[key]
    if "chs_info" in raw and "n_chans" not in raw:
        defaults["n_chans"] = len(raw["chs_info"])

    needed = set(required_params) | (
        set(raw) & set(_EEG_PARAMS) - {"input_window_seconds"}
    )
    out: dict[str, Any] = {}
    if "n_chans" in needed or "chs_info" in needed:
        out["n_chans"] = int(defaults["n_chans"])
    if "chs_info" in needed:
        out["chs_info"] = raw.get("chs_info") or ("_chs", "eeg", out["n_chans"])
    for key in ("n_outputs", "n_times", "sfreq"):
        if key in needed:
            out[key] = defaults[key]
    return out


def _resolve_kwargs(
    name: str, required_params: Sequence[str], signal_params: Any
) -> tuple[dict[str, Any], str, tuple[int, int]]:
    spec = dict(
        _SUMMARY_KWARGS.get(name) or _fallback_kwargs(required_params, signal_params)
    )
    init_kwargs: dict[str, Any] = {}
    parts: list[str] = []
    for key, val in spec.items():
        if isinstance(val, tuple) and len(val) == 3 and val[0] == "_chs":
            init_kwargs[key] = _build_chs_info(val[1], val[2])
            parts.append(f"{key}=chs_info")
        elif key == "chs_info":
            init_kwargs[key] = val
            parts.append("chs_info=chs_info")
        else:
            init_kwargs[key] = val
            parts.append(f"{key}={val!r}")
    n_chans = int(
        init_kwargs.get("n_chans")
        or (len(init_kwargs["chs_info"]) if "chs_info" in init_kwargs else 22)
    )
    n_times = int(init_kwargs.get("n_times", 1000))
    return init_kwargs, f"{name}({', '.join(parts)})", (n_chans, n_times)


def _parse_categorization(name: str, cls: type[EEGModuleMixin]) -> str:
    doc = cls.__doc__ or ""
    found: list[str] = []
    for badge in _BADGE_RE.findall(doc):
        category = _BADGE_TO_CATEGORY.get(badge)
        if category is not None and category not in found:
            found.append(category)
    if not found:
        raise ValueError(f"{name} has no categorization badges in its docstring.")
    return ",".join(found)


def _output_kind(name: str) -> str:
    if name in _TOKENIZERS:
        return "reconstruction"
    if name == "SignalJEPA":
        return "(batch, tokens, dim)"
    if name in _OUTPUT_B_OUT_TIME:
        return "(batch, outputs, time)"
    if name in _OUTPUT_B_TIME_OUT:
        return "(batch, time, outputs)"
    return "(batch, outputs)"


def _capabilities(
    name: str,
    cls: type[EEGModuleMixin],
    modality: str,
    model: Any = None,
) -> dict[str, str]:
    """The capability columns of a model; ``reset_head`` is read from the built
    ``model`` when one is given."""
    from braindecode.models.util import _modalities

    head = name not in _TOKENIZERS
    flags = {
        "Cropped decoding": cls._dense_conversion,
        # the rule adapt_input applies: one input of EEG electrodes
        "adapt_input": "EEG" in _modalities(modality) and len(cls._time_inputs) == 1,
        "reset_head": head and model is not None and model._can_reset_head(),
        # every model but a tokenizer; test_return_features checks each one
        "return_features": head,
    }
    return {col: "yes" if flag else "no" for col, flag in flags.items()}


def _count_parameters(
    name: str,
    cls: type[EEGModuleMixin],
    init_kwargs: dict[str, Any],
    probe_shape: tuple[int, int],
) -> tuple[int, Any]:
    import torch
    from torch import nn

    device = "cpu" if name in _CPU_SUMMARY_MODELS else "meta"
    with warnings.catch_warnings(), torch.device(device):
        warnings.simplefilter("ignore")
        model = cls(**init_kwargs)
        if any(isinstance(p, nn.UninitializedParameter) for p in model.parameters()):
            with torch.no_grad():
                model.eval()(torch.zeros(1, *probe_shape))
    return sum(p.numel() for p in model.parameters()), model


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.is_file():
        return []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        return [
            {k: (v or "").strip() for k, v in row.items() if k is not None}
            for row in reader
            if (row.get("Model") or "").strip()
        ]


def _serialize_csv(rows: Sequence[dict[str, str]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows({col: row.get(col, "") for col in SUMMARY_COLUMNS} for row in rows)
    return buf.getvalue()


def generate_summary_rows(
    names: Iterable[str] | None = None,
    *,
    csv_path: Path = SUMMARY_CSV_PATH,
) -> list[dict[str, str]]:
    """Compute canonical ``summary.csv`` rows for registered models."""
    from braindecode.models.util import models_dict, models_mandatory_parameters

    existing = {row["Model"]: row for row in _read_rows(csv_path)}
    mandatory = {name: (req, sp) for name, req, sp in models_mandatory_parameters}
    if names is None:
        # Keep the committed order; new models go last, in registry order.
        names = [n for n in existing if n in mandatory]
        names += [n for n in mandatory if n not in existing]

    generated: list[dict[str, str]] = []
    for name in names:
        cls = models_dict[name]
        if cls._time_axis is None or cls._channel_axis is None:
            raise ValueError(f"{name} must declare _time_axis and _channel_axis.")
        req, sp = mandatory[name]
        init_kwargs, expr, probe_shape = _resolve_kwargs(name, req, sp)
        editorial = dict(existing.get(name, {}))
        # An empty Sampling Frequency cell takes the constructor default of sfreq.
        sfreq = inspect.signature(cls.__init__).parameters["sfreq"].default
        if not editorial.get("Sampling Frequency (Hz)") and sfreq is not None:
            editorial["Sampling Frequency (Hz)"] = f"{sfreq:g}"
        for col in EDITORIAL_COLUMNS:
            if not editorial.get(col):
                raise ValueError(f"{name} is missing editorial column {col!r}.")
        n_params, model = _count_parameters(name, cls, init_kwargs, probe_shape)
        generated.append(
            {
                **{col: editorial[col] for col in EDITORIAL_COLUMNS},
                "Model": name,
                "Hyperparameters": ", ".join(req),
                "#Parameters": str(n_params),
                "get_#Parameters": expr,
                "Categorization": _parse_categorization(name, cls),
                "Time axis": cls._time_axis,
                "Channel axis": cls._channel_axis,
                "Output": _output_kind(name),
                **_capabilities(name, cls, editorial["Modality"], model),
            }
        )
    return generated


def check_summary_csv(csv_path: Path = SUMMARY_CSV_PATH) -> list[str]:
    """Diff lines between ``csv_path`` and a fresh generation, empty if equal."""
    text = csv_path.read_text(encoding="utf-8") if csv_path.is_file() else ""
    expected = _serialize_csv(generate_summary_rows(csv_path=csv_path))
    if text == expected:
        return []
    return list(
        difflib.unified_diff(
            text.splitlines(),
            expected.splitlines(),
            "summary.csv on disk",
            "generated",
            lineterm="",
        )
    )


def _models_for_files(files: Sequence[str]) -> list[str] | None:
    """Return model names affected by ``files``, ``[]`` if none, or ``None`` for all."""
    from braindecode import models as bd_models
    from braindecode.models.util import models_mandatory_parameters

    registered = {name for name, _, _ in models_mandatory_parameters}
    models_dir = SUMMARY_CSV_PATH.parent
    selected: list[str] = []
    for raw_file in files:
        path = Path(raw_file)
        if not path.is_absolute():
            repo_path = SUMMARY_CSV_PATH.parents[2] / path
            path = repo_path if repo_path.exists() else Path.cwd() / path
        path = path.resolve()
        if path.parent != models_dir or path.suffix != ".py":
            continue
        if path.stem in _ALL_MODELS_FILES:
            return None
        selected += [
            sym
            for sym, module in bd_models._LAZY.items()
            if module == path.stem and sym in registered and sym not in selected
        ]
    return selected


def update_summary_csv(
    files: Sequence[str] | None = None,
    *,
    csv_path: Path = SUMMARY_CSV_PATH,
) -> bool:
    """Regenerate all or touched model rows in ``csv_path``. Return True if modified."""
    targets = None if files is None else _models_for_files(files)
    if targets == []:
        return False
    rows = _read_rows(csv_path)
    if targets is not None and set(targets) <= {r["Model"] for r in rows}:
        fresh = {
            r["Model"]: r for r in generate_summary_rows(targets, csv_path=csv_path)
        }
        rows = [fresh.get(r["Model"], r) for r in rows]
    else:
        rows = generate_summary_rows(csv_path=csv_path)
    rendered = _serialize_csv(rows)
    if csv_path.is_file() and csv_path.read_text(encoding="utf-8") == rendered:
        return False
    csv_path.write_text(rendered, encoding="utf-8")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m braindecode.models._summary",
        description="Regenerate or check braindecode/models/summary.csv.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify summary.csv matches generated metadata without modifying disk.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Only update models defined in the given braindecode/models/*.py files.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=SUMMARY_CSV_PATH,
        help="Override path to summary.csv (for testing).",
    )
    args = parser.parse_args(argv)

    if args.check:
        diffs = check_summary_csv(args.csv)
        print("\n".join(diffs), file=sys.stderr, end="\n" if diffs else "")
        return 1 if diffs else 0

    update_summary_csv(args.files, csv_path=args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
