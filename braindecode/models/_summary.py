# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3

"""Deterministic generator and drift check for ``braindecode/models/summary.csv``."""

from __future__ import annotations

import argparse
import contextlib
import csv
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
    {"CareSleepNet", "SeizureTransformer", "SleepFMStager"}
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
_INITIAL_CROPPED_MODELS: frozenset[str] = frozenset(
    {"Deep4Net", "EEGNet", "EMG2QwertyNet", "NeuroPose", "ShallowFBCSPNet"}
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
_NON_MODEL_LAZY_EXPORTS: frozenset[str] = frozenset(
    {
        "EEGModuleMixin",
        "HybridNet",
        "TCN",
        "TFMTokenizerOutput",
        "_init_models_dict",
        "build_model_config",
        "extract_channel_locations_from_chs_info",
        "models_mandatory_parameters",
        "positions_from_chs_info",
    }
)

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
    "ATCNet": {"n_chans": 22, "n_outputs": 4, "n_times": 1000},
    "AXON": {
        "n_chans": 22,
        "chs_info": ("_chs", "eeg", 22),
        "n_outputs": 4,
        "n_times": 1000,
    },
    "AttentionBaseNet": {"n_chans": 22, "n_outputs": 4, "n_times": 1000},
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
    "Deep4Net": {"n_chans": 22, "n_outputs": 4, "n_times": 1000},
    "DeepSleepNet": {"n_chans": 1, "n_outputs": 5, "n_times": 7680, "sfreq": 256},
    "EEGConformer": {"n_chans": 22, "n_outputs": 4, "n_times": 1000},
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
    "EEGSym": {
        "n_chans": 22,
        "chs_info": ("_chs", "eeg", 22),
        "n_outputs": 4,
        "n_times": 1000,
        "sfreq": 250,
    },
    "EEGMiner": {"n_chans": 62, "n_outputs": 2, "n_times": 2560, "sfreq": 128},
    "EEGSimpleConv": {"n_chans": 22, "n_outputs": 4, "n_times": 320, "sfreq": 80},
    "EEGTCNet": {"n_chans": 22, "n_outputs": 4, "n_times": 1000, "kern_length": 32},
    "Labram": {
        "n_chans": 128,
        "chs_info": ("_chs", "labram", 128),
        "n_outputs": 4,
        "n_times": 1000,
    },
    "MSVTNet": {"n_chans": 22, "n_outputs": 4, "n_times": 1000},
    "MSCFormer": {"n_chans": 22, "n_outputs": 4, "n_times": 1000},
    "TMSANet": {"n_chans": 22, "n_outputs": 4, "n_times": 1000},
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
    "TIDNet": {"n_chans": 22, "n_outputs": 4, "n_times": 1000},
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
    "SSTDPN": {"n_chans": 22, "n_outputs": 4, "n_times": 1000, "sfreq": 250},
    "BENDR": {"n_chans": 22, "n_outputs": 4, "n_times": 1000},
    "LUNA": {"n_chans": 22, "n_outputs": 4, "n_times": 512, "sfreq": 128},
    "MAPA": {
        "n_chans": 22,
        "chs_info": ("_chs", "seeg", 22),
        "n_outputs": 4,
        "n_times": 2048,
        "sfreq": 2048,
    },
    "MEDFormer": {"n_chans": 22, "n_outputs": 4, "n_times": 1000},
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
    "BBL": {"n_chans": 22, "n_outputs": 4, "n_times": 125, "sfreq": 250.0},
    "TCFormer": {"n_chans": 22, "n_outputs": 4, "n_times": 1000},
    "TFMTokenizer": {"n_chans": 3, "n_outputs": 2, "n_times": 1000, "sfreq": 200},
    "EEGDINO": {"n_chans": 19, "n_outputs": 4, "n_times": 1000},
    "DANCE": {
        "n_chans": 19,
        "chs_info": ("_chs", "eeg", 19),
        "n_outputs": 4,
        "n_times": 6400,
        "sfreq": 200,
    },
    "STEEGFormer": {"n_chans": 22, "n_outputs": 4, "n_times": 1000},
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
                    [
                        0.08 * np.cos(i),
                        0.08 * np.sin(i),
                        0.05,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ]
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


def _has_strided_layer(model: Any) -> bool:
    for module in model.modules():
        stride = getattr(module, "stride", None)
        if isinstance(stride, int) and stride > 1:
            return True
        if isinstance(stride, tuple) and any(s > 1 for s in stride):
            return True
    return False


def _cropped_capability(
    name: str,
    cls: type[EEGModuleMixin],
    model: Any,
    base_cls: type[EEGModuleMixin],
) -> str:
    if hasattr(base_cls, "_dense_conversion"):
        ok = (
            cls._time_axis != "fixed"
            and bool(getattr(cls, "_dense_conversion", True))
            and (cls._time_axis == "dense" or _has_strided_layer(model))
        )
        return "yes" if ok else "no"
    return "yes" if name in _INITIAL_CROPPED_MODELS else "no"


def _adapt_capability(cls: type[EEGModuleMixin], model: Any) -> str:
    if hasattr(model, "_can_adapt_input"):
        return "yes" if model._can_adapt_input() else "no"
    return "yes" if hasattr(cls, "adapt_input") else "no"


def _reset_head_capability(
    name: str,
    cls: type[EEGModuleMixin],
    model: Any,
    base_cls: type[EEGModuleMixin],
) -> str:
    if name in _TOKENIZERS:
        return "no"
    if hasattr(model, "_can_reset_head"):
        return "yes" if model._can_reset_head() else "no"
    return "yes" if cls.reset_head is not base_cls.reset_head else "no"


def _return_features_capability(
    name: str,
    cls: type[EEGModuleMixin],
    base_cls: type[EEGModuleMixin],
) -> str:
    if name in _TOKENIZERS:
        return "no"
    if "return_features" in inspect.signature(cls.forward).parameters:
        return "yes"
    if getattr(cls, "_native_features", False) or hasattr(base_cls, "_pool_embedding"):
        return "yes"
    return "no"


def _instantiate_and_measure(
    name: str,
    cls: type[EEGModuleMixin],
    base_cls: type[EEGModuleMixin],
    init_kwargs: dict[str, Any],
    probe_shape: tuple[int, int],
) -> tuple[int, str, str, str]:
    import torch
    from torch import nn

    use_cpu = name in _CPU_SUMMARY_MODELS
    device_ctx = contextlib.nullcontext() if use_cpu else torch.device("meta")
    device = "cpu" if use_cpu else "meta"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with device_ctx:
            model = cls(**init_kwargs)
            if any(
                isinstance(p, nn.UninitializedParameter) for p in model.parameters()
            ):
                n_chans, n_times = probe_shape
                with torch.no_grad():
                    model.eval()(torch.zeros(1, n_chans, n_times, device=device))
        n_params = sum(p.numel() for p in model.parameters())
        cropped = _cropped_capability(name, cls, model, base_cls)
        adapt_input = _adapt_capability(cls, model)
        reset_head = _reset_head_capability(name, cls, model, base_cls)
    return n_params, cropped, adapt_input, reset_head


def _read_existing_csv(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not csv_path.is_file():
        return [], []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        headers = list(reader.fieldnames or ())
        rows = [
            {k: (v or "").strip() for k, v in row.items() if k is not None}
            for row in reader
            if (row.get("Model") or "").strip()
        ]
    return headers, rows


def _serialize_csv(rows: Sequence[dict[str, str]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=list(SUMMARY_COLUMNS),
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({col: row.get(col, "") for col in SUMMARY_COLUMNS})
    return buf.getvalue()


def generate_summary_rows(
    names: Iterable[str] | None = None,
    *,
    csv_path: Path = SUMMARY_CSV_PATH,
) -> list[dict[str, str]]:
    """Compute canonical ``summary.csv`` rows for registered models."""
    from braindecode import models as bd_models
    from braindecode.models.base import EEGModuleMixin
    from braindecode.models.util import models_mandatory_parameters

    _, existing_rows = _read_existing_csv(csv_path)
    existing_by_name = {row["Model"]: row for row in existing_rows}
    mandatory_by_name = {
        name: (req, sp) for name, req, sp in models_mandatory_parameters
    }

    if names is None:
        from braindecode.models.util import models_dict

        registered = dict(models_dict)
        ordered_names = [
            row["Model"] for row in existing_rows if row["Model"] in registered
        ]
        seen = set(ordered_names)
        for name, _, _ in models_mandatory_parameters:
            if name in registered and name not in seen:
                ordered_names.append(name)
                seen.add(name)
        for name in sorted(registered):
            if name not in seen:
                ordered_names.append(name)
                seen.add(name)
        resolve_cls = lambda n: registered[n]  # noqa: E731
    else:
        ordered_names = list(names)
        resolve_cls = lambda n: getattr(bd_models, n)  # noqa: E731

    generated: list[dict[str, str]] = []
    for name in ordered_names:
        cls = resolve_cls(name)
        req, sp = mandatory_by_name[name]
        editorial = existing_by_name.get(name, {})
        init_kwargs, expr, probe_shape = _resolve_kwargs(name, req, sp)
        n_params, cropped, adapt_input, reset_head = _instantiate_and_measure(
            name, cls, EEGModuleMixin, init_kwargs, probe_shape
        )
        time_axis = cls._time_axis
        channel_axis = cls._channel_axis
        if time_axis is None or channel_axis is None:
            raise ValueError(f"{name} must declare _time_axis and _channel_axis.")

        row: dict[str, str] = {
            "Model": name,
            "Application": editorial.get("Application", ""),
            "Type": editorial.get("Type", ""),
            "Sampling Frequency (Hz)": editorial.get("Sampling Frequency (Hz)", ""),
            "Hyperparameters": ", ".join(req),
            "#Parameters": str(n_params),
            "get_#Parameters": expr,
            "Categorization": _parse_categorization(name, cls),
            "Modality": editorial.get("Modality", ""),
            "Time axis": time_axis,
            "Channel axis": channel_axis,
            "Output": _output_kind(name),
            "Cropped decoding": cropped,
            "adapt_input": adapt_input,
            "reset_head": reset_head,
            "return_features": _return_features_capability(name, cls, EEGModuleMixin),
        }
        for col in EDITORIAL_COLUMNS:
            if not row[col]:
                raise ValueError(f"{name} is missing editorial column {col!r}.")
        generated.append(row)
    return generated


def check_summary_csv(csv_path: Path = SUMMARY_CSV_PATH) -> list[str]:
    """Compare ``csv_path`` on disk against freshly generated summary rows."""
    if not csv_path.is_file():
        return [f"Missing summary CSV: {csv_path}"]

    raw_text = csv_path.read_text(encoding="utf-8")
    diffs: list[str] = []
    if raw_text.startswith("\ufeff"):
        diffs.append("summary.csv starts with a UTF-8 BOM; write plain UTF-8.")

    headers, disk_rows = _read_existing_csv(csv_path)
    if headers != list(SUMMARY_COLUMNS):
        diffs.append(
            f"Header mismatch: got {headers}, expected {list(SUMMARY_COLUMNS)}"
        )

    expected_rows = generate_summary_rows(csv_path=csv_path)
    disk_names = [r["Model"] for r in disk_rows]
    expected_names = [r["Model"] for r in expected_rows]
    if disk_names != expected_names:
        missing = sorted(set(expected_names) - set(disk_names))
        extra = sorted(set(disk_names) - set(expected_names))
        if missing:
            diffs.append(f"Missing models in summary.csv: {missing}")
        if extra:
            diffs.append(f"Unexpected models in summary.csv: {extra}")
        if not missing and not extra:
            diffs.append("Row order in summary.csv differs from generator order.")

    disk_by_name = {r["Model"]: r for r in disk_rows}
    for exp in expected_rows:
        name = exp["Model"]
        got = disk_by_name.get(name)
        if got is None:
            continue
        for col in SUMMARY_COLUMNS:
            if got.get(col) != exp[col]:
                diffs.append(
                    f"{name}.{col}: disk={got.get(col)!r} != expected={exp[col]!r}"
                )

    if not diffs and raw_text != _serialize_csv(expected_rows):
        diffs.append(
            "Formatting or line endings in summary.csv differ from canonical CSV."
        )
    return diffs


def _models_for_files(files: Sequence[str]) -> list[str] | None:
    """Return model names affected by ``files``, ``[]`` if none, or ``None`` for all."""
    from braindecode import models as bd_models

    models_dir = SUMMARY_CSV_PATH.parent.resolve()
    repo_root = SUMMARY_CSV_PATH.parents[2].resolve()
    module_to_models: dict[str, list[str]] = {}
    for sym, mod_name in bd_models._LAZY.items():
        if sym not in _NON_MODEL_LAZY_EXPORTS:
            module_to_models.setdefault(mod_name, []).append(sym)

    selected: list[str] = []
    seen: set[str] = set()
    for raw_file in files:
        path = Path(raw_file)
        if path.is_absolute():
            resolved = path.resolve()
        elif (repo_root / path).exists():
            resolved = (repo_root / path).resolve()
        else:
            resolved = (Path.cwd() / path).resolve()
        if resolved.parent != models_dir or resolved.suffix != ".py":
            continue
        stem = resolved.stem
        if stem in _ALL_MODELS_FILES:
            return None
        for model_name in module_to_models.get(stem, ()):
            if model_name not in seen:
                selected.append(model_name)
                seen.add(model_name)
    return selected


def update_summary_csv(
    files: Sequence[str] | None = None,
    *,
    csv_path: Path = SUMMARY_CSV_PATH,
) -> bool:
    """Regenerate all or touched model rows in ``csv_path``. Return True if modified."""
    target_names: list[str] | None = None
    if files is not None:
        target_names = _models_for_files(files)
        if target_names == []:
            return False

    headers, existing_rows = _read_existing_csv(csv_path)
    if target_names is not None and headers == list(SUMMARY_COLUMNS):
        existing_names = {r["Model"] for r in existing_rows}
        if set(target_names) <= existing_names:
            updated_by_name = {
                r["Model"]: r
                for r in generate_summary_rows(names=target_names, csv_path=csv_path)
            }
            merged = [updated_by_name.get(r["Model"], r) for r in existing_rows]
            rendered = _serialize_csv(merged)
            current = csv_path.read_text(encoding="utf-8") if csv_path.is_file() else ""
            if rendered == current:
                return False
            csv_path.write_text(rendered, encoding="utf-8")
            return True

    full_rows = generate_summary_rows(csv_path=csv_path)
    rendered = _serialize_csv(full_rows)
    current = csv_path.read_text(encoding="utf-8") if csv_path.is_file() else ""
    if rendered == current:
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
        if diffs:
            for line in diffs:
                print(line, file=sys.stderr)
            return 1
        return 0

    update_summary_csv(args.files, csv_path=args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
