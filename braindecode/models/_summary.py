"""Deterministic generator and drift checker for ``braindecode/models/summary.csv``."""

from __future__ import annotations

import argparse
import csv
import inspect
import io
import re
import sys
import warnings
from collections.abc import Iterable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn

_MODELS_DIR = Path(__file__).resolve().parent
_DEFAULT_CSV_PATH = _MODELS_DIR / "summary.csv"
_NON_MODEL_STEMS: frozenset[str] = frozenset(
    {"__init__", "_summary", "base", "config", "hybrid", "util"}
)

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

# Hand-written columns, kept from the existing CSV. The code has no reliable
# source for Application, Type or Modality. An empty Sampling Frequency cell is
# filled from the constructor default of ``sfreq`` when the model has one.
EDITORIAL_COLUMNS: tuple[str, ...] = (
    "Application",
    "Type",
    "Sampling Frequency (Hz)",
    "Modality",
)

# Models whose __init__ extracts host scalars, builds torchaudio kernels on CPU,
# or runs CPU-bound graph initialization that has no PyTorch meta kernel.
_CPU_MODELS: frozenset[str] = frozenset(
    {
        "BaRISTA",
        "CORTEG",
        "DGCNN",
        "EEGPT",
        "EEGSimpleConv",
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

_BADGE_TO_CATEGORY: dict[str, str] = {
    ":bdg-success:`Convolution`": "Convolution",
    ":bdg-secondary:`Recurrent`": "Recurrent",
    ":bdg-info:`Attention/Transformer`": "Attention/Transformer",
    ":bdg-primary:`Filterbank`": "Filterbank",
    ":bdg-warning:`Interpretability`": "Interpretability",
    ":bdg-danger:`Foundation Model`": "Foundation Model",
    ":bdg-dark-line:`Channel`": "Channel",
    ":bdg-light:`Graph Neural Network`": "Graph Neural Network",
    ":bdg-dark:`SPD`": "SPD",
}
_BADGE_RE = re.compile(r":bdg-[\w-]+:`[^`]+`")

_OUTPUT_DESCRIPTIONS: dict[str, str] = {
    "BrainTokenizer": "(batch, n_chans, n_times) reconstruction",
    "CareSleepNet": "(batch, n_outputs, n_epochs) logits",
    "DANCE": "(batch, num_latents, n_outputs) logits",
    "EMG2QwertyNet": "(batch, n_steps, n_outputs) logits",
    "MetaNeuromotorHand": "(batch, n_steps, n_outputs) logits",
    "NeuroPose": "(batch, n_times, n_outputs) pose",
    "NeuroRVQTokenizer": "tuple of (batch, n_chans, n_times) reconstructions",
    "SeizureTransformer": "(batch, n_outputs, n_times) logits",
    "SensingDynamics": "(batch, n_times, n_outputs) pose",
    "SignalJEPA": "(batch, n_tokens, emb_dim) features",
    "SleepFMStager": "(batch, n_outputs, n_patches) logits",
    "TFMTokenizer": "(batch, n_chans, n_freqs, n_frames) reconstruction",
    "VEMG2Pose": "(batch, n_steps, n_outputs) pose",
}

_TOKENIZERS: frozenset[str] = frozenset(
    {"BrainTokenizer", "NeuroRVQTokenizer", "TFMTokenizer"}
)

_CHS_INFO_3CH_COORDS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("C1", (-0.03793783, 0.02633745, 0.12977061)),
    ("C2", (0.03589271, 0.02635814, 0.12841234)),
    ("C3", (-0.06714873, 0.02335823, 0.10451068)),
)


def _registry_specs() -> dict[str, tuple[list[str], Any]]:
    from braindecode.models.util import models_mandatory_parameters

    return {
        name: (list(req), raw_sp) for name, req, raw_sp in models_mandatory_parameters
    }


def _read_csv_rows(csv_path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    if not csv_path.is_file():
        return (), []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        fieldnames = tuple(reader.fieldnames or ())
        rows = [
            {k: (v or "").strip() for k, v in row.items() if k is not None}
            for row in reader
            if (row.get("Model") or "").strip()
        ]
    return fieldnames, rows


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


def _parse_categorization(model_name: str, cls: type) -> str:
    docstring = cls.__doc__ or ""
    categories: list[str] = []
    for badge in _BADGE_RE.findall(docstring):
        category = _BADGE_TO_CATEGORY.get(badge)
        if category is None:
            raise ValueError(f"{model_name} has unknown categorization badge: {badge}")
        if category not in categories:
            categories.append(category)
    if not categories:
        raise ValueError(f"{model_name} has no categorization badges in its docstring.")
    return ",".join(categories)


def _default_sfreq(cls: type) -> str:
    """Return the constructor default of ``sfreq`` as text, or ``""`` if none."""
    param = inspect.signature(cls.__init__).parameters.get("sfreq")
    default = None if param is None else param.default
    if isinstance(default, bool) or not isinstance(default, (int, float)):
        return ""
    return f"{default:g}"


def _summary_model_kwargs(
    model_name: str,
    required_params: Sequence[str],
    raw_signal_params: Any,
) -> tuple[dict[str, Any], int, int, str]:
    import numpy as np

    from braindecode.models.util import _EEG_PARAMS, _draw_chs_info

    raw_sp = (
        raw_signal_params()
        if callable(raw_signal_params)
        else (dict(raw_signal_params) if raw_signal_params is not None else {})
    )
    sp_eeg = {k: v for k, v in raw_sp.items() if k in _EEG_PARAMS}

    chs_3ch = [
        {"ch_name": name, "kind": "eeg", "loc": np.array([*xyz, *[0.0] * 9])}
        for name, xyz in _CHS_INFO_3CH_COORDS
    ]
    sp: dict[str, Any] = {
        "n_times": 1000,
        "sfreq": 250.0,
        "n_outputs": 2,
        "chs_info": chs_3ch,
        "n_chans": len(chs_3ch),
        "input_window_seconds": 4.0,
    }
    sp.update(sp_eeg)
    if "chs_info" in sp_eeg and "n_chans" not in sp_eeg:
        sp["n_chans"] = len(sp_eeg["chs_info"])
    if (
        "n_chans" in sp_eeg
        and "chs_info" not in sp_eeg
        and "chs_info" in required_params
    ):
        n_chans = int(sp_eeg["n_chans"])
        sp["chs_info"] = _draw_chs_info(n_chans, np.random.default_rng(n_chans))
    if "input_window_seconds" not in sp_eeg:
        sp["input_window_seconds"] = sp["n_times"] / sp["sfreq"]
    if "sfreq" not in sp_eeg:
        sp["sfreq"] = sp["n_times"] / sp["input_window_seconds"]
    if "n_times" not in sp_eeg:
        sp["n_times"] = int(round(sp["input_window_seconds"] * sp["sfreq"]))

    keys = list(dict.fromkeys([*required_params, *sp_eeg.keys()]))
    if "chs_info" in keys and "chs_info" not in required_params:
        idx = keys.index("chs_info")
        if "n_chans" in keys:
            keys.pop(idx)
        else:
            keys[idx] = "n_chans"
    if "chs_info" in keys and "n_chans" in keys and "n_chans" not in required_params:
        keys.remove("n_chans")

    kwargs: dict[str, Any] = {
        k: (deepcopy(sp[k]) if k == "chs_info" else sp[k]) for k in keys
    }
    args_str = ", ".join(
        "chs_info=chs_info" if k == "chs_info" else f"{k}={v!r}"
        for k, v in kwargs.items()
    )
    get_params_expr = f"{model_name}({args_str})"
    return kwargs, int(sp["n_chans"]), int(sp["n_times"]), get_params_expr


def _instantiate_for_summary(
    model_name: str,
    cls: type,
    kwargs: dict[str, Any],
    n_chans: int,
    n_times: int,
) -> nn.Module:
    import torch
    from torch import nn

    device = "cpu" if model_name in _CPU_MODELS else "meta"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if device == "meta":
            with torch.device("meta"):
                model = cls(**kwargs)
        else:
            model = cls(**kwargs)
        if any(isinstance(p, nn.UninitializedParameter) for p in model.parameters()):
            model.eval()
            with torch.no_grad():
                model(torch.zeros(2, n_chans, n_times, device=device))
    return model


def _has_strided_layer(model: nn.Module) -> bool:
    from torch import nn

    strided_types = (
        nn.Conv1d,
        nn.Conv2d,
        nn.MaxPool1d,
        nn.MaxPool2d,
        nn.AvgPool1d,
        nn.AvgPool2d,
    )
    for module in model.modules():
        if not isinstance(module, strided_types):
            continue
        stride = module.stride
        if isinstance(stride, int) and stride > 1:
            return True
        if isinstance(stride, tuple) and any(s > 1 for s in stride):
            return True
    return False


def _cropped_decoding_value(cls: type, model: nn.Module) -> str:
    from torch import nn

    from braindecode.models.base import EEGModuleMixin

    time_axis = getattr(cls, "_time_axis", None)
    if hasattr(EEGModuleMixin, "_dense_conversion"):
        supported = (
            time_axis != "fixed"
            and bool(getattr(cls, "_dense_conversion", True))
            and (time_axis == "dense" or _has_strided_layer(model))
        )
        return "yes" if supported else "no"
    supported = time_axis == "dense" and not any(
        isinstance(m, (nn.modules.conv._ConvTransposeNd, nn.Upsample))
        for m in model.modules()
    )
    return "yes" if supported else "no"


def _adapt_input_value(cls: type, modality: str) -> str:
    if not callable(getattr(cls, "adapt_input", None)):
        return "no"
    explicit = getattr(cls, "_supports_adapt_input", None)
    if isinstance(explicit, str):
        return explicit
    if isinstance(explicit, bool):
        return "yes" if explicit else "no"
    modalities = {m.strip() for m in modality.split(",") if m.strip()}
    if bool({"EEG", "iEEG", "ECoG"} & modalities) and cls.__name__ not in {
        "SleepFM",
        "SleepFMStager",
    }:
        return "yes"
    return "no"


def _reset_head_value(cls: type, model: nn.Module) -> str:
    from braindecode.models.base import EEGModuleMixin

    can_reset = getattr(model, "_can_reset_head", None)
    if callable(can_reset):
        return "yes" if can_reset() else "no"
    return "yes" if cls.reset_head is not EEGModuleMixin.reset_head else "no"


def _return_features_value(cls: type) -> str:
    from braindecode.models.base import EEGModuleMixin

    if "return_features" in inspect.signature(cls.forward).parameters:
        return "yes"
    if hasattr(EEGModuleMixin, "_pool_embedding") and cls.__name__ not in _TOKENIZERS:
        return "yes"
    return "no"


def models_for_files(files: Iterable[str | Path]) -> list[str]:
    """Return sorted registered model names defined in ``files``."""
    stems: set[str] = set()
    for raw_path in files:
        path = Path(raw_path)
        if path.suffix != ".py":
            continue
        resolved = path.resolve()
        if resolved.parent == _MODELS_DIR or path.parent.name == "models":
            if path.stem not in _NON_MODEL_STEMS:
                stems.add(path.stem)
    if not stems:
        return []
    from braindecode import models as _models_pkg

    specs = _registry_specs()
    lazy_map = _models_pkg._LAZY
    return sorted(name for name in specs if lazy_map.get(name) in stems)


def generate_summary_rows(
    names: Iterable[str] | None = None,
    *,
    csv_path: Path | None = None,
) -> list[dict[str, str]]:
    """Compute canonical ``summary.csv`` rows for ``names`` (all registered models if ``None``)."""
    from braindecode import models as _models_pkg

    target_csv = _DEFAULT_CSV_PATH if csv_path is None else Path(csv_path)
    _, existing_rows = _read_csv_rows(target_csv)
    editorial_by_name = {
        row["Model"]: {col: row.get(col, "") for col in EDITORIAL_COLUMNS}
        for row in existing_rows
    }

    specs = _registry_specs()
    if names is None:
        from braindecode.models.util import models_dict

        target_names = sorted(models_dict)
        classes = {name: models_dict[name] for name in target_names}
    else:
        target_names = sorted(dict.fromkeys(names))
        classes = {name: getattr(_models_pkg, name) for name in target_names}

    rows: list[dict[str, str]] = []
    for name in target_names:
        if name not in specs:
            raise ValueError(f"{name} is missing from models_mandatory_parameters.")
        cls = classes[name]
        required_params, raw_sp = specs[name]
        editorial = editorial_by_name.get(name, {})

        kwargs, n_chans, n_times, get_params_expr = _summary_model_kwargs(
            name, required_params, raw_sp
        )
        model = _instantiate_for_summary(name, cls, kwargs, n_chans, n_times)
        n_params = sum(p.numel() for p in model.parameters())

        time_axis = cls.__dict__.get("_time_axis") or getattr(cls, "_time_axis", None)
        channel_axis = cls.__dict__.get("_channel_axis") or getattr(
            cls, "_channel_axis", None
        )
        if time_axis not in {"fixed", "free", "dense"}:
            raise ValueError(f"{name} has invalid _time_axis={time_axis!r}.")
        if channel_axis not in {"fixed", "free"}:
            raise ValueError(f"{name} has invalid _channel_axis={channel_axis!r}.")

        modality = editorial.get("Modality", "")
        rows.append(
            {
                "Model": name,
                "Application": editorial.get("Application", ""),
                "Type": editorial.get("Type", ""),
                # A hand-written value wins; the constructor default fills empty cells.
                "Sampling Frequency (Hz)": editorial.get("Sampling Frequency (Hz)")
                or _default_sfreq(cls),
                "Hyperparameters": ", ".join(required_params),
                "#Parameters": str(n_params),
                "get_#Parameters": get_params_expr,
                "Categorization": _parse_categorization(name, cls),
                "Modality": modality,
                "Time axis": time_axis,
                "Channel axis": channel_axis,
                "Output": _OUTPUT_DESCRIPTIONS.get(name, "(batch, n_outputs) logits"),
                "Cropped decoding": _cropped_decoding_value(cls, model),
                "adapt_input": _adapt_input_value(cls, modality),
                "reset_head": _reset_head_value(cls, model),
                "return_features": _return_features_value(cls),
            }
        )
    return rows


def check_summary_csv(
    csv_path: Path | None = None,
    *,
    names: Iterable[str] | None = None,
) -> list[str]:
    """Return human-readable drift messages between ``csv_path`` and generated rows."""
    target_csv = _DEFAULT_CSV_PATH if csv_path is None else Path(csv_path)
    if not target_csv.is_file():
        return [f"Missing summary CSV: {target_csv}"]

    fieldnames, disk_rows = _read_csv_rows(target_csv)
    diffs: list[str] = []
    if fieldnames != SUMMARY_COLUMNS:
        diffs.append(
            f"Columns mismatch: csv={list(fieldnames)} != expected={list(SUMMARY_COLUMNS)}"
        )

    disk_by_name = {row["Model"]: row for row in disk_rows}
    expected_rows = generate_summary_rows(names=names, csv_path=target_csv)

    if names is None:
        disk_names = [row["Model"] for row in disk_rows]
        expected_names = [row["Model"] for row in expected_rows]
        missing = sorted(set(expected_names) - set(disk_names))
        extra = sorted(set(disk_names) - set(expected_names))
        for name in missing:
            diffs.append(f"Missing model row in {target_csv.name}: {name}")
        for name in extra:
            diffs.append(f"Unknown model row in {target_csv.name}: {name}")
        if not missing and not extra and disk_names != expected_names:
            diffs.append(f"Model rows in {target_csv.name} are not sorted by Model.")

    for exp_row in expected_rows:
        name = exp_row["Model"]
        got_row = disk_by_name.get(name)
        if got_row is None:
            if names is not None:
                diffs.append(f"Missing model row in {target_csv.name}: {name}")
            continue
        for col in SUMMARY_COLUMNS:
            got_val = got_row.get(col, "")
            exp_val = exp_row[col]
            if got_val != exp_val:
                diffs.append(f"{name}[{col}]: csv={got_val!r} != generated={exp_val!r}")
    return diffs


def write_summary_csv(
    csv_path: Path | None = None,
    *,
    files: Sequence[str | Path] | None = None,
) -> bool:
    """Regenerate all rows (or only models defined in ``files``) and write if changed."""
    target_csv = _DEFAULT_CSV_PATH if csv_path is None else Path(csv_path)
    if files is not None:
        target_names = models_for_files(files)
        if not target_names:
            return False
        _, disk_rows = _read_csv_rows(target_csv)
        merged = {row["Model"]: row for row in disk_rows}
        for row in generate_summary_rows(names=target_names, csv_path=target_csv):
            merged[row["Model"]] = row
        out_rows = [merged[name] for name in sorted(merged)]
    else:
        out_rows = generate_summary_rows(names=None, csv_path=target_csv)

    rendered = _serialize_csv(out_rows)
    existing = target_csv.read_text(encoding="utf-8") if target_csv.is_file() else None
    if existing == rendered:
        return False
    target_csv.write_text(rendered, encoding="utf-8")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m braindecode.models._summary",
        description="Generate or check braindecode/models/summary.csv.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify summary.csv matches generated metadata without writing.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Update or check only models defined in the given model .py files.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=_DEFAULT_CSV_PATH,
        help="Path to summary.csv (defaults to braindecode/models/summary.csv).",
    )
    args = parser.parse_args(argv)

    if args.check:
        target_names: list[str] | None = None
        if args.files is not None:
            target_names = models_for_files(args.files)
            if not target_names:
                return 0
        diffs = check_summary_csv(args.csv, names=target_names)
        if diffs:
            for line in diffs:
                print(line, file=sys.stderr)
            return 1
        return 0

    write_summary_csv(args.csv, files=args.files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
