# mypy: ignore-errors
"""
Low-level Zarr I/O helpers for Hub integration.

These functions keep the Zarr serialization details isolated from hub.py.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from numbers import Real
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.utils import _soft_import

zarr = _soft_import("zarr", purpose="hugging face integration", strict=False)


def _is_non_bool_real(value):
    return not isinstance(value, (bool, np.bool_)) and isinstance(
        value, (Real, np.integer, np.floating)
    )


def _prepare_info_for_json(obj, path="info", *, _in_numeric_sequence=False):
    """Normalize an MNE Info value to strict JSON without losing sequence NaNs."""
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()

    if isinstance(obj, dict):
        return {
            key: _prepare_info_for_json(value, f"{path}.{key}")
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        values = list(obj)
        has_none = any(value is None for value in values)
        has_number = any(_is_non_bool_real(value) for value in values)
        all_none = bool(values) and all(value is None for value in values)
        if all_none or (has_none and has_number):
            raise ValueError(
                f"{path} is ambiguous: numeric sequences cannot contain JSON null"
            )

        is_numeric_sequence = bool(values) and all(
            _is_non_bool_real(value) for value in values
        )
        return [
            _prepare_info_for_json(
                value,
                f"{path}[{index}]",
                _in_numeric_sequence=is_numeric_sequence,
            )
            for index, value in enumerate(values)
        ]

    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if _is_non_bool_real(obj):
        value = float(obj)
        if np.isnan(value):
            if _in_numeric_sequence:
                return None
            raise ValueError(f"{path} contains unsupported NaN")
        if np.isposinf(value):
            raise ValueError(f"{path} contains positive infinity")
        if np.isneginf(value):
            raise ValueError(f"{path} contains negative infinity")
        return value
    if obj is None or isinstance(obj, str):
        return obj
    raise ValueError(f"{path} contains a non-JSON-serializable value")


def _restore_nan_from_json(obj):
    """Restore NaN values from None in JSON-loaded attributes.

    JSON null is reserved for NaN only in non-boolean numeric sequences.
    All-null lists are therefore the representation of validated all-NaN
    sequences, while ordinary null values elsewhere remain unchanged.
    """
    if isinstance(obj, dict):
        return {k: _restore_nan_from_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > 0 and all(
            value is None or _is_non_bool_real(value) for value in obj
        ):
            return [np.nan if x is None else x for x in obj]
        return [_restore_nan_from_json(v) for v in obj]
    return obj


def _annotations_for_json(annotations):
    if annotations is None:
        return None
    extras = [dict(extra) for extra in annotations.extras]
    # MNE extras permit scalar strings and floats. A tagged value keeps NaN
    # distinct from the literal label "nan" in a strict JSON store.
    for extra in extras:
        for key, value in extra.items():
            if isinstance(value, float) and not np.isfinite(value):
                extra[key] = {"float": str(value)}
    values = dict(
        description=annotations.description.tolist(),
        orig_time=annotations.orig_time.isoformat() if annotations.orig_time else None,
        ch_names=[list(chs) for chs in annotations.ch_names],
        extras=extras,
    )
    for key in ("onset", "duration"):
        values[key] = [
            v if np.isfinite(v) else str(v) for v in getattr(annotations, key).tolist()
        ]
    if hasattr(annotations, "hed_string"):
        values.update(
            hed_string=list(annotations.hed_string),
            hed_version=annotations._hed_version,
        )
    return values


def _annotations_from_json(values):
    if values is None:
        return None
    values = dict(values)
    if values["orig_time"] is not None:
        values["orig_time"] = datetime.fromisoformat(values["orig_time"])
    values["extras"] = [
        {key: float(v["float"]) if isinstance(v, dict) else v for key, v in row.items()}
        for row in values["extras"]
    ]
    if "hed_string" in values:
        if hasattr(mne, "HEDAnnotations") and _soft_import(
            "hed", "HED annotations", strict=False
        ):
            return mne.HEDAnnotations(**values)
        warnings.warn(
            "Loading HED annotations as MNE HEDAnnotations requires MNE >= 1.12 and hedtools. "
            "Keeping the HED strings in the annotation extras.",
            RuntimeWarning,
            stacklevel=2,
        )
        values.pop("hed_version")
        for row, tags in zip(values["extras"], values.pop("hed_string")):
            row["HED"] = tags
    return mne.Annotations(**values)


def _raw_from_zarr(data, info, state):
    """Restore a continuous recording on its acquisition sample clock."""
    raw = mne.io.RawArray(data, info, first_samp=state.get("first_samp", 0))
    annotations = _annotations_from_json(state.get("annotations"))
    if annotations is not None:
        if annotations.orig_time is None:
            # Raw stores onsets on the acquisition clock. set_annotations expects
            # undated onsets relative to the first sample in this recording.
            annotations.onset -= raw.first_time
        raw.set_annotations(annotations)
    return raw


def _epochs_from_zarr(data, info, metadata, state):
    """Restore Epochs independently of per-trial window coordinates and labels."""
    if "events" in state:
        events = state["events"]
        kwargs = dict(
            tmin=state["tmin"],
            event_id=state["event_id"],
            selection=state["selection"],
            raw_sfreq=state["raw_sfreq"],
            on_missing="ignore",  # a retained event ID may have no remaining epochs
        )
        if "drop_log" in state:
            log = state["drop_log"]
            kwargs["drop_log"] = tuple(
                tuple(log["reasons"].get(str(i), ())) for i in range(log["length"])
            )
    else:
        # Older stores only recorded crop coordinates and training targets.
        starts = metadata["i_start_in_trial"].to_numpy(dtype=int)
        if len(np.unique(starts)) != len(starts):
            warnings.warn(
                "This older Zarr store has repeated event samples and no "
                "acquisition event clock. Assigning consecutive epochs for MNE.",
                RuntimeWarning,
            )
            starts = np.arange(len(starts)) * data.shape[-1]
        codes = metadata.get("target")
        if codes is None or not pd.api.types.is_integer_dtype(codes):
            codes = np.ones(len(starts), dtype=int)
        events = np.column_stack([starts, np.zeros(len(starts), dtype=int), codes])
        kwargs = {}
    # The samples already contain any projection and baseline correction.
    epochs = mne.EpochsArray(
        data, info, events=events, metadata=metadata, proj=False, **kwargs
    )
    if "tmin" in state and epochs.tmin != state["tmin"]:
        # EpochsArray rounds tmin to the current sample grid. A decimation
        # offset or shift_time can start the epoch between those grid points.
        epochs.shift_time(state["tmin"], relative=False)
        # MNE's next decimation uses the preloaded sample clock.
        epochs._raw_times = epochs.times
    if state.get("baseline") is not None:
        epochs.baseline = tuple(state["baseline"])
    epochs.set_annotations(_annotations_from_json(state.get("annotations")))
    return epochs


def _save_window_metadata(grp, metadata, targets_from):
    if not isinstance(metadata, pd.DataFrame):
        # Fixed windows can compute each metadata cell lazily.
        metadata = pd.DataFrame(
            {
                col: [metadata.loc[i, col] for i in range(len(metadata))]
                for col in metadata.columns
            }
        )
    if targets_from == "metadata":
        targets = metadata["target"]
        kind = pd.api.types.infer_dtype(targets.to_numpy(), skipna=False)
        if kind == "string":
            # Preserve literal labels such as "01" and "NA".
            grp.attrs["string_targets"] = True
        elif kind.startswith("mixed"):
            # A TSV cell has no representation for a list or mixed scalar types.
            # NumPy scalars in a recording description have native equivalents.
            metadata = metadata.assign(
                target=targets.apply(
                    json.dumps, default=lambda obj: obj.tolist(), ensure_ascii=False
                )
            )
            grp.attrs["json_targets"] = True
    store_path = getattr(grp.store, "path", getattr(grp.store, "root", None))
    metadata.to_csv(Path(store_path) / grp.path / "metadata.tsv", sep="\t", index=True)


def _load_window_metadata(grp):
    store_path = getattr(grp.store, "path", getattr(grp.store, "root", None))
    json_targets = grp.attrs.get("json_targets", False)
    text_targets = json_targets or grp.attrs.get("string_targets", False)
    # pandas also checks converter results for missing-value strings, including "NA".
    metadata = pd.read_csv(
        Path(store_path) / grp.path / "metadata.tsv",
        sep="\t",
        index_col=0,
        converters={"target": str.encode} if text_targets else None,
    )
    if text_targets:
        metadata["target"] = metadata["target"].map(
            json.loads if json_targets else bytes.decode
        )
    return metadata


def _save_windows_to_zarr(grp, epochs, compressor):
    """Save windowed data to Zarr group.

    Uses 1 window per chunk so the one-time decompression to memmap
    (on lazy load) stays fast per recording.
    """
    data_array = epochs.get_data().astype(np.float32)
    compressors_list = [compressor] if compressor is not None else None

    # 1 window per chunk: aligned with __getitem__(index) access pattern.
    # Each chunk is (1, n_ch, n_t) — typically 50-300 KB, fast to decompress.
    n_windows_per_chunk = 1

    grp.create_array(
        "data",
        data=data_array,
        chunks=(
            n_windows_per_chunk,
            data_array.shape[1],
            data_array.shape[2],
        ),
        compressors=compressors_list,
    )

    grp.create_array("events", data=epochs.events, compressors=compressors_list)
    grp.create_array("selection", data=epochs.selection, compressors=compressors_list)
    state = dict(
        event_id={key: int(code) for key, code in epochs.event_id.items()},
        tmin=float(epochs.tmin),
        # Reading an Epochs FIF can produce a one-element array here.
        raw_sfreq=np.asarray(epochs._raw_sfreq).item(),
        baseline=epochs.baseline,
        annotations=_annotations_for_json(epochs.annotations),
    )
    # Most windowed recordings have no rejected epochs. Store only the reasons.
    if any(epochs.drop_log):
        state["drop_log"] = dict(
            length=len(epochs.drop_log),
            reasons={
                str(i): reasons for i, reasons in enumerate(epochs.drop_log) if reasons
            },
        )
    grp.attrs["mne"] = state


def _save_raw_to_zarr(grp, raw, compressor, chunk_size):
    """Save RawDataset continuous raw data to Zarr group (low-level function)."""
    continuous_data = raw.get_data()
    continuous_float = continuous_data.astype(np.float32)
    compressors_list = [compressor] if compressor is not None else None

    max_samples_per_chunk = max(1, chunk_size // continuous_float.shape[0])
    n_samples_per_chunk = min(continuous_float.shape[1], max_samples_per_chunk)

    grp.create_array(
        "data",
        data=continuous_float,
        chunks=(continuous_float.shape[0], n_samples_per_chunk),
        compressors=compressors_list,
    )

    grp.attrs["mne"] = dict(
        first_samp=int(raw.first_samp),
        annotations=_annotations_for_json(raw.annotations),
    )


def _create_compressor(compression, compression_level):
    """Create a Zarr v3 compressor codec."""
    if zarr is False:
        raise ImportError(
            "Zarr is not installed. Install with: pip install braindecode[hub]"
        )

    if compression is None or compression not in ("blosc", "zstd", "gzip"):
        return None

    name = "zstd" if compression == "blosc" else compression
    return {"name": name, "configuration": {"level": compression_level}}
