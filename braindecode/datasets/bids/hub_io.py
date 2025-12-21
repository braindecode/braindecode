# mypy: ignore-errors
"""
Low-level Zarr I/O helpers for Hub integration.

These functions keep the Zarr serialization details isolated from hub.py.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from mne.utils import _soft_import

zarr = _soft_import("zarr", purpose="hugging face integration", strict=False)


def _sanitize_for_json(obj):
    """Replace NaN/Inf with None for valid JSON."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    return obj


def _restore_nan_from_json(obj):
    """Restore NaN values from None in JSON-loaded data."""
    if isinstance(obj, dict):
        return {k: _restore_nan_from_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > 0 and all(isinstance(x, (int, float, type(None))) for x in obj):
            return [np.nan if x is None else x for x in obj]
        return [_restore_nan_from_json(v) for v in obj]
    return obj


def _save_windows_to_zarr(
    grp, data, metadata, description, info, compressor, target_name
):
    """Save windowed data to Zarr group (low-level function)."""
    data_array = data.astype(np.float32)
    compressors_list = [compressor] if compressor is not None else None

    grp.create_array(
        "data",
        data=data_array,
        chunks=(1, data_array.shape[1], data_array.shape[2]),
        compressors=compressors_list,
    )

    store_path = getattr(grp.store, "path", getattr(grp.store, "root", None))
    metadata_path = Path(store_path) / grp.path / "metadata.tsv"
    metadata.to_csv(metadata_path, sep="\t", index=True)

    grp.attrs["description"] = json.loads(description.to_json(date_format="iso"))
    grp.attrs["info"] = _sanitize_for_json(info)

    if target_name is not None:
        grp.attrs["target_name"] = target_name


def _save_eegwindows_to_zarr(
    grp, raw, metadata, description, info, targets_from, last_target_only, compressor
):
    """Save EEG continuous raw data to Zarr group (low-level function)."""
    continuous_data = raw.get_data()
    continuous_float = continuous_data.astype(np.float32)
    compressors_list = [compressor] if compressor is not None else None

    grp.create_array(
        "data",
        data=continuous_float,
        chunks=(continuous_float.shape[0], min(10000, continuous_float.shape[1])),
        compressors=compressors_list,
    )

    store_path = getattr(grp.store, "path", getattr(grp.store, "root", None))
    metadata_path = Path(store_path) / grp.path / "metadata.tsv"
    metadata.to_csv(metadata_path, sep="\t", index=True)

    grp.attrs["description"] = json.loads(description.to_json(date_format="iso"))
    grp.attrs["info"] = _sanitize_for_json(info)
    grp.attrs["targets_from"] = targets_from
    grp.attrs["last_target_only"] = last_target_only


def _load_windows_from_zarr(grp, preload):
    """Load windowed data from Zarr group (low-level function)."""
    store_path = getattr(grp.store, "path", getattr(grp.store, "root", None))
    metadata_path = Path(store_path) / grp.path / "metadata.tsv"
    metadata = pd.read_csv(metadata_path, sep="\t", index_col=0)

    description = pd.Series(grp.attrs["description"])
    info_dict = _restore_nan_from_json(grp.attrs["info"])

    if preload:
        data = grp["data"][:]
    else:
        data = grp["data"][:]
        warnings.warn(
            "Lazy loading from Zarr not fully implemented yet. "
            "Loading all data into memory.",
            UserWarning,
        )

    target_name = grp.attrs.get("target_name", None)

    return data, metadata, description, info_dict, target_name


def _load_eegwindows_from_zarr(grp, preload):
    """Load EEG continuous raw data from Zarr group (low-level function)."""
    store_path = getattr(grp.store, "path", getattr(grp.store, "root", None))
    metadata_path = Path(store_path) / grp.path / "metadata.tsv"
    metadata = pd.read_csv(metadata_path, sep="\t", index_col=0)

    description = pd.Series(grp.attrs["description"])
    info_dict = _restore_nan_from_json(grp.attrs["info"])

    if preload:
        data = grp["data"][:]
    else:
        data = grp["data"][:]
        warnings.warn(
            "Lazy loading from Zarr not fully implemented yet. "
            "Loading all data into memory.",
            UserWarning,
        )

    targets_from = grp.attrs.get("targets_from", "metadata")
    last_target_only = grp.attrs.get("last_target_only", True)

    return data, metadata, description, info_dict, targets_from, last_target_only


def _save_raw_to_zarr(grp, raw, description, info, target_name, compressor):
    """Save RawDataset continuous raw data to Zarr group (low-level function)."""
    continuous_data = raw.get_data()
    continuous_float = continuous_data.astype(np.float32)
    compressors_list = [compressor] if compressor is not None else None

    grp.create_array(
        "data",
        data=continuous_float,
        chunks=(continuous_float.shape[0], min(10000, continuous_float.shape[1])),
        compressors=compressors_list,
    )

    grp.attrs["description"] = json.loads(description.to_json(date_format="iso"))
    grp.attrs["info"] = _sanitize_for_json(info)

    if target_name is not None:
        grp.attrs["target_name"] = target_name


def _load_raw_from_zarr(grp, preload):
    """Load RawDataset continuous raw data from Zarr group (low-level function)."""
    description = pd.Series(grp.attrs["description"])
    info_dict = _restore_nan_from_json(grp.attrs["info"])

    if preload:
        data = grp["data"][:]
    else:
        data = grp["data"][:]
        warnings.warn(
            "Lazy loading from Zarr not fully implemented yet. "
            "Loading all data into memory.",
            UserWarning,
        )

    target_name = grp.attrs.get("target_name", None)

    return data, description, info_dict, target_name


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
