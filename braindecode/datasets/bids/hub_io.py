# mypy: ignore-errors
"""
Low-level Zarr I/O helpers for Hub integration.

These functions keep the Zarr serialization details isolated from hub.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from mne.utils import _soft_import

zarr = _soft_import("zarr", purpose="hugging face integration", strict=False)


def _sanitize_for_json(obj):
    """Replace NaN/Inf with None so attributes serialize to valid JSON.

    ``zarr.json`` must be spec-compliant JSON, but Python's ``json``
    module writes ``NaN``/``Infinity`` literals by default, which
    RFC 8259 forbids and non-Python zarr readers reject. MNE ``Info``
    dicts routinely contain NaN (e.g. in channel ``loc`` arrays), so
    they are sanitized before being stored as attributes and restored
    with ``_restore_nan_from_json`` on load.
    """
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
    """Restore NaN values from None in JSON-loaded attributes.

    Inverse of ``_sanitize_for_json`` for numeric arrays: restores NaN
    on load so ``mne.Info.from_json_dict`` gets proper NaN arrays. Also
    covers stores written by older braindecode versions, which sanitized
    NaN/Inf → None the same way.
    """
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
    """Save windowed data to Zarr group.

    Uses 1 window per chunk so the one-time decompression to memmap
    (on lazy load) stays fast per recording.
    """
    data_array = data.astype(np.float32)
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

    store_path = getattr(grp.store, "path", getattr(grp.store, "root", None))
    metadata_path = Path(store_path) / grp.path / "metadata.tsv"
    metadata.to_csv(metadata_path, sep="\t", index=True)

    grp.attrs["description"] = json.loads(description.to_json(date_format="iso"))
    grp.attrs["info"] = _sanitize_for_json(info)

    if target_name is not None:
        grp.attrs["target_name"] = target_name


def _save_eegwindows_to_zarr(
    grp,
    raw,
    metadata,
    description,
    info,
    targets_from,
    last_target_only,
    compressor,
    chunk_size,
):
    """Save EEG continuous raw data to Zarr group (low-level function)."""
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
        data = grp["data"]  # lazy zarr.Array reference

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
        data = grp["data"]  # lazy zarr.Array reference

    targets_from = grp.attrs.get("targets_from", "metadata")
    last_target_only = grp.attrs.get("last_target_only", True)

    return data, metadata, description, info_dict, targets_from, last_target_only


def _save_raw_to_zarr(grp, raw, description, info, target_name, compressor, chunk_size):
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
        data = grp["data"]  # lazy zarr.Array reference

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
