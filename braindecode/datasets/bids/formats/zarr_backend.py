# mypy: ignore-errors
"""
Zarr storage backend for Hub integration.

This backend stores EEG data in Zarr v3 format with optional compression.
It supports chunked storage but does **not** support lazy loading
(``preload=False`` loads all data into memory).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy
from mne._fiff.meas_info import Info
from mne.utils import _soft_import

import braindecode

from ...registry import get_dataset_class, get_dataset_type
from .. import hub_validation
from . import register_format
from .utils import _restore_nan_from_json, _sanitize_for_json

zarr = _soft_import("zarr", purpose="hugging face integration", strict=False)


# ---------------------------------------------------------------------------
# Low-level Zarr save helpers
# ---------------------------------------------------------------------------

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


def _save_windows_to_zarr(
    grp, data, metadata, description, info, compressor, target_name, chunk_size
):
    """Save windowed data to Zarr group (low-level function)."""
    data_array = data.astype(np.float32)
    compressors_list = [compressor] if compressor is not None else None

    max_windows_per_chunk = max(
        1, chunk_size // (data_array.shape[1] * data_array.shape[2])
    )
    n_windows_per_chunk = min(data_array.shape[0], max_windows_per_chunk)

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


# ---------------------------------------------------------------------------
# Low-level Zarr load helpers
# ---------------------------------------------------------------------------

def _load_windows_from_zarr(grp):
    """Load windowed data from Zarr group (low-level function)."""
    store_path = getattr(grp.store, "path", getattr(grp.store, "root", None))
    metadata_path = Path(store_path) / grp.path / "metadata.tsv"
    metadata = pd.read_csv(metadata_path, sep="\t", index_col=0)

    description = pd.Series(grp.attrs["description"])
    info_dict = _restore_nan_from_json(grp.attrs["info"])

    data = grp["data"][:]

    target_name = grp.attrs.get("target_name", None)

    return data, metadata, description, info_dict, target_name


def _load_eegwindows_from_zarr(grp):
    """Load EEG continuous raw data from Zarr group (low-level function)."""
    store_path = getattr(grp.store, "path", getattr(grp.store, "root", None))
    metadata_path = Path(store_path) / grp.path / "metadata.tsv"
    metadata = pd.read_csv(metadata_path, sep="\t", index_col=0)

    description = pd.Series(grp.attrs["description"])
    info_dict = _restore_nan_from_json(grp.attrs["info"])

    data = grp["data"][:]

    targets_from = grp.attrs.get("targets_from", "metadata")
    last_target_only = grp.attrs.get("last_target_only", True)

    return data, metadata, description, info_dict, targets_from, last_target_only


def _load_raw_from_zarr(grp):
    """Load RawDataset continuous raw data from Zarr group (low-level function)."""
    description = pd.Series(grp.attrs["description"])
    info_dict = _restore_nan_from_json(grp.attrs["info"])

    data = grp["data"][:]

    target_name = grp.attrs.get("target_name", None)

    return data, description, info_dict, target_name


# ---------------------------------------------------------------------------
# ZarrBackend
# ---------------------------------------------------------------------------

class ZarrBackend:
    """Zarr storage backend — compressed, eager-loaded."""

    name = "zarr"

    def validate_dependencies(self):
        if zarr is False:
            raise ImportError(
                "Zarr is not installed. Install with: pip install braindecode[hub]"
            )

    def get_data_filename(self) -> str:
        return "dataset.zarr"

    def build_format_info(self, format_params: dict) -> dict:
        return {
            "format": "zarr",
            "compression": format_params.get("compression", "blosc"),
            "compression_level": format_params.get("compression_level", 5),
            "chunk_size": format_params.get("chunk_size", 5_000_000),
        }

    def convert_datasets(self, datasets, output_path: Path, format_params: dict):
        """Convert a list of datasets to Zarr format on disk."""
        self.validate_dependencies()

        if output_path.exists():
            raise FileExistsError(
                f"{output_path} already exists. Set overwrite=True to replace it."
            )

        compression = format_params.get("compression", "blosc")
        compression_level = format_params.get("compression_level", 5)
        chunk_size = format_params.get("chunk_size", 5_000_000)

        # Create zarr store (zarr v3 API)
        root = zarr.open(str(output_path), mode="w")

        # Validate uniformity across all datasets using shared validation
        dataset_type, _, _ = hub_validation.validate_dataset_uniformity(datasets)

        first_ds = datasets[0]

        # Store global metadata
        root.attrs["n_datasets"] = len(datasets)
        root.attrs["dataset_type"] = dataset_type
        root.attrs["braindecode_version"] = braindecode.__version__

        # Track dependency versions for reproducibility
        root.attrs["mne_version"] = mne.__version__
        root.attrs["numpy_version"] = np.__version__
        root.attrs["pandas_version"] = pd.__version__
        root.attrs["zarr_version"] = zarr.__version__
        root.attrs["scipy_version"] = scipy.__version__

        # Save preprocessing kwargs
        for kwarg_name in [
            "raw_preproc_kwargs",
            "window_kwargs",
            "window_preproc_kwargs",
        ]:
            if hasattr(first_ds, kwarg_name):
                kwargs = getattr(first_ds, kwarg_name)
                if kwargs:
                    root.attrs[kwarg_name] = json.dumps(kwargs)

        # Create compressor
        compressor = _create_compressor(compression, compression_level)

        # Save each recording
        for i_ds, ds in enumerate(datasets):
            grp = root.create_group(f"recording_{i_ds}")

            if dataset_type == "WindowsDataset":
                data = ds.windows.get_data()
                metadata = ds.windows.metadata
                description = ds.description
                info_dict = ds.windows.info.to_json_dict()
                target_name = ds.target_name if hasattr(ds, "target_name") else None

                _save_windows_to_zarr(
                    grp, data, metadata, description, info_dict,
                    compressor, target_name, chunk_size,
                )

            elif dataset_type == "EEGWindowsDataset":
                raw = ds.raw
                metadata = ds.metadata
                description = ds.description
                info_dict = ds.raw.info.to_json_dict()
                targets_from = ds.targets_from
                last_target_only = ds.last_target_only

                _save_eegwindows_to_zarr(
                    grp, raw, metadata, description, info_dict,
                    targets_from, last_target_only, compressor, chunk_size,
                )

            elif dataset_type == "RawDataset":
                raw = ds.raw
                description = ds.description
                info_dict = ds.raw.info.to_json_dict()
                target_name = ds.target_name if hasattr(ds, "target_name") else None

                _save_raw_to_zarr(
                    grp, raw, description, info_dict,
                    target_name, compressor, chunk_size,
                )

    def load_datasets(self, input_path: Path, preload: bool):
        """Load datasets from a Zarr store on disk."""
        self.validate_dependencies()

        if not input_path.exists():
            raise FileNotFoundError(f"{input_path} does not exist.")

        root = zarr.open(str(input_path), mode="r")

        n_datasets = root.attrs["n_datasets"]
        dataset_type = root.attrs["dataset_type"]

        WindowsDataset = get_dataset_class("WindowsDataset")
        EEGWindowsDataset = get_dataset_class("EEGWindowsDataset")
        RawDataset = get_dataset_class("RawDataset")
        BaseConcatDataset = get_dataset_class("BaseConcatDataset")

        if not preload:
            warnings.warn(
                "Lazy loading (preload=False) is not supported by the Zarr "
                "backend. All data will be loaded into memory. Use the 'mne' "
                "format for lazy loading support.",
                UserWarning,
            )

        datasets = []
        for i_ds in range(n_datasets):
            grp = root[f"recording_{i_ds}"]

            if dataset_type == "WindowsDataset":
                data, metadata, description, info_dict, target_name = (
                    _load_windows_from_zarr(grp)
                )

                info = Info.from_json_dict(info_dict)
                targets = metadata["target"].values
                if np.issubdtype(targets.dtype, np.integer):
                    event_ids = targets
                else:
                    event_ids = np.ones(len(metadata), dtype=int)
                events = np.column_stack(
                    [
                        metadata["i_start_in_trial"].values.astype(int),
                        np.zeros(len(metadata), dtype=int),
                        event_ids,
                    ]
                )
                epochs = mne.EpochsArray(data, info, events=events, metadata=metadata)
                ds = WindowsDataset(epochs, description)
                if target_name is not None:
                    ds.target_name = target_name

            elif dataset_type == "EEGWindowsDataset":
                (
                    data, metadata, description, info_dict,
                    targets_from, last_target_only,
                ) = _load_eegwindows_from_zarr(grp)

                info = Info.from_json_dict(info_dict)
                raw = mne.io.RawArray(data, info)
                ds = EEGWindowsDataset(
                    raw=raw,
                    metadata=metadata,
                    description=description,
                    targets_from=targets_from,
                    last_target_only=last_target_only,
                )

            elif dataset_type == "RawDataset":
                data, description, info_dict, target_name = _load_raw_from_zarr(grp)

                info = Info.from_json_dict(info_dict)
                raw = mne.io.RawArray(data, info)
                ds = RawDataset(raw, description)
                if target_name is not None:
                    ds.target_name = target_name

            else:
                raise ValueError(f"Unsupported dataset_type: {dataset_type}")

            datasets.append(ds)

        concat_ds = BaseConcatDataset(datasets)

        # Restore preprocessing kwargs
        for kwarg_name in [
            "raw_preproc_kwargs",
            "window_kwargs",
            "window_preproc_kwargs",
        ]:
            if kwarg_name in root.attrs:
                kwargs = json.loads(root.attrs[kwarg_name])
                for ds in datasets:
                    setattr(ds, kwarg_name, kwargs)

        return concat_ds


register_format(ZarrBackend())
