"""
Format converters for Hugging Face Hub integration.

This module provides Zarr format converters to transform EEG datasets for
efficient storage and fast random access during training on the Hugging Face Hub.
"""

# Authors: Kuntal Kokate
#
# License: BSD (3-clause)

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False

# Import dataset classes for type checking
if TYPE_CHECKING:
    from ..datasets.base import BaseConcatDataset

# Runtime imports for isinstance checks
from ..datasets.base import BaseDataset, EEGWindowsDataset, WindowsDataset


# =============================================================================
# Zarr Format Converters
# =============================================================================


def convert_to_zarr(
    dataset: BaseConcatDataset,
    output_path: Union[str, Path],
    compression: str = "blosc",
    compression_level: int = 5,
    overwrite: bool = False,
) -> Path:
    """Convert BaseConcatDataset to Zarr format.

    Zarr provides cloud-native chunked storage, optimized for random access
    during training. This is the format used for Hugging Face Hub uploads,
    based on comprehensive benchmarking showing:
    - Fastest random access: 0.010 ms (critical for PyTorch DataLoader)
    - Fast save/load: 0.46s / 0.12s
    - Good compression: ~23% size reduction with blosc

    Parameters
    ----------
    dataset : BaseConcatDataset
        The dataset to convert.
    output_path : str | Path
        Path where the Zarr directory will be created.
    compression : str, default="blosc"
        Compression algorithm. Options: "blosc" (recommended), "zstd", "gzip", None.
        blosc uses zstd codec by default, providing best balance of speed and compression.
    compression_level : int, default=5
        Compression level (0-9). Level 5 provides optimal balance based on benchmarks.
    overwrite : bool, default=False
        Whether to overwrite existing directory.

    Returns
    -------
    Path
        Path to the created Zarr directory.

    Notes
    -----
    The chunking strategy is optimized for random access:
    - Windowed data: Each window is a separate chunk (1, n_channels, n_times)
    - Raw data: Chunks of (n_channels, 10000) samples

    Examples
    --------
    >>> dataset = NMT(path=path, preload=True)
    >>> # Use default settings (optimal from benchmarks)
    >>> zarr_path = convert_to_zarr(dataset, "dataset.zarr")
    >>>
    >>> # Or customize compression
    >>> zarr_path = convert_to_zarr(
    ...     dataset, "dataset.zarr",
    ...     compression="blosc",
    ...     compression_level=5
    ... )
    """
    if not ZARR_AVAILABLE:
        raise ImportError(
            "Zarr is not installed. Install with: pip install zarr"
        )

    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Set overwrite=True to replace it."
        )

    # Create zarr store
    store = zarr.DirectoryStore(output_path)
    root = zarr.group(store=store, overwrite=overwrite)

    # Determine dataset types - check first dataset for consistency
    first_ds = dataset.datasets[0]

    # Identify dataset type using isinstance (order matters - check subclasses first)
    if isinstance(first_ds, WindowsDataset):
        dataset_type = "WindowsDataset"
    elif isinstance(first_ds, EEGWindowsDataset):
        dataset_type = "EEGWindowsDataset"
    elif isinstance(first_ds, BaseDataset):
        # BaseDataset represents continuous (non-windowed) raw data
        raise NotImplementedError(
            "Saving continuous BaseDataset (non-windowed raw data) to Hub is not yet "
            "supported. Please create windows from your dataset using "
            "braindecode.preprocessing.create_windows_from_events() or "
            "create_fixed_length_windows() before uploading to Hub."
        )
    else:
        raise TypeError(f"Unsupported dataset type: {type(first_ds)}")

    # Store global metadata
    root.attrs["n_datasets"] = len(dataset.datasets)
    root.attrs["dataset_type"] = dataset_type
    root.attrs["braindecode_version"] = "1.0"

    # Save preprocessing kwargs
    for kwarg_name in ["raw_preproc_kwargs", "window_kwargs", "window_preproc_kwargs"]:
        if hasattr(dataset, kwarg_name):
            kwargs = getattr(dataset, kwarg_name)
            if kwargs:
                root.attrs[kwarg_name] = json.dumps(kwargs)

    # Determine compressor
    if compression == "blosc":
        compressor = zarr.Blosc(cname="zstd", clevel=compression_level)
    elif compression == "zstd":
        compressor = zarr.Blosc(cname="zstd", clevel=compression_level)
    elif compression == "gzip":
        compressor = zarr.Blosc(cname="gzip", clevel=compression_level)
    else:
        compressor = None

    # Save each recording
    for i_ds, ds in enumerate(dataset.datasets):
        grp = root.create_group(f"recording_{i_ds}")

        if dataset_type == "WindowsDataset":
            _save_windows_dataset_zarr(grp, ds, compressor)
        elif dataset_type == "EEGWindowsDataset":
            _save_eegwindows_dataset_zarr(grp, ds, compressor)

    return output_path


def _save_windows_dataset_zarr(
    grp: zarr.Group, ds: WindowsDataset, compressor
) -> None:
    """Save a WindowsDataset (with mne.Epochs) to Zarr group."""
    # Get all windows data
    data = ds.windows.get_data()  # Shape: (n_epochs, n_channels, n_times)

    # Save data with chunking for random access
    grp.create_dataset(
        "data",
        data=data.astype(np.float32),
        chunks=(1, data.shape[1], data.shape[2]),
        compressor=compressor,
    )

    # Save metadata
    metadata_json = ds.windows.metadata.to_json(orient="split", date_format="iso")
    grp.attrs["metadata"] = metadata_json

    # Save description
    description_json = ds.description.to_json(date_format="iso")
    grp.attrs["description"] = description_json

    # Save MNE info
    info_dict = _mne_info_to_dict(ds.windows.info)
    grp.attrs["info"] = json.dumps(info_dict)

    # Save target name
    if hasattr(ds, "target_name") and ds.target_name is not None:
        grp.attrs["target_name"] = ds.target_name


def _save_eegwindows_dataset_zarr(
    grp: zarr.Group, ds: EEGWindowsDataset, compressor
) -> None:
    """Save an EEGWindowsDataset (windowed data with mne.Raw) to Zarr group."""
    # EEGWindowsDataset has windowed data stored in raw attribute with metadata
    # We need to extract each window based on crop_inds
    windows_list = []
    for i in range(len(ds)):
        X, y, crop_inds = ds[i]
        windows_list.append(X)

    data = np.stack(windows_list, axis=0)  # Shape: (n_windows, n_channels, n_times)

    # Save data with chunking for random access
    grp.create_dataset(
        "data",
        data=data.astype(np.float32),
        chunks=(1, data.shape[1], data.shape[2]),
        compressor=compressor,
    )

    # Save metadata
    metadata_json = ds.metadata.to_json(orient="split", date_format="iso")
    grp.attrs["metadata"] = metadata_json

    # Save description
    description_json = ds.description.to_json(date_format="iso")
    grp.attrs["description"] = description_json

    # Save MNE info
    info_dict = _mne_info_to_dict(ds.raw.info)
    grp.attrs["info"] = json.dumps(info_dict)

    # Save targets_from and last_target_only settings
    grp.attrs["targets_from"] = ds.targets_from
    grp.attrs["last_target_only"] = ds.last_target_only


def load_from_zarr(
    input_path: Union[str, Path],
    preload: bool = True,
    ids_to_load: Optional[List[int]] = None,
):
    """Load BaseConcatDataset from Zarr format.

    Zarr is the format used for braindecode Hub datasets, providing
    the fastest random access performance for training with PyTorch.

    Parameters
    ----------
    input_path : str | Path
        Path to the Zarr directory.
    preload : bool, default=True
        Whether to load data into memory. If False, uses lazy loading
        (data is loaded on-demand during training).
    ids_to_load : list of int | None
        Specific recording IDs to load. If None, loads all.

    Returns
    -------
    BaseConcatDataset
        The loaded dataset.

    Examples
    --------
    >>> # Load from local zarr directory
    >>> dataset = load_from_zarr("dataset.zarr", preload=True)
    >>>
    >>> # Load from Hugging Face Hub (handled automatically)
    >>> from braindecode.datasets import BaseConcatDataset
    >>> dataset = BaseConcatDataset.from_pretrained("username/dataset-name")
    """
    if not ZARR_AVAILABLE:
        raise ImportError(
            "Zarr is not installed. Install with: pip install zarr"
        )

    # Prevent circular import
    from ..datasets.base import BaseConcatDataset

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist.")

    # Open zarr store
    store = zarr.DirectoryStore(input_path)
    root = zarr.group(store=store)

    n_datasets = root.attrs["n_datasets"]
    dataset_type = root.attrs.get("dataset_type", None)

    # For backwards compatibility with old format
    if dataset_type is None:
        # Old format used is_windowed attribute
        is_windowed = root.attrs.get("is_windowed", False)
        dataset_type = "WindowsDataset" if is_windowed else "BaseDataset"

    # Determine which datasets to load
    if ids_to_load is None:
        ids_to_load = list(range(n_datasets))

    datasets = []
    for i_ds in ids_to_load:
        grp = root[f"recording_{i_ds}"]

        if dataset_type == "WindowsDataset":
            ds = _load_windows_dataset_zarr(grp, preload)
        elif dataset_type == "EEGWindowsDataset":
            ds = _load_eegwindows_dataset_zarr(grp, preload)
        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")

        datasets.append(ds)

    # Create concat dataset
    concat_ds = BaseConcatDataset(datasets)

    # Restore preprocessing kwargs
    for kwarg_name in [
        "raw_preproc_kwargs",
        "window_kwargs",
        "window_preproc_kwargs",
    ]:
        if kwarg_name in root.attrs:
            kwargs = json.loads(root.attrs[kwarg_name])
            setattr(concat_ds, kwarg_name, kwargs)

    return concat_ds


def _load_windows_dataset_zarr(grp: zarr.Group, preload: bool):
    """Load a WindowsDataset (with mne.Epochs) from Zarr group."""
    import mne

    # Load metadata
    metadata = pd.read_json(grp.attrs["metadata"], orient="split")

    # Load description
    description = pd.read_json(grp.attrs["description"], typ="series")

    # Load info
    info_dict = json.loads(grp.attrs["info"])
    info = _dict_to_mne_info(info_dict)

    # Load data
    if preload:
        data = grp["data"][:]
    else:
        data = grp["data"][:]
        warnings.warn(
            "Lazy loading from Zarr not fully implemented yet. "
            "Loading all data into memory.",
            UserWarning,
        )

    # Create Epochs object
    events = np.column_stack([
        metadata["i_start_in_trial"].values,
        np.zeros(len(metadata), dtype=int),
        metadata["target"].values,
    ])

    epochs = mne.EpochsArray(data, info, events=events, metadata=metadata)

    # Create dataset
    ds = WindowsDataset(epochs, description)

    # Restore target name
    if "target_name" in grp.attrs:
        ds.target_name = grp.attrs["target_name"]

    return ds


def _load_eegwindows_dataset_zarr(grp: zarr.Group, preload: bool):
    """Load an EEGWindowsDataset (windowed data with mne.Raw) from Zarr group."""
    import mne

    # Load metadata
    metadata = pd.read_json(grp.attrs["metadata"], orient="split")

    # Load description
    description = pd.read_json(grp.attrs["description"], typ="series")

    # Load info
    info_dict = json.loads(grp.attrs["info"])
    info = _dict_to_mne_info(info_dict)

    # Load data
    if preload:
        data = grp["data"][:]
    else:
        data = grp["data"][:]
        warnings.warn(
            "Lazy loading from Zarr not fully implemented yet. "
            "Loading all data into memory.",
            UserWarning,
        )

    # EEGWindowsDataset stores windowed data but uses mne.Raw instead of mne.Epochs
    # We need to reconstruct the data in a format that EEGWindowsDataset expects
    # Concatenate all windows into a single continuous raw array
    n_windows, n_channels, n_times_per_window = data.shape
    continuous_data = data.reshape(n_channels, n_windows * n_times_per_window)

    # Create Raw object
    raw = mne.io.RawArray(continuous_data, info)

    # Load EEGWindowsDataset-specific attributes
    targets_from = grp.attrs.get("targets_from", "metadata")
    last_target_only = grp.attrs.get("last_target_only", True)

    # Create dataset
    ds = EEGWindowsDataset(
        raw=raw,
        metadata=metadata,
        description=description,
        targets_from=targets_from,
        last_target_only=last_target_only,
    )

    return ds


# =============================================================================
# Utility Functions
# =============================================================================


def _mne_info_to_dict(info) -> Dict:
    """Convert MNE Info object to dictionary for JSON serialization.

    Parameters
    ----------
    info : mne.Info
        MNE Info object.

    Returns
    -------
    dict
        Serializable dictionary with essential info.
    """
    return {
        "ch_names": info["ch_names"],
        "sfreq": float(info["sfreq"]),
        "ch_types": [str(ch_type) for ch_type in info.get_channel_types()],
        "lowpass": float(info["lowpass"]) if info["lowpass"] is not None else None,
        "highpass": float(info["highpass"]) if info["highpass"] is not None else None,
    }


def _dict_to_mne_info(info_dict: Dict):
    """Convert dictionary back to MNE Info object.

    Parameters
    ----------
    info_dict : dict
        Dictionary with channel info.

    Returns
    -------
    mne.Info
        MNE Info object.
    """
    import mne

    info = mne.create_info(
        ch_names=info_dict["ch_names"],
        sfreq=info_dict["sfreq"],
        ch_types=info_dict["ch_types"],
    )

    # Use _unlock() context manager to set lowpass/highpass
    # These cannot be set directly in newer MNE versions
    with info._unlock():
        if info_dict.get("lowpass") is not None:
            info["lowpass"] = info_dict["lowpass"]
        if info_dict.get("highpass") is not None:
            info["highpass"] = info_dict["highpass"]

    return info


def get_format_info(dataset: "BaseConcatDataset") -> Dict:
    """Get dataset information for Hub metadata.

    Validates that all datasets in the concat have uniform properties
    (channels, sampling frequency) and raises an error if not.

    Parameters
    ----------
    dataset : BaseConcatDataset
        The dataset to analyze.

    Returns
    -------
    dict
        Dictionary with dataset statistics and format info.

    Raises
    ------
    ValueError
        If datasets have inconsistent channels or sampling frequencies.
    """
    if len(dataset.datasets) == 0:
        raise ValueError("Cannot get format info for empty dataset")

    # Determine dataset type from first dataset
    first_ds = dataset.datasets[0]
    if isinstance(first_ds, WindowsDataset):
        dataset_type = "WindowsDataset"
        first_ch_names = first_ds.windows.ch_names
        first_sfreq = first_ds.windows.info["sfreq"]
    elif isinstance(first_ds, EEGWindowsDataset):
        dataset_type = "EEGWindowsDataset"
        first_ch_names = first_ds.raw.ch_names
        first_sfreq = first_ds.raw.info["sfreq"]
    else:
        raise TypeError(f"Unsupported dataset type: {type(first_ds)}")

    # Validate uniformity across all datasets
    for i, ds in enumerate(dataset.datasets):
        if dataset_type == "WindowsDataset":
            if not isinstance(ds, WindowsDataset):
                raise ValueError(
                    f"Mixed dataset types in concat: dataset 0 is WindowsDataset "
                    f"but dataset {i} is {type(ds).__name__}"
                )
            if ds.windows.ch_names != first_ch_names:
                raise ValueError(
                    f"Inconsistent channel names: dataset 0 has {first_ch_names} "
                    f"but dataset {i} has {ds.windows.ch_names}"
                )
            if ds.windows.info["sfreq"] != first_sfreq:
                raise ValueError(
                    f"Inconsistent sampling frequencies: dataset 0 has {first_sfreq} Hz "
                    f"but dataset {i} has {ds.windows.info['sfreq']} Hz"
                )
        elif dataset_type == "EEGWindowsDataset":
            if not isinstance(ds, EEGWindowsDataset):
                raise ValueError(
                    f"Mixed dataset types in concat: dataset 0 is EEGWindowsDataset "
                    f"but dataset {i} is {type(ds).__name__}"
                )
            if ds.raw.ch_names != first_ch_names:
                raise ValueError(
                    f"Inconsistent channel names: dataset 0 has {first_ch_names} "
                    f"but dataset {i} has {ds.raw.ch_names}"
                )
            if ds.raw.info["sfreq"] != first_sfreq:
                raise ValueError(
                    f"Inconsistent sampling frequencies: dataset 0 has {first_sfreq} Hz "
                    f"but dataset {i} has {ds.raw.info['sfreq']} Hz"
                )

    # Calculate dataset size
    total_samples = 0
    total_size_mb = 0

    for ds in dataset.datasets:
        if dataset_type == "WindowsDataset":
            data = ds.windows.get_data()
            total_samples += data.shape[0]
        elif dataset_type == "EEGWindowsDataset":
            # For EEGWindowsDataset, count number of windows from metadata
            total_samples += len(ds.metadata)
            # Estimate size by extracting windows
            for i in range(len(ds)):
                X, _, _ = ds[i]
                total_size_mb += X.nbytes / (1024 * 1024)
            continue  # Skip the size calculation below

        total_size_mb += data.nbytes / (1024 * 1024)

    n_recordings = len(dataset.datasets)

    return {
        "n_recordings": n_recordings,
        "total_samples": total_samples,
        "total_size_mb": round(total_size_mb, 2),
    }
