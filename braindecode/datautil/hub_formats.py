"""
Format converters for Hugging Face Hub integration.

This module provides Zarr format converters to transform EEG datasets for
efficient storage and fast random access during training on the Hugging Face Hub.

This is a high-level wrapper around hub_formats_core that works with dataset objects.
"""

# Authors: Kuntal Kokate
#
# License: BSD (3-clause)

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np

# Import core functions from hub module
from ..datasets import hub as hub_core
# Import registry for dynamic class lookup (avoids circular imports)
from ..datasets.registry import get_dataset_class, get_dataset_type

# Optional dependencies
ZARR_AVAILABLE = False
try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    pass

# Import dataset classes for type checking only
if TYPE_CHECKING:
    from ..datasets.base import BaseConcatDataset


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

    # Identify dataset type using registry
    dataset_type = get_dataset_type(first_ds)

    if dataset_type == "BaseDataset":
        # BaseDataset represents abstract base class
        raise NotImplementedError(
            "Saving BaseDataset to Hub is not yet supported. "
            "Please create windows from your dataset using "
            "braindecode.preprocessing.create_windows_from_events() or "
            "create_fixed_length_windows() before uploading to Hub."
        )
    elif dataset_type not in ["WindowsDataset", "EEGWindowsDataset", "RawDataset"]:
        raise TypeError(f"Unsupported dataset type: {dataset_type}")

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
    compressor = hub_core._create_compressor(compression, compression_level)

    # Save each recording
    for i_ds, ds in enumerate(dataset.datasets):
        grp = root.create_group(f"recording_{i_ds}")

        if dataset_type == "WindowsDataset":
            _save_windows_dataset_zarr(grp, ds, compressor)
        elif dataset_type == "EEGWindowsDataset":
            _save_eegwindows_dataset_zarr(grp, ds, compressor)
        elif dataset_type == "RawDataset":
            _save_raw_dataset_zarr(grp, ds, compressor)

    return output_path


def _save_windows_dataset_zarr(
    grp: zarr.Group, ds: WindowsDataset, compressor
) -> None:
    """Save a WindowsDataset (with mne.Epochs) to Zarr group."""
    # Extract data from dataset
    data = ds.windows.get_data()  # Shape: (n_epochs, n_channels, n_times)
    metadata = ds.windows.metadata
    description = ds.description
    info_dict = hub_core._mne_info_to_dict(ds.windows.info)
    target_name = ds.target_name if hasattr(ds, "target_name") else None

    # Call core function to save
    hub_core._save_windows_to_zarr(
        grp, data, metadata, description, info_dict, compressor, target_name
    )


def _save_eegwindows_dataset_zarr(
    grp: zarr.Group, ds: EEGWindowsDataset, compressor
) -> None:
    """Save an EEGWindowsDataset (continuous raw with metadata) to Zarr group."""
    # Get continuous raw data and metadata from dataset
    raw = ds.raw
    metadata = ds.metadata
    description = ds.description
    info_dict = hub_core._mne_info_to_dict(ds.raw.info)
    targets_from = ds.targets_from
    last_target_only = ds.last_target_only

    # Call core function to save (saves continuous raw directly)
    hub_core._save_eegwindows_to_zarr(
        grp, raw, metadata, description, info_dict,
        targets_from, last_target_only, compressor
    )


def _save_raw_dataset_zarr(
    grp: zarr.Group, ds, compressor
) -> None:
    """Save a RawDataset (continuous raw data without windows) to Zarr group."""
    # Get continuous raw data from dataset
    raw = ds.raw
    description = ds.description
    info_dict = hub_core._mne_info_to_dict(ds.raw.info)
    target_name = ds.target_name if hasattr(ds, "target_name") else None

    # Call core function to save
    hub_core._save_raw_to_zarr(
        grp, raw, description, info_dict, target_name, compressor
    )


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
            "Zarr is not installed. Install with: pip install braindecode[hub]"
        )

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
        elif dataset_type == "RawDataset":
            ds = _load_raw_dataset_zarr(grp, preload)
        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")

        datasets.append(ds)

    # Create concat dataset using registry
    BaseConcatDataset = get_dataset_class("BaseConcatDataset")
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

    # Load raw data using core function
    data, metadata, description, info_dict, target_name = hub_core._load_windows_from_zarr(
        grp, preload
    )

    # Convert info dict back to MNE Info
    info = hub_core._dict_to_mne_info(info_dict)

    # Create Epochs object
    events = np.column_stack([
        metadata["i_start_in_trial"].values,
        np.zeros(len(metadata), dtype=int),
        metadata["target"].values,
    ])

    epochs = mne.EpochsArray(data, info, events=events, metadata=metadata)

    # Create dataset using registry
    WindowsDataset = get_dataset_class("WindowsDataset")
    ds = WindowsDataset(epochs, description)

    # Restore target name
    if target_name is not None:
        ds.target_name = target_name

    return ds


def _load_eegwindows_dataset_zarr(grp: zarr.Group, preload: bool):
    """Load an EEGWindowsDataset (windowed data with mne.Raw) from Zarr group."""
    import mne

    # Load raw data using core function
    data, metadata, description, info_dict, targets_from, last_target_only = (
        hub_core._load_eegwindows_from_zarr(grp, preload)
    )

    # Convert info dict back to MNE Info
    info = hub_core._dict_to_mne_info(info_dict)

    # EEGWindowsDataset stores continuous raw data [n_channels, n_timepoints]
    # Data is already in continuous format, use directly
    raw = mne.io.RawArray(data, info)

    # Create dataset using registry
    EEGWindowsDataset = get_dataset_class("EEGWindowsDataset")
    ds = EEGWindowsDataset(
        raw=raw,
        metadata=metadata,
        description=description,
        targets_from=targets_from,
        last_target_only=last_target_only,
    )

    return ds


def _load_raw_dataset_zarr(grp: zarr.Group, preload: bool):
    """Load a RawDataset (continuous raw data without windows) from Zarr group."""
    import mne

    # Load raw data using core function
    data, description, info_dict, target_name = (
        hub_core._load_raw_from_zarr(grp, preload)
    )

    # Convert info dict back to MNE Info
    info = hub_core._dict_to_mne_info(info_dict)

    # Data is in continuous format [n_channels, n_timepoints]
    raw = mne.io.RawArray(data, info)

    # Create dataset using registry
    RawDataset = get_dataset_class("RawDataset")
    ds = RawDataset(raw, description)

    # Restore target name
    if target_name is not None:
        ds.target_name = target_name

    return ds


# =============================================================================
# Utility Functions
# =============================================================================


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

    # Determine dataset type from first dataset using registry
    first_ds = dataset.datasets[0]
    dataset_type = get_dataset_type(first_ds)

    if dataset_type == "WindowsDataset":
        first_ch_names = first_ds.windows.ch_names
        first_sfreq = first_ds.windows.info["sfreq"]
    elif dataset_type == "EEGWindowsDataset":
        first_ch_names = first_ds.raw.ch_names
        first_sfreq = first_ds.raw.info["sfreq"]
    elif dataset_type == "RawDataset":
        first_ch_names = first_ds.raw.ch_names
        first_sfreq = first_ds.raw.info["sfreq"]
    elif dataset_type == "BaseDataset":
        raise NotImplementedError(
            "Hub operations with BaseDataset are not supported. "
            "Please create windows from your dataset first."
        )
    else:
        raise TypeError(f"Unsupported dataset type: {dataset_type}")

    # Validate uniformity across all datasets
    for i, ds in enumerate(dataset.datasets):
        # Check type consistency using registry
        ds_type = get_dataset_type(ds)
        if ds_type != dataset_type:
            raise ValueError(
                f"Mixed dataset types in concat: dataset 0 is {dataset_type} "
                f"but dataset {i} is {ds_type}"
            )

        if dataset_type == "WindowsDataset":
            if ds.windows.ch_names != first_ch_names:
                raise ValueError(
                    f"Inconsistent channel names: dataset 0 has {first_ch_names} "
                    f"but dataset {i} has {ds.windows.ch_names}"
                )
            if ds.windows.info["sfreq"] != first_sfreq:
                raise ValueError(
                    f"Inconsistent sampling frequencies: dataset 0 has {first_sfreq} Hz "
                    f"but dataset {i} has {ds.windows.info['sfreq']} Hz. "
                    f"Please resample all datasets to a common frequency before saving. "
                    f"Example:\n"
                    f"  from braindecode.preprocessing import preprocess, Preprocessor\n"
                    f"  preprocessors = [Preprocessor('resample', sfreq={first_sfreq})]\n"
                    f"  preprocess(concat_ds, preprocessors)"
                )
        elif dataset_type == "EEGWindowsDataset":
            if ds.raw.ch_names != first_ch_names:
                raise ValueError(
                    f"Inconsistent channel names: dataset 0 has {first_ch_names} "
                    f"but dataset {i} has {ds.raw.ch_names}"
                )
            if ds.raw.info["sfreq"] != first_sfreq:
                raise ValueError(
                    f"Inconsistent sampling frequencies: dataset 0 has {first_sfreq} Hz "
                    f"but dataset {i} has {ds.raw.info['sfreq']} Hz. "
                    f"Please resample all datasets to a common frequency before saving. "
                    f"Example:\n"
                    f"  from braindecode.preprocessing import preprocess, Preprocessor\n"
                    f"  preprocessors = [Preprocessor('resample', sfreq={first_sfreq})]\n"
                    f"  preprocess(concat_ds, preprocessors)"
                )
        elif dataset_type == "RawDataset":
            if ds.raw.ch_names != first_ch_names:
                raise ValueError(
                    f"Inconsistent channel names: dataset 0 has {first_ch_names} "
                    f"but dataset {i} has {ds.raw.ch_names}"
                )
            if ds.raw.info["sfreq"] != first_sfreq:
                raise ValueError(
                    f"Inconsistent sampling frequencies: dataset 0 has {first_sfreq} Hz "
                    f"but dataset {i} has {ds.raw.info['sfreq']} Hz. "
                    f"Please resample all datasets to a common frequency before saving. "
                    f"Example:\n"
                    f"  from braindecode.preprocessing import preprocess, Preprocessor\n"
                    f"  preprocessors = [Preprocessor('resample', sfreq={first_sfreq})]\n"
                    f"  preprocess(concat_ds, preprocessors)"
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
        elif dataset_type == "RawDataset":
            # RawDataset has continuous raw data without windows
            # Use number of timepoints as "samples"
            data = ds.raw.get_data()
            total_samples += data.shape[1]  # Number of timepoints

        total_size_mb += data.nbytes / (1024 * 1024)

    n_recordings = len(dataset.datasets)

    return {
        "n_recordings": n_recordings,
        "total_samples": total_samples,
        "total_size_mb": round(total_size_mb, 2),
    }
