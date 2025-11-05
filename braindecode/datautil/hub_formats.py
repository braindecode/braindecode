"""
Format converters for Hugging Face Hub integration.

This module provides converters to transform EEG datasets from MNE's .fif format
to Hub-compatible formats (HDF5, Zarr, NumPy+Parquet) for efficient storage and
fast random access during training.
"""

# Authors: Kuntal Kokate, Bruno Aristimunha
#
# License: BSD (3-clause)

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd

try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from ..datasets.base import BaseConcatDataset, BaseDataset, WindowsDataset


# =============================================================================
# HDF5 Format Converters
# =============================================================================


def convert_to_hdf5(
    dataset: BaseConcatDataset,
    output_path: Union[str, Path],
    compression: str = "gzip",
    compression_level: int = 4,
    overwrite: bool = False,
) -> Path:
    """Convert BaseConcatDataset to HDF5 format.

    HDF5 provides hierarchical storage with excellent random access performance
    and built-in compression. Recommended for most use cases.

    Parameters
    ----------
    dataset : BaseConcatDataset
        The dataset to convert.
    output_path : str | Path
        Path where the HDF5 file will be saved.
    compression : str, default="gzip"
        Compression algorithm. Options: "gzip", "lzf", None.
    compression_level : int, default=4
        Compression level (0-9 for gzip). Higher means more compression but
        slower write/read. 4 is a good balance.
    overwrite : bool, default=False
        Whether to overwrite existing file.

    Returns
    -------
    Path
        Path to the created HDF5 file.

    Examples
    --------
    >>> dataset = NMT(path=path, preload=True)
    >>> hdf5_path = convert_to_hdf5(dataset, "dataset.h5")
    """
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Set overwrite=True to replace it."
        )

    # Determine if we have raw or windowed data
    is_windowed = hasattr(dataset.datasets[0], "windows")

    with h5py.File(output_path, "w") as f:
        # Store global metadata
        _save_global_metadata_hdf5(f, dataset, is_windowed)

        # Save each recording
        for i_ds, ds in enumerate(dataset.datasets):
            grp = f.create_group(f"recording_{i_ds}")

            if is_windowed:
                _save_windowed_dataset_hdf5(
                    grp, ds, compression, compression_level
                )
            else:
                _save_raw_dataset_hdf5(grp, ds, compression, compression_level)

    return output_path


def _save_global_metadata_hdf5(
    f: h5py.File, dataset: BaseConcatDataset, is_windowed: bool
) -> None:
    """Save global dataset metadata to HDF5 file."""
    f.attrs["n_datasets"] = len(dataset.datasets)
    f.attrs["is_windowed"] = is_windowed
    f.attrs["braindecode_version"] = "1.0"  # For compatibility tracking

    # Save preprocessing kwargs if available
    for kwarg_name in ["raw_preproc_kwargs", "window_kwargs", "window_preproc_kwargs"]:
        if hasattr(dataset, kwarg_name):
            kwargs = getattr(dataset, kwarg_name)
            if kwargs:
                f.attrs[kwarg_name] = json.dumps(kwargs)


def _save_windowed_dataset_hdf5(
    grp: h5py.Group, ds: WindowsDataset, compression: str, compression_level: int
) -> None:
    """Save a windowed dataset to HDF5 group."""
    # Get all windows data
    data = ds.windows.get_data()  # Shape: (n_epochs, n_channels, n_times)

    # Save data
    grp.create_dataset(
        "data",
        data=data.astype(np.float32),
        compression=compression,
        compression_opts=compression_level if compression == "gzip" else None,
        chunks=(1, data.shape[1], data.shape[2]),  # Chunk by epoch for random access
    )

    # Save metadata
    metadata = ds.windows.metadata
    metadata_json = metadata.to_json(orient="split", date_format="iso")
    grp.attrs["metadata"] = metadata_json

    # Save description
    description_json = ds.description.to_json(date_format="iso")
    grp.attrs["description"] = description_json

    # Save MNE info
    info_dict = _mne_info_to_dict(ds.windows.info)
    grp.attrs["info"] = json.dumps(info_dict)

    # Save target name if available
    if hasattr(ds, "target_name") and ds.target_name is not None:
        grp.attrs["target_name"] = ds.target_name


def _save_raw_dataset_hdf5(
    grp: h5py.Group, ds: BaseDataset, compression: str, compression_level: int
) -> None:
    """Save a raw continuous dataset to HDF5 group."""
    # Get raw data
    data = ds.raw.get_data()  # Shape: (n_channels, n_times)

    # Save data
    grp.create_dataset(
        "data",
        data=data.astype(np.float32),
        compression=compression,
        compression_opts=compression_level if compression == "gzip" else None,
        chunks=(data.shape[0], min(10000, data.shape[1])),  # Chunk for efficient access
    )

    # Save description
    description_json = ds.description.to_json(date_format="iso")
    grp.attrs["description"] = description_json

    # Save MNE info
    info_dict = _mne_info_to_dict(ds.raw.info)
    grp.attrs["info"] = json.dumps(info_dict)

    # Save target name
    if hasattr(ds, "target_name") and ds.target_name is not None:
        grp.attrs["target_name"] = ds.target_name


def load_from_hdf5(
    input_path: Union[str, Path],
    preload: bool = True,
    ids_to_load: Optional[List[int]] = None,
):
    """Load BaseConcatDataset from HDF5 format.

    Parameters
    ----------
    input_path : str | Path
        Path to the HDF5 file.
    preload : bool, default=True
        Whether to load data into memory. If False, uses lazy loading.
    ids_to_load : list of int | None
        Specific recording IDs to load. If None, loads all.

    Returns
    -------
    BaseConcatDataset
        The loaded dataset.

    Examples
    --------
    >>> dataset = load_from_hdf5("dataset.h5", preload=True)
    >>> dataset = load_from_hdf5("dataset.h5", ids_to_load=[0, 1, 2])
    """
    # Import at runtime to avoid circular dependency
    from ..datasets.base import BaseConcatDataset

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist.")

    with h5py.File(input_path, "r") as f:
        n_datasets = f.attrs["n_datasets"]
        is_windowed = f.attrs["is_windowed"]

        # Determine which datasets to load
        if ids_to_load is None:
            ids_to_load = list(range(n_datasets))

        datasets = []
        for i_ds in ids_to_load:
            grp = f[f"recording_{i_ds}"]

            if is_windowed:
                ds = _load_windowed_dataset_hdf5(grp, preload)
            else:
                ds = _load_raw_dataset_hdf5(grp, preload)

            datasets.append(ds)

        # Create concat dataset
        concat_ds = BaseConcatDataset(datasets)

        # Restore preprocessing kwargs
        for kwarg_name in [
            "raw_preproc_kwargs",
            "window_kwargs",
            "window_preproc_kwargs",
        ]:
            if kwarg_name in f.attrs:
                kwargs = json.loads(f.attrs[kwarg_name])
                setattr(concat_ds, kwarg_name, kwargs)

    return concat_ds


def _load_windowed_dataset_hdf5(grp: h5py.Group, preload: bool):
    """Load a windowed dataset from HDF5 group."""
    import mne
    # Import at runtime to avoid circular dependency
    from ..datasets.base import WindowsDataset

    # Load metadata
    metadata = pd.read_json(grp.attrs["metadata"], orient="split")

    # Load description
    description = pd.read_json(grp.attrs["description"], typ="series")

    # Load info
    info_dict = json.loads(grp.attrs["info"])
    info = _dict_to_mne_info(info_dict)

    # Load or reference data
    if preload:
        data = grp["data"][:]
    else:
        # For lazy loading, we'll need to keep reference to file
        # This is a simplified version - full implementation would need
        # a lazy data wrapper
        data = grp["data"][:]
        warnings.warn(
            "Lazy loading from HDF5 not fully implemented yet. "
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

    # Restore target name if available
    if "target_name" in grp.attrs:
        ds.target_name = grp.attrs["target_name"]

    return ds


def _load_raw_dataset_hdf5(grp: h5py.Group, preload: bool):
    """Load a raw dataset from HDF5 group."""
    import mne
    # Import at runtime to avoid circular dependency
    from ..datasets.base import BaseDataset

    # Load description
    description = pd.read_json(grp.attrs["description"], typ="series")

    # Load info
    info_dict = json.loads(grp.attrs["info"])
    info = _dict_to_mne_info(info_dict)

    # Load or reference data
    if preload:
        data = grp["data"][:]
    else:
        # Similar to above, simplified lazy loading
        data = grp["data"][:]
        warnings.warn(
            "Lazy loading from HDF5 not fully implemented yet. "
            "Loading all data into memory.",
            UserWarning,
        )

    # Create Raw object
    raw = mne.io.RawArray(data, info)

    # Get target name
    target_name = grp.attrs.get("target_name", None)

    # Create dataset
    ds = BaseDataset(raw, description, target_name=target_name)

    return ds


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

    Zarr provides cloud-native chunked storage, ideal for very large datasets
    and streaming from cloud storage.

    Parameters
    ----------
    dataset : BaseConcatDataset
        The dataset to convert.
    output_path : str | Path
        Path where the Zarr directory will be created.
    compression : str, default="blosc"
        Compression algorithm. Options: "blosc", "zstd", "gzip", None.
    compression_level : int, default=5
        Compression level (0-9). Higher means more compression.
    overwrite : bool, default=False
        Whether to overwrite existing directory.

    Returns
    -------
    Path
        Path to the created Zarr directory.

    Examples
    --------
    >>> dataset = NMT(path=path, preload=True)
    >>> zarr_path = convert_to_zarr(dataset, "dataset.zarr")
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

    # Determine if we have raw or windowed data
    is_windowed = hasattr(dataset.datasets[0], "windows")

    # Store global metadata
    root.attrs["n_datasets"] = len(dataset.datasets)
    root.attrs["is_windowed"] = is_windowed
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

        if is_windowed:
            _save_windowed_dataset_zarr(grp, ds, compressor)
        else:
            _save_raw_dataset_zarr(grp, ds, compressor)

    return output_path


def _save_windowed_dataset_zarr(
    grp: zarr.Group, ds: WindowsDataset, compressor
) -> None:
    """Save a windowed dataset to Zarr group."""
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


def _save_raw_dataset_zarr(grp: zarr.Group, ds: BaseDataset, compressor) -> None:
    """Save a raw continuous dataset to Zarr group."""
    # Get raw data
    data = ds.raw.get_data()  # Shape: (n_channels, n_times)

    # Save data
    grp.create_dataset(
        "data",
        data=data.astype(np.float32),
        chunks=(data.shape[0], min(10000, data.shape[1])),
        compressor=compressor,
    )

    # Save description
    description_json = ds.description.to_json(date_format="iso")
    grp.attrs["description"] = description_json

    # Save MNE info
    info_dict = _mne_info_to_dict(ds.raw.info)
    grp.attrs["info"] = json.dumps(info_dict)

    # Save target name
    if hasattr(ds, "target_name") and ds.target_name is not None:
        grp.attrs["target_name"] = ds.target_name


def load_from_zarr(
    input_path: Union[str, Path],
    preload: bool = True,
    ids_to_load: Optional[List[int]] = None,
):
    """Load BaseConcatDataset from Zarr format.

    Parameters
    ----------
    input_path : str | Path
        Path to the Zarr directory.
    preload : bool, default=True
        Whether to load data into memory.
    ids_to_load : list of int | None
        Specific recording IDs to load. If None, loads all.

    Returns
    -------
    BaseConcatDataset
        The loaded dataset.

    Examples
    --------
    >>> dataset = load_from_zarr("dataset.zarr", preload=True)
    """
    if not ZARR_AVAILABLE:
        raise ImportError(
            "Zarr is not installed. Install with: pip install zarr"
        )

    # Import at runtime to avoid circular dependency
    from ..datasets.base import BaseConcatDataset

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist.")

    # Open zarr store
    store = zarr.DirectoryStore(input_path)
    root = zarr.group(store=store)

    n_datasets = root.attrs["n_datasets"]
    is_windowed = root.attrs["is_windowed"]

    # Determine which datasets to load
    if ids_to_load is None:
        ids_to_load = list(range(n_datasets))

    datasets = []
    for i_ds in ids_to_load:
        grp = root[f"recording_{i_ds}"]

        if is_windowed:
            ds = _load_windowed_dataset_zarr(grp, preload)
        else:
            ds = _load_raw_dataset_zarr(grp, preload)

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


def _load_windowed_dataset_zarr(grp: zarr.Group, preload: bool):
    """Load a windowed dataset from Zarr group."""
    import mne
    # Import at runtime to avoid circular dependency
    from ..datasets.base import WindowsDataset

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


def _load_raw_dataset_zarr(grp: zarr.Group, preload: bool):
    """Load a raw dataset from Zarr group."""
    import mne
    # Import at runtime to avoid circular dependency
    from ..datasets.base import BaseDataset

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

    # Create Raw object
    raw = mne.io.RawArray(data, info)

    # Get target name
    target_name = grp.attrs.get("target_name", None)

    # Create dataset
    ds = BaseDataset(raw, description, target_name=target_name)

    return ds


# =============================================================================
# NumPy + Parquet Format Converters
# =============================================================================


def convert_to_npz_parquet(
    dataset: BaseConcatDataset,
    output_path: Union[str, Path],
    compression: str = "zstd",
    overwrite: bool = False,
) -> Path:
    """Convert BaseConcatDataset to NumPy (.npz) + Parquet format.

    Uses .npz for signal arrays and Parquet for metadata. Simple and lightweight,
    but less efficient for very large datasets.

    Parameters
    ----------
    dataset : BaseConcatDataset
        The dataset to convert.
    output_path : str | Path
        Path where the directory will be created.
    compression : str, default="zstd"
        Compression for Parquet files. Options: "zstd", "gzip", "snappy", None.
    overwrite : bool, default=False
        Whether to overwrite existing directory.

    Returns
    -------
    Path
        Path to the created directory.

    Examples
    --------
    >>> dataset = NMT(path=path, preload=True)
    >>> npz_path = convert_to_npz_parquet(dataset, "dataset_npz")
    """
    if not PYARROW_AVAILABLE:
        raise ImportError(
            "PyArrow is not installed. Install with: pip install pyarrow"
        )

    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Set overwrite=True to replace it."
        )

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=overwrite)

    # Determine if we have raw or windowed data
    is_windowed = hasattr(dataset.datasets[0], "windows")

    # Save global metadata
    global_metadata = {
        "n_datasets": len(dataset.datasets),
        "is_windowed": is_windowed,
        "braindecode_version": "1.0",
    }

    # Save preprocessing kwargs
    for kwarg_name in ["raw_preproc_kwargs", "window_kwargs", "window_preproc_kwargs"]:
        if hasattr(dataset, kwarg_name):
            kwargs = getattr(dataset, kwarg_name)
            if kwargs:
                global_metadata[kwarg_name] = json.dumps(kwargs)

    with open(output_path / "metadata.json", "w") as f:
        json.dump(global_metadata, f, indent=2)

    # Save each recording
    all_descriptions = []
    all_metadata = []

    for i_ds, ds in enumerate(dataset.datasets):
        # Save data as npz
        data_path = output_path / f"recording_{i_ds}.npz"

        if is_windowed:
            data = ds.windows.get_data().astype(np.float32)
            np.savez_compressed(data_path, data=data)

            # Collect metadata
            metadata_df = ds.windows.metadata.copy()
            metadata_df["recording_id"] = i_ds
            all_metadata.append(metadata_df)
        else:
            data = ds.raw.get_data().astype(np.float32)
            np.savez_compressed(data_path, data=data)

        # Save info and description
        info_dict = _mne_info_to_dict(
            ds.windows.info if is_windowed else ds.raw.info
        )
        info_path = output_path / f"recording_{i_ds}_info.json"
        with open(info_path, "w") as f:
            json.dump(info_dict, f, indent=2)

        # Collect descriptions
        desc_dict = ds.description.to_dict()
        desc_dict["recording_id"] = i_ds
        if hasattr(ds, "target_name") and ds.target_name is not None:
            desc_dict["target_name"] = ds.target_name
        all_descriptions.append(desc_dict)

    # Save all descriptions as Parquet
    descriptions_df = pd.DataFrame(all_descriptions)
    pq.write_table(
        pa.Table.from_pandas(descriptions_df),
        output_path / "descriptions.parquet",
        compression=compression,
    )

    # Save all window metadata as Parquet (if windowed)
    if is_windowed and all_metadata:
        metadata_df = pd.concat(all_metadata, ignore_index=True)
        pq.write_table(
            pa.Table.from_pandas(metadata_df),
            output_path / "windows_metadata.parquet",
            compression=compression,
        )

    return output_path


def load_from_npz_parquet(
    input_path: Union[str, Path],
    preload: bool = True,
    ids_to_load: Optional[List[int]] = None,
):
    """Load BaseConcatDataset from NumPy + Parquet format.

    Parameters
    ----------
    input_path : str | Path
        Path to the directory.
    preload : bool, default=True
        Whether to load data into memory.
    ids_to_load : list of int | None
        Specific recording IDs to load. If None, loads all.

    Returns
    -------
    BaseConcatDataset
        The loaded dataset.

    Examples
    --------
    >>> dataset = load_from_npz_parquet("dataset_npz", preload=True)
    """
    if not PYARROW_AVAILABLE:
        raise ImportError(
            "PyArrow is not installed. Install with: pip install pyarrow"
        )

    # Import at runtime to avoid circular dependency
    from ..datasets.base import BaseConcatDataset, BaseDataset, WindowsDataset

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist.")

    # Load global metadata
    with open(input_path / "metadata.json", "r") as f:
        global_metadata = json.load(f)

    n_datasets = global_metadata["n_datasets"]
    is_windowed = global_metadata["is_windowed"]

    # Load descriptions
    descriptions_df = pq.read_table(input_path / "descriptions.parquet").to_pandas()

    # Load window metadata if windowed
    if is_windowed:
        windows_metadata_path = input_path / "windows_metadata.parquet"
        if windows_metadata_path.exists():
            windows_metadata_df = pq.read_table(windows_metadata_path).to_pandas()
        else:
            windows_metadata_df = None
    else:
        windows_metadata_df = None

    # Determine which datasets to load
    if ids_to_load is None:
        ids_to_load = list(range(n_datasets))

    datasets = []
    for i_ds in ids_to_load:
        # Load data
        data_path = input_path / f"recording_{i_ds}.npz"
        data = np.load(data_path)["data"]

        # Load info
        info_path = input_path / f"recording_{i_ds}_info.json"
        with open(info_path, "r") as f:
            info_dict = json.load(f)
        info = _dict_to_mne_info(info_dict)

        # Load description
        description = descriptions_df[descriptions_df["recording_id"] == i_ds].iloc[0]
        description = description.drop("recording_id")
        target_name = description.get("target_name", None)
        if "target_name" in description.index:
            description = description.drop("target_name")
        description = pd.Series(description)

        if is_windowed:
            # Load window metadata
            if windows_metadata_df is not None:
                metadata = windows_metadata_df[
                    windows_metadata_df["recording_id"] == i_ds
                ].drop(columns=["recording_id"])
            else:
                # Create basic metadata
                metadata = pd.DataFrame({
                    "i_window_in_trial": range(len(data)),
                    "i_start_in_trial": range(len(data)),
                    "i_stop_in_trial": range(1, len(data) + 1),
                    "target": 0,
                })

            # Create Epochs
            import mne
            events = np.column_stack([
                metadata["i_start_in_trial"].values,
                np.zeros(len(metadata), dtype=int),
                metadata["target"].values,
            ])

            epochs = mne.EpochsArray(data, info, events=events, metadata=metadata)
            ds = WindowsDataset(epochs, description)

            if target_name:
                ds.target_name = target_name
        else:
            # Create Raw
            import mne
            raw = mne.io.RawArray(data, info)
            ds = BaseDataset(raw, description, target_name=target_name)

        datasets.append(ds)

    # Create concat dataset
    concat_ds = BaseConcatDataset(datasets)

    # Restore preprocessing kwargs
    for kwarg_name in [
        "raw_preproc_kwargs",
        "window_kwargs",
        "window_preproc_kwargs",
    ]:
        if kwarg_name in global_metadata:
            kwargs = json.loads(global_metadata[kwarg_name])
            setattr(concat_ds, kwarg_name, kwargs)

    return concat_ds


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


def get_format_info(dataset: BaseConcatDataset) -> Dict:
    """Get recommended format based on dataset characteristics.

    Parameters
    ----------
    dataset : BaseConcatDataset
        The dataset to analyze.

    Returns
    -------
    dict
        Dictionary with format recommendations and dataset stats.
    """
    # Calculate dataset size
    is_windowed = hasattr(dataset.datasets[0], "windows")
    total_samples = 0
    total_size_mb = 0

    for ds in dataset.datasets:
        if is_windowed:
            data = ds.windows.get_data()
        else:
            data = ds.raw.get_data()

        total_samples += data.shape[0] if is_windowed else 1
        total_size_mb += data.nbytes / (1024 * 1024)

    n_recordings = len(dataset.datasets)

    # Recommendations
    if total_size_mb < 100:
        recommended = "npz_parquet"
        reason = "Small dataset - simple format sufficient"
    elif total_size_mb < 1000:
        recommended = "hdf5"
        reason = "Medium dataset - HDF5 offers best balance"
    else:
        recommended = "zarr"
        reason = "Large dataset - Zarr optimal for cloud storage and streaming"

    return {
        "n_recordings": n_recordings,
        "total_samples": total_samples,
        "total_size_mb": round(total_size_mb, 2),
        "is_windowed": is_windowed,
        "recommended_format": recommended,
        "reason": reason,
    }
