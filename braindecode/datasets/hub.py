"""
Hugging Face Hub integration for EEG datasets.

This module provides push_to_hub() and pull_from_hub() functionality
for braindecode datasets, similar to the model Hub integration.
"""

# Authors: Kuntal Kokate
#
# License: BSD (3-clause)

import io
import json
import logging
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import mne
import numpy as np
import pandas as pd
import scipy
from mne._fiff.meas_info import Info
from mne.utils import _soft_import

if TYPE_CHECKING:
    from .base import BaseDataset

import braindecode

# Import shared validation utilities
from . import hub_validation

# Import registry for dynamic class lookup (avoids circular imports)
from .registry import get_dataset_class, get_dataset_type

# Lazy import zarr and huggingface_hub
zarr = _soft_import("zarr", purpose="hugging face integration", strict=False)
huggingface_hub = _soft_import(
    "huggingface_hub", purpose="hugging face integration", strict=False
)

log = logging.getLogger(__name__)


class HubDatasetMixin:
    """
    Mixin class for Hugging Face Hub integration with EEG datasets.

    This class adds `push_to_hub()` and `pull_from_hub()` methods to
    BaseConcatDataset, enabling easy upload and download of datasets
    to/from the Hugging Face Hub.

    Examples
    --------
    >>> # Push dataset to Hub
    >>> dataset = NMT(path=path, preload=True)
    >>> dataset.push_to_hub(
    ...     repo_id="username/nmt-dataset",
    ...     commit_message="Add NMT dataset"
    ... )
    >>>
    >>> # Load dataset from Hub
    >>> dataset = BaseConcatDataset.pull_from_hub("username/nmt-dataset")
    """

    datasets: List["BaseDataset"]  # Attribute provided by inheriting class

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: Optional[str] = None,
        private: bool = False,
        token: Optional[str] = None,
        create_pr: bool = False,
        compression: str = "blosc",
        compression_level: int = 5,
    ) -> str:
        """
        Upload the dataset to the Hugging Face Hub in Zarr format.

        The dataset is converted to Zarr format with blosc compression, which provides
        optimal random access performance for PyTorch training (based on comprehensive
        benchmarking).

        Parameters
        ----------
        repo_id : str
            Repository ID on the Hugging Face Hub (e.g., "username/dataset-name").
        commit_message : str | None
            Commit message. If None, a default message is generated.
        private : bool, default=False
            Whether to create a private repository.
        token : str | None
            Hugging Face API token. If None, uses cached token.
        create_pr : bool, default=False
            Whether to create a Pull Request instead of directly committing.
        compression : str, default="blosc"
            Compression algorithm for Zarr. Options: "blosc", "zstd", "gzip", None.
        compression_level : int, default=5
            Compression level (0-9). Level 5 provides optimal balance.

        Returns
        -------
        str
            URL of the uploaded dataset on the Hub.

        Raises
        ------
        ImportError
            If huggingface-hub is not installed.
        ValueError
            If the dataset is empty or format is invalid.

        Examples
        --------
        >>> dataset = NMT(path=path, preload=True)
        >>> # Upload with default settings (zarr with blosc compression)
        >>> url = dataset.push_to_hub(
        ...     repo_id="myusername/nmt-dataset",
        ...     commit_message="Upload NMT EEG dataset"
        ... )
        >>>
        >>> # Or customize compression
        >>> url = dataset.push_to_hub(
        ...     repo_id="myusername/nmt-dataset",
        ...     compression="blosc",
        ...     compression_level=5
        ... )
        """
        if huggingface_hub is False or zarr is False:
            raise ImportError(
                "huggingface-hub or zarr is not installed. Install with: "
                "pip install braindecode[hub]"
            )

        # Create API instance
        _ = huggingface_hub.HfApi(token=token)

        # Create repository if it doesn't exist
        try:
            huggingface_hub.create_repo(
                repo_id=repo_id,
                token=token,
                private=private,
                repo_type="dataset",
                exist_ok=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create repository: {e}")

        # Create a temporary directory for upload
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Convert dataset to Zarr format
            log.info("Converting dataset to Zarr format...")
            dataset_path = tmp_path / "dataset.zarr"
            self._convert_to_zarr_inline(
                dataset_path,
                compression,
                compression_level,
            )

            # Save dataset metadata
            self._save_dataset_card(tmp_path)

            # Save format info
            format_info_path = tmp_path / "format_info.json"
            with open(format_info_path, "w") as f:
                format_info = self._get_format_info_inline()
                json.dump(
                    {
                        "format": "zarr",
                        "compression": compression,
                        "compression_level": compression_level,
                        "braindecode_version": braindecode.__version__,
                        **format_info,
                    },
                    f,
                    indent=2,
                )

            # Default commit message
            if commit_message is None:
                commit_message = (
                    f"Upload EEG dataset in Zarr format "
                    f"({len(self.datasets)} recordings)"
                )

            # Upload folder to Hub
            log.info(f"Uploading to Hugging Face Hub ({repo_id})...")
            try:
                url = huggingface_hub.upload_folder(
                    repo_id=repo_id,
                    folder_path=str(tmp_path),
                    repo_type="dataset",
                    commit_message=commit_message,
                    token=token,
                    create_pr=create_pr,
                )
                log.info(f"Dataset uploaded successfully to {repo_id}")
                log.info(f"URL: https://huggingface.co/datasets/{repo_id}")
                return url
            except Exception as e:
                raise RuntimeError(f"Failed to upload dataset: {e}")

    def _save_dataset_card(self, path: Path) -> None:
        """Generate and save a dataset card (README.md) with metadata.

        Parameters
        ----------
        path : Path
            Directory where README.md will be saved.
        """
        # Get info, which also validates uniformity across all datasets
        format_info = self._get_format_info_inline()

        n_recordings = len(self.datasets)
        first_ds = self.datasets[0]

        # Get dataset-specific info based on type using registry
        dataset_type = get_dataset_type(first_ds)

        n_windows = format_info["total_samples"]

        if dataset_type == "WindowsDataset":
            n_channels = len(first_ds.windows.ch_names)
            data_type = "Windowed (from Epochs object)"
            sfreq = first_ds.windows.info["sfreq"]
        elif dataset_type == "EEGWindowsDataset":
            n_channels = len(first_ds.raw.ch_names)
            sfreq = first_ds.raw.info["sfreq"]
            data_type = "Windowed (from Raw object)"
        elif dataset_type == "RawDataset":
            n_channels = len(first_ds.raw.ch_names)
            sfreq = first_ds.raw.info["sfreq"]
            data_type = "Continuous (Raw)"
        else:
            raise TypeError(f"Unsupported dataset type: {dataset_type}")

        # Create README content and save
        readme_content = _generate_readme_content(
            format_info=format_info,
            n_recordings=n_recordings,
            n_channels=n_channels,
            sfreq=sfreq,
            data_type=data_type,
            n_windows=n_windows,
        )

        # Save README
        readme_path = path / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)

    @classmethod
    def pull_from_hub(
        cls,
        repo_id: str,
        preload: bool = True,
        token: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        force_download: bool = False,
        **kwargs,
    ):
        """
        Load a dataset from the Hugging Face Hub.

        Parameters
        ----------
        repo_id : str
            Repository ID on the Hugging Face Hub (e.g., "username/dataset-name").
        preload : bool, default=True
            Whether to preload the data into memory. If False, uses lazy loading
            (when supported by the format).
        token : str | None
            Hugging Face API token. If None, uses cached token.
        cache_dir : str | Path | None
            Directory to cache the downloaded dataset. If None, uses default
            cache directory (~/.cache/huggingface/datasets).
        force_download : bool, default=False
            Whether to force re-download even if cached.
        **kwargs
            Additional arguments (currently unused).

        Returns
        -------
        BaseConcatDataset
            The loaded dataset.

        Raises
        ------
        ImportError
            If huggingface-hub is not installed.
        FileNotFoundError
            If the repository or dataset files are not found.

        Examples
        --------
        >>> from braindecode.datasets import BaseConcatDataset
        >>> dataset = BaseConcatDataset.pull_from_hub("username/nmt-dataset")
        >>> print(f"Loaded {len(dataset)} windows")
        >>>
        >>> # Use with PyTorch
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        """
        if zarr is False or huggingface_hub is False:
            raise ImportError(
                "huggingface hub functionality is not installed. Install with: "
                "pip install braindecode[hub]"
            )

        log.info(f"Loading dataset from Hugging Face Hub ({repo_id})...")

        try:
            # Download the entire dataset directory
            dataset_dir = huggingface_hub.snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                cache_dir=cache_dir,
                force_download=force_download,
            )

            # Load format info
            format_info_path = Path(dataset_dir) / "format_info.json"
            if format_info_path.exists():
                with open(format_info_path, "r") as f:
                    format_info = json.load(f)

                # Verify it's zarr format
                if format_info.get("format") != "zarr":
                    raise ValueError(
                        f"Dataset format is '{format_info.get('format')}', but only "
                        "'zarr' format is supported. Please re-upload the dataset."
                    )
            else:
                format_info = {}

            # Load zarr dataset
            zarr_path = Path(dataset_dir) / "dataset.zarr"
            if not zarr_path.exists():
                raise FileNotFoundError(
                    f"Zarr dataset not found at {zarr_path}. "
                    "The dataset may be in an unsupported format."
                )

            dataset = cls._load_from_zarr_inline(zarr_path, preload)

            log.info(f"Dataset loaded successfully from {repo_id}")
            log.info(f"Recordings: {len(dataset.datasets)}")
            log.info(
                f"Total windows/samples: {format_info.get('total_samples', 'N/A')}"
            )

            return dataset

        except huggingface_hub.utils.HfHubHTTPError as e:
            if e.response.status_code == 404:
                raise FileNotFoundError(
                    f"Dataset '{repo_id}' not found on Hugging Face Hub. "
                    "Please check the repository ID and ensure it exists."
                )
            else:
                raise RuntimeError(f"Failed to download dataset: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from Hub: {e}")

    def _convert_to_zarr_inline(
        self,
        output_path: Path,
        compression: str,
        compression_level: int,
    ) -> None:
        """Convert dataset to Zarr format (inline implementation)."""

        if zarr is False or huggingface_hub is False:
            raise ImportError(
                "huggingface hub functionality is not installed. Install with: "
                "pip install braindecode[hub]"
            )

        if output_path.exists():
            raise FileExistsError(
                f"{output_path} already exists. Set overwrite=True to replace it."
            )

        # Create zarr store (zarr v3 API)
        root = zarr.open(str(output_path), mode="w")

        # Validate uniformity across all datasets using shared validation
        dataset_type, _, _ = hub_validation.validate_dataset_uniformity(self.datasets)

        # Keep reference to first dataset for preprocessing kwargs
        first_ds = self.datasets[0]

        # Store global metadata
        root.attrs["n_datasets"] = len(self.datasets)
        root.attrs["dataset_type"] = dataset_type
        root.attrs["braindecode_version"] = braindecode.__version__

        # Track dependency versions for reproducibility
        root.attrs["mne_version"] = mne.__version__
        root.attrs["numpy_version"] = np.__version__
        root.attrs["pandas_version"] = pd.__version__
        root.attrs["zarr_version"] = zarr.__version__
        root.attrs["scipy_version"] = scipy.__version__

        # Save preprocessing kwargs (check first dataset, assuming uniform preprocessing)
        # These are typically set by windowing functions on individual datasets
        for kwarg_name in [
            "raw_preproc_kwargs",
            "window_kwargs",
            "window_preproc_kwargs",
        ]:
            # Check first dataset for these attributes
            if hasattr(first_ds, kwarg_name):
                kwargs = getattr(first_ds, kwarg_name)
                if kwargs:
                    root.attrs[kwarg_name] = json.dumps(kwargs)

        # Create compressor
        compressor = _create_compressor(compression, compression_level)

        # Save each recording
        for i_ds, ds in enumerate(self.datasets):
            grp = root.create_group(f"recording_{i_ds}")

            if dataset_type == "WindowsDataset":
                # Extract data from WindowsDataset
                data = ds.windows.get_data()
                metadata = ds.windows.metadata
                description = ds.description
                info_dict = ds.windows.info.to_json_dict()
                target_name = ds.target_name if hasattr(ds, "target_name") else None

                # Save using inlined function
                _save_windows_to_zarr(
                    grp, data, metadata, description, info_dict, compressor, target_name
                )

            elif dataset_type == "EEGWindowsDataset":
                # Get continuous raw data and metadata from EEGWindowsDataset
                raw = ds.raw
                metadata = ds.metadata
                description = ds.description
                info_dict = ds.raw.info.to_json_dict()
                targets_from = ds.targets_from
                last_target_only = ds.last_target_only

                # Save using inlined function (saves continuous raw directly)
                _save_eegwindows_to_zarr(
                    grp,
                    raw,
                    metadata,
                    description,
                    info_dict,
                    targets_from,
                    last_target_only,
                    compressor,
                )

            elif dataset_type == "RawDataset":
                # Get continuous raw data from RawDataset
                raw = ds.raw
                description = ds.description
                info_dict = ds.raw.info.to_json_dict()
                target_name = ds.target_name if hasattr(ds, "target_name") else None

                # Save using inlined function
                _save_raw_to_zarr(
                    grp, raw, description, info_dict, target_name, compressor
                )

    def _get_format_info_inline(self):
        """Get format info (inline implementation).

        This is an inline version of hub_formats.get_format_info() that avoids
        circular import.
        """
        if len(self.datasets) == 0:
            raise ValueError("Cannot get format info for empty dataset")

        # Validate uniformity across all datasets using shared validation
        dataset_type, _, _ = hub_validation.validate_dataset_uniformity(self.datasets)

        # Calculate dataset size
        # BaseConcatDataset's __len__ already sums len(ds) for all datasets
        total_samples = len(self)
        total_size_mb = 0

        for ds in self.datasets:
            if dataset_type == "WindowsDataset":
                # Use MNE's internal _size property to avoid loading data
                total_size_mb += ds.windows._size / (1024 * 1024)
            elif dataset_type == "EEGWindowsDataset":
                # Use raw object's size (not extracted windows)
                total_size_mb += ds.raw._size / (1024 * 1024)
            elif dataset_type == "RawDataset":
                total_size_mb += ds.raw._size / (1024 * 1024)

        n_recordings = len(self.datasets)

        return {
            "n_recordings": n_recordings,
            "total_samples": total_samples,
            "total_size_mb": round(total_size_mb, 2),
        }

    @staticmethod
    def _load_from_zarr_inline(input_path: Path, preload: bool):
        """Load dataset from Zarr format (inline implementation).

        This is an inline version of hub_formats.load_from_zarr() that avoids
        circular import by using hub_formats_core directly.
        """
        if not input_path.exists():
            raise FileNotFoundError(f"{input_path} does not exist.")

        # Open zarr store (zarr v3 API)
        root = zarr.open(str(input_path), mode="r")

        n_datasets = root.attrs["n_datasets"]
        dataset_type = root.attrs["dataset_type"]

        # Get dataset classes from registry
        WindowsDataset = get_dataset_class("WindowsDataset")
        EEGWindowsDataset = get_dataset_class("EEGWindowsDataset")
        RawDataset = get_dataset_class("RawDataset")
        BaseConcatDataset = get_dataset_class("BaseConcatDataset")

        datasets = []
        for i_ds in range(n_datasets):
            grp = root[f"recording_{i_ds}"]

            if dataset_type == "WindowsDataset":
                # Load using inlined function
                data, metadata, description, info_dict, target_name = (
                    _load_windows_from_zarr(grp, preload)
                )

                # Convert to MNE objects and create dataset
                info = Info.from_json_dict(info_dict)
                events = np.column_stack(
                    [
                        metadata["i_start_in_trial"].values,
                        np.zeros(len(metadata), dtype=int),
                        metadata["target"].values,
                    ]
                )
                epochs = mne.EpochsArray(data, info, events=events, metadata=metadata)
                ds = WindowsDataset(epochs, description)
                if target_name is not None:
                    ds.target_name = target_name

            elif dataset_type == "EEGWindowsDataset":
                # Load using inlined function
                (
                    data,
                    metadata,
                    description,
                    info_dict,
                    targets_from,
                    last_target_only,
                ) = _load_eegwindows_from_zarr(grp, preload)

                # Convert to MNE objects and create dataset
                # Data is already in continuous format [n_channels, n_timepoints]
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
                # Load using inlined function
                data, description, info_dict, target_name = _load_raw_from_zarr(
                    grp, preload
                )

                # Convert to MNE objects and create dataset
                # Data is in continuous format [n_channels, n_timepoints]
                info = Info.from_json_dict(info_dict)
                raw = mne.io.RawArray(data, info)
                ds = RawDataset(raw, description)
                if target_name is not None:
                    ds.target_name = target_name

            else:
                raise ValueError(f"Unsupported dataset_type: {dataset_type}")

            datasets.append(ds)

        # Create concat dataset
        concat_ds = BaseConcatDataset(datasets)

        # Restore preprocessing kwargs (set on individual datasets, not concat)
        for kwarg_name in [
            "raw_preproc_kwargs",
            "window_kwargs",
            "window_preproc_kwargs",
        ]:
            if kwarg_name in root.attrs:
                kwargs = json.loads(root.attrs[kwarg_name])
                # Set on each individual dataset (where they were originally stored)
                for ds in datasets:
                    setattr(ds, kwarg_name, kwargs)

        return concat_ds


# =============================================================================
# Core Zarr I/O Utilities
# =============================================================================


def _save_windows_to_zarr(
    grp, data, metadata, description, info, compressor, target_name
):
    """Save windowed data to Zarr group (low-level function)."""
    # Save data with chunking for random access
    # In Zarr v3, use create_array with compressors parameter
    data_array = data.astype(np.float32)

    # Zarr v3 expects compressors as a list
    compressors_list = [compressor] if compressor is not None else None

    grp.create_array(
        "data",
        data=data_array,
        chunks=(1, data_array.shape[1], data_array.shape[2]),
        compressors=compressors_list,
    )

    # Save metadata
    metadata_json = metadata.to_json(orient="split", date_format="iso")
    grp.attrs["metadata"] = metadata_json
    # Save dtypes to preserve them across platforms (int32 vs int64, etc.)
    metadata_dtypes = metadata.dtypes.apply(str).to_json()
    grp.attrs["metadata_dtypes"] = metadata_dtypes

    # Save description
    description_json = description.to_json(date_format="iso")
    grp.attrs["description"] = description_json

    # Save MNE info
    grp.attrs["info"] = json.dumps(info)

    # Save target name if provided
    if target_name is not None:
        grp.attrs["target_name"] = target_name


def _save_eegwindows_to_zarr(
    grp, raw, metadata, description, info, targets_from, last_target_only, compressor
):
    """Save EEG continuous raw data to Zarr group (low-level function)."""
    # Extract continuous data from Raw [n_channels, n_timepoints]
    continuous_data = raw.get_data()

    # Save continuous data with chunking optimized for window extraction
    # Chunk size: all channels, 10000 timepoints for efficient random access
    # In Zarr v3, use create_array with compressors parameter
    continuous_float = continuous_data.astype(np.float32)

    # Zarr v3 expects compressors as a list
    compressors_list = [compressor] if compressor is not None else None

    grp.create_array(
        "data",
        data=continuous_float,
        chunks=(continuous_float.shape[0], min(10000, continuous_float.shape[1])),
        compressors=compressors_list,
    )

    # Save metadata
    metadata_json = metadata.to_json(orient="split", date_format="iso")
    grp.attrs["metadata"] = metadata_json
    # Save dtypes to preserve them across platforms (int32 vs int64, etc.)
    metadata_dtypes = metadata.dtypes.apply(str).to_json()
    grp.attrs["metadata_dtypes"] = metadata_dtypes

    # Save description
    description_json = description.to_json(date_format="iso")
    grp.attrs["description"] = description_json

    # Save MNE info
    grp.attrs["info"] = json.dumps(info)

    # Save EEGWindowsDataset-specific attributes
    grp.attrs["targets_from"] = targets_from
    grp.attrs["last_target_only"] = last_target_only


def _load_windows_from_zarr(grp, preload):
    """Load windowed data from Zarr group (low-level function)."""
    # Load metadata
    metadata = pd.read_json(io.StringIO(grp.attrs["metadata"]), orient="split")
    # Restore dtypes to preserve them across platforms (int32 vs int64, etc.)
    dtypes_dict = pd.read_json(io.StringIO(grp.attrs["metadata_dtypes"]), typ="series")
    for col, dtype_str in dtypes_dict.items():
        metadata[col] = metadata[col].astype(dtype_str)

    # Load description
    description = pd.read_json(io.StringIO(grp.attrs["description"]), typ="series")

    # Load info
    info_dict = json.loads(grp.attrs["info"])

    # Load data
    if preload:
        data = grp["data"][:]
    else:
        data = grp["data"][:]
        # TODO: Implement lazy loading properly
        warnings.warn(
            "Lazy loading from Zarr not fully implemented yet. "
            "Loading all data into memory.",
            UserWarning,
        )

    # Load target name
    target_name = grp.attrs.get("target_name", None)

    return data, metadata, description, info_dict, target_name


def _load_eegwindows_from_zarr(grp, preload):
    """Load EEG continuous raw data from Zarr group (low-level function)."""
    # Load metadata
    metadata = pd.read_json(io.StringIO(grp.attrs["metadata"]), orient="split")
    # Restore dtypes to preserve them across platforms (int32 vs int64, etc.)
    dtypes_dict = pd.read_json(io.StringIO(grp.attrs["metadata_dtypes"]), typ="series")
    for col, dtype_str in dtypes_dict.items():
        metadata[col] = metadata[col].astype(dtype_str)

    # Load description
    description = pd.read_json(io.StringIO(grp.attrs["description"]), typ="series")

    # Load info
    info_dict = json.loads(grp.attrs["info"])

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

    # Load EEGWindowsDataset-specific attributes
    targets_from = grp.attrs.get("targets_from", "metadata")
    last_target_only = grp.attrs.get("last_target_only", True)

    return data, metadata, description, info_dict, targets_from, last_target_only


def _save_raw_to_zarr(grp, raw, description, info, target_name, compressor):
    """Save RawDataset continuous raw data to Zarr group (low-level function)."""
    # Extract continuous data from Raw [n_channels, n_timepoints]
    continuous_data = raw.get_data()

    # Save continuous data with chunking optimized for efficient access
    # Chunk size: all channels, 10000 timepoints for efficient random access
    # In Zarr v3, use create_array with compressors parameter
    continuous_float = continuous_data.astype(np.float32)

    # Zarr v3 expects compressors as a list
    compressors_list = [compressor] if compressor is not None else None

    grp.create_array(
        "data",
        data=continuous_float,
        chunks=(continuous_float.shape[0], min(10000, continuous_float.shape[1])),
        compressors=compressors_list,
    )

    # Save description
    description_json = description.to_json(date_format="iso")
    grp.attrs["description"] = description_json

    # Save MNE info
    grp.attrs["info"] = json.dumps(info)

    # Save target name if provided
    if target_name is not None:
        grp.attrs["target_name"] = target_name


def _load_raw_from_zarr(grp, preload):
    """Load RawDataset continuous raw data from Zarr group (low-level function)."""
    # Load description
    description = pd.read_json(io.StringIO(grp.attrs["description"]), typ="series")

    # Load info
    info_dict = json.loads(grp.attrs["info"])

    # Load data
    if preload:
        data = grp["data"][:]
    else:
        data = grp["data"][:]
        # TODO: Implement lazy loading properly
        warnings.warn(
            "Lazy loading from Zarr not fully implemented yet. "
            "Loading all data into memory.",
            UserWarning,
        )

    # Load target name
    target_name = grp.attrs.get("target_name", None)

    return data, description, info_dict, target_name


def _create_compressor(compression, compression_level):
    """Create a Zarr v3 compressor codec.

    Returns compressor dict compatible with Zarr v3 create_array API.

    Parameters
    ----------
    compression : str or None
        Compression algorithm: "blosc", "zstd", "gzip", or None.
    compression_level : int
        Compression level (0-9).

    Returns
    -------
    dict or None
        Compressor configuration dict or None if no compression.
    """
    if zarr is False:
        raise ImportError(
            "Zarr is not installed. Install with: pip install braindecode[hub]"
        )

    if compression is None or compression not in ("blosc", "zstd", "gzip"):
        return None

    # Map blosc to zstd (Zarr v3 uses zstd as the primary implementation)
    name = "zstd" if compression == "blosc" else compression

    return {"name": name, "configuration": {"level": compression_level}}


# TODO: improve content
def _generate_readme_content(
    format_info,
    n_recordings: int,
    n_channels: int,
    sfreq,
    data_type: str,
    n_windows: int,
    format: str = "zarr",
):
    """Generate README.md content for a dataset uploaded to the Hub."""
    # Use safe access for total size and format sfreq nicely
    total_size_mb = (
        format_info.get("total_size_mb", 0.0) if isinstance(format_info, dict) else 0.0
    )
    sfreq_str = f"{sfreq:g}" if sfreq is not None else "N/A"

    return f"""---
tags:
- braindecode
- eeg
- neuroscience
- brain-computer-interface
license: unknown
---

# EEG Dataset

This dataset was created using [braindecode](https://braindecode.org), a library for deep learning with EEG/MEG/ECoG signals.

## Dataset Information

| Property | Value |
|---|---:|
| Number of recordings | {n_recordings} |
| Dataset type | {data_type} |
| Number of channels | {n_channels} |
| Sampling frequency | {sfreq_str} Hz |
| Number of windows / samples | {n_windows} |
| Total size | {total_size_mb:.2f} MB |
| Storage format | {format} |

## Usage

To load this dataset::

    .. code-block:: python

        from braindecode.datasets import BaseConcatDataset

        # Load dataset from Hugging Face Hub
        dataset = BaseConcatDataset.pull_from_hub("username/dataset-name")

        # Access data
        X, y, metainfo = dataset[0]
        # X: EEG data (n_channels, n_times)
        # y: label/target
        # metainfo: window indices

## Using with PyTorch DataLoader

::

    from torch.utils.data import DataLoader

    # Create DataLoader for training
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    # Training loop
    for X, y, metainfo in train_loader:
        # X shape: [batch_size, n_channels, n_times]
        # y shape: [batch_size]
        # metainfo shape: [batch_size, 2] (start and end indices)
        # Process your batch...

## Dataset Format

This dataset is stored in **Zarr** format, optimized for:
- Fast random access during training (critical for PyTorch DataLoader)
- Efficient compression with blosc
- Cloud-native storage compatibility

For more information about braindecode, visit: https://braindecode.org
"""
