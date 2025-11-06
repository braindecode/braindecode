"""
Hugging Face Hub integration for EEG datasets.

This module provides push_to_hub() and from_pretrained() functionality
for braindecode datasets, similar to the model Hub integration.
"""

# Authors: Kuntal Kokate
#
# License: BSD (3-clause)

import json
import logging
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np

import braindecode

# Import registry for dynamic class lookup (avoids circular imports)
from .registry import get_dataset_class, get_dataset_type, is_registered_dataset

try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False

try:
    from huggingface_hub import (
        HfApi,
        create_repo,
        hf_hub_download,
        snapshot_download,
        upload_folder,
    )
    from huggingface_hub.utils import HfHubHTTPError

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

log = logging.getLogger(__name__)


# =============================================================================
# Core Zarr I/O Utilities (inlined to avoid circular imports)
# =============================================================================


def _mne_info_to_dict(info):
    """Convert MNE Info object to dictionary for JSON serialization."""
    return {
        "ch_names": info["ch_names"],
        "sfreq": float(info["sfreq"]),
        "ch_types": [str(ch_type) for ch_type in info.get_channel_types()],
        "lowpass": float(info["lowpass"]) if info["lowpass"] is not None else None,
        "highpass": float(info["highpass"]) if info["highpass"] is not None else None,
    }


def _dict_to_mne_info(info_dict):
    """Convert dictionary back to MNE Info object."""
    import mne

    info = mne.create_info(
        ch_names=info_dict["ch_names"],
        sfreq=info_dict["sfreq"],
        ch_types=info_dict["ch_types"],
    )

    # Use _unlock() context manager to set lowpass/highpass
    with info._unlock():
        if info_dict.get("lowpass") is not None:
            info["lowpass"] = info_dict["lowpass"]
        if info_dict.get("highpass") is not None:
            info["highpass"] = info_dict["highpass"]

    return info


def _save_windows_to_zarr(grp, data, metadata, description, info, compressor, target_name):
    """Save windowed data to Zarr group (low-level function)."""
    import pandas as pd

    # Save data with chunking for random access
    grp.create_dataset(
        "data",
        data=data.astype(np.float32),
        chunks=(1, data.shape[1], data.shape[2]),
        compressor=compressor,
    )

    # Save metadata
    metadata_json = metadata.to_json(orient="split", date_format="iso")
    grp.attrs["metadata"] = metadata_json

    # Save description
    description_json = description.to_json(date_format="iso")
    grp.attrs["description"] = description_json

    # Save MNE info
    grp.attrs["info"] = json.dumps(info)

    # Save target name if provided
    if target_name is not None:
        grp.attrs["target_name"] = target_name


def _save_eegwindows_to_zarr(grp, data, metadata, description, info, targets_from, last_target_only, compressor):
    """Save EEG windowed data to Zarr group (low-level function)."""
    import pandas as pd

    # Save data with chunking for random access
    grp.create_dataset(
        "data",
        data=data.astype(np.float32),
        chunks=(1, data.shape[1], data.shape[2]),
        compressor=compressor,
    )

    # Save metadata
    metadata_json = metadata.to_json(orient="split", date_format="iso")
    grp.attrs["metadata"] = metadata_json

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
    import pandas as pd

    # Load metadata
    metadata = pd.read_json(grp.attrs["metadata"], orient="split")

    # Load description
    description = pd.read_json(grp.attrs["description"], typ="series")

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

    # Load target name
    target_name = grp.attrs.get("target_name", None)

    return data, metadata, description, info_dict, target_name


def _load_eegwindows_from_zarr(grp, preload):
    """Load EEG windowed data from Zarr group (low-level function)."""
    import pandas as pd

    # Load metadata
    metadata = pd.read_json(grp.attrs["metadata"], orient="split")

    # Load description
    description = pd.read_json(grp.attrs["description"], typ="series")

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


def _create_compressor(compression, compression_level):
    """Create a Zarr compressor object."""
    if not ZARR_AVAILABLE:
        raise ImportError("Zarr is not installed. Install with: pip install zarr")

    if compression == "blosc":
        return zarr.Blosc(cname="zstd", clevel=compression_level)
    elif compression == "zstd":
        return zarr.Blosc(cname="zstd", clevel=compression_level)
    elif compression == "gzip":
        return zarr.Blosc(cname="gzip", clevel=compression_level)
    else:
        return None


class HubDatasetMixin:
    """
    Mixin class for Hugging Face Hub integration with EEG datasets.

    This class adds `push_to_hub()` and `from_pretrained()` methods to
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
    >>> dataset = BaseConcatDataset.from_pretrained("username/nmt-dataset")
    """

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
        if not HF_HUB_AVAILABLE:
            raise ImportError(
                "huggingface-hub is not installed. Install with: "
                "pip install braindecode[hub]"
            )

        # Note: No need to check for empty datasets - PyTorch's ConcatDataset
        # already prevents empty datasets in __init__

        if not ZARR_AVAILABLE:
            raise ImportError("Zarr is not installed. Install with: pip install zarr")

        # Create API instance
        api = HfApi(token=token)

        # Create repository if it doesn't exist
        try:
            create_repo(
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
                url = upload_folder(
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

        if dataset_type == "WindowsDataset":
            n_channels = len(first_ds.windows.ch_names)
            sfreq = first_ds.windows.info["sfreq"]
            n_windows = format_info["total_samples"]
            data_type = "Windowed (Epochs)"
        elif dataset_type == "EEGWindowsDataset":
            n_channels = len(first_ds.raw.ch_names)
            sfreq = first_ds.raw.info["sfreq"]
            n_windows = format_info["total_samples"]
            data_type = "Windowed (EEG)"
        else:
            raise TypeError(f"Unsupported dataset type: {dataset_type}")

        # Create README content
        readme_content = f"""---
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

- **Number of recordings**: {n_recordings}
- **Number of channels**: {n_channels}
- **Sampling frequency**: {sfreq} Hz
- **Data type**: {data_type}
- **Number of windows**: {n_windows}
- **Total size**: {format_info['total_size_mb']:.2f} MB
- **Storage format**: zarr

## Usage

To load this dataset:

```python
from braindecode.datasets import BaseConcatDataset

# Load dataset from Hugging Face Hub
dataset = BaseConcatDataset.from_pretrained("username/dataset-name")

# Access data
X, y, metainfo = dataset[0]
# X: EEG data (n_channels, n_times)
# y: label/target
# metainfo: window indices
```

## Using with PyTorch DataLoader

```python
from torch.utils.data import DataLoader

# Create DataLoader for training
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Training loop
for X, y, _ in train_loader:
    # X shape: [batch_size, n_channels, n_times]
    # y shape: [batch_size]
    # Process your batch...
```

## Dataset Format

This dataset is stored in **Zarr** format, optimized for:
- Fast random access during training (critical for PyTorch DataLoader)
- Efficient compression with blosc
- Cloud-native storage compatibility

For more information about braindecode, visit: https://braindecode.org
"""

        # Save README
        readme_path = path / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)

    @classmethod
    def from_pretrained(
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
        >>> dataset = BaseConcatDataset.from_pretrained("username/nmt-dataset")
        >>> print(f"Loaded {len(dataset)} windows")
        >>>
        >>> # Use with PyTorch
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        """
        if not HF_HUB_AVAILABLE:
            raise ImportError(
                "huggingface-hub is not installed. Install with: "
                "pip install braindecode[hub]"
            )

        if not ZARR_AVAILABLE:
            raise ImportError("Zarr is not installed. Install with: pip install zarr")

        log.info(f"Loading dataset from Hugging Face Hub ({repo_id})...")

        try:
            # Download the entire dataset directory
            dataset_dir = snapshot_download(
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

        except HfHubHTTPError as e:
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
        if output_path.exists():
            raise FileExistsError(
                f"{output_path} already exists. Set overwrite=True to replace it."
            )

        # Create zarr store
        store = zarr.DirectoryStore(output_path)
        root = zarr.group(store=store, overwrite=False)

        # Determine dataset type using registry
        first_ds = self.datasets[0]
        dataset_type = get_dataset_type(first_ds)

        if dataset_type == "BaseDataset":
            raise NotImplementedError(
                "Saving continuous BaseDataset (non-windowed raw data) to Hub is not yet "
                "supported. Please create windows from your dataset using "
                "braindecode.preprocessing.create_windows_from_events() or "
                "create_fixed_length_windows() before uploading to Hub."
            )
        elif dataset_type not in ["WindowsDataset", "EEGWindowsDataset"]:
            raise TypeError(f"Unsupported dataset type: {dataset_type}")

        # Store global metadata
        root.attrs["n_datasets"] = len(self.datasets)
        root.attrs["dataset_type"] = dataset_type
        root.attrs["braindecode_version"] = "1.0"

        # Save preprocessing kwargs
        for kwarg_name in ["raw_preproc_kwargs", "window_kwargs", "window_preproc_kwargs"]:
            if hasattr(self, kwarg_name):
                kwargs = getattr(self, kwarg_name)
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
                info_dict = _mne_info_to_dict(ds.windows.info)
                target_name = ds.target_name if hasattr(ds, "target_name") else None

                # Save using inlined function
                _save_windows_to_zarr(
                    grp, data, metadata, description, info_dict, compressor, target_name
                )

            elif dataset_type == "EEGWindowsDataset":
                # Extract windows from EEGWindowsDataset
                windows_list = []
                for i in range(len(ds)):
                    X, y, crop_inds = ds[i]
                    windows_list.append(X)

                data = np.stack(windows_list, axis=0)
                metadata = ds.metadata
                description = ds.description
                info_dict = _mne_info_to_dict(ds.raw.info)
                targets_from = ds.targets_from
                last_target_only = ds.last_target_only

                # Save using inlined function
                _save_eegwindows_to_zarr(
                    grp, data, metadata, description, info_dict,
                    targets_from, last_target_only, compressor
                )

    def _get_format_info_inline(self):
        """Get format info (inline implementation).

        This is an inline version of hub_formats.get_format_info() that avoids
        circular import.
        """
        if len(self.datasets) == 0:
            raise ValueError("Cannot get format info for empty dataset")

        # Determine dataset type from first dataset using registry
        first_ds = self.datasets[0]
        dataset_type = get_dataset_type(first_ds)

        if dataset_type == "WindowsDataset":
            first_ch_names = first_ds.windows.ch_names
            first_sfreq = first_ds.windows.info["sfreq"]
        elif dataset_type == "EEGWindowsDataset":
            first_ch_names = first_ds.raw.ch_names
            first_sfreq = first_ds.raw.info["sfreq"]
        else:
            raise TypeError(f"Unsupported dataset type: {dataset_type}")

        # Validate uniformity across all datasets
        for i, ds in enumerate(self.datasets):
            # Check if all datasets are the same type
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
                        f"but dataset {i} has {ds.windows.info['sfreq']} Hz"
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
                        f"but dataset {i} has {ds.raw.info['sfreq']} Hz"
                    )

        # Calculate dataset size
        total_samples = 0
        total_size_mb = 0

        for ds in self.datasets:
            if dataset_type == "WindowsDataset":
                data = ds.windows.get_data()
                total_samples += data.shape[0]
                total_size_mb += data.nbytes / (1024 * 1024)
            elif dataset_type == "EEGWindowsDataset":
                total_samples += len(ds.metadata)
                for i in range(len(ds)):
                    X, _, _ = ds[i]
                    total_size_mb += X.nbytes / (1024 * 1024)

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
        import mne

        if not input_path.exists():
            raise FileNotFoundError(f"{input_path} does not exist.")

        # Open zarr store
        store = zarr.DirectoryStore(input_path)
        root = zarr.group(store=store)

        n_datasets = root.attrs["n_datasets"]
        dataset_type = root.attrs.get("dataset_type", None)

        # For backwards compatibility
        if dataset_type is None:
            is_windowed = root.attrs.get("is_windowed", False)
            dataset_type = "WindowsDataset" if is_windowed else "BaseDataset"

        # Get dataset classes from registry
        WindowsDataset = get_dataset_class("WindowsDataset")
        EEGWindowsDataset = get_dataset_class("EEGWindowsDataset")
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
                info = _dict_to_mne_info(info_dict)
                events = np.column_stack([
                    metadata["i_start_in_trial"].values,
                    np.zeros(len(metadata), dtype=int),
                    metadata["target"].values,
                ])
                epochs = mne.EpochsArray(data, info, events=events, metadata=metadata)
                ds = WindowsDataset(epochs, description)
                if target_name is not None:
                    ds.target_name = target_name

            elif dataset_type == "EEGWindowsDataset":
                # Load using inlined function
                data, metadata, description, info_dict, targets_from, last_target_only = (
                    _load_eegwindows_from_zarr(grp, preload)
                )

                # Convert to MNE objects and create dataset
                info = _dict_to_mne_info(info_dict)
                n_windows, n_channels, n_times_per_window = data.shape
                continuous_data = data.reshape(n_channels, n_windows * n_times_per_window)
                raw = mne.io.RawArray(continuous_data, info)
                ds = EEGWindowsDataset(
                    raw=raw,
                    metadata=metadata,
                    description=description,
                    targets_from=targets_from,
                    last_target_only=last_target_only,
                )

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
