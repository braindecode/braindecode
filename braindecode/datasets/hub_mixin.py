"""
Hugging Face Hub integration mixin for EEG datasets.

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

import braindecode

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

        # Create API instance
        api = HfApi(token=token)

        # Prevent circular import
        from ..datautil.hub_formats import convert_to_zarr, get_format_info

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
            convert_to_zarr(
                self,
                dataset_path,
                compression=compression,
                compression_level=compression_level,
            )

            # Save dataset metadata
            self._save_dataset_card(tmp_path)

            # Save format info
            format_info_path = tmp_path / "format_info.json"
            with open(format_info_path, "w") as f:
                json.dump(
                    {
                        "format": "zarr",
                        "compression": compression,
                        "compression_level": compression_level,
                        "braindecode_version": braindecode.__version__,
                        **get_format_info(self),
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
        from ..datautil.hub_formats import WindowsDataset, EEGWindowsDataset, get_format_info

        # Get info, which also validates uniformity across all datasets
        format_info = get_format_info(self)

        n_recordings = len(self.datasets)
        first_ds = self.datasets[0]

        # Get dataset-specific info based on type
        if isinstance(first_ds, WindowsDataset):
            n_channels = len(first_ds.windows.ch_names)
            sfreq = first_ds.windows.info["sfreq"]
            n_windows = format_info["total_samples"]
            data_type = "Windowed (Epochs)"
        elif isinstance(first_ds, EEGWindowsDataset):
            n_channels = len(first_ds.raw.ch_names)
            sfreq = first_ds.raw.info["sfreq"]
            n_windows = format_info["total_samples"]
            data_type = "Windowed (EEG)"
        else:
            raise TypeError(f"Unsupported dataset type: {type(first_ds)}")

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

        log.info(f"Loading dataset from Hugging Face Hub ({repo_id})...")

        # Prevent circular import
        from ..datautil.hub_formats import load_from_zarr

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

            dataset = load_from_zarr(zarr_path, preload=preload)

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
