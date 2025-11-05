"""
Hugging Face Hub integration mixin for EEG datasets.

This module provides push_to_hub() and from_pretrained() functionality
for braindecode datasets, similar to the model Hub integration.
"""

# Authors: Kuntal Kokate
#
# License: BSD (3-clause)

import json
import warnings
from pathlib import Path
from typing import Optional, Union

try:
    from huggingface_hub import (
        HfApi,
        create_repo,
        hf_hub_download,
        upload_folder,
    )
    from huggingface_hub.utils import HfHubHTTPError
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


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
                "pip install huggingface-hub"
            )

        if len(self.datasets) == 0:
            raise ValueError("Cannot upload an empty dataset")

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

        # Import converter (lazy import to avoid circular dependency)
        from ..datautil.hub_formats import convert_to_zarr, get_format_info

        # Create a temporary directory for upload
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Convert dataset to Zarr format
            print("Converting dataset to Zarr format...")
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
            print(f"Uploading to Hugging Face Hub ({repo_id})...")
            try:
                url = upload_folder(
                    repo_id=repo_id,
                    folder_path=str(tmp_path),
                    repo_type="dataset",
                    commit_message=commit_message,
                    token=token,
                    create_pr=create_pr,
                )
                print(f"✅ Dataset uploaded successfully!")
                print(f"   URL: https://huggingface.co/datasets/{repo_id}")
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
        # Lazy import to avoid circular dependency
        from ..datautil.hub_formats import get_format_info

        is_windowed = hasattr(self.datasets[0], "windows")

        # Gather dataset information
        n_recordings = len(self.datasets)

        # Get info about first recording
        first_ds = self.datasets[0]
        if is_windowed:
            n_channels = len(first_ds.windows.ch_names)
            sfreq = first_ds.windows.info["sfreq"]
            n_windows = len(first_ds.windows)
        else:
            n_channels = len(first_ds.raw.ch_names)
            sfreq = first_ds.raw.info["sfreq"]
            n_windows = "N/A (continuous data)"

        # Get format info
        format_info = get_format_info(self)

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
- **Data type**: {'Windowed (Epochs)' if is_windowed else 'Continuous (Raw)'}
- **Number of windows**: {n_windows}
- **Total size**: {format_info['total_size_mb']:.2f} MB
- **Storage format**: {format_info.get('recommended_format', 'hdf5')}

## Usage

To load this dataset:

```python
from braindecode.datasets import BaseConcatDataset

# Load dataset from Hugging Face Hub
dataset = BaseConcatDataset.from_pretrained("{path.name.replace('tmpdir', 'username/dataset-name')}")

# Access data
for i in range(len(dataset)):
    X, y = dataset[i]
    # X: EEG data array (shape: [n_channels, n_times])
    # y: label/target
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

# Iterate over batches
for X_batch, y_batch in train_loader:
    # X_batch shape: [batch_size, n_channels, n_times]
    # y_batch shape: [batch_size]
    pass
```

## Citation

If you use this dataset in your research, please cite braindecode:

```bibtex
@article{{schirrmeister2017deep,
  title={{Deep learning with convolutional neural networks for EEG decoding and visualization}},
  author={{Schirrmeister, Robin Tibor and Springenberg, Jost Tobias and Fiederer, Lukas Dominique Josef and Glasstetter, Martin and Eggensperger, Katharina and Tangermann, Michael and Hutter, Frank and Burgard, Wolfram and Ball, Tonio}},
  journal={{Human brain mapping}},
  volume={{38}},
  number={{11}},
  pages={{5391--5420}},
  year={{2017}},
  publisher={{Wiley Online Library}}
}}
```

## Dataset Format

This dataset is stored in **{format_info.get('recommended_format', 'zarr')}** format, optimized for:
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
                "pip install huggingface-hub"
            )

        print(f"Loading dataset from Hugging Face Hub ({repo_id})...")

        # Lazy import loader to avoid circular dependency
        from ..datautil.hub_formats import load_from_zarr
        from huggingface_hub import snapshot_download

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

            print(f"✅ Dataset loaded successfully!")
            print(f"   Recordings: {len(dataset.datasets)}")
            print(f"   Total windows/samples: {format_info.get('total_samples', 'N/A')}")

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
