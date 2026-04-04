# mypy: ignore-errors
"""
Hugging Face Hub integration for EEG datasets.

This module provides push_to_hub() and pull_from_hub() functionality
for braindecode datasets, similar to the model Hub integration.

.. warning::
    The format is **BIDS-inspired**, not **BIDS-compliant**. The metadata
    files are BIDS-compliant, but the data is stored in a backend-specific
    format (Zarr or MNE/FIF) for efficient training, which is not a valid
    BIDS EEG format.

The format follows a BIDS-inspired sourcedata structure:
- sourcedata/braindecode/
  - dataset_description.json  (BIDS-compliant)
  - participants.tsv          (BIDS-compliant)
  - dataset.zarr/ or dataset/ (backend-specific data store)
  - sub-<label>/
    - eeg/
      - *_events.tsv          (BIDS-compliant)
      - *_channels.tsv        (BIDS-compliant)
      - *_eeg.json            (BIDS-compliant)
"""

# Authors: Kuntal Kokate
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

import contextlib
import json
import logging
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import pandas as pd
from mne.utils import _soft_import

if TYPE_CHECKING:
    from ..base import BaseDataset

import braindecode

# Import registry for dynamic class lookup (avoids circular imports)
from ..registry import get_dataset_type

# Hub format and validation utilities
from . import hub_format, hub_validation
from .formats import get_format_backend
from .formats.mne_backend import MneBackend
from .formats.registry import resolve_backend_params
from .formats.zarr_backend import ZarrBackend

# Lazy import huggingface_hub
huggingface_hub = _soft_import(
    "huggingface_hub", purpose="hugging face integration", strict=False
)

log = logging.getLogger(__name__)

_LOCK_FILE = "format_info.json"


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
    ... )
    >>>
    >>> # Load dataset from Hub
    >>> dataset = BaseConcatDataset.pull_from_hub("username/nmt-dataset")
    """

    datasets: List["BaseDataset"]  # Attribute provided by inheriting class

    def push_to_hub(
        self,
        repo_id: str,
        private: bool = False,
        token: Optional[str] = None,
        backend_params: Union[dict, ZarrBackend, MneBackend, None] = None,
        pipeline_name: str = "braindecode",
        local_cache_dir: str | Path | None = None,
        **kwargs,
    ) -> str:
        """
        Upload the dataset to the Hugging Face Hub in a BIDS-like structure.

        The dataset is converted to the chosen format (Zarr or MNE/FIF) and
        stored in a BIDS sourcedata-like structure with events.tsv,
        channels.tsv, and participants.tsv sidecar files.

        Parameters
        ----------
        repo_id : str
            Repository ID on the Hugging Face Hub (e.g., "username/dataset-name").
        private : bool, default=False
            Whether to create a private repository.
        token : str | None
            Hugging Face API token. If None, uses cached token.
        backend_params : dict | ZarrBackend | MneBackend | None
            Backend-specific parameters. Pass a backend instance or a plain
            dict with a ``"format"`` discriminator key. If ``None``, defaults
            to ``ZarrBackend()`` (Zarr with default settings).

            **Zarr backend** (``ZarrBackend`` / ``{"format": "zarr", ...}``):

            - ``compression`` : str — Compression algorithm
              (default ``"blosc"``). Options: ``"blosc"``, ``"zstd"``,
              ``"gzip"``, ``None``.
            - ``compression_level`` : int — Compression level 0-9
              (default ``5``).
            - ``chunk_size`` : int — Samples per chunk
              (default ``5_000_000``).

            **MNE backend** (``MneBackend`` / ``{"format": "mne", ...}``):

            - ``split_size`` : str — Max file size before splitting
              (default ``"2GB"``). E.g. ``"2GB"``, ``"500MB"``.

            Examples::

                # Zarr with defaults
                ds.push_to_hub("repo")

                # MNE with defaults (dict form)
                ds.push_to_hub("repo", backend_params={"format": "mne"})

                # MNE with custom split size
                ds.push_to_hub("repo", backend_params=MneBackend(split_size="1GB"))

        pipeline_name : str, default="braindecode"
            Name of the processing pipeline for BIDS sourcedata.
        local_cache_dir : str | Path | None
            Local directory to use for temporary files during upload. If None, uses
            the system temp directory and cleans it up after upload. If provided,
            the directory is used as a persistent cache:

            - If the directory is empty (or does not exist), the cache is built
              there and a lock file (``format_info.json``) is written once
              the cache is complete, before the upload starts. The file
              contains the conversion parameters as JSON.
            - If the lock file is present and its JSON parameters match the
              current call, cache creation is skipped and the upload resumes
              directly (useful for retrying interrupted uploads).
            - If the lock file is present but its JSON parameters differ from
              the current call, a ``ValueError`` is raised.
            - If the directory is non-empty but the lock file is absent, a
              ``ValueError`` is raised listing the files found.
        **kwargs
            Additional arguments passed to huggingface_hub.upload_large_folder().

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
        >>> # Upload with BIDS-like structure
        >>> url = dataset.push_to_hub(
        ...     repo_id="myusername/nmt-dataset",
        ... )
        """
        if huggingface_hub is False:
            raise ImportError(
                "huggingface-hub is not installed. Install with: "
                "pip install braindecode[hub]"
            )

        backend = resolve_backend_params(backend_params)
        backend.validate_dependencies()

        # Create API instance
        hf_api = huggingface_hub.HfApi(token=token)

        # Create repository if it doesn't exist
        try:
            hf_api.create_repo(
                repo_id=repo_id,
                private=private,
                repo_type="dataset",
                exist_ok=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create repository: {e}")

        format_info = self._get_format_info_inline()
        format_info_lock = {
            **backend.build_format_info(),
            "pipeline_name": pipeline_name,
            "braindecode_version": braindecode.__version__,
            **format_info,
        }

        # Determine upload directory and whether to build the cache
        with contextlib.ExitStack() as stack:
            if local_cache_dir is None:
                tmpdir = stack.enter_context(tempfile.TemporaryDirectory())
                tmp_path = Path(tmpdir)
                build_cache = True
            else:
                tmp_path = Path(local_cache_dir)
                lock_path = tmp_path / _LOCK_FILE
                if lock_path.exists():
                    with open(lock_path, "r", encoding="utf-8") as _f:
                        _lock_params = json.load(_f)
                    if _lock_params != format_info_lock:
                        raise ValueError(
                            f"Lock file found at '{lock_path}' but its "
                            f"parameters {_lock_params} differ from the "
                            f"current call parameters {format_info_lock}. "
                            "Provide an empty directory or match the "
                            "original parameters."
                        )
                    log.info(
                        f"Lock file found at '{lock_path}', skipping cache "
                        "creation and resuming upload."
                    )
                    build_cache = False
                else:
                    if tmp_path.exists():
                        existing = list(tmp_path.iterdir())
                        if existing:
                            entries = ", ".join(p.name for p in existing)
                            raise ValueError(
                                f"local_cache_dir '{tmp_path}' is not empty and "
                                f"has no lock file. Found: {entries}. Provide an "
                                "empty directory or one previously prepared by "
                                "push_to_hub()."
                            )
                    else:
                        tmp_path.mkdir(parents=True)
                    build_cache = True

            if build_cache:
                self._build_local_cache(tmp_path, format_info_lock)

            # Upload folder to Hub
            log.info(f"Uploading to Hugging Face Hub ({repo_id})...")
            try:
                url = hf_api.upload_large_folder(
                    repo_id=repo_id,
                    folder_path=str(tmp_path),
                    repo_type="dataset",
                    **kwargs,
                )
                log.info(f"Dataset uploaded successfully to {repo_id}")
                log.info(f"URL: https://huggingface.co/datasets/{repo_id}")
                return url
            except Exception as e:
                raise RuntimeError(f"Failed to upload dataset: {e}")

    def _build_local_cache(
        self,
        tmp_path,
        format_info_lock,
    ):
        """Build the local cache directory with the dataset and BIDS-like structure.
        This folder will be uploaded to the Hub."""
        pipeline_name = format_info_lock["pipeline_name"]

        # Create BIDS-like sourcedata structure
        log.info("Creating BIDS-like sourcedata structure...")
        bids_layout = hub_format.BIDSSourcedataLayout(
            tmp_path, pipeline_name=pipeline_name
        )
        sourcedata_dir = bids_layout.create_structure()

        # Save dataset_description.json
        bids_layout.save_dataset_description()

        # Save participants.tsv
        descriptions = [ds.description for ds in self.datasets]
        bids_layout.save_participants(descriptions)

        # Save BIDS sidecar files for each recording
        self._save_bids_sidecar_files(bids_layout)

        # Convert dataset using the appropriate format backend
        backend = resolve_backend_params(format_info_lock)
        data_filename = backend.get_data_filename()
        if data_filename is not None:
            dataset_path = sourcedata_dir / data_filename
        else:
            # Backend writes per-subject files into the BIDS tree
            dataset_path = sourcedata_dir
        log.info(f"Converting dataset to {backend.name} format...")
        backend.convert_datasets(self.datasets, dataset_path)

        # Save dataset metadata (README.md)
        self._save_dataset_card(tmp_path)

        # Save format info
        # This marks the cache as complete
        format_info_path = tmp_path / _LOCK_FILE
        with open(format_info_path, "w", encoding="utf-8") as f:
            json.dump(format_info_lock, f, indent=2)

    def _save_dataset_card(self, path: Path, bids_inspired: bool = True) -> None:
        """Generate and save a dataset card (README.md) with metadata.

        Parameters
        ----------
        path : Path
            Directory where README.md will be saved.
        bids_inspired : bool
            Whether to include BIDS-inspired format documentation.
        """
        # Get info, which also validates uniformity across all datasets
        format_info = self._get_format_info_inline()

        n_recordings = len(self.datasets)
        first_ds = self.datasets[0]

        # Get dataset-specific info based on type using registry
        dataset_type = get_dataset_type(first_ds)

        n_windows = format_info["total_samples"]

        # Compute total duration across all recordings
        total_duration = 0.0
        if dataset_type == "WindowsDataset":
            n_channels = len(first_ds.windows.ch_names)
            data_type = "Windowed (from Epochs object)"
            sfreq = first_ds.windows.info["sfreq"]
            for ds in self.datasets:
                epoch_length = ds.windows.tmax - ds.windows.tmin
                total_duration += len(ds.windows) * epoch_length
        elif dataset_type == "EEGWindowsDataset":
            n_channels = len(first_ds.raw.ch_names)
            sfreq = first_ds.raw.info["sfreq"]
            data_type = "Windowed (from Raw object)"
            for ds in self.datasets:
                total_duration += ds.raw.n_times / ds.raw.info["sfreq"]
        elif dataset_type == "RawDataset":
            n_channels = len(first_ds.raw.ch_names)
            sfreq = first_ds.raw.info["sfreq"]
            data_type = "Continuous (Raw)"
            for ds in self.datasets:
                total_duration += ds.raw.n_times / ds.raw.info["sfreq"]
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
            total_duration=total_duration,
        )

        # Save README
        readme_path = path / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)

    def _save_bids_sidecar_files(
        self, bids_layout: "hub_format.BIDSSourcedataLayout"
    ) -> None:
        """Save BIDS-compliant sidecar files for each recording.

        This creates events.tsv, channels.tsv, and EEG sidecar JSON files
        for each recording in a BIDS-like directory structure.

        Parameters
        ----------
        bids_layout : BIDSSourcedataLayout
            BIDS layout object for path generation.
        """
        dataset_type = get_dataset_type(self.datasets[0])

        for i_ds, ds in enumerate(self.datasets):
            # Get BIDS entities from description
            description = ds.description if ds.description is not None else pd.Series()

            # Get BIDSPath for this recording using mne_bids
            bids_path = bids_layout.get_bids_path(description)

            # Create subject directory
            bids_path.mkdir(exist_ok=True)

            # Get metadata and info based on dataset type
            # Also compute recording_duration, recording_type, and epoch_length
            recording_duration = None
            recording_type = None
            epoch_length = None

            if dataset_type == "WindowsDataset":
                info = ds.windows.info
                metadata = ds.windows.metadata
                sfreq = info["sfreq"]
                # WindowsDataset contains pre-cut epochs
                recording_type = "epoched"
                # Use MNE's tmax - tmin for epoch length
                epoch_length = ds.windows.tmax - ds.windows.tmin
                # Total duration = number of epochs * epoch length
                n_epochs = len(ds.windows)
                recording_duration = n_epochs * epoch_length
            elif dataset_type == "EEGWindowsDataset":
                info = ds.raw.info
                metadata = ds.metadata
                sfreq = info["sfreq"]
                # EEGWindowsDataset has continuous raw with window metadata
                recording_type = "epoched"
                # Use MNE Raw's duration property
                recording_duration = ds.raw.duration
                # Compute epoch_length from metadata if available
                if metadata is not None and len(metadata) > 0:
                    i_start = metadata["i_start_in_trial"].iloc[0]
                    i_stop = metadata["i_stop_in_trial"].iloc[0]
                    epoch_length = (i_stop - i_start) / sfreq
            elif dataset_type == "RawDataset":
                info = ds.raw.info
                metadata = None
                sfreq = info["sfreq"]
                # RawDataset is continuous
                recording_type = "continuous"
                # Use MNE Raw's duration property
                recording_duration = ds.raw.duration
            else:
                continue

            # Determine task name from description or BIDSPath
            task_name = bids_path.task or "unknown"

            # Save BIDS sidecar files using mne_bids BIDSPath
            hub_format.save_bids_sidecar_files(
                bids_path=bids_path,
                info=info,
                metadata=metadata,
                sfreq=sfreq,
                task_name=str(task_name),
                recording_duration=recording_duration,
                recording_type=recording_type,
                epoch_length=epoch_length,
            )

            log.debug(
                f"Saved BIDS sidecar files for recording {i_ds} to {bids_path.directory}"
            )

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
        if huggingface_hub is False:
            raise ImportError(
                "huggingface-hub is not installed. Install with: "
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
            format_info_path = Path(dataset_dir) / _LOCK_FILE
            if format_info_path.exists():
                with open(format_info_path, "r") as f:
                    format_info = json.load(f)
            else:
                format_info = {}

            # Auto-detect format (default to zarr for backward compatibility)
            format_name = format_info.get("format", "zarr")
            backend_cls = get_format_backend(format_name)
            backend = backend_cls()
            backend.validate_dependencies()

            pipeline_name = format_info.get("pipeline_name", "braindecode")
            data_filename = backend.get_data_filename()

            # Find data path (try sourcedata, derivatives, then root)
            if data_filename is not None:
                data_path = (
                    Path(dataset_dir) / "sourcedata" / pipeline_name / data_filename
                )
                if not data_path.exists():
                    data_path = (
                        Path(dataset_dir)
                        / "derivatives"
                        / pipeline_name
                        / data_filename
                    )
                if not data_path.exists():
                    data_path = Path(dataset_dir) / data_filename
            else:
                # Backend uses per-subject files; point to sourcedata dir
                data_path = Path(dataset_dir) / "sourcedata" / pipeline_name
                if not data_path.exists():
                    data_path = Path(dataset_dir) / "derivatives" / pipeline_name

            if not data_path.exists():
                raise FileNotFoundError(
                    f"Dataset not found at {data_path}. "
                    "The dataset may be in an unsupported format."
                )

            dataset = backend.load_datasets(data_path, preload)

            # Load BIDS metadata if available
            cls._load_bids_metadata(dataset, Path(dataset_dir), pipeline_name)

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

    @classmethod
    def _load_bids_metadata(
        cls,
        dataset,
        dataset_dir: Path,
        pipeline_name: str,
    ) -> None:
        """Load BIDS metadata from sidecar files and attach to dataset.

        Parameters
        ----------
        dataset : BaseConcatDataset
            The loaded dataset to attach metadata to.
        dataset_dir : Path
            Root directory of the downloaded dataset.
        pipeline_name : str
            Name of the processing pipeline.
        """
        # Try sourcedata first, fall back to derivatives for backwards compatibility
        sourcedata_dir = dataset_dir / "sourcedata" / pipeline_name
        if not sourcedata_dir.exists():
            sourcedata_dir = dataset_dir / "derivatives" / pipeline_name

        # Load participants.tsv if available
        participants_path = sourcedata_dir / "participants.tsv"
        if participants_path.exists():
            try:
                participants_df = pd.read_csv(participants_path, sep="\t")
                # Store as attribute on the concat dataset
                dataset.participants = participants_df
                log.debug(
                    f"Loaded participants info for {len(participants_df)} subjects"
                )
            except Exception as e:
                log.warning(f"Failed to load participants.tsv: {e}")

        # Create layout for path generation
        bids_layout = hub_format.BIDSSourcedataLayout(
            dataset_dir, pipeline_name=pipeline_name
        )

        # Try to load events.tsv files and attach to individual datasets
        for i_ds, ds in enumerate(dataset.datasets):
            description = ds.description if ds.description is not None else pd.Series()

            # Get BIDSPath for this recording
            bids_path = bids_layout.get_bids_path(description)

            # Load events.tsv if available
            events_path = bids_path.copy().update(suffix="events", extension=".tsv")
            if events_path.fpath.exists():
                try:
                    events_df = pd.read_csv(events_path.fpath, sep="\t")
                    ds.bids_events = events_df
                    log.debug(f"Loaded events for recording {i_ds}")
                except Exception as e:
                    log.warning(f"Failed to load events for recording {i_ds}: {e}")

            # Load channels.tsv if available
            channels_path = bids_path.copy().update(suffix="channels", extension=".tsv")
            if channels_path.fpath.exists():
                try:
                    channels_df = pd.read_csv(channels_path.fpath, sep="\t")
                    ds.bids_channels = channels_df
                    log.debug(f"Loaded channels for recording {i_ds}")
                except Exception as e:
                    log.warning(f"Failed to load channels for recording {i_ds}: {e}")

    # ------------------------------------------------------------------
    # Backward-compatible wrappers (delegate to format backends)
    # ------------------------------------------------------------------

    def _convert_to_zarr_inline(
        self,
        output_path,
        compression="blosc",
        compression_level=5,
        chunk_size=5_000_000,
    ):
        """Backward-compatible wrapper — delegates to ZarrBackend."""
        backend = ZarrBackend(
            compression=compression,
            compression_level=compression_level,
            chunk_size=chunk_size,
        )
        backend.convert_datasets(self.datasets, output_path)

    @staticmethod
    def _load_from_zarr_inline(input_path, preload=True):
        """Backward-compatible wrapper — delegates to ZarrBackend."""
        backend = ZarrBackend()
        return backend.load_datasets(input_path, preload)

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


def _generate_readme_content(
    format_info,
    n_recordings: int,
    n_channels: int,
    sfreq,
    data_type: str,
    n_windows: int,
    total_duration: float | None = None,
    format: str = "zarr",
):
    """Generate README.md content for a dataset uploaded to the Hub.

    Parameters
    ----------
    format_info : dict
        Dictionary containing format metadata (e.g., total_size_mb).
    n_recordings : int
        Number of recordings in the dataset.
    n_channels : int
        Number of EEG channels.
    sfreq : float or None
        Sampling frequency in Hz.
    data_type : str
        Type of dataset (e.g., "Windowed", "Continuous").
    n_windows : int
        Number of windows/samples in the dataset.
    total_duration : float or None
        Total duration in seconds across all recordings.
    format : str
        Storage format (default: "zarr").

    Returns
    -------
    str
        Markdown content for the README.md file.
    """
    total_size_mb = (
        format_info.get("total_size_mb", 0.0) if isinstance(format_info, dict) else 0.0
    )
    sfreq_str = f"{sfreq:g}" if sfreq is not None else "N/A"

    duration_str = (
        str(timedelta(seconds=int(total_duration))) if total_duration else "N/A"
    )

    return f"""---
tags:
- braindecode
- eeg
- neuroscience
- brain-computer-interface
- deep-learning
license: unknown
---

# EEG Dataset

This dataset was created using [braindecode](https://braindecode.org), a deep
learning library for EEG/MEG/ECoG signals.

## Dataset Information

| Property | Value |
|----------|------:|
| Recordings | {n_recordings} |
| Type | {data_type} |
| Channels | {n_channels} |
| Sampling frequency | {sfreq_str} Hz |
| Total duration | {duration_str} |
| Windows/samples | {n_windows:,} |
| Size | {total_size_mb:.2f} MB |
| Format | {format} |

## Quick Start

```python
from braindecode.datasets import BaseConcatDataset

# Load from Hugging Face Hub
dataset = BaseConcatDataset.pull_from_hub("username/dataset-name")

# Access a sample
X, y, metainfo = dataset[0]
# X: EEG data [n_channels, n_times]
# y: target label
# metainfo: window indices
```

## Training with PyTorch

```python
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

for X, y, metainfo in loader:
    # X: [batch_size, n_channels, n_times]
    # y: [batch_size]
    pass  # Your training code
```

## BIDS-inspired Structure

This dataset uses a **BIDS-inspired** organization. Metadata files follow BIDS
conventions, while data is stored in Zarr format for efficient deep learning.

**BIDS-style metadata:**
- `dataset_description.json` - Dataset information
- `participants.tsv` - Subject metadata
- `*_events.tsv` - Trial/window events
- `*_channels.tsv` - Channel information
- `*_eeg.json` - Recording parameters

**Data storage:**
- `dataset.zarr/` - Zarr format (optimized for random access)

```
sourcedata/braindecode/
├── dataset_description.json
├── participants.tsv
├── dataset.zarr/
└── sub-<label>/
    └── eeg/
        ├── *_events.tsv
        ├── *_channels.tsv
        └── *_eeg.json
```

### Accessing Metadata

```python
# Participants info
if hasattr(dataset, "participants"):
    print(dataset.participants)

# Events for a recording
if hasattr(dataset.datasets[0], "bids_events"):
    print(dataset.datasets[0].bids_events)

# Channel info
if hasattr(dataset.datasets[0], "bids_channels"):
    print(dataset.datasets[0].bids_channels)
```

---

*Created with [braindecode](https://braindecode.org)*
"""
