"""
Format converters for Hugging Face Hub integration.

This module provides Zarr format converters to transform EEG datasets for
efficient storage and fast random access during training on the Hugging Face Hub.

This module provides a standalone functional API that delegates to the
HubDatasetMixin methods for all actual implementations.
"""

# Authors: Kuntal Kokate
#
# License: BSD (3-clause)

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from errno import ENOLCK, ENOSYS, ENOTSUP, EOPNOTSUPP
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from filelock import FileLock

# Import registry for dynamic class lookup
from ..datasets.registry import get_dataset_class

# Import dataset classes for type checking only
if TYPE_CHECKING:
    from ..datasets.base import BaseConcatDataset


# =============================================================================
# Zarr Format Converters
# =============================================================================


@contextmanager
def _store_access(path, *, reading=False):
    """Keep a local open or directory replacement on one complete store."""
    lock = FileLock(path.parent / f".{path.name}.braindecode.lock")
    try:
        lock.acquire()
    except OSError as error:
        if reading:
            # Read-only media, or a filesystem without locks (e.g. NFS without
            # lockd), cannot host the lock. Detect a concurrent replacement.
            before = path.stat()
            yield
            if not os.path.samestat(before, path.stat()):
                raise OSError(
                    f"Zarr store {path} changed while reading it. Retry the load."
                )
            return
        if error.errno not in (ENOLCK, ENOSYS, ENOTSUP, EOPNOTSUPP):
            raise
        lock = None  # the filesystem has no locks: replace the store unlocked
    try:
        previous = path.parent / f".{path.name}_memmap" / ".previous-store"
        if previous.is_dir():  # a process stopped between publication and cleanup
            if not path.exists():
                previous.rename(path)
            elif not reading:
                shutil.rmtree(previous)
        yield previous
    finally:
        if lock is not None:
            lock.release()


def convert_to_zarr(
    dataset: BaseConcatDataset,
    output_path: Union[str, Path],
    compression: str = "blosc",
    compression_level: int = 5,
    chunk_size: int = 5_000_000,
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
    chunk_size : int, default=5_000_000
        Target chunk size for Zarr storage.
        For continuous data (e.g., RawDataset, EEGWindowsDataset), this is interpreted
        as the approximate number of samples per chunk; larger values create fewer but
        larger chunks on disk.
        For pre-cut windowed data (WindowsDataset), this controls how many windows are
        grouped together in each chunk (i.e., multiple windows can be stored per chunk).
    overwrite : bool, default=False
        Whether to replace an existing directory after writing the complete new
        store. The two stores must fit on the same filesystem during export.
        Open lazy readers retain their original signal in the sibling memmap
        cache. Remove old caches only when those readers and their workers have
        finished.

    Returns
    -------
    Path
        Path to the created Zarr directory.

    Notes
    -----
    The chunking strategy is optimized for random access:
    - Windowed data: Windows are stored in chunks of shape
      (n_windows_in_chunk, n_channels, n_times), where n_windows_in_chunk is
      determined by ``chunk_size``.
    - Raw / continuous data: Time is split into chunks of approximately
      ``chunk_size`` samples (per channel), e.g., (n_channels, n_times_per_chunk).

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
    output_path = Path(output_path)

    exists_message = f"{output_path} already exists. Set overwrite=True to replace it."
    if output_path.exists() and not overwrite:
        raise FileExistsError(exists_message)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_path.parent) as tmp:
        staged = Path(tmp) / "store"
        dataset._convert_to_zarr_inline(
            staged, compression, compression_level, chunk_size
        )
        with _store_access(output_path) as previous:
            if output_path.exists():
                if not overwrite:
                    raise FileExistsError(exists_message)
                if output_path.is_symlink() or not output_path.is_dir():
                    raise ValueError(f"Expected a Zarr directory at {output_path}")
                previous.parent.mkdir(exist_ok=True)
                output_path.rename(previous)
            try:
                staged.rename(output_path)
            except BaseException:
                if previous.exists():
                    previous.rename(output_path)
                raise
            if previous.exists():
                shutil.rmtree(previous)

    return output_path


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
    # Delegate to HubDatasetMixin static method
    BaseConcatDataset = get_dataset_class("BaseConcatDataset")

    # Readers through a symlink must share the writer's lock and recording cache.
    path = Path(input_path).resolve()
    if not path.parent.is_dir():
        raise FileNotFoundError(f"{path} does not exist.")
    with _store_access(path, reading=True):
        dataset = BaseConcatDataset._load_from_zarr_inline(path, preload)

    # Filter to specific IDs if requested
    if ids_to_load is not None:
        # Get only the requested datasets
        filtered_datasets = [dataset.datasets[i] for i in ids_to_load]
        dataset = BaseConcatDataset(filtered_datasets)

    return dataset


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
    # Delegate to HubDatasetMixin method
    return dataset._get_format_info_inline()
