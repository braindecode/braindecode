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

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

# Import registry for dynamic class lookup
from ..datasets.registry import get_dataset_class

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
    output_path = Path(output_path)

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_path} already exists. Set overwrite=True to replace it."
            )
        # Remove existing directory if overwrite is True
        shutil.rmtree(output_path)

    # Delegate to HubDatasetMixin method
    dataset._convert_to_zarr_inline(output_path, compression, compression_level)

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

    # Load full dataset using mixin method
    dataset = BaseConcatDataset._load_from_zarr_inline(Path(input_path), preload)

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
