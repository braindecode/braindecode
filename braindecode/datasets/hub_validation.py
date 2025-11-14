"""
Shared validation utilities for Hub format operations.

This module provides validation functions used by hub.py to avoid code duplication.
"""

# Authors: Kuntal Kokate
#
# License: BSD (3-clause)

from typing import Any, List, Tuple

from .registry import get_dataset_type


def validate_dataset_uniformity(
    datasets: List[Any],
) -> Tuple[str, List[str], float]:
    """
    Validate all datasets have uniform type, channels, and sampling frequency.

    Parameters
    ----------
    datasets : list
        List of dataset objects to validate.

    Returns
    -------
    dataset_type : str
        The validated dataset type (WindowsDataset, EEGWindowsDataset, or RawDataset).
    first_ch_names : list of str
        Channel names from the first dataset.
    first_sfreq : float
        Sampling frequency from the first dataset.

    Raises
    ------
    ValueError
        If datasets have mixed types, inconsistent channels, or inconsistent
        sampling frequencies.
    TypeError
        If dataset type is not supported.
    """
    if not datasets:
        raise ValueError("No datasets provided for validation.")

    first_ds = datasets[0]
    dataset_type = get_dataset_type(first_ds)

    # Get reference channel names and sampling frequency from the first dataset
    first_ch_names, first_sfreq = _get_ch_names_and_sfreq(first_ds, dataset_type)

    # Validate all datasets have uniform properties
    for i, ds in enumerate(datasets):
        ds_type = get_dataset_type(ds)
        if ds_type != dataset_type:
            raise ValueError(
                f"Mixed dataset types in concat: dataset 0 is {dataset_type} "
                f"but dataset {i} is {ds_type}"
            )

        ch_names, sfreq = _get_ch_names_and_sfreq(ds, dataset_type)

        if ch_names != first_ch_names:
            raise ValueError(
                f"Inconsistent channel names: dataset 0 has {first_ch_names} "
                f"but dataset {i} has {ch_names}"
            )

        if sfreq != first_sfreq:
            _raise_sfreq_error(first_sfreq, sfreq, i)

    return dataset_type, first_ch_names, first_sfreq


def _get_ch_names_and_sfreq(ds: Any, dataset_type: str) -> Tuple[List[str], float]:
    """Return (ch_names, sfreq) for supported dataset types."""
    if dataset_type == "WindowsDataset":
        obj = ds.windows
    elif dataset_type in ("EEGWindowsDataset", "RawDataset"):
        obj = ds.raw
    else:
        raise TypeError(f"Unsupported dataset type: {dataset_type}")

    return obj.ch_names, obj.info["sfreq"]


def _raise_sfreq_error(expected: float, actual: float, idx: int):
    """
    Raise standardized sampling frequency error.

    Parameters
    ----------
    expected : float
        Expected sampling frequency from dataset 0.
    actual : float
        Actual sampling frequency from current dataset.
    idx : int
        Index of the dataset with inconsistent sampling frequency.

    Raises
    ------
    ValueError
        Always raised with standardized error message.
    """
    raise ValueError(
        f"Inconsistent sampling frequencies: dataset 0 has {expected} Hz "
        f"but dataset {idx} has {actual} Hz. "
        f"Please resample all datasets to a common frequency before saving. "
        f"Use braindecode.preprocessing.preprocess("
        f"[Preprocessor(Resample(sfreq={expected}))], concat_ds) "
        f"to resample your datasets."
    )
