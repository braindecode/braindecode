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
    # Get type and properties from first dataset
    first_ds = datasets[0]
    dataset_type = get_dataset_type(first_ds)

    # Extract reference channel names and sampling frequency
    if dataset_type == "WindowsDataset":
        first_ch_names = first_ds.windows.ch_names
        first_sfreq = first_ds.windows.info["sfreq"]
    elif dataset_type == "EEGWindowsDataset":
        first_ch_names = first_ds.raw.ch_names
        first_sfreq = first_ds.raw.info["sfreq"]
    elif dataset_type == "RawDataset":
        first_ch_names = first_ds.raw.ch_names
        first_sfreq = first_ds.raw.info["sfreq"]
    else:
        raise TypeError(f"Unsupported dataset type: {dataset_type}")

    # Validate all datasets have uniform properties
    for i, ds in enumerate(datasets):
        # Check dataset type consistency
        ds_type = get_dataset_type(ds)
        if ds_type != dataset_type:
            raise ValueError(
                f"Mixed dataset types in concat: dataset 0 is {dataset_type} "
                f"but dataset {i} is {ds_type}"
            )

        # Validate based on dataset type
        if dataset_type == "WindowsDataset":
            _validate_windows_dataset(ds, first_ch_names, first_sfreq, i)
        elif dataset_type == "EEGWindowsDataset":
            _validate_eegwindows_dataset(ds, first_ch_names, first_sfreq, i)
        elif dataset_type == "RawDataset":
            _validate_raw_dataset(ds, first_ch_names, first_sfreq, i)

    return dataset_type, first_ch_names, first_sfreq


def _validate_windows_dataset(
    ds: Any, first_ch_names: List[str], first_sfreq: float, idx: int
):
    """Validate WindowsDataset properties."""
    if ds.windows.ch_names != first_ch_names:
        raise ValueError(
            f"Inconsistent channel names: dataset 0 has {first_ch_names} "
            f"but dataset {idx} has {ds.windows.ch_names}"
        )
    if ds.windows.info["sfreq"] != first_sfreq:
        _raise_sfreq_error(first_sfreq, ds.windows.info["sfreq"], idx)


def _validate_eegwindows_dataset(
    ds: Any, first_ch_names: List[str], first_sfreq: float, idx: int
):
    """Validate EEGWindowsDataset properties."""
    if ds.raw.ch_names != first_ch_names:
        raise ValueError(
            f"Inconsistent channel names: dataset 0 has {first_ch_names} "
            f"but dataset {idx} has {ds.raw.ch_names}"
        )
    if ds.raw.info["sfreq"] != first_sfreq:
        _raise_sfreq_error(first_sfreq, ds.raw.info["sfreq"], idx)


def _validate_raw_dataset(
    ds: Any, first_ch_names: List[str], first_sfreq: float, idx: int
):
    """Validate RawDataset properties."""
    if ds.raw.ch_names != first_ch_names:
        raise ValueError(
            f"Inconsistent channel names: dataset 0 has {first_ch_names} "
            f"but dataset {idx} has {ds.raw.ch_names}"
        )
    if ds.raw.info["sfreq"] != first_sfreq:
        _raise_sfreq_error(first_sfreq, ds.raw.info["sfreq"], idx)


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
        f"Use braindecode.preprocessing.Resample with Resample(sfreq={expected})."
    )
