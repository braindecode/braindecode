# Authors: Kuntal Kokate
#
# License: BSD-3

"""Simple tests for Zarr format converters."""

import inspect

import mne
import numpy as np
import pandas as pd
import pytest

from braindecode.datasets import BNCI2014_001, BaseConcatDataset, RawDataset
from braindecode.datasets.registry import get_dataset_type
from braindecode.datautil import hub_formats
from braindecode.datautil.hub_formats import (
    convert_to_zarr,
    get_format_info,
    load_from_zarr,
)
from braindecode.preprocessing import create_windows_from_events


@pytest.fixture()
def setup_concat_windows_dataset():
    """Create a small WindowsDataset (mne.Epochs) for testing.

    Uses BNCI2014_001 dataset with real EEG data.
    """
    dataset = BNCI2014_001(subject_ids=[1])
    return create_windows_from_events(
        concat_ds=BaseConcatDataset([dataset.datasets[0]]),
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        use_mne_epochs=True,  # Creates WindowsDataset
    )


@pytest.fixture()
def setup_concat_eegwindows_dataset():
    """Create a small EEGWindowsDataset (mne.Raw) for testing.

    Uses BNCI2014_001 dataset with real EEG data.
    """
    dataset = BNCI2014_001(subject_ids=[1])
    return create_windows_from_events(
        concat_ds=BaseConcatDataset([dataset.datasets[0]]),
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        use_mne_epochs=False,  # Creates EEGWindowsDataset
    )


def test_zarr_round_trip(setup_concat_windows_dataset, tmp_path):
    """Test saving and loading WindowsDataset in Zarr format."""
    pytest.importorskip("zarr")

    dataset = setup_concat_windows_dataset
    zarr_path = tmp_path / "dataset.zarr"

    # Save to Zarr
    convert_to_zarr(dataset, zarr_path, compression="blosc", compression_level=5)

    assert zarr_path.exists()
    # In zarr v3, the metadata file is zarr.json instead of .zgroup
    assert (zarr_path / "zarr.json").exists()

    # Load from Zarr
    loaded = load_from_zarr(zarr_path, preload=True)

    assert len(loaded.datasets) == len(dataset.datasets)
    assert loaded.datasets[0].windows.info["sfreq"] == dataset.datasets[0].windows.info["sfreq"]


def test_zarr_round_trip_eegwindows(setup_concat_eegwindows_dataset, tmp_path):
    """Test saving and loading EEGWindowsDataset in Zarr format."""
    pytest.importorskip("zarr")

    dataset = setup_concat_eegwindows_dataset
    zarr_path = tmp_path / "eegwindows_dataset.zarr"

    # Save to Zarr
    convert_to_zarr(dataset, zarr_path, compression="blosc", compression_level=5)

    assert zarr_path.exists()
    # In zarr v3, the metadata file is zarr.json instead of .zgroup
    assert (zarr_path / "zarr.json").exists()

    # Load from Zarr
    loaded = load_from_zarr(zarr_path, preload=True)

    assert len(loaded.datasets) == len(dataset.datasets)
    assert loaded.datasets[0].raw.info["sfreq"] == dataset.datasets[0].raw.info["sfreq"]
    # Verify it's actually an EEGWindowsDataset
    assert get_dataset_type(loaded.datasets[0]) == "EEGWindowsDataset"


def test_get_format_info(setup_concat_windows_dataset):
    """Test getting format information from WindowsDataset."""
    dataset = setup_concat_windows_dataset

    info = get_format_info(dataset)

    assert "n_recordings" in info
    assert "total_samples" in info
    assert "total_size_mb" in info
    assert info["n_recordings"] == len(dataset.datasets)
    assert info["total_samples"] > 0
    assert info["total_size_mb"] > 0


def test_get_format_info_eegwindows(setup_concat_eegwindows_dataset):
    """Test getting format information from EEGWindowsDataset."""
    dataset = setup_concat_eegwindows_dataset

    info = get_format_info(dataset)

    assert "n_recordings" in info
    assert "total_samples" in info
    assert "total_size_mb" in info
    assert info["n_recordings"] == len(dataset.datasets)
    assert info["total_samples"] > 0
    assert info["total_size_mb"] > 0


def test_zarr_round_trip_rawdataset(tmp_path):
    """Test saving and loading RawDataset in Zarr format."""
    pytest.importorskip("zarr")

    # Create a simple RawDataset
    ch_names = ["C3", "C4", "Cz"]
    sfreq = 100
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    data = np.random.randn(3, 1000)
    raw = mne.io.RawArray(data, info)
    raw_ds = RawDataset(raw, pd.Series({"subject": "1"}))

    # Create BaseConcatDataset
    dataset = BaseConcatDataset([raw_ds])

    zarr_path = tmp_path / "rawdataset.zarr"

    # Save to Zarr
    convert_to_zarr(dataset, zarr_path, compression="blosc", compression_level=5)

    assert zarr_path.exists()
    # In zarr v3, the metadata file is zarr.json instead of .zgroup
    assert (zarr_path / "zarr.json").exists()

    # Load from Zarr
    loaded = load_from_zarr(zarr_path, preload=True)

    assert len(loaded.datasets) == len(dataset.datasets)
    assert loaded.datasets[0].raw.info["sfreq"] == dataset.datasets[0].raw.info["sfreq"]
    # Verify it's actually a RawDataset
    assert get_dataset_type(loaded.datasets[0]) == "RawDataset"


def test_get_format_info_rawdataset():
    """Test getting format information from RawDataset."""
    # Create a simple RawDataset
    ch_names = ["C3", "C4", "Cz"]
    sfreq = 100
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    data = np.random.randn(3, 1000)
    raw = mne.io.RawArray(data, info)
    raw_ds = RawDataset(raw, pd.Series({"subject": "1"}))

    dataset = BaseConcatDataset([raw_ds])

    info = get_format_info(dataset)

    assert "n_recordings" in info
    assert "total_samples" in info
    assert "total_size_mb" in info
    assert info["n_recordings"] == len(dataset.datasets)
    assert info["total_samples"] > 0
    assert info["total_size_mb"] > 0


def test_zarr_load_specific_ids(tmp_path):
    """Test loading specific recording IDs with ids_to_load parameter."""
    pytest.importorskip("zarr")

    # Create multiple RawDatasets
    ch_names = ["C3", "C4", "Cz"]
    sfreq = 100
    datasets = []
    for i in range(3):
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        data = np.random.randn(3, 1000) * (i + 1)  # Different data for each
        raw = mne.io.RawArray(data, info)
        raw_ds = RawDataset(raw, pd.Series({"subject": str(i)}))
        datasets.append(raw_ds)

    concat_dataset = BaseConcatDataset(datasets)
    zarr_path = tmp_path / "multi_recording.zarr"

    # Save all recordings
    convert_to_zarr(concat_dataset, zarr_path)

    # Load only specific IDs
    loaded = load_from_zarr(zarr_path, preload=True, ids_to_load=[0, 2])

    # Should only have 2 recordings (IDs 0 and 2)
    assert len(loaded.datasets) == 2
    # Verify they are the correct recordings by checking subject metadata
    # Note: pandas JSON serialization may convert string numbers to ints
    assert str(loaded.datasets[0].description["subject"]) == "0"
    assert str(loaded.datasets[1].description["subject"]) == "2"


def test_zarr_overwrite_protection(tmp_path):
    """Test overwrite parameter prevents accidental data loss."""
    pytest.importorskip("zarr")

    # Create a simple dataset
    ch_names = ["C3", "C4"]
    sfreq = 100
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    data = np.random.randn(2, 500)
    raw = mne.io.RawArray(data, info)
    raw_ds = RawDataset(raw, pd.Series({"subject": "1"}))
    dataset = BaseConcatDataset([raw_ds])

    zarr_path = tmp_path / "overwrite_test.zarr"

    # First save should work
    convert_to_zarr(dataset, zarr_path)
    assert zarr_path.exists()

    # Second save without overwrite=True should fail
    with pytest.raises(FileExistsError, match="already exists"):
        convert_to_zarr(dataset, zarr_path, overwrite=False)

    # Second save with overwrite=True should work
    convert_to_zarr(dataset, zarr_path, overwrite=True)
    assert zarr_path.exists()


def test_no_lazy_imports_in_hub_formats():
    """Verify that hub_formats module has global imports only."""

    # Get all functions in the hub_formats module
    functions = [
        obj for name, obj in inspect.getmembers(hub_formats)
        if inspect.isfunction(obj) and obj.__module__ == hub_formats.__name__
    ]

    # Check that no functions have lazy imports
    for func in functions:
        source = inspect.getsource(func)
        # Functions should not have 'from ..datasets' imports inside them
        lines = source.split('\n')
        for line in lines:
            # Skip the function definition line
            if line.strip().startswith('def '):
                continue
            # Check for import statements inside function body
            if 'from ..datasets' in line and 'import' in line:
                # This should only be at module level, not in functions
                assert False, f"{func.__name__} has lazy import: {line.strip()}"
