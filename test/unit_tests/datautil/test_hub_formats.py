# Authors: Kuntal Kokate
#
# License: BSD-3

"""Simple tests for Zarr format converters."""

import errno
import inspect
import multiprocessing as mp
from pathlib import Path

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
    # Zarr v3 uses zarr.json for metadata (not .zgroup from v2)
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
    # Zarr v3 uses zarr.json for metadata (not .zgroup from v2)
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
    # Zarr v3 uses zarr.json for metadata (not .zgroup from v2)
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


def _signal_and_target(dataset):
    x, y = dataset[0]
    return x.item(), y


def _read_zarr_generations(old, path):
    fresh = load_from_zarr(path, preload=False)
    return _signal_and_target(old), _signal_and_target(fresh)


def _recording(value):
    info = mne.create_info(["C3"], 100, "eeg")
    raw = mne.io.RawArray(np.full((1, 10), value), info)
    return BaseConcatDataset([RawDataset(raw, {"label": value}, target_name="label")])


@pytest.mark.parametrize("legacy", [False, True])
def test_zarr_overwrite_protection(tmp_path, legacy):
    """Replacing a store preserves the recording each reader opened."""
    zarr = pytest.importorskip("zarr")
    path = convert_to_zarr(_recording(1), tmp_path / "recording.zarr")
    if legacy:
        del zarr.open_group(path, mode="a").attrs["store_id"]
    old = load_from_zarr(path, preload=False)
    with pytest.raises(FileExistsError, match="already exists"):
        convert_to_zarr(_recording(2), path)

    invalid = _recording(2)
    invalid.datasets[0].raw_preproc_kwargs = [{"scale": np.inf}]
    with pytest.raises(ValueError, match="raw_preproc_kwargs"):
        convert_to_zarr(invalid, path, overwrite=True)
    assert _signal_and_target(load_from_zarr(path)) == (1, 1)

    convert_to_zarr(_recording(2), path, overwrite=True)
    assert _signal_and_target(load_from_zarr(path)) == (2, 2)
    assert _read_zarr_generations(old, path) == ((1, 1), (2, 2))
    with mp.get_context("spawn").Pool(1) as pool:
        assert pool.apply_async(_read_zarr_generations, (old, path)).get(120) == (
            (1, 1),
            (2, 2),
        )


def test_zarr_publication_rollback(tmp_path, monkeypatch):
    pytest.importorskip("zarr")
    path = convert_to_zarr(_recording(1), tmp_path / "recording.zarr")
    rename = Path.rename
    backup = None

    def fail_publication(source, destination):
        nonlocal backup
        if source == path:
            backup = destination
        elif destination == path and source != backup:
            raise OSError("cannot publish")
        return rename(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", fail_publication)
        with pytest.raises(OSError, match="cannot publish"):
            convert_to_zarr(_recording(2), path, overwrite=True)
    assert _signal_and_target(load_from_zarr(path)) == (1, 1)

    # A killed writer cannot run its rollback after moving the previous store.
    path.rename(backup)
    assert _signal_and_target(load_from_zarr(path)) == (1, 1)
    assert path.is_dir()


def test_zarr_read_only_store_replacement(tmp_path, monkeypatch):
    pytest.importorskip("zarr")
    path = convert_to_zarr(_recording(1), tmp_path / "recording.zarr")
    replacement = convert_to_zarr(_recording(2), tmp_path / "replacement.zarr")
    cache = path.parent / f".{path.name}_memmap"

    def deny_lock(lock):
        raise PermissionError(errno.EACCES, "read-only lock file")

    monkeypatch.setattr(hub_formats.FileLock, "acquire", deny_lock)
    assert _signal_and_target(load_from_zarr(path)) == (1, 1)

    open_memmap = np.lib.format.open_memmap

    def switch_store(*args, **kwargs):
        path.rename(tmp_path / "previous.zarr")
        replacement.rename(path)
        return open_memmap(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(np.lib.format, "open_memmap", switch_store)
        with pytest.raises(OSError, match="changed while"):
            load_from_zarr(path, preload=False)
    assert not list(cache.rglob("*.npy"))
    assert _signal_and_target(load_from_zarr(path, preload=False)) == (2, 2)


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
