# Authors: Kuntal Kokate
#
# License: BSD-3

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from braindecode.datasets import BaseConcatDataset, MOABBDataset
from braindecode.datautil.hub_formats import (
    convert_to_zarr,
    get_format_info,
    load_from_zarr,
)
from braindecode.preprocessing import create_windows_from_events

# MOABB fake dataset configuration
bnci_kwargs = {
    "n_sessions": 1,
    "n_runs": 2,
    "n_subjects": 2,
    "paradigm": "imagery",
    "duration": 100.0,  # Shorter for faster tests
    "sfreq": 250,
    "event_list": ("feet", "left_hand", "right_hand", "tongue"),
    "channels": ("C3", "Cz", "C4"),
}


@pytest.fixture()
def setup_concat_raw_dataset():
    """Create a small raw dataset for testing."""
    return MOABBDataset(
        dataset_name="FakeDataset", subject_ids=[1], dataset_kwargs=bnci_kwargs
    )


@pytest.fixture()
def setup_concat_windows_dataset(setup_concat_raw_dataset):
    """Create a windowed dataset for testing."""
    moabb_dataset = setup_concat_raw_dataset
    return create_windows_from_events(
        concat_ds=moabb_dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        use_mne_epochs=True,  # Required for .windows attribute
    )


# =============================================================================
# Zarr Format Tests
# =============================================================================


def test_zarr_round_trip_windows(setup_concat_windows_dataset, tmpdir):
    """Test Zarr conversion and loading preserves windowed data."""
    pytest.importorskip("zarr")

    dataset = setup_concat_windows_dataset
    output_path = Path(tmpdir) / "test_dataset.zarr"

    # Convert to Zarr
    result_path = convert_to_zarr(dataset, output_path, overwrite=False)
    assert result_path.exists()
    assert result_path.is_dir()

    # Load from Zarr
    loaded_dataset = load_from_zarr(output_path, preload=True)

    # Verify dataset structure
    assert isinstance(loaded_dataset, BaseConcatDataset)
    assert len(loaded_dataset.datasets) == len(dataset.datasets)

    # Verify data integrity
    for i, (original_ds, loaded_ds) in enumerate(
        zip(dataset.datasets, loaded_dataset.datasets)
    ):
        original_data = original_ds.windows.get_data()
        loaded_data = loaded_ds.windows.get_data()

        assert original_data.shape == loaded_data.shape
        np.testing.assert_allclose(original_data, loaded_data, rtol=1e-5)

        # Verify metadata
        assert original_ds.windows.metadata.shape == loaded_ds.windows.metadata.shape


def test_zarr_compression(setup_concat_windows_dataset, tmpdir):
    """Test Zarr compression options."""
    pytest.importorskip("zarr")

    dataset = setup_concat_windows_dataset

    # Test with blosc compression (default)
    path_blosc = Path(tmpdir) / "blosc.zarr"
    convert_to_zarr(dataset, path_blosc, compression="blosc", compression_level=5)
    assert path_blosc.exists()

    # Test without compression
    path_no_comp = Path(tmpdir) / "no_compression.zarr"
    convert_to_zarr(dataset, path_no_comp, compression=None)
    assert path_no_comp.exists()

    # Both should be loadable
    loaded_blosc = load_from_zarr(path_blosc)
    loaded_no_comp = load_from_zarr(path_no_comp)

    assert len(loaded_blosc.datasets) == len(loaded_no_comp.datasets)

    # Verify data is the same
    for i in range(len(dataset.datasets)):
        blosc_data = loaded_blosc.datasets[i].windows.get_data()
        no_comp_data = loaded_no_comp.datasets[i].windows.get_data()
        np.testing.assert_allclose(blosc_data, no_comp_data, rtol=1e-5)


def test_zarr_overwrite(setup_concat_windows_dataset, tmpdir):
    """Test Zarr overwrite functionality."""
    pytest.importorskip("zarr")

    dataset = setup_concat_windows_dataset
    output_path = Path(tmpdir) / "test_overwrite.zarr"

    # First save
    convert_to_zarr(dataset, output_path)

    # Try to save again without overwrite - should raise error
    with pytest.raises(FileExistsError):
        convert_to_zarr(dataset, output_path, overwrite=False)

    # Save with overwrite - should succeed
    convert_to_zarr(dataset, output_path, overwrite=True)


def test_zarr_partial_load(setup_concat_windows_dataset, tmpdir):
    """Test loading only specific datasets from Zarr."""
    pytest.importorskip("zarr")

    dataset = setup_concat_windows_dataset
    output_path = Path(tmpdir) / "test_partial.zarr"

    # Convert to Zarr
    convert_to_zarr(dataset, output_path)

    # Load only first dataset
    loaded_dataset = load_from_zarr(output_path, ids_to_load=[0])

    assert len(loaded_dataset.datasets) == 1

    # Verify the loaded dataset is correct
    original_data = dataset.datasets[0].windows.get_data()
    loaded_data = loaded_dataset.datasets[0].windows.get_data()
    np.testing.assert_allclose(original_data, loaded_data, rtol=1e-5)


def test_zarr_compression_levels(setup_concat_windows_dataset, tmpdir):
    """Test different Zarr compression levels."""
    pytest.importorskip("zarr")

    dataset = setup_concat_windows_dataset

    # Test different compression levels
    for level in [1, 5, 9]:
        path = Path(tmpdir) / f"compression_level_{level}.zarr"
        convert_to_zarr(dataset, path, compression="blosc", compression_level=level)
        assert path.exists()

        # Verify it's loadable
        loaded = load_from_zarr(path)
        assert len(loaded.datasets) == len(dataset.datasets)


def test_zarr_preserves_preprocessing_kwargs(setup_concat_windows_dataset, tmpdir):
    """Test that Zarr preserves preprocessing kwargs."""
    pytest.importorskip("zarr")

    dataset = setup_concat_windows_dataset
    output_path = Path(tmpdir) / "test_kwargs.zarr"

    # Add some preprocessing kwargs to the dataset
    dataset.raw_preproc_kwargs = {"test": "value"}
    dataset.window_kwargs = {"window_size": 1000}

    # Convert and load
    convert_to_zarr(dataset, output_path)
    loaded_dataset = load_from_zarr(output_path)

    # Verify kwargs are preserved
    assert hasattr(loaded_dataset, "raw_preproc_kwargs")
    assert loaded_dataset.raw_preproc_kwargs == {"test": "value"}
    assert hasattr(loaded_dataset, "window_kwargs")
    assert loaded_dataset.window_kwargs == {"window_size": 1000}


# =============================================================================
# Utility Function Tests
# =============================================================================


def test_get_format_info_windows(setup_concat_windows_dataset):
    """Test get_format_info returns correct information for windowed dataset."""
    dataset = setup_concat_windows_dataset
    info = get_format_info(dataset)

    assert "n_recordings" in info
    assert "total_samples" in info
    assert "total_size_mb" in info

    assert info["n_recordings"] == len(dataset.datasets)
    assert info["total_size_mb"] > 0


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_load_nonexistent_file():
    """Test that loading nonexistent file raises appropriate error."""
    pytest.importorskip("zarr")

    with pytest.raises(FileNotFoundError):
        load_from_zarr("/nonexistent/path.zarr")


def test_zarr_import_error(setup_concat_windows_dataset, tmpdir, monkeypatch):
    """Test that appropriate error is raised when zarr is not installed."""
    # Mock zarr as unavailable
    monkeypatch.setattr("braindecode.datautil.hub_formats.ZARR_AVAILABLE", False)

    output_path = Path(tmpdir) / "test.zarr"

    with pytest.raises(ImportError, match="Zarr is not installed"):
        convert_to_zarr(setup_concat_windows_dataset, output_path)

    with pytest.raises(ImportError, match="Zarr is not installed"):
        load_from_zarr(output_path)


def test_zarr_channel_info_preservation(setup_concat_windows_dataset, tmpdir):
    """Test that channel information is preserved through Zarr conversion."""
    pytest.importorskip("zarr")

    dataset = setup_concat_windows_dataset
    output_path = Path(tmpdir) / "test_channels.zarr"

    # Get original channel info
    original_ch_names = dataset.datasets[0].windows.ch_names
    original_sfreq = dataset.datasets[0].windows.info["sfreq"]

    # Convert and load
    convert_to_zarr(dataset, output_path)
    loaded_dataset = load_from_zarr(output_path)

    # Verify channel info is preserved
    loaded_ch_names = loaded_dataset.datasets[0].windows.ch_names
    loaded_sfreq = loaded_dataset.datasets[0].windows.info["sfreq"]

    assert original_ch_names == loaded_ch_names
    assert original_sfreq == loaded_sfreq
