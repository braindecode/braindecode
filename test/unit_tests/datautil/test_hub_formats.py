# Authors: Kuntal Kokate, Bruno Aristimunha
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
    convert_to_hdf5,
    convert_to_npz_parquet,
    convert_to_zarr,
    get_format_info,
    load_from_hdf5,
    load_from_npz_parquet,
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
    )


# =============================================================================
# HDF5 Format Tests
# =============================================================================


def test_hdf5_round_trip_windows(setup_concat_windows_dataset, tmpdir):
    """Test HDF5 conversion and loading preserves windowed data."""
    dataset = setup_concat_windows_dataset
    output_path = Path(tmpdir) / "test_dataset.h5"

    # Convert to HDF5
    result_path = convert_to_hdf5(dataset, output_path, overwrite=False)
    assert result_path.exists()
    assert result_path == output_path

    # Load from HDF5
    loaded_dataset = load_from_hdf5(output_path, preload=True)

    # Verify dataset structure
    assert isinstance(loaded_dataset, BaseConcatDataset)
    assert len(loaded_dataset.datasets) == len(dataset.datasets)

    # Verify data integrity
    for i, (original_ds, loaded_ds) in enumerate(
        zip(dataset.datasets, loaded_dataset.datasets)
    ):
        original_data = original_ds.windows.get_data()
        loaded_data = loaded_ds.windows.get_data()

        assert original_data.shape == loaded_data.shape, \
            f"Data shape mismatch at dataset {i}"
        np.testing.assert_allclose(
            original_data, loaded_data, rtol=1e-5,
            err_msg=f"Data values mismatch at dataset {i}"
        )

        # Verify metadata
        assert original_ds.windows.metadata.shape == loaded_ds.windows.metadata.shape
        # Note: We compare key columns, as some metadata might be reconstructed
        for col in ["target"]:
            if col in original_ds.windows.metadata.columns:
                assert col in loaded_ds.windows.metadata.columns
                np.testing.assert_array_equal(
                    original_ds.windows.metadata[col].values,
                    loaded_ds.windows.metadata[col].values,
                )


def test_hdf5_round_trip_raw(setup_concat_raw_dataset, tmpdir):
    """Test HDF5 conversion and loading preserves raw data."""
    dataset = setup_concat_raw_dataset
    output_path = Path(tmpdir) / "test_dataset_raw.h5"

    # Convert to HDF5
    convert_to_hdf5(dataset, output_path)
    assert output_path.exists()

    # Load from HDF5
    loaded_dataset = load_from_hdf5(output_path, preload=True)

    # Verify dataset structure
    assert len(loaded_dataset.datasets) == len(dataset.datasets)

    # Verify data integrity
    for i, (original_ds, loaded_ds) in enumerate(
        zip(dataset.datasets, loaded_dataset.datasets)
    ):
        original_data = original_ds.raw.get_data()
        loaded_data = loaded_ds.raw.get_data()

        assert original_data.shape == loaded_data.shape
        np.testing.assert_allclose(original_data, loaded_data, rtol=1e-5)


def test_hdf5_overwrite(setup_concat_windows_dataset, tmpdir):
    """Test HDF5 overwrite functionality."""
    dataset = setup_concat_windows_dataset
    output_path = Path(tmpdir) / "test_overwrite.h5"

    # First save
    convert_to_hdf5(dataset, output_path)
    assert output_path.exists()

    # Try to save again without overwrite - should raise error
    with pytest.raises(FileExistsError):
        convert_to_hdf5(dataset, output_path, overwrite=False)

    # Save with overwrite - should succeed
    convert_to_hdf5(dataset, output_path, overwrite=True)
    assert output_path.exists()


def test_hdf5_compression_levels(setup_concat_windows_dataset, tmpdir):
    """Test different HDF5 compression levels."""
    dataset = setup_concat_windows_dataset

    # Test with no compression
    path_no_comp = Path(tmpdir) / "no_compression.h5"
    convert_to_hdf5(dataset, path_no_comp, compression=None)

    # Test with gzip compression
    path_gzip = Path(tmpdir) / "gzip_compression.h5"
    convert_to_hdf5(dataset, path_gzip, compression="gzip", compression_level=4)

    # Both should exist and be loadable
    assert path_no_comp.exists()
    assert path_gzip.exists()

    loaded_no_comp = load_from_hdf5(path_no_comp)
    loaded_gzip = load_from_hdf5(path_gzip)

    assert len(loaded_no_comp.datasets) == len(loaded_gzip.datasets)

    # Compressed file should typically be smaller (though not guaranteed for tiny datasets)
    size_no_comp = path_no_comp.stat().st_size
    size_gzip = path_gzip.stat().st_size
    # Just verify both are reasonable sizes
    assert size_no_comp > 0
    assert size_gzip > 0


def test_hdf5_partial_load(setup_concat_windows_dataset, tmpdir):
    """Test loading only specific datasets from HDF5."""
    dataset = setup_concat_windows_dataset
    output_path = Path(tmpdir) / "test_partial.h5"

    # Convert to HDF5
    convert_to_hdf5(dataset, output_path)

    # Load only first dataset
    loaded_dataset = load_from_hdf5(output_path, ids_to_load=[0])

    assert len(loaded_dataset.datasets) == 1


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


def test_zarr_round_trip_raw(setup_concat_raw_dataset, tmpdir):
    """Test Zarr conversion and loading preserves raw data."""
    pytest.importorskip("zarr")

    dataset = setup_concat_raw_dataset
    output_path = Path(tmpdir) / "test_dataset_raw.zarr"

    # Convert to Zarr
    convert_to_zarr(dataset, output_path)
    assert output_path.exists()

    # Load from Zarr
    loaded_dataset = load_from_zarr(output_path, preload=True)

    # Verify data integrity
    assert len(loaded_dataset.datasets) == len(dataset.datasets)

    for i, (original_ds, loaded_ds) in enumerate(
        zip(dataset.datasets, loaded_dataset.datasets)
    ):
        original_data = original_ds.raw.get_data()
        loaded_data = loaded_ds.raw.get_data()

        assert original_data.shape == loaded_data.shape
        np.testing.assert_allclose(original_data, loaded_data, rtol=1e-5)


def test_zarr_compression(setup_concat_windows_dataset, tmpdir):
    """Test Zarr compression options."""
    pytest.importorskip("zarr")

    dataset = setup_concat_windows_dataset

    # Test with blosc compression
    path_blosc = Path(tmpdir) / "blosc.zarr"
    convert_to_zarr(dataset, path_blosc, compression="blosc")
    assert path_blosc.exists()

    # Test without compression
    path_no_comp = Path(tmpdir) / "no_compression.zarr"
    convert_to_zarr(dataset, path_no_comp, compression=None)
    assert path_no_comp.exists()

    # Both should be loadable
    loaded_blosc = load_from_zarr(path_blosc)
    loaded_no_comp = load_from_zarr(path_no_comp)

    assert len(loaded_blosc.datasets) == len(loaded_no_comp.datasets)


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


# =============================================================================
# NumPy + Parquet Format Tests
# =============================================================================


def test_npz_parquet_round_trip_windows(setup_concat_windows_dataset, tmpdir):
    """Test NumPy+Parquet conversion and loading preserves windowed data."""
    pytest.importorskip("pyarrow")

    dataset = setup_concat_windows_dataset
    output_path = Path(tmpdir) / "test_dataset_npz"

    # Convert to npz+parquet
    result_path = convert_to_npz_parquet(dataset, output_path, overwrite=False)
    assert result_path.exists()
    assert result_path.is_dir()

    # Check expected files exist
    assert (output_path / "metadata.json").exists()
    assert (output_path / "descriptions.parquet").exists()
    assert (output_path / "windows_metadata.parquet").exists()

    # Load from npz+parquet
    loaded_dataset = load_from_npz_parquet(output_path, preload=True)

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


def test_npz_parquet_round_trip_raw(setup_concat_raw_dataset, tmpdir):
    """Test NumPy+Parquet conversion and loading preserves raw data."""
    pytest.importorskip("pyarrow")

    dataset = setup_concat_raw_dataset
    output_path = Path(tmpdir) / "test_dataset_raw_npz"

    # Convert to npz+parquet
    convert_to_npz_parquet(dataset, output_path)
    assert output_path.exists()

    # Load from npz+parquet
    loaded_dataset = load_from_npz_parquet(output_path, preload=True)

    # Verify data integrity
    assert len(loaded_dataset.datasets) == len(dataset.datasets)

    for i, (original_ds, loaded_ds) in enumerate(
        zip(dataset.datasets, loaded_dataset.datasets)
    ):
        original_data = original_ds.raw.get_data()
        loaded_data = loaded_ds.raw.get_data()

        assert original_data.shape == loaded_data.shape
        np.testing.assert_allclose(original_data, loaded_data, rtol=1e-5)


def test_npz_parquet_compression(setup_concat_windows_dataset, tmpdir):
    """Test NumPy+Parquet compression options."""
    pytest.importorskip("pyarrow")

    dataset = setup_concat_windows_dataset

    # Test with zstd compression
    path_zstd = Path(tmpdir) / "zstd_npz"
    convert_to_npz_parquet(dataset, path_zstd, compression="zstd")
    assert path_zstd.exists()

    # Test without compression
    path_no_comp = Path(tmpdir) / "no_compression_npz"
    convert_to_npz_parquet(dataset, path_no_comp, compression=None)
    assert path_no_comp.exists()

    # Both should be loadable
    loaded_zstd = load_from_npz_parquet(path_zstd)
    loaded_no_comp = load_from_npz_parquet(path_no_comp)

    assert len(loaded_zstd.datasets) == len(loaded_no_comp.datasets)


def test_npz_parquet_overwrite(setup_concat_windows_dataset, tmpdir):
    """Test NumPy+Parquet overwrite functionality."""
    pytest.importorskip("pyarrow")

    dataset = setup_concat_windows_dataset
    output_path = Path(tmpdir) / "test_overwrite_npz"

    # First save
    convert_to_npz_parquet(dataset, output_path)

    # Try to save again without overwrite - should raise error
    with pytest.raises(FileExistsError):
        convert_to_npz_parquet(dataset, output_path, overwrite=False)

    # Save with overwrite - should succeed
    convert_to_npz_parquet(dataset, output_path, overwrite=True)


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
    assert "is_windowed" in info
    assert "recommended_format" in info
    assert "reason" in info

    assert info["n_recordings"] == len(dataset.datasets)
    assert info["is_windowed"] is True
    assert info["recommended_format"] in ["hdf5", "zarr", "npz_parquet"]


def test_get_format_info_raw(setup_concat_raw_dataset):
    """Test get_format_info returns correct information for raw dataset."""
    dataset = setup_concat_raw_dataset
    info = get_format_info(dataset)

    assert info["n_recordings"] == len(dataset.datasets)
    assert info["is_windowed"] is False
    assert info["total_size_mb"] > 0


# =============================================================================
# Cross-Format Compatibility Tests
# =============================================================================


def test_all_formats_produce_same_data(setup_concat_windows_dataset, tmpdir):
    """Test that all formats preserve the same data."""
    pytest.importorskip("zarr")
    pytest.importorskip("pyarrow")

    dataset = setup_concat_windows_dataset

    # Convert to all formats
    hdf5_path = Path(tmpdir) / "test.h5"
    zarr_path = Path(tmpdir) / "test.zarr"
    npz_path = Path(tmpdir) / "test_npz"

    convert_to_hdf5(dataset, hdf5_path)
    convert_to_zarr(dataset, zarr_path)
    convert_to_npz_parquet(dataset, npz_path)

    # Load from all formats
    loaded_hdf5 = load_from_hdf5(hdf5_path)
    loaded_zarr = load_from_zarr(zarr_path)
    loaded_npz = load_from_npz_parquet(npz_path)

    # Verify all produce the same data
    for i in range(len(dataset.datasets)):
        original_data = dataset.datasets[i].windows.get_data()
        hdf5_data = loaded_hdf5.datasets[i].windows.get_data()
        zarr_data = loaded_zarr.datasets[i].windows.get_data()
        npz_data = loaded_npz.datasets[i].windows.get_data()

        np.testing.assert_allclose(original_data, hdf5_data, rtol=1e-5)
        np.testing.assert_allclose(original_data, zarr_data, rtol=1e-5)
        np.testing.assert_allclose(original_data, npz_data, rtol=1e-5)


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_load_nonexistent_file():
    """Test that loading nonexistent file raises appropriate error."""
    with pytest.raises(FileNotFoundError):
        load_from_hdf5("/nonexistent/path.h5")

    pytest.importorskip("zarr")
    with pytest.raises(FileNotFoundError):
        load_from_zarr("/nonexistent/path.zarr")

    pytest.importorskip("pyarrow")
    with pytest.raises(FileNotFoundError):
        load_from_npz_parquet("/nonexistent/path")


def test_zarr_import_error(setup_concat_windows_dataset, tmpdir, monkeypatch):
    """Test that appropriate error is raised when zarr is not installed."""
    # Mock zarr as unavailable
    monkeypatch.setattr("braindecode.datautil.hub_formats.ZARR_AVAILABLE", False)

    output_path = Path(tmpdir) / "test.zarr"

    with pytest.raises(ImportError, match="Zarr is not installed"):
        convert_to_zarr(setup_concat_windows_dataset, output_path)

    with pytest.raises(ImportError, match="Zarr is not installed"):
        load_from_zarr(output_path)


def test_pyarrow_import_error(setup_concat_windows_dataset, tmpdir, monkeypatch):
    """Test that appropriate error is raised when pyarrow is not installed."""
    # Mock pyarrow as unavailable
    monkeypatch.setattr("braindecode.datautil.hub_formats.PYARROW_AVAILABLE", False)

    output_path = Path(tmpdir) / "test_npz"

    with pytest.raises(ImportError, match="PyArrow is not installed"):
        convert_to_npz_parquet(setup_concat_windows_dataset, output_path)

    with pytest.raises(ImportError, match="PyArrow is not installed"):
        load_from_npz_parquet(output_path)
