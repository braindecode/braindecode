# Authors: Kuntal Kokate
#
# License: BSD-3

"""Simple tests for Hugging Face Hub integration."""


import inspect
import warnings
from unittest import mock

import mne
import numpy as np
import pandas as pd
import pytest
import scipy
import zarr
from numcodecs import Blosc, GZip, Zstd

from braindecode.datasets import (
    BNCI2014_001,
    BaseConcatDataset,
    RawDataset,
)
from braindecode.datasets.hub import _create_compressor
from braindecode.datasets.registry import get_dataset_class, get_dataset_type
from braindecode.preprocessing import create_windows_from_events


@pytest.fixture()
def setup_concat_windows_dataset():
    """Create a small windowed dataset for testing.

    Uses BNCI2014_001 dataset with subject_ids=[1, 2] which downloads
    real EEG data for two subjects.
    """
    # Download data for subjects 1 and 2
    dataset = BNCI2014_001(subject_ids=[1, 2])

    # Use first 2 recordings to keep tests fast while testing multiple subjects
    windowed = create_windows_from_events(
        concat_ds=BaseConcatDataset(dataset.datasets[:2]),
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        use_mne_epochs=True,
    )
    return windowed


def test_hub_mixin_methods_exist(setup_concat_windows_dataset):
    """Test that Hub mixin methods are available on BaseConcatDataset."""
    dataset = setup_concat_windows_dataset

    # Check that methods exist
    assert hasattr(dataset, "push_to_hub")
    assert hasattr(dataset, "from_pretrained")
    assert callable(dataset.push_to_hub)
    assert callable(dataset.from_pretrained)


def test_dataset_card_generation(setup_concat_windows_dataset, tmp_path):
    """Test that dataset card (README) is generated correctly."""
    pytest.importorskip("huggingface_hub")

    dataset = setup_concat_windows_dataset

    # Call the internal method to generate dataset card
    dataset._save_dataset_card(tmp_path)

    # Check that README.md was created
    readme_path = tmp_path / "README.md"
    assert readme_path.exists()

    # Check content
    content = readme_path.read_text()
    assert "braindecode" in content.lower()
    assert "zarr" in content.lower()
    assert str(len(dataset.datasets)) in content


def test_zarr_save_and_load(setup_concat_windows_dataset, tmp_path):
    """Test basic Zarr save and load functionality."""
    pytest.importorskip("zarr")

    dataset = setup_concat_windows_dataset
    zarr_path = tmp_path / "dataset.zarr"

    # Test via internal methods (avoiding HuggingFace Hub)
    dataset._convert_to_zarr_inline(
        zarr_path,
        compression="blosc",
        compression_level=5,
    )

    assert zarr_path.exists()

    # Load it back
    loaded = dataset._load_from_zarr_inline(zarr_path, preload=True)

    assert isinstance(loaded, BaseConcatDataset)
    assert len(loaded.datasets) == len(dataset.datasets)


def test_no_lazy_imports_in_hub_module():
    """Verify that hub module uses registry pattern instead of lazy imports.

    Tests with mocked zarr and huggingface_hub to ensure modules can be imported
    even when optional dependencies are not available.
    """
    # Mock zarr and huggingface_hub to simulate missing dependencies
    with mock.patch.dict('sys.modules', {
        'zarr': mock.MagicMock(),
        'huggingface_hub': mock.MagicMock(),
    }):
        # Import hub module (should work even with mocked dependencies)
        from braindecode.datasets import hub

        # Get all functions in the hub module
        functions = [
            obj for name, obj in inspect.getmembers(hub)
            if inspect.isfunction(obj) and obj.__module__ == hub.__name__
        ]

        # Check that no functions have 'from .base import' (which would be circular)
        for func in functions:
            source = inspect.getsource(func)
            # No function should have 'from .base import' or 'from ..datasets.base import'
            assert "from .base import" not in source, f"{func.__name__} has circular import"
            assert "from ..datasets.base import" not in source, f"{func.__name__} has circular import"


def test_registry_pattern_works():
    """Test that registry pattern allows access to dataset classes without circular imports."""
    # Get classes from registry
    WindowsDataset = get_dataset_class("WindowsDataset")
    EEGWindowsDataset = get_dataset_class("EEGWindowsDataset")
    BaseConcatDataset = get_dataset_class("BaseConcatDataset")

    # Verify classes are actual types
    assert isinstance(WindowsDataset, type)
    assert isinstance(EEGWindowsDataset, type)
    assert isinstance(BaseConcatDataset, type)

    # Test get_dataset_type with actual dataset
    dataset = BNCI2014_001(subject_ids=[1])
    windowed = create_windows_from_events(
        concat_ds=BaseConcatDataset([dataset.datasets[0]]),
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        use_mne_epochs=True,
    )

    # Check type detection works
    assert get_dataset_type(windowed.datasets[0]) == "WindowsDataset"
    assert get_dataset_type(windowed) == "BaseConcatDataset"


def test_eegwindows_lossless_round_trip(tmp_path):
    """Test that EEGWindowsDataset has lossless round-trip with continuous data preserved."""
    pytest.importorskip("zarr")

    # Create EEGWindowsDataset with continuous raw (use_mne_epochs=False)
    dataset = BNCI2014_001(subject_ids=[1])
    windowed = create_windows_from_events(
        concat_ds=BaseConcatDataset([dataset.datasets[0]]),
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        use_mne_epochs=False,  # Creates EEGWindowsDataset
    )

    # Store original continuous raw data
    original_raw_data = windowed.datasets[0].raw.get_data()
    original_metadata = windowed.datasets[0].metadata.copy()


    # Save to Zarr
    zarr_path = tmp_path / "dataset.zarr"
    windowed._convert_to_zarr_inline(
        zarr_path,
        compression="blosc",
        compression_level=5,
    )

    # Load back
    loaded = windowed._load_from_zarr_inline(zarr_path, preload=True)

    # Verify it's an EEGWindowsDataset
    assert get_dataset_type(loaded.datasets[0]) == "EEGWindowsDataset"

    # Verify continuous raw data is preserved (allowing for float32 precision)
    loaded_raw_data = loaded.datasets[0].raw.get_data()
    assert loaded_raw_data.shape == original_raw_data.shape, \
        f"Shape mismatch: {loaded_raw_data.shape} vs {original_raw_data.shape}"
    # Use allclose since we save as float32 (some precision loss expected)
    np.testing.assert_allclose(
        original_raw_data,
        loaded_raw_data,
        rtol=1e-6,  # Relative tolerance
        atol=1e-7,  # Absolute tolerance
        err_msg="Continuous raw data not preserved within float32 precision!"
    )

    # Verify metadata is preserved
    pd.testing.assert_frame_equal(original_metadata, loaded.datasets[0].metadata)

    # Verify windows can still be extracted correctly
    for i in range(min(5, len(windowed.datasets[0]))):  # Test first 5 windows
        orig_X, orig_y, orig_inds = windowed.datasets[0][i]
        load_X, load_y, load_inds = loaded.datasets[0][i]

        np.testing.assert_array_equal(orig_X, load_X, err_msg=f"Window {i} data mismatch")
        assert orig_y == load_y, f"Window {i} target mismatch"
        assert orig_inds == load_inds, f"Window {i} crop indices mismatch"


def test_rawdataset_basic_save_load(tmp_path):
    """Test basic RawDataset save and load functionality."""
    pytest.importorskip("zarr")

    # Create a simple RawDataset with synthetic data
    n_channels = 3
    n_times = 1000
    sfreq = 100
    ch_names = [f"ch{i}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    # Create synthetic data
    data = np.random.randn(n_channels, n_times).astype(np.float64)
    raw = mne.io.RawArray(data, info)

    # Create RawDataset
    description = pd.Series({"subject": "test_subject", "session": "1"})
    raw_dataset = RawDataset(raw, description)
    raw_dataset.target_name = "mock_target"

    # Create BaseConcatDataset
    concat_ds = BaseConcatDataset([raw_dataset])

    # Save to Zarr
    zarr_path = tmp_path / "rawdataset.zarr"
    concat_ds._convert_to_zarr_inline(
        zarr_path,
        compression="blosc",
        compression_level=5,
    )

    assert zarr_path.exists()

    # Load back
    loaded = concat_ds._load_from_zarr_inline(zarr_path, preload=True)

    # Verify dataset type
    assert get_dataset_type(loaded.datasets[0]) == "RawDataset"

    # Verify data preservation (within float32 precision)
    loaded_data = loaded.datasets[0].raw.get_data()
    np.testing.assert_allclose(
        data,
        loaded_data,
        rtol=1e-6,
        atol=1e-7,
        err_msg="RawDataset data not preserved within float32 precision!"
    )

    # Verify description
    pd.testing.assert_series_equal(description, loaded.datasets[0].description)

    # Verify target_name
    assert loaded.datasets[0].target_name == "mock_target"

    # Verify MNE info properties
    assert loaded.datasets[0].raw.info["sfreq"] == sfreq
    assert loaded.datasets[0].raw.ch_names == ch_names


def test_rawdataset_lossless_round_trip(tmp_path):
    """Test that RawDataset has lossless round-trip with continuous data preserved."""
    pytest.importorskip("zarr")

    # Use BNCI2014_001 to get real RawDataset
    dataset = BNCI2014_001(subject_ids=[1])

    # Get the first recording (which is a BaseDataset wrapping a RawDataset internally)
    # We need to create a RawDataset from it
    first_recording = dataset.datasets[0]
    raw_dataset = RawDataset(first_recording.raw, first_recording.description)

    # Create BaseConcatDataset with RawDataset
    concat_ds = BaseConcatDataset([raw_dataset])

    # Store original data
    original_raw_data = raw_dataset.raw.get_data()
    original_description = raw_dataset.description.copy()

    # Save to Zarr
    zarr_path = tmp_path / "rawdataset.zarr"
    concat_ds._convert_to_zarr_inline(
        zarr_path,
        compression="blosc",
        compression_level=5,
    )

    # Load back
    loaded = concat_ds._load_from_zarr_inline(zarr_path, preload=True)

    # Verify it's a RawDataset
    assert get_dataset_type(loaded.datasets[0]) == "RawDataset"

    # Verify continuous raw data is preserved (allowing for float32 precision)
    loaded_raw_data = loaded.datasets[0].raw.get_data()
    assert loaded_raw_data.shape == original_raw_data.shape, \
        f"Shape mismatch: {loaded_raw_data.shape} vs {original_raw_data.shape}"

    np.testing.assert_allclose(
        original_raw_data,
        loaded_raw_data,
        rtol=1e-6,
        atol=1e-7,
        err_msg="Continuous raw data not preserved within float32 precision!"
    )

    # Verify description is preserved (ignore series name attribute)
    pd.testing.assert_series_equal(
        original_description, loaded.datasets[0].description, check_names=False
    )


def test_rawdataset_mixed_concat(tmp_path):
    """Test that mixed RawDataset concat works correctly."""
    pytest.importorskip("zarr")

    # Use BNCI2014_001 with 2 subjects
    dataset = BNCI2014_001(subject_ids=[1, 2])

    # Create RawDatasets from first 2 recordings
    raw_ds1 = RawDataset(dataset.datasets[0].raw, dataset.datasets[0].description)
    raw_ds2 = RawDataset(dataset.datasets[1].raw, dataset.datasets[1].description)

    # Create BaseConcatDataset
    concat_ds = BaseConcatDataset([raw_ds1, raw_ds2])

    # Save to Zarr
    zarr_path = tmp_path / "mixed_rawdataset.zarr"
    concat_ds._convert_to_zarr_inline(
        zarr_path,
        compression="blosc",
        compression_level=5,
    )

    # Load back
    loaded = concat_ds._load_from_zarr_inline(zarr_path, preload=True)

    # Verify we have 2 datasets
    assert len(loaded.datasets) == 2

    # Verify both are RawDatasets
    assert get_dataset_type(loaded.datasets[0]) == "RawDataset"
    assert get_dataset_type(loaded.datasets[1]) == "RawDataset"

    # Verify data shapes match
    assert loaded.datasets[0].raw.get_data().shape == raw_ds1.raw.get_data().shape
    assert loaded.datasets[1].raw.get_data().shape == raw_ds2.raw.get_data().shape


def test_rawdataset_dataset_card(tmp_path):
    """Test that dataset card (README) is generated correctly for RawDataset."""
    pytest.importorskip("huggingface_hub")

    # Create a simple RawDataset
    n_channels = 3
    n_times = 1000
    sfreq = 100
    ch_names = [f"ch{i}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    data = np.random.randn(n_channels, n_times)
    raw = mne.io.RawArray(data, info)
    description = pd.Series({"subject": "test"})
    raw_dataset = RawDataset(raw, description)

    # Create BaseConcatDataset
    concat_ds = BaseConcatDataset([raw_dataset])

    # Generate dataset card
    concat_ds._save_dataset_card(tmp_path)

    # Check that README.md was created
    readme_path = tmp_path / "README.md"
    assert readme_path.exists()

    # Check content
    content = readme_path.read_text()
    assert "braindecode" in content.lower()
    assert "Continuous (Raw)" in content  # Data type should be Continuous
    assert str(n_channels) in content
    assert str(sfreq) in content


def test_rawdataset_format_info():
    """Test that format info is correctly computed for RawDataset."""
    # Create two simple RawDatasets
    n_channels = 3
    n_times_1 = 1000
    n_times_2 = 1500
    sfreq = 100
    ch_names = [f"ch{i}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    # Dataset 1
    data1 = np.random.randn(n_channels, n_times_1)
    raw1 = mne.io.RawArray(data1, info)
    description1 = pd.Series({"subject": "1"})
    raw_ds1 = RawDataset(raw1, description1)

    # Dataset 2
    data2 = np.random.randn(n_channels, n_times_2)
    raw2 = mne.io.RawArray(data2, info)
    description2 = pd.Series({"subject": "2"})
    raw_ds2 = RawDataset(raw2, description2)

    # Create BaseConcatDataset
    concat_ds = BaseConcatDataset([raw_ds1, raw_ds2])

    # Get format info
    format_info = concat_ds._get_format_info_inline()

    # Verify
    assert format_info["n_recordings"] == 2
    assert format_info["total_samples"] == n_times_1 + n_times_2  # Total timepoints
    assert format_info["total_size_mb"] > 0


def test_mixed_dataset_types_not_supported(tmp_path):
    """Test that mixing different dataset types raises clear error."""
    pytest.importorskip("zarr")

    # Create a WindowsDataset
    dataset = BNCI2014_001(subject_ids=[1])
    windowed = create_windows_from_events(
        concat_ds=BaseConcatDataset([dataset.datasets[0]]),
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        use_mne_epochs=True,  # Creates WindowsDataset
    )
    windows_ds = windowed.datasets[0]

    # Create a RawDataset from same source
    raw_ds = RawDataset(dataset.datasets[0].raw, dataset.datasets[0].description)

    # Mix different types in BaseConcatDataset
    mixed_concat = BaseConcatDataset([windows_ds, raw_ds])

    # Try to save - should raise ValueError
    zarr_path = tmp_path / "mixed_types.zarr"
    with pytest.raises(ValueError) as exc_info:
        mixed_concat._convert_to_zarr_inline(
            zarr_path,
            compression="blosc",
            compression_level=5,
        )

    # Verify error message is clear and helpful
    error_msg = str(exc_info.value)
    assert "Mixed dataset types in concat" in error_msg
    assert "WindowsDataset" in error_msg
    assert "RawDataset" in error_msg


def test_inconsistent_channels_not_supported(tmp_path):
    """Test that datasets with different channels raise clear error."""
    pytest.importorskip("zarr")

    # Create RawDataset with 3 channels
    n_channels_1 = 3
    ch_names_1 = [f"ch{i}" for i in range(n_channels_1)]
    info_1 = mne.create_info(ch_names=ch_names_1, sfreq=100, ch_types="eeg")
    data_1 = np.random.randn(n_channels_1, 1000)
    raw_1 = mne.io.RawArray(data_1, info_1)
    raw_ds1 = RawDataset(raw_1, pd.Series({"subject": "1"}))

    # Create RawDataset with 5 channels (DIFFERENT!)
    n_channels_2 = 5
    ch_names_2 = [f"ch{i}" for i in range(n_channels_2)]
    info_2 = mne.create_info(ch_names=ch_names_2, sfreq=100, ch_types="eeg")
    data_2 = np.random.randn(n_channels_2, 1000)
    raw_2 = mne.io.RawArray(data_2, info_2)
    raw_ds2 = RawDataset(raw_2, pd.Series({"subject": "2"}))

    # Mix different channel counts
    mixed_concat = BaseConcatDataset([raw_ds1, raw_ds2])

    # Try to save - should raise ValueError
    zarr_path = tmp_path / "inconsistent_channels.zarr"
    with pytest.raises(ValueError) as exc_info:
        mixed_concat._convert_to_zarr_inline(
            zarr_path,
            compression="blosc",
            compression_level=5,
        )

    # Verify error message mentions inconsistent channels
    error_msg = str(exc_info.value)
    assert "Inconsistent channel names" in error_msg


def test_inconsistent_sfreq_not_supported(tmp_path):
    """Test that datasets with different sampling frequencies raise clear error."""
    pytest.importorskip("zarr")

    # Create RawDataset with 100 Hz
    ch_names = ["ch0", "ch1", "ch2"]
    info_1 = mne.create_info(ch_names=ch_names, sfreq=100, ch_types="eeg")
    data_1 = np.random.randn(3, 1000)
    raw_1 = mne.io.RawArray(data_1, info_1)
    raw_ds1 = RawDataset(raw_1, pd.Series({"subject": "1"}))

    # Create RawDataset with 250 Hz (DIFFERENT!)
    info_2 = mne.create_info(ch_names=ch_names, sfreq=250, ch_types="eeg")
    data_2 = np.random.randn(3, 2500)
    raw_2 = mne.io.RawArray(data_2, info_2)
    raw_ds2 = RawDataset(raw_2, pd.Series({"subject": "2"}))

    # Mix different sampling frequencies
    mixed_concat = BaseConcatDataset([raw_ds1, raw_ds2])

    # Try to save - should raise ValueError
    zarr_path = tmp_path / "inconsistent_sfreq.zarr"
    with pytest.raises(ValueError) as exc_info:
        mixed_concat._convert_to_zarr_inline(
            zarr_path,
            compression="blosc",
            compression_level=5,
        )

    # Verify error message mentions inconsistent sampling frequency with helpful guidance
    error_msg = str(exc_info.value)
    assert "Inconsistent sampling frequencies" in error_msg
    assert "100" in error_msg or "100.0" in error_msg
    assert "250" in error_msg or "250.0" in error_msg
    # Check for helpful resampling instructions
    assert "resample" in error_msg.lower()
    assert "preprocess" in error_msg
    assert "Preprocessor" in error_msg


def test_different_lengths_allowed(tmp_path):
    """Test that datasets with different lengths (but same channels/sfreq) are allowed."""
    pytest.importorskip("zarr")

    # Create RawDataset with 1000 timepoints
    ch_names = ["ch0", "ch1", "ch2"]
    sfreq = 100
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    data_1 = np.random.randn(3, 1000)
    raw_1 = mne.io.RawArray(data_1, info)
    raw_ds1 = RawDataset(raw_1, pd.Series({"subject": "1"}))

    # Create RawDataset with 2000 timepoints (DIFFERENT LENGTH - should be OK!)
    data_2 = np.random.randn(3, 2000)
    raw_2 = mne.io.RawArray(data_2, info)
    raw_ds2 = RawDataset(raw_2, pd.Series({"subject": "2"}))

    # Mix different lengths (this should work!)
    concat_ds = BaseConcatDataset([raw_ds1, raw_ds2])

    # Save should succeed
    zarr_path = tmp_path / "different_lengths.zarr"
    concat_ds._convert_to_zarr_inline(
        zarr_path,
        compression="blosc",
        compression_level=5,
    )

    # Verify it saved successfully
    assert zarr_path.exists()

    # Load back and verify
    loaded = concat_ds._load_from_zarr_inline(zarr_path, preload=True)
    assert len(loaded.datasets) == 2
    assert loaded.datasets[0].raw.get_data().shape[1] == 1000
    assert loaded.datasets[1].raw.get_data().shape[1] == 2000


def test_lazy_loading_support(tmp_path):
    """Test that datasets can be loaded with preload parameter.

    Note: Lazy loading (preload=False) is not fully implemented yet,
    so data is always loaded into memory with a warning.
    """
    pytest.importorskip("zarr")

    # Create and save a windowed dataset
    dataset = BNCI2014_001(subject_ids=[1])
    windowed = create_windows_from_events(
        concat_ds=BaseConcatDataset([dataset.datasets[0]]),
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        use_mne_epochs=True,
    )

    zarr_path = tmp_path / "lazy_test.zarr"
    windowed._convert_to_zarr_inline(zarr_path, compression="blosc", compression_level=5)

    # Load with preload=False (should warn and load anyway)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        loaded_lazy = windowed._load_from_zarr_inline(zarr_path, preload=False)
        # Check that warning about lazy loading was issued
        assert any("Lazy loading from Zarr not fully implemented" in str(warning.message)
                   for warning in w)

    # Currently always loads into memory
    assert loaded_lazy.datasets[0].windows.preload

    # Load with preload=True
    loaded_eager = windowed._load_from_zarr_inline(zarr_path, preload=True)
    assert loaded_eager.datasets[0].windows.preload

    # Both should have same number of windows
    assert len(loaded_lazy.datasets[0]) == len(loaded_eager.datasets[0])


def test_preprocessing_kwargs_preserved(tmp_path):
    """Test that preprocessing kwargs (window_kwargs, etc.) are preserved."""
    pytest.importorskip("zarr")

    # Create windowed dataset with specific kwargs
    dataset = BNCI2014_001(subject_ids=[1])
    windowed = create_windows_from_events(
        concat_ds=BaseConcatDataset([dataset.datasets[0]]),
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        use_mne_epochs=True,
    )

    # Manually set preprocessing kwargs (these are set by windowing functions)
    windowed.datasets[0].window_kwargs = {
        "trial_start_offset_samples": 0,
        "trial_stop_offset_samples": 0,
    }
    windowed.datasets[0].window_preproc_kwargs = [{"fn": "pick_types", "eeg": True}]

    # Save to Zarr
    zarr_path = tmp_path / "with_kwargs.zarr"
    windowed._convert_to_zarr_inline(zarr_path, compression="blosc", compression_level=5)

    # Load back
    loaded = windowed._load_from_zarr_inline(zarr_path, preload=True)

    # Verify kwargs are preserved
    assert hasattr(loaded.datasets[0], "window_kwargs")
    assert loaded.datasets[0].window_kwargs == windowed.datasets[0].window_kwargs
    assert hasattr(loaded.datasets[0], "window_preproc_kwargs")
    assert loaded.datasets[0].window_preproc_kwargs == windowed.datasets[0].window_preproc_kwargs


@pytest.mark.parametrize("use_mne_epochs", [True, False])
def test_zarr_round_trip_parametrized(tmp_path, use_mne_epochs):
    """Parametrized test for WindowsDataset and EEGWindowsDataset round-trip."""
    pytest.importorskip("zarr")

    # Create dataset with parametrized type
    dataset = BNCI2014_001(subject_ids=[1])
    windowed = create_windows_from_events(
        concat_ds=BaseConcatDataset([dataset.datasets[0]]),
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        use_mne_epochs=use_mne_epochs,
    )

    # Determine expected type
    expected_type = "WindowsDataset" if use_mne_epochs else "EEGWindowsDataset"

    # Save to Zarr
    zarr_path = tmp_path / f"{expected_type.lower()}_parametrized.zarr"
    windowed._convert_to_zarr_inline(zarr_path, compression="blosc", compression_level=5)

    # Load back
    loaded = windowed._load_from_zarr_inline(zarr_path, preload=True)

    # Verify type
    assert get_dataset_type(loaded.datasets[0]) == expected_type

    # Verify data
    assert len(loaded.datasets) == len(windowed.datasets)

    # Verify windows/raw preserved
    if use_mne_epochs:
        assert loaded.datasets[0].windows.info["sfreq"] == windowed.datasets[0].windows.info["sfreq"]
        # Check a few windows
        for i in range(min(3, len(loaded.datasets[0]))):
            orig_X, _, _ = windowed.datasets[0][i]
            load_X, _, _ = loaded.datasets[0][i]
            np.testing.assert_allclose(orig_X, load_X, rtol=1e-6, atol=1e-7)
    else:
        assert loaded.datasets[0].raw.info["sfreq"] == windowed.datasets[0].raw.info["sfreq"]
        # Verify continuous raw data preserved
        np.testing.assert_allclose(
            windowed.datasets[0].raw.get_data(),
            loaded.datasets[0].raw.get_data(),
            rtol=1e-6,
            atol=1e-7
        )


def test_compression_types(tmp_path):
    """Test different compression types: gzip, zstd, None."""
    pytest.importorskip("zarr")

    # Create a simple RawDataset
    ch_names = ["ch0", "ch1"]
    sfreq = 100
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    data = np.random.randn(2, 500)
    raw = mne.io.RawArray(data, info)
    raw_ds = RawDataset(raw, pd.Series({"subject": "1"}))
    concat_ds = BaseConcatDataset([raw_ds])

    # Test gzip compression
    zarr_path_gzip = tmp_path / "test_gzip.zarr"
    concat_ds._convert_to_zarr_inline(
        zarr_path_gzip,
        compression="gzip",
        compression_level=6,
    )
    assert zarr_path_gzip.exists()
    loaded_gzip = concat_ds._load_from_zarr_inline(zarr_path_gzip, preload=True)
    np.testing.assert_allclose(
        raw_ds.raw.get_data(),
        loaded_gzip.datasets[0].raw.get_data(),
        rtol=1e-6,
    )

    # Test zstd compression
    zarr_path_zstd = tmp_path / "test_zstd.zarr"
    concat_ds._convert_to_zarr_inline(
        zarr_path_zstd,
        compression="zstd",
        compression_level=3,
    )
    assert zarr_path_zstd.exists()
    loaded_zstd = concat_ds._load_from_zarr_inline(zarr_path_zstd, preload=True)
    np.testing.assert_allclose(
        raw_ds.raw.get_data(),
        loaded_zstd.datasets[0].raw.get_data(),
        rtol=1e-6,
    )

    # Test no compression (None)
    zarr_path_none = tmp_path / "test_none.zarr"
    concat_ds._convert_to_zarr_inline(
        zarr_path_none,
        compression=None,
        compression_level=5,
    )
    assert zarr_path_none.exists()
    loaded_none = concat_ds._load_from_zarr_inline(zarr_path_none, preload=True)
    np.testing.assert_allclose(
        raw_ds.raw.get_data(),
        loaded_none.datasets[0].raw.get_data(),
        rtol=1e-6,
    )


def test_dependency_version_metadata(tmp_path):
    """Test that dependency versions are saved in Zarr metadata."""
    pytest.importorskip("zarr")

    # Create a simple dataset
    ch_names = ["ch0"]
    sfreq = 100
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    data = np.random.randn(1, 100)
    raw = mne.io.RawArray(data, info)
    raw_ds = RawDataset(raw, pd.Series({"subject": "1"}))
    concat_ds = BaseConcatDataset([raw_ds])

    # Save to Zarr
    zarr_path = tmp_path / "test_versions.zarr"
    concat_ds._convert_to_zarr_inline(zarr_path, compression="blosc", compression_level=5)

    # Open Zarr and verify version metadata
    store = zarr.DirectoryStore(str(zarr_path))
    root = zarr.group(store=store)

    # Check that all dependency versions are saved
    assert "mne_version" in root.attrs
    assert "numpy_version" in root.attrs
    assert "pandas_version" in root.attrs
    assert "zarr_version" in root.attrs
    assert "scipy_version" in root.attrs

    # Verify versions match current environment
    assert root.attrs["mne_version"] == mne.__version__
    assert root.attrs["numpy_version"] == np.__version__
    assert root.attrs["pandas_version"] == pd.__version__
    assert root.attrs["zarr_version"] == zarr.__version__
    assert root.attrs["scipy_version"] == scipy.__version__


def test_load_nonexistent_file_error(tmp_path):
    """Test that loading from non-existent path raises FileNotFoundError."""
    pytest.importorskip("zarr")

    nonexistent_path = tmp_path / "does_not_exist.zarr"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        BaseConcatDataset._load_from_zarr_inline(nonexistent_path, preload=True)


def test_overwrite_existing_file_error(tmp_path):
    """Test that saving to existing path raises FileExistsError without overwrite."""
    pytest.importorskip("zarr")

    # Create a simple dataset
    ch_names = ["ch0"]
    sfreq = 100
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    data = np.random.randn(1, 100)
    raw = mne.io.RawArray(data, info)
    raw_ds = RawDataset(raw, pd.Series({"subject": "1"}))
    concat_ds = BaseConcatDataset([raw_ds])

    # Save once
    zarr_path = tmp_path / "test_overwrite.zarr"
    concat_ds._convert_to_zarr_inline(zarr_path, compression="blosc", compression_level=5)

    # Try to save again without overwrite - should fail
    with pytest.raises(FileExistsError, match="already exists"):
        concat_ds._convert_to_zarr_inline(zarr_path, compression="blosc", compression_level=5)


def test_create_compressor_function():
    """Test the _create_compressor function with different compression types."""
    pytest.importorskip("zarr")

    # Test blosc
    compressor = _create_compressor("blosc", 5)
    assert isinstance(compressor, Blosc)
    assert compressor.clevel == 5
    assert compressor.cname == "zstd"

    # Test gzip
    compressor = _create_compressor("gzip", 6)
    assert isinstance(compressor, GZip)
    assert compressor.level == 6

    # Test zstd
    compressor = _create_compressor("zstd", 3)
    assert isinstance(compressor, Zstd)
    assert compressor.level == 3

    # Test None compression
    compressor = _create_compressor(None, 5)
    assert compressor is None
