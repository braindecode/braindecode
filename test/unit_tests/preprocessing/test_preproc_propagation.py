"""Test to verify that preprocessing kwargs are properly propagated during windowing."""
import mne
import numpy as np
import pytest

from braindecode.datasets import BaseConcatDataset, RawDataset
from braindecode.preprocessing import Preprocessor, preprocess
from braindecode.preprocessing.windowers import (
    create_fixed_length_windows,
    create_windows_from_events,
)


@pytest.fixture
def simple_raw():
    """Create a simple MNE Raw object for testing."""
    sfreq = 100  # Hz
    duration = 10  # seconds
    n_channels = 3

    # Create some dummy data
    data = np.random.randn(n_channels, duration * sfreq)
    info = mne.create_info(ch_names=["C3", "Cz", "C4"], sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info)

    # Add some annotations for windowing
    onsets = [1, 3, 5, 7]
    durations = [1, 1, 1, 1]
    descriptions = ["event1", "event2", "event1", "event2"]
    annotations = mne.Annotations(onsets, durations, descriptions)
    raw.set_annotations(annotations)

    return raw


def test_raw_preproc_kwargs_propagation_eeg_windows(simple_raw):
    """Test that raw_preproc_kwargs is propagated to EEGWindowsDataset and includes windowing."""
    # Create dataset
    ds = RawDataset(simple_raw, description={"subject": 1})
    concat_ds = BaseConcatDataset([ds])

    # Apply preprocessing
    preprocessors = [
        Preprocessor("filter", l_freq=1.0, h_freq=40.0),
    ]
    preprocess(concat_ds, preprocessors)

    # Verify preprocessing was recorded
    assert len(ds.raw_preproc_kwargs) == 1
    assert ds.raw_preproc_kwargs[0]["fn"] == "filter"

    # Create windows (use_mne_epochs=False creates EEGWindowsDataset)
    windows_concat_ds = create_windows_from_events(
        concat_ds,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=50,
        window_stride_samples=25,
        drop_last_window=False,
        use_mne_epochs=False,
    )

    # Verify raw_preproc_kwargs was propagated AND windowing step was added
    windows_ds = windows_concat_ds.datasets[0]
    assert hasattr(windows_ds, "raw_preproc_kwargs")
    assert len(windows_ds.raw_preproc_kwargs) == 2  # filter + windowing
    assert windows_ds.raw_preproc_kwargs[0]["fn"] == "filter"
    assert windows_ds.raw_preproc_kwargs[1]["fn"] == "create_windows_from_events"
    assert "__class_path__" in windows_ds.raw_preproc_kwargs[1]
    assert "kwargs" in windows_ds.raw_preproc_kwargs[1]


def test_raw_preproc_kwargs_propagation_windows_dataset(simple_raw):
    """Test that raw_preproc_kwargs is propagated to WindowsDataset (mne.Epochs) and includes windowing."""
    # Create dataset
    ds = RawDataset(simple_raw, description={"subject": 1})
    concat_ds = BaseConcatDataset([ds])

    # Apply preprocessing
    preprocessors = [
        Preprocessor("filter", l_freq=1.0, h_freq=40.0),
    ]
    preprocess(concat_ds, preprocessors)

    # Verify preprocessing was recorded
    assert len(ds.raw_preproc_kwargs) == 1
    assert ds.raw_preproc_kwargs[0]["fn"] == "filter"

    # Create windows (use_mne_epochs=True creates WindowsDataset with mne.Epochs)
    windows_concat_ds = create_windows_from_events(
        concat_ds,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=50,
        window_stride_samples=25,
        drop_last_window=False,
        use_mne_epochs=True,
    )

    # Verify raw_preproc_kwargs was propagated AND windowing step was added
    windows_ds = windows_concat_ds.datasets[0]
    assert hasattr(windows_ds, "raw_preproc_kwargs")
    assert len(windows_ds.raw_preproc_kwargs) == 2  # filter + windowing
    assert windows_ds.raw_preproc_kwargs[0]["fn"] == "filter"
    assert windows_ds.raw_preproc_kwargs[1]["fn"] == "create_windows_from_events"
    assert "__class_path__" in windows_ds.raw_preproc_kwargs[1]
    assert "kwargs" in windows_ds.raw_preproc_kwargs[1]


def test_raw_preproc_kwargs_propagation_fixed_length(simple_raw):
    """Test that raw_preproc_kwargs is propagated with create_fixed_length_windows and includes windowing."""
    # Create dataset
    ds = RawDataset(simple_raw, description={"subject": 1}, target_name="subject")
    concat_ds = BaseConcatDataset([ds])

    # Apply preprocessing
    preprocessors = [
        Preprocessor("filter", l_freq=1.0, h_freq=40.0),
    ]
    preprocess(concat_ds, preprocessors)

    # Verify preprocessing was recorded
    assert len(ds.raw_preproc_kwargs) == 1

    # Create windows
    windows_concat_ds = create_fixed_length_windows(
        concat_ds,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=100,
        window_stride_samples=100,
        drop_last_window=False,
    )

    # Verify raw_preproc_kwargs was propagated AND windowing step was added
    windows_ds = windows_concat_ds.datasets[0]
    assert hasattr(windows_ds, "raw_preproc_kwargs")
    assert len(windows_ds.raw_preproc_kwargs) == 2  # filter + windowing
    assert windows_ds.raw_preproc_kwargs[0]["fn"] == "filter"
    assert windows_ds.raw_preproc_kwargs[1]["fn"] == "create_fixed_length_windows"
    assert "__class_path__" in windows_ds.raw_preproc_kwargs[1]
    assert "kwargs" in windows_ds.raw_preproc_kwargs[1]


def test_preprocessing_before_and_after_windowing(simple_raw):
    """Test preprocessing on raw, then windowing, then preprocessing on windows.

    This test verifies that:
    1. Preprocessing applied to RawDataset is stored in raw_preproc_kwargs
    2. Windowing operation is added to raw_preproc_kwargs
    3. Additional preprocessing on WindowsDataset is stored in window_preproc_kwargs
    4. The raw_preproc_kwargs (including windowing) remains intact after window preprocessing
    """
    # Create dataset
    ds = RawDataset(simple_raw, description={"subject": 1})
    concat_ds = BaseConcatDataset([ds])

    # Step 1: Apply preprocessing on raw data
    preprocessors_raw = [
        Preprocessor("filter", l_freq=1.0, h_freq=40.0),
    ]
    preprocess(concat_ds, preprocessors_raw)

    # Verify preprocessing was recorded
    assert len(ds.raw_preproc_kwargs) == 1
    assert ds.raw_preproc_kwargs[0]["fn"] == "filter"

    # Step 2: Create windows
    windows_concat_ds = create_windows_from_events(
        concat_ds,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=50,
        window_stride_samples=25,
        drop_last_window=False,
        use_mne_epochs=True,  # Use mne.Epochs (WindowsDataset)
    )

    # Verify propagation and windowing step addition
    windows_ds = windows_concat_ds.datasets[0]
    assert hasattr(windows_ds, "raw_preproc_kwargs")
    assert len(windows_ds.raw_preproc_kwargs) == 2  # filter + windowing
    assert windows_ds.raw_preproc_kwargs[0]["fn"] == "filter"
    assert windows_ds.raw_preproc_kwargs[1]["fn"] == "create_windows_from_events"
    assert hasattr(windows_ds, "window_preproc_kwargs")
    assert len(windows_ds.window_preproc_kwargs) == 0

    # Step 3: Apply preprocessing on windows
    preprocessors_windows = [
        Preprocessor("crop", tmin=0, tmax=0.4),
    ]
    preprocess(windows_concat_ds, preprocessors_windows)

    # Verify both kwargs are preserved
    assert hasattr(windows_ds, "raw_preproc_kwargs")
    assert len(windows_ds.raw_preproc_kwargs) == 2  # filter + windowing
    assert windows_ds.raw_preproc_kwargs[0]["fn"] == "filter"
    assert windows_ds.raw_preproc_kwargs[1]["fn"] == "create_windows_from_events"

    assert hasattr(windows_ds, "window_preproc_kwargs")
    assert len(windows_ds.window_preproc_kwargs) == 1
    assert windows_ds.window_preproc_kwargs[0]["fn"] == "crop"


def test_window_kwargs_contains_windowing_parameters(simple_raw):
    """Test that window_kwargs contains all windowing parameters."""
    # Create dataset
    ds = RawDataset(simple_raw, description={"subject": 1})
    concat_ds = BaseConcatDataset([ds])

    # Apply preprocessing on raw data
    preprocessors_raw = [
        Preprocessor("filter", l_freq=1.0, h_freq=40.0),
    ]
    preprocess(concat_ds, preprocessors_raw)

    # Create windows
    windows_concat_ds = create_windows_from_events(
        concat_ds,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=50,
        window_stride_samples=25,
        drop_last_window=False,
        use_mne_epochs=True,
    )

    windows_ds = windows_concat_ds.datasets[0]

    # Verify window_kwargs exists and contains windowing parameters
    assert hasattr(windows_ds, "window_kwargs")
    assert isinstance(windows_ds.window_kwargs, list)
    assert len(windows_ds.window_kwargs) > 0

    # Verify it contains the function name and parameters
    func_name, params = windows_ds.window_kwargs[0]
    assert func_name == "create_windows_from_events"
    assert "window_size_samples" in params
    assert params["window_size_samples"] == 50
    assert "window_stride_samples" in params
    assert params["window_stride_samples"] == 25
