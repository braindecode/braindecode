# Authors: Christian Kothe <christian.kothe@intheon.io>
#
# License: BSD-3

import copy

import mne
import numpy as np
import pandas as pd
import pytest

from braindecode.datasets import BaseConcatDataset, BaseDataset
from braindecode.preprocessing import preprocess
from braindecode.preprocessing.eegprep_preprocess import (
    ReinterpolateRemovedChannels,
    RemoveBadChannelsNoLocs,
    RemoveBadWindows,
    RemoveCommonAverageReference,
    RemoveDCOffset,
    RemoveFlatChannels,
    Resample,
)

# Check if eegprep is available
try:
    import eegprep  # noqa: F401
    EEGPREP_AVAILABLE = True
except ImportError:
    EEGPREP_AVAILABLE = False


# Create module-level dataset with synthetic data
# We can't use fixtures with scope='module' as the dataset objects are modified
# inplace during preprocessing. To avoid the long setup time caused by calling
# the dataset/windowing functions multiple times, we instantiate the dataset
# objects once and deep-copy them in fixture.
def _create_synthetic_dataset(include_non_eeg=False, correlated_data=False,
                              duration=10, have_dc_offsets=True):
    """Create a synthetic dataset with DC offset for testing.

    Parameters
    ----------
    include_non_eeg : bool
        If True, includes additional non-EEG channels (misc type) to test
        that they are properly handled during preprocessing.
    correlated_data : bool
        If True, creates correlated channel data by multiplying with a random
        mixing matrix. This is useful for testing bad channel detection which
        relies on inter-channel correlations.
    duration : float
        Duration of the recording in seconds. Default is 10 seconds.
    have_dc_offsets : bool
        If True, adds DC offsets to each channel. Default is True. Set to False
        for tests that require zero-mean data (e.g., bad window detection).
    """
    rng = np.random.RandomState(42)

    # Create synthetic EEG data with standard 10-20 channel names for interpolation
    # Use a subset of standard 10-20 channels that have known locations
    eeg_ch_names = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T7', 'C3', 'Cz', 'C4', 'T8',
        'P7', 'P3', 'Pz', 'P4', 'P8',
        'O1', 'Oz', 'O2',
        'FC1', 'FC2'
    ]
    n_eeg_channels = len(eeg_ch_names)

    # Optionally add non-EEG channels
    if include_non_eeg:
        non_eeg_ch_names = ['EOG_L', 'EOG_R', 'EMG_1']
        ch_names = eeg_ch_names + non_eeg_ch_names
        ch_types = ['eeg'] * n_eeg_channels + ['misc'] * len(non_eeg_ch_names)
        n_channels = len(ch_names)
    else:
        ch_names = eeg_ch_names
        ch_types = ['eeg'] * n_eeg_channels
        n_channels = n_eeg_channels

    sfreq = 250  # Hz
    n_samples = int(sfreq * duration)

    # Generate random data
    data = rng.randn(n_channels, n_samples) * 1e-5  # Scale to realistic EEG amplitudes

    # Optionally create correlated data by mixing channels
    if correlated_data:
        # Create a random mixing matrix to generate correlated channels
        # This simulates realistic EEG where channels are spatially correlated
        # Add positive offset to increase baseline correlations between channels
        mixing_matrix = rng.randn(n_eeg_channels, n_eeg_channels) + 2.0
        # Mix only the EEG channels (not non-EEG if present)
        data[:n_eeg_channels, :] = mixing_matrix @ data[:n_eeg_channels, :]

    # Optionally add DC offset to each channel (different offsets per channel)
    if have_dc_offsets:
        dc_offsets = rng.uniform(-100, 100, size=(n_channels, 1))
        data += dc_offsets

    # Create MNE info and Raw object
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=ch_types
    )
    raw = mne.io.RawArray(data, info)

    # Set standard 10-20 montage for channel locations (needed for interpolation)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')

    # Create BaseDataset with minimal metadata
    desc = pd.Series({'subject': 1, 'session': 1})
    base_dataset = BaseDataset(raw, desc, target_name=None)

    # Wrap in BaseConcatDataset
    concat_ds = BaseConcatDataset([base_dataset])

    return concat_ds


raw_ds = _create_synthetic_dataset()


@pytest.fixture
def base_concat_ds():
    """Fixture returning a deepcopy of the synthetic dataset."""
    return copy.deepcopy(raw_ds)


@pytest.mark.skipif(not EEGPREP_AVAILABLE, reason="eegprep not installed")
def test_remove_dc_offset(base_concat_ds):
    """Test that RemoveDCOffset removes DC offset from EEG channels."""
    # Apply RemoveDCOffset preprocessor
    preprocessors = [RemoveDCOffset()]
    preprocess(base_concat_ds, preprocessors)

    # Get the processed data
    processed_raw = base_concat_ds.datasets[0].raw
    processed_data = processed_raw.get_data()

    # Compute median along time axis (axis=1) for each channel
    medians = np.median(processed_data, axis=1)

    # Assert that medians are very close to zero
    np.testing.assert_allclose(medians, 0, atol=1e-4)


@pytest.mark.skipif(not EEGPREP_AVAILABLE, reason="eegprep not installed")
def test_non_eeg_channels_preserved():
    """Test that non-EEG channels are separated and reintroduced correctly."""
    # Create dataset with non-EEG channels
    ds_with_non_eeg = _create_synthetic_dataset(include_non_eeg=True)

    # Get original data to check DC offsets before processing
    original_raw = ds_with_non_eeg.datasets[0].raw
    eeg_picks = mne.pick_types(original_raw.info, eeg=True, misc=False, exclude=[])
    non_eeg_picks = mne.pick_types(original_raw.info, eeg=False, misc=True, exclude=[])

    original_data = original_raw.get_data()
    original_eeg_medians = np.median(original_data[eeg_picks, :], axis=1)
    original_non_eeg_medians = np.median(original_data[non_eeg_picks, :], axis=1)

    # Verify that both EEG and non-EEG channels have DC offsets initially
    assert len(non_eeg_picks) > 0  # Verify we actually have non-EEG channels
    assert np.max(np.abs(original_eeg_medians)) > 10  # Should have large DC offset
    assert np.max(np.abs(original_non_eeg_medians)) > 10  # Should have large DC offset

    # Apply RemoveDCOffset preprocessor (should only affect EEG channels)
    preprocessors = [RemoveDCOffset()]
    preprocess(ds_with_non_eeg, preprocessors)

    # Get processed data
    processed_raw = ds_with_non_eeg.datasets[0].raw
    processed_data = processed_raw.get_data()

    # Get EEG and non-EEG channel picks for processed data
    processed_eeg_picks = mne.pick_types(processed_raw.info, eeg=True, misc=False, exclude=[])
    processed_non_eeg_picks = mne.pick_types(processed_raw.info, eeg=False, misc=True, exclude=[])

    # Verify that all channels are still present
    assert len(processed_eeg_picks) == len(eeg_picks)
    assert len(processed_non_eeg_picks) == len(non_eeg_picks)

    # Compute medians for EEG and non-EEG channels
    eeg_medians = np.median(processed_data[processed_eeg_picks, :], axis=1)
    non_eeg_medians = np.median(processed_data[processed_non_eeg_picks, :], axis=1)

    # Assert that EEG channels have DC offset removed (median ~0)
    np.testing.assert_allclose(eeg_medians, 0, atol=1e-4)

    # Assert that non-EEG channels still have their DC offset (median NOT ~0)
    # They should be essentially unchanged from the original
    np.testing.assert_allclose(non_eeg_medians, original_non_eeg_medians, rtol=1e-3)


@pytest.mark.skipif(not EEGPREP_AVAILABLE, reason="eegprep not installed")
def test_remove_flat_channels(base_concat_ds):
    """Test that RemoveFlatChannels removes flat-lined channels."""
    # Get original channel info
    original_raw = base_concat_ds.datasets[0].raw
    original_n_channels = len(original_raw.ch_names)
    original_ch_names = original_raw.ch_names.copy()

    # Make one channel flat (set to constant value) for entire duration
    # Access internal _data attribute to modify in place
    flat_channel_idx = 10
    flat_channel_name = original_ch_names[flat_channel_idx]
    original_raw._data[flat_channel_idx, :] = 0.0  # Set channel to constant value

    # Apply RemoveFlatChannels preprocessor
    # Using max_flatline_duration of 5 seconds - our 10-second flatline exceeds this
    preprocessors = [RemoveFlatChannels(max_flatline_duration=5.0)]
    preprocess(base_concat_ds, preprocessors)

    # Get processed data
    processed_raw = base_concat_ds.datasets[0].raw
    processed_n_channels = len(processed_raw.ch_names)
    processed_ch_names = processed_raw.ch_names

    # Assert that one channel was removed
    assert processed_n_channels == original_n_channels - 1

    # Assert that the flat channel is no longer present
    assert flat_channel_name not in processed_ch_names


@pytest.mark.skipif(not EEGPREP_AVAILABLE, reason="eegprep not installed")
def test_reinterpolate_removed_channels(base_concat_ds):
    """Test that ReinterpolateRemovedChannels restores removed channels."""
    # Get original channel info
    original_raw = base_concat_ds.datasets[0].raw
    original_n_channels = len(original_raw.ch_names)
    original_ch_names = original_raw.ch_names.copy()

    # Make one channel flat (set to constant value) for entire duration
    # Access internal _data attribute to modify in place
    flat_channel_idx = 10
    flat_channel_name = original_ch_names[flat_channel_idx]
    original_raw._data[flat_channel_idx, :] = 0.0  # Set channel to constant value

    # Apply RemoveFlatChannels followed by ReinterpolateRemovedChannels
    # Note: RemoveFlatChannels needs record_orig_chanlocs=True by default
    preprocessors = [
        RemoveFlatChannels(max_flatline_duration=5.0),
        ReinterpolateRemovedChannels(),
    ]
    preprocess(base_concat_ds, preprocessors)

    # Get processed data
    processed_raw = base_concat_ds.datasets[0].raw
    processed_n_channels = len(processed_raw.ch_names)
    processed_ch_names = processed_raw.ch_names

    # Assert that channel count is back to original
    assert processed_n_channels == original_n_channels

    # Assert that the flat channel name is back in the channel list
    assert flat_channel_name in processed_ch_names

    # Get the reinterpolated channel data
    reinterpolated_ch_idx = processed_ch_names.index(flat_channel_name)
    reinterpolated_data = processed_raw.get_data()[reinterpolated_ch_idx, :]

    # Assert that the reinterpolated channel is not flat (not all zeros)
    # It should have been reconstructed from neighboring channels
    assert not np.allclose(reinterpolated_data, 0.0, atol=1e-10)
    # Also verify it has some variance (not constant)
    assert np.std(reinterpolated_data) > 1e-10


@pytest.mark.skipif(not EEGPREP_AVAILABLE, reason="eegprep not installed")
def test_resample(base_concat_ds):
    """Test that Resample changes the sampling rate and data size."""
    # Get original sampling rate and data shape
    original_raw = base_concat_ds.datasets[0].raw
    original_sfreq = original_raw.info['sfreq']
    original_n_samples = original_raw.n_times
    original_duration = original_raw.times[-1]

    # Resample to a different rate
    new_sfreq = 128.0  # Resample from 250 Hz to 128 Hz
    preprocessors = [Resample(sfreq=new_sfreq)]
    preprocess(base_concat_ds, preprocessors)

    # Get processed data
    processed_raw = base_concat_ds.datasets[0].raw
    processed_sfreq = processed_raw.info['sfreq']
    processed_n_samples = processed_raw.n_times
    processed_duration = processed_raw.times[-1]

    # Assert that sampling rate changed
    assert processed_sfreq == new_sfreq
    assert processed_sfreq != original_sfreq

    # Assert that number of samples changed proportionally
    expected_n_samples = int(original_n_samples * new_sfreq / original_sfreq)
    # Allow for small rounding differences
    assert abs(processed_n_samples - expected_n_samples) <= 2

    # Assert that duration remains approximately the same
    np.testing.assert_allclose(processed_duration, original_duration, rtol=0.01)


@pytest.mark.skipif(not EEGPREP_AVAILABLE, reason="eegprep not installed")
def test_remove_common_average_reference(base_concat_ds):
    """Test that RemoveCommonAverageReference removes the common average."""
    # Apply RemoveCommonAverageReference preprocessor
    preprocessors = [RemoveCommonAverageReference()]
    preprocess(base_concat_ds, preprocessors)

    # Get the processed data
    processed_raw = base_concat_ds.datasets[0].raw
    processed_data = processed_raw.get_data()

    # Compute mean across channels (axis=0) for each time point
    channel_means = np.mean(processed_data, axis=0)

    # Assert that the mean across channels is very close to zero for all time points
    # This verifies that the common average reference was applied
    np.testing.assert_allclose(channel_means, 0, atol=1e-4)


@pytest.mark.skipif(not EEGPREP_AVAILABLE, reason="eegprep not installed")
def test_remove_bad_channels_no_locs():
    """Test that RemoveBadChannelsNoLocs removes channels with high noise."""
    # Create a smaller dataset with correlated data for faster processing
    # Use fewer channels and shorter duration
    rng = np.random.RandomState(123)

    # Create a custom smaller dataset for this test
    ch_names = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4',
                'P3', 'Pz', 'P4', 'O1', 'Oz', 'O2', 'FC1', 'FC2']
    n_channels = len(ch_names)
    sfreq = 250
    duration = 60  # 60 seconds as suggested
    n_samples = int(sfreq * duration)

    # Generate correlated data
    data = rng.randn(n_channels, n_samples) * 1e-5
    # Add positive offset to mixing matrix to increase baseline correlations
    # This prevents channels from being flagged as bad due to low natural correlation
    mixing_matrix = rng.randn(n_channels, n_channels) + 2.0
    data = mixing_matrix @ data

    # Create MNE Raw object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * n_channels)
    raw = mne.io.RawArray(data, info)

    # Create dataset
    desc = pd.Series({'subject': 1, 'session': 1})
    base_dataset = BaseDataset(raw, desc, target_name=None)
    concat_ds = BaseConcatDataset([base_dataset])

    # Add high-amplitude Gaussian noise to two channels (indices 5 and 10)
    # This makes them uncorrelated from the other channels
    bad_channel_indices = [5, 10]
    bad_channel_names = [ch_names[i] for i in bad_channel_indices]
    noise_scale = 10 * np.mean(np.std(data, axis=1))  # 10x the mean std

    for idx in bad_channel_indices:
        concat_ds.datasets[0].raw._data[idx, :] += rng.randn(n_samples) * noise_scale

    # Get original channel count
    original_n_channels = len(concat_ds.datasets[0].raw.ch_names)

    # Apply RemoveBadChannelsNoLocs
    # Using default parameters which should be sensitive enough to detect the noise
    preprocessors = [RemoveBadChannelsNoLocs()]
    preprocess(concat_ds, preprocessors)

    # Get processed data
    processed_raw = concat_ds.datasets[0].raw
    processed_n_channels = len(processed_raw.ch_names)
    processed_ch_names = processed_raw.ch_names

    # Assert that channels were removed (at least the 2 we corrupted)
    assert processed_n_channels < original_n_channels

    # Assert that the bad channels are no longer present
    for bad_ch in bad_channel_names:
        assert bad_ch not in processed_ch_names


@pytest.mark.skipif(not EEGPREP_AVAILABLE, reason="eegprep not installed")
def test_remove_bad_windows():
    """Test that RemoveBadWindows removes time periods with high-amplitude artifacts."""
    # Create a long dataset without DC offsets (required for bad window detection)
    ds = _create_synthetic_dataset(duration=1000, have_dc_offsets=False)

    # Get original duration
    original_raw = ds.datasets[0].raw
    original_duration = original_raw.times[-1]
    original_n_samples = original_raw.n_times
    sfreq = original_raw.info['sfreq']

    print(f'Original duration: {original_duration}s, samples: {original_n_samples}')

    # Calculate baseline amplitude
    baseline_data = original_raw.get_data()
    baseline_scale = np.mean(np.std(baseline_data, axis=1))

    # Inject 10 seconds of high-amplitude noise into a consecutive stretch
    # Start at 100 seconds into the recording
    noise_duration = 10  # seconds
    noise_start_time = 100  # seconds
    noise_start_sample = int(noise_start_time * sfreq)
    noise_n_samples = int(noise_duration * sfreq)
    noise_end_sample = noise_start_sample + noise_n_samples

    # Add 10x amplitude noise to all channels in this window
    noise_scale = 10 * baseline_scale
    rng = np.random.RandomState(456)
    n_channels = len(original_raw.ch_names)

    for ch_idx in range(n_channels):
        ds.datasets[0].raw._data[ch_idx, noise_start_sample:noise_end_sample] += \
            rng.randn(noise_n_samples) * noise_scale

    print(f'Injected noise: {noise_duration}s at t={noise_start_time}s')
    print(f'Noise scale: {noise_scale:.6e} (10x baseline: {baseline_scale:.6e})')

    # Apply RemoveBadWindows
    preprocessors = [RemoveBadWindows()]
    preprocess(ds, preprocessors)

    # Get processed data
    processed_raw = ds.datasets[0].raw
    processed_duration = processed_raw.times[-1]
    processed_n_samples = processed_raw.n_times

    # Calculate how much was removed
    duration_removed = original_duration - processed_duration

    print(f'Processed duration: {processed_duration}s, samples: {processed_n_samples}')
    print(f'Duration removed: {duration_removed}s')

    # Assert that at least 5 seconds were removed (conservative estimate)
    # The algorithm uses overlapping windows, so may not remove the full 10s
    assert duration_removed >= 5.0, \
        f"Expected at least 5s removed, but only {duration_removed}s was removed"

    # Also verify that some but not all data was removed
    assert processed_duration < original_duration
    assert processed_duration > original_duration * 0.5  # Should keep most of the data
