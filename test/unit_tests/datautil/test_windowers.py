# Authors: Lukas Gemein <l.gemein@gmail.com>
#          Robin Tibor Schirrmeister <robintibor@gmail.com>
#          Maciej Sliwowski <maciek.sliwowski@gmail.com>
#
# License: BSD-3

import mne
import numpy as np
import pytest

from braindecode.datasets.moabb_datasets import MOABBFetcher
from braindecode.datautil import FixedLengthWindower
from braindecode.datautil.windowers import Windower


@pytest.fixture(scope="module")
def mapping_ds_targets():
    mapping = {"tongue": 0, "left_hand": 1, "right_hand": 2, "feet": 3}
    ds = MOABBFetcher(dataset_name="BNCI2014001", subject=4)
    targets = [mapping[m] for m in ds[0].annotations.description]
    return mapping, ds, targets


def test_one_supercrop_per_original_trial(mapping_ds_targets):
    mapping, ds, targets = mapping_ds_targets
    windower = Windower(trial_start_offset_samples=0,
                        trial_stop_offset_samples=1000,
                        supercrop_size_samples=1000,
                        supercrop_stride_samples=1,
                        mapping=mapping)
    windows = windower(ds[0])
    description = windows.events[:, -1]
    assert len(description) == 48
    np.testing.assert_array_equal(np.array(targets), description)


def test_stride_has_no_effect(mapping_ds_targets):
    mapping, ds, targets = mapping_ds_targets
    windower = Windower(trial_start_offset_samples=0,
                        trial_stop_offset_samples=1000,
                        supercrop_size_samples=1000,
                        supercrop_stride_samples=1000,
                        mapping=mapping)
    windows = windower(ds[0])
    description = windows.events[:, -1]
    assert len(description) == 48
    np.testing.assert_array_equal(np.array(targets), description)


def test_trial_start_offset(mapping_ds_targets):
    mapping, ds, targets = mapping_ds_targets
    windower = Windower(trial_start_offset_samples=-250,
                        trial_stop_offset_samples=250,
                        supercrop_size_samples=250,
                        supercrop_stride_samples=250,
                        mapping=mapping)
    windows = windower(ds[0])
    description = windows.events[:, -1]
    assert len(description) == 96
    np.testing.assert_array_equal(targets, description[0::2])
    np.testing.assert_array_equal(targets, description[1::2])


def test_shifting_last_supercrop_back_in(mapping_ds_targets):
    mapping, ds, targets = mapping_ds_targets
    windower = Windower(trial_start_offset_samples=-250,
                        trial_stop_offset_samples=250,
                        supercrop_size_samples=250,
                        supercrop_stride_samples=300,
                        mapping=mapping)
    windows = windower(ds[0])
    description = windows.events[:, -1]
    assert len(description) == 96
    np.testing.assert_array_equal(targets, description[0::2])
    np.testing.assert_array_equal(targets, description[1::2])


def test_dropping_last_incomplete_supercrop(mapping_ds_targets):
    mapping, ds, targets = mapping_ds_targets
    windower = Windower(trial_start_offset_samples=-250,
                        trial_stop_offset_samples=250,
                        supercrop_size_samples=250,
                        supercrop_stride_samples=300,
                        drop_samples=True, mapping=mapping)
    windows = windower(ds[0])
    description = windows.events[:, -1]
    assert len(description) == 48
    np.testing.assert_array_equal(targets, description)


def test_maximally_overlapping_supercrops(mapping_ds_targets):
    mapping, ds, targets = mapping_ds_targets
    windower = Windower(trial_start_offset_samples=-2,
                        trial_stop_offset_samples=1000,
                        supercrop_size_samples=1000,
                        supercrop_stride_samples=1,
                        mapping=mapping)
    windows = windower(ds[0])
    description = windows.events[:, -1]
    assert len(description) == 48 * 3
    np.testing.assert_array_equal(targets, description[0::3])
    np.testing.assert_array_equal(targets, description[1::3])
    np.testing.assert_array_equal(targets, description[2::3])


def test_single_sample_size_supercrops(mapping_ds_targets):
    mapping, ds, targets = mapping_ds_targets
    windower = Windower(trial_start_offset_samples=0,
                        trial_stop_offset_samples=1000,
                        supercrop_size_samples=1,
                        supercrop_stride_samples=1,
                        mapping=mapping)
    windows = windower(ds[0])
    description = windows.events[:, -1]
    assert len(description) == 48000
    np.testing.assert_array_equal(targets, description[::1000])
    np.testing.assert_array_equal(targets, description[999::1000])


def test_overlapping_trial_offsets(mapping_ds_targets):
    mapping, ds, targets = mapping_ds_targets
    windower = Windower(trial_start_offset_samples=-2000,
                        trial_stop_offset_samples=1000,
                        supercrop_size_samples=1000,
                        supercrop_stride_samples=1000,
                        mapping=mapping)
    with pytest.raises(AssertionError, match='trials overlap not implemented'):
        windower(ds[0])


# TODO: add tests for case with drop_last_sample==False
def test_fixed_length_windower():
    rng = np.random.RandomState(42)
    info = mne.create_info(ch_names=['0', '1'], sfreq=50, ch_types='eeg')
    data = rng.randn(2, 1000)
    raw = mne.io.RawArray(data=data, info=info)

    # test case:
    # (window_size_samples, overlap_size_samples, drop_last_samples,
    # trial_start_offset_samples, n_windows)
    test_cases = [
        (100, 10, True, 0., 11),
        # TODO: does using trial_start_offset_samples have sense?
        # (100, 10, True, -0.5, 11),
        (100, 50, True, 0., 19),
        (None, 50, True, 0., 1)
    ]

    for i, test_case in enumerate(test_cases):
        (window_size, overlap_size, drop_last_samples,
         trial_start_offset_samples, n_windows) = test_case
        windower = FixedLengthWindower(
            window_size_samples=window_size,
            overlap_size_samples=overlap_size,
            drop_last_samples=drop_last_samples,
            trial_start_offset_samples=trial_start_offset_samples)

        epochs = windower(raw)
        epochs_data = epochs.get_data()
        if window_size is None:
            window_size = data.shape[1]
        idxs = np.arange(
            0, data.shape[1] - window_size + 1, window_size - overlap_size)

        assert len(idxs) == epochs_data.shape[0], \
            f"Number of epochs different than expected for test case {i}"
        assert window_size == epochs_data.shape[2], \
            f"Window size different than expected for test case {i}"
        for j, idx in enumerate(idxs):
            np.testing.assert_allclose(
                data[:, idx: idx + window_size], epochs_data[j, :],
                err_msg=f"Epochs different for test case {i} for epoch {j}"
            )
