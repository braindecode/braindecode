# Authors: Lukas Gemein <l.gemein@gmail.com>
#          Robin Tibor Schirrmeister <robintibor@gmail.com>
#          Maciej Sliwowski <maciek.sliwowski@gmail.com>
#
# License: BSD-3

import numpy as np
import pandas as pd
import pytest
import mne

from braindecode.datasets.specific_datasets import fetch_data_with_moabb
from braindecode.datautil import FixedLengthWindower, EventWindower
from braindecode.datasets.dataset import BaseDataset


@pytest.fixture(scope="module")
def raw_info_targets():
    raws, info = fetch_data_with_moabb(dataset_name="BNCI2014001", subject_ids=4)
    events = mne.find_events(raws[0])
    targets = events[:, -1]
    return raws[0], info.iloc[0], targets


def test_one_supercrop_per_original_trial(raw_info_targets):
    raw, info, targets = raw_info_targets
    base_ds = BaseDataset(raw, info)
    windower = EventWindower(
        trial_start_offset_samples=0, trial_stop_offset_samples=1000,
        supercrop_size_samples=1000, supercrop_stride_samples=1)
    windows = windower(base_ds)
    description = windows.events[:, -1]
    assert len(description) == len(targets)
    np.testing.assert_array_equal(description, targets)


def test_stride_has_no_effect(raw_info_targets):
    raw, info, targets = raw_info_targets
    base_ds = BaseDataset(raw, info)
    windower = EventWindower(
        trial_start_offset_samples=0, trial_stop_offset_samples=1000,
        supercrop_size_samples=1000, supercrop_stride_samples=1000)
    windows = windower(base_ds)
    description = windows.events[:,-1]
    assert len(description) == len(targets)
    np.testing.assert_array_equal(description, targets)


def test_trial_start_offset(raw_info_targets):
    raw, info, targets = raw_info_targets
    base_ds = BaseDataset(raw, info)
    windower = EventWindower(
        trial_start_offset_samples=-250, trial_stop_offset_samples=250,
        supercrop_size_samples=250, supercrop_stride_samples=250)
    windows = windower(base_ds)
    description = windows.events[:,-1]
    assert len(description) == len(targets) * 2
    np.testing.assert_array_equal(description[0::2], targets)
    np.testing.assert_array_equal(description[1::2], targets)


def test_shifting_last_supercrop_back_in(raw_info_targets):
    raw, info, targets = raw_info_targets
    base_ds = BaseDataset(raw, info)
    windower = EventWindower(
        trial_start_offset_samples=-250, trial_stop_offset_samples=250,
        supercrop_size_samples=250, supercrop_stride_samples=300)
    windows = windower(base_ds)
    description = windows.events[:,-1]
    assert len(description) == len(targets) * 2
    np.testing.assert_array_equal(description[0::2], targets)
    np.testing.assert_array_equal(description[1::2], targets)


def test_dropping_last_incomplete_supercrop(raw_info_targets):
    raw, info, targets = raw_info_targets
    base_ds = BaseDataset(raw, info)
    windower = EventWindower(
        trial_start_offset_samples=-250, trial_stop_offset_samples=250,
        supercrop_size_samples=250, supercrop_stride_samples=300,
        drop_samples=True)
    windows = windower(base_ds)
    description = windows.events[:,-1]
    assert len(description) == len(targets)
    np.testing.assert_array_equal(description, targets)


def test_maximally_overlapping_supercrops(raw_info_targets):
    raw, info, targets = raw_info_targets
    base_ds = BaseDataset(raw, info)
    windower = EventWindower(
        trial_start_offset_samples=-2, trial_stop_offset_samples=1000,
        supercrop_size_samples=1000, supercrop_stride_samples=1)
    windows = windower(base_ds)
    description = windows.events[:,-1]
    assert len(description) == len(targets) * 3
    np.testing.assert_array_equal(description[0::3], targets)
    np.testing.assert_array_equal(description[1::3], targets)
    np.testing.assert_array_equal(description[2::3], targets)


def test_single_sample_size_supercrops(raw_info_targets):
    raw, info, targets = raw_info_targets
    base_ds = BaseDataset(raw, info)
    windower = EventWindower(
        trial_start_offset_samples=0, trial_stop_offset_samples=1000,
        supercrop_size_samples=1, supercrop_stride_samples=1)
    windows = windower(base_ds)
    description = windows.events[:,-1]
    assert len(description) == len(targets) * 1000
    np.testing.assert_array_equal(description[::1000], targets)
    np.testing.assert_array_equal(description[999::1000], targets)


def test_overlapping_trial_offsets(raw_info_targets):
    raw, info, targets = raw_info_targets
    base_ds = BaseDataset(raw, info)
    windower = EventWindower(
        trial_start_offset_samples=-2000, trial_stop_offset_samples=1000,
        supercrop_size_samples=1000, supercrop_stride_samples=1000)
    with pytest.raises(AssertionError, match='trials overlap not implemented'):
        windower(base_ds)


@pytest.fixture(scope="module")
def setUp():
    rng = np.random.RandomState(42)
    sfreq = 50
    info = mne.create_info(ch_names=['0', '1'], sfreq=sfreq, ch_types='eeg')
    data = rng.randn(2, 1000)
    raw = mne.io.RawArray(data=data, info=info)
    df = pd.DataFrame(zip([True], ["M"], [48]),
                      columns=["pathological", "gender", "age"])
    base_ds = BaseDataset(raw, df, target="age")

    # test case:
    # (window_size_samples, stride_size_samples, drop_last_samples,
    # trial_start_offset_samples, n_windows)
    test_cases = [
        (100, 100-10, True, 0., 11),
        # TODO: does using trial_start_offset_samples have sense?
        # no, not for FixedLengthWindower
        # (100, 100-10, True, -0.5, 11),
        (100, 100-50, True, 0., 19),
        (None, 100-50, True, 0., 1)
    ]
    return base_ds, test_cases


def test_windower_results(setUp):
    base_ds, test_cases = setUp
    for test_case in test_cases:
        (window_size, stride_size, drop_last_samples,
         trial_start_offset_samples, n_windows) = test_case
        if window_size is None:
            window_size = base_ds.raw.n_times
        windower = FixedLengthWindower(
            supercrop_size_samples=window_size,
            supercrop_stride_samples=stride_size,
            drop_samples=drop_last_samples,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=-trial_start_offset_samples + window_size)

        epochs = windower(base_ds)
        epochs_data = epochs.get_data()
        if window_size is None:
            window_size = base_ds.raw.get_data().shape[1]
        idxs = np.arange(0,
                         base_ds.raw.get_data().shape[1] - window_size + 1,
                         stride_size)

        assert len(idxs) == epochs_data.shape[0]
        assert window_size == epochs_data.shape[2]
        for i, idx in enumerate(idxs):
            np.testing.assert_allclose(
                base_ds.raw.get_data()[:, idx: idx + window_size],
                epochs_data[i, :])
