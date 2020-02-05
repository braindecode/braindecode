# Authors: Lukas Gemein <l.gemein@gmail.com>
#          Robin Tibor Schirrmeister <robintibor@gmail.com>
#          Maciej Sliwowski <maciek.sliwowski@gmail.com>
#
# License: BSD-3

import mne
import numpy as np
import pandas as pd
import pytest

from braindecode.datasets.base import BaseDataset, BaseConcatDataset
from braindecode.datasets.datasets import fetch_data_with_moabb
from braindecode.datautil import (
    create_windows_from_events, create_fixed_length_windows)


@pytest.fixture(scope="module")
def concat_ds_targets():
    raws, description = fetch_data_with_moabb(
        dataset_name="BNCI2014001", subject_ids=4)
    events = mne.find_events(raws[0])
    targets = events[:, -1]
    ds = BaseDataset(raws[0], description.iloc[0])
    concat_ds = BaseConcatDataset([ds])
    return concat_ds, targets


def test_one_supercrop_per_original_trial(concat_ds_targets):
    concat_ds, targets = concat_ds_targets
    windows = create_windows_from_events(
        concat_ds=concat_ds,
        trial_start_offset_samples=0, trial_stop_offset_samples=1000,
        supercrop_size_samples=1000, supercrop_stride_samples=1,
        drop_samples=False)
    description = windows.datasets[0].windows.metadata["target"].to_list()
    assert len(description) == len(targets)
    np.testing.assert_array_equal(description, targets)


def test_stride_has_no_effect(concat_ds_targets):
    concat_ds, targets = concat_ds_targets
    windows = create_windows_from_events(
        concat_ds=concat_ds,
        trial_start_offset_samples=0, trial_stop_offset_samples=1000,
        supercrop_size_samples=1000, supercrop_stride_samples=1000,
        drop_samples=False)
    description = windows.datasets[0].windows.metadata["target"].to_list()
    assert len(description) == len(targets)
    np.testing.assert_array_equal(description, targets)


def test_trial_start_offset(concat_ds_targets):
    concat_ds, targets = concat_ds_targets
    windows = create_windows_from_events(
        concat_ds=concat_ds,
        trial_start_offset_samples=-250, trial_stop_offset_samples=250,
        supercrop_size_samples=250, supercrop_stride_samples=250,
        drop_samples=False)
    description = windows.datasets[0].windows.metadata["target"].to_list()
    assert len(description) == len(targets) * 2
    np.testing.assert_array_equal(description[0::2], targets)
    np.testing.assert_array_equal(description[1::2], targets)


def test_shifting_last_supercrop_back_in(concat_ds_targets):
    concat_ds, targets = concat_ds_targets
    windows = create_windows_from_events(
        concat_ds=concat_ds,
        trial_start_offset_samples=-250, trial_stop_offset_samples=250,
        supercrop_size_samples=250, supercrop_stride_samples=300,
        drop_samples=False)
    description = windows.datasets[0].windows.metadata["target"].to_list()
    assert len(description) == len(targets) * 2
    np.testing.assert_array_equal(description[0::2], targets)
    np.testing.assert_array_equal(description[1::2], targets)


def test_dropping_last_incomplete_supercrop(concat_ds_targets):
    concat_ds, targets = concat_ds_targets
    windows = create_windows_from_events(
        concat_ds=concat_ds,
        trial_start_offset_samples=-250, trial_stop_offset_samples=250,
        supercrop_size_samples=250, supercrop_stride_samples=300,
        drop_samples=True)
    description = windows.datasets[0].windows.metadata["target"].to_list()
    assert len(description) == len(targets)
    np.testing.assert_array_equal(description, targets)


def test_maximally_overlapping_supercrops(concat_ds_targets):
    concat_ds, targets = concat_ds_targets
    windows = create_windows_from_events(
        concat_ds=concat_ds,
        trial_start_offset_samples=-2, trial_stop_offset_samples=1000,
        supercrop_size_samples=1000, supercrop_stride_samples=1,
        drop_samples=False)
    description = windows.datasets[0].windows.metadata["target"].to_list()
    assert len(description) == len(targets) * 3
    np.testing.assert_array_equal(description[0::3], targets)
    np.testing.assert_array_equal(description[1::3], targets)
    np.testing.assert_array_equal(description[2::3], targets)


def test_single_sample_size_supercrops(concat_ds_targets):
    concat_ds, targets = concat_ds_targets
    windows = create_windows_from_events(
        concat_ds=concat_ds,
        trial_start_offset_samples=0, trial_stop_offset_samples=1000,
        supercrop_size_samples=1, supercrop_stride_samples=1,
        drop_samples=False)
    description = windows.datasets[0].windows.metadata["target"].to_list()
    assert len(description) == len(targets) * 1000
    np.testing.assert_array_equal(description[::1000], targets)
    np.testing.assert_array_equal(description[999::1000], targets)


def test_overlapping_trial_offsets(concat_ds_targets):
    concat_ds, _ = concat_ds_targets
    with pytest.raises(NotImplementedError,
                       match='Trial overlap not implemented.'):
        create_windows_from_events(
            concat_ds=concat_ds,
            trial_start_offset_samples=-2000, trial_stop_offset_samples=1000,
            supercrop_size_samples=1000, supercrop_stride_samples=1000,
            drop_samples=False)


@pytest.mark.parametrize(
    'start_offset_samples,supercrop_size_samples,supercrop_stride_samples,drop_samples',
    [(0, 100, 90, True),
     (0, 100, 50, True),
     (0, 50, 50, True),
     (0, 50, 50, False),
     (0, None, 50, True),
     (5, 10, 20, True),
     (5, 10, 20, False)]
)
def test_fixed_length_windower(start_offset_samples, supercrop_size_samples,
                               supercrop_stride_samples, drop_samples):
    rng = np.random.RandomState(42)
    info = mne.create_info(ch_names=['0', '1'], sfreq=50, ch_types='eeg')
    data = rng.randn(2, 1000)
    raw = mne.io.RawArray(data=data, info=info)
    desc = pd.Series({'pathological': True, 'gender': 'M', 'age': 48})
    base_ds = BaseDataset(raw, desc, target_name="age")
    concat_ds = BaseConcatDataset([base_ds])

    if supercrop_size_samples is None:
        supercrop_size_samples = base_ds.raw.n_times
    stop_offset_samples = data.shape[1] - start_offset_samples
    epochs = create_fixed_length_windows(
        concat_ds, start_offset_samples=start_offset_samples,
        stop_offset_samples=stop_offset_samples,
        supercrop_size_samples=supercrop_size_samples,
        supercrop_stride_samples=supercrop_stride_samples,
        drop_samples=drop_samples)

    epochs_data = epochs.datasets[0].windows.get_data()

    idxs = np.arange(
        start_offset_samples,
        stop_offset_samples - supercrop_size_samples + 1,
        supercrop_stride_samples)
    if not drop_samples and idxs[-1] != stop_offset_samples - supercrop_size_samples:
        idxs = np.append(idxs, stop_offset_samples - supercrop_size_samples)

    assert len(idxs) == epochs_data.shape[0], (
        f"Number of epochs different than expected")
    assert supercrop_size_samples == epochs_data.shape[2], (
        f"Window size different than expected")
    for j, idx in enumerate(idxs):
        np.testing.assert_allclose(
            base_ds.raw.get_data()[:, idx:idx + supercrop_size_samples],
            epochs_data[j, :],
            err_msg=f"Epochs different for epoch {j}"
        )
