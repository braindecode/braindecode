# Authors: Lukas Gemein <l.gemein@gmail.com>
#          Robin Tibor Schirrmeister <robintibor@gmail.com>
#          Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD-3

import copy

import mne
import numpy as np
import pandas as pd
import pytest

from braindecode.datasets.base import BaseDataset, BaseConcatDataset
from braindecode.datasets.moabb import fetch_data_with_moabb
from braindecode.preprocessing import (
    create_windows_from_events, create_fixed_length_windows)
from braindecode.preprocessing.preprocess import Preprocessor, preprocess
from braindecode.preprocessing.windowers import create_windows_from_target_channels
from braindecode.util import create_mne_dummy_raw


def _get_raw(tmpdir_factory, description=None):
    _, fnames = create_mne_dummy_raw(
        n_channels=2, n_times=20000, sfreq=100, description=description,
        savedir=tmpdir_factory.mktemp('data'), save_format='fif',
        random_state=87)
    raw = mne.io.read_raw_fif(fnames['fif'], preload=False, verbose=None)
    return raw


@pytest.fixture(scope="module")
def concat_ds_targets():
    raws, description = fetch_data_with_moabb(
        dataset_name="BNCI2014001", subject_ids=4)
    events, _ = mne.events_from_annotations(raws[0])
    targets = events[:, -1] - 1
    ds = BaseDataset(raws[0], description.iloc[0])
    concat_ds = BaseConcatDataset([ds])
    return concat_ds, targets


@pytest.fixture(scope="session")
def lazy_loadable_dataset(tmpdir_factory):
    """Make a dataset of fif files that can be loaded lazily.
    """
    raw = _get_raw(tmpdir_factory)
    base_ds = BaseDataset(raw, description=pd.Series({'file_id': 1}))
    concat_ds = BaseConcatDataset([base_ds])

    return concat_ds


def test_windows_from_events_preload_false(lazy_loadable_dataset):
    windows = create_windows_from_events(
        concat_ds=lazy_loadable_dataset, trial_start_offset_samples=0,
        trial_stop_offset_samples=0, window_size_samples=100,
        window_stride_samples=100, drop_last_window=False)

    assert all([not ds.windows.preload for ds in windows.datasets])


def test_windows_from_events_n_jobs(lazy_loadable_dataset):
    longer_dataset = BaseConcatDataset([lazy_loadable_dataset.datasets[0]] * 8)
    windows = [create_windows_from_events(
        concat_ds=longer_dataset, trial_start_offset_samples=0,
        trial_stop_offset_samples=0, window_size_samples=100,
        window_stride_samples=100, drop_last_window=False, preload=True,
        n_jobs=n_jobs) for n_jobs in [1, 2]]

    assert windows[0].description.equals(windows[1].description)
    for ds1, ds2 in zip(windows[0].datasets, windows[1].datasets):
        # assert ds1.windows == ds2.windows  # Runs locally, fails in CI
        assert np.allclose(ds1.windows.get_data(), ds2.windows.get_data())
        assert pd.Series(ds1.windows.info).to_json() == \
               pd.Series(ds2.windows.info).to_json()
        assert ds1.description.equals(ds2.description)
        assert np.array_equal(ds1.y, ds2.y)
        assert np.array_equal(ds1.crop_inds, ds2.crop_inds)


def test_windows_from_events_mapping_filter(tmpdir_factory):
    raw = _get_raw(tmpdir_factory, 5 * ['T0', 'T1'])
    base_ds = BaseDataset(raw, description=pd.Series({'file_id': 1}))
    concat_ds = BaseConcatDataset([base_ds])

    windows = create_windows_from_events(
        concat_ds=concat_ds, trial_start_offset_samples=0,
        trial_stop_offset_samples=0, window_size_samples=100,
        window_stride_samples=100, drop_last_window=False, mapping={'T1': 0})
    description = windows.datasets[0].windows.metadata['target'].to_list()

    assert len(description) == 5
    np.testing.assert_array_equal(description, np.zeros(5))
    # dataset should contain only 'T1' events
    np.testing.assert_array_equal(
        (raw.time_as_index(raw.annotations.onset[1::2], use_rounding=True)),
        windows.datasets[0].windows.events[:, 0])


def test_windows_from_events_different_events(tmpdir_factory):
    description_expected = 5 * ['T0', 'T1'] + 4 * ['T2', 'T3'] + 2 * ['T1']
    raw = _get_raw(tmpdir_factory, description_expected[:10])
    base_ds = BaseDataset(raw, description=pd.Series({'file_id': 1}))

    raw_1 = _get_raw(tmpdir_factory, description_expected[10:])
    base_ds_1 = BaseDataset(raw_1, description=pd.Series({'file_id': 2}))
    concat_ds = BaseConcatDataset([base_ds, base_ds_1])

    windows = create_windows_from_events(
        concat_ds=concat_ds, trial_start_offset_samples=0,
        trial_stop_offset_samples=0, window_size_samples=100,
        window_stride_samples=100, drop_last_window=False)
    description = []
    events = []
    for ds in windows.datasets:
        description += ds.windows.metadata['target'].to_list()
        events += ds.windows.events[:, 0].tolist()

    assert len(description) == 20
    np.testing.assert_array_equal(description,
                                  5 * [0, 1] + 4 * [2, 3] + 2 * [1])
    np.testing.assert_array_equal(
        np.concatenate(
            [raw.time_as_index(raw.annotations.onset, use_rounding=True),
             raw_1.time_as_index(raw.annotations.onset, use_rounding=True)]),
        events)


def test_fixed_length_windows_preload_false(lazy_loadable_dataset):
    windows = create_fixed_length_windows(
        concat_ds=lazy_loadable_dataset, start_offset_samples=0,
        stop_offset_samples=100, window_size_samples=100,
        window_stride_samples=100, drop_last_window=False, preload=False)

    assert all([not ds.windows.preload for ds in windows.datasets])


def test_one_window_per_original_trial(concat_ds_targets):
    concat_ds, targets = concat_ds_targets
    windows = create_windows_from_events(
        concat_ds=concat_ds,
        trial_start_offset_samples=0, trial_stop_offset_samples=0,
        window_size_samples=1000, window_stride_samples=1,
        drop_last_window=False)
    description = windows.datasets[0].windows.metadata["target"].to_list()
    assert len(description) == len(targets)
    np.testing.assert_array_equal(description, targets)


def test_stride_has_no_effect(concat_ds_targets):
    concat_ds, targets = concat_ds_targets
    windows = create_windows_from_events(
        concat_ds=concat_ds,
        trial_start_offset_samples=0, trial_stop_offset_samples=0,
        window_size_samples=1000, window_stride_samples=1000,
        drop_last_window=False)
    description = windows.datasets[0].windows.metadata["target"].to_list()
    assert len(description) == len(targets)
    np.testing.assert_array_equal(description, targets)


def test_trial_start_offset(concat_ds_targets):
    concat_ds, targets = concat_ds_targets
    windows = create_windows_from_events(
        concat_ds=concat_ds,
        trial_start_offset_samples=-250, trial_stop_offset_samples=-750,
        window_size_samples=250, window_stride_samples=250,
        drop_last_window=False)
    description = windows.datasets[0].windows.metadata["target"].to_list()
    assert len(description) == len(targets) * 2
    np.testing.assert_array_equal(description[0::2], targets)
    np.testing.assert_array_equal(description[1::2], targets)


def test_shifting_last_window_back_in(concat_ds_targets):
    concat_ds, targets = concat_ds_targets
    windows = create_windows_from_events(
        concat_ds=concat_ds,
        trial_start_offset_samples=-250, trial_stop_offset_samples=-750,
        window_size_samples=250, window_stride_samples=300,
        drop_last_window=False)
    description = windows.datasets[0].windows.metadata["target"].to_list()
    assert len(description) == len(targets) * 2
    np.testing.assert_array_equal(description[0::2], targets)
    np.testing.assert_array_equal(description[1::2], targets)


def test_dropping_last_incomplete_window(concat_ds_targets):
    concat_ds, targets = concat_ds_targets
    windows = create_windows_from_events(
        concat_ds=concat_ds,
        trial_start_offset_samples=-250, trial_stop_offset_samples=-750,
        window_size_samples=250, window_stride_samples=300,
        drop_last_window=True)
    description = windows.datasets[0].windows.metadata["target"].to_list()
    assert len(description) == len(targets)
    np.testing.assert_array_equal(description, targets)


def test_maximally_overlapping_windows(concat_ds_targets):
    concat_ds, targets = concat_ds_targets
    windows = create_windows_from_events(
        concat_ds=concat_ds,
        trial_start_offset_samples=-2, trial_stop_offset_samples=0,
        window_size_samples=1000, window_stride_samples=1,
        drop_last_window=False)
    description = windows.datasets[0].windows.metadata["target"].to_list()
    assert len(description) == len(targets) * 3
    np.testing.assert_array_equal(description[0::3], targets)
    np.testing.assert_array_equal(description[1::3], targets)
    np.testing.assert_array_equal(description[2::3], targets)


def test_single_sample_size_windows(concat_ds_targets):
    concat_ds, targets = concat_ds_targets
    # reduce dataset for faster test, only first 3 events
    targets = targets[:3]
    underlying_raw = concat_ds.datasets[0].raw
    annotations = underlying_raw.annotations
    underlying_raw.set_annotations(annotations[:3])
    # have to supply explicit mapping as only two classes appear in first 3
    # targets
    windows = create_windows_from_events(
        concat_ds=concat_ds,
        trial_start_offset_samples=0, trial_stop_offset_samples=0,
        window_size_samples=1, window_stride_samples=1,
        drop_last_window=False, mapping=dict(tongue=3, left_hand=1,
                                             right_hand=2, feet=4))
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
            trial_start_offset_samples=-2000, trial_stop_offset_samples=0,
            window_size_samples=1000, window_stride_samples=1000,
            drop_last_window=False)


@pytest.mark.parametrize('drop_bad_windows,preload',
                         [(True, False), (True, False)])
def test_drop_bad_windows(concat_ds_targets, drop_bad_windows, preload):
    concat_ds, _ = concat_ds_targets
    windows_from_events = create_windows_from_events(
        concat_ds=concat_ds, trial_start_offset_samples=0,
        trial_stop_offset_samples=0, window_size_samples=100,
        window_stride_samples=100, drop_last_window=False, preload=preload,
        drop_bad_windows=drop_bad_windows)

    windows_fixed_length = create_fixed_length_windows(
        concat_ds=concat_ds, start_offset_samples=0, stop_offset_samples=1000,
        window_size_samples=1000, window_stride_samples=1000,
        drop_last_window=False, preload=preload, drop_bad_windows=drop_bad_windows)

    assert (windows_from_events.datasets[0].windows._bad_dropped ==
            drop_bad_windows)
    assert (windows_fixed_length.datasets[0].windows._bad_dropped ==
            drop_bad_windows)


def test_windows_from_events_(lazy_loadable_dataset):
    msg = '"trial_stop_offset_samples" too large\\. Stop of last trial ' \
          '\\(19900\\) \\+ "trial_stop_offset_samples" \\(250\\) must be ' \
          'smaller than length of recording \\(20000\\)\\.'
    with pytest.raises(ValueError, match=msg):
        create_windows_from_events(
            concat_ds=lazy_loadable_dataset, trial_start_offset_samples=0,
            trial_stop_offset_samples=250, window_size_samples=100,
            window_stride_samples=100, drop_last_window=False)


@pytest.mark.parametrize(
    'start_offset_samples,window_size_samples,window_stride_samples,drop_last_window,mapping',
    [(0, 100, 90, True, None),
     (0, 100, 50, True, {48: 0}),
     (0, 50, 50, True, None),
     (0, 50, 50, False, None),
     (0, None, 50, True, None),
     (5, 10, 20, True, None),
     (5, 10, 39, False, None)]
)
def test_fixed_length_windower(start_offset_samples, window_size_samples,
                               window_stride_samples, drop_last_window, mapping):
    rng = np.random.RandomState(42)
    info = mne.create_info(ch_names=['0', '1'], sfreq=50, ch_types='eeg')
    data = rng.randn(2, 1000)
    raw = mne.io.RawArray(data=data, info=info)
    desc = pd.Series({'pathological': True, 'gender': 'M', 'age': 48})
    base_ds = BaseDataset(raw, desc, target_name="age")
    concat_ds = BaseConcatDataset([base_ds])

    if window_size_samples is None:
        window_size_samples = base_ds.raw.n_times
    stop_offset_samples = data.shape[1] - start_offset_samples
    epochs_ds = create_fixed_length_windows(
        concat_ds, start_offset_samples=start_offset_samples,
        stop_offset_samples=stop_offset_samples,
        window_size_samples=window_size_samples,
        window_stride_samples=window_stride_samples,
        drop_last_window=drop_last_window, mapping=mapping)

    if mapping is not None:
        assert base_ds.description[base_ds.target_name] == 48
        assert all(epochs_ds.datasets[0].windows.metadata['target'] == 0)

    epochs_data = epochs_ds.datasets[0].windows.get_data()

    idxs = np.arange(
        start_offset_samples,
        stop_offset_samples - window_size_samples + 1,
        window_stride_samples)
    if not drop_last_window and idxs[-1] != stop_offset_samples - window_size_samples:
        idxs = np.append(idxs, stop_offset_samples - window_size_samples)

    assert len(idxs) == epochs_data.shape[0], (
        'Number of epochs different than expected')
    assert window_size_samples == epochs_data.shape[2], (
        'Window size different than expected')
    for j, idx in enumerate(idxs):
        np.testing.assert_allclose(
            base_ds.raw.get_data()[:, idx:idx + window_size_samples],
            epochs_data[j, :],
            err_msg=f'Epochs different for epoch {j}'
        )


def test_fixed_length_windower_n_jobs(lazy_loadable_dataset):
    longer_dataset = BaseConcatDataset([lazy_loadable_dataset.datasets[0]] * 8)
    windows = [create_fixed_length_windows(
        concat_ds=longer_dataset, start_offset_samples=0,
        stop_offset_samples=None, window_size_samples=100,
        window_stride_samples=100, drop_last_window=True, preload=True,
        n_jobs=n_jobs) for n_jobs in [1, 2]]

    assert windows[0].description.equals(windows[1].description)
    for ds1, ds2 in zip(windows[0].datasets, windows[1].datasets):
        # assert ds1.windows == ds2.windows  # Runs locally, fails in CI
        assert np.allclose(ds1.windows.get_data(), ds2.windows.get_data())
        assert pd.Series(ds1.windows.info).to_json() == \
               pd.Series(ds2.windows.info).to_json()
        assert ds1.description.equals(ds2.description)
        assert np.array_equal(ds1.y, ds2.y)
        assert np.array_equal(ds1.crop_inds, ds2.crop_inds)


def test_windows_from_events_cropped(lazy_loadable_dataset):
    """Test windowing from events on cropped data.

    Cropping raw data changes the `first_samp` attribute of the Raw object, and
    so it is important to test this is taken into account by the windowers.
    """
    tmin, tmax = 100, 120

    ds = copy.deepcopy(lazy_loadable_dataset)
    ds.datasets[0].raw.annotations.crop(tmin, tmax)

    crop_ds = copy.deepcopy(lazy_loadable_dataset)
    crop_transform = Preprocessor('crop', tmin=tmin, tmax=tmax)
    preprocess(crop_ds, [crop_transform])

    # Extract windows
    windows1 = create_windows_from_events(
        concat_ds=ds, trial_start_offset_samples=0, trial_stop_offset_samples=0,
        window_size_samples=100, window_stride_samples=100,
        drop_last_window=False)
    windows2 = create_windows_from_events(
        concat_ds=crop_ds, trial_start_offset_samples=0,
        trial_stop_offset_samples=0, window_size_samples=100,
        window_stride_samples=100, drop_last_window=False)
    assert (windows1[0][0] == windows2[0][0]).all()

    # Make sure events that fall outside of recording will trigger an error
    with pytest.raises(
            ValueError, match='"trial_stop_offset_samples" too large'):
        create_windows_from_events(
            concat_ds=ds, trial_start_offset_samples=0,
            trial_stop_offset_samples=10000, window_size_samples=100,
            window_stride_samples=100, drop_last_window=False)
    with pytest.raises(
            ValueError, match='"trial_stop_offset_samples" too large'):
        create_windows_from_events(
            concat_ds=crop_ds, trial_start_offset_samples=0,
            trial_stop_offset_samples=2001, window_size_samples=100,
            window_stride_samples=100, drop_last_window=False)


def test_windows_fixed_length_cropped(lazy_loadable_dataset):
    """Test fixed length windowing on cropped data.

    Cropping raw data changes the `first_samp` attribute of the Raw object, and
    so it is important to test this is taken into account by the windowers.
    """
    tmin, tmax = 100, 120

    ds = copy.deepcopy(lazy_loadable_dataset)
    ds.datasets[0].raw.annotations.crop(tmin, tmax)

    crop_ds = copy.deepcopy(lazy_loadable_dataset)
    crop_transform = Preprocessor('crop', tmin=tmin, tmax=tmax)
    preprocess(crop_ds, [crop_transform])

    # Extract windows
    sfreq = ds.datasets[0].raw.info['sfreq']
    tmin_samples, tmax_samples = int(tmin * sfreq), int(tmax * sfreq)

    windows1 = create_fixed_length_windows(
        concat_ds=ds, start_offset_samples=tmin_samples,
        stop_offset_samples=tmax_samples, window_size_samples=100,
        window_stride_samples=100, drop_last_window=True)
    windows2 = create_fixed_length_windows(
        concat_ds=crop_ds, start_offset_samples=0,
        stop_offset_samples=None, window_size_samples=100,
        window_stride_samples=100, drop_last_window=True)
    assert (windows1[0][0] == windows2[0][0]).all()


def test_epochs_kwargs(lazy_loadable_dataset):
    picks = ['ch0']
    on_missing = 'warning'
    flat = {'eeg': 3e-6}
    reject = {'eeg': 43e-6}

    windows = create_windows_from_events(
        concat_ds=lazy_loadable_dataset, trial_start_offset_samples=0,
        trial_stop_offset_samples=0, window_size_samples=100,
        window_stride_samples=100, drop_last_window=False, picks=picks,
        on_missing=on_missing, flat=flat, reject=reject)

    epochs = windows.datasets[0].windows
    assert epochs.ch_names == picks
    assert epochs.reject == reject
    assert epochs.flat == flat
    for ds in windows.datasets:
        assert ds.window_kwargs == [
            ('create_windows_from_events', {
                'infer_mapping': True, 'infer_window_size_stride': False,
                'trial_start_offset_samples': 0, 'trial_stop_offset_samples': 0,
                'window_size_samples': 100, 'window_stride_samples': 100,
                'drop_last_window': False, 'mapping': {'test': 0}, 'preload': False,
                'drop_bad_windows': True, 'picks': picks, 'reject': reject,
                'flat': flat, 'on_missing': on_missing,
                'accepted_bads_ratio': 0.0})
        ]

    windows = create_fixed_length_windows(
        concat_ds=lazy_loadable_dataset, start_offset_samples=0,
        stop_offset_samples=None, window_size_samples=100,
        window_stride_samples=100, drop_last_window=False, picks=picks,
        on_missing=on_missing, flat=flat, reject=reject)

    epochs = windows.datasets[0].windows
    assert epochs.ch_names == picks
    assert epochs.reject == reject
    assert epochs.flat == flat
    for ds in windows.datasets:
        assert ds.window_kwargs == [
            ('create_fixed_length_windows', {
                'start_offset_samples': 0, 'stop_offset_samples': None,
                'window_size_samples': 100, 'window_stride_samples': 100,
                'drop_last_window': False, 'mapping': None, 'preload': False,
                'drop_bad_windows': True, 'picks': picks, 'reject': reject,
                'flat': flat, 'targets_from': 'metadata', 'last_target_only': True,
                'on_missing': on_missing}),
            ('WindowsDataset', {
                'targets_from': 'metadata',
                'last_target_only': True,
            })
        ]


def test_window_sizes_from_events(concat_ds_targets):
    # no fixed window size, no offsets
    expected_n_samples = 1000
    concat_ds, targets = concat_ds_targets
    windows = create_windows_from_events(
        concat_ds=concat_ds,
        trial_start_offset_samples=0, trial_stop_offset_samples=0,
        drop_last_window=False)
    x, y, ind = windows[0]
    assert x.shape[-1] == ind[-1] - ind[-2]
    assert x.shape[-1] == expected_n_samples

    # no fixed window size, positive trial start offset
    expected_n_samples = 999
    concat_ds, targets = concat_ds_targets
    windows = create_windows_from_events(
        concat_ds=concat_ds,
        trial_start_offset_samples=1, trial_stop_offset_samples=0,
        drop_last_window=False)
    x, y, ind = windows[0]
    assert x.shape[-1] == ind[-1] - ind[-2]
    assert x.shape[-1] == expected_n_samples

    # no fixed window size, negative trial start offset
    expected_n_samples = 1001
    concat_ds, targets = concat_ds_targets
    windows = create_windows_from_events(
        concat_ds=concat_ds,
        trial_start_offset_samples=-1, trial_stop_offset_samples=0,
        drop_last_window=False)
    x, y, ind = windows[0]
    assert x.shape[-1] == ind[-1] - ind[-2]
    assert x.shape[-1] == expected_n_samples

    # no fixed window size, positive trial stop offset
    expected_n_samples = 1001
    concat_ds, targets = concat_ds_targets
    windows = create_windows_from_events(
        concat_ds=concat_ds,
        trial_start_offset_samples=0, trial_stop_offset_samples=1,
        drop_last_window=False)
    x, y, ind = windows[0]
    assert x.shape[-1] == ind[-1] - ind[-2]
    assert x.shape[-1] == expected_n_samples

    # no fixed window size, negative trial stop offset
    expected_n_samples = 999
    concat_ds, targets = concat_ds_targets
    windows = create_windows_from_events(
        concat_ds=concat_ds,
        trial_start_offset_samples=0, trial_stop_offset_samples=-1,
        drop_last_window=False)
    x, y, ind = windows[0]
    assert x.shape[-1] == ind[-1] - ind[-2]
    assert x.shape[-1] == expected_n_samples

    # fixed window size, trial offsets should not change window size
    expected_n_samples = 250
    concat_ds, targets = concat_ds_targets
    windows = create_windows_from_events(
        concat_ds=concat_ds,
        trial_start_offset_samples=3, trial_stop_offset_samples=8,
        window_size_samples=250, window_stride_samples=250,
        drop_last_window=False)
    x, y, ind = windows[0]
    assert x.shape[-1] == ind[-1] - ind[-2]
    assert x.shape[-1] == expected_n_samples


def test_window_sizes_too_large(concat_ds_targets):
    concat_ds, targets = concat_ds_targets
    # Window size larger than all trials
    window_size = len(concat_ds.datasets[0]) + 1
    with pytest.raises(
            ValueError, match=f'Window size {window_size} exceeds trial durat'):
        create_windows_from_events(
            concat_ds=concat_ds,
            window_size_samples=window_size,
            window_stride_samples=window_size,
            trial_start_offset_samples=0,
            trial_stop_offset_samples=0,
            drop_last_window=False,
        )

    with pytest.raises(
            ValueError, match=f'Window size {window_size} exceeds trial durat'):
        create_fixed_length_windows(
            concat_ds=concat_ds,
            window_size_samples=window_size,
            window_stride_samples=window_size,
            drop_last_window=False,
        )

    # Window size larger than one single trial
    annots = concat_ds.datasets[0].raw.annotations
    annot_0 = annots[0]
    # Window equal original trials size
    window_size = int(
        annot_0["duration"] * concat_ds.datasets[0].raw.info['sfreq'])

    # Make first trial 1 second shorter
    annot_0["duration"] -= 1

    # Replace first trial by a new shorter one
    annots.delete(0)
    del annot_0["orig_time"]
    annots.append(**annot_0)
    concat_ds.datasets[0].raw.set_annotations(annots)
    with pytest.warns(
            UserWarning,
            match=".* are being dropped as the window size .*"
    ):
        create_windows_from_events(
            concat_ds=concat_ds,
            window_size_samples=window_size,
            window_stride_samples=window_size,
            trial_start_offset_samples=0,
            trial_stop_offset_samples=0,
            drop_last_window=False,
            accepted_bads_ratio=0.5,
            on_missing='ignore'
        )


@pytest.fixture(scope="module")
def dataset_target_time_series():
    rng = np.random.RandomState(42)
    signal_sfreq = 50
    info = mne.create_info(ch_names=['0', '1', 'target_0', 'target_1'],
                           sfreq=signal_sfreq,
                           ch_types=['eeg', 'eeg', 'misc', 'misc'])
    signal = rng.randn(2, 1000)
    targets = np.full((2, 1000), np.nan)
    targets_sfreq = 10
    targets_stride = int(signal_sfreq / targets_sfreq)
    targets[:, ::targets_stride] = rng.randn(2, int(targets.shape[1] / targets_stride))

    raw = mne.io.RawArray(np.concatenate([signal, targets]), info=info)
    desc = pd.Series({'pathological': True, 'gender': 'M', 'age': 48})
    base_dataset = BaseDataset(raw, desc, target_name=None)
    concat_ds = BaseConcatDataset([base_dataset])
    windows_dataset = create_windows_from_target_channels(
        concat_ds,
        window_size_samples=100,
    )
    return concat_ds, windows_dataset, targets, signal


def test_windower_from_target_channels(dataset_target_time_series):
    _, windows_dataset, targets, signal = dataset_target_time_series
    assert len(windows_dataset) == 180
    for i in range(180):
        epoch, y, window_inds = windows_dataset[i]
        target_idx = i * 5 + 100
        np.testing.assert_array_almost_equal(targets[:, target_idx], y)
        np.testing.assert_array_almost_equal(signal[:, target_idx - 99: target_idx + 1], epoch)
        np.testing.assert_array_almost_equal(np.array([i, i*5, target_idx]), window_inds)


def test_windower_from_target_channels_all_targets(dataset_target_time_series):
    concat_ds, _, targets, signal = dataset_target_time_series
    windows_dataset = create_windows_from_target_channels(
        concat_ds,
        window_size_samples=100,
        last_target_only=False
    )
    assert len(windows_dataset) == 180
    for i in range(180):
        epoch, y, window_inds = windows_dataset[i]
        target_idx = i * 5 + 100
        np.testing.assert_array_almost_equal(targets[:, target_idx-99: target_idx + 1], y)
        np.testing.assert_array_almost_equal(signal[:, target_idx - 99: target_idx + 1], epoch)
        np.testing.assert_array_almost_equal(np.array([i, i*5, target_idx]), window_inds)
