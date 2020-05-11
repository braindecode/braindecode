# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

import mne
import numpy as np
import pandas as pd
import pytest

from braindecode.datasets import WindowsDataset, BaseDataset, BaseConcatDataset
from braindecode.datasets.moabb import fetch_data_with_moabb


@pytest.fixture(scope="module")
# TODO: add test for transformers and case when subject_info is used
def set_up():
    rng = np.random.RandomState(42)
    info = mne.create_info(ch_names=['0', '1'], sfreq=50, ch_types='eeg')
    raw = mne.io.RawArray(data=rng.randn(2, 1000), info=info)
    desc = pd.Series({'pathological': True, 'gender': 'M', 'age': 48})
    base_dataset = BaseDataset(raw, desc, target_name='age')

    events = np.array([[100, 0, 1],
                       [200, 0, 2],
                       [300, 0, 1],
                       [400, 0, 4],
                       [500, 0, 3]])
    window_idxs = [(0, 0, 100),
                      (0, 100, 200),
                      (1, 0, 100),
                      (2, 0, 100),
                      (2, 50, 150)]
    i_window_in_trial, i_start_in_trial, i_stop_in_trial = list(
        zip(*window_idxs))
    metadata = pd.DataFrame(
        {'sample': events[:, 0],
         'x': events[:, 1],
         'target': events[:, 2],
         'i_window_in_trial': i_window_in_trial,
         'i_start_in_trial': i_start_in_trial,
         'i_stop_in_trial': i_stop_in_trial})

    mne_epochs = mne.Epochs(raw=raw, events=events, metadata=metadata)
    windows_dataset = WindowsDataset(mne_epochs, desc)

    return raw, base_dataset, mne_epochs, windows_dataset, events, window_idxs


@pytest.fixture(scope="module")
def concat_ds_targets():
    raws, description = fetch_data_with_moabb(
        dataset_name="BNCI2014001", subject_ids=4)
    events, _ = mne.events_from_annotations(raws[0])
    targets = events[:, -1] - 1
    ds = [BaseDataset(raws[i], description.iloc[i]) for i in range(3)]
    concat_ds = BaseConcatDataset(ds)
    return concat_ds, targets


def test_get_item(set_up):
    _, _, mne_epochs, windows_dataset, events, window_idxs  = set_up
    for i, epoch in enumerate(mne_epochs.get_data()):
        x, y, inds = windows_dataset[i]
        np.testing.assert_allclose(epoch, x)
        assert events[i, 2] == y, f'Y not equal for epoch {i}'
        np.testing.assert_array_equal(window_idxs[i], inds,
                                      f'window inds not equal for epoch {i}')


def test_len_windows_dataset(set_up):
    _, _, mne_epochs, windows_dataset, _, _  = set_up
    assert len(mne_epochs.events) == len(windows_dataset)


def test_len_base_dataset(set_up):
    raw, base_dataset, _, _, _, _ = set_up
    assert len(raw) == len(base_dataset)


def test_len_concat_dataset(concat_ds_targets):
    concat_ds = concat_ds_targets[0]
    assert len(concat_ds) == sum([len(c) for c in concat_ds.datasets])


def test_target_in_subject_info(set_up):
    raw, _, _, _, _, _ = set_up
    desc = pd.Series({'pathological': True, 'gender': 'M', 'age': 48})
    with pytest.raises(ValueError, match="'does_not_exist' not in description"):
        BaseDataset(raw, desc, target_name='does_not_exist')


def test_description_concat_dataset(concat_ds_targets):
    concat_ds = concat_ds_targets[0]
    assert isinstance(concat_ds.description, pd.DataFrame)
    assert concat_ds.description.shape[0] == len(concat_ds.datasets)


def test_split_concat_dataset(concat_ds_targets):
    concat_ds = concat_ds_targets[0]
    splits = concat_ds.split('run')

    for k, v in splits.items():
        assert k == v.description['run'].values
        assert isinstance(v, BaseConcatDataset)

    assert len(concat_ds) == sum([len(v) for v in splits.values()])
