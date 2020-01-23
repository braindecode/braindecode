# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#
# License: BSD (3-clause)

import mne
import numpy as np
import pandas as pd
import pytest

from braindecode.datasets import WindowsDataset


# TODO: add test for transformers and case when subject_info is used
@pytest.fixture(scope="module")
def epochs_events_supercrop_dataset():
    rng = np.random.RandomState(42)
    info = mne.create_info(ch_names=['0', '1'],
                           sfreq=50, ch_types='eeg')
    raw = mne.io.RawArray(data=rng.randn(2, 1000), info=info)
    events = np.array([[100, 0, 1],
                       [200, 0, 2],
                       [300, 0, 1],
                       [400, 0, 4],
                       [500, 0, 3]])
    supercrop_idxs = [(0, 0, 100),
                      (0, 100, 200),
                      (1, 0, 100),
                      (2, 0, 100),
                      (2, 50, 150)]
    metadata = pd.DataFrame(
        {'sample': events[:, 0],
         'x': events[:, 1],
         'target': events[:, 2],
         'supercrop_inds': supercrop_idxs,
         'subject_info': None})
    mne_epochs = mne.Epochs(raw=raw, events=events, metadata=metadata)

    windows_dataset = WindowsDataset(mne_epochs, target="target")
    return mne_epochs, events, supercrop_idxs, windows_dataset


def test_get_item(epochs_events_supercrop_dataset):
    mne_epochs, events, supercrop_idxs, windows_dataset = epochs_events_supercrop_dataset
    epochs_data = mne_epochs.get_data()
    for i in range(len(epochs_data)):
        x, y, inds = windows_dataset[i]
        np.testing.assert_allclose(epochs_data[i], x)
        assert events[i, 2] == y, f'Y not equal for epoch {i}'
        assert supercrop_idxs[i] == inds, f'Supercrop inds not equal for epoch {i}'


def test_len(epochs_events_supercrop_dataset):
    mne_epochs, _, _, windows_dataset = epochs_events_supercrop_dataset
    assert len(mne_epochs.get_data()) == len(windows_dataset)


def test_target_subject_info_is_none(epochs_events_supercrop_dataset):
    mne_epochs = epochs_events_supercrop_dataset[0]
    with pytest.raises(AssertionError):
        WindowsDataset(mne_epochs, target='is_none')


def test_target_in_subject_info(epochs_events_supercrop_dataset):
    mne_epochs = epochs_events_supercrop_dataset[0]
    with pytest.raises(AssertionError):
        WindowsDataset(mne_epochs, target='does_not_exist')
