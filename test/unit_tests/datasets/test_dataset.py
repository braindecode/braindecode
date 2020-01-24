# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

import mne
import pytest
import numpy as np
import pandas as pd

from braindecode.datasets import WindowsDataset, BaseDataset


@pytest.fixture(scope="module")
def setUp():
    rng = np.random.RandomState(42)
    info = mne.create_info(ch_names=['0', '1'], sfreq=50, ch_types='eeg')
    raw = mne.io.RawArray(data=rng.randn(2, 1000), info=info)
    events = np.array([[100, 0, 1],
                       [200, 0, 2],
                       [300, 0, 1],
                       [400, 0, 4],
                       [500, 0, 3]])
    supercrop_inds = [(0, 0, 100),
                      (0, 100, 200),
                      (1, 0, 100),
                      (2, 0, 100),
                      (2, 50, 150)]
    i_supercrop_in_trial, i_start_in_trial, i_stop_in_trial = list(
        zip(*supercrop_inds))
    metadata = pd.DataFrame(
        {'sample': events[:, 0],
         'x': events[:, 1],
         'target': events[:, 2],
         'i_supercrop_in_trial': i_supercrop_in_trial,
         'i_start_in_trial': i_start_in_trial,
         'i_stop_in_trial': i_stop_in_trial})

    df = pd.DataFrame(zip([True], ["M"], [48]),
                      columns=["pathological", "gender", "age"])
    mne_epochs = mne.Epochs(raw=raw, events=events, metadata=metadata)
    epochs_data = mne_epochs.get_data()
    windows_dataset = WindowsDataset(mne_epochs, metadata)
    return epochs_data, windows_dataset, events, supercrop_inds, raw, metadata, df


def test_get_item(setUp):
    epochs_data, windows_dataset, events, supercrop_inds, raw, metadata, df = setUp
    for i in range(len(epochs_data)):
        x, y, inds = windows_dataset[i]
        np.testing.assert_allclose(epochs_data[i],
                                   np.squeeze(x.get_data(), axis=0))
        assert events[i, 2] == y, f'Y not equal for epoch {i}'
        np.testing.assert_array_equal(supercrop_inds[i], inds,
                                      f'Supercrop inds not equal for epoch {i}')

def test_len(setUp):
    epochs_data, windows_dataset, events, supercrop_inds, raw, metadata, df = setUp
    assert len(epochs_data) == len(windows_dataset)


def test_target_subject_info_is_none(setUp):
    epochs_data, windows_dataset, events, supercrop_inds, raw, metadata, df = setUp
    with pytest.raises(AssertionError, match="'is_none' not in info"):
        BaseDataset(raw, df, target='is_none')


def test_target_in_subject_info(setUp):
    epochs_data, windows_dataset, events, supercrop_inds, raw, metadata, df = setUp
    base_ds = BaseDataset(raw, df, target='age')
