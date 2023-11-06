# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD-3

import os

import mne
import numpy as np
import h5py
import pytest
import torch

from braindecode.util import create_mne_dummy_raw, set_random_seeds


def test_create_mne_dummy_raw(tmp_path):
    n_channels, n_times, sfreq = 2, 10000, 100
    raw, fnames = create_mne_dummy_raw(
        n_channels, n_times, sfreq, savedir=tmp_path,
        save_format=['fif', 'hdf5'])

    assert isinstance(raw, mne.io.RawArray)
    assert len(raw.ch_names) == n_channels
    assert raw.n_times == n_times
    assert raw.info['sfreq'] == sfreq
    assert isinstance(fnames, dict)
    assert os.path.isfile(fnames['fif'])
    assert os.path.isfile(fnames['hdf5'])

    raw = mne.io.read_raw_fif(fnames['fif'], preload=False, verbose=None)
    with h5py.File(fnames['hdf5'], 'r') as hf:
        _ = np.array(hf['fake_raw'])


def test_set_random_seeds_raise_value_error():
    with pytest.raises(ValueError, match="cudnn_benchmark expected to be bool or None, got 'abc'"):
        set_random_seeds(100, True, "abc")


def test_set_random_seeds_warning():
    torch.backends.cudnn.benchmark = True
    with pytest.warns(UserWarning,
                      match="torch.backends.cudnn.benchmark was set to True which may results in "
                            "lack of reproducibility. In some cases to ensure reproducibility you "
                            "may need to set torch.backends.cudnn.benchmark to False."):
        set_random_seeds(100, True)
