# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD-3

import os

import mne
import numpy as np
import h5py

from braindecode.util import create_mne_raw


def test_create_mne_raw(tmp_path):
    n_channels, n_times, sfreq = 2, 10000, 100
    raw, fnames = create_mne_raw(
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
    with h5py.File(fnames['hdf5']) as hf:
        _ = np.array(hf['fake_raw'])
