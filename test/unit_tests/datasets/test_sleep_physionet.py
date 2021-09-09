# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)

import pickle

from braindecode.datasets.sleep_physionet import SleepPhysionet
from braindecode.datasets.base import BaseConcatDataset


def test_sleep_physionet():
    sp = SleepPhysionet(
        subject_ids=[0], recording_ids=[1], preload=True, load_eeg_only=True)
    assert isinstance(sp, BaseConcatDataset)


def test_all_signals():
    sp = SleepPhysionet(
        subject_ids=[0], recording_ids=[1], preload=True, load_eeg_only=False)
    assert len(sp.datasets[0].raw.ch_names) == 7


def test_crop_wake():
    sp = SleepPhysionet(
        subject_ids=[0], recording_ids=[1], preload=True, load_eeg_only=True,
        crop_wake_mins=30)
    sfreq = sp.datasets[0].raw.info['sfreq']
    duration_h = len(sp) / (3600 * sfreq)
    assert duration_h < 7 and duration_h > 6


def test_serializable():
    """Make sure the object can be pickled. There used to be a bug (<=0.5.1)
    where the object couldn't be pickled because raw.exclude was a dict_keys
    object.
    """
    sp = SleepPhysionet(
        subject_ids=[0], recording_ids=[1], preload=True, load_eeg_only=True)
    pickle.dumps(sp)


def test_ch_names_orig_units_match():
    sp = SleepPhysionet(
        subject_ids=[0], recording_ids=[1], preload=True, load_eeg_only=True)
    assert all([ds.raw._orig_units.keys() == set(ds.raw.ch_names)
                for ds in sp.datasets])
