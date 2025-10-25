# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)
import pickle

import pytest

from braindecode.datasets.base import BaseConcatDataset
from braindecode.datasets.sleep_physio_challe_18 import (
    SleepPhysionetChallenge2018 as PC18,
)
from braindecode.datasets.sleep_physionet import SleepPhysionet


@pytest.fixture(params=[SleepPhysionet])
def sleep_class(request):
    if request.param == SleepPhysionet:
        sleep_obj = SleepPhysionet(
            subject_ids=[0], load_eeg_only=True,
            recording_ids=[1], preload=True,
        )
    else:
        sleep_obj = PC18(
            subject_ids=[0], load_eeg_only=True,
        )
    return sleep_obj


def test_sleep_physionet(sleep_class):

    assert isinstance(sleep_class, BaseConcatDataset)


def test_all_signals():
    sp = SleepPhysionet(
        subject_ids=[0], recording_ids=[1], preload=True, load_eeg_only=False
    )
    assert len(sp.datasets[0].raw.ch_names) == 7


def test_crop_wake():
    sp = SleepPhysionet(
        subject_ids=[0],
        recording_ids=[1],
        preload=True,
        load_eeg_only=True,
        crop_wake_mins=30,
    )
    sfreq = sp.datasets[0].raw.info["sfreq"]
    duration_h = len(sp) / (3600 * sfreq)
    assert duration_h < 7 and duration_h > 6


def test_serializable(sleep_class):
    """Make sure the object can be pickled. There used to be a bug (<=0.5.1)
    where the object couldn't be pickled because raw.exclude was a dict_keys
    object.
    """
    pickle.dumps(sleep_class)


def test_ch_names_orig_units_match():
    sp = SleepPhysionet(
        subject_ids=[0], recording_ids=[1], preload=True, load_eeg_only=True
    )
    assert all([ds.raw._orig_units.keys() == set(ds.raw.ch_names) for ds in sp.datasets])
