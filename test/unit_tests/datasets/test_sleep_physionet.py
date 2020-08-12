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
