from braindecode.datasets.sleep_physionet import SleepPhysionet
from braindecode.datasets.base import BaseConcatDataset


def test_sleep_physionet():
    """
    Test if the given amount of the given amount.

    Args:
    """
    sp = SleepPhysionet(
        subject_ids=[0], recording_ids=[1], preload=True, load_eeg_only=True)
    assert isinstance(sp, BaseConcatDataset)


def test_all_signals():
    """
    Test for all signals.

    Args:
    """
    sp = SleepPhysionet(
        subject_ids=[0], recording_ids=[1], preload=True, load_eeg_only=False)
    assert len(sp.datasets[0].raw.ch_names) == 7


def test_crop_wake():
    """
    Test if a crop

    Args:
    """
    sp = SleepPhysionet(
        subject_ids=[0], recording_ids=[1], preload=True, load_eeg_only=True,
        crop_wake_mins=30)
    sfreq = sp.datasets[0].raw.info['sfreq']
    duration_h = len(sp) / (3600 * sfreq)
    assert duration_h < 7 and duration_h > 6
