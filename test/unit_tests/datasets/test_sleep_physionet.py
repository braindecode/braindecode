# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)
import pickle

import mne
import numpy as np
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


@pytest.mark.network
def test_sleep_physionet(sleep_class):

    assert isinstance(sleep_class, BaseConcatDataset)


@pytest.mark.network
def test_all_signals():
    sp = SleepPhysionet(
        subject_ids=[0], recording_ids=[1], preload=True, load_eeg_only=False
    )
    assert len(sp.datasets[0].raw.ch_names) == 7


@pytest.mark.network
def test_crop_wake():
    sp = SleepPhysionet(
        subject_ids=[0],
        recording_ids=[1],
        preload=True,
        load_eeg_only=True,
        crop_wake_mins=30,
    )
    raw = sp.datasets[0].raw
    sfreq = raw.info["sfreq"]
    sleep_event_inds = np.flatnonzero(
        [description[-1] in "1234R" for description in raw.annotations.description]
    )
    first_sleep = raw.annotations[sleep_event_inds[0]]
    last_sleep = raw.annotations[sleep_event_inds[-1]]
    crop_stop = raw.first_time + raw.n_times / sfreq

    assert first_sleep["onset"] - raw.first_time == pytest.approx(30 * 60, abs=1 / sfreq)
    assert crop_stop - last_sleep["onset"] - last_sleep["duration"] == pytest.approx(
        30 * 60, abs=1 / sfreq
    )


@pytest.mark.network
def test_serializable(sleep_class):
    """Make sure the object can be pickled. There used to be a bug (<=0.5.1)
    where the object couldn't be pickled because raw.exclude was a dict_keys
    object.
    """
    pickle.dumps(sleep_class)


@pytest.mark.network
def test_ch_names_orig_units_match():
    sp = SleepPhysionet(
        subject_ids=[0], recording_ids=[1], preload=True, load_eeg_only=True
    )
    assert all([ds.raw._orig_units.keys() == set(ds.raw.ch_names) for ds in sp.datasets])


@pytest.mark.parametrize(
    (
        "recording_duration",
        "first_sleep_onset",
        "last_sleep_onset",
        "last_sleep_duration",
        "expected_tmin",
        "expected_tmax",
    ),
    [
        pytest.param(400, 100, 200, 45, 40, 305, id="within-recording"),
        pytest.param(250, 20, 100, 30, 0, 190, id="clamp-start"),
        pytest.param(250, 100, 200, 30, 40, 250, id="clamp-stop"),
    ],
)
def test_crop_wake_keeps_full_last_annotation(
    monkeypatch,
    recording_duration,
    first_sleep_onset,
    last_sleep_onset,
    last_sleep_duration,
    expected_tmin,
    expected_tmax,
):
    sfreq = 100.0
    raw = mne.io.RawArray(
        np.zeros((1, int(recording_duration * sfreq))),
        mne.create_info(["EEG Fpz-Cz"], sfreq, ch_types="eeg"),
        verbose=False,
    )
    annotations = mne.Annotations(
        onset=[
            0.0,
            first_sleep_onset,
            last_sleep_onset,
            last_sleep_onset + last_sleep_duration,
        ],
        duration=[
            first_sleep_onset,
            30.0,
            last_sleep_duration,
            recording_duration - last_sleep_onset - last_sleep_duration,
        ],
        description=[
            "Sleep stage W",
            "Sleep stage 1",
            "Sleep stage R",
            "Sleep stage W",
        ],
    )
    crop_kwargs = {}
    original_crop = raw.crop

    def crop_spy(*args, **kwargs):
        crop_kwargs.update(kwargs)
        return original_crop(*args, **kwargs)

    monkeypatch.setattr(mne.io, "read_raw_edf", lambda *args, **kwargs: raw)
    monkeypatch.setattr(mne, "read_annotations", lambda *args, **kwargs: annotations)
    monkeypatch.setattr(raw, "crop", crop_spy)

    cropped, _ = SleepPhysionet._load_raw(
        "SC4001E0-PSG.edf",
        "SC4001EC-Hypnogram.edf",
        preload=True,
        crop_wake_mins=1,
    )

    assert cropped.first_samp == int(expected_tmin * sfreq)
    assert cropped.n_times == int((expected_tmax - expected_tmin) * sfreq)
    assert crop_kwargs["tmin"] == pytest.approx(expected_tmin)
    assert crop_kwargs["tmax"] == pytest.approx(expected_tmax)
    assert crop_kwargs.get("include_tmax") is False
