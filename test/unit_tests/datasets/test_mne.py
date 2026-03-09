# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD-3


import mne
import numpy as np

from braindecode.datasets.mne import create_from_mne_epochs, create_from_mne_raw


def test_create_from_single_raw():
    n_channels = 50
    n_times = 500
    sfreq = 100

    rng = np.random.RandomState(34834)
    data = rng.rand(n_channels, n_times)
    ch_names = [f"ch{i}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)

    n_anns = 10
    inds = np.linspace(0, n_times, n_anns, endpoint=False).astype(int)
    onsets = raw.times[inds]
    durations = np.ones(n_anns) * 0.2
    descriptions = ["test_trial"] * len(durations)
    anns = mne.Annotations(onsets, durations, descriptions)
    raw = raw.set_annotations(anns)

    windows = create_from_mne_raw([raw], 0, 0, 5, 2, False)

    # windows per trial: 0-5,2-7,4-9,6-11,...,14-19,15-20
    assert len(windows) == 9 * n_anns
    for i_w, (x, y, (i_w_in_t, i_start, i_stop)) in enumerate(windows):
        assert i_w_in_t == i_w % 9
        i_t = i_w // 9
        assert i_start == inds[i_t] + i_w_in_t * 2 - (i_w_in_t == 8)
        assert i_stop == inds[i_t] + i_w_in_t * 2 - (i_w_in_t == 8) + 5
        np.testing.assert_allclose(x, data[:, i_start:i_stop], atol=1e-5, rtol=1e-5)


def test_create_from_two_raws_with_varying_trial_lengths():
    n_channels = 50
    n_times = 500
    sfreq = 100
    rng = np.random.RandomState(34834)
    raws = []
    datas = []
    for i_raw in range(2):
        data = rng.rand(n_channels, n_times)
        ch_names = [f"ch{i}" for i in range(n_channels)]
        ch_types = ["eeg"] * n_channels
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)

        n_anns = 10
        inds = np.linspace(0, n_times, n_anns, endpoint=False).astype(int)
        onsets = raw.times[inds]
        if i_raw == 0:
            trial_dur = 0.2  # in sec
        else:
            trial_dur = 0.1
        durations = np.ones(n_anns) * trial_dur
        descriptions = ["test_trial"] * len(durations)
        anns = mne.Annotations(onsets, durations, descriptions)
        raw = raw.set_annotations(anns)
        raws.append(raw)
        datas.append(data)

    windows = create_from_mne_raw(raws, 0, 0, 5, 2, False)

    # windows per trial: 0-5,2-7,4-9,6-11,...,14-19,15-20
    # and then: 0-5,2-7,4-9,5-10
    assert len(windows) == 9 * n_anns + 4 * n_anns
    for i_w, (x, y, (i_w_in_t, i_start, i_stop)) in enumerate(windows):
        if i_w < 9 * n_anns:
            assert i_w_in_t == i_w % 9
            i_t = i_w // 9
            assert i_start == inds[i_t] + i_w_in_t * 2 - (i_w_in_t == 8)
            assert i_stop == inds[i_t] + i_w_in_t * 2 - (i_w_in_t == 8) + 5
            np.testing.assert_allclose(
                x, datas[0][:, i_start:i_stop], atol=1e-5, rtol=1e-5
            )
        else:
            assert i_w_in_t == (i_w - n_anns * 9) % 4
            i_t = (i_w - n_anns * 9) // 4
            assert i_start == inds[i_t] + i_w_in_t * 2 - (i_w_in_t == 3)
            assert i_stop == inds[i_t] + i_w_in_t * 2 - (i_w_in_t == 3) + 5


def test_create_from_mne_epochs():
    n_channels = 50
    n_times = 500
    sfreq = 100
    rng = np.random.RandomState(34834)
    all_epochs = []
    datas = []
    for i_raw in range(2):
        data = rng.rand(n_channels, n_times)
        ch_names = [f"ch{i}" for i in range(n_channels)]
        ch_types = ["eeg"] * n_channels
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)

        n_anns = 10
        inds = np.linspace(0, n_times, n_anns, endpoint=False).astype(int)
        onsets = raw.times[inds]
        if i_raw == 0:
            trial_dur = 0.2  # in sec
        else:
            trial_dur = 0.1
        durations = np.ones(n_anns) * trial_dur
        descriptions = ["test_trial"] * len(durations)
        anns = mne.Annotations(onsets, durations, descriptions)
        raw = raw.set_annotations(anns)
        events, event_id = mne.events_from_annotations(
            raw,
        )
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            preload=True,
            baseline=None,
            tmin=0,
            tmax=trial_dur - 1e-2,
        )
        all_epochs.append(epochs)
        datas.append(data)

    windows = create_from_mne_epochs(
        all_epochs,
        window_size_samples=5,
        window_stride_samples=2,
        drop_last_window=False,
    )

    # windows per trial: 0-5,2-7,4-9,6-11,...,14-19,15-20
    # and then: 0-5,2-7,4-9,5-10
    assert len(windows) == 9 * n_anns + 4 * n_anns
    for i_w, (x, y, (i_w_in_t, i_start, i_stop)) in enumerate(windows):
        if i_w < 9 * n_anns:
            assert i_w_in_t == i_w % 9
            i_t = i_w // 9
            assert i_start == inds[i_t] + i_w_in_t * 2 - (i_w_in_t == 8)
            assert i_stop == inds[i_t] + i_w_in_t * 2 - (i_w_in_t == 8) + 5
            np.testing.assert_allclose(
                x, datas[0][:, i_start:i_stop], atol=1e-5, rtol=1e-5
            )
        else:
            assert i_w_in_t == (i_w - n_anns * 9) % 4
            i_t = (i_w - n_anns * 9) // 4
            assert i_start == inds[i_t] + i_w_in_t * 2 - (i_w_in_t == 3)
            assert i_stop == inds[i_t] + i_w_in_t * 2 - (i_w_in_t == 3) + 5


def test_create_from_mne_epochs_no_mapping():
    """without mapping, targets should be raw integer event codes."""
    sfreq = 100
    n_channels = 2
    n_times = 100
    info = mne.create_info(
        [f"ch{i}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    data = np.random.randn(2, n_channels, n_times)
    events = np.array([[0, 0, 1], [100, 0, 2]])
    epochs = mne.EpochsArray(data, info, events=events, tmin=0, baseline=None)

    windows_ds = create_from_mne_epochs(
        [epochs],
        window_size_samples=50,
        window_stride_samples=50,
        drop_last_window=True,
    )
    targets = [ds.windows.metadata["target"].iloc[0] for ds in windows_ds.datasets]
    assert 1 in targets
    assert 2 in targets


def test_create_from_mne_epochs_with_mapping():
    """mapping should correctly remap integer event codes to target values."""
    sfreq = 100
    n_channels = 2
    n_times = 100
    info = mne.create_info(
        [f"ch{i}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    data = np.random.randn(2, n_channels, n_times)
    events = np.array([[0, 0, 1], [100, 0, 2]])
    epochs = mne.EpochsArray(data, info, events=events, tmin=0, baseline=None)

    mapping = {1: 0, 2: 1}
    windows_ds = create_from_mne_epochs(
        [epochs],
        window_size_samples=50,
        window_stride_samples=50,
        drop_last_window=True,
        mapping=mapping,
    )
    targets = [ds.windows.metadata["target"].iloc[0] for ds in windows_ds.datasets]
    assert 0 in targets
    assert 1 in targets


def test_create_from_mne_epochs_with_picks():
    """picks should correctly subset channels."""
    sfreq = 100
    n_channels = 4
    n_times = 100
    info = mne.create_info(
        [f"ch{i}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    data = np.random.randn(2, n_channels, n_times)
    events = np.array([[0, 0, 1], [100, 0, 2]])
    epochs = mne.EpochsArray(data, info, events=events, tmin=0, baseline=None)

    windows_ds = create_from_mne_epochs(
        [epochs],
        window_size_samples=50,
        window_stride_samples=50,
        drop_last_window=True,
        picks=[0, 1],
    )
    assert windows_ds.datasets[0].windows.get_data().shape[1] == 2


def test_create_from_mne_epochs_with_preload():
    """preload=True should preload epoch data."""
    sfreq = 100
    n_channels = 2
    n_times = 100
    info = mne.create_info(
        [f"ch{i}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    data = np.random.randn(2, n_channels, n_times)
    events = np.array([[0, 0, 1], [100, 0, 2]])
    epochs = mne.EpochsArray(data, info, events=events, tmin=0, baseline=None)

    windows_ds = create_from_mne_epochs(
        [epochs],
        window_size_samples=50,
        window_stride_samples=50,
        drop_last_window=True,
        preload=True,
    )
    assert windows_ds.datasets[0].windows.preload is True
