"""Get epochs from mne.Raw
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#          David Sabbagh <dav.sabbagh@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import mne
import pandas as pd

from ..datasets.base import WindowsDataset, BaseConcatDataset


def create_windows_from_events(
        concat_ds, trial_start_offset_samples, trial_stop_offset_samples,
        supercrop_size_samples, supercrop_stride_samples, drop_samples,
        mapping=None):
    """Windower that creates supercrops/windows based on events in mne.Raw.

    The function fits supercrops of supercrop_size_samples in
    trial_start_offset_samples to trial_stop_offset_samples separated by
    supercrop_stride_samples. If the last supercrop does not end
    at trial_stop_offset_samples, it creates another overlapping supercrop that
    ends at trial_stop_offset_samples if drop_samples is set to False.

    in mne: tmin (s)                    trial onset        onset + duration (s)
    trial:  |--------------------------------|--------------------------------|
    here:   trial_start_offset_samples                trial_stop_offset_samples

    Parameters
    ----------
    concat_ds: BaseConcatDataset
        a concat of base datasets each holding raw and description
    trial_start_offset_samples: int
        start offset from original trial onsets in samples
    trial_stop_offset_samples: int
        stop offset from original trial stop in samples
    supercrop_size_samples: int
        supercrop size
    supercrop_stride_samples: int
        stride between supercrops
    drop_samples: bool
        whether or not have a last overlapping supercrop/window, when
        supercrops/windows do not equally divide the continuous signal
    mapping: dict(str: int)
        mapping from event description to target value

    Returns
    -------
    windows_ds: WindowsDataset
        Dataset containing the extracted windows.
    """
    _check_windowing_arguments(
        trial_start_offset_samples, trial_stop_offset_samples,
        supercrop_size_samples, supercrop_stride_samples)

    list_of_windows_ds = []
    for ds in concat_ds.datasets:
        if mapping is None:
            # mapping event descriptions to integers from 0 on
            mapping = {v: k for k, v in enumerate(
                np.unique(ds.raw.annotations.description))}

        events, _ = mne.events_from_annotations(ds.raw, mapping)
        onsets = events[:, 0]
        stops = onsets + (ds.raw.annotations.duration
                          * ds.raw.info['sfreq']).astype(int)

        if stops[-1] + trial_stop_offset_samples > len(ds):
            raise ValueError('"trial_stop_offset_samples" too large. Stop of '
                             f'last trial ({stops[-1]}) + '
                             f'"trial_stop_offset_samples" '
                             f'({trial_stop_offset_samples}) must be smaller '
                             f'then length of recording {len(ds)}.')

        description = events[:, -1]
        i_trials, i_supercrop_in_trials, starts, stops = _compute_supercrop_inds(
            onsets, stops, trial_start_offset_samples,
            trial_stop_offset_samples, supercrop_size_samples,
            supercrop_stride_samples, drop_samples)

        events = [[start, supercrop_size_samples, description[i_trials[i_start]]]
                   for i_start, start in enumerate(starts)]
        events = np.array(events)

        if any(np.diff(events[:, 0]) <= 0):
            raise NotImplementedError('Trial overlap not implemented.')

        description = events[:, -1]

        metadata = pd.DataFrame(
            zip(i_supercrop_in_trials, starts, stops, description),
            columns=["i_supercrop_in_trial", "i_start_in_trial",
                     "i_stop_in_trial", "target"])

        # supercrop size - 1, since tmax is inclusive
        mne_epochs = mne.Epochs(
            ds.raw, events, mapping ,baseline=None, tmin=0,
            tmax=(supercrop_size_samples - 1) / ds.raw.info["sfreq"],
            metadata=metadata)
        windows_ds = WindowsDataset(mne_epochs, ds.description)
        list_of_windows_ds.append(windows_ds)

    return BaseConcatDataset(list_of_windows_ds)


def create_fixed_length_windows(
        concat_ds, start_offset_samples, stop_offset_samples,
        supercrop_size_samples, supercrop_stride_samples, drop_samples,
        mapping=None):
    """Windower that creates sliding supercrops/windows.

    Parameters
    ----------
    concat_ds: ConcatDataset
        a concat of base datasets each holding raw and descpription
    start_offset_samples: int
        start offset from beginning of recording in samples
    stop_offset_samples: int
        stop offset from beginning of recording in samples.
    supercrop_size_samples: int
        supercrop size
    supercrop_stride_samples: int
        stride between supercrops
    drop_samples: bool
        whether or not have a last overlapping supercrop/window, when
        supercrops/windows do not equally divide the continuous signal
    mapping: dict(str: int)
        mapping from event description to target value

    Returns
    -------
    windows_ds: WindowsDataset
        Dataset containing the extracted windows.
    """
    _check_windowing_arguments(
        start_offset_samples, stop_offset_samples,
        supercrop_size_samples, supercrop_stride_samples)

    list_of_windows_ds = []
    for ds in concat_ds.datasets:
        # already includes last incomplete supercrop start
        stop = (ds.raw.n_times
                if stop_offset_samples is None
                else stop_offset_samples)
        starts = np.arange(
            start_offset_samples, stop, supercrop_stride_samples)

        last_allowed_ind = stop - supercrop_size_samples

        if starts[-1] > last_allowed_ind:
            starts = starts[:-1]
        if not drop_samples:
            # if last supercrop does not end at trial stop, make it stop
            # there
            if starts[-1] < last_allowed_ind:
                starts[-1] = last_allowed_ind

        # TODO: handle multi-target case / non-integer target case
        target = -1 if ds.target is None else ds.target
        if mapping is not None:
            target = mapping[target]
        if not isinstance(target, (np.integer, int)):
           raise ValueError(f"Mapping from '{target}' to int is required")

        fake_events = [[start, supercrop_size_samples, target]
                        for i_start, start in enumerate(starts)]
        metadata = pd.DataFrame(zip(
            np.arange(len(fake_events)), starts, starts + supercrop_size_samples,
            len(fake_events) * [target]),
            columns=["i_supercrop_in_trial", "i_start_in_trial",
                     "i_stop_in_trial", "target"])

        # supercrop size - 1, since tmax is inclusive
        mne_epochs =  mne.Epochs(
            ds.raw, fake_events, baseline=None,
            tmin=0, tmax=(supercrop_size_samples - 1) / ds.raw.info["sfreq"],
            metadata=metadata)
        windows_ds = WindowsDataset(mne_epochs, ds.description)
        list_of_windows_ds.append(windows_ds)

    return BaseConcatDataset(list_of_windows_ds)


def _compute_supercrop_inds(
        starts, stops, start_offset, stop_offset, size, stride, drop_samples):
    """Create supercrop starts from trial onsets (shifted by offset) to trial
    end separated by stride as long as supercrop size fits into trial

    Parameters
    ----------
    starts: array-like
        trial starts in samples
    stops: array-like
        trial stops in samples
    start_offset: int
        start offset from original trial onsets in samples
    stop_offset: int
        stop offset from original trial stop in samples
    size: int
        supercrop size
    stride: int
        stride between supercrops
    drop_samples: bool
        toggles of shifting last supercrop within range or dropping last samples

    Returns
    -------
    result_lists: (list, list, list, list)
        trial, i_supercrop_in_trial, start sample and stop sample of supercrops
    """

    starts = np.array([starts]) if isinstance(starts, int) else starts
    stops = np.array([stops]) if isinstance(stops, int) else stops

    starts += start_offset
    stops += stop_offset

    i_supercrop_in_trials, i_trials, supercrop_starts = [], [], []
    for start_i, (start, stop) in enumerate(zip(starts, stops)):
        # between original trial onsets (shifted by start_offset) and stops,
        # generate possible supercrop starts with given stride
        possible_starts = np.arange(
            start, stop, stride)

        # possible supercrop start is actually a start, if supercrop size fits
        # in trial start and stop
        for i_supercrop, s in enumerate(possible_starts):
            if (s + size) <= stop:
                supercrop_starts.append(s)
                i_supercrop_in_trials.append(i_supercrop)
                i_trials.append(start_i)

        # if the last supercrop start + supercrop size is not the same as
        # stop + stop_offset, create another supercrop that overlaps and stops
        # at onset + stop_offset
        if not drop_samples:
            if supercrop_starts[-1] + size != stop:
                supercrop_starts.append(stop - size)
                i_supercrop_in_trials.append(i_supercrop_in_trials[-1] + 1)
                i_trials.append(start_i)

    # update stops to now be event stops instead of trial stops
    supercrop_stops = np.array(supercrop_starts) + size
    if not (len(i_supercrop_in_trials) == len(supercrop_starts) ==
            len(supercrop_stops)):
        raise ValueError(f'{len(i_supercrop_in_trials)} == '
                         f'{len(supercrop_starts)} == {len(supercrop_stops)}')
    return i_trials, i_supercrop_in_trials, supercrop_starts, supercrop_stops


def _check_windowing_arguments(
        trial_start_offset_samples, trial_stop_offset_samples,
        supercrop_size_samples, supercrop_stride_samples):
    assert supercrop_size_samples > 0, (
        "supercrop size has to be larger than 0")
    assert supercrop_stride_samples > 0, (
        "supercrop stride has to be larger than 0")
    assert isinstance(trial_start_offset_samples, (int, np.integer))
    if trial_stop_offset_samples is not None:
        assert isinstance(trial_stop_offset_samples, (int, np.integer))
    assert isinstance(supercrop_size_samples, (int, np.integer))
    assert isinstance(supercrop_stride_samples, (int, np.integer))
