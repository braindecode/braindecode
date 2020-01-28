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


class Windower(object):
    """
    A windower that creates a mne Epochs objects.
    """
    def __init__(self, trial_start_offset_samples, trial_stop_offset_samples,
                 supercrop_size_samples, supercrop_stride_samples,
                 drop_samples=False, mapping=None):
        self.trial_start_offset_samples = trial_start_offset_samples
        self.trial_stop_offset_samples = trial_stop_offset_samples
        assert supercrop_size_samples > 0, (
            "supercrop size has to be larger than 0")
        self.size = supercrop_size_samples
        assert supercrop_stride_samples > 0, (
            "supercrop stride has to be larger than 0")
        self.stride = supercrop_stride_samples
        self.drop_samples = drop_samples
        self.mapping = mapping
        # TODO: assert values are integers
        # TODO: assert start < stop

    # TODO: handle case we don't get a raw
    def __call__(self, raw, events, metadata):
        # supercrop size - 1, since tmax is inclusive
        return mne.Epochs(raw, events, baseline=None,
                          tmin=0, tmax=(self.size-1)/raw.info["sfreq"],
                          metadata=metadata)


class EventWindower(Windower):
    """
    A Windower that creates supercrops/windows based on events in mne.Raw.
    Therefore, it fits supercrops of supercrop_size_samples in
    trial_start_offset_samples to trial_stop_offset_samples separated by
    supercrop_stride_samples. If the last supercrop does not end
    at trial_stop_offset_samples, creates another overlapping supercrop that
    ends at trial_stop_offset_samples if drop_samples is set to False.

    in mne: tmin (s)                    trial onset        onset + duration (s)
    trial:  |--------------------------------|--------------------------------|
    here:   trial_start_offset_samples                trial_stop_offset_samples

    Parameters
    ----------
    trial_start_offset_samples: int
        start offset from original trial onsets in samples
    trial_stop_offset_samples: int
        stop offset from original trial onsets in samples
    supercrop_size_samples: int
        supercrop size
    supercrop_stride_samples: int
        stride between supercrops
    drop_samples: bool
        whether or not have a last overlapping supercrop/window, when
        supercrops/windows do not equally devide the continuous signal
    mapping: dict(str: int)
        mapping from event description to target value
    """
    def __call__(self, base_ds):
        events = mne.find_events(base_ds.raw)
        onsets = events[:, 0]
        description = events[:, -1]
        i_trials, i_supercrop_in_trials, starts, stops = _supercrop_starts(
            onsets, self.trial_start_offset_samples,
            self.trial_stop_offset_samples, self.size, self.stride,
            self.drop_samples)
        events = [[start, self.size, description[i_trials[i_start]]]
                  for i_start, start in enumerate(starts)]
        events = np.array(events)
        assert (np.diff(events[:,0]) > 0).all(), (
            "trials overlap not implemented")
        description = events[:, -1]
        if self.mapping is not None:
            # Apply remapping of targets
            description = np.array([self.mapping[d] for d in description])
            events[:, -1] = description

        metadata = pd.DataFrame(
            zip(i_supercrop_in_trials, starts, stops, description),
            columns=["i_supercrop_in_trial", "i_start_in_trial",
                     "i_stop_in_trial", "target"])

        return super().__call__(base_ds.raw, events, metadata=metadata)


class FixedLengthWindower(Windower):
    """
    A Windower that creates supercrops/windows based on fake events that equally
    divide the continuous signal.

    Parameters
    ----------
    trial_start_offset_samples: int
        start offset from original trial onsets in samples
    trial_stop_offset_samples: int
        stop offset from original trial onsets in samples
    supercrop_size_samples: int
        supercrop size
    supercrop_stride_samples: int
        stride between supercrops
    drop_samples: bool
        whether or not have a last overlapping supercrop/window, when
        supercrops/windows do not equally devide the continuous signal
    mapping: dict(str: int)
        mapping from event description to target value
    """
    def __call__(self, base_ds):
        # already includes last incomplete supercrop start
        starts = np.arange(0, base_ds.raw.n_times, self.stride)
        # 1/0
        if self.drop_samples:
            starts = starts[:-1]
        else:
            # if last supercrop does not end at trial stop, make it stop
            # there
            if starts[-1] != base_ds.raw.n_times - self.size:
                starts[-1] = base_ds.raw.n_times - self.size

        # TODO: handle multi-target case
        assert len(base_ds.info[base_ds.target]) == 1, (
            "multi-target not supported")
        description = base_ds.info[base_ds.target].iloc[0]
        # https://github.com/numpy/numpy/issues/2951
        if not isinstance(description, np.integer):
            assert self.mapping is not None, (
                f"a mapping from '{description}' to int is required")
            description = self.mapping[description]
        events = [[start, self.size, description]
                  for i_start, start in enumerate(starts)]
        metadata = pd.DataFrame(
            zip(np.arange(len(events)), starts, starts + self.size,
                len(events) *[description]),
            columns=["i_supercrop_in_trial", "i_start_in_trial",
                     "i_stop_in_trial", "target"])
        return super().__call__(base_ds.raw, events, metadata=metadata)


def _supercrop_starts(onsets, start_offset, stop_offset, size, stride,
                      drop_samples=False):
    """
    Create supercrop starts from trial onsets (shifted by offset) to trial end
    separated by stride as long as supercrop size fits into trial

    Parameters
    ----------
    onsets: array-like
        trial onsets in samples
    start_offset: int
        start offset from original trial onsets in samples
    stop_offset: int
        stop offset from original trial onsets in samples
    size: int
        supercrop size
    stride: int
        stride between supercrops
    drop_samples: bool
        toggles of shifting last supercrop within range or dropping last samples

    Returns
    -------
    """
    # trial ends are defined by trial starts (onsets maybe shifted by offset)
    # and end
    stops = onsets + stop_offset
    i_supercrop_in_trials, i_trials, starts = [], [], []
    for onset_i, onset in enumerate(onsets):
        # between original trial onsets (shifted by start_offset) and stops,
        # generate possible supercrop starts with given stride
        possible_starts = np.arange(
            onset+start_offset, onset+stop_offset, stride)

        # possible supercrop start is actually a start, if supercrop size fits
        # in trial start and stop
        for i_supercrop, s in enumerate(possible_starts):
            if (s + size) <= stops[onset_i]:
                starts.append(s)
                i_supercrop_in_trials.append(i_supercrop)
                i_trials.append(onset_i)

        # if the last supercrop start + supercrop size is not the same as
        # onset + stop_offset, create another supercrop that overlaps and stops
        # at onset + stop_offset
        if not drop_samples:
            if starts[-1] + size != onset + stop_offset:
                starts.append(onset + stop_offset - size)
                i_supercrop_in_trials.append(i_supercrop_in_trials[-1] + 1)
                i_trials.append(onset_i)

    # update stops to now be event stops instead of trial stops
    stops = np.array(starts) + size
    assert len(i_supercrop_in_trials) == len(starts) == len(stops)
    return i_trials, i_supercrop_in_trials, starts, stops
