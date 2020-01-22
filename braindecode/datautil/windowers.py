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
from ..util import round_list_to_int


class BaseWindower(object):
    """Fixed onset windower

    ToDo: samples or seconds

    Parameters
    ----------
    window_size_samples : int | None
        size of one window in samples
    overlap_size_samples : int
        size of overlap for window in samples
    drop_last_samples : bool
        Whether to drop the last samples when they don't fit into window. If
        True, there might be samples at the end of your data that is not used.
        If False, an additional window will be extracted to include the 
        remaining samples - this means this additional window will overlap with
        the previous window.

    """

    def __init__(
        self,
        window_size_samples,
        overlap_size_samples,
        drop_last_samples,
            trial_start_offset_samples=0,
        mapping=None,
    ):
        self.window_size_samples = window_size_samples
        self.overlap_size_samples = overlap_size_samples
        self.drop_last_samples = drop_last_samples
        self.trial_start_offset_samples = trial_start_offset_samples
        self.mapping = mapping

    def __call__(self, raw):
        return NotImplementedError


class FixedLengthWindower(BaseWindower):
    """Fixed onset windower

    ToDo: samples or seconds

    Parameters
    ----------
    window_size_samples : int | None
        size of one window in samples
    overlap_size_samples : int
        size of overlap for window in samples
    drop_last_samples : bool
       See BaseWindower 

    """

    def __init__(
        self,
        window_size_samples,
        overlap_size_samples,
        drop_last_samples,
        trial_start_offset_samples=0,
    ):
        super().__init__(window_size_samples, overlap_size_samples,
                         drop_last_samples, trial_start_offset_samples)

    def _include_last_samples(self, raw, events, id_holder):
        last_valid_window_start = raw.n_times - self.window_size_samples
        if events[-1, 0] < last_valid_window_start:
            events = np.concatenate(
                (events, [[last_valid_window_start, 0, id_holder]])
            )
        return events

    def __call__(self, raw):
        """[summary]
        ToDo: id=1???

        Parameters
        ----------
        raw : mne.io.Raw | mne.io.RawArray
            [description]
        """
        id_holder = 1
        fs = raw.info["sfreq"]

        if self.window_size_samples is None:
            self.window_size_samples = raw.n_times
            duration = raw.times[-1]
        else:
            duration = self.window_size_samples / fs

        overlap = self.overlap_size_samples / fs
        events = mne.make_fixed_length_events(
            raw, id=id_holder, duration=duration, overlap=overlap, stop=None
        )

        if not self.drop_last_samples:
            events = self._include_last_samples(raw, events, id_holder)

        return mne.Epochs(
            raw,
            events,
            tmin=self.trial_start_offset_samples / fs,
            tmax=(self.trial_start_offset_samples + (
                    self.window_size_samples - 1)) / fs,
            baseline=None,
            preload=False,
        )


class EventWindower(BaseWindower):
    """Fixed onset windower

    ToDo: id=1???
    ToDo: plus epsilon 1e-6 on duration; otherwise perfect fitting
          chunk_durations will SOMETIMES(!!!) not fit
    ToDo: debug case where drop_last_samples = False and window_size fits
          perfectly into trial length
    Parameters
    ----------
    window_size_samples : int | None
        size of one window in samples
    stride_samples : int
        size of overlap for window in samples
    drop_last_samples : bool
        See BaseWindower
    """

    def __init__(
        self,
        window_size_samples,
        stride_samples,
        drop_last_samples,
        trial_start_offset_samples=0,
        mapping=None,
    ):
        assert stride_samples > 0, "stride has to be larger than 0"
        super().__init__(window_size_samples, None, drop_last_samples,
                         trial_start_offset_samples)
        self.stride_samples = stride_samples
        if mapping is not None:
            assert all(
                [isinstance(v, int) for v in mapping.values()]
            ), "mapping dictionary must provided as {description: int}"

        self.mapping = mapping

    def _include_last_samples(self, raw, mapping):
        onsets, durations, descriptions = list(), list(), list()
        for onset, duration, description in zip(
            raw.annotations.onset,
            raw.annotations.duration,
            raw.annotations.description,
        ):
            fs = raw.info["sfreq"]
            new_onset = round_list_to_int(
                (onset + duration) * fs - self.window_size_samples)
            onsets_in_samples = round_list_to_int(
                [onset * fs for onset in raw.annotations.onset])
            if new_onset not in onsets_in_samples:
                onsets.append(new_onset)
                durations.append(duration)
                descriptions.append(description)

        raw.set_annotations(
            mne.Annotations(
                onsets,
                durations,
                descriptions,
                orig_time=raw.annotations.orig_time,
            )
        )
        last_events, _ = mne.events_from_annotations(
            raw, mapping, chunk_duration=None
        )
        return last_events

    def __call__(self, raw, mapping=None):
        """[summary]

        Parameters
        ----------
        raw  : mne.io.Raw
            [description]
        """
        if mapping is None:
            mapping = self.mapping
        fs = raw.info["sfreq"]

        #raw.annotations.duration += 1e-6  # see ToDo

        events, events_ids = mne.events_from_annotations(
            raw, mapping, chunk_duration=self.stride_samples / fs,
        )
        if not self.drop_last_samples:
            last_events = self._include_last_samples(raw, mapping)
            events = np.concatenate((events, last_events), axis=0)
            # Reorder events
            events = events[np.argsort(events[:, 0], axis=0)]

        metadata = {
            "event_onset_idx": events[:, 0],
            "trial_number": range(len(events)),
            "target": events[:, -1],
        }
        metadata = pd.DataFrame(metadata)
        # metadata['subject'] =

        return mne.Epochs(
            raw,
            events,
            tmin=self.trial_start_offset_samples / fs,
            tmax=(self.window_size_samples - 1) / fs,
            baseline=None,
            preload=False,
            metadata=metadata,
        )


class Windower(object):
    """
    A windower that creates a mne Epochs objects. Therefore, it fits supercrops
    of supercrop_size_samples in trial_start_offset_samples to
    trial_stop_offset_samples separated by supercrop_stride_samples. If the last
    supercrop does not end at trial_stop_offset_samples, creates another
    overlapping supercrop that ends at trial_stop_offset_samples if drop_samples
    is set to False.

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
    mapping: dict(str: int)
        mapping from event description to target value
    """
    def __init__(self, trial_start_offset_samples, trial_stop_offset_samples,
                 supercrop_size_samples, supercrop_stride_samples,
                 drop_samples=False, mapping=None, ignore_events=False):
        self.start_offset = trial_start_offset_samples
        self.stop_offset = trial_stop_offset_samples
        assert supercrop_size_samples > 0, (
            "supercrop size has to be larger than 0")
        self.size = supercrop_size_samples
        assert supercrop_stride_samples > 0, (
            "supercrop stride has to be larger than 0")
        self.stride = supercrop_stride_samples
        self.drop_samples = drop_samples
        self.mapping = mapping
        self.ignore_events = ignore_events

    # TODO: handle case we don't get a raw
    def __call__(self, raw):
        fs = raw.info["sfreq"]
        # FixedLengthWindower
        if self.ignore_events:
            # fake event onsets by equally slicing the signal
            # already includes last incomplete supercrop start
            starts = np.arange(0, raw.n_times, self.stride)
            if self.drop_samples:
                starts = starts[:-1]
            else:
                # if last supercrop does not end at trial stop, make it stop
                # there
                if starts[-1] != raw.n_times - self.size:
                    starts[-1] = raw.n_times - self.size 

            # TODO: Description should be the target of the entire recording
            # TODO: How do we know this?
            # TODO: get it from mapping? or from raw.info where we populated
            # TODO: some information?
            description = 0
            events = [[start, self.size, description]
                      for i_start, start in enumerate(starts)]
        else:
            # EventWindower: if annotations are given in raw, use them to get
            # onsets
            assert (hasattr(raw, "annotations") and
                    hasattr(raw.annotations, "onset") and
                    len(raw.annotations.onset) > 0)
            onsets = round_list_to_int(raw.annotations.onset*fs)
            i_trials, starts, stops = _supercrop_starts(
                onsets, self.start_offset, self.stop_offset, self.size,
                self.stride, self.drop_samples)
            description = raw.annotations.description
            if self.mapping is not None:
                description = [self.mapping[d] for d in description]

            # create events for supercrops as supercrop start, supercrop size
            # and supercrop description
            events = [[start, self.size, description[i_trials[i_start]]]
                      for i_start, start in enumerate(starts)]

        assert (np.diff(np.array(events)[:,0]) > 0).all(), (
            "trials overlap not implemented")
        # supercrop size - 1, since tmax is inclusive
        return mne.Epochs(raw, events, tmin=0, tmax=(self.size-1)/fs,
                          baseline=None)

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
    starts: list
        valid supercrop starts
    """
    # trial ends are defined by trial starts (onsets maybe shifted by offset)
    # and end
    stops = onsets + stop_offset
    i_trials, starts = [], []
    for onset_i, onset in enumerate(onsets):
        # between original trial onsets (shifted by start_offset) and stops,
        # generate possible supercrop starts with given stride
        possible_starts = np.arange(
            onset+start_offset, onset+stop_offset, stride)

        # possible supercrop start is actually a start, if supercrop size fits
        # in trial start and stop
        for s in possible_starts:
            if (s + size) <= stops[onset_i]:
                starts.append(s)
                i_trials.append(onset_i)

        # if the last supercrop start + supercrop size is not the same as
        # onset + stop_offset, create another supercrop that overlaps and stops
        # at onset + stop_offset
        if not drop_samples:
            if starts[-1] + size != onset + stop_offset:
                starts.append(onset + stop_offset - size)
                i_trials.append(onset_i)

    # update stops to now be event stops instead of trial stops
    stops = np.array(starts) + size
    assert len(i_trials) == len(starts) == len(stops)
    return i_trials, np.array(starts), np.array(stops)
