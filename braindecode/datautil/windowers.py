"""Get epochs from mne.Raw
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#          David Sabbagh <dav.sabbagh@gmail.com>
#
# License: BSD (3-clause)

import copy

import numpy as np
import mne
import pandas as pd


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
        tmin=0,
        mapping=None,
    ):
        self.window_size_samples = window_size_samples
        self.overlap_size_samples = overlap_size_samples
        self.drop_last_samples = drop_last_samples
        self.tmin = tmin
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
        tmin=0,
    ):
        super().__init__(
            window_size_samples, overlap_size_samples, drop_last_samples, tmin
        )

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
        raw : mne.io.Raw
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
            tmin=self.tmin,
            tmax=(self.window_size_samples - 1) / fs,
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
    overlap_size_samples : int
        size of overlap for window in samples
    drop_last_samples : bool
        See BaseWindower
    """

    def __init__(
        self,
        window_size_samples,
        stride_samples,
        drop_last_samples,
        tmin=0,
        mapping=None,
    ):
        super().__init__(
            window_size_samples, None, drop_last_samples, tmin=tmin
        )
        self.chunk_duration_samples = stride_samples
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
            new_onset = onset + duration - (self.window_size_samples - 1) / fs
            if new_onset not in raw.annotations.onset:
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

        raw.annotations.duration += 1e-6  # see ToDo

        events, events_ids = mne.events_from_annotations(
            raw, mapping, chunk_duration=self.chunk_duration_samples / fs,
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
            tmin=self.tmin,
            tmax=(self.window_size_samples - 1) / fs,
            baseline=None,
            preload=False,
            metadata=metadata,
        )
