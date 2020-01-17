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
    ):
        self.window_size_samples = window_size_samples
        self.overlap_size_samples = overlap_size_samples
        self.drop_last_samples = drop_last_samples
        self.tmin = tmin

    def include_last_samples(self, raw):
        last_valid_window_start = raw.n_times - self.window_size_samples
        if events[-1, 0] < last_valid_window_start:
            events = np.concatenate(
                (events, [[last_valid_window_start, 0, id_holder]])
            )
        return events

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
            events = self.include_last_samples(raw)

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
        chunk_duration_samples,
        drop_last_samples,
        tmin=0,
        mapping=None,
    ):
        super().__init__(window_size_samples, None, drop_last_samples, tmin=tmin)
        self.chunk_duration_samples = chunk_duration_samples
        self.mapping_rev = {v: k for k, v in mapping.items()}

    def _include_last_samples(self, raw):
        onsets, durations, descriptions = list(), list(), list()
        for onset, duration, description in zip(
                raw.annotations.onset, 
                raw.annotations.duration, 
                raw.annotations.description):
            if onset + duration % (self.window_size_samples - 1) / fs > 0:
                onsets.append(onset)
                durations.append(duration)
                descriptions.append(description)

        raw.annotations = mne.Annotations(
            onsets, durations, descriptions, 
            orig_time=raw.annotations.orig_time)
        last_events, _ = mne.events_from_annotations(
            raw,
            self.mapping_rev,
            chunk_duration=None
        )

        return last_events

    def __call__(self, raw):
        """[summary]
        ToDo: id=1???
        ToDo: plus epsilon 1e-6 on duration; otherwise perfect fitting 
              chunk_durations will SOMETIMES(!!!) not fit

        Parameters
        ----------
        raw  : mne.io.Raw
            [description]
        """

        fs = raw.info["sfreq"]

        raw.annotations.duration += 1e-6  # see ToDo

        events, events_ids = mne.events_from_annotations(
            raw,
            self.mapping_rev,
            chunk_duration=self.chunk_duration_samples / fs,
        )

        if not self.drop_last_samples:
            raise NotImplementedError  # TO TEST!
    
            last_events = self._include_last_samples(raw)
            events = np.concatenate((events, last_events), axis=0)
            # Reorder events

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
