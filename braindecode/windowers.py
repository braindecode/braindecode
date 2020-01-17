"""Get epochs from mne.Raw
"""

import numpy as np
from sklearn.base import TransformerMixin
import mne
import pandas as pd


class BaseWindower(TransformerMixin):
    """Fixed onset windower
    ToDo: samples or seconds

    Parameters
    ----------
    window_size_samples : int | None
        size of one window in samples
    overlap_size_samples : int
        size of overlap for window in samples
    drop_last_samples : bool, default=True
        whether to drop the last samples when they don't fit into window

    """

    def __init__(self, window_size_samples, overlap_size_samples,
                 drop_last_samples=True, tmin=0, mapping=None):
        self.window_size_samples = window_size_samples
        self.overlap_size_samples = overlap_size_samples
        self.drop_last_samples = drop_last_samples
        self.tmin = tmin
        self.mapping = mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
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
    drop_last_samples : bool, default=True
        whether to drop the last samples when they don't fit into window

    """

    def __init__(self, window_size_samples, overlap_size_samples,
                 drop_last_samples=True, tmin=0):
        super().__init__(window_size_samples, overlap_size_samples,
                         drop_last_samples, tmin)

    def transform(self, X):
        """[summary]
        ToDo: id=1???

        Parameters
        ----------
        X : mne.io.Raw
            [description]
        """
        id_holder = 1
        fs = X.info['sfreq']

        if self.window_size_samples is None:
            self.window_size_samples = X.n_times
            duration = X.times[-1]
        else:
            duration = self.window_size_samples / fs

        overlap = self.overlap_size_samples / fs
        events = mne.make_fixed_length_events(
            X, id=id_holder, duration=duration, overlap=overlap, stop=None)

        if not self.drop_last_samples:
            last_valid_window_start = X.n_times - self.window_size_samples
            if events[-1, 0] < last_valid_window_start:
                events = np.concatenate((events, [[last_valid_window_start, 0,
                                                  id_holder]]))

        windows = mne.Epochs(
            X, events, tmin=self.tmin, tmax=(self.window_size_samples - 1) / fs,
            baseline=None, preload=False)

        return windows


class EventWindower(BaseWindower):
    """Fixed onset windower
    ToDo: samples or seconds

    Parameters
    ----------
    window_size_samples : int | None
        size of one window in samples
    overlap_size_samples : int
        size of overlap for window in samples
    drop_last_samples : bool, default=True
        whether to drop the last samples when they don't fit into window

    """
    def __init__(self, window_size_samples, tmin=0, chunk_duration_samples=None,
                 mapping=None):
        super().__init__(window_size_samples, overlap_size_samples=None,
                         tmin=tmin)
        self.chunk_duration_samples = chunk_duration_samples
        if mapping is not None:
            self.mapping = mapping

    def transform(self, X):
        """[summary]
        ToDo: id=1???
        ToDo: plus epsilon 1e-6 on duration; otherwise perfect fitting 
              chunk_durations will SOMETIMES(!!!) not fit

        Parameters
        ----------
        X : mne.io.Raw
            [description]
        """

        fs = X.info['sfreq']

        X.annotations.duration += 1e-6  # see ToDo

        events, events_ids = mne.events_from_annotations(
            X, self.mapping, chunk_duration=self.chunk_duration_samples / fs)

        metadata = {
            'event_onset_idx': events[:, 0],
            'trial_number': range(len(events)),
            'target': events[:, -1]
        }
        metadata = pd.DataFrame(metadata)
        # metadata['subject'] = 

        windows = mne.Epochs(
            X, events, tmin=self.tmin, tmax=(self.window_size_samples - 1) / fs,
            baseline=None, preload=False, metadata=metadata)

        return windows
