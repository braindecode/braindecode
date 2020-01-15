"""Get epochs from mne.Raw
"""

import numpy as np
from sklearn.base import TransformerMixin
import mne

class Windower(TransformerMixin):
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

    def __init__(self, window_size_samples, overlap_size_samples, drop_last_samples=True, tmin=0):
        self.window_size_samples = window_size_samples
        self.overlap_size_samples = overlap_size_samples
        self.drop_last_samples = drop_last_samples
        self.tmin = tmin

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """[summary]
        ToDo: understand why in case windower = Windower(window_size_samples=1,
                                                         overlap_size_samples=0) last event is dropped
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
        events = mne.make_fixed_length_events(X, id=id_holder, duration=duration, overlap=overlap, stop=X.times[-1])

        if not self.drop_last_samples:
            last_valid_window_start = X.n_times - self.window_size_samples
            if events[-1, 0] < last_valid_window_start:
                events = np.concatenate((events, [[last_valid_window_start, 0, id_holder]]))

        windows = mne.Epochs(X, events, tmin=self.tmin, tmax=(self.window_size_samples - 1) / fs,
                             baseline=None, preload=False)

        return windows
