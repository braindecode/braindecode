"""Transformers that work on Raw or Epochs objects.
ToDo: decide whether window transformers should work on mne.Epochs or numpy.Arrays
ToDo: should transformer also transform y (e.g. cutting continuous labelled data)?
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#          David Sabbagh <dav.sabbagh@gmail.com>
#
# License: BSD (3-clause)

from sklearn.base import TransformerMixin
import mne


class FilterRawTransformer(TransformerMixin):
    """Apply mne filter on raw data

    Parameters
    ----------
    mne_filter_kwargs : **kwargs
        kwargs passed to mne.io.Raw.filter
    """
    def __init__(self, **mne_filter_kwargs):
        self.mne_filter_kwargs = mne_filter_kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Apply filter
        
        Parameters
        ----------
        X : mne.io.Raw
            raw data to filter
        """
        return X.filter(**self.mne_filter_kwargs)


class ZscoreRawTransformer(TransformerMixin):
    """Zscore raw data channel wise
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Zscore Normalize raw data channel wise

        Parameters
        ----------
        X : mne.io.Raw
            raw data to normalize

        Returns
        -------
        X : mne.io.Raw
            normalized raw data

        """
        X = X.apply_function(lambda x: x - x.mean())
        X = X.apply_function(lambda x: x / x.std())

        return X


class FilterWindowTransformer(TransformerMixin):
    """FIR filter for windowed data.

    Parameters
    ----------
    sfreq : int | float
        sampling frequency of data when applying filter
    l_freq : int | float | None
        see mne.filter.create_filter
    h_freq : int | float | None
        see mne.filter.create_filter
    kwargs : **kwargs
        see mne.filter.create_filter
    overlap_kwargs  : **kwargs
        see mne.filter._overlap_add_filter
    """
    def __init__(self, sfreq, l_freq=None, h_freq=None, kwargs=None, overlap_kwargs=None):
        if kwargs is None:
            kwargs = dict()
        if overlap_kwargs is None:
            overlap_kwargs = dict()
        self.filter = mne.filter.create_filter(None, sfreq, l_freq=l_freq, h_freq=h_freq, method='fir', **kwargs)
        self.overlap_kwargs = overlap_kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = mne.filter._overlap_add_filter(X, self.filter, **self.overlap_kwargs)
        return X


class ZscoreWindowTransformer(TransformerMixin):
    """Zscore windowed data channel wise
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Zscore Normalize windowed data channel wise

        Parameters
        ----------
        X : ndarray, shape (n_channels, window_size)
            windowed data to normalize

        Returns
        -------
        X : ndarray, shape (n_channels, window_size)
            normalized windowed data

        """
        X -= X.mean(axis=-1, keepdims=True)
        X /= X.std(axis=-1, keepdims=True)
        return X
