"""Preprocessors that work on Raw or Epochs objects.
ToDo: should transformer also transform y (e.g. cutting continuous labelled
      data)?
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#          David Sabbagh <dav.sabbagh@gmail.com>
#
# License: BSD (3-clause)

from collections.abc import Iterable
from functools import partial

import numpy as np


class MNEPreproc():
    """Preprocessor for an MNE-raw/epoch.

    Parameters
    ----------
    fn: str or callable
        if str, the raw/epoch object must have a member function with that name.
        if callable, directly apply the callable to the mne raw/epoch.
    kwargs:
        Keyword arguments will be forwarded to the mne function
    """
    def __init__(self, fn, **kwargs):
        self.fn = fn
        self.kwargs = kwargs

    def apply(self, raw_or_epochs):
        if callable(self.fn):
            self.fn(raw_or_epochs.load_data(), **self.kwargs)
        else:
            if not hasattr(raw_or_epochs, self.fn):
                raise AttributeError(
                    f'MNE object does not have {self.fn} method.')
            getattr(raw_or_epochs.load_data(), self.fn)(**self.kwargs)


class NumpyPreproc(MNEPreproc):
    """Preprocessor that directy operates on the underlying numpy array of an mne raw/epoch.

    Parameters
    ----------
    fn: callable
        Function that preprocesses the numpy array
    channel_wise: bool
        Whether to apply the functiona
    kwargs:
        Keyword arguments will be forwarded to the function
    """
    def __init__(self, fn, channel_wise=False, **kwargs):
        # use apply function of mne which will directly apply it to numpy array
        partial_fn = partial(fn, **kwargs)
        mne_kwargs = dict(fun=partial_fn, channel_wise=channel_wise)
        super().__init__(fn='apply_function', **mne_kwargs)


def preprocess(concat_ds, preprocessors):
    """Apply several preprocessors to a concat dataset.

    Parameters
    ----------
    concat_ds: A concat of BaseDataset or WindowsDataset
        datasets to be preprocessed
    preprocessors: list(MNEPreproc) #TODO: correct object stuffs
        List of preprocessors to apply to the dataset

    Returns
    -------
    concat_ds:
    """
    assert isinstance(preprocessors, Iterable)
    for elem in preprocessors:
        assert hasattr(elem, 'apply'), (
            "Expect preprocessor object to have apply method")

    for ds in concat_ds.datasets:
        if hasattr(ds, "raw"):
            _preprocess(ds.raw, preprocessors)
        elif hasattr(ds, "windows"):
            _preprocess(ds.windows, preprocessors)
        else:
            raise ValueError(
                'Can only perprocess concatenation of BaseDataset or '
                'WindowsDataset, with either a `raw` or `windows` attribute.')

    # Recompute cumulative sizes as the transforms might have changed them
    # XXX: Ultimately, the best solution would be to have cumulative_size be
    #      a property of BaseConcatDataset.
    concat_ds.cumulative_sizes = concat_ds.cumsum(concat_ds.datasets)


def _preprocess(raw_or_epochs, preprocessors):
    """Apply preprocessor(s) to Raw or Epochs object.

    Parameters
    ----------
    raw_or_epochs: mne.io.Raw or mne.Epochs
        Object to preprocess.
    preprocessors: list(MNEPreproc) #TODO: correct object stuffs
        List of preprocessors to apply to the dataset
    """
    for preproc in preprocessors:
        preproc.apply(raw_or_epochs)


def zscore(data):
    """Zscore continuous or windowed data in-place

    Parameters
    ----------
    data: np.ndarray (n_channels x n_times) or (n_windows x n_channels x
    n_times)
        continuous or windowed signal

    Returns
    -------
    zscored: np.ndarray (n_channels x n_times) or (n_windows x n_channels x
    n_times)
        normalized continuous or windowed data
    ..note:
        If this function is supposed to preprocess continuous data, it should be
        given to raw.apply_function().
    """
    zscored = data - np.mean(data, keepdims=True, axis=-1)
    zscored = zscored / np.std(zscored, keepdims=True, axis=-1)
    # TODO: the overriding of protected '_data' should be implemented in the
    # TODO: dataset when transforms are applied to windows
    if hasattr(data, '_data'):
        data._data = zscored
    return zscored


def scale(data, factor):
    """Scale continuous or windowed data in-place

    Parameters
    ----------
    data: np.ndarray (n_channels x n_times) or (n_windows x n_channels x
    n_times)
        continuous or windowed signal
    factor: float
        multiplication factor

    Returns
    -------
    scaled: np.ndarray (n_channels x n_times) or (n_windows x n_channels x
    n_times)
        normalized continuous or windowed data
    ..note:
        If this function is supposed to preprocess continuous data, it should be
        given to raw.apply_function().
    """
    scaled = np.multiply(data, factor)
    # TODO: the overriding of protected '_data' should be implemented in the
    # TODO: dataset when transforms are applied to windows
    if hasattr(data, '_data'):
        data._data = scaled
    return scaled
