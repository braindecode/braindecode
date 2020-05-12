"""Transforms that work on Raw or Epochs objects.
ToDo: should transformer also transform y (e.g. cutting continuous labelled
      data)?
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#          David Sabbagh <dav.sabbagh@gmail.com>
#
# License: BSD (3-clause)

from collections import OrderedDict
from collections.abc import Iterable

import numpy as np


def transform(concat_ds, transforms):
    """Apply a number of transformers to a concat dataset.

    Parameters
    ----------
    concat_ds: A concat of BaseDataset or WindowsDataset
        datasets to be transformed
    transforms: list(str | callable, dict)
        dict with function names of mne.raw or a custom transform and function
        kwargs

    Returns
    -------
    concat_ds:
    """
    assert isinstance(transforms, Iterable)
    for elem in transforms:
        assert isinstance(transforms, Iterable) and len(elem) == 2, (
            "Expect transform function and keyword args per transformation "
            f"got {elem} instead")
    for ds in concat_ds.datasets:
        if hasattr(ds, "raw"):
            _transform(ds.raw, transforms)
        elif hasattr(ds, "windows"):
            _transform(ds.windows, transforms)
        else:
            raise ValueError(
                'Can only transform concatenation of BaseDataset or '
                'WindowsDataset, with either a `raw` or `windows` attribute.')

    # Recompute cumulative sizes as the transforms might have changed them
    # XXX: Ultimately, the best solution would be to have cumulative_size be
    #      a property of BaseConcatDataset.
    concat_ds.cumulative_sizes = concat_ds.cumsum(concat_ds.datasets)


def _transform(raw_or_epochs, transforms):
    """Apply transform(s) to Raw or Epochs object.

    Parameters
    ----------
    raw_or_epochs: mne.io.Raw or mne.Epochs
        Object to transform.
    transforms: list(str | callable, dict)
        List of two elements iterables. First element is either str or callable.
        If str, it represents the name of a method of Raw or Epochs to be called.
        If callable, the callable will be applied to the Raw or Epochs object.
        Values are dictionaries of keyword arguments passed to the transform
        function.

    ..note:
        The methods or callables that are used must modify the Raw or Epochs
        object inplace, otherwise they won't have any effect.
    """
    for transform, transform_kwargs in transforms:
        if callable(transform):
            transform(raw_or_epochs.load_data(), **transform_kwargs)
        else:
            if not hasattr(raw_or_epochs, transform):
                raise AttributeError(
                    f'MNE object does not have {transform} method.')
            getattr(raw_or_epochs.load_data(), transform)(**transform_kwargs)


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
        If this function is supposed to transform continuous data, it should be
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
        If this function is supposed to transform continuous data, it should be
        given to raw.apply_function().
    """
    scaled = np.multiply(data, factor)
    # TODO: the overriding of protected '_data' should be implemented in the
    # TODO: dataset when transforms are applied to windows
    if hasattr(data, '_data'):
        data._data = scaled
    return scaled
