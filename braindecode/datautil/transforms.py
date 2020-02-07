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

import mne


def transform_concat_ds(concat_ds, transforms):
    """Apply a number of transformers to a concat dataset.

    Parameters
    ----------
    concat_ds: A concat of BaseDataset or WindowsDataset
        datasets to be transformed
    transforms: dict(str | callable: dict)
        dict with function names of mne.raw or a custom transform and function
        kwargs

    Returns
    -------
    concat_ds:
    """
    if not isinstance(transforms, OrderedDict):
        raise TypeError(
            "Order of transforms matters! Please provide an OrderedDict.")
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
    transforms: OrderedDict
        Keys are either str or callable. If str, it represents the name of a
        method of Raw or Epochs to be called. If callable, the callable will be
        applied to the Raw or Epochs object.
        Values are dictionaries of keyword arguments passed to the transform
        function.

    ..note:
        The methods or callables that are used must modify the Raw or Epochs
        object inplace, otherwise they won't have any effect.
    """
    for transform, transform_kwargs in transforms.items():
        if callable(transform):
            transform(raw_or_epochs, **transform_kwargs)
        else:
            if not hasattr(raw_or_epochs, transform):
                raise AttributeError(
                    f'MNE object does not have {transform} method.')
            getattr(raw_or_epochs.load_data(), transform)(**transform_kwargs)


class FilterRaw(object):
    """Apply mne filter on raw data

    Parameters
    ----------
    mne_filter_kwargs : **kwargs
        kwargs passed to mne.io.Raw.filter
    """

    def __init__(self, **mne_filter_kwargs):
        self.mne_filter_kwargs = mne_filter_kwargs

    def __call__(self, raw):
        """Apply filter

        Parameters
        ----------
        raw : mne.io.Raw
            raw data to filter
        """
        return raw.filter(**self.mne_filter_kwargs)


class ZscoreRaw(object):
    """Zscore raw data channel wise
    """

    def __call__(self, raw):
        """Zscore Normalize raw data channel wise

        Parameters
        ----------
        raw : mne.io.Raw
            raw data to normalize

        Returns
        -------
        raw : mne.io.Raw
            normalized raw data

        """
        raw = raw.apply_function(lambda x: x - x.mean())
        return raw.apply_function(lambda x: x / x.std())


class FilterWindow(object):
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

    def __init__(
        self, sfreq, l_freq=None, h_freq=None, kwargs=None, overlap_kwargs=None
    ):
        if kwargs is None:
            kwargs = dict()
        if overlap_kwargs is None:
            overlap_kwargs = dict()
        self.filter = mne.filter.create_filter(
            None, sfreq, l_freq=l_freq, h_freq=h_freq, method="fir", **kwargs
        )
        self.overlap_kwargs = overlap_kwargs

    def __call__(self, windows):
        return mne.filter._overlap_add_filter(
            windows, self.filter, **self.overlap_kwargs
        )


class ZscoreWindow(object):
    """Zscore windowed data channel wise
    """

    def __call__(self, windows):
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
        windows -= windows.mean(axis=-1, keepdims=True)
        return windows / windows.std(axis=-1, keepdims=True)
