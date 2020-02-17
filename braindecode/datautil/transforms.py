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


def zscore(continuous_data):
    """Zscore raw data

    Parameters
    ----------
    continuous_data: np.ndarray
        continuous signal

    Returns
    -------
    normalized: np.ndarray
        normalized data
    ..note:
        This function is supposed to be given to raw.apply_function().
    """
    continuous_data = continuous_data - continuous_data.mean(axis=-1)
    return continuous_data / continuous_data.std(axis=-1)
