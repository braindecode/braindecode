"""Preprocessors that work on Raw or Epochs objects.
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#          David Sabbagh <dav.sabbagh@gmail.com>
#
# License: BSD (3-clause)

from warnings import warn
from functools import partial
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.utils import deprecated
from joblib import Parallel, delayed

from braindecode.datasets.base import BaseConcatDataset, BaseDataset, WindowsDataset
from braindecode.datautil.serialization import (
    load_concat_dataset, _check_save_dir_empty)


class Preprocessor(object):
    """Preprocessor for an MNE Raw or Epochs object.

    Applies the provided preprocessing function to the data of a Raw or Epochs
    object.
    If the function is provided as a string, the method with that name will be
    used (e.g., 'pick_channels', 'filter', etc.).
    If it is provided as a callable and `apply_on_array` is True, the
    `apply_function` method of Raw and Epochs object will be used to apply the
    function on the internal arrays of Raw and Epochs.
    If `apply_on_array` is False, the callable must directly modify the Raw or
    Epochs object (e.g., by calling its method(s) or modifying its attributes).

    Parameters
    ----------
    fn: str or callable
        If str, the Raw/Epochs object must have a method with that name.
        If callable, directly apply the callable to the object.
    apply_on_array : bool
        Ignored if `fn` is not a callable. If True, the `apply_function` of Raw
        and Epochs object will be used to run `fn` on the underlying arrays
        directly. If False, `fn` must directly modify the Raw or Epochs object.
    kwargs:
        Keyword arguments to be forwarded to the MNE function.
    """
    def __init__(self, fn, *, apply_on_array=True, **kwargs):
        if hasattr(fn, '__name__') and fn.__name__ == '<lambda>':
            warn('Preprocessing choices with lambda functions cannot be saved.')
        if callable(fn) and apply_on_array:
            channel_wise = kwargs.pop('channel_wise', False)
            picks = kwargs.pop('picks', None)
            n_jobs = kwargs.pop('n_jobs', 1)
            kwargs = dict(fun=partial(fn, **kwargs), channel_wise=channel_wise,
                          picks=picks, n_jobs=n_jobs)
            fn = 'apply_function'
        self.fn = fn
        self.kwargs = kwargs

    def apply(self, raw_or_epochs):
        try:
            self._try_apply(raw_or_epochs)
        except RuntimeError:
            # Maybe the function needs the data to be loaded and the data was
            # not loaded yet. Not all MNE functions need data to be loaded,
            # most importantly the 'crop' function can be lazily applied
            # without preloading data which can make the overall preprocessing
            # pipeline substantially faster.
            raw_or_epochs.load_data()
            self._try_apply(raw_or_epochs)

    def _try_apply(self, raw_or_epochs):
        if callable(self.fn):
            self.fn(raw_or_epochs, **self.kwargs)
        else:
            if not hasattr(raw_or_epochs, self.fn):
                raise AttributeError(
                    f'MNE object does not have a {self.fn} method.')
            getattr(raw_or_epochs, self.fn)(**self.kwargs)


@deprecated(extra='will be removed in 0.7.0. Use Preprocessor with '
                  '`apply_on_array=False` instead.')
class MNEPreproc(Preprocessor):
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
        super().__init__(fn, apply_on_array=False, **kwargs)


@deprecated(extra='will be removed in 0.7.0. Use Preprocessor with '
                  '`apply_on_array=True` instead.')
class NumpyPreproc(Preprocessor):
    """Preprocessor that directly operates on the underlying numpy array of an mne raw/epoch.

    Parameters
    ----------
    fn: callable
        Function that preprocesses the numpy array
    channel_wise: bool
        Whether to apply the function channel-wise.
    kwargs:
        Keyword arguments will be forwarded to the function
    """
    def __init__(self, fn, channel_wise=False, **kwargs):
        assert callable(fn), 'fn must be callable.'
        super().__init__(fn, apply_on_array=True, channel_wise=channel_wise,
                         **kwargs)


def preprocess(concat_ds, preprocessors, save_dir=None, overwrite=False,
               n_jobs=None):
    """Apply preprocessors to a concat dataset.

    Parameters
    ----------
    concat_ds: BaseConcatDataset
        A concat of BaseDataset or WindowsDataset datasets to be preprocessed.
    preprocessors: list(Preprocessor)
        List of Preprocessor objects to apply to the dataset.
    save_dir : str | None
        If a string, the preprocessed data will be saved under the specified
        directory and the datasets in ``concat_ds`` will be reloaded with
        `preload=False`.
    overwrite : bool
        When `save_dir` is provided, controls whether to delete the old
        subdirectories that will be written to under `save_dir`. If False and
        the corresponding subdirectories already exist, a ``FileExistsError``
        will be raised.
    n_jobs : int | None
        Number of jobs for parallel execution.

    Returns
    -------
    BaseConcatDataset:
        Preprocessed dataset.
    """
    # In case of serialization, make sure directory is available before
    # preprocessing
    if save_dir is not None and not overwrite:
        _check_save_dir_empty(save_dir)

    if not isinstance(preprocessors, Iterable):
        raise ValueError(
            'preprocessors must be a list of Preprocessor objects.')
    for elem in preprocessors:
        assert hasattr(elem, 'apply'), (
            'Preprocessor object needs an `apply` method.')

    list_of_ds = Parallel(n_jobs=n_jobs)(
        delayed(_preprocess)(ds, i, preprocessors, save_dir, overwrite)
        for i, ds in enumerate(concat_ds.datasets))

    if save_dir is not None:  # Reload datasets and replace in concat_ds
        concat_ds_reloaded = load_concat_dataset(
            save_dir, preload=False, target_name=None)
        _replace_inplace(concat_ds, concat_ds_reloaded)
    else:
        if n_jobs is None or n_jobs == 1:  # joblib did not make copies, the
            # preprocessing happened in-place
            # Recompute cumulative sizes as transforms might have changed them
            concat_ds.cumulative_sizes = concat_ds.cumsum(concat_ds.datasets)
        else:  # joblib made copies
            _replace_inplace(concat_ds, BaseConcatDataset(list_of_ds))

    return concat_ds


def _replace_inplace(concat_ds, new_concat_ds):
    """Replace subdatasets and preproc_kwargs of a BaseConcatDataset inplace.

    Parameters
    ----------
    concat_ds : BaseConcatDataset
        Dataset to modify inplace.
    new_concat_ds : BaseConcatDataset
        Dataset to use to modify ``concat_ds``.
    """
    if len(concat_ds.datasets) != len(new_concat_ds.datasets):
        raise ValueError('Both inputs must have the same length.')
    for i in range(len(new_concat_ds.datasets)):
        concat_ds.datasets[i] = new_concat_ds.datasets[i]

    concat_kind = 'raw' if hasattr(concat_ds.datasets[0], 'raw') else 'window'
    preproc_kwargs_attr = concat_kind + '_preproc_kwargs'
    if hasattr(new_concat_ds, preproc_kwargs_attr):
        setattr(concat_ds, preproc_kwargs_attr,
                getattr(new_concat_ds, preproc_kwargs_attr))


def _preprocess(ds, ds_index, preprocessors, save_dir=None, overwrite=False):
    """Apply preprocessor(s) to Raw or Epochs object.

    Parameters
    ----------
    ds: BaseDataset | WindowsDataset
        Dataset object to preprocess.
    ds_index : int
        Index of the BaseDataset in its BaseConcatDataset. Ignored if save_dir
        is None.
    preprocessors: list(Preprocessor)
        List of preprocessors to apply to the dataset.
    save_dir : str | None
        If provided, save the preprocessed BaseDataset in the
        specified directory.
    overwrite : bool
        If True, overwrite existing file with the same name.
    """

    def _preprocess_raw_or_epochs(raw_or_epochs, preprocessors):
        for preproc in preprocessors:
            preproc.apply(raw_or_epochs)

    if hasattr(ds, 'raw'):
        _preprocess_raw_or_epochs(ds.raw, preprocessors)
    elif hasattr(ds, 'windows'):
        _preprocess_raw_or_epochs(ds.windows, preprocessors)
    else:
        raise ValueError(
            'Can only preprocess concatenation of BaseDataset or '
            'WindowsDataset, with either a `raw` or `windows` attribute.')

    # Store preprocessing keyword arguments in the dataset
    _set_preproc_kwargs(ds, preprocessors)

    if save_dir is not None:
        concat_ds = BaseConcatDataset([ds])
        concat_ds.save(save_dir, overwrite=overwrite, offset=ds_index)
    else:
        return ds


def _get_preproc_kwargs(preprocessors):
    preproc_kwargs = []
    for p in preprocessors:
        # in case of a mne function, fn is a str, kwargs is a dict
        func_name = p.fn
        func_kwargs = p.kwargs
        # in case of another function
        # if apply_on_array=False
        if callable(p.fn):
            func_name = p.fn.__name__
        # if apply_on_array=True
        else:
            if 'fun' in p.fn:
                func_name = p.kwargs['fun'].func.__name__
                func_kwargs = p.kwargs['fun'].keywords
        preproc_kwargs.append((func_name, func_kwargs))
    return preproc_kwargs


def _set_preproc_kwargs(ds, preprocessors):
    """Record preprocessing keyword arguments in BaseDataset or WindowsDataset.

    Parameters
    ----------
    ds : BaseDataset | WindowsDataset
        Dataset in which to record preprocessing keyword arguments.
    preprocessors : list
        List of preprocessors.
    """
    preproc_kwargs = _get_preproc_kwargs(preprocessors)
    if isinstance(ds, WindowsDataset):
        kind = 'window'
    elif isinstance(ds, BaseDataset):
        kind = 'raw'
    else:
        raise TypeError(
            f'ds must be a BaseDataset or a WindowsDataset, got {type(ds)}')
    setattr(ds, kind + '_preproc_kwargs', preproc_kwargs)


def exponential_moving_standardize(
        data, factor_new=0.001, init_block_size=None, eps=1e-4
):
    r"""Perform exponential moving standardization.

    Compute the exponental moving mean :math:`m_t` at time `t` as
    :math:`m_t=\mathrm{factornew} \cdot mean(x_t) + (1 - \mathrm{factornew}) \cdot m_{t-1}`.

    Then, compute exponential moving variance :math:`v_t` at time `t` as
    :math:`v_t=\mathrm{factornew} \cdot (m_t - x_t)^2 + (1 - \mathrm{factornew}) \cdot v_{t-1}`.

    Finally, standardize the data point :math:`x_t` at time `t` as:
    :math:`x'_t=(x_t - m_t) / max(\sqrt{->v_t}, eps)`.


    Parameters
    ----------
    data: np.ndarray (n_channels, n_times)
    factor_new: float
    init_block_size: int
        Standardize data before to this index with regular standardization.
    eps: float
        Stabilizer for division by zero variance.

    Returns
    -------
    standardized: np.ndarray (n_channels, n_times)
        Standardized data.
    """
    data = data.T
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    if init_block_size is not None:
        i_time_axis = 0
        init_mean = np.mean(
            data[0:init_block_size], axis=i_time_axis, keepdims=True
        )
        init_std = np.std(
            data[0:init_block_size], axis=i_time_axis, keepdims=True
        )
        init_block_standardized = (
            data[0:init_block_size] - init_mean) / np.maximum(eps, init_std)
        standardized[0:init_block_size] = init_block_standardized
    return standardized.T


def exponential_moving_demean(data, factor_new=0.001, init_block_size=None):
    r"""Perform exponential moving demeanining.

    Compute the exponental moving mean :math:`m_t` at time `t` as
    :math:`m_t=\mathrm{factornew} \cdot mean(x_t) + (1 - \mathrm{factornew}) \cdot m_{t-1}`.

    Deman the data point :math:`x_t` at time `t` as:
    :math:`x'_t=(x_t - m_t)`.

    Parameters
    ----------
    data: np.ndarray (n_channels, n_times)
    factor_new: float
    init_block_size: int
        Demean data before to this index with regular demeaning.

    Returns
    -------
    demeaned: np.ndarray (n_channels, n_times)
        Demeaned data.
    """
    data = data.T
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    demeaned = np.array(demeaned)
    if init_block_size is not None:
        i_time_axis = 0
        init_mean = np.mean(
            data[0:init_block_size], axis=i_time_axis, keepdims=True
        )
        demeaned[0:init_block_size] = data[0:init_block_size] - init_mean
    return demeaned.T


@deprecated(extra='will be removed in 0.7.0. Use sklearn.preprocessing.scale '
                  'instead.')
def zscore(data):
    """Zscore normalize continuous or windowed data in-place.

    Parameters
    ----------
    data: np.ndarray (n_channels, n_times) or (n_windows, n_channels, n_times)
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


@deprecated(extra='will be removed in 0.7.0. Use numpy.multiply instead.')
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


def filterbank(raw, frequency_bands, drop_original_signals=True,
               order_by_frequency_band=False, **mne_filter_kwargs):
    """Applies multiple bandpass filters to the signals in raw. The raw will be
    modified in-place and number of channels in raw will be updated to
    len(frequency_bands) * len(raw.ch_names) (-len(raw.ch_names) if
    drop_original_signals).

    Parameters
    ----------
    raw: mne.io.Raw
        The raw signals to be filtered.
    frequency_bands: list(tuple)
        The frequency bands to be filtered for (e.g. [(4, 8), (8, 13)]).
    drop_original_signals: bool
        Whether to drop the original unfiltered signals
    order_by_frequency_band: bool
        If True will return channels odered by frequency bands, so if there
        are channels Cz, O1 and filterbank ranges [(4,8), (8,13)], returned
        channels will be [Cz_4-8, O1_4-8, Cz_8-13, O1_8-13]. If False, order
        will be [Cz_4-8, Cz_8-13, O1_4-8, O1_8-13].
    mne_filter_kwargs: dict
        Keyword arguments for filtering supported by mne.io.Raw.filter().
        Please refer to mne for a detailed explanation.
    """
    if not frequency_bands:
        raise ValueError(f"Expected at least one frequency band, got"
                         f" {frequency_bands}")
    if not all([len(ch_name) < 8 for ch_name in raw.ch_names]):
        warn("Try to use shorter channel names, since frequency band "
             "annotation requires an estimated 4-8 chars depending on the "
             "frequency ranges. Will truncate to 15 chars (mne max).")
    original_ch_names = raw.ch_names
    all_filtered = []
    for (l_freq, h_freq) in frequency_bands:
        filtered = raw.copy()
        filtered.filter(l_freq=l_freq, h_freq=h_freq, **mne_filter_kwargs)
        # mne automatically changes the highpass/lowpass info values
        # when applying filters and channels cant be added if they have
        # different such parameters. Not needed when making picks as
        # high pass is not modified by filter if pick is specified
        filtered.info["highpass"] = raw.info["highpass"]
        filtered.info["lowpass"] = raw.info["lowpass"]
        # add frequency band annotation to channel names
        # truncate to a max of 15 characters, since mne does not allow for more
        filtered.rename_channels({
            old_name: (old_name + f"_{l_freq}-{h_freq}")[-15:]
            for old_name in filtered.ch_names})
        all_filtered.append(filtered)
    raw.add_channels(all_filtered)
    if not order_by_frequency_band:
        # order channels by name and not by frequency band:
        # index the list with a stepsize of the number of channels for each of
        # the original channels
        chs_by_freq_band = []
        for i in range(len(original_ch_names)):
            chs_by_freq_band.extend(raw.ch_names[i::len(original_ch_names)])
        raw.reorder_channels(chs_by_freq_band)
    if drop_original_signals:
        raw.drop_channels(original_ch_names)
