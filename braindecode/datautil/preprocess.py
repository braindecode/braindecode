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
from warnings import warn

import numpy as np
import pandas as pd
import mne


class MNEPreproc(object):
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
        try:
            self._try_apply(raw_or_epochs)
        except RuntimeError:
            # Maybe the function needs the data to be loaded
            # and the data was not loaded yet
            # Not all mne functions need data to be loaded,
            # most importantly the 'crop' function can be
            # lazily applied without preloading data
            # which can make overall preprocessing pipeline
            # substantially faster
            raw_or_epochs.load_data()
            self._try_apply(raw_or_epochs)

    def _try_apply(self, raw_or_epochs):
        if callable(self.fn):
            self.fn(raw_or_epochs, **self.kwargs)
        else:
            if not hasattr(raw_or_epochs, self.fn):
                raise AttributeError(
                    f'MNE object does not have {self.fn} method.')
            getattr(raw_or_epochs, self.fn)(**self.kwargs)


class NumpyPreproc(MNEPreproc):
    """Preprocessor that directly operates on the underlying numpy array of an mne raw/epoch.

    Parameters
    ----------
    fn: callable
        Function that preprocesses the numpy array
    channel_wise: bool
        Whether to apply the function
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
                                          data[0:init_block_size] - init_mean
                                  ) / np.maximum(eps, init_std)
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
               **mne_filter_kwargs):
    """Applies multiple bandpass filters to the signals in raw. The raw will be
    modified in-place and number of channels in raw will be updated to
    len(frequency_bands) * len(raw.ch_names) (-len(raw.ch_names) if
    drop_original_signals).

    Parameters
    ----------
    raw: Instance of mne.io.Raw
        The raw signals to be filtered
    frequency_bands: list(tuple)
        The frequency bands to be filtered for (e.g. [(4, 8), (8, 13)])
    drop_original_signals: bool
        Whether to drop the original unfiltered signals
    mne_filter_kwargs: dict
        Keyworkd arguments for filtering supported by mne.io.Raw.filter().
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
    # reorder channels by frequency band
    chs_by_freq_band = [
        ch for i in range(len(original_ch_names))
        for ch in raw.ch_names[i::len(original_ch_names)]]
    raw.reorder_channels(chs_by_freq_band)
    if drop_original_signals:
        raw.drop_channels(original_ch_names)
