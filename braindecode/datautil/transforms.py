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

import mne


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


class FilterBankRaw(object):
    """
    Implement filter bank on raw object.

    Parameters
    ----------
    filters_kwargs : list of dictionaries. 
        Each dictionnary contains mne filter creation parameters for a given 
        frequency band. See mne.filter.create_filter for possible parameters

    Returns
    ----------
    filtered_data : raw object
        contains the original data plus additionnal channels with the filtered 
        data. New channels have the original channel name with _FB{i} appended
        with i the indice of the filter
    """

    def __init__(self, filters_kwargs):
        self.filters_kwargs = filters_kwargs

    def __call__(self, raw):
        filtered_data = raw.copy()
        for filter_idx in range(len(self.filters_kwargs)):
            original_data = (
                raw.copy()
            )  # reset to original data to not filter succesively the same data.
            # create new raw with filtered data only
            raw_single_band = original_data.filter(
                **self.filters_kwargs[filter_idx]
            )

            # filter only desired channels for this FB
            if "picks" in self.filters_kwargs[filter_idx]:
                raw_single_band = raw_single_band.pick(
                    self.filters_kwargs[filter_idx]["picks"]
                )
            else:
                raw_single_band = raw_single_band.pick("data")
                # mne automatically changes the highpass/lowpass info values 
                # when applying filters and channels cant be added if they have 
                # different such parameters. Not needed when making picks as 
                # high pass is not modified by filter if pick is specified
                raw_single_band.info["highpass"] = raw.info["highpass"]
                raw_single_band.info["lowpass"] = raw.info["lowpass"]

            # modify channel names of the new filtered data  #still has the 
            # issue that further picks will most likely not have the filter band 
            # ones if picked by channel names
            channel_single_band = raw_single_band.info["ch_names"]
            channel_single_band_original = channel_single_band
            channel_single_band = [
                "{0}_FB{1}".format(channel_single_band, filter_idx + 1)
                for channel_single_band in channel_single_band
            ]
            raw_single_band.rename_channels(
                dict(zip(channel_single_band_original, channel_single_band))
            )  # map old channel name to new one and use mne-rename

            filtered_data.add_channels([raw_single_band])
        return filtered_data


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
        windows : ndarray, shape (n_channels, window_size)
            windowed data to normalize

        Returns
        -------
        windows : ndarray, shape (n_channels, window_size)
            normalized windowed data

        """
        windows -= windows.mean(axis=-1, keepdims=True)
        return windows / windows.std(axis=-1, keepdims=True)
