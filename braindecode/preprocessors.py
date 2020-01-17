"""Preprocessing transformers.
"""

from sklearn.base import TransformerMixin


class FilteringTransformer(TransformerMixin):
    """
    """

    def __init__(self, **mne_filter_kwargs):
        self.mne_filter_kwargs = mne_filter_kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """[summary]
        
        Parameters
        ----------
        X : mne.io.Raw
            [description]
        """
        return X.filter(**self.mne_filter_kwargs)


class ZscoreTransformer(TransformerMixin):
    """
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.apply_function(lambda x: x - x.mean())
        X = X.apply_function(lambda x: x / x.std())

        return X


class SpatialTransformer(TransformerMixin):
    """
    spatial transformer. is it done here or after epoching?
    """

    def __init__(self, **spatial_filter_kwargs):
        self.spatial_filter_kwargs = spatial_filter_kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        return X


class FilterBankTransformer(TransformerMixin):
    """
    should be done in the filter transformer? 
    
    filter_kwargs: list of mne-filter-parameters dictionnaries
    """

    def __init__(self, filters_kwargs):
        self.filters_kwargs = filters_kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        filtered_data = X.copy()
        for filter_idx in range(len(self.filters_kwargs)):
            original_data = (
                X.copy()
            )  # <- Need to test this! #reset to original data to not filter succesively the same data.
            # create new raw with filtered data only
            raw_single_band = original_data.filter(**self.filters_kwargs[filter_idx])

            # filter only desired channels for this FB
            if "picks" in self.filters_kwargs[filter_idx]:
                raw_single_band = raw_single_band.pick(
                    self.filters_kwargs[filter_idx]["picks"]
                )
            else:
                raw_single_band = raw_single_band.pick("data")
                # mne automatically changes the highpass/lowpass info values when applying filters and channels cant be added if they have differnet such parameters
                # not needed when making picks as high pass is not modified by filter if pick is specified
                raw_single_band.info["highpass"] = X.info["highpass"]
                raw_single_band.info["lowpass"] = X.info["lowpass"]

            # modify channel names of the new filtered data  #still has the issue that further picks will most likely not have the filter band ones if picked by channel names
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
