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
        