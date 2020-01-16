"""Transformers that work on Raw or Epochs objects.
"""

from sklearn.base import TransformerMixin


class FilteringRawTransformer(TransformerMixin):
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


class ZscoreRawTransformer(TransformerMixin):
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


class FilteringWindowTransformer(TransformerMixin):
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


class ZscoreWindowTransformer(TransformerMixin):
    """
    XXX: Might be merged with ZscorePreprocessor
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.apply_function(lambda x: x - x.mean())
        X = X.apply_function(lambda x: x / x.std())

        return X
        