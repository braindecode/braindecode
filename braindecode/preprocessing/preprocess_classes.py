"""Preprocessor objects based on mne methods.
"""

# Author: Bruna Lopes <brunajaflopes@gmail.com>
#
# License: BSD-3

import mne.io
from braindecode.util import _update_moabb_docstring
from braindecode.preprocessing import Preprocessor


class Resample(Preprocessor):
    doc = """
    Subclass of Preprocessor to perform resampling using mne.Epochs/mne.io.Raw's resample method.

    Parameters
    ----------
    kwargs:
        Keyword arguments to be passed to mne.Epochs's resample method.
    """
    try:
        base_class = mne.io.Raw.resample
        _doc__ = _update_moabb_docstring(base_class, doc)
    except ModuleNotFoundError:
        pass

    def __init__(self, **kwargs):
        # Ignore "fn" parameter -> the only preprocessing that is going to
        # be used here is mne.Epochs's resample
        # Removing this string parameter, because it's going to execute
        # directly the said preprocess

        fn = 'resample'
        self.fn = fn
        self.kwargs = kwargs

        super().__init__(fn, **kwargs)


class DropChannels(Preprocessor):
    doc = """
    Subclass of Preprocessor to drop specific channels using mne.Epochs/mne.io.Raw's method.

    Parameters
    ----------
    kwargs:
        Keyword arguments to be passed to mne.Epochs's resample method.
    """
    try:
        base_class = mne.io.Raw.drop_channels
        _doc__ = _update_moabb_docstring(base_class, doc)
    except ModuleNotFoundError:
        pass

    def __init__(self, **kwargs):
        fn = 'drop_channels'
        self.fn = fn
        self.kwargs = kwargs
        # Init parent
        super().__init__(fn, **kwargs)


class SetEEGReference(Preprocessor):
    doc = """
    Subclass of Preprocessor to specify the reference for EEG signals
    using mne.Epochs/mne.io.Raw's method.

    Parameters
    ----------
    kwargs:
        Keyword arguments to be passed to mne.Epochs's resample method.
    """
    try:
        from mne.channels import channels
        base_class = mne.io.Raw.set_eeg_reference
        _doc__ = _update_moabb_docstring(base_class, doc)
    except ModuleNotFoundError:
        pass

    def __init__(self, **kwargs):
        fn = 'set_eeg_reference'
        self.fn = fn
        self.kwargs = kwargs

        super().__init__(fn, **kwargs)


class Filter(Preprocessor):
    doc = """
    Subclass of Preprocessor to perform filtering using mne.Epochs/mne.io.Raw's method.

    Parameters
    ----------
    kwargs:
        Keyword arguments to be passed to mne.Epochs's resample method.
    """
    try:
        base_class = mne.io.Raw.filter
        _doc__ = _update_moabb_docstring(base_class, doc)
    except ModuleNotFoundError:
        pass

    def __init__(self, **kwargs):
        fn = 'filter'
        self.fn = fn
        self.kwargs = kwargs

        super().__init__(fn, **kwargs)


class Pick(Preprocessor):
    doc = """
    Subclass of Preprocessor to pick a subset of channels using mne.Epochs/mne.io.Raw's method.

    Parameters
    ----------
    kwargs:
        Keyword arguments to be passed to mne.Epochs's resample method.
    """
    try:
        base_class = mne.io.Raw.pick
        _doc__ = _update_moabb_docstring(base_class, doc)
    except ModuleNotFoundError:
        pass

    def __init__(self, **kwargs):
        fn = 'pick'
        self.fn = fn
        self.kwargs = kwargs

        super().__init__(fn, **kwargs)


class Crop(Preprocessor):
    doc = """
    Subclass of Preprocessor to crop a time interval using mne.Epochs/mne.io.Raw's method.

    Parameters
    ----------
    kwargs:
        Keyword arguments to be passed to mne.Epochs's resample method.
    """
    try:
        base_class = mne.io.Raw.pick
        _doc__ = _update_moabb_docstring(base_class, doc)
    except ModuleNotFoundError:
        pass

    def __init__(self, **kwargs):
        fn = 'crop'
        self.fn = fn
        self.kwargs = kwargs

        super().__init__(fn, **kwargs)
