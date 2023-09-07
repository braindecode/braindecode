"""Preprocessor objects based on mne methods.
"""

# Author: Bruna Lopes <brunajaflopes@gmail.com>
#
# License: BSD-3

import mne.io
from braindecode.util import _update_moabb_docstring
from braindecode.preprocessing import Preprocessor


class Resample(Preprocessor):
    try:
        base_class = mne.io.Raw.resample
        __doc__ = base_class.__doc__
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
    try:
        base_class = mne.io.Raw.drop_channels
        __doc__ = base_class.__doc__
    except ModuleNotFoundError:
        pass

    def __init__(self, **kwargs):
        fn = 'drop_channels'
        self.fn = fn
        self.kwargs = kwargs
        # Init parent
        super().__init__(fn, **kwargs)


class SetEEGReference(Preprocessor):
    try:
        from mne.channels import channels
        base_class = mne.io.Raw.set_eeg_reference
        __doc__ = _update_moabb_docstring(base_class, doc)
    except ModuleNotFoundError:
        pass

    def __init__(self, **kwargs):
        fn = 'set_eeg_reference'
        self.fn = fn
        self.kwargs = kwargs

        super().__init__(fn, **kwargs)


class Filter(Preprocessor):
    try:
        base_class = mne.io.Raw.filter
        __doc__ = _update_moabb_docstring(base_class, doc)
    except ModuleNotFoundError:
        pass

    def __init__(self, **kwargs):
        fn = 'filter'
        self.fn = fn
        self.kwargs = kwargs

        super().__init__(fn, **kwargs)


class Pick(Preprocessor):
    try:
        base_class = mne.io.Raw.pick
        __doc__ = base_class.__doc__
    except ModuleNotFoundError:
        pass

    def __init__(self, **kwargs):
        fn = 'pick'
        self.fn = fn
        self.kwargs = kwargs

        super().__init__(fn, **kwargs)


class Crop(Preprocessor):
    try:
        base_class = mne.io.Raw.pick
        __doc__ = base_class.__doc__
    except ModuleNotFoundError:
        pass

    def __init__(self, **kwargs):
        fn = 'crop'
        self.fn = fn
        self.kwargs = kwargs

        super().__init__(fn, **kwargs)
