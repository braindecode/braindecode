"""Preprocessor objects based on mne methods."""

# Authors: Bruna Lopes <brunajaflopes@gmail.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3
import inspect

import mne.channels
import mne.io
import mne.preprocessing

from braindecode.preprocessing.preprocess import Preprocessor
from braindecode.util import _update_moabb_docstring


def _generate_init_method(func):
    """
    Generate an __init__ method for a class based on the function's signature.
    """
    parameters = list(inspect.signature(func).parameters.values())
    param_names = [param.name for param in parameters]

    def init_method(self, *args, **kwargs):
        for name, value in zip(param_names, args):
            setattr(self, name, value)
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.kwargs = kwargs

    init_method.__signature__ = inspect.signature(func)
    return init_method


def _generate_mne_pre_processor(function):
    """
    Generate a class based on an MNE function for preprocessing.
    """
    class_name = "".join(word.title() for word in function.__name__.split("_")).replace(
        "Eeg", "EEG"
    )
    import_path = f"{function.__module__}.{function.__name__}"
    doc = f" See more details in {import_path}"

    base_classes = (Preprocessor,)
    class_attrs = {
        "__init__": _generate_init_method(function),
        "__doc__": _update_moabb_docstring(function, doc),
        "fn": function.__name__,
    }
    generated_class = type(class_name, base_classes, class_attrs)

    return generated_class


# List of MNE functions to generate classes for
mne_functions = [
    # From mne.filter
    mne.filter.resample,
    # From mne.io.Raw
    mne.io.Raw.drop_channels,
    mne.io.Raw.filter,
    mne.io.Raw.crop,
    mne.io.Raw.pick,
    mne.io.Raw.set_eeg_reference,
    mne.io.Raw.interpolate_bads,
    mne.io.Raw.notch_filter,
    mne.io.Raw.savgol_filter,
    mne.io.Raw.set_montage,
    mne.io.Raw.rename_channels,
    mne.io.Raw.reorder_channels,
    mne.io.Raw.add_reference_channels,
    mne.io.Raw.apply_hilbert,
    mne.io.Raw.apply_proj,
    # From mne.preprocessing
    mne.preprocessing.compute_current_source_density,
    mne.preprocessing.fix_stim_artifact,
    # From mne.channels
    mne.channels.equalize_channels,
]

# Automatically generate and add classes to the global namespace
for function in mne_functions:
    class_obj = _generate_mne_pre_processor(function)
    globals()[class_obj.__name__] = class_obj

# Define __all__ based on the generated class names
__all__ = [
    class_obj.__name__
    for class_obj in globals().values()
    if isinstance(class_obj, type)
]

# Clean up unnecessary variables
del mne_functions, function, class_obj
