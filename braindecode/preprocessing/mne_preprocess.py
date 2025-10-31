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


def _generate_mne_pre_processor(function, is_standalone=False):
    """
    Generate a class based on an MNE function for preprocessing.

    Parameters
    ----------
    function : callable
        The MNE function to wrap.
    is_standalone : bool
        If True, the function is a standalone function (e.g., from mne.preprocessing)
        that takes raw as the first argument. If False, it's a method on Raw objects.
    """
    class_name = "".join(word.title() for word in function.__name__.split("_")).replace(
        "Eeg", "EEG"
    )
    import_path = f"{function.__module__}.{function.__name__}"
    doc = f" See more details in {import_path}"

    base_classes = (Preprocessor,)

    if is_standalone:
        # For standalone functions, store the actual function object
        class_attrs = {
            "__init__": _generate_init_method(function),
            "__doc__": _update_moabb_docstring(function, doc),
            "fn": function,  # Store the function itself, not the name
        }
    else:
        # For methods, store the function name as before
        class_attrs = {
            "__init__": _generate_init_method(function),
            "__doc__": _update_moabb_docstring(function, doc),
            "fn": function.__name__,
        }

    generated_class = type(class_name, base_classes, class_attrs)

    return generated_class


# List of MNE functions to generate classes for
# Format: (function, is_standalone)
mne_functions = [
    # From mne.filter
    (mne.filter.resample, False),
    # From mne.io.Raw methods
    (mne.io.Raw.add_channels, False),
    (mne.io.Raw.add_events, False),
    (mne.io.Raw.add_proj, False),
    (mne.io.Raw.add_reference_channels, False),
    (mne.io.Raw.anonymize, False),
    (mne.io.Raw.apply_gradient_compensation, False),
    (mne.io.Raw.apply_hilbert, False),
    (mne.io.Raw.apply_proj, False),
    (mne.io.Raw.crop, False),
    (mne.io.Raw.crop_by_annotations, False),
    (mne.io.Raw.del_proj, False),
    (mne.io.Raw.drop_channels, False),
    (mne.io.Raw.filter, False),
    (mne.io.Raw.fix_mag_coil_types, False),
    (mne.io.Raw.interpolate_bads, False),
    (mne.io.Raw.notch_filter, False),
    (mne.io.Raw.pick, False),
    (mne.io.Raw.rename_channels, False),
    (mne.io.Raw.reorder_channels, False),
    (mne.io.Raw.rescale, False),
    (mne.io.Raw.resample, False),
    (mne.io.Raw.savgol_filter, False),
    (mne.io.Raw.set_annotations, False),
    (mne.io.Raw.set_channel_types, False),
    (mne.io.Raw.set_eeg_reference, False),
    (mne.io.Raw.set_meas_date, False),
    (mne.io.Raw.set_montage, False),
    # Standalone functions from mne.preprocessing
    (mne.preprocessing.compute_current_source_density, True),
    (mne.preprocessing.fix_stim_artifact, True),
    # Standalone functions from mne.channels
    (mne.channels.equalize_channels, True),
]

# Automatically generate and add classes to the global namespace
for function, is_standalone in mne_functions:
    class_obj = _generate_mne_pre_processor(function, is_standalone=is_standalone)
    globals()[class_obj.__name__] = class_obj

# Define __all__ based on the generated class names
__all__ = [
    class_obj.__name__
    for class_obj in globals().values()
    if isinstance(class_obj, type)
    and issubclass(class_obj, Preprocessor)
    and class_obj != Preprocessor
]

# Clean up unnecessary variables
del mne_functions, function, is_standalone, class_obj
