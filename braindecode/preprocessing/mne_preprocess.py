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


def _is_standalone_function(func):
    """
    Determine if a function is standalone based on its module.
    
    Standalone functions are those in mne.preprocessing, mne.channels, mne.filter, etc.
    that are not methods of mne.io.Raw.
    """
    # Check if it's a method of Raw by seeing if it's bound or unbound method
    if hasattr(mne.io.Raw, func.__name__):
        return False
    # Otherwise, it's a standalone function
    return True


def _generate_init_method(func, force_copy_false=False):
    """
    Generate an __init__ method for a class based on the function's signature.
    
    Parameters
    ----------
    func : callable
        The function to wrap.
    force_copy_false : bool
        If True, forces copy=False by default for functions that have a copy parameter.
    """
    parameters = list(inspect.signature(func).parameters.values())
    param_names = [param.name for param in parameters]

    def init_method(self, *args, **kwargs):
        # For standalone functions with copy parameter, set copy=False by default
        if force_copy_false and 'copy' in param_names and 'copy' not in kwargs:
            kwargs['copy'] = False
        
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

    Parameters
    ----------
    function : callable
        The MNE function to wrap. Automatically determines if it's standalone
        or a Raw method based on the function's module and name.
    """
    class_name = "".join(word.title() for word in function.__name__.split("_")).replace(
        "Eeg", "EEG"
    )
    import_path = f"{function.__module__}.{function.__name__}"
    doc = f" See more details in {import_path}"

    base_classes = (Preprocessor,)
    
    # Automatically determine if function is standalone
    is_standalone = _is_standalone_function(function)
    
    # Check if function has a 'copy' parameter
    sig = inspect.signature(function)
    has_copy_param = 'copy' in sig.parameters
    force_copy_false = is_standalone and has_copy_param

    if is_standalone:
        # For standalone functions, store the actual function object
        class_attrs = {
            "__init__": _generate_init_method(function, force_copy_false=force_copy_false),
            "__doc__": _update_moabb_docstring(function, doc),
            "fn": function,  # Store the function itself, not the name
            "_is_standalone": True,
        }
    else:
        # For methods, store the function name as before
        class_attrs = {
            "__init__": _generate_init_method(function),
            "__doc__": _update_moabb_docstring(function, doc),
            "fn": function.__name__,
            "_is_standalone": False,
        }

    generated_class = type(class_name, base_classes, class_attrs)

    return generated_class


# List of MNE functions to generate classes for
mne_functions = [
    # From mne.filter
    mne.filter.resample,
    # From mne.io.Raw methods
    mne.io.Raw.add_channels,
    mne.io.Raw.add_events,
    mne.io.Raw.add_proj,
    mne.io.Raw.add_reference_channels,
    mne.io.Raw.anonymize,
    mne.io.Raw.apply_gradient_compensation,
    mne.io.Raw.apply_hilbert,
    mne.io.Raw.apply_proj,
    mne.io.Raw.crop,
    mne.io.Raw.crop_by_annotations,
    mne.io.Raw.del_proj,
    mne.io.Raw.drop_channels,
    mne.io.Raw.filter,
    mne.io.Raw.fix_mag_coil_types,
    mne.io.Raw.interpolate_bads,
    mne.io.Raw.interpolate_to,
    mne.io.Raw.notch_filter,
    mne.io.Raw.pick,
    mne.io.Raw.pick_channels,
    mne.io.Raw.pick_types,
    mne.io.Raw.rename_channels,
    mne.io.Raw.reorder_channels,
    mne.io.Raw.rescale,
    mne.io.Raw.resample,
    mne.io.Raw.savgol_filter,
    mne.io.Raw.set_annotations,
    mne.io.Raw.set_channel_types,
    mne.io.Raw.set_eeg_reference,
    mne.io.Raw.set_meas_date,
    mne.io.Raw.set_montage,
    # Standalone functions from mne.preprocessing
    mne.preprocessing.annotate_amplitude,
    mne.preprocessing.annotate_break,
    mne.preprocessing.annotate_movement,
    mne.preprocessing.annotate_muscle_zscore,
    mne.preprocessing.annotate_nan,
    mne.preprocessing.compute_current_source_density,
    mne.preprocessing.compute_bridged_electrodes,
    mne.preprocessing.fix_stim_artifact,
    mne.preprocessing.interpolate_bridged_electrodes,
    mne.preprocessing.maxwell_filter,
    mne.preprocessing.oversampled_temporal_projection,
    mne.preprocessing.realign_raw,
    mne.preprocessing.regress_artifact,
    # Standalone functions from mne.channels
    mne.channels.combine_channels,
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
    and issubclass(class_obj, Preprocessor)
    and class_obj != Preprocessor
]

# Clean up unnecessary variables
del mne_functions, function, class_obj
