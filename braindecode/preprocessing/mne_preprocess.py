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
    func_name = func.__name__
    parameters = list(inspect.signature(func).parameters.values())
    param_names = [
        param.name
        for param in parameters[1:]  # Skip 'self' or 'raw' or 'epochs'
    ]
    all_mandatory = [
        param.name
        for param in parameters[1:]  # Skip 'self' or 'raw' or 'epochs'
        if param.default == inspect.Parameter.empty
    ]

    def init_method(self, *args, **kwargs):
        used = []
        mandatory = list(all_mandatory)
        init_kwargs = {}

        # For standalone functions with copy parameter, set copy=False by default
        if force_copy_false and "copy" in param_names and "copy" not in kwargs:
            kwargs["copy"] = False

        for name, value in zip(param_names, args):
            init_kwargs[name] = value
            used.append(name)
            if name in mandatory:
                mandatory.remove(name)
        for name, value in kwargs.items():
            if name in used:
                raise TypeError(f"Multiple values for argument '{name}'")
            if name not in param_names:
                raise TypeError(
                    f"'{name}' is an invalid keyword argument for {func_name}()"
                )
            init_kwargs[name] = value
            if name in mandatory:
                mandatory.remove(name)
        if len(mandatory) > 0:
            raise TypeError(
                f"{func_name}() missing required arguments: {', '.join(mandatory)}"
            )
        Preprocessor.__init__(self, fn=func_name, apply_on_array=False, **init_kwargs)

    init_method.__signature__ = inspect.signature(func)
    return init_method


def _generate_repr_method(class_name):
    def repr_method(self):
        args_str = ", ".join(f"{k}={v.__repr__()}" for k, v in self.kwargs.items())
        return f"{class_name}({args_str})"

    return repr_method


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

    # Automatically determine if function is standalone
    is_standalone = _is_standalone_function(function)

    # Create a wrapper note that references the original MNE function
    # For Raw methods, use mne.io.Raw.method_name format with :meth:
    # For standalone functions, use the function name only with :func:
    if not is_standalone:
        ref_path = f"mne.io.Raw.{function.__name__}"
        ref_role = "meth"
    else:
        # For standalone functions, try common MNE public APIs
        # These are more likely to be in intersphinx inventory
        func_name = function.__name__
        if function.__module__.startswith("mne.preprocessing"):
            ref_path = f"mne.preprocessing.{func_name}"
        elif function.__module__.startswith("mne.channels"):
            ref_path = f"mne.channels.{func_name}"
        elif function.__module__.startswith("mne.filter"):
            ref_path = f"mne.filter.{func_name}"
        else:
            ref_path = f"{function.__module__}.{func_name}"
        ref_role = "func"

    # Use proper Sphinx cross-reference for intersphinx linking
    wrapper_note = (
        f"Braindecode preprocessor wrapper for :{ref_role}:`~{ref_path}`.\n\n"
    )

    base_classes = (Preprocessor,)

    # Check if function has a 'copy' parameter
    sig = inspect.signature(function)
    has_copy_param = "copy" in sig.parameters
    force_copy_false = is_standalone and has_copy_param
    # Automatically determine if function is standalone
    is_standalone = _is_standalone_function(function)

    # Check if function has a 'copy' parameter
    sig = inspect.signature(function)
    has_copy_param = "copy" in sig.parameters
    force_copy_false = is_standalone and has_copy_param
    class_attrs = {
        "__init__": _generate_init_method(function, force_copy_false),
        "__doc__": wrapper_note + (function.__doc__ or ""),
        "__repr__": _generate_repr_method(class_name),
        "fn": function if is_standalone else function.__name__,
        "_is_standalone": is_standalone,
    }
    generated_class = type(class_name, base_classes, class_attrs)

    return generated_class


# List of MNE functions to generate classes for
mne_functions = [
    # From mne.filter
    mne.filter.resample,
    mne.filter.filter_data,
    mne.filter.notch_filter,
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
    mne.preprocessing.equalize_bads,
    mne.preprocessing.find_bad_channels_lof,
    mne.preprocessing.fix_stim_artifact,
    mne.preprocessing.interpolate_bridged_electrodes,
    mne.preprocessing.maxwell_filter,
    mne.preprocessing.oversampled_temporal_projection,
    mne.preprocessing.realign_raw,
    mne.preprocessing.regress_artifact,
    # Standalone functions from mne.channels
    mne.channels.combine_channels,
    mne.channels.equalize_channels,
    mne.channels.rename_channels,
    # Top-level mne functions for referencing
    mne.add_reference_channels,
    mne.set_bipolar_reference,
    mne.set_eeg_reference,
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
