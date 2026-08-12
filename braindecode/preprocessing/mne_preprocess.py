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
from braindecode.util import _clean_docstring_sections

# Standalone MNE functions that are safe to wrap as a *callable* preprocessor.
#
# When ``fn`` is a callable and ``apply_on_array=False``, ``Preprocessor`` calls
# ``fn(raw, **kwargs)`` and reassigns the dataset to whatever ``fn`` returns. That
# is only correct for functions that return the modified ``Raw``/``Epochs``
# instance. Many standalone MNE functions instead return auxiliary data
# (annotations, a list of bad channels, a ``(inst, ...)`` tuple, ...), so passing
# them as callables would silently replace the recording with that auxiliary
# object. We therefore restrict the callable path to functions verified to return
# the modified instance; every other standalone function keeps the existing
# (string-name) behaviour. See :gh:`885`.
_SAFE_STANDALONE_FUNCTIONS = frozenset(
    {
        # returns the Raw with the CSD-transformed data
        mne.preprocessing.compute_current_source_density,
        # returns the Raw with the new bipolar reference channel
        mne.set_bipolar_reference,
        # returns the Raw with the OTP-cleaned data
        mne.preprocessing.oversampled_temporal_projection,
    }
)


# ---------------------------------------------------------------------------
# Wrapper factories for standalone functions that return auxiliary data.
#
# These functions cannot be passed as callables directly because
# ``Preprocessor.apply`` would replace the recording with the auxiliary return
# value (e.g. an Annotations object or a list of bad channels).  Each factory
# returns a new callable that calls the original function, applies its side
# effects onto the recording, and returns the (still-valid) recording.
# See :gh:`1055`.
# ---------------------------------------------------------------------------


def _wrap_annotation_return(func, also_bads=False):
    """Wrap a function that returns Annotations (or ``(Annotations, bads)``).

    The wrapper calls *func*, adds the returned annotations to the recording
    with :meth:`mne.io.Raw.set_annotations`, and—when *also_bads* is
    ``True``—extends ``raw.info['bads']`` with the bad-channel list returned
    as the second tuple element.  The recording is returned unchanged otherwise.
    """

    def wrapper(raw, **kwargs):
        result = func(raw, **kwargs)
        if isinstance(result, tuple):
            annot = result[0]
            bads = list(result[1]) if also_bads else []
        else:
            annot = result
            bads = []
        raw.set_annotations(raw.annotations + annot)
        if bads:
            raw.info["bads"] = list(set(raw.info["bads"] + bads))
        return raw

    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = func.__qualname__
    wrapper.__module__ = func.__module__
    wrapper.__doc__ = func.__doc__
    return wrapper


def _wrap_bad_channels_return(func):
    """Wrap a function that returns a list of bad channel names.

    The wrapper calls *func*, extends ``raw.info['bads']`` with the returned
    channel names (deduplicating), and returns the recording.  If the function
    was called with ``return_scores=True`` (so it returns a ``(bads, scores)``
    tuple), only the first element is used.
    """

    def wrapper(raw, **kwargs):
        result = func(raw, **kwargs)
        bads = result[0] if isinstance(result, tuple) else result
        raw.info["bads"] = list(set(raw.info["bads"] + list(bads)))
        return raw

    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = func.__qualname__
    wrapper.__module__ = func.__module__
    wrapper.__doc__ = func.__doc__
    return wrapper


def _wrap_bridged_electrodes(func):
    """Wrap :func:`mne.preprocessing.compute_bridged_electrodes`.

    The function returns ``(bridged_idx, ed_matrix)``.  The wrapper marks
    every channel that appears in a bridged pair as bad by extending
    ``raw.info['bads']``, then returns the recording.
    """

    def wrapper(raw, **kwargs):
        bridged_idx, _ = func(raw, **kwargs)
        ch_names = raw.ch_names
        new_bads = [ch_names[idx] for pair in bridged_idx for idx in pair]
        raw.info["bads"] = list(set(raw.info["bads"] + new_bads))
        return raw

    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = func.__qualname__
    wrapper.__module__ = func.__module__
    wrapper.__doc__ = func.__doc__
    return wrapper


# Map each "aux-return" standalone function to its safe wrapper callable.
# These wrappers apply the function's side effects to the recording and return
# the recording itself, so they can be used with the callable path in
# ``Preprocessor`` without corrupting the dataset.
_AUX_RETURN_WRAPPERS = {
    mne.preprocessing.annotate_amplitude: _wrap_annotation_return(
        mne.preprocessing.annotate_amplitude, also_bads=True
    ),
    mne.preprocessing.annotate_nan: _wrap_annotation_return(
        mne.preprocessing.annotate_nan
    ),
    mne.preprocessing.annotate_break: _wrap_annotation_return(
        mne.preprocessing.annotate_break
    ),
    mne.preprocessing.annotate_movement: _wrap_annotation_return(
        mne.preprocessing.annotate_movement
    ),
    mne.preprocessing.annotate_muscle_zscore: _wrap_annotation_return(
        mne.preprocessing.annotate_muscle_zscore
    ),
    mne.preprocessing.compute_bridged_electrodes: _wrap_bridged_electrodes(
        mne.preprocessing.compute_bridged_electrodes
    ),
    mne.preprocessing.find_bad_channels_lof: _wrap_bad_channels_return(
        mne.preprocessing.find_bad_channels_lof
    ),
}


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
        # Choose the callable to store in this Preprocessor instance.
        #
        # - Functions in ``_SAFE_STANDALONE_FUNCTIONS`` return the modified
        #   Raw/Epochs directly and are passed as-is.
        # - Functions in ``_AUX_RETURN_WRAPPERS`` return auxiliary data; we
        #   use a pre-built wrapper that applies the side effects and returns
        #   the recording.  See :gh:`1055`.
        # - Every other standalone function keeps the legacy string-name
        #   behaviour (``_apply_str``).
        if func in _SAFE_STANDALONE_FUNCTIONS:
            fn = func
        elif func in _AUX_RETURN_WRAPPERS:
            fn = _AUX_RETURN_WRAPPERS[func]
        else:
            fn = func_name
        Preprocessor.__init__(self, fn=fn, apply_on_array=False, **init_kwargs)

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
    class_attrs = {
        "__init__": _generate_init_method(function, force_copy_false),
        "__doc__": wrapper_note + _clean_docstring_sections(function.__doc__ or ""),
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
