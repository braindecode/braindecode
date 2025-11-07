"""Preprocessor objects based on mne methods."""

# Authors: Bruna Lopes <brunajaflopes@gmail.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3
import inspect

import mne.io

from braindecode.preprocessing.preprocess import Preprocessor
from braindecode.util import _update_moabb_docstring


def _generate_init_method(func):
    """
    Generate an __init__ method for a class based on the function's signature.
    """
    func_name = func.__name__
    parameters = list(inspect.signature(func).parameters.values())
    param_names = [param.name for param in parameters if param.name != "self"]
    all_mandatory = [
        param.name
        for param in parameters
        if param.default == inspect.Parameter.empty and param.name != "self"
    ]

    def init_method(self, *args, **kwargs):
        used = []
        mandatory = list(all_mandatory)
        init_kwargs = {}
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
        "__repr__": _generate_repr_method(class_name),
    }
    generated_class = type(class_name, base_classes, class_attrs)

    return generated_class


# List of MNE functions to generate classes for
mne_functions = [
    mne.filter.resample,
    mne.io.Raw.drop_channels,
    mne.io.Raw.filter,
    mne.io.Raw.crop,
    mne.io.Raw.pick,
    mne.io.Raw.set_eeg_reference,
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
