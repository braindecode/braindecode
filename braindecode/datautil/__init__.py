"""
Utilities for data manipulation.
"""

from .serialization import (
    _check_save_dir_empty,
    load_concat_dataset,
    save_concat_dataset,
)


def __getattr__(name):
    # ideas from https://stackoverflow.com/a/57110249/1469195
    import importlib
    from warnings import warn

    if name == "create_from_X_y":
        warn(
            "create_from_X_y has been moved to datasets, please use from braindecode.datasets import create_from_X_y"
        )
        xy = importlib.import_module("..datasets.xy", __package__)
        return xy.create_from_X_y
    if name in ["create_from_mne_raw", "create_from_mne_epochs"]:
        warn(
            f"{name} has been moved to datasets, please use from braindecode.datasets import {name}"
        )
        mne = importlib.import_module("..datasets.mne", __package__)
        return mne.__dict__[name]
    if name in [
        "scale",
        "exponential_moving_demean",
        "exponential_moving_standardize",
        "filterbank",
        "preprocess",
        "Preprocessor",
    ]:
        warn(
            f"{name} has been moved to preprocessing, please use from braindecode.preprocessing import {name}"
        )
        preprocess = importlib.import_module("..preprocessing.preprocess", __package__)
        return preprocess.__dict__[name]
    if name in ["create_windows_from_events", "create_fixed_length_windows"]:
        warn(
            f"{name} has been moved to preprocessing, please use from braindecode.preprocessing import {name}"
        )
        windowers = importlib.import_module("..preprocessing.windowers", __package__)
        return windowers.__dict__[name]

    raise AttributeError("No possible import named " + name)


__all__ = ["load_concat_dataset", "save_concat_dataset", "_check_save_dir_empty"]
