import re

import pytest

from braindecode import datasets, datautil, preprocessing


@pytest.mark.parametrize(
    ("destination", "name"),
    [
        (datasets, "create_from_X_y"),
        (datasets, "create_from_mne_raw"),
        (datasets, "create_from_mne_epochs"),
        (preprocessing, "exponential_moving_demean"),
        (preprocessing, "exponential_moving_standardize"),
        (preprocessing, "filterbank"),
        (preprocessing, "preprocess"),
        (preprocessing, "Preprocessor"),
        (preprocessing, "create_windows_from_events"),
        (preprocessing, "create_fixed_length_windows"),
    ],
)
def test_moved_attribute(destination, name):
    message = f"please use from {destination.__name__} import {name}"
    with pytest.warns(UserWarning, match=re.escape(message)):
        alias = getattr(datautil, name)
    assert alias is getattr(destination, name)


@pytest.mark.parametrize("name", ["scale", "non_existent"])
def test_missing_attribute(name):
    with pytest.raises(AttributeError, match=f"No possible import named {name}"):
        getattr(datautil, name)
