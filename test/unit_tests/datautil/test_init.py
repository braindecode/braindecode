import pytest

from braindecode.datautil import __getattr__


# Define a fixture for the names and their expected results
@pytest.fixture(params=[
    ("create_from_X_y", "create_from_X_y has been moved to datasets, please use from braindecode.datasets import create_from_X_y", "..datasets.xy", "create_from_X_y"),
    ("create_from_mne_raw", "create_from_mne_raw has been moved to datasets, please use from braindecode.datasets import create_from_mne_raw", "..datasets.mne", "create_from_mne_raw"),
    ("non_existent", "No possible import named non_existent", None, None)
])
def name_and_result(request):
    return request.param


# Use the fixture in the test
def test_getattr(name_and_result):
    name, expected_warning, expected_module, expected_function = name_and_result

    if expected_module is not None:
        with pytest.warns(UserWarning, match=expected_warning):
            assert callable(__getattr__(name))
    else:
        with pytest.raises(AttributeError, match=expected_warning):
            __getattr__(name)
