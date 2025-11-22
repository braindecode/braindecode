import numpy as np
import pytest

from braindecode.datautil.util import _get_n_outputs, ms_to_samples, samples_to_ms


def test_ms_to_samples():
    assert ms_to_samples(1000, 44100) == 44100
    assert ms_to_samples(500, 44100) == 22050
    assert ms_to_samples(0, 44100) == 0
    assert ms_to_samples(1000, 0) == 0


def test_samples_to_ms():
    assert samples_to_ms(44100, 44100) == 1000
    assert samples_to_ms(22050, 44100) == 500
    assert samples_to_ms(0, 44100) == 0
    with pytest.raises(ZeroDivisionError):
        assert samples_to_ms(44100, 0) == float("inf")


def test_get_n_outputs_regression():

    # Test _get_n_outputs method
    assert _get_n_outputs(y=None, classes=None, mode="regression") is None
    assert (
        _get_n_outputs(y=np.array([0, 1, 2, 3, 4]), classes=None, mode="regression")
        == 1
    )
    assert (
        _get_n_outputs(
            y=np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),
            classes=None,
            mode="regression",
        )
        == 5
    )
