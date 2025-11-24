import pytest

from braindecode.datautil.util import ms_to_samples, samples_to_ms


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
        assert samples_to_ms(44100, 0) == float('inf')
