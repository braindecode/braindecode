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


@pytest.mark.parametrize("target_name", ["age", ["age", "score", "bmi"]])
def test_infer_n_outputs_of_windows_with_several_targets(target_name):
    import mne

    from braindecode.datasets import BaseConcatDataset, RawDataset
    from braindecode.datautil import infer_signal_properties
    from braindecode.preprocessing import create_fixed_length_windows

    info = mne.create_info(["C3", "C4"], 100.0, "eeg")
    datasets = [
        RawDataset(
            mne.io.RawArray(np.zeros((2, 300)), info, verbose="error"),
            {"age": 30.0 + i, "score": 1.5, "bmi": 22.0},
            target_name=target_name,
        )
        for i in range(2)
    ]
    windows = create_fixed_length_windows(
        BaseConcatDataset(datasets),
        window_size_samples=100,
        window_stride_samples=100,
        drop_last_window=False,
        # an empty mapping gives several targets as a list
        mapping=None if isinstance(target_name, str) else {},
    )
    n_targets = 1 if isinstance(target_name, str) else len(target_name)
    signal_kwargs = infer_signal_properties(windows, mode="regression")
    assert signal_kwargs["n_outputs"] == n_targets
