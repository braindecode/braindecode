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


def test_classification_output_count_from_probabilities_and_dense_indices():
    probabilities = np.tile(np.array([0.85, 0.05, 0.05, 0.05]), (2, 3, 1)).transpose(
        0, 2, 1
    )
    assert _get_n_outputs(probabilities, None, "classification") == 4
    with pytest.raises(ValueError, match="number of classes"):
        _get_n_outputs(probabilities, [0, 1], "classification")
    indices = np.array([[1, 3, 4], [4, 3, 1]])
    assert _get_n_outputs(indices, None, "classification") == 3


@pytest.mark.parametrize("lazy_metadata", [False, True])
@pytest.mark.parametrize("target_name", ["age", ["age", "score", "bmi"]])
def test_infer_n_outputs_of_windows_with_several_targets(target_name, lazy_metadata):
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
        on_last_window="drop",
        # an empty mapping gives several targets as a list
        mapping=None if isinstance(target_name, str) else {},
        lazy_metadata=lazy_metadata,
    )
    n_targets = 1 if isinstance(target_name, str) else len(target_name)
    signal_kwargs = infer_signal_properties(windows, mode="regression")
    assert signal_kwargs["n_outputs"] == n_targets


@pytest.mark.parametrize("lazy_metadata", [False, True])
def test_window_target_transform_sets_output_count(lazy_metadata):
    import mne

    from braindecode.datasets import BaseConcatDataset, RawDataset
    from braindecode.datautil import infer_signal_properties
    from braindecode.preprocessing import create_fixed_length_windows

    raw = mne.io.RawArray(np.zeros((1, 120)), mne.create_info(["C3"], 100, "eeg"))
    windows = create_fixed_length_windows(
        BaseConcatDataset(
            [
                RawDataset(raw, {"label": y, "study": "demo"}, target_name="label")
                for y in range(3)
            ]
        ),
        window_size_samples=60,
        window_stride_samples=60,
        on_last_window="drop",
        lazy_metadata=lazy_metadata,
    )
    windows.target_transform = lambda y: y % 2
    assert [windows[i][1] for i in range(len(windows))] == [0, 0, 1, 1, 0, 0]
    assert infer_signal_properties(windows)["n_outputs"] == 2

    windows.target_transform = lambda y: np.eye(4)[y] * 0.8 + 0.05
    assert infer_signal_properties(windows)["n_outputs"] == windows[0][1].shape[0] == 4
    metadata = windows.get_metadata()
    assert metadata["target"].tolist() == [0, 0, 1, 1, 2, 2]
    assert metadata["i_dataset"].tolist() == [0, 0, 1, 1, 2, 2]
    assert metadata["study"].tolist() == ["demo"] * len(windows)
