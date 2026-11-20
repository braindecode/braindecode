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


@pytest.mark.parametrize(
    "y, n_outputs",
    [
        (np.array([2, 0, 2, 1]), 3),
        # dense class indices (n, n_times) count the classes, not the 8 time steps
        (np.arange(32).reshape(4, 8) % 3, 3),
        (np.arange(32.0).reshape(4, 8) % 3, 3),
    ],
    ids=["labels", "dense_int_indices", "dense_float_indices"],
)
def test_n_outputs_counts_the_distinct_labels_of_any_shape(y, n_outputs):
    from braindecode.datautil import infer_signal_properties

    signal_kwargs = infer_signal_properties(np.zeros((len(y), 2, 8)), y)
    assert signal_kwargs["n_outputs"] == n_outputs


def test_classes_name_the_labels():
    from braindecode.datautil import infer_signal_properties

    X, soft = np.zeros((4, 2, 8)), np.full((4, 3), 1 / 3)
    assert infer_signal_properties(X, [0, 1, 0, 1], classes=[0, 1, 2])["n_outputs"] == 3
    # class probabilities are not labels: classes alone sizes the model
    assert infer_signal_properties(X, soft, classes=[0, 1, 2])["n_outputs"] == 3
    with pytest.raises(ValueError, match=r"Labels \[2\] are not in classes \[0, 1\]"):
        infer_signal_properties(X, [0, 2, 0, 1], classes=[0, 1])


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


def _windows_of_labels(labels, lazy_metadata):
    """Two windows per recording, each recording stored with one label."""
    import mne

    from braindecode.datasets import BaseConcatDataset, RawDataset
    from braindecode.preprocessing import create_fixed_length_windows

    raw = mne.io.RawArray(np.zeros((1, 120)), mne.create_info(["C3"], 100, "eeg"))
    recordings = [RawDataset(raw, {"label": y}, target_name="label") for y in labels]
    return create_fixed_length_windows(
        BaseConcatDataset(recordings), window_size_samples=60, lazy_metadata=lazy_metadata
    )


@pytest.mark.parametrize("lazy_metadata", [False, True])
def test_classes_name_the_stored_window_labels(lazy_metadata):
    from braindecode.datautil import infer_signal_properties

    windows = _windows_of_labels("abc", lazy_metadata)
    # The metadata keeps the stored labels, the model trains on the transformed ones.
    windows.target_transform = {"a": 0, "b": 1, "c": 0}.__getitem__
    assert windows.get_metadata()["target"].tolist() == list("aabbcc")
    assert infer_signal_properties(windows)["n_outputs"] == 2
    # classes names the stored labels, as in v2.0.0.
    assert infer_signal_properties(windows, classes=list("abc"))["n_outputs"] == 3
    with pytest.raises(ValueError, match=r"Labels \['c'\] are not in classes"):
        infer_signal_properties(windows, classes=list("ab"))
    # vectors from the transform are not labels either: classes alone sizes the model
    windows.target_transform = lambda label: np.eye(3)["abc".index(label)]
    assert infer_signal_properties(windows, classes=list("abc"))["n_outputs"] == 3
