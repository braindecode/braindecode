from pathlib import Path

TUTORIAL = (
    Path(__file__).resolve().parents[2]
    / "examples/advanced_training/plot_transfer_learning.py"
)


def test_transfer_learning_tuab_path_contract():
    tutorial = TUTORIAL.read_text(encoding="utf-8")
    loader_call = tutorial.split("source_recordings = TUHAbnormal(", maxsplit=1)[
        1
    ].split(")", maxsplit=1)[0]

    assert 'path="/path/to/tuh_eeg_abnormal",' in loader_call
    assert 'version="v3.0.1",' in loader_call
    assert 'path="/path/to/tuh_eeg_abnormal/v3.0.1/edf"' not in loader_call


def test_transfer_learning_defers_target_test_access_until_evaluation():
    tutorial = TUTORIAL.read_text(encoding="utf-8")
    training, evaluation = tutorial.split(
        "# Evaluate once on the held-out target test set", maxsplit=1
    )

    assert (
        'target_recordings.description.groupby(["split", "pathological"])'
        not in training
    )
    assert "target_test_windows" not in training
    assert 'target_windows["test"]' not in training
    assert '_ = fine_tune_clf.fit(target_windows["train"], y=None)' in training
    assert (
        'target_test_windows = window_recordings(target_recording_splits["test"])'
        in evaluation
    )
    assert "test_metadata = target_test_windows.get_metadata()" in evaluation


def test_transfer_learning_gallery_execution_is_cpu_fixed():
    tutorial = TUTORIAL.read_text(encoding="utf-8")

    assert 'device = "cpu"' in tutorial
    assert "torch.cuda.is_available()" not in tutorial
    assert "cuda=True" not in tutorial
