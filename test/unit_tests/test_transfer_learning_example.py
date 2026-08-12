import ast
from pathlib import Path

import pytest

TUTORIAL = (
    Path(__file__).resolve().parents[2]
    / "examples/advanced_training/plot_transfer_learning.py"
)


def _load_tutorial_function(name):
    """Load one dependency-free helper without executing the gallery."""
    tree = ast.parse(TUTORIAL.read_text(encoding="utf-8"), filename=str(TUTORIAL))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(functions) == 1, f"Expected one tutorial helper named {name!r}"

    namespace = {}
    module = ast.Module(body=functions, type_ignores=[])
    # The parsed source is a repository-owned tutorial, not untrusted input.
    exec(  # nosec B102
        compile(module, filename=str(TUTORIAL), mode="exec"), namespace
    )
    return namespace[name]


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


def test_synthetic_recording_ids_depend_only_on_domain_identity():
    make_recording_description = _load_tutorial_function("make_recording_description")

    for split in ("train", "valid", "test"):
        for pathological in (0, 1):
            description = make_recording_description(
                domain="target",
                participant_index=7,
                split=split,
                pathological=pathological,
            )

            assert description["recording_id"] == "target-participant-007"
            assert description["split"] == split
            assert description["pathological"] == pathological


def test_target_recording_disjointness_detects_identity_reuse():
    assert_disjoint_recording_ids = _load_tutorial_function(
        "assert_disjoint_recording_ids"
    )
    ids_by_split = {
        "train": {"target-participant-001"},
        "valid": {"target-participant-002"},
        "test": {"target-participant-001"},
    }

    with pytest.raises(AssertionError, match="train.*test"):
        assert_disjoint_recording_ids(ids_by_split)


def test_transfer_learning_gallery_execution_is_cpu_fixed():
    tutorial = TUTORIAL.read_text(encoding="utf-8")

    assert 'device = "cpu"' in tutorial
    assert "torch.cuda.is_available()" not in tutorial
    assert "cuda=True" not in tutorial
