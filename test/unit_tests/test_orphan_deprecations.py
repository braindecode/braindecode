"""The orphaned public API of #1442 warns and keeps working until 2.2."""

import numpy as np
import pytest
import torch

from braindecode.datasets.bbci import load_bbci_sets_from_folder
from braindecode.datasets.registry import is_registered_dataset
from braindecode.modules import IntermediateOutputWrapper
from braindecode.training.callbacks import MaxNormConstraintCallback
from braindecode.util import corr, cov, get_balanced_batches, read_all_file_names


def _intermediate_output_wrapper():
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())
    out = IntermediateOutputWrapper(["0"], model)(torch.zeros(2, 4))
    return out[0].shape


def _read_all_file_names(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    return read_all_file_names(str(tmp_path), ".txt")


@pytest.mark.parametrize(
    "name, call, expected",
    [
        ("corr", lambda tmp: corr(np.eye(2, 5), np.eye(2, 5)).shape, (2, 2)),
        ("cov", lambda tmp: cov(np.eye(2, 5), np.eye(2, 5)).shape, (2, 2)),
        (
            "get_balanced_batches",
            lambda tmp: [len(b) for b in get_balanced_batches(5, None, False, 2)],
            [3, 2],
        ),
        ("read_all_file_names", _read_all_file_names, None),
        ("is_registered_dataset", lambda tmp: is_registered_dataset(1, "nope"), False),
        (
            "load_bbci_sets_from_folder",
            lambda tmp: load_bbci_sets_from_folder(str(tmp)),
            [],
        ),
        (
            "IntermediateOutputWrapper",
            lambda tmp: _intermediate_output_wrapper(),
            (2, 3),
        ),
        (
            "MaxNormConstraintCallback",
            lambda tmp: type(MaxNormConstraintCallback()).__name__,
            "MaxNormConstraintCallback",
        ),
    ],
)
def test_orphaned_api_warns_and_still_works(name, call, expected, tmp_path):
    with pytest.warns(FutureWarning, match=f"{name} is deprecated") as record:
        result = call(tmp_path)
    assert len([w for w in record if name in str(w.message)]) == 1
    assert record[0].filename == __file__
    if name == "read_all_file_names":
        assert [p.endswith("a.txt") for p in result] == [True]
    elif expected is not None:
        assert result == expected
