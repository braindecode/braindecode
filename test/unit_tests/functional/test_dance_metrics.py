# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: MIT
from __future__ import annotations

import pytest
import torch

from braindecode.training import f1_event, f1_sample


@pytest.mark.parametrize(
    "pred,gt,expected",
    [
        # both IoU > 0.5 and class matches -> perfect
        ([(0.01, 0.51, 1), (0.6, 0.9, 2)], [(0.0, 0.5, 1), (0.6, 0.9, 2)], 1.0),
        # perfect overlap but wrong class -> FP + FN
        ([(0.0, 0.5, 2)], [(0.0, 0.5, 1)], 0.0),
        # nothing to detect, nothing predicted -> trivially perfect
        ([], [], 1.0),
        # events on only one side -> all FP / all FN
        ([(0.0, 0.5, 1)], [], 0.0),
        ([], [(0.0, 0.5, 1)], 0.0),
        # 1 TP + 1 FP -> P=0.5, R=1 -> F1=2/3
        ([(0.0, 0.5, 1), (0.0, 0.01, 1)], [(0.0, 0.5, 1)], 2.0 / 3.0),
    ],
    ids=["perfect", "class-mismatch", "empty-empty", "fp-only", "fn-only", "half-prec"],
)
def test_f1_event(pred, gt, expected):
    torch.testing.assert_close(
        torch.tensor(f1_event(pred, gt, iou_threshold=0.5)),
        torch.tensor(expected),
        atol=1e-6,
        rtol=0,
    )


def test_f1_sample_macro_excludes_background_row0():
    # 3 classes (row 0 = background, rows 1-2 real), 4 tokens. CLASS-0 CONTRACT:
    # row 0 is EXCLUDED from the macro mean; only rows 1 and 2 are averaged.
    gt = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]])  # (3, T=4)
    pred = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 1, 0]])
    # row1: tp=1 fp=0 fn=1 -> F1=2/3 ; row2: tp=1 fp=1 fn=1 -> F1=1/2
    # macro over rows 1..2 only: (2/3 + 1/2) / 2
    torch.testing.assert_close(
        torch.tensor(f1_sample(pred, gt)),
        torch.tensor((2.0 / 3.0 + 0.5) / 2.0),
        atol=1e-6,
        rtol=0,
    )
