# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: MIT
from __future__ import annotations

import torch

from braindecode.training import f1_event, f1_sample


def test_f1_event_perfect_match():
    gt = [(0.0, 0.5, 1), (0.6, 0.9, 2)]
    pred = [(0.01, 0.51, 1), (0.6, 0.9, 2)]  # both IoU > 0.5, class matches
    assert f1_event(pred, gt, iou_threshold=0.5) == 1.0


def test_f1_event_class_mismatch_is_fp():
    gt = [(0.0, 0.5, 1)]
    pred = [(0.0, 0.5, 2)]  # perfect overlap but wrong class -> FP + FN
    assert f1_event(pred, gt, iou_threshold=0.5) == 0.0


def test_f1_event_half_precision():
    gt = [(0.0, 0.5, 1)]
    pred = [(0.0, 0.5, 1), (0.0, 0.01, 1)]  # 1 TP, 1 FP -> P=0.5,R=1 -> F1=2/3
    torch.testing.assert_close(
        torch.tensor(f1_event(pred, gt)), torch.tensor(2.0 / 3.0), atol=1e-6, rtol=0
    )


def test_f1_sample_macro_excludes_background_row0():
    # 3 classes (row 0 = background, rows 1-2 real), 4 tokens. CLASS-0 CONTRACT:
    # row 0 is EXCLUDED from the macro mean; only rows 1 and 2 are averaged.
    #   gt = [[bg], class1=[0,1,1,0], class2=[1,0,0,1]]
    #   pred= [[bg], class1=[0,1,0,0], class2=[1,0,1,0]]
    gt = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]])  # (3, T=4)
    pred = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 1, 0]])
    # row1: tp=1 fp=0 fn=1 -> P=1 R=1/2 -> F1=2/3
    # row2: tp=1 fp=1 fn=1 -> P=1/2 R=1/2 -> F1=1/2
    # macro over rows 1..2 only: (2/3 + 1/2) / 2
    val = f1_sample(pred, gt)
    torch.testing.assert_close(
        torch.tensor(val), torch.tensor((2.0 / 3.0 + 0.5) / 2.0), atol=1e-6, rtol=0
    )
