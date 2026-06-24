# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: MIT
from __future__ import annotations

import torch

from braindecode.functional import (
    detr_to_dense_probs,
    events_to_mask,
    extract_events_from_detr_batch,
    iou_1d,
    pairwise_iou_1d,
)


def test_iou_1d_elementwise_overlap():
    # ELEMENTWISE: all inputs share shape (1,).
    s1 = torch.tensor([0.0]); e1 = torch.tensor([0.5])
    s2 = torch.tensor([0.25]); e2 = torch.tensor([0.75])
    iou = iou_1d(s1, e1, s2, e2)
    assert iou.shape == (1,)
    # inter=0.25, union=0.75 -> 1/3
    torch.testing.assert_close(iou, torch.tensor([1.0 / 3.0]), atol=1e-5, rtol=0)


def test_iou_1d_disjoint_is_zero():
    iou = iou_1d(
        torch.tensor([0.0]), torch.tensor([0.1]),
        torch.tensor([0.5]), torch.tensor([0.6]),
    )
    torch.testing.assert_close(iou, torch.zeros(1), atol=1e-6, rtol=0)


def test_pairwise_iou_1d_shape_and_values():
    # PAIRWISE: (Q,) x (T,) -> (Q, T).
    s1 = torch.tensor([0.0, 0.5]); e1 = torch.tensor([0.5, 1.0])
    s2 = torch.tensor([0.25]); e2 = torch.tensor([0.75])
    iou = pairwise_iou_1d(s1, e1, s2, e2)
    assert iou.shape == (2, 1)
    torch.testing.assert_close(iou[0, 0], torch.tensor(1.0 / 3.0), atol=1e-5, rtol=0)


def test_detr_to_dense_probs_shape_and_normalized():
    # Queries span the full [0, 1) window so every timestep is covered.
    preds = {
        "class": torch.randn(2, 5, 4),
        "start": torch.zeros(2, 5),
        "end": torch.ones(2, 5),
    }
    # num_latents passed DIRECTLY -> T == num_latents (no frequency ambiguity).
    out = detr_to_dense_probs(preds, num_latents=256, n_classes=4)
    assert out.shape == (2, 256, 4)
    assert torch.all(out >= 0)  # accumulated softmax probabilities are non-negative
    sums = out.sum(dim=-1)
    # Fully covered timesteps are renormalized to a proper distribution (sum == 1).
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=0)


def test_detr_to_dense_probs_uncovered_timesteps_are_zero():
    # A single query covering only the first half leaves the second half idle.
    preds = {
        "class": torch.randn(1, 1, 3),
        "start": torch.tensor([[0.0]]),
        "end": torch.tensor([[0.5]]),
    }
    out = detr_to_dense_probs(preds, num_latents=10, n_classes=3)
    sums = out.sum(dim=-1)
    # Covered first half normalizes to 1; idle second half stays exactly 0.
    torch.testing.assert_close(sums[0, :5], torch.ones(5), atol=1e-5, rtol=0)
    torch.testing.assert_close(sums[0, 5:], torch.zeros(5), atol=1e-6, rtol=0)


def test_events_to_mask_background_and_class_rows():
    # CLASS-0 CONTRACT: row 0 = background where idle; real class rows set in span.
    # n_classes=3, n_times=10; one class-2 event over tokens [3, 6).
    mask = events_to_mask([(3, 6, 2)], n_classes=3, n_times=10)
    assert mask.shape == (3, 10)
    # class-2 row is 1 inside the span, 0 outside
    assert mask[2, 3] == 1.0 and mask[2, 5] == 1.0 and mask[2, 6] == 0.0
    # background row 0 is 1 where idle (no event), 0 inside the event span
    assert mask[0, 0] == 1.0 and mask[0, 9] == 1.0
    assert mask[0, 3] == 0.0 and mask[0, 5] == 0.0
    # a class-0 "event" is ignored (no-object), background stays 1 there
    mask2 = events_to_mask([(0, 4, 0)], n_classes=3, n_times=6)
    assert torch.all(mask2[0] == 1.0)
    assert torch.all(mask2[1:] == 0.0)


def test_extract_events_skips_background_argmax():
    # n_classes=3; query 0 argmaxes to class 2, query 1 argmaxes to class 0 (bg).
    outputs = {
        "class": torch.tensor(
            [[[0.0, 0.0, 5.0], [5.0, 0.0, 0.0]]]  # (B=1, Q=2, n_classes=3)
        ),
        "start": torch.tensor([[0.25, 0.10]]),
        "end": torch.tensor([[0.50, 0.90]]),
    }
    events = extract_events_from_detr_batch(outputs, duration=8.0)
    assert len(events) == 1  # one window
    # only the class-2 query survives (class-0 query is dropped as no-object)
    assert len(events[0]) == 1
    s, e, label, conf = events[0][0]
    assert label == 2
    torch.testing.assert_close(torch.tensor(s), torch.tensor(2.0), atol=1e-5, rtol=0)
    torch.testing.assert_close(torch.tensor(e), torch.tensor(4.0), atol=1e-5, rtol=0)
    assert 0.0 <= conf <= 1.0
