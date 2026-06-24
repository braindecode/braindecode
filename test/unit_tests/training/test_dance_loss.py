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
from braindecode.training import DanceLoss
from braindecode.training.losses import HungarianMatcher


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


def _targets(b=2, max_events=3, n_classes=4):
    return {
        "start": torch.tensor([[0.1, 0.5, 0.0]] * b),
        "end": torch.tensor([[0.2, 0.7, 0.0]] * b),
        "class": torch.tensor([[1, 2, 0]] * b),  # 0 = padding
    }


def _preds(b=2, q=100, n_classes=4, num_latents=256):
    # ``dense`` time dim MUST equal the loss's ``num_latents`` (the fixed latent
    # grid); the real DANCE dense head always emits ``num_latents`` tokens.
    return {
        "class": torch.randn(b, q, n_classes, requires_grad=True),
        "start": torch.rand(b, q, requires_grad=True),
        "end": torch.rand(b, q, requires_grad=True),
        "dense": torch.randn(b, num_latents, n_classes, requires_grad=True),
    }


def test_matcher_returns_matched_structures():
    matcher = HungarianMatcher(weight_class=1.0, weight_iou=5.0)
    mp, mt, matches = matcher(_preds(), _targets())
    assert "class" in mp and "start" in mp
    assert mt["start"].shape == mp["start"].shape


def test_dance_loss_finite():
    loss_fn = DanceLoss(num_latents=256)
    loss, details = loss_fn(_preds(), _targets(), duration=32.0)
    assert torch.isfinite(loss)
    assert {"class_loss", "iou_loss", "dense_loss", "consistency_loss"} <= set(details)


def test_dance_loss_decreases_on_overfit():
    torch.manual_seed(0)
    loss_fn = DanceLoss(num_latents=64)
    preds = _preds(num_latents=64)
    targets = _targets()
    params = [preds["class"], preds["start"], preds["end"], preds["dense"]]
    opt = torch.optim.Adam(params, lr=0.05)
    first = None
    for step in range(40):
        opt.zero_grad()
        loss, _ = loss_fn(preds, targets, duration=32.0)
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss)
    assert float(loss) < first  # overfit batch -> loss goes down


import importlib.util
from pathlib import Path

_EX = Path(__file__).resolve().parents[3] / "examples" / "applied_examples" / \
    "plot_dance_event_detection.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("_dance_ex", _EX)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_target_builder_synthetic():
    ex = _load_example()
    # window [10s, 42s] (32 s); one event 16-24 s, class 2
    ann = [(16.0, 24.0, 2), (100.0, 110.0, 3)]  # 2nd is outside the window
    tgt = ex.dance_target_builder(
        ann, window_onset=10.0, window_duration=32.0,
        max_events=5, num_latents=256,
    )
    assert tgt["start"].shape == (5,)
    assert tgt["class"].tolist().count(0) == 4  # only one in-window event
    # event maps to [ (16-10)/32, (24-10)/32 ] = [0.1875, 0.4375]
    torch.testing.assert_close(tgt["start"][0], torch.tensor(0.1875), atol=1e-4, rtol=0)
    torch.testing.assert_close(tgt["end"][0], torch.tensor(0.4375), atol=1e-4, rtol=0)
    assert int(tgt["class"][0]) == 2
    # dense target: tokens for [0.1875*256, 0.4375*256] = [48, 112] set to 2
    assert int(tgt["dense"][48]) == 2 and int(tgt["dense"][111]) == 2
    assert int(tgt["dense"][0]) == 0


def test_dance_tutorial_pipeline_runs_end_to_end():
    """Execute the tutorial's data path + one train/eval step on the synthetic
    recording. Guards against the gallery never running under html-noplot."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from braindecode.models import DANCE
    from braindecode.preprocessing import create_fixed_length_windows
    from braindecode.training import DanceLoss, f1_event
    from braindecode.functional import extract_events_from_detr_batch

    ex = _load_example()  # defined in Task 10's test section
    sfreq, window_s, n_classes, num_latents = 200.0, 32.0, 4, 256
    win = int(window_s * sfreq)  # 6400
    concat_ds = ex.build_synthetic_event_dataset(
        n_chans=19, sfreq=sfreq, n_seconds=128.0, n_classes=n_classes, seed=1
    )
    raw = concat_ds.datasets[0].raw
    events = ex.annotations_to_events(raw, n_classes)
    assert len(events) > 0  # synthetic recording really has events

    windows_ds = create_fixed_length_windows(
        concat_ds, window_size_samples=win, window_stride_samples=win,
        drop_last_window=True, preload=True, use_mne_epochs=False,
    )
    assert len(windows_ds) >= 2  # 128 s / 32 s -> 4 windows

    samples = []
    for i in range(len(windows_ds)):
        x, _, crop = windows_ds[i]
        eeg = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        tgt = ex.dance_target_builder(
            events, float(crop[1]) / sfreq, window_s,
            max_events=16, num_latents=num_latents,
        )
        assert eeg.shape == (19, win)
        assert tgt["dense"].shape == (num_latents,)
        samples.append((eeg, tgt))
    # at least one window must contain a real (non-padding) event
    assert any(int((s[1]["class"] != 0).sum()) > 0 for s in samples)

    loader = DataLoader(samples, batch_size=2, collate_fn=ex.dance_collate)
    model = DANCE(
        n_outputs=n_classes, n_chans=19, chs_info=raw.info["chs"],
        n_times=win, sfreq=sfreq, input_window_seconds=window_s,
    )
    criterion = DanceLoss(num_latents=num_latents)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    batch = next(iter(loader))
    out = model.detect(batch["eeg"])
    assert out["class"].shape == (2, 100, n_classes)
    assert out["dense"].shape == (2, num_latents, n_classes)
    loss, details = criterion(out, batch, duration=window_s)
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
        ev = extract_events_from_detr_batch(model.detect(batch["eeg"]), window_s)
    assert len(ev) == 2  # one event-list per window in the batch
    gt = [
        (float(s) * window_s, float(e) * window_s, int(c))
        for s, e, c in zip(batch["start"][0], batch["end"][0], batch["class"][0])
        if int(c) != 0
    ]
    preds = [(s, e, c) for (s, e, c, _conf) in ev[0]]
    f1 = f1_event(preds, gt, iou_threshold=0.5)  # finite, in [0, 1]
    assert 0.0 <= f1 <= 1.0
