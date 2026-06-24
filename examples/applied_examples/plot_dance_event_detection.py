# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: MIT
"""
DANCE: event detection on long EEG windows
===========================================

This tutorial shows how to use the :class:`~braindecode.models.DANCE` model for
**event detection** on long EEG windows with plain PyTorch (no skorch, no
Lightning). DANCE reframes EEG decoding as a DETR-style set-prediction problem:
instead of one label per window, it predicts a *set* of ``(start, end, class)``
events directly from a continuous recording.

The pipeline is:

1. take a long, continuous EEG recording carrying onset/duration class
   annotations,
2. cut it into fixed-length (32 s) windows with
   :func:`~braindecode.preprocessing.create_fixed_length_windows`,
3. turn each window's annotations into a DANCE target dict (normalized
   ``start``/``end`` in ``[0, 1]`` plus a dense per-token class map),
4. train :class:`~braindecode.models.DANCE` with
   :class:`~braindecode.training.DanceLoss`, and
5. evaluate with the event-level :func:`~braindecode.training.f1_event` and the
   per-token :func:`~braindecode.training.f1_sample` metrics.

.. note::
    To keep this example fully self-contained and runnable offline (it executes
    end-to-end in the docs build), we use a small **synthetic** recording built
    with :class:`mne.io.RawArray` and :class:`mne.Annotations`. For real data,
    load a continuous recording with
    :class:`~braindecode.datasets.MOABBDataset` (e.g. a P300 paradigm) and read
    its event annotations the same way. The long-window event-detection framing
    on a specific MOABB dataset is *not* shown in an executed cell here because
    most MOABB P300 datasets ship short, pre-epoched trials rather than the
    continuous raw recording this pipeline expects; adapt the data-loading step
    to your continuous recording while keeping the rest of the pipeline
    unchanged.

.. note::
    DANCE follows a DETR-style **class-0 = background / no-object** convention:
    real event classes are ``1 .. n_outputs - 1`` and class ``0`` is reserved
    for "no event". The target builder and both F1 metrics honour this
    convention automatically.
"""  # noqa: D205, E501
from __future__ import annotations

import torch


def dance_target_builder(
    annotations, window_onset, window_duration, max_events, num_latents
):
    """Build a DANCE target dict from one window's (onset, dur, class) events.

    Annotations are absolute-time ``(start_s, end_s, class_int)`` tuples; only
    those overlapping ``[window_onset, window_onset + window_duration]`` are
    kept, clipped, and normalized to ``[0, 1]`` within the window.

    CLASS-0 CONTRACT (Global Constraints): class id ``0`` is the single shared
    no-object/background id. Unused event slots are padded with class ``0``
    (``cls[kept:]`` stays 0), idle dense tokens stay 0 (= background), and any
    annotation whose ``int(c) == 0`` is skipped (cannot be told apart from
    padding, by design). Real classes are ``1..n_outputs-1``. This is exactly
    what the matcher and dense CE target in Task 9 expect.
    """
    start = torch.zeros(max_events)
    end = torch.zeros(max_events)
    cls = torch.zeros(max_events, dtype=torch.long)
    w0, wd = window_onset, window_duration
    kept = 0
    for s, e, c in annotations:
        s_c = max(s, w0)
        e_c = min(e, w0 + wd)
        if e_c <= s_c or int(c) == 0 or kept >= max_events:
            continue
        start[kept] = (s_c - w0) / wd
        end[kept] = (e_c - w0) / wd
        cls[kept] = int(c)
        kept += 1
    dense = torch.zeros(num_latents, dtype=torch.long)
    s_tok = (start * num_latents).clamp(0, num_latents).long()
    e_tok = (end * num_latents).clamp(0, num_latents).long()
    for i in range(kept):
        a, b = int(s_tok[i]), int(e_tok[i])
        if a < b:
            dense[a:b] = int(cls[i])
    return {"start": start, "end": end, "class": cls, "dense": dense}


def dance_collate(batch):
    """Collate ``[(eeg, target_dict), ...]`` into a batched DANCE dict."""
    eeg = torch.stack([b[0] for b in batch])
    out = {"eeg": eeg}
    for key in ("start", "end", "class", "dense"):
        out[key] = torch.stack([b[1][key] for b in batch])
    return out


def build_synthetic_event_dataset(
    n_chans=19, sfreq=200.0, n_seconds=192.0, n_classes=4, seed=0
):
    """A self-contained long EEG recording with onset/duration class events.

    Returns a ``BaseConcatDataset`` of one ``BaseDataset`` wrapping an
    ``mne.io.RawArray`` (montaged so ``chs_info`` carries real locations for
    the ChannelMerger) with ``mne.Annotations`` of ``(onset, duration,
    "classK")`` for K in ``1..n_classes-1`` (class 0 = background, never
    annotated). Realistic stand-in for a continuous P300/event recording.
    """
    import mne
    import numpy as np

    from braindecode.datasets.base import BaseConcatDataset, BaseDataset

    rng = np.random.default_rng(seed)
    n_samples = int(n_seconds * sfreq)
    data = rng.standard_normal((n_chans, n_samples)) * 1e-6  # ~uV scale
    # Real 10-20 labels (they exist in the standard_1020 montage, unlike
    # "E1".."E19"), so set_montage actually assigns finite locations and DANCE
    # can derive non-degenerate positions to drive the ChannelMerger.
    ch_names = [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
        "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
    ][:n_chans]
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="error")
    # Real 10-20-ish locations so DANCE derives non-degenerate positions.
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(
        montage, match_case=False, on_missing="ignore", verbose="error"
    )
    # Scatter events across the recording: ~1 event / 6 s, duration 0.5-2 s,
    # class in 1..n_classes-1. Inject signal so the model has something to learn.
    onsets, durations, descriptions = [], [], []
    t = 3.0
    while t < n_seconds - 3.0:
        dur = float(rng.uniform(0.5, 2.0))
        cls = int(rng.integers(1, n_classes))
        s0, s1 = int(t * sfreq), int((t + dur) * sfreq)
        data[:, s0:s1] += cls * 5e-6  # class-dependent bump
        onsets.append(t)
        durations.append(dur)
        descriptions.append(f"class{cls}")
        t += float(rng.uniform(4.0, 8.0))
    raw.set_annotations(
        mne.Annotations(onsets, durations, descriptions), verbose="error"
    )
    ds = BaseDataset(raw, description={"subject": 0})
    return BaseConcatDataset([ds])


def annotations_to_events(raw, n_classes):
    """Read absolute-time ``(start_s, end_s, class_int)`` events from a raw's
    annotations (``"classK"`` -> K). Class 0 is never produced."""
    events = []
    for ann in raw.annotations:
        desc = str(ann["description"])
        if not desc.startswith("class"):
            continue
        cls = int(desc[len("class"):])
        if cls <= 0 or cls >= n_classes:
            continue
        events.append((float(ann["onset"]), float(ann["onset"] + ann["duration"]), cls))
    return events


# %%
# Build a long-window recording and cut 32 s windows
# ----------------------------------------------------
import numpy as np
from torch.utils.data import DataLoader

from braindecode.functional import extract_events_from_detr_batch
from braindecode.models import DANCE
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.training import DanceLoss, f1_event, f1_sample

SFREQ, WINDOW_S, N_CLASSES, NUM_LATENTS, MAX_EVENTS = 200.0, 32.0, 4, 256, 16
WINDOW_SAMPLES = int(WINDOW_S * SFREQ)  # 6400

concat_ds = build_synthetic_event_dataset(
    n_chans=19, sfreq=SFREQ, n_seconds=192.0, n_classes=N_CLASSES
)
raw = concat_ds.datasets[0].raw
chs_info = raw.info["chs"]
all_events = annotations_to_events(raw, N_CLASSES)

windows_ds = create_fixed_length_windows(
    concat_ds,
    window_size_samples=WINDOW_SAMPLES,
    window_stride_samples=WINDOW_SAMPLES,
    drop_last_window=True,
    preload=True,
    use_mne_epochs=False,  # guarantees EEGWindowsDataset -> (X, y, crop_inds)
)

# %%
# Map each window's annotations to DANCE targets
# -----------------------------------------------
# Each fixed-length window starts at i * WINDOW_S seconds; build its target dict
# by selecting the recording events overlapping that window.
samples = []
for i in range(len(windows_ds)):
    x, _, crop_inds = windows_ds[i]
    eeg = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    window_onset = float(crop_inds[1]) / SFREQ  # i_start_in_trial / sfreq
    target = dance_target_builder(
        all_events, window_onset=window_onset, window_duration=WINDOW_S,
        max_events=MAX_EVENTS, num_latents=NUM_LATENTS,
    )
    samples.append((eeg, target))
loader = DataLoader(samples, batch_size=4, shuffle=True, collate_fn=dance_collate)

# %%
# Build the model and the criterion
# ----------------------------------
model = DANCE(
    n_outputs=N_CLASSES, n_chans=len(chs_info), chs_info=chs_info,
    n_times=WINDOW_SAMPLES, sfreq=SFREQ, input_window_seconds=WINDOW_S,
)
criterion = DanceLoss(num_latents=NUM_LATENTS)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# %%
# Train a few steps
# -----------------
model.train()
for epoch in range(2):
    for batch in loader:
        optimizer.zero_grad()
        out = model.detect(batch["eeg"])
        loss, details = criterion(out, batch, duration=WINDOW_S)
        loss.backward()
        optimizer.step()

# %%
# Evaluate with F1-event and F1-sample
# ------------------------------------
from braindecode.functional import events_to_mask

model.eval()
ev_f1s, samp_f1s = [], []
with torch.no_grad():
    for batch in loader:
        out = model.detect(batch["eeg"])
        pred_events = extract_events_from_detr_batch(out, duration=WINDOW_S)
        for bi in range(batch["eeg"].shape[0]):
            # ground-truth events of this window, in seconds within the window
            gt = [
                (float(s) * WINDOW_S, float(e) * WINDOW_S, int(c))
                for s, e, c in zip(batch["start"][bi], batch["end"][bi], batch["class"][bi])
                if int(c) != 0
            ]
            preds = [(s, e, c) for (s, e, c, _conf) in pred_events[bi]]
            ev_f1s.append(f1_event(preds, gt, iou_threshold=0.5))
            pred_mask = events_to_mask(
                [(int(s / WINDOW_S * NUM_LATENTS), int(e / WINDOW_S * NUM_LATENTS), c)
                 for s, e, c in preds],
                N_CLASSES, NUM_LATENTS,
            )
            gt_mask = events_to_mask(
                [(int(s / WINDOW_S * NUM_LATENTS), int(e / WINDOW_S * NUM_LATENTS), c)
                 for s, e, c in gt],
                N_CLASSES, NUM_LATENTS,
            )
            samp_f1s.append(f1_sample(pred_mask, gt_mask))
print(f"F1-event={np.mean(ev_f1s):.3f}  F1-sample={np.mean(samp_f1s):.3f}")
