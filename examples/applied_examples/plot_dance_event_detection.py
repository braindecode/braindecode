# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: MIT
"""
DANCE: event detection on long EEG windows
===========================================

(Tutorial body added in a later step.)
"""  # noqa: D205
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
