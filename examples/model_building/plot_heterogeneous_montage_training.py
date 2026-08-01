""".. _heterogeneous-montage-training:

Training on recordings with different channels (heterogeneous montages)
=======================================================================

EEG datasets often do not share the same channels: recordings come from
different caps, montages, or vendors. Stacking such recordings into a single
training batch is normally impossible -- the signal tensors have different
numbers of channels and cannot be concatenated.

This example shows braindecode's tools for training a **position-aware** model
across recordings with different channel sets:

* :meth:`~braindecode.datasets.BaseConcatDataset.set_return_ch_pos` makes every
  window carry its electrode positions ``(n_ch, 3)`` (x, y, z), so channel
  identity is encoded by *where* each electrode sits rather than by its row
  index.
* :func:`~braindecode.datasets.pad_channels_collate` pads each batch to the
  largest channel count present and returns a boolean ``ch_mask`` marking the
  real (non-padded) channels.

A model that consumes positions (and ignores padded channels via the mask) can
then train on the mixed collection. Here we use a tiny illustrative model;
swap in any position-aware architecture (e.g. ``REVE``).

.. contents:: This example covers:
   :local:
   :depth: 2

"""

# Authors: The braindecode developers
#
# License: BSD (3-clause)

import mne
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from braindecode import EEGClassifier
from braindecode.datasets import (
    BaseConcatDataset,
    RawDataset,
    pad_channels_collate,
)
from braindecode.preprocessing import create_fixed_length_windows

mne.set_log_level("ERROR")

######################################################################
# Build two recordings with different channel sets
# ------------------------------------------------
#
# We synthesise two short recordings: one with 8 channels and one with 10, both
# using ``standard_1020`` electrode positions. In practice these would be real
# datasets loaded with different montages.


def make_recording(ch_names, label, seed):
    info = mne.create_info(ch_names, sfreq=100.0, ch_types="eeg")
    data = np.random.RandomState(seed).randn(len(ch_names), 4000) * 1e-6
    raw = mne.io.RawArray(data, info)
    raw.set_montage("standard_1020")
    return RawDataset(raw, description={"label": label})


chs_a = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"]  # 8 channels
chs_b = ["Fz", "Cz", "Pz", "Oz", "T7", "T8", "O1", "O2", "F7", "F8"]  # 10 channels

concat = BaseConcatDataset(
    [
        make_recording(chs_a, label=0, seed=1),
        make_recording(chs_b, label=1, seed=2),
        make_recording(chs_a, label=0, seed=3),
        make_recording(chs_b, label=1, seed=4),
    ]
)

######################################################################
# Window the recordings and enable channel positions
# --------------------------------------------------
#
# After cutting fixed-length windows we assign the per-recording ``label`` as
# the target and turn on position returning for the whole collection.

windows = create_fixed_length_windows(
    concat,
    window_size_samples=200,
    window_stride_samples=200,
    drop_last_window=True,
    preload=True,
)
windows.set_target("label")
windows.set_return_ch_pos(True)

######################################################################
# Inspect a heterogeneous batch
# -----------------------------
#
# ``pad_channels_collate`` pads each batch to the largest channel count (10
# here) and returns a boolean ``ch_mask``. Because we mix 8- and 10-channel
# recordings, batches contain both -- the mask tells real channels from padding.

loader = DataLoader(
    windows, batch_size=4, shuffle=True, collate_fn=pad_channels_collate
)
X, y, crop_inds, ch_pos, ch_mask = next(iter(loader))
print("X       :", tuple(X.shape))  # (batch, max_ch, n_times)
print("ch_pos  :", tuple(ch_pos.shape))  # (batch, max_ch, 3)
print("ch_mask :", tuple(ch_mask.shape))  # (batch, max_ch) bool
print("real channels per sample:", ch_mask.sum(1).tolist())

######################################################################
# A minimal position-aware model
# ------------------------------
#
# This tiny model is permutation-invariant over channels: it embeds each
# channel from a small signal summary plus its (x, y, z) position, then
# **masked-mean-pools** over channels so padded channels are ignored. Any model
# that accepts ``forward(x, pos=None, ch_mask=None)`` works the same way -- this
# is the signature braindecode routes positions and the mask into.


class TinyPositionalNet(nn.Module):
    def __init__(self, n_outputs=2, dim=32):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(6, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.head = nn.Linear(dim, n_outputs)

    def forward(self, x, pos=None, ch_mask=None):
        # Per-channel signal summary: mean, std, max over time -> (B, C, 3).
        feat = torch.stack([x.mean(-1), x.std(-1), x.amax(-1)], dim=-1)
        if pos is None:
            pos = torch.zeros_like(feat)
        h = self.embed(torch.cat([feat, pos], dim=-1))  # (B, C, dim)
        if ch_mask is not None:
            m = ch_mask.unsqueeze(-1).float()
            h = (h * m).sum(1) / m.sum(1).clamp(min=1)  # masked mean over channels
        else:
            h = h.mean(1)
        return self.head(h)


######################################################################
# Train with EEGClassifier
# ------------------------
#
# We pass ``pad_channels_collate`` as the iterator's ``collate_fn``. braindecode
# routes the signal, positions and mask into the model's ``forward`` for you.

clf = EEGClassifier(
    TinyPositionalNet(n_outputs=2),
    max_epochs=3,
    batch_size=4,
    train_split=None,
    classes=[0, 1],
    iterator_train__collate_fn=pad_channels_collate,
    iterator_train__shuffle=True,
    iterator_train__drop_last=False,
)
clf.fit(windows, y=None)

######################################################################
# Next steps
# ----------
#
# The data layer here makes a heterogeneous collection **batchable** and feeds
# positions plus a channel mask to the model. How a model *normalizes* and
# consumes variable channels (e.g. applying ``ch_mask`` inside attention, or
# mapping electrode positions to a canonical space) is model-specific and the
# natural next step when adapting a real architecture such as ``REVE``.
