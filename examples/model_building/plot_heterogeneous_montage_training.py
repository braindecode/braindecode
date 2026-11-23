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

import matplotlib.pyplot as plt
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
# using ``colin27_1020`` electrode positions. In practice these would be real
# datasets loaded with different montages.


def make_recording(ch_names, label, seed):
    info = mne.create_info(ch_names, sfreq=100.0, ch_types="eeg")
    data = np.random.RandomState(seed).randn(len(ch_names), 4000) * 1e-6
    raw = mne.io.RawArray(data, info)
    raw.set_montage("colin27_1020")
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
    on_last_window="drop",
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
    windows,
    batch_size=4,
    shuffle=True,
    collate_fn=pad_channels_collate,
    generator=torch.Generator().manual_seed(0),  # a batch that mixes both caps
)
X, y, crop_inds, ch_pos, ch_mask = next(iter(loader))
print("X       :", tuple(X.shape))  # (batch, max_ch, n_times)
print("ch_pos  :", tuple(ch_pos.shape))  # (batch, max_ch, 3)
print("ch_mask :", tuple(ch_mask.shape))  # (batch, max_ch) bool
print("real channels per sample:", ch_mask.sum(1).tolist())

######################################################################
# Panel A places the two caps on the scalp. Panel B shows the mask of the
# batch: each row is one window, each column one channel slot of the padded
# tensor. Windows from the 8-channel cap leave the last two slots padded.

BLUE, VERMILION, INK = "#0072B2", "#D55E00", "#14181F"
plt.rcdefaults()  # importing MOABB applies seaborn's "whitegrid" style globally
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)

ch_pos = mne.channels.make_standard_montage("colin27_1020").get_positions()["ch_pos"]
# Electrode positions seen from above, in cm, centered on the drawn electrodes
xy_all = 100 * np.array([ch_pos[ch][:2] for ch in chs_a + chs_b])
center = xy_all.mean(axis=0)
radius = 1.15 * np.linalg.norm(xy_all - center, axis=1).max()
fig, (ax_cap, ax_mask) = plt.subplots(
    1, 2, figsize=(9, 3.8), gridspec_kw={"width_ratios": [1, 1.3]}
)

# A: electrode positions seen from above (x, y in cm)
for chs, color, label in [
    (chs_a, BLUE, f"{len(chs_a)}-channel cap"),
    (chs_b, VERMILION, f"{len(chs_b)}-channel cap"),
]:
    xy = 100 * np.array([ch_pos[ch][:2] for ch in chs]) - center
    ax_cap.scatter(xy[:, 0], xy[:, 1], s=60, color=color, label=label, zorder=3)
    for ch, (x, y) in zip(chs, xy):
        ax_cap.annotate(
            ch,
            (x, y),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=color,
        )
ax_cap.add_patch(plt.Circle((0, 0), radius, fill=False, color="#B0B0B0", lw=0.8))
ax_cap.set_aspect("equal")
ax_cap.set_xlim(-1.05 * radius, 1.05 * radius)
ax_cap.set_ylim(-1.05 * radius, 1.05 * radius)
ax_cap.axis("off")
ax_cap.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
ax_cap.set_title("A  Two montages", loc="left", fontweight="bold", color=INK)

# B: real (cap color) versus padded (white) channel slots of the batch
n_real = ch_mask.sum(1).numpy()
for row, mask_row in enumerate(ch_mask.numpy()):
    color = BLUE if n_real[row] == len(chs_a) else VERMILION
    for slot, is_real in enumerate(mask_row):
        ax_mask.add_patch(
            plt.Rectangle(
                (slot, row),
                1,
                1,
                facecolor=color if is_real else "white",
                edgecolor="#B0B0B0",
                lw=0.6,
            )
        )
    ax_mask.text(
        len(mask_row) + 0.2,
        row + 0.5,
        f"{n_real[row]} real",
        va="center",
        color=color,
        fontsize=9,
    )
ax_mask.set_xlim(0, ch_mask.shape[1])
ax_mask.set_ylim(ch_mask.shape[0], 0)
ax_mask.set_xticks(np.arange(ch_mask.shape[1]) + 0.5, np.arange(ch_mask.shape[1]))
ax_mask.set_yticks(np.arange(ch_mask.shape[0]) + 0.5, np.arange(ch_mask.shape[0]))
ax_mask.set_xlabel("Channel slot in the padded batch")
ax_mask.set_ylabel("Window in the batch")
ax_mask.set_title(
    "B  Channel mask (white = padding)", loc="left", fontweight="bold", color=INK
)
fig.tight_layout()

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
