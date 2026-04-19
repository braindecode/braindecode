# type: ignore
r""".. _channel-interpolation-tutorial:

Adapting Foundation Models to Arbitrary Channel Sets
=====================================================

Most pretrained EEG foundation models are trained on a fixed,
canonical channel montage (e.g. LaBraM uses a specific 128-channel
set, BIOT a specific 18-channel TCP montage).  When you want to
fine-tune one of these models on a dataset whose channel layout is
different — either a different number of channels or a different
spatial arrangement — you hit a wall: the model simply cannot be
instantiated on your data.

braindecode ships an **experimental** solution for this: the
``Interpolated*`` family of model wrappers.  They insert a frozen
(by default) spatial-interpolation matrix before the backbone so
that, from the backbone's perspective, the input always has the
canonical channel set it was trained on.

This tutorial covers:

* what the underlying :class:`ChannelInterpolationLayer` does,
* how to use the shipped :class:`InterpolatedLaBraM`,
  :class:`InterpolatedBIOT` and :class:`InterpolatedSignalJEPA`
  variants on your own channel layout,
* the two interpolation ``mode``\\ s (``"name_match"`` vs
  ``"always"``) and when to prefer each,
* the ``trainable`` flag for data-driven spatial filters.

.. warning::

   The ``Interpolated*`` API is experimental and may change without
   a deprecation cycle.

.. contents:: This example covers:
   :local:
   :depth: 2
"""

# Authors: Pierre Guetschel <pierre.guetschel@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
import mne
import numpy as np
import torch

from braindecode.datasets import MOABBDataset
from braindecode.models import (
    InterpolatedBIOT,
    InterpolatedLaBraM,
    InterpolatedSignalJEPA,
    Labram,
)
from braindecode.modules import ChannelInterpolationLayer

torch.manual_seed(0)
np.random.seed(0)

######################################################################
# The problem: your channels are not the model's channels
# --------------------------------------------------------
#
# We load one subject of the BNCI2014_001 motor-imagery dataset.  Its
# 22-channel layout is a subset of the standard 10-20 montage.
#

subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])

montage = mne.channels.make_standard_montage("standard_1020")
for ds in dataset.datasets:
    ds.raw.pick_types(eeg=True)
    ds.raw.set_montage(montage)

chs_info = dataset.datasets[0].raw.info["chs"]
user_ch_names = [ch["ch_name"] for ch in chs_info]
print(f"Dataset has {len(chs_info)} channels: {user_ch_names}")

######################################################################
# LaBraM was pretrained on a canonical 128-channel layout.  Passing
# our 22-channel montage directly is rejected at construction time:
#

try:
    Labram(chs_info=chs_info, n_times=1000, n_outputs=4, patch_size=200)
except ValueError as exc:
    print(f"Labram refused our chs_info:\n  {exc}")

######################################################################
# Before the ``Interpolated*`` wrappers existed, adapting a
# foundation model to a new montage required manual surgery on the
# first spatial layer.  We can now do it in one line.
#

######################################################################
# ``InterpolatedLaBraM``: one-line channel adaptation
# ----------------------------------------------------
#
# ``InterpolatedLaBraM`` is a subclass of :class:`Labram` that
# **prepends** a frozen spatial-interpolation layer before the
# backbone.  From the user's side, it accepts any ``chs_info``; from
# the backbone's side, the input is always the canonical
# 128-channel tensor the pretrained weights expect.
#

model = InterpolatedLaBraM(
    chs_info=chs_info,
    n_times=1000,
    n_outputs=4,
    patch_size=200,
)
x = torch.randn(2, len(chs_info), 1000)
model.eval()
with torch.no_grad():
    out = model(x)
print(f"Input shape  : {tuple(x.shape)}")
print(f"Output shape : {tuple(out.shape)}")

######################################################################
# The user-facing view of the model reports the *user's* channel
# count, not the canonical one, so the model plays nicely with the
# rest of braindecode (``EEGClassifier``, cross-validation, logging):
#

print(f"model.n_chans          = {model.n_chans}")
print(f"len(model.chs_info)    = {len(model.chs_info)}")

######################################################################
# Internally, an ``interpolation_layer`` has been inserted:
#

print(model.interpolation_layer)

######################################################################
# Under the hood: ``ChannelInterpolationLayer``
# ----------------------------------------------
#
# The layer is a thin ``nn.Module`` whose forward is a single matrix
# multiplication ``W @ x``, where ``W`` has shape
# ``(n_target_chans, n_source_chans)``.
#
# Two ``mode``\\ s control how ``W`` is built:
#
# * ``"name_match"`` (default): for every target channel whose name
#   also appears in the source, use the identity mapping for that row.
#   Target channels without a source-name match fall back to MNE's
#   spatial interpolation.
# * ``"always"``: every row is computed from MNE's spline
#   interpolation based on 3D sensor positions, even when names
#   would have matched.
#
# We build both layers on top of LaBraM's canonical 128-channel set
# to compare:
#

target_chs_info = InterpolatedLaBraM._TARGET_CHS_INFO

layer_name_match = ChannelInterpolationLayer(
    src_chs_info=chs_info,
    tgt_chs_info=target_chs_info,
    mode="name_match",
)
layer_always = ChannelInterpolationLayer(
    src_chs_info=chs_info,
    tgt_chs_info=target_chs_info,
    mode="always",
)

W_name = layer_name_match.matrix
W_always = layer_always.matrix

print(f"name_match : {W_name.shape}, nonzeros = {int(W_name.count_nonzero())}")
print(f"always     : {W_always.shape}, nonzeros = {int(W_always.count_nonzero())}")

######################################################################
# ``name_match`` produces a sparse (mostly zero) matrix because the
# target rows whose name is one of our 22 input channels are
# one-hot.  ``always`` produces a dense matrix because every row is
# a spatial interpolation over all 22 input channels.
#
# The two matrices look very different — let's visualise them side
# by side:
#

fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=110)
for ax, W, title in zip(
    axes,
    [W_name.abs().numpy(), W_always.abs().numpy()],
    ['mode="name_match"', 'mode="always"'],
):
    im = ax.imshow(W, aspect="auto", cmap="magma", vmin=0, vmax=max(W.max(), 1e-6))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("source channel (user)")
    ax.set_ylabel("target channel (canonical)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle(
    "Interpolation matrix |W| — target (128) × source (22)",
    fontsize=12,
    y=1.02,
)
fig.tight_layout()
plt.show()

######################################################################
# **Which mode should you pick?**
#
# * Prefer ``"name_match"`` when many of your channels share names
#   with the canonical set: you preserve the pretrained weights'
#   original mapping for those channels and only interpolate the
#   missing ones.
# * Prefer ``"always"`` when channel names are unreliable (e.g.
#   reference schemes differ) and you trust positions more — or
#   when you want a smoother, fully spatial projection.
#
# The default is ``"name_match"`` because it is the safer bet for
# linear probing (no source channel is altered when it could have
# been passed through as-is).

######################################################################
# Other shipped variants
# ----------------------
#
# The same pattern applies to :class:`InterpolatedBIOT` (18 canonical
# bipolar channels) and :class:`InterpolatedSignalJEPA` (62 canonical
# channels).  All three accept your ``chs_info`` directly:
#

model_biot = InterpolatedBIOT(
    chs_info=chs_info,
    n_outputs=4,
    n_times=1000,
    sfreq=250,
)
with torch.no_grad():
    out_biot = model_biot(x)
print(f"InterpolatedBIOT  : input {tuple(x.shape)} -> output {tuple(out_biot.shape)}")

model_sjepa = InterpolatedSignalJEPA(
    chs_info=chs_info,
    n_outputs=4,
    n_times=512,
    sfreq=128,
)
x_sjepa = torch.randn(2, len(chs_info), 512)
with torch.no_grad():
    out_sjepa = model_sjepa(x_sjepa)
print(
    f"InterpolatedSignalJEPA: input {tuple(x_sjepa.shape)} "
    f"-> features {tuple(out_sjepa.shape)}"
)

######################################################################
# Loading pretrained weights
# --------------------------
#
# ``Interpolated*`` classes are strict subclasses of their
# backbones, so the pretrained weights on the Hugging Face Hub
# load without any renaming.  The only extra parameter is the
# interpolation matrix, which is absent from the checkpoint
# (it is derived from your ``chs_info``) — use ``strict=False``:
#
# .. code-block:: python
#
#    model = InterpolatedLaBraM.from_pretrained(
#        "braindecode/labram-pretrained",
#        chs_info=chs_info,
#        n_outputs=4,
#        n_times=1000,
#        strict=False,  # interpolation_layer.matrix is not in the checkpoint
#    )
#
# For the full fine-tuning workflow — including freezing the
# backbone and training only a new head — see
# :ref:`finetune-foundation-model`.

######################################################################
# Making the interpolation trainable
# -----------------------------------
#
# By default the interpolation matrix is a **frozen buffer**: it is
# computed once from ``chs_info`` at construction time and never
# updated during training.  This is the right choice for linear
# probing, because it guarantees that the pretrained weights see
# the exact input they were trained on.
#
# For larger fine-tuning budgets you may want to let the network
# learn a better spatial projection.  Pass ``trainable=True``:
#

model_trainable = InterpolatedLaBraM(
    chs_info=chs_info,
    n_times=1000,
    n_outputs=4,
    patch_size=200,
    trainable=True,
)
interp_params = [
    (name, p.requires_grad)
    for name, p in model_trainable.named_parameters()
    if "interpolation_layer" in name
]
print("Trainable interpolation parameters:")
for name, grad in interp_params:
    print(f"  {name}  requires_grad={grad}")

######################################################################
# In this mode the matrix is an ``nn.Parameter`` initialised from
# the same MNE spline interpolation, and is optimised jointly with
# the backbone.

######################################################################
# Summary
# -------
#
# * ``Interpolated*`` wrappers let you use any foundation model on
#   any channel layout by prepending a spatial-projection layer.
# * The projection is frozen by default (safe for linear probing)
#   and can be made trainable for larger fine-tunes.
# * Two modes — ``"name_match"`` and ``"always"`` — trade off
#   *preserve source channels* vs *smoother projection*.
# * The shipped variants are :class:`InterpolatedLaBraM`,
#   :class:`InterpolatedBIOT` and :class:`InterpolatedSignalJEPA`;
#   for custom backbones use :func:`InterpolatedModel`.
#
