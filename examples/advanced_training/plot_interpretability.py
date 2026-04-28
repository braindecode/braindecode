""".. _interpretability-tutorial:

Interpretability of EEG Decoders
================================

This tutorial walks through the gradient-based attribution and
topographic-projection utilities in :mod:`braindecode.visualization`
applied to a motor-imagery decoder.

We train a small :class:`~braindecode.models.ShallowFBCSPNet` on the BCI
Competition IV 2a dataset, pick a few correctly-classified trials, and
ask: *which channels and time samples drove the model's prediction?*

The visualization API is intentionally minimal — every method is a plain
PyTorch function on top of :func:`torch.autograd.grad`, no extra
dependencies:

- :func:`~braindecode.visualization.saliency` — ``|∂y[target] / ∂x|``
- :func:`~braindecode.visualization.input_x_gradient` — element-wise
  ``x * ∂y/∂x``
- :func:`~braindecode.visualization.integrated_gradients` — path integral
  from a baseline (Sundararajan et al., 2017)
- :func:`~braindecode.visualization.layer_grad_cam` — class-discriminative
  localization at a chosen layer
- :func:`~braindecode.visualization.project_to_topomap` — per-channel
  values to a 2-D scalp map via MNE
- :func:`~braindecode.visualization.compute_metrics` — quantitative
  comparison of attribution maps against a reference

.. contents:: This example covers:
   :local:
   :depth: 2
"""

# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

######################################################################
# Loading and preparing the data
# ------------------------------
#
# We reuse the BCI Competition IV 2a setup from the basic training
# tutorial: a single subject, bandpass-filtered to 4–38 Hz, with
# exponential moving standardization.

from numpy import multiply

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    exponential_moving_standardize,
    preprocess,
)

subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])

low_cut_hz, high_cut_hz = 4.0, 38.0
factor_new, init_block_size, factor = 1e-3, 1000, 1e6

preprocess(
    dataset,
    [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        Preprocessor(lambda data: multiply(data, factor)),
        Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),
        Preprocessor(
            exponential_moving_standardize,
            factor_new=factor_new,
            init_block_size=init_block_size,
        ),
    ],
    n_jobs=-1,
)

sfreq = dataset.datasets[0].raw.info["sfreq"]
trial_start_offset_samples = int(-0.5 * sfreq)
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    preload=True,
)

splitted = windows_dataset.split("session")
train_set, valid_set = splitted["0train"], splitted["1test"]

######################################################################
# Training a small ShallowFBCSPNet
# --------------------------------
#
# We train just long enough to get a model that classifies above chance —
# attributions of a chance-level model are dominated by noise. For a
# real analysis you would want a fully trained checkpoint.

import torch
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from braindecode import EEGClassifier
from braindecode.datautil import infer_signal_properties
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds

cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
set_random_seeds(seed=20240205, cuda=cuda)

sig_props = infer_signal_properties(train_set, mode="classification")
model = ShallowFBCSPNet(
    n_chans=sig_props["n_chans"],
    n_outputs=sig_props["n_outputs"],
    n_times=sig_props["n_times"],
    final_conv_length="auto",
).to(device)

clf = EEGClassifier(
    model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),
    optimizer__lr=0.0625 * 0.01,
    optimizer__weight_decay=0,
    batch_size=64,
    callbacks=[
        "accuracy",
        ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=10)),
    ],
    device=device,
    classes=list(range(sig_props["n_outputs"])),
)
clf.fit(train_set, y=None, epochs=10)

######################################################################
# Picking samples for interpretation
# ----------------------------------
#
# Attribution is meaningful on samples the model gets right. We pull a
# stack of validation windows and filter to the correctly-classified
# subset using :func:`~braindecode.visualization.select_correctly_classified`.

import numpy as np

from braindecode.visualization import select_correctly_classified

X_val = np.stack([x for x, *_ in valid_set]).astype(np.float32)
y_val = np.array([y for _, y, *_ in valid_set])

X_correct, y_correct = select_correctly_classified(model, X_val, y_val, device=device)
print(f"Correctly classified: {X_correct.shape[0]} / {X_val.shape[0]}")

# Take a small batch for the rest of the tutorial.
n_show = min(8, X_correct.shape[0])
X_batch = X_correct[:n_show]
y_batch = y_correct[:n_show]

######################################################################
# Computing attribution maps
# --------------------------
#
# Each function takes ``(model, x, target)`` and returns a tensor with
# the same spatial shape as ``x``. ``target`` selects which class
# probability we're attributing — here we use the model's prediction.

from braindecode.visualization import (
    input_x_gradient,
    integrated_gradients,
    layer_grad_cam,
    saliency,
)

# `final_layer.conv_classifier` is the last conv before the average-pool /
# log-softmax in ShallowFBCSPNet — a natural choice for class-discriminative
# layer-wise attributions.
target_layer = model.final_layer.conv_classifier

attributions = {
    "Saliency": saliency(model, X_batch, y_batch),
    "Input × Gradient": input_x_gradient(model, X_batch, y_batch),
    "Integrated Gradients": integrated_gradients(model, X_batch, y_batch, steps=32),
    "LayerGradCam": layer_grad_cam(model, X_batch, y_batch, target_layer),
}

# All maps share the input shape (batch, n_chans, n_times).
for name, attr in attributions.items():
    print(f"{name:>22s} → {tuple(attr.shape)}")

######################################################################
# Heatmaps: channels × time
# -------------------------
#
# We average the absolute attribution over the batch and plot one
# (n_chans, n_times) heatmap per method. Brighter cells indicate samples
# the network leaned on more strongly.

import matplotlib.pyplot as plt

chs_info = valid_set.datasets[0].windows.info["chs"]
ch_labels = [ch["ch_name"] for ch in chs_info]
times = np.arange(sig_props["n_times"]) / sfreq

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
for ax, (name, attr) in zip(axes.ravel(), attributions.items()):
    heat = attr.detach().cpu().abs().mean(dim=0).numpy()
    im = ax.imshow(
        heat,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], -0.5, len(ch_labels) - 0.5],
        cmap="magma",
    )
    ax.set_title(name)
    ax.set_xlabel("Time (s)")
    ax.set_yticks(range(len(ch_labels)))
    ax.set_yticklabels(ch_labels, fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.04)

axes[0, 0].set_ylabel("Channel")
axes[1, 0].set_ylabel("Channel")
fig.suptitle("Mean absolute attribution per (channel, time) — averaged over correct trials")
fig.tight_layout()

######################################################################
# Topographic projection
# ----------------------
#
# Time-averaged attribution per channel can be projected onto a scalp
# topomap with :func:`~braindecode.visualization.project_to_topomap`,
# which is a thin wrapper around :func:`mne.viz.plot_topomap`. Lateralised
# motor activity over the central electrodes is the textbook signature
# we expect for left/right hand motor imagery.

from braindecode.visualization import project_to_topomap

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
for ax, (name, attr) in zip(axes, attributions.items()):
    per_channel = attr.detach().cpu().abs().mean(dim=(0, 2)).numpy()
    Z = project_to_topomap(per_channel, chs_info, res=64)
    im = ax.imshow(Z, cmap="viridis", origin="lower")
    ax.set_title(name)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.045)

fig.suptitle("Time-averaged attribution projected onto the scalp")
fig.tight_layout()

######################################################################
# Quantitative comparison vs. a randomized baseline
# -------------------------------------------------
#
# A common sanity check (Adebayo et al., 2018) is to verify that
# attributions from the trained model differ from those of a
# randomly-initialized model. If they don't, the explanation is
# essentially driven by the model's architecture, not by what it
# learned.
#
# :func:`~braindecode.visualization.compute_metrics` returns 12
# attribution-quality scores per sample (cosine, Pearson, relevance
# mass / rank accuracy, each in raw / normalized / top-percentile
# variants). Here we compare the trained Saliency map against the
# randomized model's Saliency.

import copy

from braindecode.visualization import METRIC_NAMES, compute_metrics


def _reset_parameters(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


random_model = copy.deepcopy(model).apply(_reset_parameters).eval().to(device)

trained_attr = saliency(model, X_batch, y_batch).detach().cpu().numpy()
random_attr = saliency(random_model, X_batch, y_batch).detach().cpu().numpy()

scores, n_skipped = compute_metrics(trained_attr, random_attr, abs_reference=True)
print(f"\nMetrics (trained vs randomized Saliency, n_skipped={n_skipped}):")
for name, col in zip(METRIC_NAMES, scores.T):
    print(f"  {name:30s} mean = {col.mean():+.3f}  std = {col.std():.3f}")

######################################################################
# Low cosine / Pearson values here mean the trained model's saliency is
# qualitatively different from random. High values would be a red flag —
# the explanation might be an artifact of the architecture.
#
# References
# ----------
#
# .. [1] Sundararajan, M., Taly, A., & Yan, Q. (2017). *Axiomatic
#        Attribution for Deep Networks.* ICML.
# .. [2] Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations
#        from Deep Networks via Gradient-based Localization.* ICCV.
# .. [3] Adebayo, J., et al. (2018). *Sanity Checks for Saliency Maps.*
#        NeurIPS.
