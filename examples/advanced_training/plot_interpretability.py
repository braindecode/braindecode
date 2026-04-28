""".. _interpretability-tutorial:

Interpretability of EEG Decoders
================================

This tutorial walks through the interpretability utilities in
:mod:`braindecode.visualization` applied to a motor-imagery decoder. We
train a small :class:`~braindecode.models.ShallowFBCSPNet` on the BCI
Competition IV 2a dataset and ask the same question from several angles:
**what is the network actually using?**

The tour combines two families of techniques:

*Frequency-domain and activation-space analyses* (adapted from the
CuttingGardens 2023 Braindecode tutorial by R. Schirrmeister):

- :func:`~braindecode.visualization.compute_amplitude_gradients` —
  gradient of class predictions w.r.t. spectral amplitudes
- a Haufe-style transform that turns those gradients ("filters") into
  the more interpretable "patterns"
- maximally activating inputs — the receptive-field windows that excite
  each filter the most

*Gradient-based attribution* (built on plain PyTorch autograd, no extra
dependencies):

- :func:`~braindecode.visualization.saliency` — ``|∂y[target] / ∂x|``
- :func:`~braindecode.visualization.integrated_gradients` — path
  integral from a baseline (Sundararajan et al., 2017)
- :func:`~braindecode.visualization.layer_grad_cam` — class-discriminative
  localization at a chosen layer
- :func:`~braindecode.visualization.project_to_topomap` — per-channel
  values to a 2-D scalp map via :func:`mne.viz.plot_topomap`
- :func:`~braindecode.visualization.compute_metrics` — quantitative
  comparison of attribution maps against a reference

.. contents:: This example covers:
   :local:
   :depth: 2
"""

# Authors: Robin T. Schirrmeister (CuttingGardens 2023 tutorial techniques)
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

######################################################################
# Loading and preparing the data
# ------------------------------
#
# We reuse the BCI Competition IV 2a setup: a single subject,
# bandpass-filtered to 4–38 Hz, with exponential moving standardization.

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

# Class label → index mapping (BCI IV 2a is left/right hand, feet, tongue).
class_mapping = train_set.datasets[0].windows.event_id

######################################################################
# Training a small ShallowFBCSPNet
# --------------------------------
#
# Ten epochs is enough to get usable accuracy for the examples below; for
# a real analysis you'd train longer.

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

# Channel info needed by the topomaps below.
chs_info = valid_set.datasets[0].windows.info["chs"]
raw_info = valid_set.datasets[0].windows.info

######################################################################
# Amplitude gradients
# -------------------
#
# How does changing the spectral amplitude at each (channel, frequency)
# move the network's class scores? :func:`compute_amplitude_gradients`
# answers exactly this — for every output unit it computes the gradient
# of the mean class score w.r.t. the input amplitudes (over all training
# trials), so we get an interpretable ``(n_classes, n_chans, n_freqs)``
# tensor after averaging over trials.

import numpy as np

from braindecode.visualization import compute_amplitude_gradients

amp_grads_per_filter = compute_amplitude_gradients(model, train_set, batch_size=64)
avg_amp_grads = amp_grads_per_filter.mean(axis=1)  # (n_classes, n_chans, n_freqs)

n_times = train_set[0][0].shape[1]
freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)

######################################################################
# We average the per-frequency gradients inside the canonical motor
# imagery bands (alpha, 7–13 Hz, and beta, 14–30 Hz) and plot a topomap
# per class. Lateralised activity over the central electrodes is the
# textbook signature for left/right hand motor imagery.

import matplotlib.pyplot as plt
import mne
from matplotlib import cm


def _band_topomap_grid(per_class_per_chan_per_freq, bands, info, mapping, title):
    """Plot one row of topomaps per frequency band, one column per class."""
    fig, axes = plt.subplots(
        len(bands), len(mapping), figsize=(3.5 * len(mapping), 3 * len(bands))
    )
    axes = np.atleast_2d(axes)
    for row, (lo, hi) in enumerate(bands):
        i_lo = np.searchsorted(freqs, lo)
        i_hi = np.searchsorted(freqs, hi) + 1
        avg_in_band = per_class_per_chan_per_freq[:, :, i_lo:i_hi].mean(axis=2)
        vmax = np.abs(avg_in_band).max()
        for class_name, i_class in mapping.items():
            ax = axes[row, i_class]
            mne.viz.plot_topomap(
                avg_in_band[i_class],
                info,
                vlim=(-vmax, vmax),
                contours=0,
                cmap=cm.coolwarm,
                show=False,
                axes=ax,
            )
            if row == 0:
                ax.set_title(class_name.replace("_", " ").title())
        axes[row, 0].set_ylabel(f"{lo}–{hi} Hz", rotation=0, labelpad=40, va="center")
    fig.suptitle(title)
    fig.tight_layout()


_band_topomap_grid(
    avg_amp_grads,
    bands=[(7, 13), (14, 30)],
    info=raw_info,
    mapping=class_mapping,
    title="Amplitude gradients (filters)",
)

######################################################################
# Filters → patterns (Haufe transform)
# ------------------------------------
#
# Gradients (the network's "filters") mix two effects: the genuine
# class-relevant signal *and* whatever noise correlations exist between
# channels and frequencies. Haufe et al. (2014) show that multiplying
# the filters by the input covariance recovers the underlying patterns,
# which are typically much easier to read.

import einops

train_X = np.stack([x for x, *_ in train_set])
amplitudes = np.abs(np.fft.rfft(train_X, axis=-1))

amp_cov = einops.rearrange(
    np.cov(einops.rearrange(amplitudes, "trial channel freq -> (channel freq) trial")),
    "(c1 f1) (c2 f2) -> c1 f1 c2 f2",
    c1=amplitudes.shape[1], c2=amplitudes.shape[1],
    f1=amplitudes.shape[2], f2=amplitudes.shape[2],
)

patterns = einops.einsum(
    avg_amp_grads, amp_cov,
    "classes chan freq, chan freq c2 f2 -> classes c2 f2",
)

_band_topomap_grid(
    patterns,
    bands=[(7, 13), (14, 30)],
    info=raw_info,
    mapping=class_mapping,
    title="Patterns (Haufe transform of the filters)",
)

######################################################################
# Maximally activating inputs
# ---------------------------
#
# For each filter in the spatial-temporal block, find the trials whose
# activation is in the top 5%, extract the receptive-field-sized window
# centred on the peak, and plot the per-channel mean of those windows.
# Filters with a coherent pattern across their top trials are learning
# something stable; filters with no consistent signature are dead weight.

activations = []


def _capture(_, __, output):
    activations.append(output.detach().cpu().numpy())


handle = model.bnorm.register_forward_hook(_capture)
loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=False)
batch_inputs = []
for X_batch, *_ in loader:
    with torch.no_grad():
        model(X_batch.to(device))
    batch_inputs.append(X_batch.numpy())
handle.remove()

# Activations: (trials, n_filters, time_out, 1) for ShallowFBCSPNet.
activations = np.concatenate(activations).squeeze(-1)
all_X = np.concatenate(batch_inputs)
n_filters = activations.shape[1]
n_receptive_field = all_X.shape[-1] - activations.shape[-1] + 1
max_act_per_trial = activations.max(axis=2)

######################################################################
# Plot the top-5% windows for every filter on a 5×8 grid (40 filters,
# the ShallowFBCSPNet default).

n_rows, n_cols = 5, 8
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 9), sharex=True, sharey=True)
for i_filter in range(n_filters):
    threshold = np.percentile(max_act_per_trial[:, i_filter], 95)
    top_trials = np.where(max_act_per_trial[:, i_filter] >= threshold)[0]
    peak_times = activations[top_trials, i_filter].argmax(axis=-1)

    windows = np.stack(
        [
            all_X[t, :, peak : peak + n_receptive_field]
            for t, peak in zip(top_trials, peak_times)
        ]
    )
    ax = axes[i_filter // n_cols, i_filter % n_cols]
    ax.plot(windows.mean(axis=(0, 1)), color="black", lw=1.2)
    ax.plot(windows.mean(axis=1).T, color="C0", alpha=0.15, lw=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"f{i_filter}", fontsize=8)
fig.suptitle("Maximally activating inputs (top-5% trials per filter, channel-mean)")
fig.tight_layout()

######################################################################
# Gradient-based attribution
# --------------------------
#
# Now we switch from frequency-domain analysis to per-trial attribution
# in the time domain. The four functions below take ``(model, x, target)``
# and return a tensor of the same spatial shape as ``x``.

from braindecode.visualization import (
    input_x_gradient,
    integrated_gradients,
    layer_grad_cam,
    saliency,
    select_correctly_classified,
)

X_val = np.stack([x for x, *_ in valid_set]).astype(np.float32)
y_val = np.array([y for _, y, *_ in valid_set])

X_correct, y_correct = select_correctly_classified(model, X_val, y_val, device=device)
print(f"Correctly classified: {X_correct.shape[0]} / {X_val.shape[0]}")

n_show = min(8, X_correct.shape[0])
X_batch = X_correct[:n_show]
y_batch = y_correct[:n_show]

# `final_layer.conv_classifier` is the last conv before the average-pool /
# log-softmax in ShallowFBCSPNet — a natural choice for the layer-CAM target.
target_layer = model.final_layer.conv_classifier

attributions = {
    "Saliency": saliency(model, X_batch, y_batch),
    "Input × Gradient": input_x_gradient(model, X_batch, y_batch),
    "Integrated Gradients": integrated_gradients(model, X_batch, y_batch, steps=32),
    "LayerGradCam": layer_grad_cam(model, X_batch, y_batch, target_layer),
}
for name, attr in attributions.items():
    print(f"{name:>22s} → {tuple(attr.shape)}")

######################################################################
# Heatmaps over channels and time
# -------------------------------
#
# Mean absolute attribution per ``(channel, time)`` cell, averaged over
# the eight correctly-classified trials. Brighter cells indicate samples
# the network leaned on more strongly when predicting the correct class.

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
# Time-averaged attribution per channel projected onto a scalp topomap
# via :func:`~braindecode.visualization.project_to_topomap` (a thin
# wrapper around :func:`mne.viz.plot_topomap`). Compare with the
# amplitude-gradient topomaps above — the techniques operate on
# different domains but should highlight roughly the same anatomy.

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
# Sanity check: trained vs randomized model
# -----------------------------------------
#
# Adebayo et al. (2018) showed that some saliency methods produce
# essentially the same map for a trained network and one with random
# weights — i.e. the explanation is an architectural artifact, not a
# learned signal. :func:`~braindecode.visualization.compute_metrics`
# turns this into a quantitative diagnostic. Low cosine / Pearson
# values mean the trained model's saliency is qualitatively different
# from random; high values are a red flag.

import copy

from braindecode.visualization import METRIC_NAMES, compute_metrics


def _reset(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


random_model = copy.deepcopy(model).apply(_reset).eval().to(device)

trained_attr = saliency(model, X_batch, y_batch).detach().cpu().numpy()
random_attr = saliency(random_model, X_batch, y_batch).detach().cpu().numpy()

scores, n_skipped = compute_metrics(trained_attr, random_attr, abs_reference=True)
print(f"\nMetrics (trained vs randomized Saliency, n_skipped={n_skipped}):")
for name, col in zip(METRIC_NAMES, scores.T):
    print(f"  {name:30s} mean = {col.mean():+.3f}  std = {col.std():.3f}")

######################################################################
# References
# ----------
#
# .. [1] Schirrmeister, R. T., et al. (2017). *Deep learning with
#        convolutional neural networks for EEG decoding and
#        visualization.* Human Brain Mapping, 38(11), 5391–5420.
# .. [2] Haufe, S., et al. (2014). *On the interpretation of weight
#        vectors of linear models in multivariate neuroimaging.*
#        NeuroImage, 87, 96–110.
# .. [3] Sundararajan, M., Taly, A., & Yan, Q. (2017). *Axiomatic
#        Attribution for Deep Networks.* ICML.
# .. [4] Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations
#        from Deep Networks via Gradient-based Localization.* ICCV.
# .. [5] Adebayo, J., et al. (2018). *Sanity Checks for Saliency Maps.*
#        NeurIPS.
