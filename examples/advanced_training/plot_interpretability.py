""".. _interpretability-tutorial:

Interpretability of EEG Decoders
================================

This tutorial trains a small :class:`~braindecode.models.ShallowFBCSPNet`
on BCI IV 2a motor imagery, then asks which parts of the input the
network actually uses, and which attribution methods we can trust to
tell us. The tour follows the EEG-XAI benchmark of Sujatha Ravindran &
Contreras-Vidal, Sci Rep 2023 (DOI 10.1038/s41598-023-43871-8): two
sanity checks (label and weight randomization) decide whether an
attribution map reflects what the model learned, or just its
architecture.

Outline:

1. Frequency-domain analysis —
   :func:`~braindecode.visualization.amplitude_gradients_per_trial`
   gives a per-class amplitude-gradient field; the Haufe transform turns
   those filters into class-distinctive patterns.
2. Per-trial attribution — :func:`~braindecode.visualization.saliency`,
   :func:`~braindecode.visualization.integrated_gradients`, and
   :func:`~braindecode.visualization.deep_lift`. All three are thin
   wrappers around captum (install with ``pip install braindecode[viz]``);
   the paper [5]_ identifies DeepLIFT as the most robust of the three on EEG.
3. Where on the scalp does the network look — single topomap with the
   landmark electrodes labelled.
4. **Are the explanations trustworthy?** Label randomization
   (:func:`~braindecode.visualization.random_target`) and cascading
   weight randomization (:func:`~braindecode.visualization.cascading_layer_reset`)
   answer this. Quantitative scoring uses
   :func:`~braindecode.visualization.compute_metrics` (12 metrics) and
   :func:`~braindecode.visualization.compute_ssim_metrics` (4 SSIM
   variants, pure-torch).

.. contents:: This example covers:
   :local:
   :depth: 2
"""

# Authors: Robin T. Schirrmeister (CuttingGardens 2023 tutorial techniques)
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

import copy

import einops
import matplotlib.pyplot as plt
import mne
import numpy as np
import torch
from matplotlib import cm
from numpy import multiply
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from braindecode import EEGClassifier
from braindecode.datasets import MOABBDataset
from braindecode.datautil import infer_signal_properties
from braindecode.models import ShallowFBCSPNet
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    exponential_moving_standardize,
    preprocess,
)
from braindecode.util import set_random_seeds
from braindecode.visualization import (
    METRIC_NAMES,
    amplitude_gradients_per_trial,
    cascading_layer_reset,
    compute_metrics,
    compute_ssim_metrics,
    deep_lift,
    integrated_gradients,
    random_target,
    saliency,
)

######################################################################
# Loading and preparing the data
# ------------------------------
#
# We reuse the BCI Competition IV 2a setup: a single subject,
# bandpass-filtered to 4–38 Hz, with exponential moving standardization.

subject_id = 1
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])

low_cut_hz, high_cut_hz = 4.0, 38.0
ems_factor_new, ems_init_block_size = 1e-3, 1000
volt_to_microvolt = 1e6

preprocess(
    dataset,
    [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        Preprocessor(lambda data: multiply(data, volt_to_microvolt)),
        Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),
        Preprocessor(
            exponential_moving_standardize,
            factor_new=ems_factor_new,
            init_block_size=ems_init_block_size,
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

split_by_session = windows_dataset.split("session")
train_set, valid_set = split_by_session["0train"], split_by_session["1test"]

# BCI IV 2a labels in MOABB's alphabetical order; the tuple position is
# the integer class id.
LABELS = ("feet", "left_hand", "right_hand", "tongue")

######################################################################
# Training a small ShallowFBCSPNet
# --------------------------------
#
# Thirty epochs is enough to push validation accuracy well above chance
# for this subject; the resulting attribution maps are correspondingly
# sharper than at 10 epochs.

n_epochs = 30
cuda_available = torch.cuda.is_available()
device = "cuda" if cuda_available else "cpu"
set_random_seeds(seed=20240205, cuda=cuda_available)

signal_properties = infer_signal_properties(train_set, mode="classification")
model = ShallowFBCSPNet(
    n_chans=signal_properties["n_chans"],
    n_outputs=signal_properties["n_outputs"],
    n_times=signal_properties["n_times"],
    final_conv_length="auto",
).to(device)

classifier = EEGClassifier(
    model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),
    optimizer__lr=0.0625 * 0.01,
    optimizer__weight_decay=0,
    batch_size=64,
    callbacks=[
        "accuracy",
        ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=n_epochs)),
    ],
    device=device,
    classes=list(range(signal_properties["n_outputs"])),
)
classifier.fit(train_set, y=None, epochs=n_epochs)

valid_accuracy = classifier.history[-1, "valid_accuracy"]
print(f"Final validation accuracy: {valid_accuracy:.2%}")

# Channel info needed by the topomaps below.
raw_info = valid_set.datasets[0].raw.info
channels_info = raw_info["chs"]

plt.rcParams.update({"font.size": 9, "figure.dpi": 110, "savefig.bbox": "tight"})

# Okabe-Ito colorblind-safe palette, one entry per label.
LABEL_COLORS = ("#0072B2", "#009E73", "#D55E00", "#CC79A7")

######################################################################
# Amplitude gradients
# -------------------
#
# How does changing the spectral amplitude at each (channel, frequency)
# move the network's class scores? :func:`amplitude_gradients_per_trial`
# answers that question. For every output unit it computes the gradient
# of the mean class score w.r.t. the input amplitudes over all training
# trials. After averaging over trials we get an
# ``(n_classes, n_chans, n_freqs)`` tensor.

per_trial_amplitude_gradients = amplitude_gradients_per_trial(
    model, train_set, batch_size=64
)
mean_amplitude_gradients = per_trial_amplitude_gradients.mean(
    axis=1
)  # (n_classes, n_chans, n_freqs)

n_input_samples = train_set[0][0].shape[1]
frequencies_hz = np.fft.rfftfreq(n_input_samples, d=1.0 / sfreq)

######################################################################
# We average the per-frequency gradients inside the canonical motor
# imagery bands (alpha, 7–13 Hz, and beta, 14–30 Hz) and plot a topomap
# per class. Lateralised activity over the central electrodes is the
# textbook signature for left/right hand motor imagery.


def _band_topomap_grid(
    values_per_class_chan_freq, frequency_bands, mne_info, labels, title
):
    """One row per frequency band, one column per class.

    Color scale is set per row from the 98th-percentile absolute value
    so that a single high-magnitude class can't wash out the rest of the
    band. The shared colorbar on the right uses scientific notation
    folded into its label, so no floating ``1e-6`` exponent appears.
    """
    n_bands, n_classes = len(frequency_bands), len(labels)
    fig, axes = plt.subplots(
        n_bands, n_classes, figsize=(2.6 * n_classes + 1.6, 2.4 * n_bands + 1.0)
    )
    axes = np.atleast_2d(axes)
    band_images, band_color_limits = [], []
    for row_idx, (band_low_hz, band_high_hz) in enumerate(frequency_bands):
        low_freq_idx = np.searchsorted(frequencies_hz, band_low_hz)
        high_freq_idx = np.searchsorted(frequencies_hz, band_high_hz) + 1
        mean_in_band = values_per_class_chan_freq[
            :, :, low_freq_idx:high_freq_idx
        ].mean(axis=2)
        band_color_limit = np.percentile(np.abs(mean_in_band), 98)
        band_color_limits.append(band_color_limit)
        for class_idx, label in enumerate(labels):
            ax = axes[row_idx, class_idx]
            topomap_image, _ = mne.viz.plot_topomap(
                mean_in_band[class_idx],
                mne_info,
                vlim=(-band_color_limit, band_color_limit),
                contours=0,
                cmap=cm.RdBu_r,
                sensors=False,
                show=False,
                axes=ax,
            )
            if row_idx == 0:
                ax.set_title(
                    label.replace("_", " ").title(),
                    color=LABEL_COLORS[class_idx],
                    fontweight="bold",
                    fontsize=10,
                )
        band_images.append(topomap_image)
        axes[row_idx, 0].text(
            -0.18,
            0.5,
            f"{band_low_hz}–{band_high_hz} Hz",
            transform=axes[row_idx, 0].transAxes,
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#444",
        )

    # Fold the colorbar's scientific exponent into the label so it
    # doesn't float as a "1e-6" annotation outside the axes.
    largest_limit = max(band_color_limits)
    exponent = int(np.floor(np.log10(largest_limit))) if largest_limit > 0 else 0
    scale = 10**exponent
    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.99)
    fig.subplots_adjust(right=0.88, top=0.90, wspace=0.05, hspace=0.15)
    colorbar_ax = fig.add_axes([0.91, 0.18, 0.015, 0.65])
    cbar = fig.colorbar(band_images[-1], cax=colorbar_ax)
    cbar.formatter.set_scientific(False)
    cbar.ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _pos: f"{v / scale:+.1f}")
    )
    cbar.set_label(
        f"Attribution × 10$^{{{exponent}}}$ (diverging)",
        fontsize=9,
    )


_band_topomap_grid(
    mean_amplitude_gradients,
    frequency_bands=[(7, 13), (14, 30)],
    mne_info=raw_info,
    labels=LABELS,
    title="Amplitude gradients carry distinct α/β signatures across classes",
)

######################################################################
# Filters → patterns (Haufe transform)
# ------------------------------------
#
# Gradients (the network's "filters") mix two effects: the
# class-relevant signal *and* the noise correlations between channels
# and frequencies. Haufe et al. (2014) show that multiplying the
# filters by the input covariance recovers the underlying patterns.
#
# In practice the input covariance for amplitude features is dominated
# by a near-rank-1 component (the average spectrum, ~1/f). Applied
# directly, that component leaves a class-uniform offset on every
# channel and the topographies look flat. Subtracting the across-class
# mean from the gradients before the transform removes that shared
# component and reveals what is *distinctive* about each class.

train_inputs = np.stack([x for x, *_ in train_set])
train_amplitudes = np.abs(np.fft.rfft(train_inputs, axis=-1))
n_channels = train_amplitudes.shape[1]
n_frequencies = train_amplitudes.shape[2]

amplitude_covariance = einops.rearrange(
    np.cov(
        einops.rearrange(train_amplitudes, "trial channel freq -> (channel freq) trial")
    ),
    "(chan_a freq_a) (chan_b freq_b) -> chan_a freq_a chan_b freq_b",
    chan_a=n_channels,
    chan_b=n_channels,
    freq_a=n_frequencies,
    freq_b=n_frequencies,
)

class_centered_gradients = mean_amplitude_gradients - mean_amplitude_gradients.mean(
    axis=0, keepdims=True
)

haufe_patterns = einops.einsum(
    class_centered_gradients,
    amplitude_covariance,
    "classes chan freq, chan freq chan_b freq_b -> classes chan_b freq_b",
)

_band_topomap_grid(
    haufe_patterns,
    frequency_bands=[(7, 13), (14, 30)],
    mne_info=raw_info,
    labels=LABELS,
    title="Haufe patterns isolate the class-distinctive part of each gradient",
)

######################################################################
# Per-trial attribution
# ---------------------
#
# We now switch from population averages to per-trial attribution in the
# time domain. Each method below takes ``(model, x, target)`` and
# returns an attribution tensor with the same spatial shape as ``x``.
# All three are captum-backed; the paper [5]_ identifies DeepLIFT as the
# most robust of the three on EEG.

valid_inputs = np.stack([x for x, *_ in valid_set]).astype(np.float32)
valid_labels = np.array([y for _, y, *_ in valid_set])

# Keep only trials the model classifies correctly: attribution maps for
# misclassified trials would explain the *wrong* class, which muddies the
# sanity-check signal below.
valid_inputs_t = torch.as_tensor(valid_inputs, dtype=torch.float32, device=device)
valid_labels_t = torch.as_tensor(valid_labels, dtype=torch.long, device=device)
with torch.no_grad():
    correct_mask = model(valid_inputs_t).argmax(dim=1) == valid_labels_t
correctly_classified_inputs = valid_inputs_t[correct_mask]
correctly_classified_labels = valid_labels_t[correct_mask]
print(
    f"Correctly classified: {correctly_classified_inputs.shape[0]} "
    f"/ {valid_inputs.shape[0]}"
)

n_examples_to_show = min(8, correctly_classified_inputs.shape[0])
example_inputs = correctly_classified_inputs[:n_examples_to_show]
example_labels = correctly_classified_labels[:n_examples_to_show]


methods = {
    "Saliency": lambda m, x, y: saliency(m, x, y).cpu().numpy(),
    "Integrated Gradients": lambda m, x, y: integrated_gradients(m, x, y, steps=32)
    .cpu()
    .numpy(),
    "DeepLIFT": lambda m, x, y: deep_lift(m, x, y).cpu().numpy(),
}

trained_attributions = {
    name: fn(model, example_inputs, example_labels) for name, fn in methods.items()
}
for method_name, attribution in trained_attributions.items():
    print(f"  {method_name:>22s} → {attribution.shape}")

######################################################################
# Where on the scalp does the network look?
# ------------------------------------------
#
# We collapse each attribution to one value per electrode — mean absolute
# value across trials and time — and project onto the scalp. The methods
# agree closely on this dataset (see the printed pairwise correlations),
# so we plot a single panel showing their average and label C3 / Cz / C4
# for orientation.

channel_names = [ch["ch_name"] for ch in channels_info]

per_channel_per_method = {}
for method_name, attribution in trained_attributions.items():
    per_channel = np.abs(attribution).mean(axis=(0, 2))
    per_channel_per_method[method_name] = per_channel / max(per_channel.max(), 1e-12)

method_names = list(per_channel_per_method)
print("\nPer-channel attribution agreement (Pearson r):")
for i, name_a in enumerate(method_names):
    for name_b in method_names[i + 1 :]:
        r = float(
            np.corrcoef(per_channel_per_method[name_a], per_channel_per_method[name_b])[
                0, 1
            ]
        )
        print(f"  {name_a:>22s} vs {name_b:<22s} r = {r:+.3f}")

mean_per_channel = np.mean([per_channel_per_method[n] for n in method_names], axis=0)

landmark_channels = {"C3", "Cz", "C4"}
landmark_labels = [name if name in landmark_channels else "" for name in channel_names]

fig, ax = plt.subplots(figsize=(5.2, 4.4))
topomap_image, _ = mne.viz.plot_topomap(
    mean_per_channel,
    raw_info,
    vlim=(0, 1),
    contours=0,
    cmap="magma",
    sensors=True,
    names=landmark_labels,
    show=False,
    axes=ax,
)
ax.set_title(
    f"Mean attribution across {len(method_names)} methods, projected on the scalp",
    fontsize=10,
    fontweight="bold",
)
fig.colorbar(
    topomap_image,
    ax=ax,
    fraction=0.04,
    pad=0.04,
    label="Per-method normalized (0–1)",
)

######################################################################
# Are the explanations trustworthy?
# ---------------------------------
#
# A good attribution map should depend on **what** the model learned and
# on **which class** we ask about. Adebayo et al. [4]_ formalised two
# sanity checks; Sujatha Ravindran & Contreras-Vidal [5]_ applied them
# to twelve attribution methods on simulated EEG. Their headline
# finding: vanilla Saliency, the most common method in EEG, is *not*
# class- or model-specific, while DeepLIFT remains both accurate and
# robust across the temporal, spatial, and spectral regimes they tested.
#
# We can replicate the qualitative result on this small motor-imagery
# example by computing two cosine similarities for each method:
#
# - **Label randomization**: re-attribute on the trained model with a
#   wrong-class target (via :func:`~braindecode.visualization.random_target`).
#   A method that depends on the requested class should produce a
#   different map → low cosine to the trained-target attribution.
# - **Weight randomization, cascading**: walk the model's modules from
#   output to input, resetting parameters one at a time
#   (:func:`~braindecode.visualization.cascading_layer_reset`), and
#   measure how fast the attribution drifts away from the trained one.
#   A method that depends on the learned weights should drift quickly.
#
# In both checks **lower cosine = better**: the attribution is sensitive
# to the manipulation, so it is doing its job.

set_random_seeds(seed=20240205, cuda=cuda_available)
n_classes = signal_properties["n_outputs"]


def _flat_cosine(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# --- Label randomization: one cosine per method ---
random_labels = random_target(example_labels, n_classes=n_classes)
label_random_cosine = {
    name: _flat_cosine(
        trained_attributions[name], fn(model, example_inputs, random_labels)
    )
    for name, fn in methods.items()
}

# --- Cascading weight randomization: one curve per method ---
cascade_levels = [0]
cascade_cosines = {name: [1.0] for name in methods}
for level, (_layer_name, randomized_model) in enumerate(
    cascading_layer_reset(model), start=1
):
    cascade_levels.append(level)
    for name, fn in methods.items():
        rand_attr = fn(randomized_model, example_inputs, example_labels)
        cascade_cosines[name].append(
            _flat_cosine(trained_attributions[name], rand_attr)
        )

n_levels = len(cascade_levels) - 1
fully_random_cosine = {name: cascade_cosines[name][-1] for name in methods}

######################################################################
# The figure puts the two checks side by side: each method gets a bar
# in the left panel (label vs full-weight randomization) and a curve in
# the right panel (cosine versus number of layers reset, output → input).

method_palette = {
    "Saliency": "#D55E00",
    "Integrated Gradients": "#0072B2",
    "DeepLIFT": "#009E73",
}
fig, (ax_bar, ax_cascade) = plt.subplots(1, 2, figsize=(11, 4.2))

bar_x = np.arange(len(methods))
bar_width = 0.36
ax_bar.bar(
    bar_x - bar_width / 2,
    [label_random_cosine[n] for n in methods],
    bar_width,
    color=[method_palette.get(n, "#888") for n in methods],
    label="Label randomization",
    edgecolor="white",
)
ax_bar.bar(
    bar_x + bar_width / 2,
    [fully_random_cosine[n] for n in methods],
    bar_width,
    color=[method_palette.get(n, "#888") for n in methods],
    alpha=0.45,
    label="Full weight randomization",
    edgecolor="white",
    hatch="//",
)
ax_bar.set_xticks(bar_x)
ax_bar.set_xticklabels(list(methods), fontsize=9)
ax_bar.set_ylim(0, max(1.05, max(label_random_cosine.values()) + 0.05))
ax_bar.axhline(1.0, color="#a30000", lw=0.8, ls=":", alpha=0.7)
ax_bar.set_ylabel("Cosine to trained-model attribution\n(lower is better)")
ax_bar.set_title("Sanity-check scores", fontsize=10, fontweight="bold")
ax_bar.legend(loc="upper right", frameon=False, fontsize=8)
for spine in ("top", "right"):
    ax_bar.spines[spine].set_visible(False)

for name in methods:
    ax_cascade.plot(
        cascade_levels,
        cascade_cosines[name],
        marker="o",
        color=method_palette.get(name, "#888"),
        label=name,
        lw=1.6,
        markersize=4,
    )
ax_cascade.set_xlabel(f"# layers reset (output → input, total {n_levels})")
ax_cascade.set_ylabel("Cosine to trained-model attribution")
ax_cascade.set_title("Cascading weight randomization", fontsize=10, fontweight="bold")
ax_cascade.set_ylim(-0.1, 1.05)
ax_cascade.axhline(0.0, color="#888", lw=0.6, ls=":")
ax_cascade.legend(loc="best", frameon=False, fontsize=8)
for spine in ("top", "right"):
    ax_cascade.spines[spine].set_visible(False)

fig.suptitle(
    "Methods that survive both randomizations are reflecting architecture, not learning",
    fontsize=11,
    fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.94])

######################################################################
# Quantitative scoring with `compute_metrics` and `compute_ssim_metrics`
# ----------------------------------------------------------------------
#
# `compute_metrics` returns 12 cosine / Pearson / mass-accuracy variants
# per sample; `compute_ssim_metrics` adds four SSIM-based scores using a
# pure-torch implementation. Below we report a four-metric summary for
# each method against its weight-randomized counterpart. The trained
# model's attribution should score **low** on all four.


def _reset_all(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


random_weight_model = copy.deepcopy(model).apply(_reset_all).eval().to(device)

SUMMARY_METRICS = ("Cosine_absnorm", "Pearson_absnorm_vs_refnorm")
print("\nMethod scores vs fully-randomized weights (lower = more sensitive = better):")
header = (
    f"  {'Method':<22s}"
    + "".join(f"{m:>22s}" for m in SUMMARY_METRICS)
    + f"{'SSIM_absnorm':>16s}"
)
print(header)
for name, fn in methods.items():
    rand_attr = fn(random_weight_model, example_inputs, example_labels)
    cos_pearson, _ = compute_metrics(
        trained_attributions[name], rand_attr, abs_reference=True
    )
    by_name = dict(zip(METRIC_NAMES, cos_pearson.T))
    ssim_scores, _ = compute_ssim_metrics(
        trained_attributions[name], rand_attr, abs_reference=True
    )
    ssim_absnorm = float(ssim_scores[:, 1].mean())
    line = f"  {name:<22s}"
    for m in SUMMARY_METRICS:
        line += f"{by_name[m].mean():>+22.3f}"
    line += f"{ssim_absnorm:>+16.3f}"
    print(line)

######################################################################
# Take-aways
# ----------
#
# - Looking at one attribution map is not enough on EEG. Saliency, the
#   most popular technique in the literature, is often invariant to both
#   class and model — see [5]_ for the simulation evidence.
# - When in doubt, run the two checks above. The cascading curve in
#   particular distinguishes methods that genuinely depend on weights
#   (curve drops fast) from those that mostly reflect architecture
#   (curve stays near 1).
# - DeepLIFT is the consistent winner in [5]_; this tutorial includes it
#   automatically when Captum is installed (``pip install braindecode[viz]``).
#
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
# .. [4] Adebayo, J., et al. (2018). *Sanity Checks for Saliency Maps.*
#        NeurIPS.
# .. [5] Sujatha Ravindran, A., & Contreras-Vidal, J. (2023). *An
#        empirical comparison of deep learning explainability approaches
#        for EEG using simulated ground truth.* Scientific Reports, 13.
#        DOI: 10.1038/s41598-023-43871-8
