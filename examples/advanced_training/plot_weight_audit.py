""".. _weight-audit-tutorial:

Auditing what a decoder relies on from its weights
==================================================

A decoder can score well by reading a signal other than the one it claims
to decode. On the PhysioNet motor imagery dataset the cue is a target that
appears on the left or on the right of the screen. A cue-directed eye
movement can produce a slow potential at scalp electrodes, giving a
left-versus-right decoder another signal alongside the sensorimotor rhythm. Gao
(2026) [1]_ ran a preregistered audit of this shortcut on PhysioNet. In an
exploratory analysis of that audit, the decoders trained there beat
controls with the lateral eye signal regressed out, and the gap grew with
the size of the cue-directed deflection (panels a and b of
``figures/fig_eye_posthoc.png`` in the Zenodo record).

Every audit used so far (probing, erasure, EOG regression, attribution
maps) needs the data and forward passes. Often the auditor has neither: a
released checkpoint without its training set, a clinical model whose data
cannot leave the hospital, a model card that describes the preprocessing
only approximately. This example audits a released EEG decoder from its
weights with the simplest possible reader: a path-weighted statistic of
the weights, then an occlusion test on data to check the reading.

.. topic:: What the audit looks for

    EEGNet learns a filter bank (temporal kernels), one set of spatial
    filters per kernel, and a classifier over the pooled features. Each
    tensor is readable on its own: the power spectrum of a temporal kernel
    says which band it passes, the spatial weights say which electrodes it
    mixes, and the classifier says which pooled time bins separate the
    classes. Pooling these through the network's own path weights gives a
    ranking of bands, time windows and channels. The PhysioNet comparison
    tests two hypotheses: early activity below 4 Hz could carry the cue,
    while later activity at 8-30 Hz could carry sensorimotor rhythms.

.. contents:: This example covers:
   :local:
   :depth: 2

"""

# Authors: Yiheng Li <yihengl4@asu.edu>
# License: BSD (3-clause)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy import multiply
from scipy.signal import butter, freqz, sosfiltfilt
from sklearn.metrics import accuracy_score

from braindecode.datasets import BaseConcatDataset, MOABBDataset
from braindecode.models import EEGNet
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)

######################################################################
# The decoder under audit
# -----------------------
#
# Guetschel and Tangermann [2]_ trained :class:`~braindecode.models.EEGNet`
# [3]_ on twelve MOABB motor imagery datasets and released the weights on
# the Hugging Face Hub. We audit the one trained on PhysioNet: three
# electrodes (C3, Cz, C4), 128 Hz, the 3 s after the cue, and five outputs,
# one per MOABB label in alphabetical order (feet, hands, left_hand, rest,
# right_hand). Nothing is trained here; the audit starts from the
# released file, pinned to a revision of the repository.

CHECKPOINT_URL = (
    "https://huggingface.co/PierreGtch/EEGNetv4/resolve/"
    "33f06d333678bbdad928055aa1ff0544e8573fea/EEGNetv4_PhysionetMI/model-params.pkl"
)
CHANNELS = ["C3", "Cz", "C4"]
SFREQ = 128.0
L_FREQ, H_FREQ = 0.5, 40.0  # band-pass of the checkpoint's input
LEFT, RIGHT = 2, 4  # output indices of left_hand and right_hand

state = torch.hub.load_state_dict_from_url(
    CHECKPOINT_URL,
    map_location="cpu",
    weights_only=True,
    file_name="EEGNetv4_PhysionetMI.pkl",
)
# The checkpoint was saved by an older braindecode; load_state_dict maps its
# parameter names to the current ones.
model = EEGNet(n_chans=3, n_outputs=5, n_times=385, sfreq=SFREQ, drop_prob=0.25)
model.load_state_dict(state)
model.eval()

######################################################################
# Reading the weights
# -------------------
#
# The reader follows the EEGNet forward path backwards, for the contrast
# between the left- and right-hand outputs:
#
# * **Which features matter?** Subtract the classifier weights of the two
#   hand outputs. Scale each feature's contrast by the gain of its
#   batch-normalization layer in evaluation mode, then take its norm over time.
# * **Which channels?** Feature importance is pushed back through the
#   pointwise weights and the gain of the batch normalization after the
#   spatial filters to each spatial filter, then through the gain of the
#   temporal batch normalization and the absolute spatial weights to each
#   channel.
# * **Which band?** Each temporal kernel is an FIR filter; the kernels'
#   power spectra, weighted by the importance of the spatial filters they
#   feed, are averaged over each band inside the input pass-band
#   (0.5-40 Hz). A mean lets us compare bands of different widths.
# * **Which time window?** The classifier contrast energy per pooled time
#   bin, averaged in one-second windows.
#
# None of this touches data: it is what anyone can compute from a released
# checkpoint in a second. One limit to keep in mind: a spatial weight says
# how much a filter *mixes* an electrode, not how much signal arrives
# there (Haufe et al. [4]_ make this point for linear models), so the
# channel ranking is the weakest part of the reading. The band and the
# window come from the first temporal kernels and the classifier. This
# heuristic omits the second depthwise temporal convolution, pooling,
# nonlinearities and the input spectrum. Occlusion can disagree with it.
#
# First, the weights of each layer as arrays. ``numpy(force=True)`` detaches
# a parameter and moves it to the CPU; ``conv_spatial.weight`` is the
# spatial filter after its max-norm.


def batch_norm_gain(layer):
    """Return the absolute slope of a fitted batch-normalization layer."""
    return np.abs(layer.weight.numpy(force=True)) / np.sqrt(
        layer.running_var.numpy(force=True) + layer.eps
    )


w_temporal = model.conv_temporal.weight.numpy(force=True)[:, 0, 0, :]  # (F1, n_taps)
w_temporal = batch_norm_gain(model.bnorm_temporal)[:, None] * w_temporal
w_spatial = model.conv_spatial.weight.numpy(force=True)[:, 0, :, 0]  # (F1*D, n_chans)
w_point = model.conv_separable_point.weight.numpy(force=True)[:, :, 0, 0]  # (F2, F1*D)
w_head = model.final_layer.conv_classifier.weight.numpy(force=True)
w_head = w_head.reshape(model.n_outputs, model.F2, -1)

######################################################################
# Feature, then spatial filter, then channel.

contrast = batch_norm_gain(model.bnorm_2)[:, None] * (w_head[LEFT] - w_head[RIGHT])
feature_importance = np.linalg.norm(contrast, axis=1)
filter_importance = batch_norm_gain(model.bnorm_1) * (
    feature_importance @ np.abs(w_point)
)
# Spatial filter ``j`` reads temporal kernel ``j // D``, scaled by its gain.
gain_temporal = np.repeat(batch_norm_gain(model.bnorm_temporal), model.D)
channel_share = pd.Series(
    (gain_temporal * filter_importance) @ np.abs(w_spatial), CHANNELS, dtype=float
)
channel_share /= channel_share.sum()

######################################################################
# The band. Temporal kernel ``k`` feeds spatial filters ``k*D`` to
# ``k*D + D - 1``, so a kernel's importance is the sum over its ``D``
# filters. :func:`scipy.signal.freqz` gives the frequency response of each
# kernel every 0.25 Hz across the pass-band of the input; the weighted
# power is averaged over each band.

kernel_importance = filter_importance.reshape(model.F1, model.D).sum(axis=1)
# freqz reads filter taps along the first axis: (n_taps, F1, 1) gives (F1, n_freqs).
freqs, response = freqz(
    w_temporal.T[:, :, None], worN=np.arange(0.0, H_FREQ + 0.25, 0.25), fs=SFREQ
)
spectrum = kernel_importance @ np.abs(response) ** 2
passed = freqs >= L_FREQ
band = pd.cut(
    freqs[passed],
    bins=[0.0, 4.0, 8.0, 13.0, 30.0, np.inf],
    labels=[
        "slow (<4 Hz)",
        "theta (4-8 Hz)",
        "mu/alpha (8-13 Hz)",
        "beta (13-30 Hz)",
        "gamma (>30 Hz)",
    ],
    right=False,
)
band_power = pd.Series(spectrum[passed]).groupby(band, observed=False).mean()
band_share = band_power / band_power.sum()

######################################################################
# Approximate the time of each pooled classifier bin from its centre within
# the trial. Convolution and pooling give neighbouring bins overlapping
# receptive fields. Average the contrast energy within each one-second window.

energy = (contrast**2).sum(axis=0)
bin_centre_s = (np.arange(energy.size) + 0.5) * model.n_times / energy.size / SFREQ
window_energy = pd.Series(energy).groupby(np.floor(bin_centre_s).astype(int)).mean()
best = window_energy.idxmax()
window = f"{best:+d}..{best + 1:+d} s"

print(f"Band with the highest mean kernel power: {band_share.idxmax()}")
print(band_share.round(3).to_string())
print(f"Window with most classifier contrast: {window}")
print("Channel shares:", channel_share.round(3).to_dict())

######################################################################
# Compare the printed band and window with the two hypotheses above. The
# five values are normalized band means. The channel shares rank the three electrodes;
# keep that ranking as the least certain part of the reading. The next
# step checks the reading on data.
#
# Check the reading on recorded trials
# ------------------------------------
#
# Loading PhysioNet as the checkpoint saw it
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We load three subjects of the PhysioNet motor imagery dataset [5]_ through
# MOABB and rebuild the input of the checkpoint from its paper: C3, Cz and
# C4 in microvolts, 0.5-40 Hz, resampled to 128 Hz, 385 samples from the
# cue. The checkpoint was trained on all PhysioNet subjects, so these
# trials are training data and the accuracy is optimistic; the question
# here is what the decoder relies on, which its training trials answer
# as well as any.

subjects = [1, 2, 3]
dataset = MOABBDataset(dataset_name="PhysionetMI", subject_ids=subjects)
preprocess(
    dataset,
    [
        Preprocessor("pick", picks=CHANNELS),
        Preprocessor(lambda data: multiply(data, 1e6)),  # V to uV
        Preprocessor("filter", l_freq=L_FREQ, h_freq=H_FREQ),
        Preprocessor("resample", sfreq=SFREQ),
    ],
    n_jobs=1,
)

######################################################################
# MOABB also loads the hands-versus-feet runs; we keep the runs with left-
# and right-hand trials. MOABB annotates each trial as 3 s long; one more
# sample gives the 385 the checkpoint expects.

hand_runs = BaseConcatDataset(
    [ds for ds in dataset.datasets if "left_hand" in ds.raw.annotations.description]
)
windows = create_windows_from_events(
    hand_runs,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=1,
    mapping={"left_hand": 0, "right_hand": 1},
    preload=True,
)
assert windows.datasets[0].raw.ch_names == CHANNELS
X = np.stack([np.asarray(x) for x, _, _ in windows]).astype(np.float32)
y = np.array([int(label) for _, label, _ in windows])
print(f"{len(X)} trials, {X.shape[1]} channels x {X.shape[2]} samples")

######################################################################
# Removing what the weights point to
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Remove a band, a time window or a channel, and measure the change in
# left-versus-right accuracy. High-pass and low-pass perturbations check
# the low and high ends of the input spectrum. A third perturbation keeps
# the 8-30 Hz band containing the sensorimotor rhythms.
#
# Each test is a modified copy of the trials: filtered with a 4th-order
# zero-phase Butterworth filter, or multiplied by a mask that zeroes some
# samples or one channel.

highpass_4hz = butter(4, 4.0, "highpass", fs=SFREQ, output="sos")
lowpass_30hz = butter(4, 30.0, "lowpass", fs=SFREQ, output="sos")
bandpass_8_30hz = butter(4, [8.0, 30.0], "bandpass", fs=SFREQ, output="sos")
first_second = np.arange(X.shape[-1]) < int(SFREQ)
X_tests = {
    "remove\n0.5-4 Hz": sosfiltfilt(highpass_4hz, X).astype(np.float32),
    "remove\n30-40 Hz": sosfiltfilt(lowpass_30hz, X).astype(np.float32),
    "keep\n8-30 Hz only": sosfiltfilt(bandpass_8_30hz, X).astype(np.float32),
    "zero\n0-1 s": X * ~first_second,
    "zero\n1-3 s": X * first_second,
}
for i, ch in enumerate(CHANNELS):
    X_tests[f"zero\n{ch}"] = X * (np.arange(len(CHANNELS)) != i)[:, None]

######################################################################
# The decoder predicts the hand with the larger of its two hand outputs.
# The drop of a test is the accuracy on the unchanged trials minus the
# accuracy on the modified ones.

accuracy = {}
with torch.no_grad():
    for name, X_test in {"unchanged": X, **X_tests}.items():
        logits = model(torch.as_tensor(X_test))
        y_pred = logits[:, [LEFT, RIGHT]].argmax(dim=1).numpy()
        accuracy[name] = accuracy_score(y, y_pred)
accuracy = pd.Series(accuracy)
baseline = accuracy.pop("unchanged")
drops = baseline - accuracy

print(f"Left vs right accuracy: {baseline:.1%}")
for name, value in drops.items():
    print(f"  {name.replace(chr(10), ' '):>18s}: drop {value:+.1%}")
top_weights = channel_share.idxmax()
top_data = drops[[f"zero\n{ch}" for ch in CHANNELS]].idxmax().removeprefix("zero\n")
print(f"Channel ranked first by the weights: {top_weights}")
print(f"Channel whose removal costs the most: {top_data}")

######################################################################
# The chart groups frequency, time and channel perturbations by colour.
# Compare the two frequency removals with the band ranked first by the
# kernels, and the two time masks with the highest-scoring classifier window.
# The last two printed lines compare the channel rankings.

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.bar(
    drops.index,
    100 * drops.to_numpy(),
    color=["#D55E00"] * 3 + ["#0072B2"] * 2 + ["#999999"] * 3,
)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Accuracy drop (points)")
ax.set_title(
    f"PhysioNet EEGNet checkpoint, left vs right {baseline:.0%}; "
    f"weights say {band_share.idxmax()}, {window}\n"
    f"top channel: {top_weights} by the weights, {top_data} by occlusion",
    fontsize=9,
)
fig.tight_layout()

######################################################################
# What to do with the result
# --------------------------
#
# Record it with the model, in its model card on the Hugging Face Hub. A
# model loaded with
# :meth:`~braindecode.models.EEGModuleMixin.from_pretrained` carries the
# metadata of that card as ``model.card_data`` (training datasets,
# licence); see :ref:`pretraining-overlap`. The same card can state the
# band, window and channels the audit found, the method (weight statistic,
# occlusion on which subjects), the date, and whether the data check
# agreed with the weight reading. A decoder whose card says "below 4 Hz,
# 0-1 s after the cue" is a different product from one that says
# "8-30 Hz, 1-3 s", even at the same accuracy. Two more lines make the
# next audit easier: the exact preprocessing of the input, and the
# initialisation seed.
#
# Filtering and masking change the inputs. A loss of accuracy measures
# sensitivity to that change on these training subjects. Slow cortical
# activity and ocular activity can both contribute. To investigate the
# source, compare predictions with independently recorded eye movements.
# To investigate transfer, test held-out data acquired with a different
# cue arrangement.
#
# References
# ----------
#
# .. [1] Gao, Y. (2026). Do shortcut audits measure reliance in EEG
#        decoders? A preregistered semi-synthetic ground-truth test and a
#        real-data check of eye-movement use. Manuscript. Preregistration:
#        https://doi.org/10.17605/OSF.IO/8Q7N6. Code:
#        https://doi.org/10.5281/zenodo.24142936
# .. [2] Guetschel, P., & Tangermann, M. (2023). Transfer learning
#        between motor imagery datasets using deep learning: validation
#        of framework and comparison of datasets. arXiv:2311.16109.
#        Weights: https://huggingface.co/PierreGtch/EEGNetv4
# .. [3] Lawhern, V. J., et al. (2018). EEGNet: a compact convolutional
#        neural network for EEG-based brain-computer interfaces. Journal
#        of Neural Engineering, 15(5), 056013.
#        DOI: 10.1088/1741-2552/aace8c
# .. [4] Haufe, S., et al. (2014). On the interpretation of weight
#        vectors of linear models in multivariate neuroimaging.
#        NeuroImage, 87, 96-110. DOI: 10.1016/j.neuroimage.2013.10.067
# .. [5] Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N., &
#        Wolpaw, J. R. (2004). BCI2000: A general-purpose brain-computer
#        interface (BCI) system. IEEE Transactions on Biomedical
#        Engineering, 51(6), 1034-1043. DOI: 10.1109/TBME.2004.827072
