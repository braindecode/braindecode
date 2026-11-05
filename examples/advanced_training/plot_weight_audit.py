""".. _weight-audit-tutorial:

Auditing what a decoder relies on from its weights
==================================================

A decoder can score well by reading a signal other than the one it claims
to decode. On the PhysioNet motor imagery dataset the cue is a target that
appears on the left or on the right of the screen, the eyes move toward
it, and the slow potential of that eye movement reaches every frontal and
temporal electrode. A left-versus-right decoder trained on accuracy alone
takes that signal when it is easier than the sensorimotor rhythm. Gao
(2026) [1]_ ran a preregistered audit of this shortcut on PhysioNet; the
decoders they trained beat controls with the lateral eye signal regressed
out, and the gap grew with the size of the cue-directed deflection.

Every audit used so far (probing, erasure, EOG regression, attribution
maps) needs the data and forward passes. Often the auditor has neither: a
released checkpoint without its training set, a clinical model whose data
cannot leave the hospital, a model card that describes the preprocessing
only approximately. Takehi et al. (2026) [2]_ showed that language models can
read raw weights well enough to say what a small network was trained to
do, and listed "auditing without data" as an open direction. This example
takes that route on a released EEG decoder with the simplest possible
reader: a path-weighted statistic of the weights, then one occlusion test
on data to confirm it.

.. topic:: What the audit looks for

    EEGNet learns a filter bank (temporal kernels), one set of spatial
    filters per kernel, and a classifier over the pooled features. Each
    tensor is readable on its own: the spectral peak of a temporal kernel
    says which band it passes, the spatial weights say which electrodes it
    mixes, and the classifier says which pooled time bins separate the
    classes. Pooling these through the network's own path weights gives a
    data-free estimate of the band and the time window the decoder relies
    on. On PhysioNet the shortcut signature is a band below 4 Hz and a
    window in the first second after the cue; the sensorimotor signature
    is 8-30 Hz and the seconds after that.

.. contents:: This example covers:
   :local:
   :depth: 2

"""

# Authors: Yiheng Li <yihengl4@asu.edu>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import multiply
from scipy.signal import butter, sosfiltfilt

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
# Guetschel and Tangermann [3]_ trained :class:`~braindecode.models.EEGNet`
# [4]_ on twelve MOABB motor imagery datasets and released the weights on
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
LEFT, RIGHT = 2, 4  # output indices of left_hand and right_hand

state = torch.hub.load_state_dict_from_url(
    CHECKPOINT_URL,
    map_location="cpu",
    weights_only=True,
    file_name="EEGNetv4_PhysionetMI.pkl",
)
# The checkpoint predates the max-norm parametrisation of the spatial filters.
state["conv_spatial.parametrizations.weight.original"] = state.pop(
    "conv_spatial.weight"
)
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
# * **Which features matter?** The classifier weights, reshaped to
#   ``(n_outputs, F2, n_time_bins)``, are centred across the two outputs;
#   a feature's importance is the norm of that contrast times its
#   batch-norm gain.
# * **Which channels?** Feature importance is pushed back through the
#   pointwise weights and the first batch-norm gain to each spatial
#   filter, then through the absolute spatial weights to each channel.
# * **Which band?** Each temporal kernel is an FIR filter; the peak of
#   the kernels' power spectra, weighted by the importance of the spatial
#   filters they feed, gives the band.
# * **Which time window?** The classifier contrast energy per pooled time
#   bin, averaged in one-second windows.
#
# None of this touches data: it is what anyone can compute from a released
# checkpoint in a second. One limit to keep in mind: a spatial weight says
# how much a filter *mixes* an electrode, not how much signal arrives
# there (Haufe et al. [5]_ make this point for linear models), so the
# channel ranking is the weakest part of the reading. The band and the
# window come from the temporal kernels and the classifier.

BANDS = [
    (0.0, 4.0, "slow (<4 Hz)"),
    (4.0, 8.0, "theta (4-8 Hz)"),
    (8.0, 13.0, "mu/alpha (8-13 Hz)"),
    (13.0, 30.0, "beta (13-30 Hz)"),
    (30.0, np.inf, "gamma (>30 Hz)"),
]


def read_weights(model, ch_names, sfreq, window_start_s, outputs):
    """Estimate the channels, band and time window an EEGNet relies on."""

    def as_np(t):
        return t.detach().cpu().numpy()

    w_spatial = as_np(model.conv_spatial.weight)[:, 0, :, 0]  # (F1*D, n_chans)
    w_temporal = as_np(model.conv_temporal.weight)[:, 0, 0, :]  # (F1, n_taps)
    w_point = as_np(model.conv_separable_point.weight)[:, :, 0, 0]  # (F2, F1*D)
    head = next(
        m
        for m in model.final_layer.modules()
        if torch.is_tensor(getattr(m, "weight", None))
    )
    w_head = as_np(head.weight).reshape(model.n_outputs, model.F2, -1)[outputs]

    # Feature -> spatial filter -> channel.
    contrast = w_head - w_head.mean(axis=0, keepdims=True)
    feature_importance = np.abs(as_np(model.bnorm_2.weight)) * np.sqrt(
        (contrast**2).sum(axis=(0, 2))
    )
    filter_importance = np.abs(as_np(model.bnorm_1.weight)) * (
        np.abs(w_point) * feature_importance[:, None]
    ).sum(axis=0)
    channel_score = (np.abs(w_spatial) * filter_importance[:, None]).sum(axis=0)
    channel_score = channel_score / channel_score.sum()

    # Temporal kernel k feeds spatial filters k*D ... k*D+D-1.
    kernel_importance = filter_importance.reshape(model.F1, model.D).sum(axis=1)
    n_fft = 8 * w_temporal.shape[1]
    power = np.abs(np.fft.rfft(w_temporal, n=n_fft, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)
    peak_hz = float(freqs[np.argmax(kernel_importance @ power)])

    # Classifier time bins -> seconds -> one-second windows.
    energy = (feature_importance[None, :, None] * contrast**2).sum(axis=(0, 1))
    n_bins = energy.size
    centres_s = (
        window_start_s + (np.arange(n_bins) + 0.5) * model.n_times / n_bins / sfreq
    )
    second = np.floor(centres_s).astype(int)
    window_energy = {s: energy[second == s].mean() for s in np.unique(second)}
    best = int(max(window_energy, key=window_energy.get))

    return {
        "channel_score": dict(zip(ch_names, channel_score.tolist())),
        "band": next(name for low, high, name in BANDS if low <= peak_hz < high),
        "peak_hz": peak_hz,
        "window": f"{best:+d}..{best + 1:+d} s",
    }


reading = read_weights(model, CHANNELS, SFREQ, 0.0, outputs=[LEFT, RIGHT])
print(f"Band: {reading['band']} (peak {reading['peak_hz']:.1f} Hz)")
print(f"Window with most classifier contrast: {reading['window']}")
print(
    "Channel shares:", {ch: round(s, 3) for ch, s in reading["channel_score"].items()}
)

######################################################################
# The weights point to the slow band and the first second after the cue,
# the shortcut signature. The channel shares put Cz first; keep that as
# the least certain part of the reading. The next step checks it on data.
#
# Confirm on data: one occlusion test
# -----------------------------------
#
# Loading PhysioNet as the checkpoint saw it
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We load ten subjects of the PhysioNet motor imagery dataset [6]_ through
# MOABB and rebuild the input of the checkpoint from its paper: C3, Cz and
# C4 in microvolts, 0.5-40 Hz, resampled to 128 Hz, 385 samples from the
# cue. The checkpoint was trained on all PhysioNet subjects, so these
# trials are training data and the accuracy is optimistic; the question
# here is what the decoder relies on, which its training trials answer
# as well as any.

subjects = list(range(1, 11))
dataset = MOABBDataset(dataset_name="PhysionetMI", subject_ids=subjects)
preprocess(
    dataset,
    [
        Preprocessor("pick", picks=CHANNELS),
        Preprocessor(lambda data: multiply(data, 1e6)),  # V to uV
        Preprocessor("filter", l_freq=0.5, h_freq=40.0),
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
# The reading is a claim. One test checks it: remove a band, a time
# window or a channel, and see how far the left-versus-right accuracy
# drops. A decoder that reads the cue-directed eye movement loses accuracy
# when 0.5-4 Hz is removed, when only 8-30 Hz is kept, or when the first
# second is zeroed, and keeps it when 1-3 s is zeroed. A decoder that
# reads the sensorimotor rhythm does the opposite.


def _accuracy(model, X, y):
    with torch.no_grad():
        logits = model(torch.as_tensor(X))
    return float(((logits[:, RIGHT] > logits[:, LEFT]).numpy() == y).mean())


def occlusion_check(model, X, y, ch_names, sfreq):
    """Accuracy drop when a band, a time window or a channel is removed."""
    baseline = _accuracy(model, X, y)

    def drop(X_changed):
        return baseline - _accuracy(model, X_changed, y)

    def filtered(kind, freqs):
        sos = butter(4, freqs, kind, fs=sfreq, output="sos")
        return sosfiltfilt(sos, X, axis=-1).astype(np.float32)

    def zeroed(channel=slice(None), samples=slice(None)):
        X_zeroed = X.copy()
        X_zeroed[:, channel, samples] = 0.0
        return X_zeroed

    one_s = int(sfreq)
    drops = {
        "remove\n0.5-4 Hz": drop(filtered("highpass", 4.0)),
        "keep\n8-30 Hz only": drop(filtered("bandpass", [8.0, 30.0])),
        "zero\n0-1 s": drop(zeroed(samples=slice(0, one_s))),
        "zero\n1-3 s": drop(zeroed(samples=slice(one_s, None))),
    }
    for i, name in enumerate(ch_names):
        drops[f"zero\n{name}"] = drop(zeroed(channel=i))
    return baseline, drops


baseline, drops = occlusion_check(model, X, y, CHANNELS, SFREQ)
print(f"Left vs right accuracy: {baseline:.1%}")
for name, value in drops.items():
    print(f"  {name.replace(chr(10), ' '):>18s}: drop {value:+.1%}")

######################################################################
# The chart puts the data check next to what the weights said (in the
# title). Orange bars are the tests an eye-movement decoder fails, the
# blue bar the one a sensorimotor decoder fails, grey bars remove one
# electrode. On this checkpoint the orange bars cost 14-17 points each
# and zeroing 1-3 s costs nothing: the data agree with the band and the
# window read from the weights. Zeroing C3 costs the most, while the
# weights put Cz first, so the channel ranking is where the reading and
# the data part.

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.bar(
    list(drops),
    100 * np.array(list(drops.values())),
    color=["#D55E00"] * 3 + ["#0072B2"] + ["#999999"] * 3,
)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Accuracy drop (points)")
ax.set_title(
    f"PhysioNet EEGNet checkpoint, left vs right {baseline:.0%}; "
    f"weights say {reading['band']}, {reading['window']}",
    fontsize=9,
)
fig.tight_layout()

######################################################################
# What to do with the result
# --------------------------
#
# Record it with the model. A model card for a released decoder should
# carry a *relies on* field that states the band, window and channels
# the audit found, the method (weight statistic, occlusion on which
# subjects), the date, and whether the data check agreed with the weight
# reading. A decoder whose entry says "below 4 Hz, 0-1 s after the cue" is
# a different product from one that says "8-30 Hz, 1-3 s", even at the
# same accuracy. Two more lines make the next audit easier: the exact
# preprocessing of the input, and the initialisation seed. With the seed,
# the reading can be repeated on the change that training made to the
# weights, which removes channels that rank high only because of the
# starting noise [7]_.
#
# The occlusion test here is a check on the reading, not a proof of
# generalisation. The real test is cross-dataset transfer: a decoder
# that reads cue-directed eye movements falls to chance on a dataset
# whose cue is in the centre of the screen, while a decoder that reads
# the sensorimotor rhythm keeps some of its accuracy. If the weight
# reading says "eye movement", run that transfer before relying on the
# decoder.
#
# References
# ----------
#
# .. [1] Gao, Y. (2026). *Do shortcut audits measure reliance in EEG
#        decoders? A preregistered semi-synthetic ground-truth test and a
#        real-data check of eye-movement use.* Preregistered (OSF);
#        code: https://github.com/jabarrick/physio-shortcut-audit
# .. [2] Takehi, R., Teoh, J., Zhang, A., & Khattab, O. (2026). *Sparks of
#        Weight-Space Programming: LLMs can natively read & write neural
#        network weights.* MIT CSAIL.
#        https://rikiyat.github.io/blog/weight-space-programming/
# .. [3] Guetschel, P., & Tangermann, M. (2023). *Transfer learning
#        between motor imagery datasets using deep learning: validation
#        of framework and comparison of datasets.* arXiv:2311.16109.
#        Weights: https://huggingface.co/PierreGtch/EEGNetv4
# .. [4] Lawhern, V. J., et al. (2018). *EEGNet: a compact convolutional
#        neural network for EEG-based brain-computer interfaces.* Journal
#        of Neural Engineering, 15(5), 056013.
#        DOI: 10.1088/1741-2552/aace8c
# .. [5] Haufe, S., et al. (2014). *On the interpretation of weight
#        vectors of linear models in multivariate neuroimaging.*
#        NeuroImage, 87, 96-110. DOI: 10.1016/j.neuroimage.2013.10.067
# .. [6] Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N., &
#        Wolpaw, J. R. (2004). *BCI2000: A general-purpose brain-computer
#        interface (BCI) system.* IEEE Transactions on Biomedical
#        Engineering, 51(6), 1034-1043. DOI: 10.1109/TBME.2004.827072
# .. [7] Li, Y. (2026). *Reading the weights: a language model audits
#        and repairs EEG decoders without data.* Preprint.
#        DOI: 10.5281/zenodo.23934403
