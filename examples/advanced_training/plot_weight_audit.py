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
takes that route on an EEG decoder: it reads the weights first, with a
statistic that needs no language model, and then confirms the reading with
one occlusion test on held-out data.

.. topic:: What the audit looks for

    EEGNet learns a filter bank (temporal kernels), one set of spatial
    filters per kernel, and a classifier over the pooled features. Each
    tensor is readable on its own: the spectral peak of a temporal kernel
    says which band it passes, the spatial weights say which electrodes it
    mixes, and the classifier says which pooled time bins separate the
    classes. Pooling these through the network's own path weights gives a
    data-free estimate of the band and the time window the decoder relies
    on, and a channel ranking whose sharpness depends on how structured
    the spatial filters are. On PhysioNet the shortcut signature is a band
    below 4 Hz, a window in the first second after the cue and, once the
    spatial filters have structure, a ranking led by the eye-adjacent
    electrodes (Fp1, Fp2, AF7, AF8, F7, F8, FT7, FT8); the sensorimotor
    signature is 8-30 Hz, 1-4 s and C3/C4.

.. contents:: This example covers:
   :local:
   :depth: 2

"""

# Authors: Yiheng Li <yihengl4@asu.edu>
#
# License: BSD (3-clause)

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import multiply
from scipy.signal import butter, sosfiltfilt
from skorch.dataset import ValidSplit

from braindecode import EEGClassifier
from braindecode.datasets import BaseConcatDataset, MOABBDataset
from braindecode.models import EEGNet
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)
from braindecode.util import set_random_seeds

######################################################################
# Data and model
# --------------
#
# Loading PhysioNet motor imagery
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We load the PhysioNet motor imagery dataset [3]_ through MOABB. Each
# subject performed three runs of imagined left- or right-fist movement
# with 64 electrodes at 160 Hz; the cue is a target on the left or right
# edge of the screen. We train on eight subjects and hold three out,
# which keeps the docs build short. Each subject contributes about 45
# trials, so one trial is worth 0.7 accuracy points on the held-out
# set; for a real audit use more held-out subjects still.

train_subjects = [1, 2, 3, 4, 5, 6, 7, 8]
test_subjects = [9, 10, 11]
dataset = MOABBDataset(
    dataset_name="PhysionetMI", subject_ids=train_subjects + test_subjects
)

######################################################################
# Preprocessing: keep the slow signal in
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We deliberately do *not* band-pass to 8-30 Hz. The eye-movement
# potential lives below 4 Hz, and the point of the audit is to see
# whether the decoder reaches for it when it is available. A 0.5 Hz
# high-pass removes drift and nothing else. If you release a decoder
# trained on broadband data, this is the regime it saw.

preprocess(
    dataset,
    [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        Preprocessor(lambda data: multiply(data, 1e6)),  # V to uV
        Preprocessor("filter", l_freq=0.5, h_freq=None),
    ],
    n_jobs=1,
)

######################################################################
# Windows from 1 s before to 4 s after the cue
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The pre-cue second is included on purpose: a decoder that uses it is
# not decoding imagery. MOABB annotates each trial as 3 s long, so an
# offset of -1 s at the start and +1 s at the end gives one 5 s window
# per trial. MOABB also loads the hands-versus-feet runs of each
# subject; we keep only the runs that contain left- and right-hand
# trials.

hand_runs = BaseConcatDataset(
    [ds for ds in dataset.datasets if "left_hand" in ds.raw.annotations.description]
)
sfreq = hand_runs.datasets[0].raw.info["sfreq"]
windows_dataset = create_windows_from_events(
    hand_runs,
    trial_start_offset_samples=int(-1.0 * sfreq),
    trial_stop_offset_samples=int(1.0 * sfreq),
    mapping={"left_hand": 0, "right_hand": 1},
    preload=True,
)
window_start_s = -1.0
ch_names = windows_dataset.datasets[0].raw.ch_names

by_subject = windows_dataset.split("subject")
train_set = BaseConcatDataset([by_subject[str(s)] for s in train_subjects])
test_set = BaseConcatDataset([by_subject[str(s)] for s in test_subjects])


def _to_arrays(windows):
    X = np.stack([np.asarray(x) for x, _, _ in windows]).astype(np.float32)
    y = np.array([int(label) for _, label, _ in windows])
    return X, y


X_train, y_train = _to_arrays(train_set)
X_test, y_test = _to_arrays(test_set)
n_chans, n_times = X_train.shape[1:]
print(f"{len(X_train)} training trials, {len(X_test)} held-out trials")
print(f"{n_chans} channels x {n_times} samples at {sfreq:g} Hz")

######################################################################
# Training EEGNet briefly
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# :class:`~braindecode.models.EEGNet` [4]_ is the decoder under audit.
# A few dozen epochs on CPU are enough for the purpose of this example;
# the audit does not need a well-tuned model, it needs the model that
# will be released.

seed = 20261103
set_random_seeds(seed=seed, cuda=False)
device = "cpu"

model = EEGNet(
    n_chans=n_chans,
    n_outputs=2,
    n_times=n_times,
    sfreq=sfreq,
    final_layer_with_constraint=True,
)

clf = EEGClassifier(
    model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    optimizer__lr=1e-3,
    optimizer__weight_decay=1e-4,
    train_split=ValidSplit(0.2, stratified=True, random_state=seed),
    batch_size=32,
    max_epochs=30,
    callbacks=["accuracy"],
    device=device,
    classes=[0, 1],
    verbose=0,
)
clf.fit(X_train, y_train)

held_out_accuracy = clf.score(X_test, y_test)
print(f"Held-out accuracy on subjects {test_subjects}: {held_out_accuracy:.1%}")
model = clf.module_.eval()

######################################################################
# Dump the weights as labelled text
# ---------------------------------
#
# Everything that follows works from the trained weights alone. The
# first step is to write them out in a form a reader (human, script or
# language model) can use without the model class: one line per row,
# with the spatial filters labelled by electrode name and the temporal
# kernels written as FIR taps. Batch-norm statistics are included
# because they set the sign and scale of each feature.


def weights_to_text(model, ch_names, sfreq):
    """Write every tensor of ``model`` as labelled text and return it.

    Spatial filters (shape ``(n_filters, 1, n_chans, 1)``) get one row per
    filter with ``electrode=weight`` pairs; temporal kernels (shape
    ``(n_kernels, 1, 1, n_taps)``) get one row per kernel; every other
    tensor is written row by row.
    """
    n_chans = len(ch_names)
    lines = [
        (
            f"# {model.__class__.__name__}: input (n_channels={n_chans}, n_times) "
            f"at {sfreq:g} Hz, output = class scores"
        ),
        "# channel order: " + " ".join(ch_names),
    ]
    for name, module in model.named_modules():
        for attr in ("weight", "bias", "running_mean", "running_var"):
            value = getattr(module, attr, None)
            if not torch.is_tensor(value):
                continue
            array = value.detach().cpu().numpy()
            lines.append(f"## {name}.{attr} shape={tuple(array.shape)}")
            if array.ndim == 4 and array.shape[2] == n_chans and array.shape[3] == 1:
                for i, row in enumerate(array[:, 0, :, 0]):
                    pairs = " ".join(f"{ch}={w:+.3f}" for ch, w in zip(ch_names, row))
                    lines.append(f"spatial_filter{i:02d} {pairs}")
            elif array.ndim == 4 and array.shape[1] == 1 and array.shape[2] == 1:
                lines.append(f"# FIR taps, {array.shape[3] / sfreq * 1000:.0f} ms long")
                for i, row in enumerate(array[:, 0, 0, :]):
                    lines.append(
                        f"temporal_kernel{i:02d} " + " ".join(f"{v:+.4f}" for v in row)
                    )
            elif array.ndim <= 1:
                lines.append(" ".join(f"{v:+.4f}" for v in np.atleast_1d(array)))
            else:
                for i, row in enumerate(array.reshape(array.shape[0], -1)):
                    lines.append(f"row{i:02d} " + " ".join(f"{v:+.4f}" for v in row))
    return "\n".join(lines) + "\n"


weights_text = weights_to_text(model, ch_names, sfreq)
print(f"{len(weights_text.splitlines())} lines, {len(weights_text)} characters")
print("\n".join(line[:100] for line in weights_text.splitlines()[:12]))

######################################################################
# A statistics reader: no language model needed
# ---------------------------------------------
#
# The reader below answers three questions from the weights with
# path-weighted statistics, following the EEGNet forward path backwards:
#
# * **Which features matter?** The classifier weights, reshaped to
#   ``(n_outputs, F2, n_time_bins)``, are centred across classes; a
#   feature's importance is the norm of that contrast times its
#   batch-norm gain.
# * **Which spatial filters matter?** Feature importance is pushed back
#   through the absolute pointwise weights and the first batch-norm gain.
# * **Which channels?** Absolute spatial weight times filter importance,
#   summed over filters, normalised to sum to one.
# * **Which band?** Each temporal kernel is an FIR filter; its power
#   spectrum peak, weighted by the importance of the spatial filters it
#   feeds, gives the band.
# * **Which time window?** The classifier contrast energy per pooled time
#   bin, mapped back to seconds, averaged in one-second windows.
#
# None of this touches data. It is what a maintainer can compute from a
# released checkpoint in a second. One limit to keep in mind: a spatial
# weight says how much a filter *mixes* an electrode, not how much
# signal arrives there (Haufe et al. [5]_ make this point for linear
# models). A briefly trained EEGNet keeps dense, near-initial spatial
# filters, and its channel ranking is then close to flat even when one
# electrode carries the decision. The band and the window do not have
# this problem, because they come from the temporal kernels and the
# classifier, which training shapes quickly. We therefore print the
# ranking together with its flatness, as the share held by the top
# channel against the uniform share ``1 / n_chans``, and the share held
# by two electrode groups against their uniform shares.

BANDS = [
    (0.0, 4.0, "slow (<4 Hz)"),
    (4.0, 8.0, "theta (4-8 Hz)"),
    (8.0, 13.0, "mu/alpha (8-13 Hz)"),
    (13.0, 30.0, "beta (13-30 Hz)"),
    (30.0, np.inf, "gamma (>30 Hz)"),
]


def _band_name(freq_hz):
    return next(name for low, high, name in BANDS if low <= freq_hz < high)


def read_weights(model, ch_names, sfreq, window_start_s=-1.0, n_top=5):
    """Estimate the channels, band and time window an EEGNet relies on.

    Returns a dict with the top channels (path-weighted absolute spatial
    weight, normalised), the spectral peak and band of every temporal
    kernel with its importance, the importance-weighted dominant band,
    and the one-second window where the classifier contrast is largest.
    """

    def as_np(t):
        return t.detach().cpu().numpy()

    w_spatial = as_np(model.conv_spatial.weight)[:, 0, :, 0]  # (F1*D, n_chans)
    w_temporal = as_np(model.conv_temporal.weight)[:, 0, 0, :]  # (F1, n_taps)
    w_point = as_np(model.conv_separable_point.weight)[:, :, 0, 0]  # (F2, F1*D)
    gain_1 = as_np(model.bnorm_1.weight)
    gain_2 = as_np(model.bnorm_2.weight)
    head = next(
        m
        for m in model.final_layer.modules()
        if torch.is_tensor(getattr(m, "weight", None))
    )
    w_head = as_np(head.weight).reshape(model.n_outputs, model.F2, -1)
    n_bins = w_head.shape[-1]

    # Feature -> spatial filter -> channel.
    contrast = w_head - w_head.mean(axis=0, keepdims=True)
    feature_importance = np.abs(gain_2) * np.sqrt((contrast**2).sum(axis=(0, 2)))
    filter_importance = np.abs(gain_1) * (
        np.abs(w_point) * feature_importance[:, None]
    ).sum(axis=0)
    channel_score = (np.abs(w_spatial) * filter_importance[:, None]).sum(axis=0)
    channel_score = channel_score / channel_score.sum()
    order = np.argsort(channel_score)[::-1]
    top_channels = [
        (ch_names[i], round(float(channel_score[i]), 4)) for i in order[:n_top]
    ]

    # Temporal kernel k feeds spatial filters k*D ... k*D+D-1.
    kernel_importance = filter_importance.reshape(model.F1, model.D).sum(axis=1)
    n_fft = 8 * w_temporal.shape[1]
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)
    power = np.abs(np.fft.rfft(w_temporal, n=n_fft, axis=1)) ** 2
    kernels = []
    for k, (spectrum, importance) in enumerate(zip(power, kernel_importance)):
        peak_hz = float(freqs[np.argmax(spectrum)])
        kernels.append(
            {
                "kernel": k,
                "peak_hz": round(peak_hz, 1),
                "band": _band_name(peak_hz),
                "importance": round(float(importance / kernel_importance.sum()), 3),
            }
        )
    pooled = (power * kernel_importance[:, None]).sum(axis=0)
    dominant_hz = float(freqs[np.argmax(pooled)])

    # Classifier time bins -> seconds -> one-second windows.
    energy = (feature_importance[None, :, None] * contrast**2).sum(axis=(0, 1))
    bin_centres_s = (
        window_start_s + (np.arange(n_bins) + 0.5) * model.n_times / n_bins / sfreq
    )
    window_energy = {}
    for start in np.arange(window_start_s, window_start_s + model.n_times / sfreq):
        in_window = (bin_centres_s >= start) & (bin_centres_s < start + 1)
        if in_window.any():
            window_energy[f"{start:+.0f}..{start + 1:+.0f} s"] = float(
                energy[in_window].mean()
            )
    best_window = max(window_energy, key=window_energy.get)

    return {
        "top_channels": top_channels,
        "channel_score": dict(zip(ch_names, channel_score.round(4).tolist())),
        "kernels": kernels,
        "dominant_band": _band_name(dominant_hz),
        "dominant_hz": round(dominant_hz, 1),
        "window": best_window,
        "window_energy": window_energy,
    }


EYE_ADJACENT = ["Fp1", "Fp2", "AF7", "AF8", "F7", "F8", "FT7", "FT8"]
MOTOR = ["C3", "C1", "Cz", "C2", "C4", "FC3", "FC4", "CP3", "CP4"]
GROUPS = {"eye-adjacent": EYE_ADJACENT, "motor": MOTOR}


def group_share(reading, groups):
    """Share of the reader's channel score in each group, with the uniform share."""
    n_chans = len(reading["channel_score"])
    return {
        name: (sum(reading["channel_score"][ch] for ch in names), len(names) / n_chans)
        for name, names in groups.items()
    }


reading = read_weights(model, ch_names, sfreq, window_start_s=window_start_s)
print(
    "Top channels (share of path-weighted spatial weight; "
    f"uniform share would be {1 / n_chans:.3f}):"
)
for name, score in reading["top_channels"]:
    print(f"  {name:>4s}  {score:.3f}")
print(f"Dominant band: {reading['dominant_band']} (peak {reading['dominant_hz']} Hz)")
print(
    "Per kernel:",
    [(k["kernel"], k["peak_hz"], k["importance"]) for k in reading["kernels"]],
)
print(f"Time window with most classifier contrast: {reading['window']}")
for name, (share, uniform) in group_share(reading, GROUPS).items():
    print(f"Reader's share on {name:>12s} group: {share:.3f} (uniform {uniform:.3f})")

######################################################################
# Optional: a language-model reader
# ---------------------------------
#
# A language model can read the same text and add what the statistic
# cannot: the *family* of the signal (brain rhythm, eye movement, evoked
# potential, drift) and the class rule, by reasoning over the sign of
# the spatial weights and the shape of the kernels. Takehi et al. (2026)
# report this kind of reading on small networks. On planted
# single-channel cues the statistic above localises channel, band and
# window as well as a language model does; the language model's value
# is the interpretation, and on dense filters like the ones above it can
# also use the sign pattern across lateral electrode pairs (F7 against
# F8, AF7 against AF8), which a magnitude statistic throws away.
#
# The prompt and the answer format are fixed so that the answer can be
# scored against the data check below. The call is made only when
# ``BRAINDECODE_AUDIT_LLM_KEY`` is set, so the docs build never makes a
# network request, and it goes through a function you fill in with your
# provider's client; no SDK is imported here.

AUDIT_PROMPT = f"""You are auditing an EEG decoder from its weights alone.
Architecture: EEGNet (temporal conv -> depthwise spatial conv -> separable
conv -> linear classifier). Input: {n_chans} channels at {sfreq:g} Hz,
{n_times} samples per trial from {window_start_s:+.0f} s to
{window_start_s + n_times / sfreq:+.0f} s around the cue, high-passed at
0.5 Hz, in microvolts. Task: imagined left hand (class 0) versus right
hand (class 1). The cue appeared on the left or right of the screen.

Analyse the weights below with code if you like, but do not run the
network on any input. Say which electrodes, which frequency band and which
time window the decoder relies on, which signal family that points to, and
what marks the right-hand class. End with exactly one line of JSON:
{{"top_channels": [...], "band_hz": [low, high], "window_s": [start, stop],
"signal": "brain rhythm | eye movement | evoked potential | other",
"right_hand_rule": "...", "confidence": 0.0-1.0}}

WEIGHTS
"""


def ask_language_model(prompt, api_key):
    """Send ``prompt`` to a language model and return its reply as text.

    Replace the body with your provider's call. Keep the model from
    running the network: give it the text, not the checkpoint.
    """
    raise NotImplementedError("Fill in your provider's API call here.")


api_key = os.environ.get("BRAINDECODE_AUDIT_LLM_KEY")
if api_key is None:
    print("Language-model reader skipped: BRAINDECODE_AUDIT_LLM_KEY is not set.")
else:
    print(ask_language_model(AUDIT_PROMPT + weights_text, api_key))

######################################################################
# Confirm on data: one occlusion test
# -----------------------------------
#
# The reading is a claim. One test on the held-out subjects checks it:
# zero a group of channels, or remove a band, and see how far the
# accuracy drops. The two named groups from above are compared against
# random groups of the same size, so a drop can be read against chance.
# On PhysioNet a sizeable drop for the eye-adjacent group and for the
# 0.5-4 Hz band, with little lost when the motor group goes, is the
# shortcut signature. The reader's band and window should agree with
# the band check, and its channel ranking should point to the same
# electrodes as the occlusion; when the ranking is flat, the occlusion
# is what settles the channels, and the model card should say so.


def _accuracy(model, X, y, batch_size=64):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32)
            predictions.append(model(batch).argmax(dim=1).cpu().numpy())
    return float((np.concatenate(predictions) == y).mean())


def occlusion_check(model, X, y, groups, ch_names, n_random=20, seed=0):
    """Accuracy drop when each channel group is zeroed, against random groups.

    ``groups`` maps a name to a list of channel names. For every group
    size present, ``n_random`` random groups of that size are also
    zeroed; their drops give the chance level for the named groups.
    """
    baseline = _accuracy(model, X, y)

    def masked(names):
        X_masked = X.copy()
        X_masked[:, [ch_names.index(name) for name in names], :] = 0.0
        return X_masked

    drops = {
        name: baseline - _accuracy(model, masked(names), y)
        for name, names in groups.items()
    }
    rng = np.random.default_rng(seed)
    random_drops = {}
    for size in sorted({len(names) for names in groups.values()}):
        random_drops[size] = np.array(
            [
                baseline
                - _accuracy(model, masked(rng.choice(ch_names, size, replace=False)), y)
                for _ in range(n_random)
            ]
        )
    return {"baseline": baseline, "drops": drops, "random": random_drops}


def band_check(model, X, y, sfreq):
    """Accuracy drop when the slow band is removed or only 8-30 Hz is kept."""
    baseline = _accuracy(model, X, y)

    def filtered(kind, freqs):
        sos = butter(4, freqs, kind, fs=sfreq, output="sos")
        return sosfiltfilt(sos, X, axis=-1).astype(np.float32)

    return {
        "remove 0.5-4 Hz": baseline - _accuracy(model, filtered("highpass", 4.0), y),
        "keep 8-30 Hz only": baseline
        - _accuracy(model, filtered("bandpass", [8.0, 30.0]), y),
    }


occlusion = occlusion_check(model, X_test, y_test, GROUPS, ch_names)
bands = band_check(model, X_test, y_test, sfreq)

print(f"Baseline accuracy: {occlusion['baseline']:.1%}")
for name, drop in occlusion["drops"].items():
    size = len(GROUPS[name])
    random = occlusion["random"][size]
    print(
        f"  zero {name:>12s} ({size} ch): drop {drop:+.1%}; "
        f"random {size}-channel groups: mean {random.mean():+.1%}, "
        f"95th percentile {np.percentile(random, 95):+.1%}"
    )
for name, drop in bands.items():
    print(f"  {name:>18s}: drop {drop:+.1%}")

######################################################################
# Put the reader and the data check side by side
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# First as a table: what the weights said, what the data said.

shares = group_share(reading, GROUPS)
rows = [("", "weights", "data (accuracy drop)")]
for name in GROUPS:
    share, uniform = shares[name]
    rows.append(
        (
            name,
            f"share {share:.3f} (uniform {uniform:.3f})",
            f"zeroed: {occlusion['drops'][name]:+.1%}",
        )
    )
rows.append(
    (
        "band",
        reading["dominant_band"],
        f"0.5-4 Hz removed: {bands['remove 0.5-4 Hz']:+.1%}",
    )
)
rows.append(("window", reading["window"], ""))
for row in rows:
    print(f"{row[0]:>14s}  {row[1]:<32s}  {row[2]}")

######################################################################
# Then as a bar chart of the accuracy drops. The dashed line is the
# 95th percentile of the random groups of the eye group's size, the
# level a named group has to clear before its drop means anything.

eye_size, motor_size = len(EYE_ADJACENT), len(MOTOR)
labels = [
    f"eye-adjacent\n({eye_size} ch)",
    f"motor\n({motor_size} ch)",
    f"random\n({eye_size} ch, mean)",
    "remove\n0.5-4 Hz",
    "keep\n8-30 Hz only",
]
values = [
    occlusion["drops"]["eye-adjacent"],
    occlusion["drops"]["motor"],
    occlusion["random"][eye_size].mean(),
    bands["remove 0.5-4 Hz"],
    bands["keep 8-30 Hz only"],
]
colors = ["#D55E00", "#0072B2", "#999999", "#D55E00", "#0072B2"]

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.bar(labels, np.array(values) * 100, color=colors)
ax.axhline(
    np.percentile(occlusion["random"][eye_size], 95) * 100,
    ls="--",
    color="#444",
    lw=1,
    label=f"random {eye_size}-channel groups, 95th percentile",
)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Accuracy drop (points)")
ax.set_title(
    f"Held-out subjects {test_subjects}: baseline {occlusion['baseline']:.0%}; "
    f"weights say {reading['dominant_band']}, {reading['window']}",
    fontsize=9,
)
ax.legend(fontsize=8, loc="upper left")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()

######################################################################
# What to do with the result
# --------------------------
#
# Record it with the model. A model card for a released decoder should
# carry a *relies on* field that states the channels, band and window
# the audit found, the method (weight statistic, language-model read,
# occlusion on which subjects), the date, and whether the data check
# agreed with the weight reading. A decoder whose entry says "F7/F8/AF7/
# AF8, below 4 Hz, 0-1 s after the cue" is a different product from one
# that says "C3/C4, 8-30 Hz, 1-4 s", even at the same accuracy.
#
# The occlusion test here is a check on the reading, not a proof of
# generalisation. The real test is cross-dataset transfer: a decoder
# that reads cue-directed eye movements falls to chance on a dataset
# whose cue is in the centre of the screen, while a decoder that reads
# the sensorimotor rhythm keeps some of its accuracy. If the weight
# reading says "eye movement", run that transfer before releasing.
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
# .. [3] Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N., &
#        Wolpaw, J. R. (2004). *BCI2000: A general-purpose brain-computer
#        interface (BCI) system.* IEEE Transactions on Biomedical
#        Engineering, 51(6), 1034-1043. DOI: 10.1109/TBME.2004.827072
# .. [4] Lawhern, V. J., et al. (2018). *EEGNet: a compact convolutional
#        neural network for EEG-based brain-computer interfaces.* Journal
#        of Neural Engineering, 15(5), 056013.
#        DOI: 10.1088/1741-2552/aace8c
# .. [5] Haufe, S., et al. (2014). *On the interpretation of weight
#        vectors of linear models in multivariate neuroimaging.*
#        NeuroImage, 87, 96-110. DOI: 10.1016/j.neuroimage.2013.10.067
