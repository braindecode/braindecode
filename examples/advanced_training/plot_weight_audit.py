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
takes that route on an EEG decoder with the simplest possible reader: a
path-weighted statistic of the weights, confirmed by one occlusion test on
held-out data.

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
window_start_s = -1.0
windows_dataset = create_windows_from_events(
    hand_runs,
    trial_start_offset_samples=int(window_start_s * sfreq),
    trial_stop_offset_samples=int(1.0 * sfreq),
    mapping={"left_hand": 0, "right_hand": 1},
    preload=True,
)
ch_names = windows_dataset.datasets[0].raw.ch_names

by_subject = windows_dataset.split("subject")


def _to_arrays(subjects):
    windows = BaseConcatDataset([by_subject[str(s)] for s in subjects])
    X = np.stack([np.asarray(x) for x, _, _ in windows]).astype(np.float32)
    y = np.array([int(label) for _, label, _ in windows])
    return X, y


X_train, y_train = _to_arrays(train_subjects)
X_test, y_test = _to_arrays(test_subjects)
n_chans, n_times = X_train.shape[1:]
print(f"{len(X_train)} training trials, {len(X_test)} held-out trials")

######################################################################
# Training EEGNet briefly
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# :class:`~braindecode.models.EEGNet` [4]_ is the decoder under audit.
# A few dozen epochs on CPU are enough for the purpose of this example;
# the audit does not need a well-tuned model, it needs the model that
# will be released. We build the network from a seed so that its
# initialisation can be rebuilt later.

seed = 20261103


def make_eegnet(init_seed):
    set_random_seeds(seed=init_seed, cuda=False)
    return EEGNet(
        n_chans=n_chans,
        n_outputs=2,
        n_times=n_times,
        sfreq=sfreq,
        final_layer_with_constraint=True,
    )


clf = EEGClassifier(
    make_eegnet(seed),
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    optimizer__lr=1e-3,
    optimizer__weight_decay=1e-4,
    train_split=ValidSplit(0.2, stratified=True, random_state=seed),
    batch_size=32,
    max_epochs=30,
    callbacks=["accuracy"],
    device="cpu",
    classes=[0, 1],
    verbose=0,
)
clf.fit(X_train, y_train)
print(f"Held-out accuracy: {clf.score(X_test, y_test):.1%}")
model = clf.module_.eval()

######################################################################
# Reading the weights
# -------------------
#
# The reader follows the EEGNet forward path backwards:
#
# * **Which features matter?** The classifier weights, reshaped to
#   ``(n_outputs, F2, n_time_bins)``, are centred across classes; a
#   feature's importance is the norm of that contrast times its
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
# None of this touches data: it is what a maintainer can compute from a
# released checkpoint in a second. One limit to keep in mind: a spatial
# weight says how much a filter *mixes* an electrode, not how much
# signal arrives there (Haufe et al. [5]_ make this point for linear
# models). A briefly trained EEGNet keeps dense, near-initial spatial
# filters, so its channel ranking can be close to flat even when one
# electrode carries the decision. The band and the window come from the
# temporal kernels and the classifier, which training shapes quickly. We
# therefore print the top channel's share next to the uniform share, and
# the share held by two electrode groups next to theirs.

BANDS = [
    (0.0, 4.0, "slow (<4 Hz)"),
    (4.0, 8.0, "theta (4-8 Hz)"),
    (8.0, 13.0, "mu/alpha (8-13 Hz)"),
    (13.0, 30.0, "beta (13-30 Hz)"),
    (30.0, np.inf, "gamma (>30 Hz)"),
]


def read_weights(model, ch_names, sfreq, window_start_s, spatial_weight=None):
    """Estimate the channels, band and time window an EEGNet relies on.

    ``spatial_weight`` replaces the spatial filters, e.g. by the change
    that training made to them.
    """

    def as_np(t):
        return t.detach().cpu().numpy()

    if spatial_weight is None:
        spatial_weight = model.conv_spatial.weight
    w_spatial = as_np(spatial_weight)[:, 0, :, 0]  # (F1*D, n_chans)
    w_temporal = as_np(model.conv_temporal.weight)[:, 0, 0, :]  # (F1, n_taps)
    w_point = as_np(model.conv_separable_point.weight)[:, :, 0, 0]  # (F2, F1*D)
    head = next(
        m
        for m in model.final_layer.modules()
        if torch.is_tensor(getattr(m, "weight", None))
    )
    w_head = as_np(head.weight).reshape(model.n_outputs, model.F2, -1)

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


EYE_ADJACENT = ["Fp1", "Fp2", "AF7", "AF8", "F7", "F8", "FT7", "FT8"]
MOTOR = ["C3", "C1", "Cz", "C2", "C4", "FC3", "FC4", "CP3", "CP4"]
GROUPS = {"eye-adjacent": EYE_ADJACENT, "motor": MOTOR}


def top_channels(reading, n=5):
    scores = reading["channel_score"]
    return sorted(scores, key=scores.get, reverse=True)[:n]


def print_channels(reading, label):
    top = top_channels(reading)
    share = reading["channel_score"][top[0]]
    print(f"{label}: top channels {top}")
    print(f"  top share {share:.3f} (uniform {1 / n_chans:.3f})")
    for name, names in GROUPS.items():
        group = sum(reading["channel_score"][ch] for ch in names)
        print(f"  {name} share {group:.3f} (uniform {len(names) / n_chans:.3f})")


reading = read_weights(model, ch_names, sfreq, window_start_s)
print_channels(reading, "Weights")
print(f"Band: {reading['band']} (peak {reading['peak_hz']:.1f} Hz)")
print(f"Window with most classifier contrast: {reading['window']}")

######################################################################
# Confirm on data: one occlusion test
# -----------------------------------
#
# The reading is a claim. One test on the held-out subjects checks it:
# zero a group of channels, or remove a band, and see how far the
# accuracy drops. The two named groups are compared against random
# groups of the same size, so a drop can be read against chance. On
# PhysioNet a sizeable drop for the eye-adjacent group and for the
# 0.5-4 Hz band, with little lost when the motor group goes, is the
# shortcut signature. The band check should agree with the reader's
# band; when the channel ranking is flat, the occlusion is what settles
# the channels, and the model card should say so.


def _accuracy(model, X, y):
    with torch.no_grad():
        predictions = model(torch.as_tensor(X)).argmax(dim=1).numpy()
    return float((predictions == y).mean())


def occlusion_check(model, X, y, ch_names, sfreq, n_random=20, seed=0):
    """Accuracy drop when channel groups are zeroed or bands are removed.

    Each group in ``GROUPS`` is compared with ``n_random`` random groups
    of the same size, whose drops give the chance level.
    """
    baseline = _accuracy(model, X, y)

    def drop(X_changed):
        return baseline - _accuracy(model, X_changed, y)

    def zeroed(names):
        X_zeroed = X.copy()
        X_zeroed[:, [ch_names.index(name) for name in names]] = 0.0
        return X_zeroed

    def filtered(kind, freqs):
        sos = butter(4, freqs, kind, fs=sfreq, output="sos")
        return sosfiltfilt(sos, X, axis=-1).astype(np.float32)

    rng = np.random.default_rng(seed)
    sizes = sorted({len(names) for names in GROUPS.values()})
    return {
        "baseline": baseline,
        "groups": {name: drop(zeroed(names)) for name, names in GROUPS.items()},
        "random": {
            size: np.array(
                [
                    drop(zeroed(rng.choice(ch_names, size, replace=False)))
                    for _ in range(n_random)
                ]
            )
            for size in sizes
        },
        "bands": {
            "remove 0.5-4 Hz": drop(filtered("highpass", 4.0)),
            "keep 8-30 Hz only": drop(filtered("bandpass", [8.0, 30.0])),
        },
    }


occlusion = occlusion_check(model, X_test, y_test, ch_names, sfreq)
print(f"Baseline accuracy: {occlusion['baseline']:.1%}")
for name, drop in occlusion["groups"].items():
    random_95 = np.percentile(occlusion["random"][len(GROUPS[name])], 95)
    print(f"  zero {name}: drop {drop:+.1%} (random groups, 95th pct {random_95:+.1%})")
for name, drop in occlusion["bands"].items():
    print(f"  {name}: drop {drop:+.1%}")

######################################################################
# The chart puts the data check next to what the weights said (in the
# title). The dashed line is the 95th percentile of the random groups of
# the eye group's size, the level a named group has to clear before its
# drop means anything.

eye_random = occlusion["random"][len(EYE_ADJACENT)]
bars = {
    f"eye-adjacent\n({len(EYE_ADJACENT)} ch)": occlusion["groups"]["eye-adjacent"],
    f"motor\n({len(MOTOR)} ch)": occlusion["groups"]["motor"],
    f"random\n({len(EYE_ADJACENT)} ch, mean)": eye_random.mean(),
    "remove\n0.5-4 Hz": occlusion["bands"]["remove 0.5-4 Hz"],
    "keep\n8-30 Hz only": occlusion["bands"]["keep 8-30 Hz only"],
}
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.bar(
    list(bars),
    100 * np.array(list(bars.values())),
    color=["#D55E00", "#0072B2", "#999999", "#D55E00", "#0072B2"],
)
ax.axhline(
    100 * np.percentile(eye_random, 95),
    ls="--",
    color="#444",
    lw=1,
    label="random groups, 95th percentile",
)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Accuracy drop (points)")
ax.set_title(
    f"Held-out subjects {test_subjects}: baseline {occlusion['baseline']:.0%}; "
    f"weights say {reading['band']}, {reading['window']}",
    fontsize=9,
)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()

######################################################################
# Subtract the initialisation before reading
# ------------------------------------------
#
# A trained EEGNet keeps much of its random initialisation. The spatial
# filters move a short distance from where they started, so part of
# what the weight statistic sees is the starting noise, not what
# training put there. Rebuilding the initial network from the same seed
# shows how much: compare the trained filters with the seed's
# initialisation and with another seed's.

init_model = make_eegnet(seed)
other_init = make_eegnet(seed + 1)
for name in ["conv_spatial", "conv_temporal"]:
    trained, own, other = (
        m.get_submodule(name).weight.detach().flatten()
        for m in (model, init_model, other_init)
    )
    cos_own = torch.nn.functional.cosine_similarity(trained, own, dim=0)
    cos_other = torch.nn.functional.cosine_similarity(trained, other, dim=0)
    print(
        f"{name}: cosine to own init {cos_own:.2f}, to another seed's {cos_other:.2f}"
    )

######################################################################
# Then read the training difference. Only the spatial filters are
# replaced by trained minus initial weights; the gains, temporal
# kernels and classifier stay as trained, so the importance weighting
# is unchanged. Channels that only rank high because of the starting
# noise drop out of the list.
#
# On a bench of eight EEGNets with planted cues, one channel sat in the
# top three of the plain reading for all eight models, seven of which
# never had a cue on it; in the difference reading it stayed only in
# the model where it was planted [6]_. On real decoders the difference
# reading agreed with occlusion about as well as the plain one, so
# treat it as a second reading to compare against, not a replacement.

spatial_change = model.conv_spatial.weight - init_model.conv_spatial.weight
reading_delta = read_weights(
    model, ch_names, sfreq, window_start_s, spatial_weight=spatial_change
)
print_channels(reading_delta, "Training difference")

######################################################################
# What to do with the result
# --------------------------
#
# Record it with the model. A model card for a released decoder should
# carry a *relies on* field that states the channels, band and window
# the audit found, the method (weight statistic, occlusion on which
# subjects), the date, whether the data check agreed with the weight
# reading, and the initialisation seed, which costs nothing to release
# and is what makes the difference reading possible. A decoder whose
# entry says "F7/F8/AF7/AF8, below 4 Hz, 0-1 s after the cue" is a
# different product from one that says "C3/C4, 8-30 Hz, 1-4 s", even at
# the same accuracy.
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
# .. [6] Li, Y. (2026). *Reading the weights: a language model audits
#        and repairs EEG decoders without data.* Preprint.
#        DOI: 10.5281/zenodo.23934403
