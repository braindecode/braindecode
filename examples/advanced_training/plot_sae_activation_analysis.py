""".. _sae-activation-analysis:

Sparse autoencoders on the activations of a motor-imagery decoder
==================================================================

This tutorial trains a :class:`~braindecode.models.ShallowFBCSPNet` on one
subject of the BCI Competition IV 2a motor-imagery dataset (BNCI2014_001 via
MOABB), captures the activations of one of its layers, and fits a sparse
autoencoder (SAE) to those activations with
`SAE Lens <https://github.com/decoderesearch/SAELens>`_. Braindecode supplies
the data pipeline, the decoder and the activation capture/substitution hooks;
SAE Lens supplies the dictionary, the Top-K activation, its losses and the
training loop.

.. topic:: Why fit a sparse autoencoder to EEG-model activations?

    A trained decoder mixes many signals into every hidden unit, which makes
    individual units hard to read. A sparse autoencoder learns an
    *overcomplete dictionary*: each activation vector is rewritten as a
    combination of a small number of learned directions ("features"), so
    that individual features tend to be more selective than the raw units.
    Sparsity is a useful constraint, not a guarantee that these directions
    are independent physiological concepts, and two questions must be kept
    apart: *how well does the dictionary reconstruct the activations?* and
    *does the model still make the same predictions when it is fed the
    reconstruction?* The second question is the one that matters for using
    features to explain a decoder.

The recipe below is model-agnostic. We fit everything (decoder, scaling
statistics, dictionary, feature selection) on the *training session* and
evaluate once on the *test session* recorded on a different day. All numbers
come from a single subject and a short CPU training budget, so they describe
this run only and support no scientific claim about motor imagery.

Install the optional dependency with ``pip install 'braindecode[sae]'``
(or ``pip install sae-lens==6.51.3`` in a source checkout). SAE Lens brings a
substantial language-model dependency stack, but no language model or
pretrained SAE is downloaded here; dictionaries trained on LLMs are not EEG
dictionaries.

.. note::

   Documentation builds without SAE Lens render this source without executing
   it. Install the optional dependency to run the example and generate figures.

.. contents:: This example covers:
   :local:
   :depth: 2
"""

# Authors: Vandit Shah <shahvanditt@gmail.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
# License: BSD (3-clause)

import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import multiply

from braindecode import EEGClassifier
from braindecode.datasets import MOABBDataset
from braindecode.models import ShallowFBCSPNet
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    exponential_moving_standardize,
    preprocess,
)
from braindecode.util import set_random_seeds
from braindecode.visualization import (
    capture_activations,
    run_with_activation_substitution,
)

try:
    from sae_lens.config import LoggingConfig, SAETrainerConfig
    from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig
    from sae_lens.training.sae_trainer import SAETrainer
except ImportError as error:
    raise ImportError(
        "This optional tutorial requires: pip install sae-lens==6.51.3"
    ) from error

######################################################################
# Loading and preparing the data
# ------------------------------
#
# Loading
# ~~~~~~~
#
# We use subject 3 of BCI Competition IV 2a (BNCI2014_001), the same
# recording as :ref:`interpretability-tutorial`. It contains 22 EEG
# channels at 250 Hz and four motor-imagery classes (feet, left hand,
# right hand, tongue), with two sessions of 288 trials recorded on
# different days.

subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])

######################################################################
# Preprocessing
# ~~~~~~~~~~~~~
#
# The standard trial-wise BCI IV 2a preprocessing: keep EEG channels,
# convert V to µV, band-pass 4–38 Hz to retain the mu and beta bands where
# motor-imagery effects live, and apply exponential moving standardization
# to remove slow per-channel drifts.

low_cut_hz, high_cut_hz = 4.0, 38.0
factor_new, init_block_size = 1e-3, 1000

preprocess(
    dataset,
    [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        Preprocessor(lambda data: multiply(data, 1e6)),
        Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),
        Preprocessor(
            exponential_moving_standardize,
            factor_new=factor_new,
            init_block_size=init_block_size,
        ),
    ],
    n_jobs=-1,
)

######################################################################
# Windowing and split
# ~~~~~~~~~~~~~~~~~~~
#
# Windows run from 0.5 s before the cue to the end of the trial. We split
# by session: ``"0train"`` is used for every fitting step below, and
# ``"1test"`` (a different day) is touched only once, at evaluation time.
# Splitting *before* any statistic is computed is what makes the held-out
# numbers meaningful; overlapping windows from one recording must never
# straddle the split.

sfreq = dataset.datasets[0].raw.info["sfreq"]
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=int(-0.5 * sfreq),
    trial_stop_offset_samples=0,
    preload=True,
)
split_by_session = windows_dataset.split("session")
train_set, test_set = split_by_session["0train"], split_by_session["1test"]

# BCI IV 2a labels in MOABB's alphabetical order; the position is the class id.
LABELS = ("feet", "left_hand", "right_hand", "tongue")

# Dense tensors for the hooks below (288 windows × 22 channels × 1125 samples).
x_train = torch.as_tensor(np.stack([x for x, *_ in train_set]), dtype=torch.float32)
y_train = torch.as_tensor([y for _, y, _ in train_set])
x_test = torch.as_tensor(np.stack([x for x, *_ in test_set]), dtype=torch.float32)
y_test = torch.as_tensor([y for _, y, _ in test_set])
print(f"Train windows: {tuple(x_train.shape)}; test windows: {tuple(x_test.shape)}")

######################################################################
# Training the decoder
# --------------------
#
# We train a :class:`~braindecode.models.ShallowFBCSPNet` for 30 epochs on
# the training session with the settings of :ref:`bcic-iv-2a-moabb-trial`.
# ``train_split=None`` keeps the test session out of training entirely; on
# a laptop CPU this takes a few tens of seconds. The SAE analysis below applies to
# any trained network, so a stronger decoder would only make the features
# more interesting to look at.

device = "cpu"
set_random_seeds(seed=20240205, cuda=False)

model = ShallowFBCSPNet(
    n_chans=x_train.shape[1],
    n_outputs=len(LABELS),
    n_times=x_train.shape[2],
    final_conv_length="auto",
)
classifier = EEGClassifier(
    model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    optimizer__lr=6.25e-4,
    optimizer__weight_decay=0,
    batch_size=64,
    max_epochs=30,
    train_split=None,
    callbacks=["accuracy"],
    classes=list(range(len(LABELS))),
    device=device,
    verbose=0,
)
classifier.fit(train_set, y=None)

model = classifier.module_.eval()
model.requires_grad_(False)
with torch.no_grad():
    baseline_logits = model(x_test)
baseline_pred = baseline_logits.argmax(1)
baseline_accuracy = (baseline_pred == y_test).float().mean().item()
print(f"Test-session accuracy of the trained decoder: {baseline_accuracy:.1%}")

######################################################################
# Capturing activations from one layer
# ------------------------------------
#
# ShallowFBCSPNet is temporal convolution → spatial filter → square →
# average pool → log. We hook the output of the log, ``pool_nonlin_exp``,
# so that each activation vector is the log band-power of the 40 learned
# filters in one pooling window: shape ``(windows, 40, 69, 1)``. Every
# ``(window, time bin)`` pair is one SAE observation, so we move the
# feature axis last and flatten the rest. The capture is detached; the SAE
# never updates the decoder.
#
# Per-coordinate standardization (mean and standard deviation fitted on
# *training rows only*) puts filters with very different power scales on a
# comparable footing. Because we do this ourselves, SAE Lens's internal
# ``normalize_activations`` is switched off.

layer = model.pool_nonlin_exp


def to_rows(activation):
    """(windows, features, time, 1) -> (windows * time, features)."""
    return activation.movedim(1, -1).reshape(-1, activation.shape[1])


with torch.no_grad():
    train_acts = capture_activations(model, x_train, layer).detach()
train_rows = to_rows(train_acts)
mean = train_rows.mean(0)
std = train_rows.std(0, correction=0).clamp_min(1e-6)
train_scaled = (train_rows - mean) / std
d_in = train_rows.shape[1]
n_time_bins = train_acts.shape[2]
print(f"Layer output: {tuple(train_acts.shape)}; training rows: {len(train_rows)}")

######################################################################
# Fitting the Top-K SAE with SAE Lens
# -----------------------------------
#
# SAE Lens's low-level :class:`~sae_lens.training.sae_trainer.SAETrainer`
# accepts an iterator of tensor batches, so no tokenizer or language-model
# runner is involved. We ask for a dictionary four times wider than the
# layer (160 features) with at most ``k=8`` active features per
# observation, and train for 2000 updates on batches sampled with
# replacement. The library owns the optimizer, learning-rate schedule
# (``lr_end`` is set explicitly), the norm-adjusted Top-K activation and
# the auxiliary loss that revives inactive features. A copy taken *before*
# training gives an untrained dictionary with the same initialization,
# which we use as a control later.

batch_size = 128
n_updates = 2000
expansion_factor = 4
k_active = 8

torch.manual_seed(20240205)


def activation_batches():
    while True:
        indices = torch.randint(len(train_scaled), (batch_size,), device=device)
        yield train_scaled[indices]


sae = TopKTrainingSAE(
    TopKTrainingSAEConfig(
        d_in=d_in,
        d_sae=expansion_factor * d_in,
        k=k_active,
        device=device,
        dtype="float32",
        normalize_activations="none",
        reshape_activations="none",
        apply_b_dec_to_input=True,
        rescale_acts_by_decoder_norm=True,
    ),
    use_error_term=False,
)
untrained_sae = copy.deepcopy(sae).eval()
trainer = SAETrainer(
    cfg=SAETrainerConfig(
        total_training_samples=n_updates * batch_size,
        train_batch_size_samples=batch_size,
        device=device,
        autocast=False,
        lr=1e-3,
        lr_end=1e-4,
        logger=LoggingConfig(log_to_wandb=False),
        n_checkpoints=0,
        save_final_checkpoint=False,
    ),
    sae=sae,
    data_provider=activation_batches(),
)
trainer.fit()
sae.eval()

######################################################################
# Reconstruction quality on the test session
# ------------------------------------------
#
# The *fraction of variance unexplained* (FVU) compares the reconstruction
# error with the variance of the activations around the training mean; 0 is
# perfect, 1 is no better than predicting the mean. We report it in the
# standardized coordinates the SAE was trained in and in the original
# log-power units. The mean number of active features is at most ``k``
# because Top-K rectifies the selected values.


def fvu(target, reconstruction):
    residual = (target - reconstruction).square().sum()
    total = (target - target.mean(0)).square().sum()
    return (residual / total).item()


with torch.no_grad():
    test_acts = capture_activations(model, x_test, layer).detach()
    test_rows = to_rows(test_acts)
    test_scaled = (test_rows - mean) / std
    test_codes = sae.encode(test_scaled)
    test_reconstructed = sae.decode(test_codes)
    untrained_reconstructed = untrained_sae.decode(untrained_sae.encode(test_scaled))
fvu_scaled = fvu(test_scaled, test_reconstructed)
fvu_original = fvu(test_rows, test_reconstructed * std + mean)
fvu_untrained = fvu(test_scaled, untrained_reconstructed)
active_per_row = (test_codes != 0).sum(-1).float().mean().item()
never_active = ((test_codes != 0).sum(0) == 0).float().mean().item()
print(f"Test FVU (standardized): {fvu_scaled:.3f}")
print(f"Test FVU (original log-power units): {fvu_original:.3f}")
print(f"Test FVU of the untrained dictionary: {fvu_untrained:.3f}")
print(f"Mean active features per observation: {active_per_row:.2f} (k = {k_active})")
print(f"Features never active on the test session: {never_active:.1%}")

######################################################################
# Does the decoder still agree with itself?
# -----------------------------------------
#
# Reconstruction error is measured in activation space, but we care about
# the decoder's *decisions*. :func:`~braindecode.visualization.run_with_activation_substitution`
# replaces the layer output during a forward pass with whatever the callable
# returns. The callable receives the original activation tensor: it applies
# the training-set scaling, encodes and decodes with the SAE, undoes the
# scaling, and restores the original axes and dtype.
#
# We report test accuracy and *agreement* (the fraction of test windows whose
# predicted class is unchanged) for three substitutions: the trained SAE,
# the untrained SAE with the same initialization, and replacing the whole
# layer with its training mean. If the trained SAE did not beat both
# controls, its reconstruction would not be carrying the decision-relevant
# information.


def substitute_with(dictionary):
    def substitute(output):
        flat = output.movedim(1, -1)
        rows = flat.reshape(-1, d_in).to(mean)
        codes = dictionary.encode((rows - mean) / std)
        restored = dictionary.decode(codes) * std + mean
        return restored.reshape(flat.shape).movedim(-1, 1).to(output)

    return substitute


def mean_ablation(output):
    flat = output.movedim(1, -1)
    return mean.expand(flat.shape).movedim(-1, 1).to(output)


interventions = {
    "trained SAE": substitute_with(sae),
    "untrained SAE": substitute_with(untrained_sae),
    "layer := train mean": mean_ablation,
}
results = {}
with torch.no_grad():
    for name, substitute in interventions.items():
        logits = run_with_activation_substitution(model, x_test, layer, substitute)
        pred = logits.argmax(1)
        results[name] = {
            "accuracy": (pred == y_test).float().mean().item(),
            "agreement": (pred == baseline_pred).float().mean().item(),
        }
print(f"{'intervention':>22s}  accuracy  agreement")
print(f"{'none (native model)':>22s}  {baseline_accuracy:8.1%}  {1.0:9.1%}")
for name, scores in results.items():
    print(f"{name:>22s}  {scores['accuracy']:8.1%}  {scores['agreement']:9.1%}")

######################################################################
# Which features fire for which class?
# ------------------------------------
#
# Codes are non-negative, so a feature's mean code per class summarizes
# how strongly it responds to each condition. We average codes over the
# time bins of each window, giving one ``(d_sae,)`` vector per trial, and
# compute a *selectivity index* per feature and class on the **training**
# session: the difference between the class mean and the mean over the
# other classes, divided by the pooled standard deviation. The two most
# selective features per class are chosen there; only then do we look at
# how those features behave on the test session. Choosing features on the
# evaluation data would make any apparent selectivity circular.


def per_window_codes(codes):
    return codes.reshape(-1, n_time_bins, codes.shape[-1]).mean(1)


with torch.no_grad():
    train_codes = sae.encode(train_scaled)
train_window_codes = per_window_codes(train_codes)
test_window_codes = per_window_codes(test_codes)

n_features = train_window_codes.shape[1]
selectivity = torch.zeros(len(LABELS), n_features)
for class_id in range(len(LABELS)):
    in_class = train_window_codes[y_train == class_id]
    out_class = train_window_codes[y_train != class_id]
    pooled_std = torch.cat([in_class, out_class]).std(0).clamp_min(1e-6)
    selectivity[class_id] = (in_class.mean(0) - out_class.mean(0)) / pooled_std

n_per_class = 2
selected = {
    label: selectivity[class_id].topk(n_per_class).indices.tolist()
    for class_id, label in enumerate(LABELS)
}
print("Most class-selective features (chosen on the training session):")
for label, features in selected.items():
    print(f"  {label:>10s}: {features}")

######################################################################
# The heatmap shows the mean test-session code of each selected feature
# per class, normalized by the feature's largest class mean so that all
# rows share one color scale. A feature that was selected for a class on
# the training session and still peaks on that class on the test session
# has a preference that survived a change of day. Diagonal blocks are the
# hoped-for pattern; off-diagonal maxima are honest failures, not noise to
# be explained away.

selected_features = [f for features in selected.values() for f in features]
class_means = torch.stack(
    [test_window_codes[y_test == class_id].mean(0) for class_id in range(len(LABELS))]
)  # (n_classes, d_sae)
heat = class_means[:, selected_features].T
heat = heat / heat.max(1, keepdim=True).values.clamp_min(1e-6)

fig, ax = plt.subplots(figsize=(5.5, 4.2))
image = ax.imshow(heat.numpy(), cmap="magma", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(LABELS)))
ax.set_xticklabels([label.replace("_", " ") for label in LABELS])
ax.set_yticks(range(len(selected_features)))
ax.set_yticklabels(
    [
        f"#{feature} (sel. {label.replace('_', ' ')})"
        for label, features in selected.items()
        for feature in features
    ],
    fontsize=8,
)
ax.set_xlabel("Test-session class")
ax.set_title("Mean SAE code per class (row-normalized)", fontsize=10)
fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03, label="relative activation")
fig.tight_layout()

######################################################################
# Because each activation vector is one pooling window, the codes also
# have a time course within the trial. Below we plot, for the top feature
# selected for left-hand and for right-hand imagery, the mean test code
# over time for every class. Each time bin is placed at the centre of the
# input span it summarizes (temporal filter plus pooling window); the cue
# appears 0.5 s after window onset. A feature whose class preference only
# emerges after the cue is consistent with task-related activity, whereas
# a preference that is already present before the cue, or that is flat,
# deserves suspicion.

bin_centre_samples = (
    model.pool_time_stride * np.arange(n_time_bins)
    + (model.pool_time_length - 1) / 2
    + (model.filter_time_length - 1) / 2
)
time_axis = bin_centre_samples / sfreq - 0.5
test_codes_time = test_codes.reshape(-1, n_time_bins, n_features)
LABEL_COLORS = ("#0072B2", "#009E73", "#D55E00", "#CC79A7")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, label in zip(axes, ("left_hand", "right_hand")):
    feature = selected[label][0]
    for class_id, class_label in enumerate(LABELS):
        trace = test_codes_time[y_test == class_id, :, feature].mean(0)
        ax.plot(
            time_axis,
            trace.numpy(),
            color=LABEL_COLORS[class_id],
            lw=1.8,
            label=class_label.replace("_", " "),
        )
    ax.axvline(0, color="#888", ls=":", lw=1)
    ax.set_title(f"Feature #{feature}, selected for {label.replace('_', ' ')}")
    ax.set_xlabel("Time from cue (s)")
axes[0].set_ylabel("Mean SAE code (test session)")
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()

######################################################################
# Interpretation and next steps
# -----------------------------
#
# * **Read the agreement table first.** A dictionary can have a low FVU and
#   still flip decisions if the small residual is exactly what the classifier
#   uses. Conversely, a controlled drop in accuracy is informative when the
#   untrained-dictionary and mean-ablation controls drop much further.
# * **Selectivity is a statement about this decoder**, not about the brain.
#   A feature that prefers left-hand trials reflects what the trained
#   ShallowFBCSPNet computes on this subject. Its physiological meaning
#   requires further work: relate the decoder direction to spatial and
#   spectral patterns, compare several seeds and subjects, and check
#   associations with artifacts and session before drawing conclusions.
# * **Budget and validation.** The layer, expansion factor, ``k`` and the
#   number of updates were fixed in advance for a bounded gallery run. In a
#   study, select them on a validation split of the training session (for
#   instance by run) and evaluate the test session only once.
# * **Saving a dictionary.** Store the SAE Lens configuration and weights
#   together with the ``mean``/``std`` tensors, the decoder checkpoint, the
#   layer contract and the split, then verify that the reloaded pipeline
#   reproduces the substitution results before reusing it.
#
# References
# ----------
#
# * `SAE Lens documentation <https://jbloomaus.github.io/SAELens/>`_ describes
#   the external training and inference APIs used here (tested with 6.51.3).
# * Makhzani and Frey, `k-Sparse Autoencoders
#   <https://arxiv.org/abs/1312.5663>`_, motivate sparse dictionary learning.
# * Schirrmeister et al. (2017), *Deep learning with convolutional neural
#   networks for EEG decoding and visualization*, Human Brain Mapping,
#   38(11), 5391–5420, introduce ShallowFBCSPNet.
