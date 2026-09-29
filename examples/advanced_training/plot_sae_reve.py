""".. _sae-reve-interventions:

Sparse feature interventions in a pretrained REVE
=================================================

This tutorial takes the pretrained :class:`~braindecode.models.REVE` EEG
foundation model, keeps its encoder frozen and trains only its linear
classification head on one subject of the BCI Competition IV 2a motor-imagery
dataset (BNCI2014_001 via MOABB). It then fits a Top-K sparse autoencoder
(SAE) to the token embeddings of one transformer block with
`SAE Lens <https://github.com/decoderesearch/SAELens>`_, and *intervenes* on
the learned features while the model classifies a session recorded on
another day. Braindecode supplies the data pipeline and the pretrained model;
SAE Lens supplies the dictionary, its Top-K activation, losses and training
loop.

.. topic:: What is different about REVE's tokens?

    REVE cuts every channel into 1-s patches and turns each (channel, patch)
    pair into one token. A 4D positional encoding, computed from the
    electrode's 3D coordinates and the patch index, tells the transformer
    where and when each token was recorded. There is no classification
    token: the default head flattens the tokens of the last block and applies
    a linear layer, so every token of every block can reach the decision.
    An SAE fitted to these tokens learns features that come with a *scalp
    location* and a *time*, which we can draw as topographic maps. We then
    ask a causal question: if we remove the features that were most
    selective for a class, does the model become less confident in that
    class, and more so than when removing random features or the features
    selected for another class?

We use three splits. The first four runs of the training session (*fit*)
train the head and the dictionary. Its last two runs (*validation*) were used
to choose the model setup, the block and the dictionary size. The test
session, recorded on a different day, is used only for the final evaluation.
All numbers come from a single subject, a frozen encoder and a short SAE run.
They describe this run, not REVE or motor imagery in general, and support no
scientific claim.

Install the optional dependency with ``pip install 'braindecode[sae]'`` (or
``pip install sae-lens==6.51.3`` in a source checkout). SAE Lens brings a
language-model dependency stack, but no language model or pretrained SAE is
used here. The REVE-base weights (``brain-bzh/reve-base``, about 277 MB) are
downloaded from the Hugging Face Hub the first time the example runs. They are
distributed under the REVE Responsible Use License, which you accept by
downloading them. REVE's electrode position bank is a small file that is
cached in the MNE data directory.

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
import mne
import numpy as np
import torch
from torch.nn import functional as F

from braindecode.datasets import MOABBDataset
from braindecode.models import REVE
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)
from braindecode.util import set_random_seeds

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
# We use subject 3 of BCI Competition IV 2a (BNCI2014_001), the recording
# used in :ref:`finetune-foundation-model`: 22 EEG channels at 250 Hz, four
# motor-imagery classes (feet, left hand, right hand, tongue) and two
# sessions of 288 trials, each made of six runs, recorded on different days.

subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])

######################################################################
# Preprocessing to match the pretraining
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# REVE was pretrained on EEG band-passed between 0.5 and 99.5 Hz, sampled at
# 200 Hz, z-scored per channel over each recording and clipped at 15 standard
# deviations. We apply the same steps to every run. The z-score uses only the
# run's own signal, and no label.


def standardize_and_clip(data):
    """Z-score each channel of one recording and clip at 15 SD."""
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    return np.clip((data - mean) / std, -15.0, 15.0)


preprocess(
    dataset,
    [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        Preprocessor("filter", l_freq=0.5, h_freq=99.5),
        Preprocessor("resample", sfreq=200),
        Preprocessor(standardize_and_clip),
    ],
)

######################################################################
# Windowing and split
# ~~~~~~~~~~~~~~~~~~~
#
# Each window covers the 4-s motor-imagery period that MOABB defines after
# the cue (800 samples at 200 Hz). The first four runs of session
# ``"0train"`` form the fit split, its last two runs the validation split,
# and session ``"1test"`` the test split.

raw_info = dataset.datasets[0].raw.info
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    preload=True,
)
description = windows_dataset.description
in_training_session = description["session"] == "0train"
in_last_two_runs = description["run"].isin(["4", "5"])
splits = windows_dataset.split(
    {
        "fit": description.index[in_training_session & ~in_last_two_runs].tolist(),
        "validation": description.index[
            in_training_session & in_last_two_runs
        ].tolist(),
        "test": description.index[~in_training_session].tolist(),
    }
)

# BCI IV 2a labels in MOABB's alphabetical order; the position is the class id.
LABELS = ("feet", "left_hand", "right_hand", "tongue")


def to_tensors(windows):
    x = torch.as_tensor(np.stack([x for x, *_ in windows]), dtype=torch.float32)
    y = torch.as_tensor([y for _, y, _ in windows])
    return x, y


x_fit, y_fit = to_tensors(splits["fit"])
x_val, y_val = to_tensors(splits["validation"])
x_test, y_test = to_tensors(splits["test"])
n_chans, n_times = x_fit.shape[1:]
print(
    f"Windows: fit {len(x_fit)}, validation {len(x_val)}, test {len(x_test)}; "
    f"each {n_chans} channels x {n_times} samples"
)

######################################################################
# Euclidean alignment
# ~~~~~~~~~~~~~~~~~~~
#
# The spatial covariance of the EEG drifts between sessions. As in the REVE
# evaluation on this dataset, we apply Euclidean alignment (He and Wu, 2020):
# the windows of each session are whitened with that session's mean spatial
# covariance, so that their average covariance becomes the identity. This
# uses the windows of the session but none of their labels. For the test
# session it is transductive: the test windows are aligned with statistics
# computed on all of them.


def euclidean_alignment(x):
    """Whiten windows (windows, channels, times) with their mean covariance."""
    covariance = torch.einsum("nct,ndt->cd", x, x) / (x.shape[0] * x.shape[2])
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    whitening = eigenvectors @ torch.diag(eigenvalues.rsqrt()) @ eigenvectors.T
    return whitening @ x


training_session = euclidean_alignment(torch.cat([x_fit, x_val]))
x_fit, x_val = training_session[: len(x_fit)], training_session[len(x_fit) :]
x_test = euclidean_alignment(x_test)

######################################################################
# The pretrained REVE with a frozen encoder
# -----------------------------------------
#
# Loading the weights
# ~~~~~~~~~~~~~~~~~~~
#
# :ref:`load-pretrained-models` shows the general ``from_pretrained``
# pattern. REVE looks up the 3D position of every electrode by name in its
# position bank, so we pass the channel names through ``chs_info``. A 4-s
# window at 200 Hz gives four overlapping 1-s patches per channel, i.e.
# 22 × 4 = 88 tokens of dimension 512. The checkpoint holds the encoder;
# the classification head (``final_layer``) is newly initialized. We freeze
# every weight, so the encoder is used exactly as pretrained.

set_random_seeds(seed=20240205, cuda=False)
model = REVE.from_pretrained(
    "brain-bzh/reve-base",
    n_outputs=len(LABELS),
    n_chans=n_chans,
    n_times=n_times,
    sfreq=200,
    chs_info=[{"ch_name": name} for name in raw_info["ch_names"]],
).eval()
model.requires_grad_(False)
n_blocks = len(model.transformer.layers)
n_params = sum(parameter.numel() for parameter in model.parameters())
print(f"REVE-base: {n_params / 1e6:.1f} M parameters, {n_blocks} blocks")

######################################################################
# Token streams
# ~~~~~~~~~~~~~
#
# :func:`~braindecode.visualization.capture_activations` and
# :func:`~braindecode.visualization.run_with_activation_substitution` read and
# replace the *output of a submodule*. In REVE, each block adds the outputs
# of its attention and feed-forward layers to the token stream inside the
# encoder's loop, so no submodule returns the stream after a block. Instead,
# ``model(x, return_output=True)`` returns the stream entering the first
# block and the stream after each of the 22 blocks, and ``run_blocks`` below
# continues a forward pass from any block by reusing the pretrained layers in
# the same loop. Starting from a block's output also avoids recomputing the
# blocks before it for every edit.


def token_streams(x, keep, batch_size=32):
    """Map stream index to tokens; 0 enters block 1 and ``b`` leaves block ``b``."""
    kept = {index: [] for index in keep}
    with torch.no_grad():
        for batch in x.split(batch_size):
            streams = model(batch, return_output=True)
            for index in keep:
                kept[index].append(streams[index])
    return {index: torch.cat(parts) for index, parts in kept.items()}


def run_blocks(tokens, first):
    """Apply blocks ``first`` to 22 (counted from 1), then the head."""
    for attn, ff in model.transformer.layers[first - 1 :]:
        tokens = attn(tokens) + tokens
        tokens = ff(tokens) + tokens
    return model.final_layer(tokens)


######################################################################
# We keep the output of block 18, where the dictionary will be fitted (the
# choice is explained below), and of the last block, which the head reads.

block = 18
fit_tokens = token_streams(x_fit, keep=(block, n_blocks))
val_tokens = token_streams(x_val, keep=(n_blocks,))
test_tokens = token_streams(x_test, keep=(block, n_blocks))

######################################################################
# Training the linear head
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# Because the encoder is frozen, the last block's tokens are computed once
# and only ``final_layer`` (a layer normalization of the flattened tokens and
# a linear layer) is trained, on the fit windows with full-batch AdamW. This
# takes a few seconds on a CPU. We then check that continuing the forward
# pass from block 18 reproduces the model's own output.


def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


head = model.final_layer
head.requires_grad_(True)
optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.1)
torch.manual_seed(0)
for _ in range(300):
    optimizer.zero_grad()
    F.cross_entropy(head(fit_tokens[n_blocks]), y_fit).backward()
    optimizer.step()
head.requires_grad_(False)

with torch.no_grad():
    fit_accuracy = accuracy(head(fit_tokens[n_blocks]), y_fit)
    val_logits = head(val_tokens[n_blocks])
    baseline_logits = head(test_tokens[n_blocks])
    difference = run_blocks(test_tokens[block][:8], block + 1) - model(x_test[:8])
baseline_pred = baseline_logits.argmax(1)
baseline_accuracy = accuracy(baseline_logits, y_test)
print(f"Largest difference to model(x): {difference.abs().max():.1e}")
print(
    f"Accuracy (chance 25%): fit {fit_accuracy:.1%}, "
    f"validation {accuracy(val_logits, y_val):.1%}, test {baseline_accuracy:.1%}"
)

######################################################################
# Which blocks influence the prediction?
# --------------------------------------
#
# We skip one block at a time, i.e. pass its input unchanged to the next
# block, and count how many validation predictions stay the same. Only the
# validation windows are used, because this check informs a modelling choice.


def skip_block_predictions(x, batch_size=32):
    """Predictions when each block in turn is skipped."""
    predictions = {b: [] for b in range(1, n_blocks + 1)}
    with torch.no_grad():
        for batch in x.split(batch_size):
            streams = model(batch, return_output=True)
            for b in predictions:
                # streams[b - 1] enters block b; skipping b feeds it to b + 1.
                predictions[b].append(run_blocks(streams[b - 1], b + 1).argmax(1))
    return {b: torch.cat(parts) for b, parts in predictions.items()}


val_pred = val_logits.argmax(1)
skipped = skip_block_predictions(x_val)
unchanged = np.array(
    [(pred == val_pred).float().mean().item() for pred in skipped.values()]
)
for start in range(0, n_blocks, 11):
    print(
        "Unchanged when skipped: "
        + ", ".join(
            f"{b}: {unchanged[b - 1]:.0%}" for b in range(start + 1, start + 12)
        )
    )

fig, ax = plt.subplots(figsize=(7.5, 2.8))
ax.bar(
    np.arange(1, n_blocks + 1),
    unchanged,
    color=["#D55E00" if b == block else "#999999" for b in range(1, n_blocks + 1)],
)
ax.set_xticks(np.arange(1, n_blocks + 1))
ax.set_xlabel("Skipped block (orange: block used for the SAE)")
ax.set_ylabel("Unchanged predictions")
ax.set_ylim(0, 1)
ax.set_title("Validation runs: effect of skipping one REVE block", fontsize=10)
fig.tight_layout()

######################################################################
# Every block matters to some extent: unlike a head that reads a single
# classification token, REVE's flattening head sees every token of the last
# block, and each block rewrites all tokens. When we ran this example,
# skipping the first block changed about three quarters of the validation
# decisions and skipping block 9 about two thirds, while each of the last
# eight blocks still changed 12–22% of them.
#
# How the settings were chosen
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The choices below were made on the validation runs only, before the test
# session was used, in runs that are not part of this example:
#
# * **Model setup.** With Euclidean alignment, the frozen encoder with this
#   flattening head reached 52–59% validation accuracy across the training
#   lengths and initializations we tried, against 45–49% for the
#   attention-pooling head (``attention_pooling=True``); without alignment,
#   31–38% and 27–32%. Also fine-tuning the last two blocks, tried before we
#   added the alignment, reached at most 31% and costs much more on a CPU.
# * **Block.** We fitted SAEs with 2048 features and ``k=32`` to the outputs
#   of blocks 6, 10, 14 and 18, i.e. blocks followed by at least four others,
#   so that an edit has to pass through attention layers before reaching the
#   head. Feeding their reconstructions back into the model kept 53%, 44%,
#   55% and 73% of the validation decisions, so we use block 18.
# * **Dictionary size.** At block 18, 2048 features with ``k=16``, 2048 with
#   ``k=32`` and 4096 with ``k=32`` kept 69%, 73% and 75% of the validation
#   decisions. We use 2048 features and ``k=32``: 4096 features agreed on
#   only two more of the 96 validation windows, while doubling the training
#   time and leaving 182 features inactive on those windows.
#
# The number of features removed per class (32) and of random control sets
# (20) were fixed in advance.
#
# Fitting an SAE to the block-18 tokens
# -------------------------------------
#
# Capturing token embeddings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Every token of block 18 is one training row. The per-coordinate mean and
# standard deviation come from the fit tokens only, so SAE Lens's own
# ``normalize_activations`` is switched off.

d_in = fit_tokens[block].shape[-1]
n_tokens = fit_tokens[block].shape[1]
n_patches = n_tokens // n_chans
fit_rows = fit_tokens[block].reshape(-1, d_in)
mean = fit_rows.mean(0)
std = fit_rows.std(0, correction=0).clamp_min(1e-6)
fit_scaled = (fit_rows - mean) / std
print(f"Fit tokens: {tuple(fit_rows.shape)} ({n_tokens} per window)")


def standardize(tokens):
    """(windows, tokens, dim) -> standardized (windows * tokens, dim)."""
    return (tokens.reshape(-1, d_in) - mean) / std


######################################################################
# Training the Top-K SAE
# ~~~~~~~~~~~~~~~~~~~~~~
#
# SAE Lens's :class:`~sae_lens.training.sae_trainer.SAETrainer` consumes an
# iterator of tensor batches. We learn 2048 features (four times the token
# width) with at most ``k=32`` active per token, for 3000 updates on batches
# of 256 tokens sampled with replacement. A copy made before training serves
# as an untrained control with the same initialization.

batch_size = 256
n_updates = 3000
k_active = 32

torch.manual_seed(20240205)


def activation_batches():
    while True:
        yield fit_scaled[torch.randint(len(fit_scaled), (batch_size,))]


sae = TopKTrainingSAE(
    TopKTrainingSAEConfig(
        d_in=d_in,
        d_sae=4 * d_in,
        k=k_active,
        device="cpu",
        dtype="float32",
        normalize_activations="none",
        reshape_activations="none",
        apply_b_dec_to_input=True,
        rescale_acts_by_decoder_norm=True,
    ),
    use_error_term=False,
)
untrained_sae = copy.deepcopy(sae).eval()
SAETrainer(
    cfg=SAETrainerConfig(
        total_training_samples=n_updates * batch_size,
        train_batch_size_samples=batch_size,
        device="cpu",
        autocast=False,
        lr=1e-3,
        lr_end=1e-4,
        logger=LoggingConfig(log_to_wandb=False),
        n_checkpoints=0,
        save_final_checkpoint=False,
    ),
    sae=sae,
    data_provider=activation_batches(),
).fit()
_ = sae.eval()

######################################################################
# Reconstruction on the test session
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The fraction of variance unexplained (FVU) is 0 for a perfect
# reconstruction and 1 for predicting the mean. We compute it on the test
# tokens in the standardized coordinates, for the trained and the untrained
# dictionary.


def fvu(target, reconstruction):
    residual = (target - reconstruction).square().sum()
    return (residual / (target - target.mean(0)).square().sum()).item()


with torch.no_grad():
    test_scaled = standardize(test_tokens[block])
    test_codes = sae.encode(test_scaled)
    fvu_trained = fvu(test_scaled, sae.decode(test_codes))
    fvu_untrained = fvu(
        test_scaled, untrained_sae.decode(untrained_sae.encode(test_scaled))
    )
    fit_codes = sae.encode(fit_scaled)
active_on_fit = (fit_codes != 0).any(0)
print(f"Test FVU: trained {fvu_trained:.3f}, untrained {fvu_untrained:.3f}")
print(f"Mean active features per token: {(test_codes != 0).sum(-1).float().mean():.1f}")
print(f"Features never active on the fit windows: {(~active_on_fit).sum()}")

######################################################################
# Substituting the reconstruction into the model
# ----------------------------------------------
#
# We replace the output of block 18 by the dictionary's reconstruction, undo
# the standardization and let blocks 19–22 and the head finish the forward
# pass. We compare the trained SAE with the untrained one and with setting
# every token to its fit mean. Agreement is the fraction of test predictions
# that match those of the unedited model.


def reconstruct_with(dictionary):
    def substitute(tokens):
        rows = dictionary.decode(dictionary.encode(standardize(tokens)))
        return (rows * std + mean).reshape(tokens.shape)

    return substitute


def edited_logits(substitute):
    with torch.no_grad():
        return run_blocks(substitute(test_tokens[block]), block + 1)


print(f"{'substitution':>22s}  accuracy  agreement")
print(f"{'none (native model)':>22s}  {baseline_accuracy:8.1%}  {1.0:9.1%}")
for name, substitute in {
    "trained SAE": reconstruct_with(sae),
    "untrained SAE": reconstruct_with(untrained_sae),
    "tokens := fit mean": lambda tokens: mean.expand(tokens.shape).clone(),
}.items():
    logits = edited_logits(substitute)
    agreement = (logits.argmax(1) == baseline_pred).float().mean().item()
    print(f"{name:>22s}  {accuracy(logits, y_test):8.1%}  {agreement:9.1%}")

######################################################################
# Where on the scalp do class-selective features fire?
# ----------------------------------------------------
#
# We average each window's codes over its tokens and compute, on the **fit**
# windows, a selectivity index per feature and class: the difference between
# the class mean and the mean of the other classes, divided by the standard
# deviation over all windows. The 32 most selective features per class are
# chosen there, and only then examined on the test session.


def per_window(codes):
    return codes.reshape(-1, n_tokens, codes.shape[-1])


fit_window_codes = per_window(fit_codes).mean(1)
selectivity = torch.stack(
    [
        (
            fit_window_codes[y_fit == class_id].mean(0)
            - fit_window_codes[y_fit != class_id].mean(0)
        )
        / fit_window_codes.std(0).clamp_min(1e-6)
        for class_id in range(len(LABELS))
    ]
)
n_selected = 32
selected = selectivity.topk(n_selected, dim=1).indices  # (classes, n_selected)
for label, features in zip(LABELS, selected.tolist()):
    print(f"Top five selected for {label:>10s}: {features[:5]}")

######################################################################
# REVE orders the tokens channel by channel, four patches each, so a
# feature's mean code per channel is a topographic map. For the top feature
# of each class, we average its test-session code over the four patches and
# show the mean over the test trials of that class *minus* the mean over the
# other test trials: red channels are where the feature fires more for its
# class (each map has its own symmetric color scale, in code units). Because
# Euclidean alignment mixes channels, a channel here is an aligned virtual
# channel placed at the electrode's position, not the raw electrode. The maps
# describe what this frozen encoder encodes, not a validated neural source.

test_token_codes = per_window(test_codes).reshape(len(x_test), n_chans, n_patches, -1)
fig, axes = plt.subplots(1, len(LABELS), figsize=(11, 3.2))
for class_id, (ax, label) in enumerate(zip(axes, LABELS)):
    feature = selected[class_id, 0].item()
    channel_codes = test_token_codes[:, :, :, feature].mean(2)  # (trials, chans)
    contrast = (
        channel_codes[y_test == class_id].mean(0)
        - channel_codes[y_test != class_id].mean(0)
    ).numpy()
    limit = max(np.abs(contrast).max(), 1e-6)
    image, _ = mne.viz.plot_topomap(
        contrast,
        raw_info,
        axes=ax,
        show=False,
        cmap="RdBu_r",
        vlim=(-limit, limit),
        contours=0,
        extrapolate="local",
    )
    fig.colorbar(image, ax=ax, shrink=0.6, pad=0.02)
    ax.set_title(f"#{feature}, selected for {label.replace('_', ' ')}", fontsize=9)
fig.suptitle(
    "Test session: class minus other classes, most selective feature per class",
    fontsize=10,
)
fig.tight_layout()

######################################################################
# When we ran this example, the feature selected for left hand fired more
# over the right sensorimotor area and the one selected for right hand over
# the left, as expected from the contralateral organization of hand motor
# imagery. The map of the feature selected for feet showed only weak,
# scattered differences (note its much smaller color scale). One feature
# per class from one subject is suggestive at most.
#
# Intervening on class-selective features
# ---------------------------------------
#
# To remove a set of features we do not replace the tokens by the SAE
# reconstruction, which would add the reconstruction error to the edit.
# Instead we subtract only the removed features' decoded contribution from the
# original tokens. With an empty set this returns the native tokens exactly,
# so any change in the output is caused by the removed features.
#
# For each class we remove its 32 selected features and measure the drop in
# the model's probability for that class on the test trials of that class.
# Two controls tell whether a drop is specific:
#
# * **random features**: 20 sets of 32 features drawn from those active on
#   the fit windows;
# * **other classes' features**: the sets selected for the three other
#   classes, measured on the same trials.
#
# The drop that a class's own set causes on the trials of the *other* classes
# is also reported. Each set is evaluated once on all test windows.


def remove_features(features):
    def substitute(tokens):
        codes = sae.encode(standardize(tokens))
        kept = codes.clone()
        kept[:, features] = 0
        contribution = (sae.decode(codes) - sae.decode(kept)) * std
        return tokens - contribution.reshape(tokens.shape)

    return substitute


baseline_prob = baseline_logits.softmax(1)
class_trials = [y_test == class_id for class_id in range(len(LABELS))]


def probability_drops(features):
    """Edited probabilities and the drop of p(c) on the trials of each class c."""
    prob = edited_logits(remove_features(features)).softmax(1)
    drop = baseline_prob - prob
    return prob, np.array(
        [drop[t, c].mean().item() for c, t in enumerate(class_trials)]
    )


# drop_matrix[j, c]: drop of p(c) on class-c trials after removing class j's set
selected_probs, drop_matrix = zip(*(probability_drops(s) for s in selected))
drop_matrix = np.stack(drop_matrix)

candidate_features = active_on_fit.nonzero().squeeze(1)
generator = torch.Generator().manual_seed(0)
n_random = 20
random_drops = np.stack(
    [
        probability_drops(
            candidate_features[
                torch.randperm(len(candidate_features), generator=generator)[
                    :n_selected
                ]
            ]
        )[1]
        for _ in range(n_random)
    ]
)

effects = []
for class_id, (label, trials) in enumerate(zip(LABELS, class_trials)):
    prob = selected_probs[class_id]
    effects.append(
        {
            "label": label,
            "own": drop_matrix[class_id, class_id],
            "other sets": np.delete(drop_matrix[:, class_id], class_id).mean(),
            "random mean": random_drops[:, class_id].mean(),
            "random max": random_drops[:, class_id].max(),
            "own, other trials": (
                baseline_prob[~trials, class_id] - prob[~trials, class_id]
            )
            .mean()
            .item(),
            "flipped": (
                (prob[trials].argmax(1) != baseline_pred[trials]).float().mean().item()
            ),
        }
    )

print("Drop in p(class) after removing features (test session):")
print(
    f"{'class':>10s}  {'own set':>7s}  {'others':>6s}  {'random mean':>11s}  "
    f"{'random max':>10s}  {'own, other trials':>17s}  {'flipped':>7s}"
)
for row in effects:
    print(
        f"{row['label']:>10s}  {row['own']:7.3f}  {row['other sets']:6.3f}  "
        f"{row['random mean']:11.3f}  {row['random max']:10.3f}  "
        f"{row['own, other trials']:17.3f}  {row['flipped']:7.1%}"
    )

######################################################################
# The left panel summarizes the table for each class: the drop caused by the
# class's own features, by random feature sets (mean and maximum of 20) and
# by the other classes' features, on the trials of that class. The right
# panel shows every combination: row *j* removes the features selected for
# class *j*, column *c* is the drop of p(*c*) on the trials of class *c*. A
# class-specific effect would appear as a diagonal that stands out.

pretty = [label.replace("_", " ") for label in LABELS]
positions = np.arange(len(LABELS))
fig, (ax, ax_matrix) = plt.subplots(
    1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.5, 1]}
)
random_mean = np.array([row["random mean"] for row in effects])
random_max = np.array([row["random max"] for row in effects])
ax.bar(
    positions - 0.27,
    [row["own"] for row in effects],
    width=0.27,
    color="#D55E00",
    label="own features",
)
ax.bar(
    positions,
    random_mean,
    width=0.27,
    yerr=[np.zeros_like(random_mean), random_max - random_mean],
    color="#999999",
    capsize=3,
    label=f"random features (mean, max of {n_random})",
)
ax.bar(
    positions + 0.27,
    [row["other sets"] for row in effects],
    width=0.27,
    color="#0072B2",
    label="other classes' features (mean)",
)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(positions)
ax.set_xticklabels(pretty)
ax.set_ylabel("Drop in p(class) on its trials")
ax.set_title(f"Removing {n_selected} SAE features at block {block}", fontsize=10)
ax.legend(frameon=False, fontsize=8)
limit = np.abs(drop_matrix).max()
image = ax_matrix.imshow(drop_matrix, cmap="RdBu_r", vmin=-limit, vmax=limit)
for (row, column), value in np.ndenumerate(drop_matrix):
    ax_matrix.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
ax_matrix.set_xticks(positions)
ax_matrix.set_xticklabels(pretty, fontsize=8)
ax_matrix.set_yticks(positions)
ax_matrix.set_yticklabels(pretty, fontsize=8)
ax_matrix.set_xlabel("drop of p(c) on class-c trials")
ax_matrix.set_ylabel("features removed")
fig.colorbar(image, ax=ax_matrix, shrink=0.8)
fig.tight_layout()

######################################################################
# Interpretation and next steps
# -----------------------------
#
# * **What this run shows.** When we ran this example on a laptop CPU, the
#   frozen REVE with its trained head reached 76% on the test session (chance
#   25%). Its validation accuracy, on the last two runs of the training
#   session, was only 56%: single-subject estimates on a few runs vary a
#   lot. Substituting the trained SAE's reconstruction at block 18 kept
#   about 80% of the test decisions, against 34% for the untrained
#   dictionary and for mean tokens (both at chance accuracy). Removing the
#   32 features selected for left hand, right hand or tongue lowered the
#   probability of that class on its own test trials by about 0.1–0.2. That
#   is more than each of the 20 random sets (at most about 0.05) and roughly
#   three to five times the drop on the other classes' trials, while
#   removing the other classes' sets slightly raised it on average. For feet
#   the drop was small (about 0.03) and close to the largest random sets.
#   The SAE is trained in float32, so these numbers change slightly with the
#   machine and the number of threads; the accuracy and the block check did
#   not change in our runs.
# * **Read the substitution table first.** If the trained SAE's
#   reconstruction does not preserve the model's decisions much better than
#   the untrained and mean-token controls, feature-level conclusions are not
#   supported.
# * **Interventions are statements about this model.** A set that lowers
#   the probability of its class more than random sets and other classes'
#   sets shows that this frozen REVE with its trained head uses those
#   directions, on this subject and session. It does not show that the
#   features correspond to a physiological process. Relating them to
#   physiology requires spatial and spectral analyses, several subjects and
#   seeds, and checks for artifacts; with Euclidean alignment, the spatial
#   maps are also mixtures of electrodes.
# * **The effect of an edit depends on where it is made.** Later blocks can
#   compensate for an edit, and the flattening head weighs every token with
#   its own weights. Choose the block, the dictionary size, ``k`` and the
#   number of features on validation data, never on the test session.
# * **Budget.** A frozen encoder, a linear head and 3000 SAE updates on one
#   subject keep this example to a few minutes on a CPU. Fine-tuning, more
#   subjects and several seeds are needed before comparing layers or models.
#
# References
# ----------
#
# * El Ouahidi et al. (2025), `REVE: A Foundation Model for EEG - Adapting to
#   Any Setup with Large-Scale Pretraining on 25,000 Subjects
#   <https://arxiv.org/abs/2510.21585>`_, NeurIPS, introduce REVE.
# * He and Wu (2020), `Transfer Learning for Brain-Computer Interfaces: A
#   Euclidean Space Data Alignment Approach
#   <https://arxiv.org/abs/1808.05464>`_, IEEE Transactions on Biomedical
#   Engineering, introduce Euclidean alignment.
# * `SAE Lens documentation <https://jbloomaus.github.io/SAELens/>`_ describes
#   the external dictionary and tensor-batch trainer APIs used here (tested
#   with 6.51.3).
# * Makhzani and Frey, `k-Sparse Autoencoders
#   <https://arxiv.org/abs/1312.5663>`_, motivate sparse dictionary learning.
