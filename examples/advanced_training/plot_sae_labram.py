""".. _sae-labram-interventions:

Sparse feature interventions in a pretrained LaBraM
===================================================

This tutorial fine-tunes the pretrained :class:`~braindecode.models.Labram`
foundation model on one subject of the BCI Competition IV 2a motor-imagery
dataset (BNCI2014_001 via MOABB), fits a Top-K sparse autoencoder (SAE) to the
token embeddings of one transformer block with
`SAE Lens <https://github.com/decoderesearch/SAELens>`_, and then *intervenes*
on the learned features while the model classifies the held-out session.
Braindecode supplies the data pipeline, the model and the generic activation
capture/substitution hooks; SAE Lens supplies the dictionary, its Top-K
activation, losses and training loop.

.. topic:: What is different about transformer tokens?

    LaBraM cuts every channel into 1-s patches and turns each (channel, patch)
    pair into one token, plus a classification (CLS) token. An SAE fitted to
    these tokens therefore learns features that come with a *scalp location*
    and a *time*, and we can draw them as topographic maps. Because the
    classification head reads the CLS token, a feature can only influence the
    decision through the blocks that follow the one we edit. We use this to
    ask a causal question: if we remove the features that were most
    selective for a class, does the model become less confident in that
    class, and more so than when removing the same number of random
    features?

Everything that is *fitted* (the decoder, the scaling statistics, the
dictionary, the selected features) uses only the *training session*; the
*test session*, recorded on a different day, is used only for evaluation.
All numbers come from a single subject, a few epochs of CPU fine-tuning and a
short SAE run. They describe this run, not LaBraM or motor imagery in general,
and support no scientific claim.

Install the optional dependency with ``pip install sae-lens==6.51.3`` (or the
optional ``braindecode[sae]`` extra when available). SAE Lens brings a
language-model dependency stack, but no language model or pretrained SAE is
used here. The LaBraM weights are downloaded from the Hugging Face Hub
(``braindecode/labram-pretrained``, about 23 MB) the first time the example
runs.

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
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn

from braindecode import EEGClassifier
from braindecode.datasets import MOABBDataset
from braindecode.models import Labram
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
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
# We use subject 3 of BCI Competition IV 2a (BNCI2014_001), the recording
# used in :ref:`finetune-foundation-model`: 22 EEG channels at 250 Hz, four
# motor-imagery classes (feet, left hand, right hand, tongue) and two
# sessions of 288 trials recorded on different days.

subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])

######################################################################
# Preprocessing to match the pretrained model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# LaBraM was pretrained on signals in units of 0.1 mV, band-passed between
# 0.1 and 75 Hz, notch-filtered at the power-line frequency and sampled at
# 200 Hz, with 200-sample (1 s) patches. We apply the same steps to the EEG
# channels: volts are multiplied by :math:`10^4` to obtain 0.1 mV.

preprocess(
    dataset,
    [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        Preprocessor(lambda data: data * 1e4),
        Preprocessor("filter", l_freq=0.1, h_freq=75.0),
        Preprocessor("notch_filter", freqs=50.0),
        Preprocessor("resample", sfreq=200),
    ],
    n_jobs=-1,
)

######################################################################
# Windowing and split
# ~~~~~~~~~~~~~~~~~~~
#
# Each window covers the 4-s motor-imagery period that MOABB defines after
# the cue: 800 samples, i.e. four patches per channel. We split by session;
# ``"0train"`` is used for every fitting step below and ``"1test"`` only for
# evaluation.

raw_info = dataset.datasets[0].raw.info
ch_names = raw_info["ch_names"]
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    preload=True,
)
split_by_session = windows_dataset.split("session")
train_set, test_set = split_by_session["0train"], split_by_session["1test"]

# BCI IV 2a labels in MOABB's alphabetical order; the position is the class id.
LABELS = ("feet", "left_hand", "right_hand", "tongue")

x_train = torch.as_tensor(np.stack([x for x, *_ in train_set]), dtype=torch.float32)
y_train = torch.as_tensor([y for _, y, _ in train_set])
x_test = torch.as_tensor(np.stack([x for x, *_ in test_set]), dtype=torch.float32)
y_test = torch.as_tensor([y for _, y, _ in test_set])
n_chans, n_times = x_train.shape[1:]
print(f"Train windows: {tuple(x_train.shape)}; test windows: {tuple(x_test.shape)}")

######################################################################
# Loading and fine-tuning the pretrained LaBraM
# ---------------------------------------------
#
# Loading the pretrained weights
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# :ref:`load-pretrained-models` shows the general ``from_pretrained``
# pattern. Here we load the weights directly, because the published
# checkpoint was trained on 15-s inputs: its temporal embedding has one entry
# per patch position up to 15. Our model sees four patches, and we keep the
# first entries of that table. The classification
# head is new (``missing_keys`` lists only ``final_layer``). LaBraM looks up
# a spatial embedding for every channel by name; the small wrapper below binds
# this montage's channel names so that the classifier and the hooks can call
# the model with a tensor only.

device = "cpu"
set_random_seeds(seed=20240205, cuda=False)

labram = Labram(n_chans=n_chans, n_outputs=len(LABELS), n_times=n_times, sfreq=200.0)
state = load_file(hf_hub_download("braindecode/labram-pretrained", "model.safetensors"))
state["temporal_embedding"] = state["temporal_embedding"][
    :, : labram.temporal_embedding.shape[1]
]
print(labram.load_state_dict(state, strict=False))


class LabramOnMontage(nn.Module):
    """Call LaBraM with a fixed list of channel names."""

    def __init__(self, labram, ch_names):
        super().__init__()
        self.labram = labram
        self.ch_names = list(ch_names)

    def forward(self, x):
        return self.labram(x, ch_names=self.ch_names)


######################################################################
# Fine-tuning
# ~~~~~~~~~~~
#
# We fine-tune all weights for ten epochs on the training session with a
# small learning rate. On a laptop CPU this takes well under a minute. The
# model overfits quickly and cross-session four-class accuracy is modest;
# the point of this tutorial is the SAE analysis, which only requires a
# fixed model whose decisions we can probe.

classifier = EEGClassifier(
    LabramOnMontage(labram, ch_names),
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    optimizer__lr=5e-5,
    optimizer__weight_decay=0.05,
    batch_size=32,
    max_epochs=10,
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
print(f"Test-session accuracy (chance 25%): {baseline_accuracy:.1%}")

######################################################################
# Which block can reach the head?
# -------------------------------
#
# A LaBraM block outputs ``(windows, 1 + channels × patches, 200)``: the CLS
# token followed by one token per (channel, patch). Token ``1 + j`` belongs
# to channel ``j // n_patches`` and patch ``j % n_patches``. The head reads
# only the final CLS token, so an edit of the patch tokens can reach the
# decision only through the attention of the *following* blocks.
#
# A quick check makes this concrete. For a few blocks we replace every patch
# token by its training mean, leave CLS untouched, and count how many
# training-session predictions stay the same. Only the training session is
# used, because this check informs a modelling choice.

n_patches = n_times // model.labram.patch_size
n_tokens = n_chans * n_patches


def patch_rows(activation):
    """(windows, 1 + tokens, dim) -> (windows * tokens, dim), CLS removed."""
    return activation[:, 1:, :].reshape(-1, activation.shape[-1])


def with_patch_tokens(output, rows):
    """Copy of ``output`` whose patch tokens are replaced by ``rows``."""
    replacement = output.clone()
    replacement[:, 1:, :] = rows.reshape(output[:, 1:, :].shape).to(output)
    return replacement


def patch_tokens_set_to(value):
    """Substitution that sets every patch token to ``value``."""

    def substitute(output):
        return with_patch_tokens(output, value.expand(patch_rows(output).shape))

    return substitute


with torch.no_grad():
    train_pred = model(x_train).argmax(1)
    for index in (0, 3, 5, 8, 11):
        block = model.labram.blocks[index]
        block_mean = patch_rows(capture_activations(model, x_train, block)).mean(0)
        substitute = patch_tokens_set_to(block_mean)
        pred = run_with_activation_substitution(
            model, x_train, block, substitute
        ).argmax(1)
        agreement = (pred == train_pred).float().mean().item()
        print(f"Block {index + 1:2d}: {agreement:6.1%} of predictions unchanged")

######################################################################
# Replacing the patch tokens of the last block changes nothing, so an SAE
# fitted there could never influence this CLS-pooled head. We therefore work
# with block 6 of 12 (index 5), in the middle of the network: the check above
# shows that its patch tokens still matter for the decision, and six more
# blocks separate it from the head.
#
# Fitting an SAE to the patch tokens
# ----------------------------------
#
# Capturing token embeddings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The CLS token is excluded from the dictionary and never edited. The
# per-coordinate mean and standard deviation come from training tokens only,
# so SAE Lens's own ``normalize_activations`` is switched off.

layer = model.labram.blocks[5]
with torch.no_grad():
    train_rows = patch_rows(capture_activations(model, x_train, layer))
d_in = train_rows.shape[1]
mean = train_rows.mean(0)
std = train_rows.std(0, correction=0).clamp_min(1e-6)
train_scaled = (train_rows - mean) / std
print(f"Training tokens: {tuple(train_rows.shape)} ({n_tokens} per window)")

######################################################################
# Training the Top-K SAE
# ~~~~~~~~~~~~~~~~~~~~~~
#
# SAE Lens's :class:`~sae_lens.training.sae_trainer.SAETrainer` consumes an
# iterator of tensor batches. We learn 800 features (four times the embedding
# width) with at most ``k=16`` active per token, for 3000 updates on batches
# sampled with replacement. A copy made before training serves as an
# untrained control with the same initialization.

batch_size = 256
n_updates = 3000
k_active = 16

torch.manual_seed(20240205)


def activation_batches():
    while True:
        indices = torch.randint(len(train_scaled), (batch_size,), device=device)
        yield train_scaled[indices]


sae = TopKTrainingSAE(
    TopKTrainingSAEConfig(
        d_in=d_in,
        d_sae=4 * d_in,
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
SAETrainer(
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
).fit()
_ = sae.eval()

######################################################################
# Reconstruction on the test session
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The fraction of variance unexplained (FVU) is 0 for a perfect
# reconstruction and 1 for predicting the mean. We compute it on test tokens
# in the standardized coordinates, for the trained and the untrained
# dictionary.


def fvu(target, reconstruction):
    residual = (target - reconstruction).square().sum()
    return (residual / (target - target.mean(0)).square().sum()).item()


with torch.no_grad():
    test_scaled = (patch_rows(capture_activations(model, x_test, layer)) - mean) / std
    test_codes = sae.encode(test_scaled)
    fvu_trained = fvu(test_scaled, sae.decode(test_codes))
    fvu_untrained = fvu(
        test_scaled, untrained_sae.decode(untrained_sae.encode(test_scaled))
    )
    train_codes = sae.encode(train_scaled)
active_on_train = (train_codes != 0).any(0)
print(f"Test FVU: trained {fvu_trained:.3f}, untrained {fvu_untrained:.3f}")
print(f"Mean active features per token: {(test_codes != 0).sum(-1).float().mean():.2f}")
print(f"Features never active on the training session: {(~active_on_train).sum()}")

######################################################################
# Substituting the reconstruction into the model
# ----------------------------------------------
#
# :func:`~braindecode.visualization.run_with_activation_substitution` replaces
# the block output during a forward pass. The callables below keep the CLS
# token and the output's shape and dtype, standardize the patch tokens with
# the training statistics, pass them through a dictionary and undo the
# scaling. We compare the trained SAE with the untrained one and with setting
# every patch token to its training mean.


def encode_tokens(output, dictionary):
    return dictionary.encode((patch_rows(output).to(mean) - mean) / std)


def reconstruct_with(dictionary):
    def substitute(output):
        codes = encode_tokens(output, dictionary)
        return with_patch_tokens(output, dictionary.decode(codes) * std + mean)

    return substitute


def evaluate(substitute, x=x_test):
    with torch.no_grad():
        return run_with_activation_substitution(model, x, layer, substitute)


print(f"{'substitution':>24s}  accuracy  agreement")
print(f"{'none (native model)':>24s}  {baseline_accuracy:8.1%}  {1.0:9.1%}")
for name, substitute in {
    "trained SAE": reconstruct_with(sae),
    "untrained SAE": reconstruct_with(untrained_sae),
    "patch tokens := mean": patch_tokens_set_to(mean),
}.items():
    pred = evaluate(substitute).argmax(1)
    accuracy = (pred == y_test).float().mean().item()
    agreement = (pred == baseline_pred).float().mean().item()
    print(f"{name:>24s}  {accuracy:8.1%}  {agreement:9.1%}")

######################################################################
# Where on the scalp do class-selective features fire?
# ----------------------------------------------------
#
# We average each window's codes over its tokens and compute, on the
# **training** session, a selectivity index per feature and class: the
# difference between the class mean and the mean of the other classes,
# divided by the standard deviation over all windows. The 32 most selective
# features per class (4% of the dictionary) are chosen there, and only then
# are they examined on the test session.


def per_window(codes):
    return codes.reshape(-1, n_tokens, codes.shape[-1])


train_window_codes = per_window(train_codes).mean(1)
selectivity = torch.stack(
    [
        (
            train_window_codes[y_train == class_id].mean(0)
            - train_window_codes[y_train != class_id].mean(0)
        )
        / train_window_codes.std(0).clamp_min(1e-6)
        for class_id in range(len(LABELS))
    ]
)
n_selected = 32
selected = selectivity.topk(n_selected, dim=1).indices  # (classes, n_selected)
for label, features in zip(LABELS, selected.tolist()):
    print(f"Top five selected for {label:>10s}: {features[:5]}")

######################################################################
# Because every token is tied to one electrode, a feature's mean code per
# channel is a topographic map. Below, for the top feature of each class, we
# average its test-session code over the four patches and show the mean over
# the test trials of that class *minus* the mean over the other test trials:
# red channels are where the feature fires more for its class (each map has
# its own symmetric color scale, in code units). Differences
# over the sensorimotor cortex contralateral to the imagined hand would be
# consistent with the physiology of hand motor imagery; differences at edge
# or frontal electrodes may instead reflect artifacts. Either way, the maps
# describe what this fine-tuned model encodes, not a validated neural source.

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
# Intervening on class-selective features
# ---------------------------------------
#
# To remove a set of features we do not replace the tokens by the SAE
# reconstruction, which would add the reconstruction error to the edit.
# Instead we subtract only the removed features' decoded contribution from the
# original tokens. With an empty set this returns the native activations
# exactly, so any change in the output is caused by the removed features.
#
# For each class we remove its 32 selected features and measure the drop in
# the model's probability for that class on the test trials of that class. As
# a control we remove 32 features drawn at random from those active on the
# training session, repeated 20 times. A selected set whose drop clearly
# exceeds the random ones indicates that these features carry
# class-relevant information *for this model*. The same drop measured on the
# trials of the other classes checks whether the effect is specific.


def remove_features(features):
    def substitute(output):
        codes = encode_tokens(output, sae)
        kept = codes.clone()
        kept[:, features] = 0
        contribution = (sae.decode(codes) - sae.decode(kept)) * std
        return with_patch_tokens(output, patch_rows(output).to(mean) - contribution)

    return substitute


baseline_prob = baseline_logits.softmax(1)
candidate_features = active_on_train.nonzero().squeeze(1)
generator = torch.Generator().manual_seed(0)
n_random = 20
effects = []
for class_id, label in enumerate(LABELS):
    trials = y_test == class_id
    prob = evaluate(remove_features(selected[class_id]), x_test).softmax(1)
    drop = baseline_prob[:, class_id] - prob[:, class_id]
    random_drops = []
    for _ in range(n_random):
        random_set = candidate_features[
            torch.randperm(len(candidate_features), generator=generator)[:n_selected]
        ]
        random_prob = evaluate(remove_features(random_set), x_test[trials]).softmax(1)
        random_drops.append(
            (baseline_prob[trials, class_id] - random_prob[:, class_id]).mean().item()
        )
    effects.append(
        {
            "label": label,
            "selected": drop[trials].mean().item(),
            "other trials": drop[~trials].mean().item(),
            "random mean": float(np.mean(random_drops)),
            "random max": float(np.max(random_drops)),
            "flipped": (
                (prob[trials].argmax(1) != baseline_pred[trials]).float().mean().item()
            ),
        }
    )

print("Drop in p(class) after removing features (test session):")
print(
    f"{'class':>11s}  {'selected':>8s}  {'other trials':>12s}  "
    f"{'random mean':>11s}  {'random max':>10s}  {'flipped':>7s}"
)
for row in effects:
    print(
        f"{row['label']:>11s}  {row['selected']:8.3f}  {row['other trials']:12.3f}  "
        f"{row['random mean']:11.3f}  {row['random max']:10.3f}  {row['flipped']:7.1%}"
    )

######################################################################
# The bar chart summarizes the table: the drop in the probability of the
# class on its own test trials, for the selected features, for 20 random
# feature sets (mean and range) and, as a specificity check, for the
# selected features on the trials of the other classes.

positions = np.arange(len(LABELS))
fig, ax = plt.subplots(figsize=(7.5, 3.6))
random_mean = np.array([row["random mean"] for row in effects])
random_max = np.array([row["random max"] for row in effects])
ax.bar(
    positions - 0.27,
    [row["selected"] for row in effects],
    width=0.27,
    color="#D55E00",
    label="selected features, class trials",
)
ax.bar(
    positions,
    random_mean,
    width=0.27,
    yerr=[np.zeros_like(random_mean), random_max - random_mean],
    color="#999999",
    capsize=3,
    label=f"random features, class trials (mean, max of {n_random})",
)
ax.bar(
    positions + 0.27,
    [row["other trials"] for row in effects],
    width=0.27,
    color="#0072B2",
    label="selected features, other trials",
)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(positions)
ax.set_xticklabels([label.replace("_", " ") for label in LABELS])
ax.set_ylabel("Drop in p(class)")
ax.set_title(
    f"Removing {n_selected} SAE features at block 6 (test session)", fontsize=10
)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()

######################################################################
# Interpretation and next steps
# -----------------------------
#
# * **What this run shows.** When we ran this example on a laptop CPU, the
#   fine-tuned model reached about 36% on the test session (chance 25%). The
#   trained SAE's reconstruction kept about 82% of its decisions, against
#   about 53% for the untrained dictionary and for mean patch tokens.
#   Removing the selected features lowered the probability of their class
#   more than random sets did for three of the four classes, but by a similar
#   amount on the other classes' trials. These features push the model
#   towards their class wherever they fire; this run does not show that the
#   model uses them specifically on trials of that class. Exact numbers can
#   differ slightly between machines.
# * **Read the substitution table first.** If the trained SAE's reconstruction
#   does not preserve the model's decisions much better than the untrained and
#   mean-token controls, feature-level conclusions are not supported.
# * **Interventions are statements about this model.** A selected set that
#   lowers the probability of its class far more than random sets shows that
#   the fine-tuned LaBraM uses those directions, on this subject and session.
#   It does not show that the features correspond to a physiological process.
#   Relating them to physiology requires spatial and spectral analyses,
#   several subjects and seeds, and checks for artifacts.
# * **The effect of an edit depends on where it is made.** Later blocks may
#   compensate for an early edit, and an edit of the last block's patch tokens
#   does nothing to a CLS-pooled head. Choose the block, the dictionary size,
#   ``k`` and the number of features on a validation split of the training
#   session, never on the test session.
# * **Budget.** Ten epochs of fine-tuning and 3000 SAE updates on one subject
#   keep this example short on a CPU. Longer training, more subjects and
#   several seeds are needed before comparing layers or models.
#
# References
# ----------
#
# * Jiang, Zhao and Lu (2024), `Large Brain Model for Learning Generic
#   Representations with Tremendous EEG Data in BCI
#   <https://arxiv.org/abs/2405.18765>`_, ICLR, introduce LaBraM.
# * `SAE Lens documentation <https://jbloomaus.github.io/SAELens/>`_ describes
#   the external dictionary and tensor-batch trainer APIs used here (tested
#   with 6.51.3).
# * Makhzani and Frey, `k-Sparse Autoencoders
#   <https://arxiv.org/abs/1312.5663>`_, motivate sparse dictionary learning.
