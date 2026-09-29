""".. _sae-activation-analysis:

Sparse autoencoders for model-agnostic activation analysis
==================================================================

This tutorial fits a sparse autoencoder (SAE) to intermediate EEG-model
activations using `SAE Lens <https://github.com/decoderesearch/SAELens>`_.
Braindecode supplies activation capture and substitution; SAE Lens supplies
the dictionary, Top-K activation, losses and training procedure.

An overcomplete dictionary represents each activation using a small number of
learned directions. Sparsity is a useful constraint, not a guarantee that these
directions are independent physiological concepts. Reconstruction quality and
preservation of a model's predictions are different questions.

We use independent synthetic windows and a small, randomly initialized
convolutional network to make the tensor contracts visible without a download.
This is a workflow demonstration, **not EEG validation**: there are no labels,
trained EEG checkpoint, clinical conclusions or paper-reproduction claims.
The same capture/substitution calls apply to a trained Braindecode model.

Install the optional dependency with ``pip install 'braindecode[sae]'``
(or ``pip install sae-lens==6.51.3`` in a source checkout). SAE Lens brings a
substantial language-model dependency stack, but no language model or pretrained
SAE is downloaded here. Dictionaries trained on LLMs are not EEG dictionaries.
The bounded example uses CPU and 100 SAE updates.

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

import matplotlib.pyplot as plt
import torch
from torch import nn

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
# Split windows before collecting activations
# ---------------------------------------------------
#
# Independent windows let us use a simple fixed split. With real EEG, split
# subjects (or recordings for a within-subject question) *before* windowing:
# overlapping windows from one recording must not cross the split. Any
# classifier fitting must also use only training data. Use validation data to
# choose the layer, dictionary size and training budget, then freeze those
# choices before evaluating the test set once.

torch.manual_seed(87)
device = "cpu"
x = torch.randn(96, 4, 128, device=device)
x_train, x_valid, x_test = x[:64], x[64:80], x[80:]
model = (
    nn.Sequential(
        nn.Conv1d(4, 16, kernel_size=9, stride=4),
        nn.GELU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(16, 2),
    )
    .to(device)
    .eval()
)
model.requires_grad_(False)
layer = model[1]

######################################################################
# Put the feature axis last
# ---------------------------------
#
# The convolution emits ``(windows, features, time)``. Each time position is
# one SAE observation, so we move features last before flattening. Capture
# runs under ``no_grad`` and is detached explicitly: the SAE never updates
# the EEG network. Mean and standard deviation are fitted on training rows
# only, with a floor for nearly constant dimensions.

with torch.no_grad():
    train_acts = capture_activations(model, x_train, layer).detach()
train_rows = train_acts.movedim(1, -1).reshape(-1, train_acts.shape[1])
mean = train_rows.mean(0)
std = train_rows.std(0, correction=0).clamp_min(1e-6)
train_scaled = (train_rows - mean) / std
d_in = train_rows.shape[1]
print(f"Training rows: {tuple(train_rows.shape)}")

######################################################################
# Fit the external dictionary
# -----------------------------------
#
# SAE Lens's low-level trainer accepts an iterator of tensor batches; it does
# not require a tokenizer or TransformerLens model. Sampling with replacement
# bounds this demonstration to 100 updates. The library owns optimization,
# scheduling and its auxiliary loss for inactive features; we add no trainer
# or resampling implementation. This uses upstream norm-adjusted Top-K,
# rather than imposing a separate unit-norm constraint on decoder columns.
# Logging and checkpoints are disabled to avoid external services or files.

batch_size = 64


def activation_batches():
    while True:
        indices = torch.randint(len(train_scaled), (batch_size,), device=device)
        yield train_scaled[indices]


sae = TopKTrainingSAE(
    TopKTrainingSAEConfig(
        d_in=d_in,
        d_sae=4 * d_in,
        k=4,
        device=device,
        dtype="float32",
        normalize_activations="none",
        reshape_activations="none",
        apply_b_dec_to_input=True,
        rescale_acts_by_decoder_norm=True,
    ),
    use_error_term=False,
)
trainer = SAETrainer(
    cfg=SAETrainerConfig(
        total_training_samples=100 * batch_size,
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
# Inspect held-out reconstruction and feature usage
# ---------------------------------------------------------
#
# Validation reconstruction error is measured in standardized coordinates.
# The fraction of features never observed here depends on sample size; it is
# not a proof of permanently dead features, nor is inactivity specialization.
# Top-K means *at most* k nonzero features because selected values are rectified.

with torch.no_grad():
    valid_acts = capture_activations(model, x_valid, layer).detach()
    valid_rows = valid_acts.movedim(1, -1).reshape(-1, d_in)
    valid_scaled = (valid_rows - mean) / std
    codes = sae.encode(valid_scaled)
    reconstructed = sae.decode(codes)
    mse = (reconstructed - valid_scaled).square().mean()
    usage = (codes != 0).float().mean(0)
print(f"Validation standardized MSE: {mse.item():.4f}")
print(f"Mean active features: {(codes != 0).sum(-1).float().mean().item():.2f}")
print(f"Features not observed on validation: {(usage == 0).float().mean():.1%}")
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(range(len(usage)), usage.cpu().numpy())
ax.set(xlabel="SAE feature", ylabel="Validation firing fraction")
fig.tight_layout()

######################################################################
# Return reconstructions to the original coordinate system
# ----------------------------------------------------------------
#
# The substitution callable receives the original activation, not SAE codes.
# It reverses training-set scaling and restores the original axes, device and
# dtype. ``use_error_term=False`` is important: adding the reconstruction error
# back would hide reconstruction damage. Keep both networks in evaluation mode.


def reconstruct(output):
    rows = output.movedim(1, -1).reshape(-1, d_in).to(mean)
    restored = sae.decode(sae.encode((rows - mean) / std)) * std + mean
    return restored.reshape(output.movedim(1, -1).shape).movedim(-1, 1).to(output)


with torch.no_grad():
    baseline = model(x_test)
    substituted = run_with_activation_substitution(model, x_test, layer, reconstruct)
    zeroed = run_with_activation_substitution(model, x_test, layer, torch.zeros_like)
for name, prediction in [("reconstruction", substituted), ("zero layer", zeroed)]:
    print(
        f"Test logit RMS change ({name}): "
        f"{(prediction - baseline).square().mean().sqrt().item():.5f}"
    )

######################################################################
# Interpretation and next steps
# -------------------------------------
#
# These logits belong to a random network, so a small change is only a wiring
# check. For a trained EEG model, report held-out task performance with its
# unchanged native head, including random-dictionary and layer-ablation controls.
# Compare several seeds, and examine associations with subject identity,
# artifacts and acquisition site before proposing physiological interpretations.
# Good reconstruction alone establishes neither causality nor monosemanticity.
#
# To reuse a fitted dictionary, save its upstream SAE Lens configuration and
# weights **together with** these normalization tensors, the model checkpoint,
# layer/axis contract, split manifest and library version; verify reload parity.
# A dictionary or its scaling statistics cannot be transferred to a different
# model or layer merely because the dimensions match.
#
# References
# ------------------
#
# * `SAE Lens documentation <https://jbloomaus.github.io/SAELens/>`_ describes
#   the external training and inference APIs used here (tested with 6.51.3).
# * Makhzani and Frey, `k-Sparse Autoencoders
#   <https://arxiv.org/abs/1312.5663>`_, motivate sparse dictionary learning.
