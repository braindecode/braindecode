""".. _sae-labram-interventions:

Sparse feature interventions in LaBraM
==============================================

How do we intervene on a transformer's sparse features without changing its
classification head? This tutorial uses an external Top-K SAE from
`SAE Lens <https://github.com/decoderesearch/SAELens>`_ and Braindecode's generic
activation utilities. It focuses on token selection, a validation-selected
feature ablation, and controls for the reconstruction intervention.

Unlike convolutional feature maps, transformer activations already have the
embedding axis last. This standalone example concentrates on token selection
and controlled interventions, without a custom SAE or training framework.

To keep execution bounded, we use a **random, two-block LaBraM** and 48 synthetic
windows, with no dataset or weight downloads. These results check the workflow,
not the physiological meaning of features or LaBraM's predictive performance.
In particular, this is not replication of a clinical EEG-SAE study or validation
on NMT. A real experiment needs a separately validated, frozen EEG checkpoint.

Install ``pip install sae-lens==6.51.3`` separately (or the optional
``braindecode[sae]`` extra when available). No SAE dependency is required by
Braindecode itself. The external library also installs language-model tooling;
we do not use its language-model runners or pretrained LLM dictionaries.
This example runs 100 SAE updates on CPU.

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
from functools import partial

import matplotlib.pyplot as plt
import mne
import torch

from braindecode.models import Labram
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
# Freeze the model and define disjoint splits
# ---------------------------------------------------
#
# Canonical channel names select LaBraM's spatial embeddings. We retain the
# normal 200-sample patch size but shorten the input to two patches per channel.
# The model determines its actual embedding width from the patching pipeline;
# do not infer it from a paper or hard-code 200 when adapting this example.
#
# Windows here are independent. Real EEG requires subject/recording-disjoint
# splits before windowing, with all preprocessing estimates fitted on training
# data. Select the EEG checkpoint using validation task performance *before*
# SAE analysis. No test performance should decide whether to start SAE fitting.

torch.manual_seed(87)
device = "cpu"
info = mne.create_info(["C3", "C4", "CZ", "PZ"], sfreq=200, ch_types="eeg")
x = torch.randn(48, 4, 400, device=device)
x_train, x_valid, x_test = x[:32], x[32:40], x[40:]
# The model is initialized below after choosing its pooling contract.

######################################################################
# Learn a dictionary of non-CLS token embeddings
# ------------------------------------------------------
#
# Transformer block output has shape ``(windows, tokens, embedding)``. Token
# zero is CLS; all other tokens come from the patching pipeline. We exclude
# CLS when fitting the dictionary and leave it untouched during interventions.
# Mean pooling makes the classifier read the non-CLS tokens after the last
# block. For a CLS-head model, intervening on non-CLS tokens in the last block
# cannot affect CLS; choose an earlier block or explicitly study CLS instead.
# Here we use mean pooling so the experiment tests a connected path.

model = (
    Labram(
        n_chans=4,
        n_times=400,
        n_outputs=2,
        chs_info=info["chs"],
        patch_size=200,
        num_layers=2,
        num_heads=4,
        conv_out_channels=4,
        use_mean_pooling=True,
    )
    .to(device)
    .eval()
)
model.requires_grad_(False)
# The generic hooks accept a callable model: bind the channel names once.
forward = partial(model, ch_names=info["ch_names"])
layer = model.blocks[-1]
with torch.no_grad():
    train_acts = capture_activations(forward, x_train, layer).detach()
d_in = train_acts.shape[-1]
train_rows = train_acts[:, 1:, :].reshape(-1, d_in)
mean = train_rows.mean(0)
std = train_rows.std(0, correction=0).clamp_min(1e-6)
train_scaled = (train_rows - mean) / std
print(f"Block output: {tuple(train_acts.shape)}; training rows: {len(train_rows)}")

######################################################################
# Train with the upstream tensor-batch trainer
# ----------------------------------------------------
#
# The SAE's dictionary is specific to this frozen model, layer and training
# distribution. Fixed standardization lives outside SAE Lens, so its internal
# normalization is disabled. Upstream norm-adjusted Top-K and auxiliary loss
# are used without adding a custom optimizer, callback or resampling routine.
# A copy made before fitting provides a same-initialization untrained control.

batch_size = 64


def activation_batches():
    while True:
        indices = torch.randint(len(train_scaled), (batch_size,), device=device)
        yield train_scaled[indices]


sae = TopKTrainingSAE(
    TopKTrainingSAEConfig(
        d_in=d_in,
        d_sae=4 * d_in,
        k=8,
        device=device,
        dtype="float32",
        normalize_activations="none",
        reshape_activations="none",
        apply_b_dec_to_input=True,
        rescale_acts_by_decoder_norm=True,
    ),
    use_error_term=False,
)
random_sae = copy.deepcopy(sae).eval()
SAETrainer(
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
).fit()
sae.eval()

######################################################################
# Select a feature using validation only
# ----------------------------------------------
#
# We select the most frequently active feature on validation, not the feature
# producing the largest test effect. This is a simple, predeclared selection
# rule, not a claim about a clinical concept. Reconstruction MSE is in
# standardized coordinates; firing frequency depends on the sampled windows.

with torch.no_grad():
    valid_acts = capture_activations(forward, x_valid, layer).detach()
    valid_scaled = (valid_acts[:, 1:, :].reshape(-1, d_in) - mean) / std
    valid_codes = sae.encode(valid_scaled)
    frequency = (valid_codes != 0).float().mean(0)
    feature = int(frequency.argmax())
    valid_mse = (sae.decode(valid_codes) - valid_scaled).square().mean()
print(f"Validation MSE: {valid_mse.item():.4f}; selected feature: {feature}")

######################################################################
# Reconstruct, then ablate in the SAE coordinate system
# -------------------------------------------------------------
#
# Ablation sets one nonnegative code to zero before decoding; SAE codes cannot
# replace transformer embeddings directly. Both reconstruction and ablation
# invert the same train-only normalization. The callable preserves CLS and the
# output's original shape, dtype and device without mutating its input.
# We compare feature ablation to SAE reconstruction, not just to the native
# model, to separate its incremental effect from reconstruction damage.


def substitute(output, dictionary=sae, ablate=False):
    rows = output[:, 1:, :].reshape(-1, d_in).to(mean)
    codes = dictionary.encode((rows - mean) / std)
    if ablate:
        codes = codes.clone()
        codes[:, feature] = 0
    restored = dictionary.decode(codes) * std + mean
    replacement = output.clone()
    replacement[:, 1:, :] = restored.reshape(output[:, 1:, :].shape).to(output)
    return replacement


def zero_tokens(output):
    replacement = output.clone()
    replacement[:, 1:, :] = 0
    return replacement


with torch.no_grad():
    baseline = forward(x_test)
    predictions = {
        "SAE": run_with_activation_substitution(forward, x_test, layer, substitute),
        "random SAE": run_with_activation_substitution(
            forward, x_test, layer, lambda h: substitute(h, dictionary=random_sae)
        ),
        "feature ablated": run_with_activation_substitution(
            forward, x_test, layer, lambda h: substitute(h, ablate=True)
        ),
        "zero tokens": run_with_activation_substitution(
            forward, x_test, layer, zero_tokens
        ),
    }
changes = [
    (value - baseline).square().mean().sqrt().item() for value in predictions.values()
]
increment = predictions["feature ablated"] - predictions["SAE"]
print(f"Ablation vs reconstruction logit RMS: {increment.square().mean().sqrt():.6f}")
fig, axes = plt.subplots(1, 2, figsize=(10, 3), width_ratios=[3, 1])
axes[0].bar(list(predictions), changes)
axes[0].set(ylabel="Logit RMS change from native model")
axes[1].bar(["Ablation vs SAE"], [increment.square().mean().sqrt().item()])
axes[1].set(ylabel="Incremental logit RMS change")
fig.suptitle("Synthetic inputs and random LaBraM: wiring check only")
fig.tight_layout()

######################################################################
# From a wiring check to an EEG experiment
# ------------------------------------------------
#
# A random classifier's small logit changes do not demonstrate faithfulness.
# Replace it with a frozen checkpoint whose task performance was qualified on
# validation data; retain its native head and evaluate task scores on the same
# held-out subjects under all predeclared interventions. If zero-token and
# random-dictionary controls do not degrade performance, the measurement may
# be insensitive. A feature's effect is conditional on this model and layer,
# not evidence that the corresponding physiology causes a clinical outcome.
#
# For a 12-block model, choose a layer on validation rather than repeating a
# sweep on test. Quantify uncertainty at subject/recording level, not treating
# correlated tokens as independent observations. Compare seeds and inspect
# nuisance associations (site, subject and artifacts). High reconstruction
# accuracy and sparse codes alone do not establish monosemanticity.
#
# Real-data use also needs an immutable split/preprocessing/checkpoint manifest,
# a bounded activation cache, and normalization tensors stored alongside the
# SAE Lens configuration and weights. Verify reload-output parity. This example
# intentionally performs no downloads, checkpoint reuse or cache deletion;
# it reports no real-data AUROC or paper-reproduction result.
#
# References
# ------------------
#
# * `SAE Lens documentation <https://jbloomaus.github.io/SAELens/>`_: the
#   dictionary and tensor-batch trainer APIs used here (tested with 6.51.3).
# * Jiang et al., `Large Brain Model for Learning Generic Representations
#   with Tremendous EEG Data in BCI <https://arxiv.org/abs/2405.18765>`_.
