""".. _sae-interpretability-tutorial:

Sparse Autoencoders for EEG Model Interpretability
==================================================

A trained network represents more concepts than it has dimensions, by letting
them share directions and rarely co-activate. A Top-K sparse autoencoder
undoes that packing: it re-expresses each activation as a combination of a
few learned directions drawn from a much wider dictionary, so individual
features stand a chance of meaning something on their own.

Fitting one is easy. Knowing whether it kept what the model actually uses is
the hard part, and it is what most of this example is about.

We use the pretrained :class:`~braindecode.models.ShallowFBCSPNet` for
subject 3 of BCI Competition IV 2a, the same checkpoint as
:ref:`interpretability-tutorial`, and decompose its penultimate feature map:

1. Load the pretrained classifier and measure its baseline performance.
2. Capture the feature map with
   :func:`~braindecode.visualization.capture_activations` and fit a Top-K
   sparse autoencoder with
   :func:`~braindecode.visualization.fit_sparse_autoencoder`.
3. Read the dictionary's health with
   :func:`~braindecode.visualization.sae_diagnostics`.
4. Put the reconstruction back into the model with
   :func:`~braindecode.visualization.run_with_activation_substitution` and
   re-measure the task, against two controls.

.. note::

   Sparse autoencoders are usually discussed for transformer residual
   streams, but nothing in the method requires one: any activation vector
   will do. Here the vectors are the 40 spatio-temporal filter responses of
   a convolutional network, which makes the point that these utilities apply
   across braindecode's model zoo rather than only to its transformers.

.. note::

   The controls are what make step 4 mean anything. A preserved score after
   substitution is also what an insensitive measurement looks like, so the
   trained dictionary is compared against an untrained one of identical
   shape, and against ablating the layer entirely.

.. contents:: This example covers:
   :local:
   :depth: 2

"""

# Authors: Vandit Shah <shahvanditt@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from numpy import multiply
from sklearn.metrics import roc_auc_score

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
    SparseAutoencoder,
    capture_activations,
    fit_sparse_autoencoder,
    run_with_activation_substitution,
    sae_diagnostics,
)

SEED = 20240205
set_random_seeds(seed=SEED, cuda=False)

######################################################################
# Data
# ----
#
# Subject 3 of BCI Competition IV 2a, four motor-imagery classes. The
# preprocessing mirrors the recipe the published checkpoint was trained
# with: keep EEG channels, convert V to µV, bandpass 4--38 Hz around the mu
# and beta rhythms where motor imagery lives, then exponential moving
# standardisation for per-channel drift. Changing any of it would put the
# input off the distribution the weights expect.
#
# The split is cross-session: session ``"0train"`` for training and
# ``"1test"``, recorded on a different day, for evaluation. That is the
# harder regime, and deliberately so. On an easy task every condition below
# would sit at ceiling and the comparison would tell us nothing.

subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])

preprocess(
    dataset,
    [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        Preprocessor(lambda data: multiply(data, 1e6)),
        Preprocessor("filter", l_freq=4.0, h_freq=38.0),
        Preprocessor(
            exponential_moving_standardize,
            factor_new=1e-3,
            init_block_size=1000,
        ),
    ],
    n_jobs=1,
)

sfreq = dataset.datasets[0].raw.info["sfreq"]
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=int(-0.5 * sfreq),
    trial_stop_offset_samples=0,
    preload=True,
)

split_by_session = windows_dataset.split("session")
train_set, valid_set = split_by_session["0train"], split_by_session["1test"]
print(f"train {len(train_set)} trials, valid {len(valid_set)} trials")


def as_arrays(windows):
    """Stack a windows dataset into ``(X, y)`` arrays."""
    xs = np.stack([windows[i][0] for i in range(len(windows))]).astype(np.float32)
    ys = np.asarray([windows[i][1] for i in range(len(windows))], dtype=np.int64)
    return xs, ys


X_train, y_train = as_arrays(train_set)
X_valid, y_valid = as_arrays(valid_set)

######################################################################
# Load the pretrained classifier
# ------------------------------
#
# Everything downstream describes a model that performs the task, so the
# classifier has to be a real one. Rather than train here, we load the
# checkpoint braindecode publishes for this subject and preprocessing, which
# keeps the docs build short and removes any doubt about whether the model
# converged.

signal_properties = infer_signal_properties(train_set, mode="classification")
classifier = EEGClassifier(
    ShallowFBCSPNet(
        n_chans=signal_properties["n_chans"],
        n_outputs=signal_properties["n_outputs"],
        n_times=signal_properties["n_times"],
        final_conv_length="auto",
    ),
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    classes=list(range(signal_properties["n_outputs"])),
)
classifier.initialize()  # builds the module without training

repo_id = "braindecode/plot_bcic_iv_2a_moabb_trial"
classifier.load_params(
    f_params=hf_hub_download(repo_id, "params.safetensors"),
    f_history=hf_hub_download(repo_id, "history.json"),
    use_safetensors=True,
)
model = classifier.module_
model.eval()


def macro_auroc(model, x, y, layer=None, substitute_fn=None):
    """One-vs-rest macro AUROC, optionally with a layer's output replaced.

    Threshold-free, so it is unaffected by class balance.
    """
    with torch.no_grad():
        inputs = torch.from_numpy(x)
        if layer is None:
            logits = model(inputs)
        else:
            logits = run_with_activation_substitution(
                model, inputs, layer, substitute_fn
            )
        probs = logits.softmax(dim=-1).cpu().numpy()
    return roc_auc_score(y, probs, multi_class="ovr", average="macro")


baseline_valid = macro_auroc(model, X_valid, y_valid)
print(f"Baseline validation AUROC: {baseline_valid:.3f}")

######################################################################
# Capture the feature map and fit the autoencoder
# -----------------------------------------------
#
# :func:`~braindecode.visualization.capture_activations` attaches forward
# hooks, runs the model, and removes them again even if the forward pass
# raises. We tap ``model.drop``, the last module before the classification
# head, so the dictionary describes exactly what the classifier reads.
#
# Its output is ``(trials, 40, time, 1)``: forty spatio-temporal filters
# over 32 pooled time positions. The autoencoder expects the feature axis
# last and one row per sample, so the tensor is permuted and flattened into
# ``(trials × time, 40)``. Reversing that permutation is the only fiddly
# part of applying an SAE to a convolutional feature map, and the
# substitution below has to undo it exactly.
#
# Only training-split activations are used, so the evaluation stays honest.
# Features that stop firing are periodically reassigned to directions the
# dictionary reconstructs poorly, since a dead feature receives no gradient
# and cannot recover on its own; ``resample_until`` stops that before the
# end of training so the last revived features are not judged before they
# have trained.

layer = model.drop


def to_tokens(activations):
    """``(trials, features, time, 1)`` to ``(trials * time, features)``."""
    return activations.squeeze(-1).permute(0, 2, 1).reshape(-1, activations.shape[1])


train_acts = capture_activations(model, torch.from_numpy(X_train), {"layer": layer})[
    "layer"
]
train_flat = to_tokens(train_acts)

sae, history = fit_sparse_autoencoder(
    train_flat,
    expansion=4,
    k=8,
    epochs=60,
    batch_size=128,
    lr=1e-3,
    seed=SEED,
    resample_steps=100,
    resample_until=0.8,
)
print(f"{train_flat.shape[0]} tokens of {train_flat.shape[1]} dims")
print(f"{sae.n_features} dictionary features, k={sae.k}")
print(f"Final reconstruction loss: {history['loss'][-1]:.4f}")

######################################################################
# Read the diagnostics
# --------------------
#
# :func:`~braindecode.visualization.sae_diagnostics` reports three numbers:
#
# ``l0``
#     Mean active features per token. Never above ``k``, and below it when
#     fewer than ``k`` encoder outputs are positive for some tokens.
#
# ``dead``
#     Fraction of features that essentially never fire. This is a firing
#     *rate* threshold rather than a never-fired test: over a few million
#     tokens almost every feature wins the top-k competition at least once,
#     so the stricter test reports zero regardless of the dictionary's
#     actual state.
#
# ``r2``
#     Fraction of activation variance reconstructed.
#
# ``r2`` is the tempting number and the misleading one. It measures how much
# *variance* survives, which is not how much of what the model *uses*
# survives. Those come apart in both directions: a dictionary can
# reconstruct nearly all the variance while dropping the directions the
# classifier reads, and a poor reconstruction can still carry an easy label.
# That is why the next section intervenes on the model instead of trusting
# ``r2``.
#
# To make that concrete, the same diagnostics are printed for an untrained
# autoencoder of identical shape. Compare the two rows before reading on.

random_sae = SparseAutoencoder.from_config(sae.get_config())
random_sae.set_activation_normalization(sae.activation_mean, sae.activation_std)

print("trained SAE:", sae_diagnostics(sae, train_flat))
print("random SAE: ", sae_diagnostics(random_sae, train_flat))

######################################################################
# The untrained dictionary has **no** dead features, and the trained one has
# roughly half. Taken on its own ``dead`` would rank them backwards. Random
# directions all fire precisely because none of them specialise: every token
# lands somewhere in an arbitrary basis, so every feature gets its turn. A
# trained dictionary concentrates the work in the features that earn it and
# lets the rest fall silent.
#
# ``r2`` separates them correctly here, and its sign is worth noticing: the
# untrained reconstruction scores below zero, meaning it is worse than
# simply predicting the mean activation. Whether that matters for the task
# is still a different question, and only the substitution answers it.

######################################################################
# Substitute the reconstruction, against controls
# -----------------------------------------------
#
# The question is causal: with the feature map replaced by the
# autoencoder's reconstruction, does the model still do its job?
# :func:`~braindecode.visualization.run_with_activation_substitution` swaps
# the output through a forward hook and reuses the trained classifier
# unchanged. Refitting a head under each condition would let it re-adapt to
# whatever the autoencoder damaged, which flatters the result.
#
# Two controls make the comparison interpretable:
#
# *random SAE*
#     The untrained autoencoder built above with
#     :meth:`~braindecode.visualization.SparseAutoencoder.from_config`, of
#     identical shape, sparsity and normalisation, whose directions are
#     arbitrary. Note that a random top-k dictionary is still a random
#     *projection*, and projections preserve linear separability, so on an
#     easy task this control passes and tells you nothing. It is informative
#     here because cross-session motor imagery is not easy.
#
# *zeroed layer*
#     The feature map replaced by zeros. This is the floor: it shows what
#     the metric looks like when the layer contributes nothing, and confirms
#     the substitution reaches the classifier at all.


def substitute_with(module):
    """Reconstruct through ``module``, restoring the conv layout."""

    def substitute(output):
        tokens = output.squeeze(-1).permute(0, 2, 1)
        recon = module.reconstruct_activations(tokens)
        return recon.permute(0, 2, 1).unsqueeze(-1)

    return substitute


scores = {
    "baseline": baseline_valid,
    "SAE": macro_auroc(
        model, X_valid, y_valid, layer=layer, substitute_fn=substitute_with(sae)
    ),
    "random SAE": macro_auroc(
        model, X_valid, y_valid, layer=layer, substitute_fn=substitute_with(random_sae)
    ),
    "zeroed layer": macro_auroc(
        model,
        X_valid,
        y_valid,
        layer=layer,
        substitute_fn=lambda out: torch.zeros_like(out),
    ),
}

print(f"\n{'condition':14s} {'AUROC':>7s} {'vs baseline':>12s}")
for name, value in scores.items():
    print(f"{name:14s} {value:>7.3f} {value - baseline_valid:>+12.3f}")

######################################################################
# Reading the result
# ------------------
#
# The comparison that matters is the trained dictionary against the two
# controls, not against the baseline alone.
#
# A small gap for the trained autoencoder means the handful of features it
# keeps active per token span what the classifier reads. That claim only
# holds if the controls visibly fail: if random directions cost as little as
# the learned ones, the measurement cannot distinguish a good dictionary
# from a meaningless one, and a near-zero gap says more about the task than
# about the dictionary.
#
# This is a single site in a network with one representational stage, so it
# shows the workflow rather than how faithfulness varies with depth. On a
# deep transformer the same three calls are repeated per block, which is how
# an operating layer gets chosen.

fig, ax = plt.subplots(figsize=(6.0, 3.4))
names = list(scores)
values = [scores[name] for name in names]
ax.bar(
    np.arange(len(names)),
    values,
    0.6,
    color=["0.3", "tab:blue", "tab:orange", "tab:red"],
)
ax.axhline(baseline_valid, color="0.3", linestyle="--", linewidth=1)
ax.axhline(0.5, color="0.7", linestyle=":", linewidth=1)
ax.set_ylim(0.4, 1.0)
ax.set_ylabel("macro AUROC (validation)")
ax.set_xticks(np.arange(len(names)))
ax.set_xticklabels(names, fontsize="small")
ax.set_title("Substituting the penultimate feature map")
fig.tight_layout()
