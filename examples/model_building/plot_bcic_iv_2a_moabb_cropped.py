# -*- coding: utf-8 -*-
""".. _bcic-iv-2a-moabb-cropped:

Cropped Decoding on BCIC IV 2a Dataset
======================================

.. contents:: This example covers:
   :local:
   :depth: 2

"""

######################################################################
# Building on the :ref:`Trialwise Decoding <bcic-iv-2a-moabb-trial>`,
# we now do more data-efficient cropped decoding!
#
# In Braindecode, there are two supported configurations created for
# training models: trialwise decoding and cropped decoding. We will
# explain this visually by comparing trialwise to cropped decoding.
#
# .. image:: /_static/trialwise_explanation.png
#    :alt: Trialwise decoding
# .. image:: /_static/cropped_explanation.png
#    :alt: Cropped decoding
#
# On the left, you see trialwise decoding:
#
# 1. A complete trial is pushed through the network.
# 2. The network produces a prediction.
# 3. The prediction is compared to the target (label) for that trial to
#    compute the loss.
#
# On the right, you see cropped decoding:
#
# 1. Instead of a complete trial, crops are pushed through the network.
# 2. For computational efficiency, multiple neighbouring crops are pushed
#    through the network simultaneously (these neighbouring crops are
#    called compute windows)
# 3. Therefore, the network produces multiple predictions (one per crop in
#    the window)
# 4. The individual crop predictions are *AVERAGED* before computing the
#    loss function
#
# This averaging of predictions of small sub-windows is the key difference
# between trialwise and cropped decoding. It was introduced in [1]_ and it impact
# on the parameters of the network.
#
# It is important to note that the averaging of predictions is only done
# during training. During testing, the network is still applied to crops
# and the predictions are averaged afterwards.
#
#
# .. note::
#
#     -  The network architecture implicitly defines the crop size (it is the
#        receptive field size, i.e., the number of timesteps the network uses
#        to make a single prediction)
#     -  The window size is a user-defined hyperparameter, called
#        ``n_times`` in Braindecode. It mostly affects runtime
#        (larger window sizes should be faster). As a rule of thumb, you can
#        set it to two times the crop size.
#     -  Crop size and window size together define how many predictions the
#        network makes per window: ``#window - #crop + 1 = #predictions``
#
# .. note::
#
#     For cropped decoding, the above training setup is mathematically
#     similar to sampling crops in your dataset, pushing them through the
#     network and training directly on the individual crops. However, the
#     if their position would be randomly selected, the crops would be less
#     correlated in contrast to the neighbourhood crops selected from a window.
#     At the same time, the above training setup is much faster as it avoids
#     redundant computations by using dilated convolutions, see [2]_.
#     However, the two setups are only mathematically related in case (1)
#     your network does not use any padding or only left padding and
#     (2) your loss function leads
#     to the same gradients when using the averaged output. The first is true
#     for our shallow and deep ConvNet models and the second is true for the
#     log-softmax outputs and negative log likelihood loss that is typically
#     used for classification in PyTorch.


######################################################################
# Loading and preprocessing the dataset
# -------------------------------------
#
# Loading and preprocessing stays the same as in the
# :ref:`Trialwise decoding tutorial <bcic-iv-2a-moabb-trial>`.

from braindecode.datasets import MOABBDataset

subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])

from numpy import multiply

from braindecode.preprocessing import (
    Preprocessor,
    exponential_moving_standardize,
    preprocess,
)

low_cut_hz = 4.0  # low cut frequency for filtering
high_cut_hz = 38.0  # high cut frequency for filtering
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000
# Factor to convert from V to uV
factor = 1e6

preprocessors = [
    Preprocessor("pick_types", eeg=True, meg=False, stim=False),
    # Keep EEG sensors
    Preprocessor(lambda data: multiply(data, factor)),  # Convert from V to uV
    Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),
    # Bandpass filter
    Preprocessor(
        exponential_moving_standardize,
        # Exponential moving standardization
        factor_new=factor_new,
        init_block_size=init_block_size,
    ),
]

# Transform the data
preprocess(dataset, preprocessors, n_jobs=-1)

######################################################################
# Create model and compute windowing parameters
# ---------------------------------------------
# In contrast to trialwise decoding, we first have to create the model
# before we can cut the dataset into windows. This is because we need to
# know the neural network parameters to know how large the sub-window
# stride should be.
#
# We first choose the compute/input window size that will be fed to the
# network during training. This has to be larger than the networks
# the number of timesteps size and can otherwise be chosen for computational
# efficiency (see explanations in the beginning of this tutorial). Here we
# choose 1000 samples, which are 4 seconds for the 250 Hz sampling rate.
#

n_times = 1000

######################################################################
# Now we create the model. To enable it to be used in cropped decoding
# efficiently, we manually set the length of the final convolution layer
# to some length that makes the number of timesteps of the ConvNet smaller
# than ``n_times`` (see ``final_conv_length=30`` in the model
# definition).
#

import torch

from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = "cuda" if cuda else "cpu"
if cuda:
    torch.backends.cudnn.benchmark = True
# Set random seed to be able to roughly reproduce results
# Note that with cudnn benchmark set to True, GPU indeterminism
# may still make results substantially different between runs.
# To obtain more consistent results at the cost of increased computation time,
# you can set `cudnn_benchmark=False` in `set_random_seeds`
# or remove `torch.backends.cudnn.benchmark = True`
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 4
classes = list(range(n_classes))
# Extract number of chans from dataset
n_chans = dataset[0][0].shape[0]

model = ShallowFBCSPNet(
    n_chans,
    n_classes,
    n_times=n_times,
    final_conv_length=30,
)

# Display torchinfo table describing the model
print(model)

# Send model to GPU
if cuda:
    _ = model.cuda()

######################################################################
# And now we transform model with strides to a model that outputs dense
# prediction, so we can use it to obtain predictions for all
# crops.
#
model.to_dense_prediction_model()

######################################################################
# To know the models' output shape without the last layer, we calculate the
# shape of model output for a dummy input.
#

n_preds_per_input = model.get_output_shape()[2]

######################################################################
# Cut the data into windows
# -------------------------
# In contrast to trialwise decoding, we have to supply an explicit
# window size and window stride to the
# :func:`braindecode.preprocessing.create_windows_from_events` function.
#

from braindecode.preprocessing import create_windows_from_events

trial_start_offset_seconds = -0.5
# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info["sfreq"]
assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])

# Calculate the trial start offset in samples.
trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

# Create windows using braindecode function for this. It needs parameters to define how
# trials should be used.
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    window_size_samples=n_times,
    window_stride_samples=n_preds_per_input,
    on_last_window="overlap",
    preload=True,
)

######################################################################
# Split the dataset
# -----------------
#
# This code is the same as in trialwise decoding.
#

splitted = windows_dataset.split("session")
train_set = splitted["0train"]  # Session train
valid_set = splitted["1test"]  # Session evaluation

######################################################################
# Training
# --------
# In difference to trialwise decoding, we now should supply
# ``cropped=True`` to the :class:`EEGClassifier
# <braindecode.classifier.EEGClassifier>`, and :class:`CroppedLoss
# <braindecode.training.CroppedLoss>` as the criterion,
# as well as ``criterion__loss_function`` as the loss function
# applied to the meaned predictions.
#
# .. note::
#
#     In this tutorial, we use some default parameters that we
#     have found to work well for motor decoding, however we strongly
#     encourage you to perform your own hyperparameter optimization using
#     cross validation on your training data.
#

from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.helper import predefined_split

from braindecode import EEGClassifier
from braindecode.training import CroppedLoss

# These values we found good for shallow network:
lr = 0.0625 * 0.01
weight_decay = 0

# For deep4 they should be:
# lr = 1 * 0.01
# weight_decay = 0.5 * 0.001

batch_size = 64
n_epochs = 4

clf = EEGClassifier(
    model,
    cropped=True,
    criterion=CroppedLoss,
    criterion__loss_function=torch.nn.functional.cross_entropy,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    iterator_train__shuffle=True,
    batch_size=batch_size,
    callbacks=[
        "accuracy",
        ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=max(1, n_epochs - 1))),
        ("early_stopping", EarlyStopping(patience=10, load_best=True)),
    ],
    device=device,
    classes=classes,
)
# Model training for a specified number of epochs. ``y`` is None as it is
# already supplied in the dataset.
clf.fit(train_set, y=None, epochs=n_epochs)

######################################################################
# Training for longer
# -------------------
#
# The gallery build above uses only ``n_epochs = 4``. When trained
# offline for up to 100 epochs with early stopping, the model reaches
# **83.7 % accuracy on the held-out session (chance = 25 %)**.
#
# We can load the pretrained checkpoint from the Hugging Face Hub and
# inspect the full training curves:

import warnings

repo_id = "braindecode/plot_bcic_iv_2a_moabb_cropped"
try:
    from huggingface_hub import hf_hub_download

    # Download before initialize(), which would discard the trained model
    f_params = hf_hub_download(repo_id, "params.safetensors")
    f_history = hf_hub_download(repo_id, "history.json")
    clf.initialize()
    clf.load_params(f_params=f_params, f_history=f_history, use_safetensors=True)
except Exception as exc:
    warnings.warn(
        f"Could not load pretrained checkpoint from {repo_id} ({exc}); "
        "continuing with the locally trained short-run model.",
        stacklevel=2,
    )

######################################################################
# Plot training curves
# ~~~~~~~~~~~~~~~~~~~~
#
# .. note::
#
#     Note that we drop further in the classification error and
#     loss as in the trialwise decoding tutorial.
#

import matplotlib.pyplot as plt
import pandas as pd

BLUE, GREY, INK = "#0072B2", "#7F7F7F", "#14181F"

plt.rcdefaults()  # importing MOABB applies seaborn's "whitegrid" style globally
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
    }
)

# Extract loss and accuracy values for plotting from history object
results_columns = ["train_loss", "valid_loss", "train_accuracy", "valid_accuracy"]
df = pd.DataFrame(
    clf.history[:, results_columns],
    columns=results_columns,
    index=clf.history[:, "epoch"],
)
df[["train_accuracy", "valid_accuracy"]] *= 100  # fractions to percent


def label_curve_ends(ax, ends, min_gap=0.07):
    """Write each curve's name after its last point, keeping names apart."""
    y_min, y_max = ax.get_ylim()
    gap = min_gap * (y_max - y_min)
    (low_name, (low_y, _)), (high_name, (high_y, _)) = sorted(
        ends.items(), key=lambda item: item[1][0]
    )
    if high_y - low_y < gap:
        middle = (high_y + low_y) / 2
        low_y, high_y = middle - gap / 2, middle + gap / 2
    for name, y in [(low_name, low_y), (high_name, high_y)]:
        ax.text(
            df.index[-1] + 0.02 * len(df),
            y,
            name,
            color=ends[name][1],
            va="center",
            fontweight="bold",
        )


fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(8, 3.2), sharex=True)

# Four balanced classes: a random guess is right 25 % of the time
ax_acc.axhline(25, color=GREY, ls="--", lw=0.9)
ax_acc.text(
    df.index[-1], 23, "Chance (25%)", color=GREY, fontsize=8.5, ha="right", va="top"
)
ax_acc.set_ylim(0, 100)
ax_acc.set_yticks([0, 25, 50, 75, 100])

for ax, metric, letter, title, ylabel in [
    (ax_loss, "loss", "A", "Cross-entropy loss", "Loss"),
    (ax_acc, "accuracy", "B", "Classification accuracy", "Accuracy (%)"),
]:
    ends = {}
    for split, color in [("Train", GREY), ("Valid", BLUE)]:
        values = df[f"{split.lower()}_{metric}"]
        ax.plot(df.index, values, color=color, lw=1.8)
        ends[split] = (values.iloc[-1], color)
    label_curve_ends(ax, ends)
    ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold", color=INK)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#EAEAEA", lw=0.7)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
fig.tight_layout()

######################################################################
# Plot Confusion Matrix
# ---------------------
#
# Generate a confusion matrix as in [2]_
#

from sklearn.metrics import ConfusionMatrixDisplay

y_true = valid_set.get_metadata().target
y_pred = clf.predict(valid_set)

label_dict = valid_set.datasets[0].window_kwargs[0][1]["mapping"]
sorted_items = sorted(label_dict.items(), key=lambda kv: kv[1])
labels = [k for k, _ in sorted_items]
class_ids = [v for _, v in sorted_items]

fig, ax = plt.subplots(figsize=(4.4, 3.8))
# Each row sums to 100 %: the share of a true class given to each prediction
ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    labels=class_ids,
    display_labels=[label.replace("_", " ") for label in labels],
    normalize="true",
    values_format=".0%",
    cmap="Blues",
    colorbar=False,
    ax=ax,
)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
fig.tight_layout()

######################################################################
# .. _cropped-inference:
#
# Predict whole trials
# --------------------
#
# The confusion matrix above counts windows. Each trial is cut into
# overlapping windows, and the network gives one prediction per crop of
# each window. To look at a whole trial, we put its windows back together
# in time. Where two windows overlap, we keep the predictions of the
# earlier window.
#
# Three functions of :mod:`braindecode.training` do this. Each one takes
# the trained network, ``clf.module_``, and windows in recording order.
#
# - :func:`~braindecode.training.predict_trials` returns all trials at
#   once.
# - :func:`~braindecode.training.iter_trial_predictions` yields one trial
#   at a time, so memory holds one trial instead of all of them.
# - :func:`~braindecode.training.iter_trial_prediction_chunks` yields each
#   trial in chunks of at most ``windows_per_chunk`` windows. Its memory
#   does not grow with the trial length, which matters when one trial is a
#   whole recording.
#
# We take the first four trials of the evaluation session, so this part
# only runs inference on a few windows.

import numpy as np
from torch.utils.data import Subset

metadata = valid_set.get_metadata()
# A trial starts at each window whose index within the trial is 0
trial_starts = np.flatnonzero(metadata["i_window_in_trial"].to_numpy() == 0)
n_trials = 4
few_trials = Subset(valid_set, range(trial_starts[n_trials]))
print(f"{n_trials} trials, {len(few_trials)} windows")

######################################################################
# ``predict_trials`` stacks the trials into an array of shape
# ``(n_trials, n_classes, n_predictions)`` and returns one target per
# trial. ``clf.predict_trials(few_trials)`` gives the same result.

from braindecode.training import predict_trials

trial_preds, trial_targets = predict_trials(
    clf.module_, few_trials, batch_size=batch_size
)
print(trial_preds.shape, trial_targets)

######################################################################
# ``iter_trial_predictions`` is a generator. It puts the network in eval
# mode when it starts, and restores the training flag of every layer once
# it is exhausted or closed. We only need the first trial. A ``break`` or
# a single ``next`` would leave the generator suspended, with the network
# still in eval mode, so we wrap it in :func:`contextlib.closing`. The
# generator is closed at the end of the ``with`` block.

from contextlib import closing

from braindecode.training import iter_trial_predictions

with closing(
    iter_trial_predictions(clf.module_, few_trials, batch_size=batch_size)
) as trials:
    trial_pred, trial_target = next(trials)

######################################################################
# Each prediction comes from one crop, and we place it at the time of the
# crop's last sample. The network outputs logits, so we apply a softmax
# over the classes to plot probabilities.

crop_size = n_times - n_preds_per_input + 1  # receptive field, in samples
# Time of the last sample of each crop, in seconds after the cue
pred_times = (
    trial_start_offset_seconds
    + (crop_size - 1 + np.arange(trial_pred.shape[1])) / sfreq
)
true_class = int(trial_target)
class_names = [label.replace("_", " ") for label in labels]
colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]  # Okabe-Ito


def plot_class_probabilities(ax, times, logits):
    """Plot the probability of each class over time, named at the line's end."""
    probas = torch.from_numpy(logits).softmax(dim=0).numpy()
    # Keep the end labels at least 0.09 apart so that close lines stay readable
    label_y = probas[:, -1].copy()
    order = np.argsort(label_y)
    for below, above in zip(order[:-1], order[1:]):
        label_y[above] = max(label_y[above], label_y[below] + 0.09)
    for class_id, name in enumerate(class_names):
        is_true = class_id == true_class
        ax.plot(
            times, probas[class_id], color=colors[class_id], lw=2.5 if is_true else 1.2
        )
        ax.text(
            times[-1] + 0.03,
            label_y[class_id],
            f"{name} (true class)" if is_true else name,
            color=colors[class_id],
            va="center",
        )
    # Reference lines: the target of the true class and the chance level
    ax.axhline(1, color=colors[true_class], lw=1, ls="--")
    ax.text(times[0] + 0.03, 1.01, "target of the true class", color=colors[true_class])
    ax.axhline(1 / n_classes, color=GREY, lw=0.9, ls="--")
    ax.text(
        times[0] + 0.03,
        1 / n_classes + 0.02,
        f"Chance ({1 / n_classes:.0%})",
        color=GREY,
    )
    ax.set(xlim=(times[0], times[-1]), ylim=(0, 1.1), ylabel="Probability")
    ax.set_xlabel("Time of the crop's last sample after the cue (s)")
    ax.spines[["top", "right"]].set_visible(False)


# sphinx_gallery_thumbnail_number = 3
fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
plot_class_probabilities(ax, pred_times, trial_pred)
ax.set_title("A  First test trial", loc="left", fontweight="bold", color=INK)

######################################################################
# ``iter_trial_prediction_chunks`` cuts each trial into chunks of at most
# ``windows_per_chunk`` windows. Each chunk records its trial and its
# ``prediction_start``, the offset of its first prediction within the
# trial's predictions. This offset counts predictions. It is not a sample
# index of the recording. With ``windows_per_chunk=1``, each window of the
# trial gives one chunk.

from braindecode.training import iter_trial_prediction_chunks

with closing(
    iter_trial_prediction_chunks(
        clf.module_, few_trials, windows_per_chunk=1, batch_size=batch_size
    )
) as chunks:
    first_trial = []
    for chunk in chunks:
        if chunk.trial_index > 0:
            break
        first_trial.append(chunk)

for chunk in first_trial:
    print(
        f"trial {chunk.trial_index}, prediction_start {chunk.prediction_start}, "
        f"prediction shape {chunk.prediction.shape}"
    )
assembled = np.concatenate([chunk.prediction for chunk in first_trial], axis=1)
max_abs_diff = np.abs(assembled - trial_pred).max()
print(f"Max abs difference to iter_trial_predictions: {max_abs_diff}")

######################################################################
# We plot the same trial from its chunks and shade each chunk. Put back
# together, the chunks give the trial of panel A. The largest
# absolute difference, printed above, is 0.0 on CPU.

fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
for i_chunk, chunk in enumerate(first_trial):
    first = chunk.prediction_start
    last = first + chunk.prediction.shape[1] - 1
    # Span from half a sample before the first prediction to half after the last
    start, stop = pred_times[first] - 0.5 / sfreq, pred_times[last] + 0.5 / sfreq
    ax.axvspan(start, stop, color=["0.9", "0.96"][i_chunk % 2], lw=0)
    # Chunk name above the axes: x in seconds, y in axes coordinates
    ax.text(
        (start + stop) / 2,
        1.0,
        f"Chunk {i_chunk}",
        ha="center",
        va="bottom",
        transform=ax.get_xaxis_transform(),
    )
plot_class_probabilities(ax, pred_times, assembled)
ax.set_title(
    "B  Same trial, rebuilt from its chunks",
    loc="left",
    fontweight="bold",
    color=INK,
    pad=16,
)

######################################################################
# The API reference of each function gives the details, such as the
# accepted ``loader_kwargs`` and the shape of the returned targets.

##########################################################################
#
#
# References
# ----------
#
# .. [1] Tangermann, M., Müller, K.R., Aertsen, A., Birbaumer, N., Braun, C.,
#        Brunner, C., Leeb, R., Mehring, C., Miller, K.J., Mueller-Putz, G.
#        and Nolte, G., 2012. Review of the BCI competition IV.
#        Frontiers in neuroscience, 6, p.55.
#
# .. [2] Schirrmeister, R.T., Springenberg, J.T., Fiederer, L.D.J., Glasstetter, M.,
#        Eggensperger, K., Tangermann, M., Hutter, F., Burgard, W. and Ball, T. (2017),
#        Deep learning with convolutional neural networks for EEG decoding and visualization.
#        Hum. Brain Mapping, 38: 5391-5420. https://onlinelibrary.wiley.com/doi/10.1002/hbm.23730.
