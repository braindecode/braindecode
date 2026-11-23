""".. _sleep-staging-physionet-eldele2021:

Sleep staging on the Sleep Physionet dataset using Eldele2021
=============================================================

This tutorial shows how to train and test a sleep staging neural network with
Braindecode. We use the attention-based model from [1]_ with the time distributed approach of [2]_
to learn on sequences of EEG windows using the openly accessible Sleep Physionet dataset [3]_ [4]_.

.. raw:: html

   <iframe src="https://wandb.ai/braindecode/braindecode-tutorials/runs/ma2m9x9c"
           style="border:none;height:512px;width:100%"></iframe>

"""
# Authors: Divyesh Narayanan <divyesh.narayanan@gmail.com>
#
# License: BSD (3-clause)


######################################################################
# Loading and preprocessing the dataset
# -------------------------------------
#
# Loading
# ~~~~~~~
#
# First, we load the data using the
# :class:`braindecode.datasets.sleep_physionet.SleepPhysionet` class. We load
# three subjects with two recordings (nights) each. Subjects 0 and 1 are used
# for training and subject 2 for validation, as for the pretrained checkpoint
# loaded below.
#

from numbers import Integral

from braindecode.datasets import SleepPhysionet

subject_ids = [0, 1, 2]
train_subject_ids = [0, 1]
valid_subject_ids = [2]
dataset = SleepPhysionet(
    subject_ids=subject_ids, recording_ids=[1, 2], crop_wake_mins=30
)

######################################################################
# Preprocessing
# ~~~~~~~~~~~~~
#
# Next, we preprocess the raw data. We convert the data to microvolts and apply
# a lowpass filter.

from numpy import multiply

from braindecode.preprocessing import Preprocessor, preprocess

high_cut_hz = 30
# Factor to convert from V to uV
factor = 1e6

preprocessors = [
    Preprocessor(
        lambda data: multiply(data, factor), apply_on_array=True
    ),  # Convert from V to uV
    Preprocessor("filter", l_freq=None, h_freq=high_cut_hz),
]

# Transform the data
preprocess(dataset, preprocessors)

######################################################################
# Extract windows
# ~~~~~~~~~~~~~~~
#
# We extract 30-s windows to be used in the classification task.
# The :class:`braindecode.models.AttnSleep` model takes a
# single channel as input. Here, the Fpz-Cz channel is used as it
# was found to give better performance than using the Pz-Oz channel

from braindecode.preprocessing import create_windows_from_events

mapping = {  # We merge stages 3 and 4 following AASM standards.
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}

window_size_s = 30
sfreq = 100
window_size_samples = window_size_s * sfreq

windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=window_size_samples,
    window_stride_samples=window_size_samples,
    picks="Fpz-Cz",  # the other option is Pz-Oz,
    preload=True,
    mapping=mapping,
)

######################################################################
# Window preprocessing
# ~~~~~~~~~~~~~~~~~~~~
#
# We also preprocess the windows by applying channel-wise z-score normalization
# in each window.

from sklearn.preprocessing import scale as standard_scale

preprocess(windows_dataset, [Preprocessor(standard_scale, channel_wise=True)])

######################################################################
# Split dataset into train and valid
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We split the dataset into training and validation sets by subject, so that
# no night of the validation subject is seen during training. ``split`` takes
# the indices of the recordings, which we look up in the ``"subject"`` column
# of the description.

subjects = windows_dataset.description["subject"]
split_ids = dict(
    train=subjects.index[subjects.isin(train_subject_ids)].tolist(),
    valid=subjects.index[subjects.isin(valid_subject_ids)].tolist(),
)
splits = windows_dataset.split(split_ids)
train_set, valid_set = splits["train"], splits["valid"]

######################################################################
# Create sequence samplers
# ------------------------
#
# Following the time distributed approach of [2]_, we need to provide our
# neural network with sequences of windows, such that the embeddings of
# multiple consecutive windows can be concatenated and provided to a final
# classifier. We can achieve this by defining Sampler objects that return
# sequences of window indices.
# To simplify the example, we train the whole model end-to-end on sequences,
# rather than using the two-step approach of [2]_ (i.e. training the feature
# extractor on single windows, then freezing its weights and training the
# classifier).
#

from braindecode.samplers import SequenceSampler

n_windows = 3  # Sequences of 3 consecutive windows
n_windows_stride = 3  # Maximally overlapping sequences

train_sampler = SequenceSampler(
    train_set.get_metadata(), n_windows, n_windows_stride, randomize=True
)
valid_sampler = SequenceSampler(valid_set.get_metadata(), n_windows, n_windows_stride)

# Print number of examples per class
print("Training examples: ", len(train_sampler))
print("Validation examples: ", len(valid_sampler))

######################################################################
# We also implement a transform to extract the label of the center window of a
# sequence to use it as target.

import numpy as np


# Use label of center window in the sequence
def get_center_label(x):
    if isinstance(x, Integral):
        return x
    return x[np.ceil(len(x) / 2).astype(int)] if len(x) > 1 else x


train_set.target_transform = get_center_label
valid_set.target_transform = get_center_label

######################################################################
# Finally, since some sleep stages appear a lot more often than others (e.g.
# most of the night is spent in the N2 stage), the classes are imbalanced. To
# avoid overfitting on the more frequent classes, we compute weights that we
# will provide to the loss function when training.

from sklearn.utils import compute_class_weight

y_train = [train_set[idx][1] for idx in train_sampler]
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)

######################################################################
# Create model
# ------------
#
# We can now create the deep learning model. In this tutorial, we use the sleep
# staging architecture introduced in [1]_, which is an attention-based
# neural network. We use the time distributed version of the model, where the
# feature vectors of a sequence of windows are concatenated and passed to a
# linear layer for classification.
#

import torch
from torch import nn

from braindecode.models import AttnSleep
from braindecode.modules import TimeDistributed
from braindecode.util import set_random_seeds

cuda = torch.cuda.is_available()  # check if CUDA is available
mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
device = "cuda" if cuda else "mps" if mps else "cpu"
if cuda:
    torch.backends.cudnn.benchmark = True
# Set random seed to be able to reproduce results
set_random_seeds(seed=31, cuda=cuda)

n_classes = 5
# Extract number of channels and time steps from dataset
n_channels, input_size_samples = train_set[0][0].shape

feat_extractor = AttnSleep(
    sfreq=sfreq,
    n_outputs=n_classes,
    n_times=input_size_samples,
    drop_prob=0.3,
)
feat_extractor.final_layer = nn.Identity()  # keep the features

model = nn.Sequential(
    TimeDistributed(feat_extractor),  # apply model on each 30-s window
    nn.Sequential(  # apply linear layer on concatenated feature vectors
        nn.Flatten(start_dim=1),
        nn.Dropout(0.5),
        nn.Linear(feat_extractor.len_last_layer * n_windows, n_classes),
    ),
)

# Send model to the selected accelerator
if device != "cpu":
    model.to(device)

######################################################################
# Training
# --------
#
# We can now train our network. :class:`braindecode.EEGClassifier` is a
# braindecode object that is responsible for managing the training of neural
# networks. It inherits from :class:`skorch.NeuralNetClassifier`, so the
# training logic is the same as in
# `Skorch <https://skorch.readthedocs.io/en/stable/>`__.
#

from skorch.callbacks import (
    EarlyStopping,
    EpochScoring,
    GradientNormClipping,
    LRScheduler,
)
from skorch.helper import predefined_split

from braindecode import EEGClassifier

lr = 1e-3
batch_size = 32
n_epochs = 3

train_bal_acc = EpochScoring(
    scoring="balanced_accuracy",
    on_train=True,
    name="train_bal_acc",
    lower_is_better=False,
)
valid_bal_acc = EpochScoring(
    scoring="balanced_accuracy",
    on_train=False,
    name="valid_bal_acc",
    lower_is_better=False,
)
callbacks = [
    ("train_bal_acc", train_bal_acc),
    ("valid_bal_acc", valid_bal_acc),
    ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=max(1, n_epochs - 1))),
    ("grad_clip", GradientNormClipping(gradient_clip_value=1.0)),
    (
        "early_stopping",
        EarlyStopping(
            monitor="valid_bal_acc",
            lower_is_better=False,
            patience=20,
            load_best=True,
        ),
    ),
]

clf = EEGClassifier(
    model,
    criterion=torch.nn.CrossEntropyLoss,
    criterion__weight=torch.Tensor(class_weights).to(device),
    criterion__label_smoothing=0.1,
    optimizer=torch.optim.Adam,
    iterator_train__shuffle=False,
    iterator_train__sampler=train_sampler,
    iterator_valid__sampler=valid_sampler,
    train_split=predefined_split(valid_set),  # using valid_set for validation
    optimizer__lr=lr,
    optimizer__weight_decay=1e-3,
    batch_size=batch_size,
    callbacks=callbacks,
    device=device,
    classes=np.unique(y_train),
)
# Model training for a specified number of epochs. ``y`` is ``None`` as it is already
# supplied in the dataset.
clf.fit(train_set, y=None, epochs=n_epochs)

######################################################################
# Training for longer
# -------------------
#
# The gallery build above uses only ``n_epochs = 3``. When trained offline
# for up to 100 epochs with early stopping, the same setup reaches
# **74.6 % balanced accuracy** on the held-out recordings (chance = 20 %).
#
# The interactive training dashboard is on
# `Weights & Biases <https://wandb.ai/braindecode/braindecode-tutorials/runs/ma2m9x9c>`_,
# and the offline trainer used to produce all tutorial checkpoints is
# available as a
# `public gist <https://gist.github.com/bruAristimunha/27d74c8410fe9d0db258a03f42efa7c6>`_.
#
# We can load the pretrained checkpoint directly from the Hugging Face Hub
# and inspect the full training curves. This checkpoint uses ``params.pt``
# (not safetensors) due to the ``TimeDistributed`` wrapper:

import warnings

repo_id = "braindecode/plot_sleep_staging_eldele2021"
try:
    from huggingface_hub import hf_hub_download

    clf.initialize()
    clf.load_params(
        f_params=hf_hub_download(repo_id, "params.pt"),
        f_history=hf_hub_download(repo_id, "history.json"),
        use_safetensors=False,
    )
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
# The loaded history contains the full offline training run. Panel A shows the
# weighted cross-entropy loss and panel B the balanced accuracy on the training
# (grey) and validation (blue) recordings. With five sleep stages, a classifier
# that guesses at random reaches a balanced accuracy of 20 %, drawn as the
# dashed line.

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

BLUE, GREY, INK = "#0072B2", "#7F7F7F", "#14181F"
# MOABB, imported by ``braindecode.datasets``, sets a seaborn style when it is
# imported, so we go back to the matplotlib defaults before our own settings.
plt.rcdefaults()
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

df = pd.DataFrame(clf.history.to_list())
fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(6.8, 5.2), sharex=True)
for ax, metric in ((ax_loss, "loss"), (ax_acc, "bal_acc")):
    for split, name, color in (("train", "Train", GREY), ("valid", "Valid", BLUE)):
        values = df[f"{split}_{metric}"]
        ax.plot(df["epoch"], values, color=color, linewidth=1.8)
        ax.annotate(
            name,
            xy=(df["epoch"].iloc[-1], values.iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontweight="bold",
            va="center",
        )
    ax.grid(axis="y", color="#EAEAEA", linewidth=0.7)
    ax.set_axisbelow(True)
ax_acc.axhline(1 / n_classes, color=GREY, linestyle="--", linewidth=0.9)
ax_acc.text(
    df["epoch"].iloc[-1],
    1 / n_classes - 0.02,
    f"Chance ({100 / n_classes:.0f}%)",
    color=GREY,
    fontsize=8.5,
    ha="right",
    va="top",
)
ax_acc.set_ylim(0, 1)
ax_acc.set_yticks([0, 0.5, 1])
ax_loss.set_ylabel("Loss")
ax_acc.set_ylabel("Balanced accuracy")
ax_acc.set_xlabel("Epoch")
ax_acc.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
ax_loss.set_title("A  Cross-entropy loss", loc="left", fontweight="bold", color=INK)
ax_acc.set_title("B  Balanced accuracy", loc="left", fontweight="bold", color=INK)
fig.tight_layout()
plt.show()

######################################################################
# Confusion matrix and classification report
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Using the pretrained weights we can evaluate on the held-out validation
# subject and display the confusion matrix and per-class metrics. Each row
# is normalized by the number of expert-labelled windows of that stage, so the
# diagonal reads as the recall of each stage.
#

from sklearn.metrics import ConfusionMatrixDisplay, classification_report

stage_names = ["Wake", "N1", "N2", "N3", "REM"]  # label order of ``mapping``
y_true = [valid_set[i][1] for i in valid_sampler]
y_pred = clf.predict(valid_set)

fig, ax = plt.subplots(figsize=(4.6, 4.2))
ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    labels=[0, 1, 2, 3, 4],
    display_labels=stage_names,
    normalize="true",
    cmap="Blues",
    values_format=".2f",
    colorbar=False,
    ax=ax,
)
ax.spines[["left", "bottom"]].set_visible(False)
ax.set_xlabel("Predicted stage")
ax.set_ylabel("Expert stage")
fig.tight_layout()
plt.show()

print(classification_report(y_true, y_pred))

######################################################################
# Hypnogram
# ~~~~~~~~~
#
# Finally we compare, for the first validation recording, the expert
# annotations (A) and the predicted sleep stages (B), drawn on the same time
# axis with the clinical stage order (Wake at the top,
# N3 at the bottom). Each step is one sequence of ``n_windows`` 30-s windows.
# The predicted hypnogram follows the main sleep cycles, but it switches
# between stages more often than the expert does.

# sphinx_gallery_thumbnail_number = 3

# The validation subject has two recordings; keep the sequences of the first.
first_rec = valid_sampler.file_ids == valid_sampler.file_ids[0]
y_true_rec = np.asarray(y_true)[first_rec]
y_pred_rec = y_pred.flatten()[first_rec]

# Row of each label in the hypnogram: Wake at the top, N3 at the bottom.
stage_height = {0: 4, 4: 3, 1: 2, 2: 1, 3: 0}
step_h = n_windows_stride * window_size_s / 3600
hours = np.arange(len(y_true_rec) + 1) * step_h

fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.2), sharex=True, sharey=True)
for ax, labels, name, color, letter in (
    (axes[0], y_true_rec, "Expert", GREY, "A"),
    (axes[1], y_pred_rec, "Predicted", BLUE, "B"),
):
    heights = [stage_height[int(label)] for label in labels]
    ax.step(hours, heights + heights[-1:], where="post", color=color, linewidth=1.2)
    ax.set_yticks([4, 3, 2, 1, 0])
    ax.set_yticklabels(["Wake", "REM", "N1", "N2", "N3"])
    ax.set_ylim(-0.4, 4.4)
    ax.set_title(f"{letter}  {name}", loc="left", fontweight="bold", color=INK)
axes[1].set_xlim(0, hours[-1])
axes[1].set_xlabel("Time (h)")
fig.tight_layout()
plt.show()

###########################################################################
# References
# ----------
#
# .. [1] E. Eldele et al., "An Attention-Based Deep Learning Approach for Sleep Stage
#         Classification With Single-Channel EEG," in IEEE Transactions on Neural Systems and
#         Rehabilitation Engineering, vol. 29, pp. 809-818, 2021, doi: 10.1109/TNSRE.2021.3076234.
#
# .. [2] Chambon, S., Galtier, M., Arnal, P., Wainrib, G. and Gramfort, A.
#       (2018)A Deep Learning Architecture for Temporal Sleep Stage
#       Classification Using Multivariate and Multimodal Time Series.
#       IEEE Trans. on Neural Systems and Rehabilitation Engineering 26:
#       (758-769)
#
# .. [3] B Kemp, AH Zwinderman, B Tuk, HAC Kamphuisen, JJL Oberyé. Analysis of
#        a sleep-dependent neuronal feedback loop: the slow-wave
#        microcontinuity of the EEG. IEEE-BME 47(9):1185-1194 (2000).
#
# .. [4] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh,
#        Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. (2000)
#        PhysioBank, PhysioToolkit, and PhysioNet: Components of a New
#        Research Resource for Complex Physiologic Signals.
#        Circulation 101(23):e215-e220
#
# .. include:: /links.inc
