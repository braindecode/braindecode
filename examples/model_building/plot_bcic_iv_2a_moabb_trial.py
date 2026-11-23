""".. _bcic-iv-2a-moabb-trial:

Basic Brain Decoding on EEG Data
========================================

This tutorial shows you how to train and test deep learning models with
Braindecode in a classical EEG setting: you have trials of data with
labels (e.g., Right Hand, Left Hand, etc.).

.. contents:: This example covers:
   :local:
   :depth: 2

"""

######################################################################
# Loading and preparing the data
# -------------------------------------
#


######################################################################
# Loading the dataset
# ~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# First, we load the data. In this tutorial, we load the BCI Competition
# IV 2a data [1]_ using braindecode's wrapper to load via
# `MOABB library <moabb_>`_ [2]_.
#
# .. note::
#    To load your own datasets either via mne or from
#    preprocessed X/y numpy arrays, see :ref:`mne-dataset-example`
#    and :ref:`custom-dataset-example`.
#

from braindecode.datasets import MOABBDataset

subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])


######################################################################
# Preprocessing
# ~~~~~~~~~~~~~
#


######################################################################
# Now we apply preprocessing like bandpass filtering to our dataset.
# You can either apply functions provided by :class:`mne.io.Raw` or
# :class:`mne.Epochs` or apply your own functions, either to the
# MNE object or the underlying numpy array.
#
# .. note::
#    Generally, braindecode prepocessing is directly applied to the loaded
#    data, and not applied on-the-fly as transformations, such as in
#    PyTorch-libraries like `<torchvision_>`_.
#

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
    Preprocessor("pick_types", eeg=True, meg=False, stim=False),  # Keep EEG sensors
    Preprocessor(lambda data: multiply(data, factor)),  # Convert from V to uV
    Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    Preprocessor(
        exponential_moving_standardize,  # Exponential moving standardization
        factor_new=factor_new,
        init_block_size=init_block_size,
    ),
]

# Transform the data
preprocess(dataset, preprocessors, n_jobs=-1)


######################################################################
# Extracting Compute Windows
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# Now we extract compute windows from the signals, these will be the inputs
# to the deep networks during training. In the case of trialwise
# decoding, we just have to decide if we want to include some part
# before and/or after the trial. For our work with this dataset,
# it was often beneficial to also include the 500 ms before the trial.
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
    preload=True,
)


######################################################################
# Splitting the dataset into training and validation sets
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# We can easily split the dataset using additional info stored in the
# description attribute, in this case ``session`` column. We select
# ``0train`` for training and ``1test`` for validation.
#

splitted = windows_dataset.split("session")
train_set = splitted["0train"]  # Session train
valid_set = splitted["1test"]  # Session evaluation


######################################################################
# Creating a model
# ----------------
#


######################################################################
# Now we create the deep learning model!
# First thing we need to do is know the properties of our signals.
# For this, we use the :func:`braindecode.datautil.infer_signal_properties` function:
#
from braindecode.datautil import infer_signal_properties

sig_props = infer_signal_properties(train_set, mode="classification")
print(sig_props)


######################################################################
# Braindecode comes with some
# predefined convolutional neural network architectures for raw
# time-domain EEG. Here, we use the :class:`ShallowFBCSPNet
# <braindecode.models.ShallowFBCSPNet>` model from [3]_. These models are
# pure `PyTorch <pytorch_>`_ deep learning models, therefore
# to use your own model, it just has to be a normal PyTorch
# :class:`torch.nn.Module`.
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

model = ShallowFBCSPNet(
    n_chans=sig_props["n_chans"],
    n_outputs=sig_props["n_outputs"],
    n_times=sig_props["n_times"],
    final_conv_length="auto",
)

# Display torchinfo table describing the model
print(model)

# Send model to GPU
if cuda:
    model = model.cuda()


######################################################################
# Model Training
# --------------
#


######################################################################
# Now we will train the network! :class:`EEGClassifier
# <braindecode.classifier.EEGClassifier>` is a Braindecode object
# responsible for managing the training of neural networks.
# It inherits from :class:`skorch.classifier.NeuralNetClassifier`,
# so the training logic is the same as in `<skorch_>`_.
#


######################################################################
# .. note::
#    In this tutorial, we use some default parameters that we
#    have found to work well for motor decoding, however we strongly
#    encourage you to perform your own hyperparameter optimization using
#    cross validation on your training data.
#

from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.helper import predefined_split

from braindecode import EEGClassifier

# We found these values to be good for the shallow network:
lr = 0.0625 * 0.01
weight_decay = 0

# For deep4 they should be:
# lr = 1 * 0.01
# weight_decay = 0.5 * 0.001

batch_size = 64
n_epochs = 4
classes = list(range(sig_props["n_outputs"]))

clf = EEGClassifier(
    model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),  # using valid_set for validation
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    callbacks=[
        "accuracy",
        ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=max(1, n_epochs - 1))),
        ("early_stopping", EarlyStopping(patience=10, load_best=True)),
    ],
    device=device,
    classes=classes,
)
# Model training for the specified number of epochs. ``y`` is ``None`` as it is
# already supplied in the dataset.
clf.fit(train_set, y=None, epochs=n_epochs)

######################################################################
# Training for longer
# -------------------
#
# The gallery build above uses only ``n_epochs = 4``. When trained
# offline for up to 100 epochs with early stopping, the model reaches
# **68.4 % accuracy on the held-out session (chance = 25 %)**.
#
# We can load the pretrained checkpoint from the Hugging Face Hub and
# inspect the full training curves. If the optional ``huggingface_hub``
# dependency is missing or the download fails, we continue with the
# locally trained short-run model.

import warnings

repo_id = "braindecode/plot_bcic_iv_2a_moabb_trial"
try:
    from huggingface_hub import hf_hub_download

    clf.initialize()
    clf.load_params(
        f_params=hf_hub_download(repo_id, "params.safetensors"),
        f_history=hf_hub_download(repo_id, "history.json"),
        use_safetensors=True,
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
# Plotting a  Confusion Matrix
# ----------------------------
#


#######################################################################
# Here we generate a confusion matrix as in [3]_.
#


from sklearn.metrics import ConfusionMatrixDisplay

y_true = valid_set.get_metadata().target
y_pred = clf.predict(valid_set)

label_dict = windows_dataset.datasets[0].window_kwargs[0][1]["mapping"]
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

#############################################################
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
# .. [2] Jayaram, Vinay, and Alexandre Barachant.
#        "MOABB: trustworthy algorithm benchmarking for BCIs."
#        Journal of neural engineering 15.6 (2018): 066011.
#
# .. [3] Schirrmeister, R.T., Springenberg, J.T., Fiederer, L.D.J., Glasstetter, M.,
#        Eggensperger, K., Tangermann, M., Hutter, F., Burgard, W. and Ball, T. (2017),
#        Deep learning with convolutional neural networks for EEG decoding and visualization.
#        Hum. Brain Mapping, 38: 5391-5420. https://onlinelibrary.wiley.com/doi/10.1002/hbm.23730.
#
# .. include:: /links.inc
