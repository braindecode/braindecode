""".. _bcic-iv-4-ecog-cropped-decoding:

Fingers flexion cropped decoding on BCIC IV 4 ECoG Dataset
==========================================================

This tutorial shows you how to train and test deep learning models with
Braindecode on ECoG BCI IV competition dataset 4 using cropped mode. For this
dataset we will predict 5 regression targets corresponding to flexion of each finger.
The targets were recorded as a time series (each 25 Hz), so this tutorial is
an example of time series target prediction.
"""

# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Mohammed Fattouh <mo.fattouh@gmail.com>
#
# License: BSD (3-clause)

######################################################################
# Loading and preparing the dataset
# ---------------------------------
#
# Loading
# ~~~~~~~
#
# First, we load the data. In this tutorial, we use the functionality of braindecode
# to load `BCI IV competition dataset 4 <http://www.bbci.de/competition/iv/#dataset4>`__.
# The dataset is available as a part of ECoG library:
# https://searchworks.stanford.edu/view/zk881ps0522
#
# The dataset contains ECoG signal and time series of 5 targets corresponding
# to each finger flexion. This is different than standard decoding setup for EEG with
# multiple trials and usually one target per trial. Here, fingers flexions change in time
# and are recorded with sampling frequency equals to 25 Hz.
#
# If this dataset is used please cite [1].
#
# [1] Miller, Kai J. "A library of human electrocorticographic data and analyses.
# "Nature human behaviour 3, no. 11 (2019): 1225-1235. https://doi.org/10.1038/s41562-019-0678-3
import copy

import numpy as np
import sklearn
from mne import set_log_level

from braindecode.datasets import BCICompetitionIVDataset4

subject_id = 1
dataset = BCICompetitionIVDataset4(subject_ids=[subject_id])

######################################################################
# Split dataset into train and test
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can easily split the dataset using additional info stored in the
# description attribute, in this case ``session`` column. We select `train` dataset
# for training and validation and `test` for final evaluation.
dataset_split = dataset.split("session")
train_set = dataset_split["train"]
test_set = dataset_split["test"]

######################################################################
# Preprocessing
# ~~~~~~~~~~~~~
#
# Now we apply preprocessing like bandpass filtering to our dataset. You
# can either apply functions provided by :class:`mne.io.Raw` or :class:`mne.Epochs`
# or apply your own functions, either to the MNE object or the underlying
# numpy array.
#
# .. note::
#    Preprocessing steps are taken from a standard EEG processing pipeline.
#    The only change is the cutoff frequency of the filter. For a proper ECoG
#    decoding other preprocessing steps may be needed.
#
# .. note::
#    These prepocessings are now directly applied to the loaded
#    data, and not on-the-fly applied as transformations in
#    PyTorch-libraries like
#    `<torchvision_>`_.
#


from braindecode.preprocessing import (
    Preprocessor,
    exponential_moving_standardize,
    preprocess,
)

low_cut_hz = 1.0  # low cut frequency for filtering
high_cut_hz = 200.0  # high cut frequency for filtering, for ECoG higher than for EEG
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000

######################################################################
# We select only first 30 seconds from the training dataset to limit time and memory
# to run this example. We split training dataset into train and validation (only 6 seconds).
valid_set = preprocess(
    copy.deepcopy(train_set), [Preprocessor("crop", tmin=24, tmax=30)], n_jobs=-1
)
preprocess(train_set, [Preprocessor("crop", tmin=0, tmax=24)], n_jobs=-1)
preprocess(test_set, [Preprocessor("crop", tmin=0, tmax=24)], n_jobs=-1)

######################################################################
# In time series targets setup, targets variables are stored in mne.Raw object as channels
# of type `misc`. Thus those channels have to be selected for further processing. However,
# many mne functions ignore `misc` channels and perform operations only on data channels
# (see `MNE's glossary on data channels <MNE-glossary-data-channels_>`_).
preprocessors = [
    # TODO: ensure that misc is not removed
    Preprocessor("pick_types", ecog=True, misc=True),
    Preprocessor(lambda x: x / 1e6, picks="ecog"),  # Convert from V to uV
    Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    Preprocessor(
        exponential_moving_standardize,  # Exponential moving standardization
        factor_new=factor_new,
        init_block_size=init_block_size,
        picks="ecog",
    ),
]
# Transform the data
preprocess(train_set, preprocessors)
preprocess(valid_set, preprocessors)
preprocess(test_set, preprocessors)

# Extract sampling frequency, check that they are same in all datasets
sfreq = train_set.datasets[0].raw.info["sfreq"]
assert all([ds.raw.info["sfreq"] == sfreq for ds in train_set.datasets])
# Extract target sampling frequency
target_sfreq = train_set.datasets[0].raw.info["temp"]["target_sfreq"]

######################################################################
# Create model
# ------------
#
# In contrast to trialwise decoding, we first have to create the model
# before we can cut the dataset into windows. This is because we need to
# know the receptive field of the network to know how large the window
# stride should be.
#
# We first choose the compute/input window size that will be fed to the
# network during training This has to be larger than the networks
# receptive field size and can otherwise be chosen for computational
# efficiency (see explanations in the beginning of this tutorial). Here we
# choose 1000 samples, which is 1 second for the 1000 Hz sampling rate.
#

n_times = 1000

######################################################################
# Now we create the deep learning model! Braindecode comes with some
# predefined convolutional neural network architectures for raw
# time-domain EEG. Here, we use the :class:`ShallowFBCSPNet
# <braindecode.models.ShallowFBCSPNet>` model from [1]_. These models are
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

n_classes = 1
# Extract number of chans and time steps from dataset
n_chans = train_set[0][0].shape[0] - 5

model = ShallowFBCSPNet(
    n_chans,
    n_classes,
    n_times=n_times,
    final_conv_length=2,
)

# Send model to GPU
if cuda:
    model.cuda()


print(model)

######################################################################
# To know the models' receptive field, we calculate the shape of model
# output for a dummy input.

n_preds_per_input = model.get_output_shape()[2]

######################################################################
# Cut Compute Windows
# ~~~~~~~~~~~~~~~~~~~
#

from braindecode.preprocessing import create_fixed_length_windows

# Create windows using braindecode function for this. It needs parameters to define how
# trials should be used.

train_set = create_fixed_length_windows(
    train_set,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=n_times,
    window_stride_samples=n_preds_per_input,
    on_last_window="overlap",
    targets_from="channels",
    last_target_only=False,
    preload=False,
)

valid_set = create_fixed_length_windows(
    valid_set,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=n_times,
    window_stride_samples=n_preds_per_input,
    on_last_window="overlap",
    targets_from="channels",
    last_target_only=False,
    preload=False,
)

test_set = create_fixed_length_windows(
    test_set,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=n_times,
    window_stride_samples=n_preds_per_input,
    on_last_window="overlap",
    targets_from="channels",
    last_target_only=False,
    preload=False,
)
######################################################################
# We select only the thumb's finger flexion to create one model per finger.
#
# .. note::
#    Methods to predict all 5 fingers flexion with the same model may be considered as well.
#    We encourage you to find your own way to use braindecode models to predict fingers flexions.
#
train_set.target_transform = lambda x: x[0:1]
valid_set.target_transform = lambda x: x[0:1]
test_set.target_transform = lambda x: x[0:1]

#####################################################################
# Training
# --------
#
# In difference to trialwise decoding, we now should supply
# ``cropped=True`` to the :class:`EEGClassifer
# <braindecode.classifier.EEGClassifier>`, and :class:`CroppedLoss
# <braindecode.training.CroppedLoss>` as the criterion,
# as well as ``criterion__loss_function`` as the loss function
# applied to the meaned predictions.
#
# .. note::
#    In this tutorial, we use some default parameters that we
#    have found to work well for EEG motor decoding, however we strongly
#    encourage you to perform your own hyperparameter optimization using
#    cross validation on your training data.
#

from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.helper import predefined_split

from braindecode import EEGRegressor
from braindecode.training import CroppedTimeSeriesEpochScoring, TimeSeriesLoss

# These values we found good for shallow network for EEG MI decoding:
lr = 0.0625 * 0.01
weight_decay = 0
batch_size = 27  # only 27 examples in train set, otherwise set to 64
n_epochs = 8

regressor = EEGRegressor(
    model,
    cropped=True,
    aggregate_predictions=False,
    criterion=TimeSeriesLoss,
    criterion__loss_function=torch.nn.functional.mse_loss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    iterator_train__shuffle=True,
    batch_size=batch_size,
    callbacks=[
        ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=max(1, n_epochs - 1))),
        (
            "r2_train",
            CroppedTimeSeriesEpochScoring(
                sklearn.metrics.r2_score,
                lower_is_better=False,
                on_train=True,
                name="r2_train",
            ),
        ),
        (
            "r2_valid",
            CroppedTimeSeriesEpochScoring(
                sklearn.metrics.r2_score,
                lower_is_better=False,
                on_train=False,
                name="r2_valid",
            ),
        ),
        ("early_stopping", EarlyStopping(patience=10, load_best=True)),
    ],
    device=device,
)
set_log_level(verbose="WARNING")

######################################################################
# Model training for a specified number of epochs. ``y`` is ``None`` as it is already supplied
# in the dataset.
regressor.fit(train_set, y=None, epochs=n_epochs)

######################################################################
# Training for longer
# -------------------
#
# The gallery build above uses only ``n_epochs = 8``. When trained
# offline on the full recording for up to 100 epochs with early stopping,
# the model reaches a test-set mean Pearson r of **0.07**.
#
# We can load the pretrained checkpoint from the Hugging Face Hub and
# inspect the full training curves:

import warnings

repo_id = "braindecode/bcic_iv_4_ecog_cropped"
try:
    from huggingface_hub import hf_hub_download

    regressor.initialize()
    regressor.load_params(
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
# Obtaining predictions and targets for the test, train, and validation dataset


def pad_and_select_predictions(preds, y):
    preds = np.pad(
        preds,
        ((0, 0), (0, 0), (y.shape[2] - preds.shape[2], 0)),
        "constant",
        constant_values=0,
    )

    mask = ~np.isnan(y[0, 0, :])
    preds = np.squeeze(preds[..., mask], 0)
    y = np.squeeze(y[..., mask], 0)
    return y.T, preds.T


preds_train, y_train = regressor.predict_trials(train_set, return_targets=True)
preds_train, y_train = pad_and_select_predictions(preds_train, y_train)

preds_valid, y_valid = regressor.predict_trials(valid_set, return_targets=True)
preds_valid, y_valid = pad_and_select_predictions(preds_valid, y_valid)

preds_test, y_test = regressor.predict_trials(test_set, return_targets=True)
preds_test, y_test = pad_and_select_predictions(preds_test, y_test)

######################################################################
# Plot Results
# ------------


######################################################################
# We plot target and predicted thumb flexion on the training, validation,
# and test sets, with the Pearson correlation of each split in its heading.
#
import matplotlib.pyplot as plt
import pandas as pd

BLUE, GREY, INK = "#0072B2", "#7F7F7F", "#14181F"
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)

splits = [
    ("A", "Training set", y_train, preds_train),
    ("B", "Validation set", y_valid, preds_valid),
    ("C", "Test set", y_test, preds_test),
]
fig, axes = plt.subplots(3, 1, figsize=(8, 7.5))
for ax, (letter, heading, y_split, preds_split) in zip(axes, splits):
    times = np.arange(y_split.shape[0]) / target_sfreq
    r = np.corrcoef(preds_split[:, 0], y_split[:, 0])[0, 1]
    ax.plot(times, y_split[:, 0], color=GREY, linewidth=1.2)
    ax.plot(times, preds_split[:, 0], color=BLUE, linewidth=1.2)
    ax.set_xlim(times[0], times[-1])
    ax.set_ylabel("Thumb flexion (a.u.)")
    ax.set_title(
        f"{letter}  {heading} (r = {r:.2f})", loc="left", fontweight="bold", color=INK
    )
axes[-1].set_xlabel("Time (s)")
for y_text, text, color in [(0.95, "Target", GREY), (0.82, "Predicted", BLUE)]:
    axes[0].text(
        0.01,
        y_text,
        text,
        transform=axes[0].transAxes,
        color=color,
        fontsize=9,
        fontweight="bold",
        va="top",
    )
fig.tight_layout()

######################################################################
# We can compute correlation coefficients for each finger
#
corr_coeffs = []
for dim in range(y_test.shape[1]):
    corr_coeffs.append(np.corrcoef(preds_test[:, dim], y_test[:, dim])[0, 1])
print("Correlation coefficient for each dimension: ", np.round(corr_coeffs, 2))

######################################################################
# Now we use the history stored by Skorch throughout training to plot
# the mean squared error loss (**A**) and the R² score (**B**) on the
# training and validation sets. An R² of 0 is what predicting the mean
# target would score.
results_columns = ["train_loss", "valid_loss", "r2_train", "r2_valid"]
df = pd.DataFrame(
    regressor.history[:, results_columns],
    columns=results_columns,
    index=regressor.history[:, "epoch"],
)

fig, (ax_loss, ax_r2) = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
for ax, columns in [
    (ax_loss, ["train_loss", "valid_loss"]),
    (ax_r2, ["r2_train", "r2_valid"]),
]:
    for column, text, color in zip(columns, ["Train", "Valid"], [GREY, BLUE]):
        ax.plot(df.index, df[column], color=color, linewidth=1.8)
        ax.annotate(
            text,
            xy=(df.index[-1], df[column].iloc[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            color=color,
            fontsize=9,
            fontweight="bold",
            va="center",
        )
    ax.grid(axis="y", color="#EAEAEA", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Epoch")
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
ax_r2.axhline(0.0, color=GREY, linestyle="--", linewidth=0.9)
ax_r2.text(
    df.index[-1], 0.0, "R² = 0", color=GREY, fontsize=8.5, ha="right", va="bottom"
)
ax_loss.set_ylabel("Mean squared error")
ax_r2.set_ylabel("R²")
ax_loss.set_title("A  Loss", loc="left", fontweight="bold", color=INK)
ax_r2.set_title("B  R² score", loc="left", fontweight="bold", color=INK)
fig.tight_layout()

#############################################################
#
#
# References
# ----------
#
# .. [1] Schirrmeister, R.T., Springenberg, J.T., Fiederer, L.D.J., Glasstetter, M.,
#        Eggensperger, K., Tangermann, M., Hutter, F., Burgard, W. and Ball, T. (2017),
#        Deep learning with convolutional neural networks for EEG decoding and visualization.
#        Hum. Brain Mapping, 38: 5391-5420. https://onlinelibrary.wiley.com/doi/10.1002/hbm.23730.
#
# .. include:: /links.inc
