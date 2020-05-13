"""
Trialwise Decoding on BCIC IV 2a Dataset
========================================

This tutorial shows you how to train and test deep learning models with Braindecode.
Whole procedure is performed on
`MOABB BCI IV <http://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014001.html>`_
dataset. Braindeocde supplies infrastructure for loading, transforming and splitting all
MOABB datasets which is also briefly presented here. You can find more on this topic in
another tutorial.

This script presents a standard processing workflow using trialwise braindecode models.
It is also compatible with `PyTorch <https://pytorch.org/>`_ deep learning models created
by user. There are no additional constraints on model's implementations except being
valid PyTorch (it does not have to inherit from any additional class or implement any
additional method).

Trialwise decoding is one out of two ways of EEG signal processing implemented in
Braindecode. Main points of trialwise decoding:

1. A complete trial is pushed through the network.
2. The network produces a prediction.
3. The prediction is compared to the target (label) for that trial to compute the loss.

We supply some default parameters that we have found to work well for
motor decoding, however we strongly encourage you to perform your own hyperparameter
optimization using cross validation on your training data.
"""

# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Robin Tibor Schirrmeister <robintibor@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD-3
from functools import partial

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from braindecode import EEGClassifier
from braindecode.datasets import MOABBDataset
from braindecode.datautil import create_windows_from_events
from braindecode.datautil.preprocess import exponential_moving_standardize
from braindecode.datautil.preprocess import preprocess, MNEPreproc, \
    NumpyPreproc
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds

mne.set_log_level('ERROR')

##########################################################################################
# Script parameters definition
# ----------------------------
seed = 20200220  # random seed to make results reproducible

# Parameters describing the dataset and transformations
subject_id = 3  # 1-9
low_cut_hz = 4.  # low cut frequency for filtering
high_cut_hz = 38.  # high cut frequency for filtering
n_classes = 4  # number of classes to predict
n_chans = 22  # number of channels in the dataset
trial_start_offset_seconds = -0.5  # offset between trail start in the raw data and dataset
input_window_samples = 1125  # length of trial in samples
# Parameters for exponential running standarization
factor_new = 1e-3
init_block_size = 1000

# Define parameters describing training
n_epochs = 5  # number of epochs of training
batch_size = 64
cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True

# Set random seed to be able to reproduce results
set_random_seeds(seed=seed, cuda=cuda)

#########################################################################################
# Create model
# ------------
# Braindecode comes with some predefined convolutional neural network architectures for
# raw time-domain EEG. Here, we use the shallow ConvNet model from
# `Deep learning with convolutional neural networks for EEG decoding and visualization <https://arxiv.org/abs/1703.05051>`_.

model = ShallowFBCSPNet(
    n_chans,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length='auto',
)

lr = 0.0625 * 0.01
weight_decay = 0

# Send model to GPU
if cuda:
    model.cuda()

##########################################################################################
# Load the dataset
# --------------------------
# Load `MOABB <https://github.com/NeuroTechX/moabb>`_ dataset using Braindecode datasets
# functionalities.
dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])

##########################################################################################
# Define data preprocessing and preprocess the data
# -------------------------------------------------
# Transform steps are defined as 2 elements tuples of `(str | callable, dict)`
# If the first element is string it has to be a name of
# `mne.Raw <https://mne.tools/stable/generated/mne.io.Raw.html>`_/`mne.Epochs <https://mne.tools/0.11/generated/mne.Epochs.html#mne.Epochs>`_
# method. The second element of a tuple defines method parameters.

preprocessors = [
    MNEPreproc(fn='pick_types', eeg=True, meg=False, stim=False), # keep only EEG sensors
    NumpyPreproc(fn=lambda x: x * 1e6), # convert from volt to microvolt, directly modifying the numpy array
    MNEPreproc(fn='filter', l_freq=low_cut_hz, h_freq=high_cut_hz), # bandpass filter
    NumpyPreproc(fn=exponential_moving_standardize, factor_new=factor_new,
                 init_block_size=init_block_size)
]

# Transform the data
preprocess(dataset, preprocessors)

##########################################################################################
# Create windows from MOABB dataset
# ---------------------------------

# Extract sampling frequency from all datasets (in general they may be different for each
# dataset).
sfreqs = [ds.raw.info['sfreq'] for ds in dataset.datasets]
assert len(np.unique(sfreqs)) == 1
# Calculate the trial start offset in samples.
trial_start_offset_samples = int(trial_start_offset_seconds * sfreqs[0])

# Create windows using braindecode function for this. It needs parameters to define how
# trials should be used.
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    window_size_samples=input_window_samples,
    window_stride_samples=input_window_samples,
    drop_last_window=False,
    preload=True,
)
##########################################################################################
# Split dataset into train and valid
# ----------------------------------
# We can easily split the dataset using additional info stored in the description
# attribute, in this case `session` column. We select `session_T` for training and
# `session_E` for validation.
splitted = windows_dataset.split('session')
train_set = splitted['session_T']
valid_set = splitted['session_E']

##########################################################################################
# EEGClassifier definition and training
# -------------------------------------
# EEGClassifier is a Braindecode object responsible for managing the training of neural
# networks. It inherits from `skorch.NeuralNetClassifier`, so the training logic is the
# same as in `skorch <https://skorch.readthedocs.io/en/stable/index.html>`_.
# EEGClassifier object takes all training hyperparameters, creates all callbacks and
# performs training. Model supplied to this class has to be a PyTorch model.
clf = EEGClassifier(
    model,
    criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),  # using valid_set for validation
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    callbacks=[
        "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ],
    device=device,
)
# Model training for a specified number of epochs. `y` is None as it is already supplied
# in the dataset.
clf.fit(train_set, y=None, epochs=n_epochs)

##########################################################################################
# Plot Results
# -------------

# Extract loss and accuracy values for plotting from history object
results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,
                  index=clf.history[:, 'epoch'])

# get percent of misclass for better visual comparison to loss
df = df.assign(train_misclass=100 - 100 * df.train_accuracy,
               valid_misclass=100 - 100 * df.valid_accuracy)

plt.style.use('seaborn')
fig, ax1 = plt.subplots(figsize=(8, 3))
df.loc[:, ['train_loss', 'valid_loss']].plot(
    ax=ax1, style=['-', ':'], marker='o', color='tab:blue', legend=False, fontsize=14)

ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
ax1.set_ylabel("Loss", color='tab:blue', fontsize=14)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

df.loc[:, ['train_misclass', 'valid_misclass']].plot(
    ax=ax2, style=['-', ':'], marker='o', color='tab:red', legend=False)
ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
ax2.set_ylabel("Misclassification Rate [%]", color='tab:red', fontsize=14)
ax2.set_ylim(ax2.get_ylim()[0], 85)  # make some room for legend
ax1.set_xlabel("Epoch", fontsize=14)

# where some data has already been plotted to ax
handles = []
handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle='-', label='Train'))
handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle=':', label='Valid'))
plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
plt.tight_layout()
