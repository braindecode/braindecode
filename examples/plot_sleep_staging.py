"""
Sleep staging on Sleep Physionet dataset
========================================

This tutorial shows how to train and test a sleep staging neural network with
Braindecode on the Sleep Physionet dataset.
"""


######################################################################
# Loading and preprocessing the dataset
# -------------------------------------
#

######################################################################
# Loading
# ~~~~~~~
#

######################################################################
# First, we load the data using the braindecode SleepPhysionet class. We load
# two recordings from two different individuals: we will use the first one to
# train our network and the second one to evaluate performance (as in the _MNE
# sleep staging example).
#
# .. _MNE: https://mne.tools/stable/auto_tutorials/sample-datasets/plot_sleep.html
#
# .. note::
#    To load your own datasets either via mne or from
#    preprocessed X/y numpy arrays, see `MNE Dataset
#    Tutorial <./plot_mne_dataset_example.html>`__ and `Numpy Dataset
#    Tutorial <./plot_custom_dataset_example.html>`__.
#

from braindecode.datasets.sleep_physionet import SleepPhysionet

dataset = SleepPhysionet(subject_ids=[0, 1], recording_ids=[1])


######################################################################
# Preprocessing
# ~~~~~~~~~~~~~
#


######################################################################
# Next, we preprocess the dataset. We apply a lowpass filter and normalize the
# data channel-wise.
#

from braindecode.datautil.preprocess import zscore
from braindecode.datautil.preprocess import MNEPreproc, NumpyPreproc

high_cut_hz = 30  # high cut frequency for filtering

preprocessors = [
    # convert from volt to microvolt, directly modifying the numpy array
    NumpyPreproc(fn=lambda x: x * 1e6),
    # bandpass filter
    MNEPreproc(fn='filter', h_freq=high_cut_hz),
    # channel-wise z-score normalization
    NumpyPreproc(fn=zscore)
]

# Transform the data
preprocess(dataset, preprocessors)

# XXX MAKE SURE THE DATA IS PREPROCESSED AS EXPECTED!!!


######################################################################
# Extract windows
# ~~~~~~~~~~~~~~~
#


######################################################################
# We now extract windows to be used in the classification task.

from braindecode.datautil.windowers import create_windows_from_events


mapping = {  # We merge stages 3 and 4 following AASM standards.
    'Sleep stage W': 1,
    'Sleep stage 1': 2,
    'Sleep stage 2': 3,
    'Sleep stage 3': 4,
    'Sleep stage 4': 4,
    'Sleep stage R': 5
}

windows_dataset = create_windows_from_events(
    dataset, trial_start_offset_samples=0, trial_stop_offset_samples=0,
    preload=True, mapping=mapping)


######################################################################
# Split dataset into train and valid
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

######################################################################
# We can easily split the dataset using additional info stored in the
# description attribute, in this case ``session`` column. We select
# ``session_T`` for training and ``session_E`` for validation.
#

splitted = windows_dataset.split('subject')
train_set = splitted['0']
valid_set = splitted['1']
# XXX Make sure this works!


######################################################################
# Create model
# ------------
#

######################################################################
# We can now create the deep learning model. Here, we use the sleep staging
# architecture introduced in `A deep learning architecture for temporal sleep
# stage classification using multivariate and multimodal time series
# <https://arxiv.org/abs/1707.03321>`__.
#

import torch
from braindecode.util import set_random_seeds
from braindecode.models import ChambonStagerNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True
seed = 87  # random seed to make results reproducible
# Set random seed to be able to reproduce results
set_random_seeds(seed=seed, cuda=cuda)

n_classes=4
# Extract number of chans and time steps from dataset
n_chans = train_set[0][0].shape[0]
input_window_samples = train_set[0][0].shape[1]

model = ShallowFBCSPNet(
    n_chans,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length='auto',
)

# Send model to GPU
if cuda:
    model.cuda()



######################################################################
# Training
# --------
#


######################################################################
# Now we train the network! EEGClassifier is a Braindecode object
# responsible for managing the training of neural networks. It inherits
# from skorch.NeuralNetClassifier, so the training logic is the same as in
# `Skorch <https://skorch.readthedocs.io/en/stable/>`__.
#


######################################################################
#    **Note**: In this tutorial, we use some default parameters that we
#    have found to work well for motor decoding, however we strongly
#    encourage you to perform your own hyperparameter optimization using
#    cross validation on your training data.
#

from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from braindecode import EEGClassifier
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


######################################################################
# Plot Results
# ------------
#


######################################################################
# Now we use the history stored by Skorch throughout training to plot
# accuracy and loss curves.
#

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
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
