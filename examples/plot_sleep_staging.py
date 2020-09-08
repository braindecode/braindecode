"""
Sleep staging on Sleep Physionet dataset
========================================

This tutorial shows how to train and test a sleep staging neural network with
Braindecode. We follow the approach of [1]_, but on the openly accessible Sleep
Physionet dataset.

References
----------
.. [1] Chambon, S., Galtier, M. N., Arnal, P. J., Wainrib, G., &
        Gramfort, A. (2018). A deep learning architecture for temporal sleep
        stage classification using multivariate and multimodal time series.
        IEEE Transactions on Neural Systems and Rehabilitation Engineering,
        26(4), 758-769.
"""
# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)


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

dataset = SleepPhysionet(
    subject_ids=[0, 1], recording_ids=[1], crop_wake_mins=30)


######################################################################
# Preprocessing
# ~~~~~~~~~~~~~
#


######################################################################
# Next, we preprocess the raw data. We apply a lowpass filter and normalize the
# data channel-wise. We omit the downsampling step of Chambon et al. (2018) as
# the Sleep Physionet data is already sampled at a lower 100 Hz.
#

from braindecode.datautil.preprocess import (
    MNEPreproc, NumpyPreproc, preprocess)

high_cut_hz = 30  # high cut frequency for filtering

preprocessors = [
    # convert from volt to microvolt, directly modifying the numpy array
    NumpyPreproc(fn=lambda x: x * 1e6),
    # bandpass filter
    MNEPreproc(fn='filter', l_freq=None, h_freq=high_cut_hz),
]

# Transform the data
preprocess(dataset, preprocessors)


######################################################################
# Extract windows
# ~~~~~~~~~~~~~~~
#


######################################################################
# We now extract windows to be used in the classification task.

from braindecode.datautil.windowers import create_windows_from_events


mapping = {  # We merge stages 3 and 4 following AASM standards.
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4
}

window_size_s = 30
sfreq = 100
window_size_samples = window_size_s * sfreq

windows_dataset = create_windows_from_events(
    dataset, trial_start_offset_samples=0, trial_stop_offset_samples=0,
    window_size_samples=window_size_samples,
    window_stride_samples=window_size_samples, preload=True, mapping=mapping)


######################################################################
# Window preprocessing
# ~~~~~~~~~~~~~~~~~~~
#


######################################################################
# We also preprocess the windows by applying channel-wise z-score normalization
# in each window.
#

from braindecode.datautil.preprocess import zscore

preprocess(windows_dataset, [MNEPreproc(fn=zscore)])


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
train_set = splitted[0]
valid_set = splitted[1]

# Print number of examples per class
print(train_set.datasets[0].windows)
print(valid_set.datasets[0].windows)


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
from braindecode.models import SleepStager

cuda = torch.cuda.is_available()  # check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True
seed = 87
# Set random seed to be able to reproduce results
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 5
# Extract number of channels and time steps from dataset
n_channels = train_set[0][0].shape[0]
input_size_samples = train_set[0][0].shape[1]

model = SleepStager(
    n_channels,
    sfreq,
    n_classes=n_classes,
    input_size_s=input_size_samples / sfreq
)

# Send model to GPU
if cuda:
    model.cuda()


######################################################################
# Training
# --------
#


######################################################################
# We can now train our network. EEGClassifier is a Braindecode object that is
# responsible for managing the training of neural networks. It inherits
# from skorch.NeuralNetClassifier, so the training logic is the same as in
# `Skorch <https://skorch.readthedocs.io/en/stable/>`__.
#


######################################################################
#    **Note**: We use different hyperparameters from Chambon et al. (2018), as
#    these hyperparameters were optimized on a different dataset (MASS SS3) and
#    with a different number of recordings. Generally speaking, it is
#    recommended to perform hyperparameter optimization if reusing this code on
#    a different dataset.
#

from skorch.helper import predefined_split
from skorch.callbacks import EpochScoring
from braindecode import EEGClassifier

lr = 5e-4
batch_size = 16
n_epochs = 5

train_bal_acc = EpochScoring(
    scoring='balanced_accuracy', on_train=True, name='train_bal_acc',
    lower_is_better=False)
valid_bal_acc = EpochScoring(
    scoring='balanced_accuracy', on_train=False, name='valid_bal_acc',
    lower_is_better=False)
callbacks = [('train_bal_acc', train_bal_acc),
             ('valid_bal_acc', valid_bal_acc)]

clf = EEGClassifier(
    model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    train_split=predefined_split(valid_set),  # using valid_set for validation
    optimizer__lr=lr,
    batch_size=batch_size,
    callbacks=callbacks,
    device=device
)
# Model training for a specified number of epochs. `y` is None as it is already
# supplied in the dataset.
clf.fit(train_set, y=None, epochs=n_epochs)


######################################################################
# Plot Results
# ------------
#


######################################################################
# We use the history stored by Skorch during training to plot the accuracy and
# loss curves.
#

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# Extract loss and accuracy values for plotting from history object
df = pd.DataFrame(clf.history.to_list())
df[['train_bal_acc', 'valid_bal_acc']] *= 100

# get percent of misclass for better visual comparison to loss
plt.style.use('seaborn')
fig, ax1 = plt.subplots(figsize=(8, 3))
df.loc[:, ['train_loss', 'valid_loss']].plot(
    ax=ax1, style=['-', ':'], marker='o', color='tab:blue', legend=False,
    fontsize=14)

ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
ax1.set_ylabel("Loss", color='tab:blue', fontsize=14)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

df.loc[:, ['train_bal_acc', 'valid_bal_acc']].plot(
    ax=ax2, style=['-', ':'], marker='o', color='tab:red', legend=False)
ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
ax2.set_ylabel('Balanced accuracy [%]', color='tab:red', fontsize=14)
ax2.set_ylim(ax2.get_ylim()[0], 85)  # make some room for legend
ax1.set_xlabel('Epoch', fontsize=14)

# where some data has already been plotted to ax
handles = []
handles.append(
    Line2D([0], [0], color='black', linewidth=1, linestyle='-', label='Train'))
handles.append(
    Line2D([0], [0], color='black', linewidth=1, linestyle=':', label='Valid'))
plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
plt.tight_layout()



######################################################################
# Finally, we also plot the confusion matrix and classification report:
#

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

y_true = valid_set.datasets[0].windows.metadata['target'].values
y_pred = clf.predict(valid_set)

print(confusion_matrix(y_true, y_pred))

print(classification_report(y_true, y_pred))
