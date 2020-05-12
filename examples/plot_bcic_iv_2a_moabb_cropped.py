"""
Cropped Decoding on BCIC IV 2a Dataset
======================================

This tutorial shows how to train a deep learning model on `MOABB BCI IV <http://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014001.html>`_
dataset in the cropped decoding setup, for details see
`Deep learning with convolutional neural networks for EEG decoding and visualization <https://arxiv.org/abs/1703.05051>`_.

In Braindecode, there are two supported configurations created for training models: trialwise decoding and cropped
decoding. We will explain this visually by comparing trialwise to cropped decoding.

.. image:: ../_static/trialwise_explanation.png
.. image:: ../_static/cropped_explanation.png

On the left, you see trialwise decoding:

1. A complete trial is pushed through the network.
2. The network produces a prediction.
3. The prediction is compared to the target (label) for that trial to compute the loss.

On the right, you see cropped decoding:

1. Instead of a complete trial, crops are pushed through the network.
2. For computational efficiency, multiple neighbouring crops are pushed through the network simultaneously (these
   neighbouring crops are called compute windows)
3. Therefore, the network produces multiple predictions (one per crop in the window)
4. The individual crop predictions are averaged before computing the loss function

Notes:

- The network architecture implicitly defines the crop size (it is the receptive field size, i.e., the number of
  timesteps the network uses to make a single prediction)
- The window size is a user-defined hyperparameter, called `input_window_samples` in Braindecode. It mostly affects runtime
  (larger window sizes should be faster). As a rule of thumb, you can set it to two times the crop size.
- Crop size and window size together define how many predictions the network makes per window: `#window−#crop+1=#predictions`

For cropped decoding, the above training setup is mathematically identical to sampling crops in your dataset, pushing
them through the network and training directly on the individual crops. At the same time, the above training setup is
much faster as it avoids redundant computations by using dilated convolutions, see our paper
`Deep learning with convolutional neural networks for EEG decoding and visualization <https://arxiv.org/abs/1703.05051>`_.
However, the two setups are only mathematically identical in case (1) your network does not use any padding and (2)
your loss function leads to the same gradients when using the averaged output. The first is true for our shallow and
deep ConvNet models and the second is true for the log-softmax outputs and negative log likelihood loss that is
typically used for classification in PyTorch.

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
from braindecode.datautil.signalproc import exponential_running_standardize
from braindecode.datautil.transforms import transform, MNETransform, \
    NumpyTransform
from braindecode.training.losses import CroppedLoss
from braindecode.models import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model, get_output_shape
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
input_window_samples = 1000  # length of trial in samples
# Parameters for exponential running standarization
factor_new = 1e-3
init_block_size = 1000

# Define parameters describing training
n_epochs = 4  # number of epochs of training
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

# For cropped decoding, we now transform the model into a model that outputs a dense
# time series of predictions. For this, we manually set the length of the finalconvolution
# layer to some length that makes the receptive field of the ConvNet smaller than the
# number of samples in a trial (see `final_conv_length=30` in the model definition).
model = ShallowFBCSPNet(
    n_chans,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length=30,
)
lr = 0.0625 * 0.01
weight_decay = 0

# Send model to GPU
if cuda:
    model.cuda()

#########################################################################################
# Prepare model for cropped decoding
# ----------------------------------
# First we transform model with strides to a model that outputs dense prediction, so we
# can use it to obtain properly predictions for all crops.
to_dense_prediction_model(model)
# We calculate the shape of model output as it depends on the input shape and model
# architecture. We save number of predictions computed per each sample by model for
# windowing function.
n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]

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

transforms = [
    MNETransform(fn='pick_types', eeg=True, meg=False, stim=False), # keep only EEG sensors
    NumpyTransform(fn=lambda x: x * 1e6), # convert from volt to mikrovolt, directly modifying the numpy array
    MNETransform(fn='filter', l_freq=low_cut_hz, h_freq=high_cut_hz), # bandpass filter
    NumpyTransform(fn=exponential_running_standardize, factor_new=factor_new,
        init_block_size=init_block_size)
]

# Transform the data
transform(dataset, transforms)


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
    window_stride_samples=n_preds_per_input,
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
#
# For cropped decoding, we have to supply EEGClassifier with `cropped=True` to modify
# behavior of model (e.g. callbacks definition). One more difference between cropped
# decoding and trialwise decoding is the `criterion` parameter specifying loss function.
# For cropped decoding, loss function has to be modified to handle multiple predictions
# made by a model.
clf = EEGClassifier(
    model,
    cropped=True,
    criterion=CroppedLoss,
    criterion__loss_function=torch.nn.functional.nll_loss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    iterator_train__shuffle=True,
    batch_size=batch_size,
    callbacks=[
        "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ],
    device=device,
)
# Model training for a specified number of epochs. `y` is None as it is already supplied
# in the dataset.
clf.fit(windows_dataset, y=None, epochs=n_epochs)

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
