"""
Data Augmentation on BCIC IV 2a Dataset
=======================================

This tutorial shows the application of transforms for data augmentation. It
follows the trialwise decoding example, but adds transforms to the training.
For visualization, the effect of a transform is shown.

"""

######################################################################
# Following trialwise decoding example

######################################################################
# Loading
# ~~~~~~~
#

from braindecode import EEGClassifier
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler
from braindecode.datasets.moabb import MOABBDataset

subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])

######################################################################
# Preprocessing
# ~~~~~~~~~~~~~
#

from braindecode.preprocessing.preprocess import (
    exponential_moving_standardize, preprocess, Preprocessor)

low_cut_hz = 4.  # low cut frequency for filtering
high_cut_hz = 38.  # high cut frequency for filtering
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000

preprocessors = [
    Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
    Preprocessor(lambda x: x * 1e6),  # Convert from V to uV
    Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                 factor_new=factor_new, init_block_size=init_block_size)
]

preprocess(dataset, preprocessors)

######################################################################
# Cut Compute Windows
# ~~~~~~~~~~~~~~~~~~~
#

from braindecode.preprocessing.windowers import create_windows_from_events

trial_start_offset_seconds = -0.5
# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info['sfreq']
assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
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
# Split dataset into train and valid
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

splitted = windows_dataset.split('session')
train_set = splitted['session_T']
valid_set = splitted['session_E']

######################################################################
# Defining a transform
# ~~~~~~~~~~~~~~~~~~~~
#

######################################################################
# Data can be manipulated by transforms. A transform is usually handled by a
# custom data loader, but can also be called using the forward function. Here I
# will use the forward function to get manipulated data for visualization.
#

# First, we need to define a transform.

from braindecode.augmentation import FrequencyShift
transform = FrequencyShift(
    probability=1.,  # defines the probability by which a sample is manipulated
    sfreq=sfreq,
    delta_freq_range=(10., 10.)  # -> fixed frequency shift for visualization
)

######################################################################
# Manipulating one session and visualizing the transformation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

# Next, I will manipulate one session to show the resulting frequency shift. I
# will manipulate the data of an mne Epoch to make usage of mne functions.

import torch

epochs = train_set.datasets[0].windows
epochs_tr = epochs.copy()

X = torch.Tensor(epochs_tr._data)
X_tr = transform.forward(X)
epochs_tr._data = X_tr

######################################################################
# The psd of the transformed session has now been shifted by 10 Hz, as one can
# see in the psd plot.

import mne
import matplotlib.pyplot as plt
import numpy as np

f, ax = plt.subplots()

psds, freqs = mne.time_frequency.psd_multitaper(epochs)
psds = 10. * np.log10(psds)
psds_mean = psds.mean(0).mean(0)
ax.plot(freqs, psds_mean, color='k', label='original')

psds, freqs = mne.time_frequency.psd_multitaper(epochs_tr)
psds = 10. * np.log10(psds)
psds_mean = psds.mean(0).mean(0)
ax.plot(freqs, psds_mean, color='r', label='shifted')

ax.set(title='Multitaper PSD (gradiometers)', xlabel='Frequency (Hz)',
       ylabel='Power Spectral Density (dB)')
ax.legend()
plt.show()

######################################################################
# Create model
# ------------
#

######################################################################
# The model to be trained is defined as usual.

from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True
seed = 20200220  # random seed to make results reproducible
# Set random seed to be able to reproduce results
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 4
# Extract number of chans and time steps from dataset
n_chans = train_set[0][0].shape[0]
input_window_samples = train_set[0][0].shape[1]

model = ShallowFBCSPNet(
    n_chans,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length='auto',
)

######################################################################
# Define final model
# ------------------
#

######################################################################
# To use the transformer in the training, a custom data loader is used in for
# the training. Multiple transforms can be passed to the data loader and will
# be applied on the batched data.

from braindecode.augmentation import AugmentedDataLoader, SignFlip

freq_shift = FrequencyShift(
    probability=.5,
    sfreq=sfreq,
    delta_freq_range=(3., 5.)  # the frequency shifts are sampled now
)

sign_flip = SignFlip(probability=.1)

transforms = [
    freq_shift,
    sign_flip
]

# Send model to GPU
if cuda:
    model.cuda()

######################################################################
# The model is now trained as in the trial wise example. The data laoder is
# used as the train iterator and the transforms passed as arguments.

lr = 0.0625 * 0.01
weight_decay = 0

batch_size = 64
n_epochs = 4

clf = EEGClassifier(
    model,
    iterator_train=AugmentedDataLoader,
    iterator_train__transforms=transforms,
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
