"""
Fingers flexion decoding on BCIC IV 4 ECoG Dataset
========================================

This tutorial shows you how to train and test deep learning models with
Braindecode on ECoG BCI IV competition dataset 4.
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
# First, we load the data. In this tutorial, we use the functionality of braindecode
# to load `BCI IV competition dataset 4
# <http://www.bbci.de/competition/iv/#dataset4>`__ [1]
#
# To run this tutorial you have to download the dataset and test labels from:
# Test labels: http://www.bbci.de/competition/iv/results/ds4/true_labels.zip
# Data: http://www.bbci.de/competition/iv/#dataset4
#
# [1] Schalk, G., Kubanek, J., Miller, K.J., Anderson, N.R., Leuthardt, E.C.,
#     Ojemann, J.G., Limbrick, D., Moran, D.W., Gerhardt, L.A., and Wolpaw, J.R.
#     Decoding Two Dimensional Movement Trajectories Using Electrocorticographic Signals
#     in Humans, J Neural Eng, 4: 264-275, 2007.
#
# .. note::
#    To load your own datasets either via mne or from
#    preprocessed X/y numpy arrays, see `MNE Dataset
#    Tutorial <./plot_mne_dataset_example.html>`__ and `Numpy Dataset
#    Tutorial <./plot_custom_dataset_example.html>`__.
#
# This dataset contains ECoG signal and time series of 5 targets corresponding
# to each finger flexion. This is different than standard decoding setup for EEG with
# multiple trials and usually one target per trial. Here, fingers flexions change in time
# and are recorded with sampling frequency equals to 25 Hz.

DATASET_PATH = '/home/maciej/projects/braindecode/BCICIV_4_mat'

import numpy as np

from braindecode.datasets.ecog_bci_competition import EcogBCICompetition4

subject_id = 1
dataset = EcogBCICompetition4(DATASET_PATH, subject_ids=[subject_id])

######################################################################
# Preprocessing
# ~~~~~~~~~~~~~
#
######################################################################
# Now we apply preprocessing like bandpass filtering to our dataset. You
# can either apply functions provided by
# `mne.Raw <https://mne.tools/stable/generated/mne.io.Raw.html>`__ or
# `mne.Epochs <https://mne.tools/0.11/generated/mne.Epochs.html#mne.Epochs>`__
# or apply your own functions, either to the MNE object or the underlying
# numpy array.
#
# .. note::
#    These prepocessings are now directly applied to the loaded
#    data, and not on-the-fly applied as transformations in
#    PyTorch-libraries like
#    `torchvision <https://pytorch.org/docs/stable/torchvision/index.html>`__.
#

from braindecode.preprocessing.preprocess import (
    exponential_moving_standardize, preprocess, Preprocessor)

low_cut_hz = 1.  # low cut frequency for filtering
high_cut_hz = 200.  # high cut frequency for filtering, for ECoG higher than for EEG
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000

# We select only first 30 seconds of signal to limit time and memory to run this example.
# To obtain results on the whole datasets one should remove this line.
preprocess(dataset, [Preprocessor('crop', tmin=0, tmax=30)])

# In time series targets setup, targets variables are stored in mne.Raw object as channels
# of type `misc`. Thus those channels have to be selected for further processing. However,
# many mne functions ignore`misc` channels and perform operations only on data channels
# (see https://mne.tools/stable/glossary.html#term-data-channels).
preprocessors = [
    Preprocessor('pick_types', ecog=True, misc=True),
    Preprocessor(lambda x: x / 1e6, picks='ecog'),  # Convert from V to uV
    Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                 factor_new=factor_new, init_block_size=init_block_size, picks='ecog')
]

# Transform the data
preprocess(dataset, preprocessors)

# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info['sfreq']
assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
# Extract target sampling frequency
target_sfreq = dataset.datasets[0].raw.info['target_sfreq']
# ######################################################################
# Split dataset into train and valid
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ######################################################################
# We can easily split the dataset using additional info stored in the
# description attribute, in this case ``session`` column. We select `train` dataset
# for training and validation and `test` for final evaluation.

subsets = dataset.split('session')
del dataset
dataset_train = subsets['train']
dataset_test = subsets['test']
del subsets
# ######################################################################
# Cut Compute Windows
# ~~~~~~~~~~~~~~~~~~~
#
#
#
######################################################################
# Now we cut out compute windows, the inputs for the deep networks during
# training. In the case of trialwise decoding of time series targets, we just have to
# decide about length windows that will be selected from the signal preceding each target.
# We use different windowing function than in standard trialwise decoding as our targets
# are stored as target channels in mne.Raw
from braindecode.preprocessing.windowers import create_windows_from_target_channels

windows_dataset = create_windows_from_target_channels(
    dataset_train,
    window_size_samples=1000,
    preload=True,
    last_target_only=True
)

windows_dataset_test = create_windows_from_target_channels(
    dataset_test,
    window_size_samples=1000,
    last_target_only=True
)

import torch
from sklearn.model_selection import train_test_split

# We can split train dataset into training and validation datasets using
# `sklearn.model_selection.train_test_split` and `torch.utils.data.Subset`
idx_train, idx_valid = train_test_split(np.arange(len(windows_dataset)),
                                        random_state=100,
                                        test_size=0.2,
                                        shuffle=False)

train_set = torch.utils.data.Subset(windows_dataset, idx_train)
valid_set = torch.utils.data.Subset(windows_dataset, idx_valid)

# ######################################################################
# # Create model
# # ------------
# #
#
#
# ######################################################################
# # Now we create the deep learning model! Braindecode comes with some
# # predefined convolutional neural network architectures for raw
# # time-domain EEG. Here, we use the shallow ConvNet model from `Deep
# # learning with convolutional neural networks for EEG decoding and
# # visualization <https://arxiv.org/abs/1703.05051>`__. These models are
# # pure `PyTorch <https://pytorch.org>`__ deep learning models, therefore
# # to use your own model, it just has to be a normal PyTorch
# # `nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__.
# #

from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True
seed = 20200220  # random seed to make results reproducible
# Set random seed to be able to reproduce results
set_random_seeds(seed=seed, cuda=cuda)

n_out_chans = train_set[0][1].shape[0]
# Extract number of chans and time steps from dataset
n_chans = train_set[0][0].shape[0]
input_window_samples = 1000  # 1 second long windows

model = ShallowFBCSPNet(
    n_chans,
    n_out_chans,
    input_window_samples=input_window_samples,
    final_conv_length='auto',
)

# We are removing the softmax layer to make it a regression model
new_model = torch.nn.Sequential()
for name, module_ in model.named_children():
    if "softmax" in name:
        continue
    new_model.add_module(name, module_)
model = new_model

# Send model to GPU
if cuda:
    model.cuda()

# ######################################################################
# Training
# --------
#
######################################################################
# Now we train the network! EEGRegressor is a Braindecode object
# responsible for managing the training of neural networks. It inherits
# from skorch.NeuralNetRegressor, so the training logic is the same as in
# `Skorch <https://skorch.readthedocs.io/en/stable/>`__.
#
######################################################################
#    **Note**: In this tutorial, we use some default parameters that we
#    have found to work well for EEG motor decoding, however we strongly
#    encourage you to perform your own hyperparameter and preprocessing optimization using
#    cross validation on your training data.
#
from skorch.callbacks import LRScheduler, EpochScoring
from skorch.helper import predefined_split
from mne import set_log_level

from braindecode import EEGRegressor

# These values we found good for shallow network for EEG MI decoding:
lr = 0.0625 * 0.01
weight_decay = 0
batch_size = 64
n_epochs = 4


def pearson_r_score(net, dataset, y):
    preds = net.predict(dataset)
    corr_coeffs = []
    for i in range(y.shape[1]):
        corr_coeffs.append(np.corrcoef(y[:, i], preds[:, i])[0, 1])
    return np.mean(corr_coeffs)


regressor = EEGRegressor(
    model,
    criterion=torch.nn.MSELoss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),  # using valid_set for validation,
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    callbacks=[
        'r2',
        ('valid_pearson_r', EpochScoring(pearson_r_score, lower_is_better=False,
                                         name='valid_pearson_r')),
        ('train_pearson_r', EpochScoring(pearson_r_score, lower_is_better=False, on_train=True,
                                         name='train_pearson_r')),
        ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ],
    device=device,
)
# Model training for a specified number of epochs. `y` is None as it is already supplied
# in the dataset.
regressor.fit(windows_dataset, y=None, epochs=n_epochs)

old_level = set_log_level(verbose='WARNING', return_old_level=True)
preds_test = regressor.predict(windows_dataset_test)
set_log_level(verbose=old_level)

y_test = np.stack([data[1] for data in windows_dataset_test])

######################################################################
# Plot Results
# ------------
######################################################################
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# We can plot target and predicted finger flexion on the test set.
plt.style.use('seaborn')
fig, ax1 = plt.subplots(figsize=(8, 3))
ax1.plot(np.arange(0, y_test.shape[0]) / target_sfreq, y_test[:, 0], label='Target')
ax1.plot(np.arange(0, preds_test.shape[0]) / target_sfreq, preds_test[:, 0], label='Predicted')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Finger flexion')
ax1.legend()

# Now we use the history stored by Skorch throughout training to plot
# accuracy and loss curves.

# Extract loss and accuracy values for plotting from history object
results_columns = ['train_loss', 'valid_loss', 'train_pearson_r', 'valid_pearson_r']
df = pd.DataFrame(regressor.history[:, results_columns], columns=results_columns,
                  index=regressor.history[:, 'epoch'])

fig, ax1 = plt.subplots(figsize=(8, 3))
df.loc[:, ['train_loss', 'valid_loss']].plot(
    ax=ax1, style=['-', ':'], marker='o', color='tab:blue', legend=False, fontsize=14)

ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
ax1.set_ylabel("Loss", color='tab:blue', fontsize=14)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

df.loc[:, ['train_pearson_r', 'valid_pearson_r']].plot(
    ax=ax2, style=['-', ':'], marker='o', color='tab:red', legend=False)
ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
ax2.set_ylabel("Pearson correlation coefficient", color='tab:red', fontsize=14)
ax1.set_xlabel("Epoch", fontsize=14)

# where some data has already been plotted to ax
handles = []
handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle='-',
                      label='Train'))
handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle=':',
                      label='Valid'))
plt.legend(handles, [h.get_label() for h in handles], fontsize=14, loc='center right')
plt.tight_layout()
