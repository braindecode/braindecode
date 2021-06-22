"""
Trialwise Decoding on BCIC IV 2a Dataset
========================================

This tutorial shows you how to train and test deep learning models with
Braindecode in a classical EEG setting: you have trials of data with
labels (e.g., Right Hand, Left Hand, etc.).

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
# First, we load the data. In this tutorial, we use the functionality of
# braindecode to load datasets through
# `MOABB <https://github.com/NeuroTechX/moabb>`__ to load the BCI
# Competition IV 2a data.
#
# .. note::
#    To load your own datasets either via mne or from
#    preprocessed X/y numpy arrays, see `MNE Dataset
#    Tutorial <./plot_mne_dataset_example.html>`__ and `Numpy Dataset
#    Tutorial <./plot_custom_dataset_example.html>`__.
#
import copy

import numpy as np

from braindecode.datasets.ecog import load_bci_iv_ecog

subject_id = 1
dataset = load_bci_iv_ecog('/home/maciej/projects/braindecode/BCICIV_4_mat',
                           subject_ids=[subject_id])
dataset = dataset.split('session')['train']
dataset_before = copy.deepcopy(dataset)

from braindecode.preprocessing.preprocess import (
    exponential_moving_standardize, preprocess, Preprocessor)

low_cut_hz = 4.  # low cut frequency for filtering
high_cut_hz = 200.  # high cut frequency for filtering
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000

preprocessors = [
    # TODO: ensure that misc is not removed
    Preprocessor('pick_types', ecog=True, misc=True),
    Preprocessor(lambda x: x / 1e6, picks='ecog'),  # Convert from V to uV
    Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                 factor_new=factor_new, init_block_size=init_block_size, picks='ecog')
]

# Transform the data
preprocess(dataset, preprocessors)

# Check whether preprocessing has not affected the targets
np.testing.assert_array_equal(
    dataset.datasets[0].raw.get_data()[-5:, :],
    dataset_before.datasets[0].raw.get_data()[-5:, :]
)
del dataset_before

#
#
# ######################################################################
# # Cut Compute Windows
# # ~~~~~~~~~~~~~~~~~~~
# #
#
#
# ######################################################################
# # Now we cut out compute windows, the inputs for the deep networks during
# # training. In the case of trialwise decoding, we just have to decide if
# # we want to cut out some part before and/or after the trial. For this
# # dataset, in our work, it often was beneficial to also cut out 500 ms
# # before the trial.
# #
#
from braindecode.preprocessing.windowers import create_windows_from_target_channels

trial_start_offset_seconds = -0.5
# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info['sfreq']
assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
# Calculate the trial start offset in samples.
trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

windows_dataset = create_windows_from_target_channels(
    dataset,
    trial_start_offset_samples=1000,
    trial_stop_offset_samples=400000 - 392000,
    window_size_samples=1000,
    preload=True,
)
#
#
# ######################################################################
# # Split dataset into train and valid
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# #
#
#
# ######################################################################
# # We can easily split the dataset using additional info stored in the
# # description attribute, in this case ``session`` column. We select
# # ``session_T`` for training and ``session_E`` for validation.
# #
#
from sklearn.model_selection import train_test_split

idx_train, idx_valid = train_test_split(np.arange(len(windows_dataset)),
                                        random_state=100,
                                        test_size=0.2)
train_set = windows_dataset.subset(idx_train)
valid_set = windows_dataset.subset(idx_valid)
#
#
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

import torch
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
n_chans = 62
input_window_samples = 1000

model = ShallowFBCSPNet(
    n_chans,
    5,
    input_window_samples=input_window_samples,
    final_conv_length='auto',
)

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
# # Training
# # --------
# #
#
#
# ######################################################################
# # Now we train the network! EEGClassifier is a Braindecode object
# # responsible for managing the training of neural networks. It inherits
# # from skorch.NeuralNetClassifier, so the training logic is the same as in
# # `Skorch <https://skorch.readthedocs.io/en/stable/>`__.
# #
#
#
# ######################################################################
# #    **Note**: In this tutorial, we use some default parameters that we
# #    have found to work well for motor decoding, however we strongly
# #    encourage you to perform your own hyperparameter optimization using
# #    cross validation on your training data.
# #
#
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from braindecode import EEGRegressor

# These values we found good for shallow network:
lr = 0.0625 * 0.01
weight_decay = 0

# For deep4 they should be:
# lr = 1 * 0.01
# weight_decay = 0.5 * 0.001

batch_size = 64
n_epochs = 4

clf = EEGRegressor(
    model,
    criterion=torch.nn.MSELoss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),  # using valid_set for validation,
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    callbacks=[
        ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ],
    device=device,
)
# Model training for a specified number of epochs. `y` is None as it is already supplied
# in the dataset.
clf.fit(windows_dataset, y=None, epochs=n_epochs)

#
# ######################################################################
# # Plot Results
# # ------------
# #
#
#
# ######################################################################
# # Now we use the history stored by Skorch throughout training to plot
# # accuracy and loss curves.
# #
#
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# import pandas as pd
#
# # Extract loss and accuracy values for plotting from history object
# results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
# df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,
#                   index=clf.history[:, 'epoch'])
#
# # get percent of misclass for better visual comparison to loss
# df = df.assign(train_misclass=100 - 100 * df.train_accuracy,
#                valid_misclass=100 - 100 * df.valid_accuracy)
#
# plt.style.use('seaborn')
# fig, ax1 = plt.subplots(figsize=(8, 3))
# df.loc[:, ['train_loss', 'valid_loss']].plot(
#     ax=ax1, style=['-', ':'], marker='o', color='tab:blue', legend=False, fontsize=14)
#
# ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
# ax1.set_ylabel("Loss", color='tab:blue', fontsize=14)
#
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
# df.loc[:, ['train_misclass', 'valid_misclass']].plot(
#     ax=ax2, style=['-', ':'], marker='o', color='tab:red', legend=False)
# ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
# ax2.set_ylabel("Misclassification Rate [%]", color='tab:red', fontsize=14)
# ax2.set_ylim(ax2.get_ylim()[0], 85)  # make some room for legend
# ax1.set_xlabel("Epoch", fontsize=14)
#
# # where some data has already been plotted to ax
# handles = []
# handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle='-',
# label='Train'))
# handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle=':',
# label='Valid'))
# plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
# plt.tight_layout()