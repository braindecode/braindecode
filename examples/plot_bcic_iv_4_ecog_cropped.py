"""
Cropped Decoding on BCIC IV 2a Dataset
========================================

Building on the Trialwise decoding tutorial, we now do more data-efficient cropped decoding!

In Braindecode, there are two supported configurations created for training models: trialwise decoding and cropped decoding. We will explain this visually by comparing trialwise to cropped decoding.
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
import sklearn

from braindecode.datasets.bcicomp import BCICompetitionDataset4

subject_id = 1
dataset = BCICompetitionDataset4(subject_ids=[subject_id])
dataset = dataset.split('session')['train']

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

# crop dataset into train, valid and test
train_dataset = copy.deepcopy(dataset)
# for speed up the tarining we will use the first 10 secs for training
preprocess(train_dataset, [Preprocessor('crop', tmin=0, tmax=10)])

valid_dataset = copy.deepcopy(dataset)
# for speed up the tarining we will use the first 10 secs for training
preprocess(valid_dataset, [Preprocessor('crop', tmin=20, tmax=30)])

test_dataset = copy.deepcopy(dataset)
# for speed up the tarining we will use the first 10 secs for training
preprocess(test_dataset, [Preprocessor('crop', tmin=30, tmax=40)])


######################################################################
# Create model and compute windowing parameters
# ---------------------------------------------
#


######################################################################
# In contrast to trialwise decoding, we first have to create the model
# before we can cut the dataset into windows. This is because we need to
# know the receptive field of the network to know how large the window
# stride should be.
#


######################################################################
# We first choose the compute/input window size that will be fed to the
# network during training This has to be larger than the networks
# receptive field size and can otherwise be chosen for computational
# efficiency (see explanations in the beginning of this tutorial). Here we
# choose 1000 samples, which are 4 seconds for the 250 Hz sampling rate.
#

input_window_samples = 1000

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

n_classes = 5
# Extract number of chans and time steps from dataset
n_chans = 62

model = ShallowFBCSPNet(
    n_chans,
    n_classes,
    final_conv_length=2,
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

from braindecode.models.util import to_dense_prediction_model, get_output_shape
to_dense_prediction_model(model)

######################################################################
# To know the models’ receptive field, we calculate the shape of model
# output for a dummy input.
#

n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]
n_preds_per_input



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
from braindecode.preprocessing.windowers import create_fixed_length_windows

# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info['sfreq']
assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])


# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info['sfreq']
assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])

# Create windows using braindecode function for this. It needs parameters to define how
# trials should be used.

train_windows_dataset = create_fixed_length_windows(
    train_dataset,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=input_window_samples,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
    targets_from='channels',
    last_target_only=False,
    preload=True
)

valid_windows_dataset = create_fixed_length_windows(
    valid_dataset,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=input_window_samples,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
    targets_from='channels',
    last_target_only=False,
    preload=True
)

test_windows_dataset = create_fixed_length_windows(
    test_dataset,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=input_window_samples,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
    targets_from='channels',
    last_target_only=False,
    preload=True
)
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

######################################################################
# Training
# --------
#


######################################################################
# In difference to trialwise decoding, we now should supply
# ``cropped=True`` to the EEGClassifier, and ``CroppedLoss`` as the
# criterion, as well as ``criterion__loss_function`` as the loss function
# applied to the meaned predictions.
#


######################################################################
# .. note::
#    In this tutorial, we use some default parameters that we
#    have found to work well for motor decoding, however we strongly
#    encourage you to perform your own hyperparameter optimization using
#    cross validation on your training data.
#

from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from braindecode.training import TimeSeriesLoss
from braindecode import EEGRegressor
from braindecode.training.scoring import CroppedTimeSeriesEpochScoring

# These values we found good for shallow network:
lr = 0.0625 * 0.01
weight_decay = 0

# For deep4 they should be:
# lr = 1 * 0.01
# weight_decay = 0.5 * 0.001

batch_size = 64
n_epochs = 4

regressor = EEGRegressor(
    model,
    cropped=True,
    aggregate_predictions=False,
    criterion=TimeSeriesLoss,
    criterion__loss_function=torch.nn.functional.mse_loss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_windows_dataset),
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    iterator_train__shuffle=True,
    batch_size=batch_size,
    callbacks=[
        ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
        ('r2_train', CroppedTimeSeriesEpochScoring(sklearn.metrics.r2_score,
                                                   lower_is_better=False,
                                                   on_train=True,
                                                   name='r2_train')
         ),
        ('r2_valid', CroppedTimeSeriesEpochScoring(sklearn.metrics.r2_score,
                                                   lower_is_better=False,
                                                   on_train=False,
                                                   name='r2_valid')
         )
    ],
    device=device,
)
# Model training for a specified number of epochs. `y` is None as it is already supplied
# in the dataset.
regressor.fit(train_windows_dataset, y=None, epochs=n_epochs)

ys_valid, i_window_in_trials_valid, i_window_stops_valid  = [], [], []

for batch in valid_windows_dataset:
    ys_valid.append(batch[1])
    i_window_in_trials_valid.append(batch[2][0])
    i_window_stops_valid.append(batch[2][2])

ys_valid = np.stack(ys_valid)

ys_test, i_window_in_trials_test, i_window_stops_test  = [], [], []
for batch in test_windows_dataset:
    ys_test.append(batch[1])
    i_window_in_trials_test.append(batch[2][0])
    i_window_stops_test.append(batch[2][2])

ys_test = np.stack(ys_test)

preds_valid = regressor.predict(valid_windows_dataset)
preds_test = regressor.predict(test_windows_dataset)

import numpy as np

# Window correlation coefficient score
for i in range(ys_test.shape[1]):
    ys_cropped = ys_test[:, i, -preds_test.shape[2]:]
    mask = ~np.isnan(ys_cropped)
    ys_masked = ys_cropped[mask]
    preds_masked = preds_test[:, i, :][mask]
    print(np.corrcoef(preds_masked, ys_masked)[0, 1])

######################################################################
import matplotlib.pyplot as plt

plt.scatter(preds_masked, ys_masked)
######################################################################
# Trial-wise correlation coefficient score
from braindecode.training.scoring import trial_preds_from_window_preds

for target_ch_idx in range(ys_test.shape[1]):
    ys_cropped = ys_test[:, target_ch_idx, -preds_test.shape[2]:]

    trials_preds = trial_preds_from_window_preds(
        np.expand_dims(preds_test[:, target_ch_idx, :], 1),
        i_window_in_trials_test,
        i_window_stops_test
    )

    trials_ys = trial_preds_from_window_preds(
        np.expand_dims(ys_cropped, 1),
        i_window_in_trials_test,
        i_window_stops_test
    )

    for trial_idx, (trial_preds, trial_ys) in enumerate(zip(trials_preds, trials_ys)):
        mask = ~np.isnan(trial_ys)
        ys_masked = trial_ys[mask]
        preds_masked = trial_preds[mask]
        print(f'corr. coeff. for target {target_ch_idx} trial {trial_idx}: ',
              np.corrcoef(preds_masked, ys_masked)[0, 1])
