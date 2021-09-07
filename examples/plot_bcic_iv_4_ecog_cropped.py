"""
Fingers flexion cropped decoding on BCIC IV 4 ECoG Dataset
==========================================================

We train a neural network to predict fingers flexion wusing cropped decoding.
"""


# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Mohammed Fattouh <mo.fattouh@gmail.com>
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
# First, we load the data. In this tutorial, we use the functionality of braindecode
# to load `BCI IV competition dataset 4
# <http://www.bbci.de/competition/iv/#dataset4>`.
# The dataset is available as a part of ECoG library:
# https://searchworks.stanford.edu/view/zk881ps0522
#
# If this dataset is used please cite [1].
#
# [1] Miller, Kai J. "A library of human electrocorticographic data and analyses.
# "Nature human behaviour 3, no. 11 (2019): 1225-1235. https://doi.org/10.1038/s41562-019-0678-3
#
# The dataset contains ECoG signal and time series of 5 targets corresponding
# to each finger flexion. This is different than standard decoding setup for EEG with
# multiple trials and usually one target per trial. Here, fingers flexions change in time
# and are recorded with sampling frequency equals to 25 Hz.
import numpy as np
import sklearn
from mne import set_log_level

from braindecode.datasets.bcicomp import BCICompetitionIVDataset4

subject_id = 1
dataset = BCICompetitionIVDataset4(subject_ids=[subject_id])


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

# We select only first 30 seconds of signal to limit time and memory to run this example.
# To obtain results on the whole datasets you should remove this line.
preprocess(dataset, [Preprocessor('crop', tmin=0, tmax=30)])

low_cut_hz = 1.  # low cut frequency for filtering
high_cut_hz = 200.  # high cut frequency for filtering, for ECoG higher than for EEG
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

# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info['sfreq']
assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
# Extract target sampling frequency
target_sfreq = dataset.datasets[0].raw.info['target_sfreq']


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
# choose 1000 samples, which is 1 second for the 1000 Hz sampling rate.
#

input_window_samples = 1000


######################################################################
# Create model
# ------------
#


######################################################################
# Now we create the deep learning model! Braindecode comes with some
# predefined convolutional neural network architectures for raw
# time-domain EEG. Here, we use the shallow ConvNet model from `Deep
# learning with convolutional neural networks for EEG decoding and
# visualization <https://arxiv.org/abs/1703.05051>`__. These models are
# pure `PyTorch <https://pytorch.org>`__ deep learning models, therefore
# to use your own model, it just has to be a normal PyTorch
# `nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__.
#

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

from braindecode.models.util import to_dense_prediction_model, get_output_shape
to_dense_prediction_model(model)


######################################################################
# To know the models’ receptive field, we calculate the shape of model
# output for a dummy input.

n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]


######################################################################
# Cut Compute Windows
# ~~~~~~~~~~~~~~~~~~~
#


from braindecode.preprocessing.windowers import create_fixed_length_windows

# Create windows using braindecode function for this. It needs parameters to define how
# trials should be used.

windows_dataset = create_fixed_length_windows(
    dataset,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=input_window_samples,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
    targets_from='channels',
    last_target_only=False,
    preload=False
)


######################################################################
# Split dataset into train and valid
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
######################################################################
# We can easily split the dataset using additional info stored in the
# description attribute, in this case ``session`` column.

subsets = windows_dataset.split('session')
train_set = subsets['train']
test_set = subsets['test']

import torch
from sklearn.model_selection import train_test_split

# We can split train dataset into training and validation datasets using
# `sklearn.model_selection.train_test_split` and `torch.utils.data.Subset`
idx_train, idx_valid = train_test_split(np.arange(len(train_set)),
                                        random_state=100,
                                        test_size=0.2,
                                        shuffle=False)
valid_set = test_set
# valid_set = torch.utils.data.Subset(train_set, idx_valid)
# train_set = torch.utils.data.Subset(train_set, idx_train)


#####################################################################
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
#    have found to work well for EEG motor decoding, however we strongly
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
n_epochs = 10

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
set_log_level(verbose='WARNING')

# Model training for a specified number of epochs. `y` is None as it is already supplied
# in the dataset.
regressor.fit(train_set, y=None, epochs=n_epochs)

# We compute predictions for each finger flexion
preds_test, y_test = regressor.predict_trials(test_set, return_targets=True)

preds_test = np.pad(preds_test,
                    ((0, 0), (0, 0), (y_test.shape[2] - preds_test.shape[2], 0)),
                    'constant',
                    constant_values=0)

mask = ~np.isnan(y_test[0, 0, :])
preds_test = np.squeeze(preds_test[..., mask], 0)
y_test = np.squeeze(y_test[..., mask], 0)


######################################################################
# Plot Results
# ------------
######################################################################
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# Now we use the history stored by Skorch throughout training to plot
# accuracy and loss curves.

# Extract loss and accuracy values for plotting from history object
results_columns = ['train_loss', 'valid_loss', 'r2_train', 'r2_valid']
df = pd.DataFrame(regressor.history[:, results_columns], columns=results_columns,
                  index=regressor.history[:, 'epoch'])

fig, ax1 = plt.subplots(figsize=(8, 3))
df.loc[:, ['train_loss', 'valid_loss']].plot(
    ax=ax1, style=['-', ':'], marker='o', color='tab:blue', legend=False, fontsize=14)

ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
ax1.set_ylabel("Loss", color='tab:blue', fontsize=14)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

df.loc[:, ['r2_train', 'r2_valid']].plot(
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

# We can compute correlation coefficients for each finger
corr_coeffs = []
for dim in range(y_test.shape[0]):
    corr_coeffs.append(
        np.corrcoef(preds_test[dim, :], y_test[dim, :])[0, 1]
    )
print('Correlation coefficient for each dimension: ', corr_coeffs)

# We plot predicted and target finger flexion
x_time = np.linspace(0, preds_test.shape[1] / sfreq, preds_test.shape[1])
fig, ax1 = plt.subplots(figsize=(8, 3))
ax1.plot(x_time, y_test[0, :], label='Target')
ax1.plot(x_time, preds_test[0, :], label='Predicted')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Finger flexion')
plt.legend()
