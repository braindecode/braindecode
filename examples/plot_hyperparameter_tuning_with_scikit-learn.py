"""
Hyperparameter tuning with scikit-learn
=======================================

This tutorial shows you how to tune hyperparameters with scikit-learn
(GridSearchCV) in the setting of trialwise decoding on dataset
BCIC IV 2a.

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

from braindecode.datasets.moabb import MOABBDataset

subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])


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
    exponential_moving_standardize, preprocess, Preprocessor, scale)

low_cut_hz = 4.  # low cut frequency for filtering
high_cut_hz = 38.  # high cut frequency for filtering
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000

preprocessors = [
    Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
    Preprocessor(scale, factor=1e6, apply_on_array=True),  # Convert from V to uV
    Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                 factor_new=factor_new, init_block_size=init_block_size)
]

# Transform the data
preprocess(dataset, preprocessors)


######################################################################
# Cut Compute Windows
# ~~~~~~~~~~~~~~~~~~~
#


######################################################################
# Now we cut out compute windows, the inputs for the deep networks during
# training. In the case of trialwise decoding, we just have to decide if
# we want to cut out some part before and/or after the trial. For this
# dataset, in our work, it often was beneficial to also cut out 500 ms
# before the trial.
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


######################################################################
# We can easily split the dataset using additional info stored in the
# description attribute, in this case ``session`` column. We select
# ``session_T`` for training and ``session_E`` for evaluation.
#

splitted = windows_dataset.split('session')
train_set = splitted['session_T']
eval_set = splitted['session_E']


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

from skorch.callbacks import LRScheduler

from braindecode import EEGClassifier
batch_size = 16
n_epochs = 4

clf = EEGClassifier(
    model,
    criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.AdamW,
    optimizer__lr=[],
    batch_size=batch_size,
    train_split=None,  # train /test split is handled by GridSearchCV
    callbacks=[
        "accuracy",
        ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ],
    device=device,
)

######################################################################
# Use scikit-learn GridSearchCV to tune hyperparameters. To be able
# to do this, we slice the braindecode datasets that by default return
# a 3-tuple to return X and y, respectively.
#

######################################################################
#    **Note**: The KFold object splits the datasets based on their
#    length which corresponds to the number of compute windows. In
#    this (trialwise) example this is fine to do. In a cropped setting
#    this is not advisable since this might split compute windows
#    of a single trial into both train and valid set.
#

from sklearn.model_selection import GridSearchCV, KFold
from skorch.helper import SliceDataset
from numpy import array
import pandas as pd

train_X = SliceDataset(train_set, idx=0)
train_y = array([y for y in SliceDataset(train_set, idx=1)])
cv = KFold(n_splits=2, shuffle=True, random_state=42)

fit_params = {'epochs': n_epochs}
param_grid = {
    'optimizer__lr': [0.00625, 0.000625, 0.0000625],
}
search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=cv,
    return_train_score=True,
    scoring='accuracy',
    refit=True,
    verbose=1,
    error_score='raise'
)

search.fit(train_X, train_y, **fit_params)

search_results = pd.DataFrame(search.cv_results_)

best_run = search_results[search_results['rank_test_score'] == 1].squeeze()
print(f"Best hyperparameters were {best_run['params']} which gave a validation "
      f"accuracy of {best_run['mean_test_score']*100:.2f}% (training "
      f"accuracy of {best_run['mean_train_score']*100:.2f}%).")

eval_X = SliceDataset(eval_set, idx=0)
eval_y = SliceDataset(eval_set, idx=1)
score = search.score(eval_X, eval_y)
print(f"Eval accuracy is {score*100:.2f}%.")
