"""
How to train, test and tune your model
======================================

This tutorial shows you how to properly train, tune and test your deep learning
models with Braindecode. We will use the BCIC IV 2a dataset as a showcase example.

The methods shown can be applied to any standard supervised trial-based decoding setting.
This tutorial will include additional parts of code like loading and preprocessing,
defining a model, and other details which are not exclusive to this page (compare
`Cropped Decoding Tutorial <./plot_bcic_iv_2a_moabb_trial.html>`__). Therefore we
will not further elaborate on these parts and you can feel free to skip them.

In general we distinguish between "usual" training and evaluation and hyperparameter search.
The Tutorial is therefore split into two parts, one for the three different training schemes
and one for the two different hyperparameter tuning methods.

"""

######################################################################
# Why should I care about model evaluation?
# -----------------------------------------
# Short answer: To produce reliable results!
#
# In machine learning, we usually follow the scheme of splitting the
# data into two parts, training and testing sets. It sounds like a
# simple division, right? But the story does not end here.
#
# While developing a ML model you
# usually have to adjust and tune hyperparameters of your model or
# pipeline (e.g., number of layers, learning rate, number of epochs).
# Deep learning models usually have many free parameters; they could
# be considered complex models with many degrees of freedom.
# If you kept using the test dataset to evaluate your adjustment
# you would run into data leakage.
#
# This means that if you use the test set to adjust the hyperparameters
# of your model, the model implicitly learns or memorizes the test set.
# Therefore, the trained model is no longer independent of the test set
# (even though they were never used for backpropagation!).
# If you perform any Hyperparameter tuning, you need a third split,
# the so-called validation set.
#
# This tutorial shows the three basic schemes for training and evaluating
# the model as well as two methods to tune your hyperparameters.
#

######################################################################
# Loading, preprocessing, defining a model, etc.
# ----------------------------------------------
#


######################################################################
# Loading
# ~~~~~~~
#

from braindecode.datasets import MOABBDataset

subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])


######################################################################
# Preprocessing
# ~~~~~~~~~~~~~
#

import numpy as np

from braindecode.preprocessing import (
    exponential_moving_standardize, preprocess, Preprocessor)

low_cut_hz = 4.  # low cut frequency for filtering
high_cut_hz = 38.  # high cut frequency for filtering
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000

preprocessors = [
    Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
    Preprocessor(lambda data, factor: np.multiply(data, factor),  # Convert from V to uV
                 factor=1e6),
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

from braindecode.preprocessing import create_windows_from_events

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
# Create model
# ~~~~~~~~~~~~
#

import torch
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 4
# Extract number of chans and time steps from dataset
n_chans = windows_dataset[0][0].shape[0]
input_window_samples = windows_dataset[0][0].shape[1]

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
# How to train and evaluate your model
# ------------------------------------
#

######################################################################
# Split dataset into train and test
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

######################################################################
# We can easily split the dataset using additional info stored in the
# description attribute, in this case the ``session`` column. We
# select ``session_T`` for training and ``session_E`` for testing.
# For other datasets, you might have to choose another column.
#
# .. note::
#    No matter which of the three schemes you use, this initial
#    two-fold split into train_set and test_set always remains the same.
#    Remember that you are not allowed to use the test_set during any
#    stage of training or tuning.
#

splitted = windows_dataset.split('session')
train_set = splitted['session_T']
test_set = splitted['session_E']


######################################################################
# Option 1: Simple Train-Test Split
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# This is the easiest training scheme to use as the dataset is only
# split into two distinct sets (``train_set`` and ``test_set``).
# This scheme uses no separate validation split and should only be
# used for the final evaluation of the (previously!) found
# hyperparameters configuration.
#
# .. warning::
#    If you make any use of the ``test_set`` during training
#    (e.g. by using EarlyStopping) there will be data leakage
#    which will make the reported generalization capability/decoding
#    performance of your model less credible.
#

from skorch.callbacks import LRScheduler

from braindecode import EEGClassifier

lr = 0.0625 * 0.01
weight_decay = 0
batch_size = 64
n_epochs = 4

clf = EEGClassifier(
    model,
    criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.AdamW,
    train_split=None,
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

# score the Model after training
y_test = test_set.get_metadata().target
test_acc = clf.score(test_set, y=y_test)
print(f"Test acc: {(test_acc * 100):.2f}%")


######################################################################
# Option 2: Train-Val-Test Split
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

######################################################################
# When evaluating different settings hyperparameters for your model,
# there is still a risk of overfitting on the test set because the
# parameters can be tweaked until the estimator performs optimally.
# For more information visit `sklearns Cross-Validation Guide
# <https://scikit-learn.org/stable/modules/cross_validation.html>`__.
# This second option splits the original ``train_set`` into two distinct
# sets, the training set and the validation set to avoid overfitting
# the hyperparameters to the test set.
#
# .. note::
#    If your dataset is really small, the validation split can become
#    quite small. This may lead to unreliable tuning results. To
#    avoid this, either use Option 3 or adjust the split ratio.
#
# To split the ``train_set`` we will make use of the
# ``train_split`` argument of ``EEGClassifier``. If you leave this empty
# (not None!), skorch will make an 80-20 train-validation split.
# If you want to control the split manually you can do that by using
# ``Subset`` from torch and ``predefined_split`` from skorch.
#

from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from skorch.helper import predefined_split, SliceDataset

X_train = SliceDataset(train_set, idx=0)
y_train = np.array([y for y in SliceDataset(train_set, idx=1)])
train_indices, val_indices = train_test_split(X_train.indices_, test_size=0.2, shuffle=False)
train_subset = Subset(train_set, train_indices)
val_subset = Subset(train_set, val_indices)

######################################################################
# .. note::
#    The parameter ``shuffle`` is set to ``False``. For time-series
#    data this should always be the case as shuffling might take
#    advantage of correlated samples, which would make the validation
#    performance less meaningful.
#

clf = EEGClassifier(
    model,
    criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(val_subset),
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    callbacks=[
        "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ],
    device=device,
)
clf.fit(train_subset, y=None, epochs=n_epochs)

# score the Model after training (optional)
y_test = test_set.get_metadata().target
test_acc = clf.score(test_set, y=y_test)
print(f"Test acc: {(test_acc * 100):.2f}%")

######################################################################
# Option 3: k-Fold Cross Validation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

######################################################################
# As mentioned above, using only one validation split might not be
# sufficient, as there might be a shift in the data distribution.
# To compensate for this, one can run a k-fold Cross Validation,
# where every sample of the training set is in the validation set once.
# After averaging over the k validation scores afterwards, you get a
# very reliable estimate of how the model would perform on unseen
# data (test set).
#
# .. note::
#    This k-Fold Cross Validation can be used without a separate
#    (holdout) test set. If there is no test set available, e.g. in a
#    competition, this scheme is highly recommended to get a reliable
#    estimate of the generalization performance.
#
# To implement this, we will make use of sklearn function
# `cross_val_score <https://scikit-learn.org/stable/modules/generated/
# sklearn.model_selection.cross_val_score.html>`__ and the `KFold
# <https://scikit-learn.org/stable/modules/generated/sklearn.model_
# selection.KFold.html>`__. cross-validator.
# The ``train_split`` argument has to be set to ``None``, as sklearn
# will take care of the splitting.
#

from sklearn.model_selection import KFold, cross_val_score

clf = EEGClassifier(
    model,
    criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.AdamW,
    train_split=None,
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    callbacks=[
        "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ],
    device=device,
)

train_val_split = KFold(n_splits=5, shuffle=False)
fit_params = {'epochs': n_epochs}
cv_results = cross_val_score(
    clf, X_train, y_train, scoring='accuracy', cv=train_val_split, fit_params=fit_params)
print(f"Validation accuracy: {np.mean(cv_results*100):.2f}"
      f"+-{np.std(cv_results*100):.2f}%")

######################################################################
# How to tune your Hyperparameters
# --------------------------------
#

######################################################################
# One way to do hyperparameter tuning is to run each configuration
# manually (via Option 2 or 3 from above) and compare the validation
# performance afterwards. In the early stages of your developement
# process this might be sufficient to get a rough understanding of
# how your hyperparameter should look like for your model to converge.
# However, this manual tuning process quickly becomes messy as the
# number of hyperparameters you want to (jointly) tune increases.
# Therefore you sould automate this process. We will present two
# different options, analogous to Option 2 and 3 from above.
#

######################################################################
# Option 1: Train-Val-Test Split
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

######################################################################
# We will again make use of the sklearn library to do the hyperparameter
# search. `GridSearchCV <https://scikit-learn.org/stable/modules/
# generated/sklearn.model_selection.GridSearchCV.html>`__ will perform
# a Grid Search over the parameters specified in ``param_grid``.
# We use grid search as a simple example, but you can use `any strategy
# you want <https://scikit-learn.org/stable/modules/classes.html#
# module-sklearn.model_selection>`__).
#

import pandas as pd
from sklearn.model_selection import GridSearchCV

train_val_split = [tuple(train_test_split(X_train.indices_, test_size=0.2, shuffle=False))]

param_grid = {
    'optimizer__lr': [0.00625, 0.000625],
}
search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=train_val_split,
    return_train_score=True,
    scoring='accuracy',
    refit=True,
    verbose=1,
    error_score='raise'
)

search.fit(X_train, y_train, **fit_params)
search_results = pd.DataFrame(search.cv_results_)

best_run = search_results[search_results['rank_test_score'] == 1].squeeze()
print(f"Best hyperparameters were {best_run['params']} which gave a validation "
      f"accuracy of {best_run['mean_test_score'] * 100:.2f}% (training "
      f"accuracy of {best_run['mean_train_score'] * 100:.2f}%).")

######################################################################
# Option 2: k-Fold Cross Validation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

######################################################################
# To perform a full k-Fold CV just replace ``train_val_split`` from
# above with the ``KFold`` cross-validator from sklearn.

train_val_split = KFold(n_splits=5, shuffle=False)

######################################################################
# .. note::
#    You might have recognized that the accuracy got better throughout
#    the experiments of this tutorial. The reason behind that is that
#    we always use the same model with the same paramters in every
#    segment to keep the tutorial short and readable. If you do your
#    own experiments you always have to reinitialize the model before
#    training.
#
