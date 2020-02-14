"""
Trialwise Decoding on BCIC IV 2a Competition Set
================================================

"""

# Authors: Maciej Sliwowski
#          Robin Tibor Schirrmeister
#
# License: BSD-3

import logging
import os.path
import sys
from collections import OrderedDict

import numpy as np
import torch
from skorch.callbacks.scoring import EpochScoring
from torch.utils.data import Dataset

from braindecode.callbacks import MaxNormConstraintCallback
from braindecode.classifier import EEGClassifier
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.datautil.signal_target import apply_to_X_y
from braindecode.datautil.signalproc import exponential_running_standardize
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.models.deep4 import Deep4Net
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.scoring import PostEpochTrainScoring
from braindecode.util import set_random_seeds

log = logging.getLogger(__name__)

data_folder = "/data/schirrmr/schirrmr/bci-competition-iv/2a-gdf/"
subject_id = 1  # 1-9
low_cut_hz = 4  # 0 or 4
model = "shallow"  # 'shallow' or 'deep'

# Set if you want to use GPU
# You can also use torch.cuda.is_available() to determine if cuda is available
# on your machine.
cuda = True

ival = [-500, 4000]
max_epochs = 100  # 1600
max_increase_epochs = 160
batch_size = 60
high_cut_hz = 38
factor_new = 1e-3
init_block_size = 1000
valid_set_fraction = 0.2


def split_into_two_sets(dataset, first_set_fraction=None, n_first_set=None):
    """
    Split set into two sets either by fraction of first set or by number
    of trials in first set.

    Parameters
    ----------
    dataset: :class:`.SignalAndTarget`
    first_set_fraction: float, optional
        Fraction of trials in first set.
    n_first_set: int, optional
        Number of trials in first set

    Returns
    -------
    first_set, second_set: :class:`.SignalAndTarget`
        The two splitted sets.
    """
    assert (first_set_fraction is None) != (
        n_first_set is None
    ), "Pass either first_set_fraction or n_first_set"
    if n_first_set is None:
        n_first_set = int(round(len(dataset.X) * first_set_fraction))
    assert n_first_set < len(dataset.X)
    first_set = apply_to_X_y(lambda a: a[:n_first_set], dataset)
    second_set = apply_to_X_y(lambda a: a[n_first_set:], dataset)
    return first_set, second_set


class EEGDataSet(Dataset):
    def __init__(self, X, y):
        self.X = X
        if self.X.ndim == 3:
            self.X = self.X[:, :, :, None]
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TrainTestSplit(object):
    def __init__(self, train_size):
        assert isinstance(train_size, (int, float))
        self.train_size = train_size

    def __call__(self, dataset, y, **kwargs):
        # can we directly use this https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        # or stick to same API
        if isinstance(self.train_size, int):
            n_train_samples = self.train_size
        else:
            n_train_samples = int(self.train_size * len(dataset))

        X, y = dataset.X, dataset.y
        return (
            EEGDataSet(X[:n_train_samples], y[:n_train_samples]),
            EEGDataSet(X[n_train_samples:], y[n_train_samples:]),
        )


train_filename = "A{:02d}T.gdf".format(subject_id)
test_filename = "A{:02d}E.gdf".format(subject_id)
train_filepath = os.path.join(data_folder, train_filename)
test_filepath = os.path.join(data_folder, test_filename)
train_label_filepath = train_filepath.replace(".gdf", ".mat")
test_label_filepath = test_filepath.replace(".gdf", ".mat")

train_loader = BCICompetition4Set2A(
    train_filepath, labels_filename=train_label_filepath
)
test_loader = BCICompetition4Set2A(
    test_filepath, labels_filename=test_label_filepath
)
train_cnt = train_loader.load()
test_cnt = test_loader.load()

# Preprocessing

train_cnt = train_cnt.drop_channels(["EOG-left", "EOG-central", "EOG-right"])
assert len(train_cnt.ch_names) == 22
# lets convert to millvolt for numerical stability of next operations
train_cnt.apply_function(fun=lambda a: a * 1e6, channel_wise=False)
train_cnt.filter(l_freq=low_cut_hz, h_freq=high_cut_hz, method='iir',
                 iir_params=dict(order=3, ftype='butter'))
train_cnt.apply_function(
    func=lambda a: exponential_running_standardize(
        a, factor_new=factor_new, init_block_size=init_block_size, eps=1e-4),
    channel_wise=False)

test_cnt = test_cnt.drop_channels(["EOG-left", "EOG-central", "EOG-right"])
assert len(test_cnt.ch_names) == 22
test_cnt.apply_function(fun=lambda a: a * 1e6, channel_wise=False)
test_cnt.filter(l_freq=low_cut_hz, h_freq=high_cut_hz, method='iir',
                iir_params=dict(order=3, ftype='butter'))
test_cnt.apply_function(func=lambda a: exponential_running_standardize(
        a, factor_new=factor_new, init_block_size=init_block_size, eps=1e-4),
    channel_wise=False)

marker_def = OrderedDict(
    [("Left Hand", [1]), ("Right Hand", [2]), ("Foot", [3]), ("Tongue", [4])]
)

train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)

train_set, valid_set = split_into_two_sets(
    train_set, first_set_fraction=1 - valid_set_fraction
)

logging.basicConfig(
    format="%(asctime)s %(levelname)s : %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)

set_random_seeds(seed=20190706, cuda=cuda)

n_classes = 4
n_chans = int(train_set.X.shape[1])
input_time_length = train_set.X.shape[2]
if model == "shallow":
    model = ShallowFBCSPNet(
        n_chans,
        n_classes,
        input_time_length=input_time_length,
        final_conv_length="auto",
    )
elif model == "deep":
    model = Deep4Net(
        n_chans,
        n_classes,
        input_time_length=input_time_length,
        final_conv_length="auto",
    )
if cuda:
    model.cuda()

clf = EEGClassifier(
    model,
    criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.Adam,
    train_split=TrainTestSplit(train_size=0.8),
    batch_size=64,
    device="cuda",
    iterator_train__drop_last=True,
    callbacks=[
        (
            "train_accuracy",
            PostEpochTrainScoring(
                "accuracy", lower_is_better=False, name="train_acc"
            ),
        ),
        (
            "valid_accuracy",
            EpochScoring(
                "accuracy",
                lower_is_better=False,
                name="valid_acc",
                on_train=False,
                use_caching=True,
            ),
        ),
        ("constraint", MaxNormConstraintCallback()),
    ],
)

clf.fit(
    np.concatenate((train_set.X, valid_set.X), axis=0),
    np.concatenate((train_set.y, valid_set.y), axis=0),
    epochs=100,
)
