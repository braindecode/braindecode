"""
Skorch Crop Decoding
=========================

Example using Skorch for crop decoding
"""

# Authors: Lukas Gemein
#          Robin Tibor Schirrmeister
#          Alexandre Gramfort
#          Maciej Sliwowski
#
# License: BSD-3

import numpy as np

import mne
from mne.io import concatenate_raws

import torch
from torch import optim
from torch.utils.data import Dataset

from skorch.net import NeuralNet
from skorch.callbacks.scoring import EpochScoring

from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds
from braindecode.datautil import CropsDataLoader
from braindecode.models.util import to_dense_prediction_model
from braindecode.experiments.scoring import CroppedTrialEpochScoring

subject_id = [22]  # carefully cherry-picked to give nice results on such limited data :)
event_codes = [5, 6, 9, 10, 13, 14]  # codes for executed and imagined hands/feet

# This will download the files if you don't have them yet,
# and then return the paths to the files.
physionet_paths = mne.datasets.eegbci.load_data(subject_id, event_codes)

# Load each of the files
raws = [
    mne.io.read_raw_edf(
        path, preload=True, stim_channel="auto", verbose="WARNING"
    )
    for path in physionet_paths
]

# Concatenate them
raw = concatenate_raws(raws)
del raws

# Find the events in this dataset
events, _ = mne.events_from_annotations(raw)

# Use only EEG channels
picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude="bads")

# Extract trials, only using EEG channels
epochs = mne.Epochs(
    raw,
    events,
    event_id=dict(hands_or_left=2, feet_or_right=3),
    tmin=1,
    tmax=4.1,
    proj=False,
    picks=picks,
    baseline=None,
    preload=True,
)

X = (epochs.get_data() * 1e6).astype(np.float32)
y = (epochs.events[:, 2] - 2).astype(np.int64)  # 2,3 -> 0,1
del epochs

# Set if you want to use GPU
# You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
cuda = False
set_random_seeds(seed=20170629, cuda=cuda)
n_classes = 2
in_chans = X.shape[1]


class EEGDataSet(Dataset):
    def __init__(self, X, y):
        self.X = X
        if self.X.ndim == 3:
            self.X = self.X[:, :, :, None]
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        i_trial, start, stop = idx
        return self.X[i_trial, :, start:stop], self.y[i_trial]


train_set = EEGDataSet(X, y)
test_set = EEGDataSet(X[70:], y=y[70:])


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


set_random_seeds(20200114, True)

# final_conv_length = auto ensures we only get a single output in the time dimension
model = ShallowFBCSPNet(
    in_chans=in_chans,
    n_classes=n_classes,
    input_time_length=train_set.X.shape[2],
    final_conv_length="auto",
).create_network()
to_dense_prediction_model(model)
if cuda:
    model.cuda()

input_time_length = X.shape[2]

# Perform forward pass to determine how many outputs per input
with torch.no_grad():
    dummy_input = torch.tensor(X[:1, :, :input_time_length, None], device="cpu")
    n_preds_per_input = model(dummy_input).shape[2]


class CroppedNLLLoss:
    def __call__(self, preds, targets):
        return torch.nn.functional.nll_loss(torch.mean(preds, dim=1), targets)


cropped_cb = CroppedTrialEpochScoring(
    "accuracy",
    on_train=False,
    name="valid_trial_accuracy",
    lower_is_better=False,
)

clf = NeuralNet(
    model,
    criterion=CroppedNLLLoss,
    optimizer=optim.AdamW,
    train_split=TrainTestSplit(train_size=40),
    optimizer__lr=0.0625 * 0.01,
    optimizer__weight_decay=0,
    batch_size=64,
    iterator_train=CropsDataLoader,
    iterator_valid=CropsDataLoader,
    iterator_train__input_time_length=input_time_length,
    iterator_train__n_preds_per_input=n_preds_per_input,
    iterator_valid__input_time_length=input_time_length,
    iterator_valid__n_preds_per_input=n_preds_per_input,
    callbacks=[("trial_accuracy", cropped_cb)],
)

clf.fit(train_set, y=None, epochs=4)
clf.predict(test_set)
