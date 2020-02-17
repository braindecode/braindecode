"""
Skorch Trialwise Decoding
=========================

Example using skorch trialwise decoding on a simple dataset.
"""

# Authors: Maciej Sliwowski
#          Robin Tibor Schirrmeister
#          Alexandre Gramfort
#
# License: BSD-3

import mne
import numpy as np
import torch
from mne.io import concatenate_raws
from sklearn.metrics import f1_score
from skorch.callbacks.scoring import EpochScoring
from torch import optim
from torch.utils.data import Dataset

from braindecode.classifier import EEGClassifier
from braindecode.models import ShallowFBCSPNet
from braindecode.scoring import PostEpochTrainScoring
from braindecode.util import set_random_seeds

subject_id = (
    22  # carefully cherry-picked to give nice results on such limited data :)
)
event_codes = [
    5,
    6,
    9,
    10,
    13,
    14,
]  # codes for executed and imagined hands/feet

# This will download the files if you don't have them yet,
# and then return the paths to the files.
physionet_paths = mne.datasets.eegbci.load_data(
    subject_id, event_codes, update_path=False
)

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
        return self.X[idx], self.y[idx]


train_set = EEGDataSet(X[:70], y[:70])
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
)
if cuda:
    model.cuda()

# It can use also NeuralNetClassifier
clf = EEGClassifier(
    model,
    criterion=torch.nn.NLLLoss,
    optimizer=optim.AdamW,
    train_split=TrainTestSplit(train_size=40),
    optimizer__lr=0.0625 * 0.01,
    optimizer__weight_decay=0,
    batch_size=64,
    callbacks=['accuracy', 'f1', 'roc_auc']
)
clf.fit(train_set, y=None, epochs=4)

preds = clf.predict(test_set.X)
y_true = test_set.y
f1_score(y_true, preds)
