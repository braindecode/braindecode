# Authors: Robin Tibor Schirrmeister
#          Maciej Sliwowski
#
# License: BSD-3

import os.path
from collections import OrderedDict

import torch
from torch import optim
from torch.utils.data import Dataset

from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.datautil import CropsDataLoader
from braindecode.datautil.signalproc import (
    bandpass_cnt,
    exponential_running_standardize,
)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.classifier import EEGClassifier
from braindecode.scoring import (CroppedTrialEpochScoring,)
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.models.deep4 import Deep4Net
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model
from braindecode.util import set_random_seeds


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


class CroppedNLLLoss:
    """Compute NLL Loss after averaging predictions across time.
    Assumes predictions are in shape:
    n_batch size x n_classes x n_predictions (in time)"""

    def __call__(self, preds, targets):
        return torch.nn.functional.nll_loss(torch.mean(preds, dim=2), targets)


data_folder = "/data/bci_competition/"
subject_id = 1  # 1-9
low_cut_hz = 4  # 0 or 4
model = "shallow"  # 'shallow' or 'deep'
cuda = False
ival = [-500, 4000]
input_time_length = 1000
max_epochs = 5
max_increase_epochs = 80
batch_size = 60
high_cut_hz = 38
factor_new = 1e-3
init_block_size = 1000
valid_set_fraction = 0.2

train_filename = "A{:02d}T.gdf".format(subject_id)
test_filename = "A{:02d}E.gdf".format(subject_id)
train_filepath = os.path.join(data_folder, train_filename)
test_filepath = os.path.join(data_folder, test_filename)
train_label_filepath = train_filepath.replace(".gdf", ".mat")
test_label_filepath = test_filepath.replace(".gdf", ".mat")

train_loader = BCICompetition4Set2A(
    train_filepath, labels_filename=train_label_filepath
)
test_loader = BCICompetition4Set2A(test_filepath, labels_filename=test_label_filepath)
raw_train = train_loader.load()
raw_test = test_loader.load()

# Preprocessing

raw_train = raw_train.drop_channels(["EOG-left", "EOG-central", "EOG-right"])
assert len(raw_train.ch_names) == 22
# lets convert to millvolt for numerical stability of next operations
raw_train = mne_apply(lambda a: a * 1e6, raw_train)
raw_train = mne_apply(
    lambda a: bandpass_cnt(
        a, low_cut_hz, high_cut_hz, raw_train.info["sfreq"], filt_order=3, axis=1
    ),
    raw_train,
)
raw_train = mne_apply(
    lambda a: exponential_running_standardize(
        a.T, factor_new=factor_new, init_block_size=init_block_size, eps=1e-4
    ).T,
    raw_train,
)

raw_test = raw_test.drop_channels(["EOG-left", "EOG-central", "EOG-right"])
assert len(raw_test.ch_names) == 22
raw_test = mne_apply(lambda a: a * 1e6, raw_test)
raw_test = mne_apply(
    lambda a: bandpass_cnt(
        a, low_cut_hz, high_cut_hz, raw_test.info["sfreq"], filt_order=3, axis=1
    ),
    raw_test,
)
raw_test = mne_apply(
    lambda a: exponential_running_standardize(
        a.T, factor_new=factor_new, init_block_size=init_block_size, eps=1e-4
    ).T,
    raw_test,
)
marker_def = OrderedDict(
    [("Left Hand", [1]), ("Right Hand", [2],), ("Foot", [3]), ("Tongue", [4])]
)

train_set = create_signal_target_from_raw_mne(raw_train, marker_def, ival)
test_set = create_signal_target_from_raw_mne(raw_test, marker_def, ival)

train_set = EEGDataSet(train_set.X, train_set.y)
test_set = EEGDataSet(test_set.X, test_set.y)

set_random_seeds(seed=20190706, cuda=cuda)

n_classes = 4
n_chans = int(train_set.X.shape[1])
if model == "shallow":
    model = ShallowFBCSPNet(
        n_chans, n_classes, input_time_length=input_time_length, final_conv_length=30
    )
elif model == "deep":
    model = Deep4Net(
        n_chans, n_classes, input_time_length=input_time_length, final_conv_length=2
    )

to_dense_prediction_model(model)

if cuda:
    model.cuda()

with torch.no_grad():
    dummy_input = torch.tensor(train_set.X[:1, :, :input_time_length], device="cpu")
    n_preds_per_input = model(dummy_input).shape[2]

out = model(dummy_input)

cropped_cb_train = CroppedTrialEpochScoring(
    "accuracy", name="train_trial_accuracy", lower_is_better=False, on_train=True
)

cropped_cb_valid = CroppedTrialEpochScoring(
    "accuracy", on_train=False, name="valid_trial_accuracy", lower_is_better=False,
)
# MaxNormDefaultConstraint and early stopping should be added to repeat previous braindecode

clf = EEGClassifier(
    model,
    criterion=CroppedNLLLoss,
    optimizer=optim.AdamW,
    train_split=TrainTestSplit(train_size=1 - valid_set_fraction),
    optimizer__lr=0.0625 * 0.01,
    optimizer__weight_decay=0,
    batch_size=32,
    iterator_train=CropsDataLoader,
    iterator_valid=CropsDataLoader,
    iterator_train__input_time_length=input_time_length,
    iterator_train__n_preds_per_input=n_preds_per_input,
    iterator_valid__input_time_length=input_time_length,
    iterator_valid__n_preds_per_input=n_preds_per_input,
    callbacks=[
        ("train_trial_accuracy", cropped_cb_train),
        ("valid_trial_accuracy", cropped_cb_valid),
    ],
)

clf.fit(train_set.X, train_set.y, epochs=20)
