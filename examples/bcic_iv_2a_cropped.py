"""
Cropped Decoding on BCIC IV 2a Competition Set
==============================================

"""

# Authors: Maciej Sliwowski
#          Robin Tibor Schirrmeister
#
# License: BSD-3

import os.path
from collections import OrderedDict

import torch
from torch import optim

from braindecode.classifier import EEGClassifier
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.datasets.croppedxy import CroppedXyDataset
from braindecode.datautil.signalproc import exponential_running_standardize
from braindecode.datautil.splitters import TrainTestSplit
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.losses import CroppedNLLLoss
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.models.deep4 import Deep4Net
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model
from braindecode.scoring import CroppedTrialEpochScoring
from braindecode.util import set_random_seeds

data_folder =  "/data/schirrmr/schirrmr/bci-competition-iv/2a-gdf/"
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
test_loader = BCICompetition4Set2A(
    test_filepath, labels_filename=test_label_filepath
)
raw_train = train_loader.load()
raw_test = test_loader.load()

# Preprocessing

raw_train = raw_train.drop_channels(["EOG-left", "EOG-central", "EOG-right"])
assert len(raw_train.ch_names) == 22
# lets convert to millvolt for numerical stability of next operations
raw_train = mne_apply(lambda a: a * 1e6, raw_train)
raw_train.filter(l_freq=low_cut_hz, h_freq=high_cut_hz, method='iir',
                 iir_params=dict(order=3, ftype='butter'))
raw_train = mne_apply(
    lambda a: exponential_running_standardize(
        a, factor_new=factor_new, init_block_size=init_block_size, eps=1e-4
    ),
    raw_train,
)

raw_test = raw_test.drop_channels(["EOG-left", "EOG-central", "EOG-right"])
assert len(raw_test.ch_names) == 22
raw_test = mne_apply(lambda a: a * 1e6, raw_test)
raw_test.filter(l_freq=low_cut_hz, h_freq=high_cut_hz, method='iir',
                iir_params=dict(order=3, ftype='butter'))
raw_test = mne_apply(
    lambda a: exponential_running_standardize(
        a, factor_new=factor_new, init_block_size=init_block_size, eps=1e-4
    ),
    raw_test,
)
marker_def = OrderedDict(
    [("Left Hand", [1]), ("Right Hand", [2],), ("Foot", [3]), ("Tongue", [4])]
)

train_set = create_signal_target_from_raw_mne(raw_train, marker_def, ival)
test_set = create_signal_target_from_raw_mne(raw_test, marker_def, ival)


set_random_seeds(seed=20190706, cuda=cuda)

n_classes = 4
n_chans = int(train_set.X.shape[1])
if model == "shallow":
    model = ShallowFBCSPNet(
        n_chans,
        n_classes,
        input_time_length=input_time_length,
        final_conv_length=30,
    )
elif model == "deep":
    model = Deep4Net(
        n_chans,
        n_classes,
        input_time_length=input_time_length,
        final_conv_length=2,
    )

to_dense_prediction_model(model)

if cuda:
    model.cuda()

with torch.no_grad():
    dummy_input = torch.tensor(
        train_set.X[:1, :, :input_time_length], device="cpu"
    )
    n_preds_per_input = model(dummy_input).shape[2]

out = model(dummy_input)

train_set = CroppedXyDataset(
    train_set.X, train_set.y,
    input_time_length=input_time_length,
    n_preds_per_input=n_preds_per_input)
test_set = CroppedXyDataset(test_set.X, test_set.y,
    input_time_length=input_time_length,
    n_preds_per_input=n_preds_per_input)

cropped_cb_train = CroppedTrialEpochScoring(
    "accuracy",
    name="train_trial_accuracy",
    lower_is_better=False,
    on_train=True,
)

cropped_cb_valid = CroppedTrialEpochScoring(
    "accuracy",
    on_train=False,
    name="valid_trial_accuracy",
    lower_is_better=False,
)
# MaxNormDefaultConstraint and early stopping should be added to repeat previous braindecode

clf = EEGClassifier(
    model,
    criterion=CroppedNLLLoss,
    optimizer=optim.AdamW,
    train_split=TrainTestSplit(train_size=1 - valid_set_fraction,
                               input_time_length=input_time_length,
                               n_preds_per_input=n_preds_per_input),
    optimizer__lr=0.0625 * 0.01,
    optimizer__weight_decay=0,
    batch_size=32,
    callbacks=[
        ("train_trial_accuracy", cropped_cb_train),
        ("valid_trial_accuracy", cropped_cb_valid),
    ],
)

clf.fit(train_set.X, train_set.y, epochs=20)
