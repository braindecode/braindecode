"""
Cropped Decoding on BCIC IV 2a Competition Set with skorch and moabb.
=====================================================================
"""

# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Robin Tibor Schirrmeister <robintibor@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD-3
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import mne

from braindecode.models.util import to_dense_prediction_model, get_output_shape

mne.set_log_level('ERROR')

from braindecode.datautil.windowers import create_windows_from_events
from braindecode.classifier import EEGClassifier
from braindecode.datasets import MOABBDataset
from braindecode.losses import CroppedNLLLoss
from braindecode.models.deep4 import Deep4Net
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.scoring import CroppedTrialEpochScoring
from braindecode.util import set_random_seeds
from braindecode.datautil.signalproc import exponential_running_standardize
from braindecode.datautil.transforms import transform_concat_ds

model_name = "shallow"  # 'shallow' or 'deep'
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

set_random_seeds(seed=20190706, cuda=cuda)

subject_id = 1  # 1-9
n_classes = 4

low_cut_hz = 4.  # 0 or 4
high_cut_hz = 38.
model = "shallow"  # 'shallow' or 'deep'
trial_start_offset_seconds = -0.5
input_time_length = 1000
max_epochs = 5
max_increase_epochs = 80
batch_size = 32
factor_new = 1e-3
init_block_size = 1000

n_chans = 22

if model_name == "shallow":
    model = ShallowFBCSPNet(
        n_chans,
        n_classes,
        input_time_length=input_time_length,
        final_conv_length=30,
    )
elif model_name == "deep":
    model = Deep4Net(
        n_chans,
        n_classes,
        input_time_length=input_time_length,
        final_conv_length=2,
    )

if cuda:
    model.cuda()

to_dense_prediction_model(model)

n_preds_per_input = get_output_shape(model, n_chans, input_time_length)[2]

dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])

standardize_func = partial(
    exponential_running_standardize, factor_new=factor_new,
    init_block_size=init_block_size)
raw_transform_dict = OrderedDict([
    ("pick_types", dict(eeg=True, meg=False, stim=False)),
    ('apply_function', dict(fun=lambda x: x*1e6, channel_wise=False)),
    ('filter', dict(l_freq=low_cut_hz, h_freq=high_cut_hz)),
    ('apply_function', dict(fun=standardize_func, channel_wise=False))
])
transform_concat_ds(dataset, raw_transform_dict)

sfreqs = [ds.raw.info['sfreq'] for ds in dataset.datasets]
assert len(np.unique(sfreqs)) == 1
trial_start_offset_samples = int(trial_start_offset_seconds * sfreqs[0])

windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    supercrop_size_samples=input_time_length,
    supercrop_stride_samples=n_preds_per_input,
    drop_samples=False
)


class TrainTestBCICIV2aSplit(object):
    def __call__(self, dataset, y, **kwargs):
        splitted = dataset.split('session')
        return splitted['session_T'], splitted['session_E']


cropped_cb_train = CroppedTrialEpochScoring(
    "accuracy",
    name="train_trial_accuracy",
    lower_is_better=False,
    on_train=True,
    input_time_length=input_time_length,
)
cropped_cb_train_f1_score = CroppedTrialEpochScoring(
    "f1_macro",
    name="train_f1_score",
    lower_is_better=False,
    on_train=True,
    input_time_length=input_time_length,
)
cropped_cb_valid = CroppedTrialEpochScoring(
    "accuracy",
    on_train=False,
    name="valid_trial_accuracy",
    lower_is_better=False,
    input_time_length=input_time_length,
)
cropped_cb_valid_f1_score = CroppedTrialEpochScoring(
    "f1_macro",
    name="valid_f1_score",
    lower_is_better=False,
    on_train=False,
    input_time_length=input_time_length,
)
# MaxNormDefaultConstraint and early stopping should be added to repeat
# previous braindecode

clf = EEGClassifier(
    model,
    criterion=CroppedNLLLoss,
    optimizer=torch.optim.AdamW,
    train_split=TrainTestBCICIV2aSplit(),
    optimizer__lr=0.0625 * 0.01,
    optimizer__weight_decay=0,
    batch_size=batch_size,
    callbacks=[
        ("train_trial_accuracy", cropped_cb_train),
        ("train_trial_f1_score", cropped_cb_train_f1_score),
        ("valid_trial_accuracy", cropped_cb_valid),
    ],
    device=device,
)

clf.fit(windows_dataset, y=None, epochs=2)
