"""
Skorch Crop Decoding
=========================

Example using Skorch for crop decoding on a simpler dataset.
"""

# Authors: Lukas Gemein
#          Robin Tibor Schirrmeister
#          Alexandre Gramfort
#          Maciej Sliwowski
#
# License: BSD-3

import mne
import numpy as np
from mne.io import concatenate_raws
from torch import optim

from braindecode.classifier import EEGClassifier
from braindecode.datasets.croppedxy import CroppedXyDataset
from braindecode.datautil.splitters import TrainTestSplit
from braindecode.losses import CroppedNLLLoss
from braindecode.models import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model, get_output_shape
from braindecode.scoring import CroppedTrialEpochScoring
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


set_random_seeds(20200114, cuda=False)

# final_conv_length = auto ensures we only get a single output in the time dimension
model = ShallowFBCSPNet(
    in_chans=in_chans,
    n_classes=n_classes,
    input_time_length=X.shape[2],
    final_conv_length="auto",
)
to_dense_prediction_model(model)
if cuda:
    model.cuda()

input_time_length = X.shape[2]

# Perform forward pass to determine how many outputs per input
n_preds_per_input = get_output_shape(model, in_chans, input_time_length)[2]


train_set = CroppedXyDataset(X[:70], y[:70],
                             input_time_length=input_time_length,
                             n_preds_per_input=n_preds_per_input)
test_set = CroppedXyDataset(X[70:], y=y[70:],
                            input_time_length=input_time_length,
                            n_preds_per_input=n_preds_per_input)

clf = EEGClassifier(
    model,
    cropped=True,
    criterion=CroppedNLLLoss,
    optimizer=optim.AdamW,
    train_split=TrainTestSplit(
        train_size=40,
        input_time_length=input_time_length,
        n_preds_per_input=n_preds_per_input,),
    optimizer__lr=0.0625 * 0.01,
    optimizer__weight_decay=0,
    batch_size=64,
    callbacks=['accuracy'],
)

clf.fit(train_set, y=None, epochs=4)
clf.predict(test_set)
