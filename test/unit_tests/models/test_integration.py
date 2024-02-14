# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#          Alexandre Gramfort
#
# License: BSD-3


import mne
import torch
import pytest

from skorch.dataset import ValidSplit

from braindecode.models.util import models_dict
from braindecode import EEGClassifier
from braindecode.datasets import BaseDataset, BaseConcatDataset
from braindecode.datasets.moabb import fetch_data_with_moabb
from braindecode.preprocessing.windowers import (
    create_windows_from_events)

# Temporary fix for the issue with this models
from braindecode.models import TCN, HybridNet, EEGResNet, SleepStagerEldele2021

models_not_working = {"TCN": TCN, "Hybrid": HybridNet,
                      "SleepStagerEldele2021": SleepStagerEldele2021,
                      "EEGResNet": EEGResNet}


bnci_kwargs = {"n_sessions": 2, "n_runs": 3,
               "n_subjects": 9, "paradigm": "imagery",
               "duration": 3869, "sfreq": 250,
               "event_list": ("left", "right"),
               "channels": ('C5', 'C3', 'C1')}


@pytest.fixture(scope="module")
def concat_ds_targets():
    raws, description = fetch_data_with_moabb(
        dataset_name="FakeDataset", subject_ids=1,
        dataset_kwargs=bnci_kwargs)

    events, _ = mne.events_from_annotations(raws[0])
    targets = events[:, -1] - 1
    ds = [BaseDataset(raws[i], description.iloc[i]) for i in range(3)]
    concat_ds = BaseConcatDataset(ds)
    return concat_ds, targets

@pytest.fixture(scope='module')
def concat_windows_dataset(concat_ds_targets):
    concat_ds, targets = concat_ds_targets
    windows_ds = create_windows_from_events(
        concat_ds=concat_ds, trial_start_offset_samples=0,
        trial_stop_offset_samples=0, window_size_samples=750,
        window_stride_samples=100, drop_last_window=False)

    return windows_ds

@pytest.mark.parametrize("model_name", models_dict.keys())
def test_model_list(model_name, concat_windows_dataset):

    if model_name in models_not_working:
        pytest.skip(f"Model {model_name} not working")

    model_class = models_dict[model_name]

    LEARNING_RATE = 0.0625 * 0.01
    BATCH_SIZE = 2
    EPOCH = 1
    seed = 2409
    valid_split = 0.2

    clf = EEGClassifier(
        module=model_class,
        optimizer=torch.optim.Adam,
        optimizer__lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        max_epochs=EPOCH,
        classes=[0, 1],
        train_split=ValidSplit(valid_split,
                               random_state=seed),
        verbose=0,
    )

    clf.fit(X=concat_windows_dataset)
