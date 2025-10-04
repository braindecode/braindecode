# Authors: Bruno Aristimunha
#
# License: BSD-3

import mne
import numpy as np
import pytest
import requests
import torch
from mne.datasets.eegbci.eegbci import EEGMI_URL
from mne.io import concatenate_raws
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from braindecode.classifier import EEGClassifier
from braindecode.datasets.xy import create_from_X_y
from braindecode.models.biot import BIOT
from braindecode.util import set_random_seeds


def check_http_issue():
    """Check if the EEGMI_URL is available."""
    try:
        response = requests.get(EEGMI_URL)
        response.raise_for_status()
        return False
    except requests.HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
        return True
    except Exception as err:
        print(f'Other error occurred: {err}')
        return True


@pytest.fixture()
def real_data():

    subject_id = 1
    # 5,6,7,10,13,14 are codes for executed and imagined hands/feet
    event_codes = [5, 6, 9, 10, 13, 14]

    # This will download the files if you don't have them yet,
    # and then return the paths to the files.
    physionet_paths = mne.datasets.eegbci.load_data(
        subject_id, event_codes, update_path=False
    )

    # Load each of the files
    parts = [
        mne.io.read_raw_edf(path, preload=True, stim_channel="auto", verbose="WARNING")
        for path in physionet_paths
    ]

    # Concatenate them
    raw = concatenate_raws(parts)

    # Find the events in this dataset
    events, _ = mne.events_from_annotations(raw)

    # Use only EEG channels
    eeg_channel_inds = mne.pick_types(
        raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
    )

    # Extract trials, only using EEG channels
    epoched = mne.Epochs(
        raw,
        events,
        dict(hands=2, feet=3),
        tmin=1,
        tmax=4.1,
        proj=False,
        picks=eeg_channel_inds,
        baseline=None,
        preload=True,
    )

    # Convert data from volt to millivolt
    # Pytorch expects float32 for input and int64 for labels.
    X = (epoched.get_data() * 1e6).astype(np.float32)
    y = (epoched.events[:, 2] - 2).astype(np.int64)  # 2,3 -> 0,1
    return X, y


# TODO: adding this test for all the models when the issue #571 is closed
@pytest.mark.skipif(check_http_issue(),
                    reason="HTTP issue occurred, skipping test.")

def test_correctness_biot(real_data):
    seed = 20200220
    set_random_seeds(seed=seed, cuda=False)

    X, y = real_data

    n_times = 450
    n_chans = X.shape[1]

    model = BIOT(
        n_outputs=2,
        n_chans=n_chans,
        n_times=n_times,
        sfreq=100,
        n_layers=1,
        att_num_heads=2,
        hop_length=50,
        emb_size=256,
    )

    train_set = create_from_X_y(
        X[:48],
        y[:48],
        drop_last_window=False,
        sfreq=100,
        window_size_samples=n_times,
        window_stride_samples=n_times,
    )

    valid_set = create_from_X_y(
        X[48:60],
        y[48:60],
        drop_last_window=False,
        sfreq=100,
        window_size_samples=n_times,
        window_stride_samples=n_times,
    )
    n_epochs = 4
    clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),
        batch_size=16,
        classes=[0, 1],
        max_epochs=n_epochs,
        callbacks=[
            "accuracy",
            ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=n_epochs - 1)),
        ],
    )

    clf.fit(train_set, y=None)

    train_loss = clf.history[:, "train_loss"]
    valid_loss = clf.history[:, "valid_loss"]
    valid_accuracy = clf.history[:, "valid_accuracy"]
    train_accuracy = clf.history[:, "train_accuracy"]

    assert valid_loss[0] > valid_loss[-1]
    assert train_loss[0] > train_loss[-1]
    assert train_accuracy[0] < train_accuracy[-1]
    assert valid_accuracy[0] < valid_accuracy[-1]
