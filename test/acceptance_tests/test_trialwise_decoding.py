# Authors: Maciej Sliwowski
#          Robin Tibor Schirrmeister
#
# License: BSD-3

import mne
import numpy as np
import torch
from mne.io import concatenate_raws
from skorch.helper import predefined_split
from torch.utils.data import Dataset, Subset

from braindecode.classifier import EEGClassifier
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds


class EpochsDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows
        self.y = np.array(self.windows.events[:, -1])
        self.y = self.y - self.y.min()

    def __getitem__(self, index):
        X = self.windows.get_data(item=index)[0].astype('float32')[:, :, None]
        y = self.y[index]
        return X, y

    def __len__(self):
        return len(self.windows.events)


def test_trialwise_decoding():
    # 5,6,7,10,13,14 are codes for executed and imagined hands/feet
    subject_id = 1
    event_codes = [5, 6, 9, 10, 13, 14]

    # This will download the files if you don't have them yet,
    # and then return the paths to the files.
    physionet_paths = mne.datasets.eegbci.load_data(
        subject_id, event_codes, update_path=False
    )

    # Load each of the files
    parts = [
        mne.io.read_raw_edf(
            path, preload=True, stim_channel="auto", verbose="WARNING"
        )
        for path in physionet_paths
    ]

    # Concatenate them
    raw = concatenate_raws(parts)
    raw.apply_function(lambda x: x * 1000000)

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

    ds = EpochsDataset(epoched)

    train_set = Subset(ds, np.arange(60))
    valid_set = Subset(ds, np.arange(60, len(ds)))

    train_valid_split = predefined_split(valid_set)

    cuda = False
    if cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    set_random_seeds(seed=20170629, cuda=cuda)
    n_classes = 2
    in_chans = train_set[0][0].shape[0]
    input_window_samples = train_set[0][0].shape[1]
    model = ShallowFBCSPNet(
        in_chans=in_chans,
        n_classes=n_classes,
        input_window_samples=input_window_samples,
        final_conv_length="auto",
    )
    if cuda:
        model.cuda()

    clf = EEGClassifier(
        model,
        cropped=False,
        criterion=torch.nn.NLLLoss,
        optimizer=torch.optim.Adam,
        train_split=train_valid_split,
        optimizer__lr=0.001,
        batch_size=30,
        callbacks=["accuracy"],
        device=device,
    )
    clf.fit(train_set, y=None, epochs=6)

    np.testing.assert_allclose(
        clf.history[:, 'train_loss'],
        np.array([
            1.623811,
            1.141197,
            0.730361,
            0.5994,
            0.738884,
            0.419241
        ]),
        rtol=1e-4,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        clf.history[:, 'valid_loss'],
        np.array([
            1.958356261253357,
            0.8494758009910583,
            0.9321595430374146,
            0.8126973509788513,
            0.7426613569259644,
            0.7547585368156433
        ]),
        rtol=1e-4,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        clf.history[:, 'train_accuracy'],
        np.array([
            0.5,
            0.8166666666666667,
            0.7,
            0.8,
            0.8666666666666667,
            0.9
        ]),
        rtol=1e-4,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        clf.history[:, 'valid_accuracy'],
        np.array([
            0.4666666666666667,
            0.6,
            0.5,
            0.5666666666666667,
            0.6,
            0.6,
        ]),
        rtol=1e-4,
        atol=1e-5,
    )
