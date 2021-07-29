# Authors: Maciej Sliwowski
#          Robin Tibor Schirrmeister
#
# License: BSD-3

import mne
import numpy as np
import torch
from mne.io import concatenate_raws
from skorch.helper import predefined_split
from torch import optim

from braindecode import EEGClassifier
from braindecode.datasets.xy import create_from_X_y
from braindecode.training.losses import CroppedLoss
from braindecode.models import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model, get_output_shape
from braindecode.util import set_random_seeds


def test_cropped_decoding():
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

    # Set if you want to use GPU
    # You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
    cuda = False
    set_random_seeds(seed=20170629, cuda=cuda)

    # This will determine how many crops are processed in parallel
    input_window_samples = 450
    n_classes = 2
    in_chans = X.shape[1]
    # final_conv_length determines the size of the receptive field of the ConvNet
    model = ShallowFBCSPNet(
        in_chans=in_chans,
        n_classes=n_classes,
        input_window_samples=input_window_samples,
        final_conv_length=12,
    )
    to_dense_prediction_model(model)

    if cuda:
        model.cuda()

    # Perform forward pass to determine how many outputs per input
    n_preds_per_input = get_output_shape(model, in_chans, input_window_samples)[2]

    train_set = create_from_X_y(X[:60], y[:60],
                                drop_last_window=False,
                                sfreq=100,
                                window_size_samples=input_window_samples,
                                window_stride_samples=n_preds_per_input)

    valid_set = create_from_X_y(X[60:], y[60:],
                                drop_last_window=False,
                                sfreq=100,
                                window_size_samples=input_window_samples,
                                window_stride_samples=n_preds_per_input)

    train_split = predefined_split(valid_set)

    clf = EEGClassifier(
        model,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.nll_loss,
        optimizer=optim.Adam,
        train_split=train_split,
        batch_size=32,
        callbacks=['accuracy'],
    )

    clf.fit(train_set, y=None, epochs=4)

    np.testing.assert_allclose(
        clf.history[:, 'train_loss'],
        np.array(
            [
                1.6269113222757976,
                1.3790630420049033,
                1.2096718986829122,
                1.0220003485679627
            ]
        ),
        rtol=1e-3,
        atol=1e-4,
    )

    np.testing.assert_allclose(
        clf.history[:, 'valid_loss'],
        np.array(
            [
                0.8756710092226664,
                0.7009139498074849,
                0.8636367797851563,
                0.6089811046918233
            ]
        ),
        rtol=1e-3,
        atol=1e-3,
    )

    np.testing.assert_allclose(
        clf.history[:, 'train_accuracy'],
        np.array(
            [
                0.5333333333333333,
                0.5833333333333334,
                0.5,
                0.6166666666666667
            ]
        ),
        rtol=1e-3,
        atol=1e-4,
    )

    np.testing.assert_allclose(
        clf.history[:, 'valid_accuracy'],
        np.array(
            [
                0.5,
                0.5666666666666667,
                0.5333333333333333,
                0.7
            ]
        ),
        rtol=1e-3,
        atol=1e-4,
    )
