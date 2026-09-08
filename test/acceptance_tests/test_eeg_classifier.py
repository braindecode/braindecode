# Authors: Maciej Sliwowski
#          Robin Tibor Schirrmeister
#          Lukas Gemein
#
# License: BSD-3
import mne
import numpy as np
import pytest
from mne.io import concatenate_raws
from skorch.helper import predefined_split
from torch import optim
from torch.nn.functional import nll_loss

from test.acceptance_tests._history_assertions import assert_learning_history

from braindecode.classifier import EEGClassifier
from braindecode.datasets.xy import create_from_X_y
from braindecode.models import ShallowFBCSPNet
from braindecode.training.losses import CroppedLoss
from braindecode.training.scoring import CroppedTrialEpochScoring
from braindecode.util import np_to_th, set_random_seeds


@pytest.mark.network
def test_eeg_classifier():
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
        n_chans=in_chans,
        n_outputs=n_classes,
        n_times=input_window_samples,
        final_conv_length=12,
    )
    model.to_dense_prediction_model()

    if cuda:
        model.cuda()

    # determine output size
    test_input = np_to_th(
        np.ones((2, in_chans, input_window_samples, 1), dtype=np.float32)
    )
    if cuda:
        test_input = test_input.cuda()
    out = model(test_input)
    n_preds_per_input = out.cpu().data.numpy().shape[2]

    train_set = create_from_X_y(
        X[:48],
        y[:48],
        drop_last_window=False,
        sfreq=100,
        window_size_samples=input_window_samples,
        window_stride_samples=n_preds_per_input,
    )

    valid_set = create_from_X_y(
        X[48:60],
        y[48:60],
        drop_last_window=False,
        sfreq=100,
        window_size_samples=input_window_samples,
        window_stride_samples=n_preds_per_input,
    )

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

    clf = EEGClassifier(
        model,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=nll_loss,
        optimizer=optim.Adam,
        train_split=predefined_split(valid_set),
        batch_size=32,
        callbacks=[
            ("train_trial_accuracy", cropped_cb_train),
            ("valid_trial_accuracy", cropped_cb_valid),
        ],
        classes=[0, 1],
    )

    clf.fit(train_set, y=None, epochs=4)
    assert_learning_history(
        clf.history,
        n_epochs=4,
        loss_keys=("train_loss", "valid_loss"),
        accuracy_keys=("train_trial_accuracy", "valid_trial_accuracy"),
        improving_accuracy_keys=("train_trial_accuracy",),
    )
    assert all(epoch["train_batch_count"] == 3 for epoch in clf.history)
    assert all(epoch["valid_batch_count"] == 1 for epoch in clf.history)
    assert all(len(epoch["batches"]) == 4 for epoch in clf.history)
    assert all("train_loss" in batch for epoch in clf.history for batch in epoch["batches"][:-1])
    assert all("valid_loss" in epoch["batches"][-1] for epoch in clf.history)
    return clf
