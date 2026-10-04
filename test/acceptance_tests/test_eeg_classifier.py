# Authors: Maciej Sliwowski
#          Robin Tibor Schirrmeister
#          Lukas Gemein
#
# License: BSD-3
import math

import numpy as np
import pytest
import torch
from skorch.helper import predefined_split
from torch import optim

from braindecode.classifier import EEGClassifier
from braindecode.datasets.xy import create_from_X_y
from braindecode.models import ShallowFBCSPNet
from braindecode.training.losses import CroppedLoss
from braindecode.training.scoring import CroppedTrialEpochScoring
from braindecode.util import set_random_seeds
from test.acceptance_tests._bnci import SFREQ, load_left_right_trials

SEED = 20170629
N_EPOCHS = 4
BATCH_SIZE = 32
INPUT_WINDOW_SAMPLES = 300


def fit_eeg_classifier(seed=SEED, lr=1e-3):
    # First 60 trials of the first session: 48 to train, 12 to validate.
    X, y, session = load_left_right_trials(tmin=0.0, tmax=4.0)
    first_session = session == np.unique(session)[0]
    X, y = X[first_session][:60], y[first_session][:60]

    set_random_seeds(seed=seed, cuda=False)
    model = ShallowFBCSPNet(
        n_chans=X.shape[1],
        n_outputs=2,
        n_times=INPUT_WINDOW_SAMPLES,
        final_conv_length=12,
    )
    model.to_dense_prediction_model()
    n_preds_per_input = model.get_output_shape()[2]

    def windows(X, y):
        return create_from_X_y(
            X,
            y,
            drop_last_window=False,
            sfreq=SFREQ,
            window_size_samples=INPUT_WINDOW_SAMPLES,
            window_stride_samples=n_preds_per_input,
        )

    train_set, valid_set = windows(X[:48], y[:48]), windows(X[48:], y[48:])
    clf = EEGClassifier(
        model,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.cross_entropy,
        optimizer=optim.Adam,
        optimizer__lr=lr,
        train_split=predefined_split(valid_set),
        batch_size=BATCH_SIZE,
        iterator_train__shuffle=True,
        iterator_train__generator=torch.Generator().manual_seed(seed),
        callbacks=[
            (
                "train_trial_accuracy",
                CroppedTrialEpochScoring(
                    "accuracy",
                    name="train_trial_accuracy",
                    lower_is_better=False,
                    on_train=True,
                ),
            ),
            (
                "valid_trial_accuracy",
                CroppedTrialEpochScoring(
                    "accuracy",
                    on_train=False,
                    name="valid_trial_accuracy",
                    lower_is_better=False,
                ),
            ),
        ],
        classes=[0, 1],
    )
    clf.fit(train_set, y=None, epochs=N_EPOCHS)
    return clf, train_set, valid_set


def _trial_accuracy(clf, dataset):
    trial_preds, trial_targets = clf.predict_trials(dataset)
    return np.mean(trial_preds.mean(axis=2).argmax(axis=1) == trial_targets)


@pytest.mark.network
def test_eeg_classifier(deterministic_algorithms):
    clf, train_set, valid_set = fit_eeg_classifier()
    history = clf.history

    # Several crops per trial, so trial scores really aggregate crops.
    assert len(train_set) > 48 and len(valid_set) > 12
    n_train_batches = math.ceil(len(train_set) / BATCH_SIZE)
    n_valid_batches = math.ceil(len(valid_set) / BATCH_SIZE)
    assert len(history) == N_EPOCHS
    assert all(epoch["train_batch_count"] == n_train_batches for epoch in history)
    assert all(epoch["valid_batch_count"] == n_valid_batches for epoch in history)
    for epoch in history:
        batches = epoch["batches"]
        assert len(batches) == n_train_batches + n_valid_batches
        assert all("train_loss" in batch for batch in batches[:n_train_batches])
        assert all("valid_loss" in batch for batch in batches[n_train_batches:])

    for key in ("train_loss", "valid_loss"):
        # Cross-entropy is non-negative; a negative loss means log-probabilities
        # and logits got mixed up.
        assert np.all(np.asarray(history[:, key]) >= 0)
    for key in ("train_trial_accuracy", "valid_trial_accuracy"):
        values = np.asarray(history[:, key])
        assert np.all((values >= 0) & (values <= 1))

    # The 48 training trials get fitted. Over 200 seeds the last/first
    # train-loss ratio was at most 0.52 and the final train trial accuracy at
    # least 44/48.
    train_loss = np.asarray(history[:, "train_loss"])
    assert train_loss[-1] < 0.75 * train_loss[0]
    assert history[-1, "train_trial_accuracy"] >= 0.8

    # Both scorers predict every crop with the final weights and average the
    # crops of each trial, which is what predict_trials returns.
    np.testing.assert_allclose(
        _trial_accuracy(clf, train_set), history[-1, "train_trial_accuracy"]
    )
    np.testing.assert_allclose(
        _trial_accuracy(clf, valid_set), history[-1, "valid_trial_accuracy"]
    )
