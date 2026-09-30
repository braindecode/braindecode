# Authors: Lukas Gemein <l.gemein@gmail.com>
#          Robin Tibor Schirrmeister <robintibor@gmail.com>
#
# License: BSD-3
import numpy as np
import torch
from skorch.helper import predefined_split

from braindecode import EEGClassifier
from braindecode.datasets import BaseConcatDataset
from braindecode.datasets.tuh import _TUHAbnormalMock
from braindecode.models import ShallowFBCSPNet
from braindecode.preprocessing import (
    Preprocessor,
    create_fixed_length_windows,
    preprocess,
)
from braindecode.training import CroppedLoss
from braindecode.util import set_random_seeds

SEED = 20210726
N_EPOCHS = 3


def fit_variable_length(seed=SEED, lr=1e-3):
    set_random_seeds(seed=seed, cuda=False)

    # create fake tuh abnormal dataset (random signals)
    tuh = _TUHAbnormalMock(path="")
    # fake variable length trials by cropping first recording
    splits = tuh.split([[i] for i in range(len(tuh.datasets))])
    preprocess(
        concat_ds=splits["0"],
        preprocessors=[
            Preprocessor("crop", tmax=300),
        ],
    )
    variable_tuh = BaseConcatDataset([splits[str(i)] for i in range(len(tuh.datasets))])
    # make sure we actually have different length trials
    assert any(np.diff([ds.raw.n_times for ds in variable_tuh.datasets]) != 0)

    # create windows
    variable_tuh_windows = create_fixed_length_windows(
        concat_ds=variable_tuh,
        window_size_samples=1000,
        window_stride_samples=1000,
        drop_last_window=False,
        mapping={True: 1, False: 0},
    )

    # create train and valid set
    splits = variable_tuh_windows.split(
        [[i] for i in range(len(variable_tuh_windows.datasets))]
    )
    train_set = BaseConcatDataset(
        [splits[str(i)] for i in range(len(tuh.datasets) - 1)]
    )
    valid_set = BaseConcatDataset([splits[str(len(tuh.datasets) - 1)]])
    x, _, _ = train_set[0]
    n_classes = len(tuh.description.pathological.unique())
    # initialize a model
    model = ShallowFBCSPNet(
        n_chans=x.shape[0],
        n_outputs=n_classes,
        n_times=x.shape[1],
    )
    model.to_dense_prediction_model()

    # create and train a classifier
    clf = EEGClassifier(
        model,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.cross_entropy,
        optimizer=torch.optim.Adam,
        optimizer__lr=lr,
        batch_size=16,
        callbacks=["accuracy"],
        train_split=predefined_split(valid_set),
        classes=list(range(n_classes)),
    )
    clf.fit(train_set, y=None, epochs=N_EPOCHS)
    return clf, train_set, valid_set


def test_variable_length_trials_cropped_decoding():
    # The mock recordings are random noise, so there is nothing to generalize;
    # this checks that variable-length recordings train and predict end to end.
    clf, train_set, valid_set = fit_variable_length()
    history = clf.history

    assert len(history) == N_EPOCHS
    for key in ("train_loss", "valid_loss"):
        # Cross-entropy is non-negative; a negative loss means log-probabilities
        # and logits got mixed up.
        values = np.asarray(history[:, key])
        assert np.all(np.isfinite(values)) and np.all(values >= 0)
    for key in ("train_accuracy", "valid_accuracy"):
        values = np.asarray(history[:, key])
        assert np.all((values >= 0) & (values <= 1))

    # The train set is tiny and gets memorized: over 200 seeds the last/first
    # train-loss ratio was at most 0.47.
    train_loss = np.asarray(history[:, "train_loss"])
    assert train_loss[-1] < 0.75 * train_loss[0]

    # One prediction per window, including the windows of the shorter recording.
    assert clf.predict(train_set).shape == (len(train_set),)
    # Trials of different lengths come back as one prediction array per trial.
    trial_preds, trial_targets = clf.predict_trials(train_set)
    n_samples = [ds.metadata["i_stop_in_trial"].max() for ds in train_set.datasets]
    assert len(trial_preds) == len(trial_targets) == len(n_samples)
    lengths = [preds.shape[1] for preds in trial_preds]
    assert len(set(lengths)) > 1 and np.argmin(lengths) == np.argmin(n_samples)
    # The single valid recording is one trial made of all its windows.
    trial_preds, trial_targets = clf.predict_trials(valid_set)
    assert trial_preds.shape[:2] == (len(valid_set.datasets), 2)
    np.testing.assert_allclose(
        np.mean(trial_preds.mean(axis=2).argmax(axis=1) == trial_targets),
        history[-1, "valid_accuracy"],
    )
