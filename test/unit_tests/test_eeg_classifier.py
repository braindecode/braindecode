# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD-3

import pytest
import numpy as np
import torch
from sklearn.base import clone
from skorch.callbacks import LRScheduler
from skorch.utils import to_tensor
from torch import optim
from torch.nn.functional import nll_loss

from braindecode import EEGClassifier
from braindecode.training import CroppedLoss


class MockDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, item):
        return torch.rand(4, 1), torch.ones(4)


class MockModule(torch.nn.Module):
    def __init__(self, preds):
        super().__init__()
        self.preds = to_tensor(preds, device='cpu')
        self.linear = torch.nn.Linear(5, 5)

    def forward(self, x):
        return self.preds


def test_trialwise_predict_and_predict_proba():
    preds = np.array(
        [
            [0.125, 0.875],
            [1., 0.],
            [0.8, 0.2],
            [0.9, 0.1],
        ]
    )
    clf = EEGClassifier(
        MockModule(preds),
        optimizer=optim.Adam,
        batch_size=32
    )
    clf.initialize()
    np.testing.assert_array_equal(preds.argmax(1), clf.predict(MockDataset()))
    np.testing.assert_array_equal(preds, clf.predict_proba(MockDataset()))


def test_cropped_predict_and_predict_proba():
    preds = np.array(
        [
            [[0.2, 0.1, 0.1, 0.1], [0.8, 0.9, 0.9, 0.9]],
            [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
            [[1.0, 1.0, 1.0, 0.2], [0.0, 0.0, 0.0, 0.8]],
            [[0.9, 0.8, 0.9, 1.0], [0.1, 0.2, 0.1, 0.0]],
        ]
    )
    clf = EEGClassifier(
        MockModule(preds),
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=nll_loss,
        optimizer=optim.Adam,
        batch_size=32
    )
    clf.initialize()
    # for cropped decoding classifier returns one label for each trial (averaged over all crops)
    np.testing.assert_array_equal(preds.mean(-1).argmax(1), clf.predict(MockDataset()))
    # for cropped decoding classifier returns values for each trial (average over all crops)
    np.testing.assert_array_equal(preds.mean(-1), clf.predict_proba(MockDataset()))


def test_cropped_predict_and_predict_proba_not_aggregate_predictions():
    preds = np.array(
        [
            [[0.2, 0.1, 0.1, 0.1], [0.8, 0.9, 0.9, 0.9]],
            [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
            [[1.0, 1.0, 1.0, 0.2], [0.0, 0.0, 0.0, 0.8]],
            [[0.9, 0.8, 0.9, 1.0], [0.1, 0.2, 0.1, 0.0]],
        ]
    )
    clf = EEGClassifier(
        MockModule(preds),
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=nll_loss,
        optimizer=optim.Adam,
        batch_size=32,
        aggregate_predictions=False
    )
    clf.initialize()
    np.testing.assert_array_equal(preds.argmax(1), clf.predict(MockDataset()))
    np.testing.assert_array_equal(preds, clf.predict_proba(MockDataset()))


def test_predict_trials():
    preds = np.array(
        [
            [[0.2, 0.1, 0.1, 0.1], [0.8, 0.9, 0.9, 0.9]],
            [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
            [[1.0, 1.0, 1.0, 0.2], [0.0, 0.0, 0.0, 0.8]],
            [[0.9, 0.8, 0.9, 1.0], [0.1, 0.2, 0.1, 0.0]],
        ]
    )
    clf = EEGClassifier(
        MockModule(preds),
        cropped=False,
        criterion=CroppedLoss,
        criterion__loss_function=nll_loss,
        optimizer=optim.Adam,
        batch_size=32
    )
    clf.initialize()
    with pytest.warns(UserWarning, match="This method was designed to predict "
                                         "trials in cropped mode."):
        clf.predict_trials(MockDataset(), return_targets=False)


def test_eeg_classifier_clonable():
    preds = np.array(
        [
            [[0.2, 0.1, 0.1, 0.1], [0.8, 0.9, 0.9, 0.9]],
            [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
            [[1.0, 1.0, 1.0, 0.2], [0.0, 0.0, 0.0, 0.8]],
            [[0.9, 0.8, 0.9, 1.0], [0.1, 0.2, 0.1, 0.0]],
        ]
    )
    clf = EEGClassifier(
        MockModule(preds),
        cropped=False,
        callbacks=[
            "accuracy",
            ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=1))],
        criterion=CroppedLoss,
        criterion__loss_function=nll_loss,
        optimizer=optim.Adam,
        batch_size=32
    )
    clone(clf)
    clf.initialize()
    clone(clf)
