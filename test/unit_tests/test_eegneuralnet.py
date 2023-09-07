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

from braindecode import EEGClassifier, EEGRegressor
from braindecode.training import CroppedLoss
from braindecode.models.base import EEGModuleMixin


class MockDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, item):
        return torch.rand(4, 1), torch.ones(4)


class MockModule(EEGModuleMixin, torch.nn.Module):
    def __init__(
            self,
            preds,
            n_outputs=None,
            n_chans=None,
            chs_info=None,
            n_times=None,
            input_window_seconds=None,
            sfreq=None,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        self.preds = to_tensor(preds, device='cpu')
        self.linear = torch.nn.Linear(self.n_times, self.n_chans)
        self.final_layer = torch.nn.Linear(self.n_chans, self.n_outputs)

    def forward(self, x):
        return self.preds


@pytest.fixture(params=[EEGClassifier, EEGRegressor])
def eegneuralnet_cls(request):
    return request.param


@pytest.fixture
def preds():
    return np.array(
        [
            [[0.2, 0.1, 0.1, 0.1], [0.8, 0.9, 0.9, 0.9]],
            [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
            [[1.0, 1.0, 1.0, 0.2], [0.0, 0.0, 0.0, 0.8]],
            [[0.9, 0.8, 0.9, 1.0], [0.1, 0.2, 0.1, 0.0]],
        ]
    )


def test_trialwise_predict_and_predict_proba(eegneuralnet_cls):
    preds = np.array(
        [
            [0.125, 0.875],
            [1., 0.],
            [0.8, 0.2],
            [0.9, 0.1],
        ]
    )
    eegneuralnet = eegneuralnet_cls(
        MockModule,
        module__preds=preds,
        module__n_outputs=2,
        module__n_chans=3,
        module__n_times=3,
        optimizer=optim.Adam,
        batch_size=32
    )
    eegneuralnet.initialize()
    target_predict = preds if isinstance(eegneuralnet, EEGRegressor) else preds.argmax(1)
    np.testing.assert_array_equal(target_predict, eegneuralnet.predict(MockDataset()))
    np.testing.assert_array_equal(preds, eegneuralnet.predict_proba(MockDataset()))


def test_cropped_predict_and_predict_proba(eegneuralnet_cls, preds):
    eegneuralnet = eegneuralnet_cls(
        MockModule,
        module__preds=preds,
        module__n_outputs=4,
        module__n_chans=3,
        module__n_times=3,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=nll_loss,
        optimizer=optim.Adam,
        batch_size=32
    )
    eegneuralnet.initialize()
    target_predict = (preds.mean(-1) if isinstance(eegneuralnet, EEGRegressor)
                      else preds.mean(-1).argmax(1))
    # for cropped decoding classifier returns one label for each trial (averaged over all crops)
    np.testing.assert_array_equal(target_predict, eegneuralnet.predict(MockDataset()))
    # for cropped decoding classifier returns values for each trial (average over all crops)
    np.testing.assert_array_equal(preds.mean(-1), eegneuralnet.predict_proba(MockDataset()))


def test_cropped_predict_and_predict_proba_not_aggregate_predictions(eegneuralnet_cls, preds):
    eegneuralnet = eegneuralnet_cls(
        MockModule,
        module__preds=preds,
        module__n_outputs=4,
        module__n_chans=3,
        module__n_times=3,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=nll_loss,
        optimizer=optim.Adam,
        batch_size=32,
        aggregate_predictions=False
    )
    eegneuralnet.initialize()
    target_predict = preds if isinstance(eegneuralnet, EEGRegressor) else preds.argmax(1)
    np.testing.assert_array_equal(target_predict, eegneuralnet.predict(MockDataset()))
    np.testing.assert_array_equal(preds, eegneuralnet.predict_proba(MockDataset()))


def test_predict_trials(eegneuralnet_cls, preds):
    eegneuralnet = eegneuralnet_cls(
        MockModule,
        module__preds=preds,
        module__n_outputs=4,
        module__n_chans=3,
        module__n_times=3,
        cropped=False,
        criterion=CroppedLoss,
        criterion__loss_function=nll_loss,
        optimizer=optim.Adam,
        batch_size=32
    )
    eegneuralnet.initialize()
    with pytest.warns(UserWarning, match="This method was designed to predict "
                                         "trials in cropped mode."):
        eegneuralnet.predict_trials(MockDataset(), return_targets=False)


def test_clonable(eegneuralnet_cls, preds):
    eegneuralnet = eegneuralnet_cls(
        MockModule,
        module__preds=preds,
        module__n_outputs=4,
        module__n_chans=3,
        module__n_times=3,
        cropped=False,
        callbacks=[
            "accuracy",
            ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=1))],
        criterion=CroppedLoss,
        criterion__loss_function=nll_loss,
        optimizer=optim.Adam,
        batch_size=32
    )
    clone(eegneuralnet)
    eegneuralnet.initialize()
    clone(eegneuralnet)
