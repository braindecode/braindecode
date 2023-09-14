# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD-3
import logging

import pandas as pd
import pytest
import numpy as np
import torch
import mne
from sklearn.base import clone
from skorch.callbacks import LRScheduler
from skorch.utils import to_tensor
from torch import optim
from torch.nn.functional import nll_loss

from braindecode import EEGClassifier, EEGRegressor
from braindecode.training import CroppedLoss
from braindecode.models.base import EEGModuleMixin
from braindecode.datasets import WindowsDataset, BaseConcatDataset
# from braindecode.models.util import models_dict
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet


class MockDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, item):
        return torch.rand(3, 10), item % 4


class MockModuleReturnMockedPreds(EEGModuleMixin, torch.nn.Module):
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
        self.final_layer = torch.nn.Conv1d(self.n_chans, self.n_outputs, self.n_times)

    def forward(self, x):
        return self.preds


class MockModuleFinalLayer(MockModuleReturnMockedPreds):
    def forward(self, x):
        return self.final_layer(x).reshape(x.shape[0], self.n_outputs)


@pytest.fixture(params=[EEGClassifier, EEGRegressor])
def eegneuralnet_cls(request):
    return request.param


@pytest.fixture
def preds():
    return np.array(
        [
            [[0.2, 0.1, 0.1, 0.1], [0.8, 0.9, 0.9, 0.9]],
            [[0.2, 0.1, 0.1, 0.1], [0.8, 0.9, 0.9, 0.9]],
            [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
            [[1.0, 1.0, 1.0, 0.2], [0.0, 0.0, 0.0, 0.8]],
            [[0.9, 0.8, 0.9, 1.0], [0.1, 0.2, 0.1, 0.0]],
        ]
    )


@pytest.fixture
def Xy():
    dataset = MockDataset()
    X, y = zip(*[dataset[i] for i in range(len(dataset))])
    return np.stack(X), np.stack(y)


@pytest.fixture
def epochs(Xy):
    X, y = Xy
    metadata = [(yi, 0, 0, 9) for yi in y]
    metadata = pd.DataFrame(
        metadata,
        columns=['target', 'i_window_in_trial', 'i_start_in_trial', 'i_stop_in_trial']
    )
    return mne.EpochsArray(
        X,
        info=mne.create_info(
            ch_names=['ch1', 'ch2', 'ch3', ],
            sfreq=10,
            ch_types='eeg',
        ),
        metadata=metadata,
    )


@pytest.fixture
def windows_dataset_metadata(epochs):
    return WindowsDataset(
        windows=epochs,
        targets_from='metadata',
        description={},
    )


@pytest.fixture
def windows_dataset_channels(epochs):
    return WindowsDataset(
        windows=epochs,
        targets_from='channels',
        description={},
    )


@pytest.fixture
def concat_dataset_metadata(windows_dataset_metadata):
    return BaseConcatDataset(
        [windows_dataset_metadata, windows_dataset_metadata]
    )


@pytest.fixture
def concat_dataset_channels(
        windows_dataset_metadata,
        windows_dataset_channels,
):
    return BaseConcatDataset(
        [windows_dataset_metadata, windows_dataset_channels]
    )


def test_trialwise_predict_and_predict_proba(eegneuralnet_cls):
    preds = np.array(
        [
            [0.125, 0.875],
            [1., 0.],
            [0.8, 0.2],
            [0.8, 0.2],
            [0.9, 0.1],
        ]
    )
    eegneuralnet = eegneuralnet_cls(
        MockModuleReturnMockedPreds,
        module__preds=preds,
        module__n_outputs=2,
        module__n_chans=3,
        module__n_times=10,
        optimizer=optim.Adam,
        batch_size=32
    )
    eegneuralnet.initialize()
    target_predict = preds if isinstance(eegneuralnet, EEGRegressor) else preds.argmax(1)
    np.testing.assert_array_equal(target_predict, eegneuralnet.predict(MockDataset()))
    np.testing.assert_array_equal(preds, eegneuralnet.predict_proba(MockDataset()))


def test_cropped_predict_and_predict_proba(eegneuralnet_cls, preds):
    eegneuralnet = eegneuralnet_cls(
        MockModuleReturnMockedPreds,
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
        MockModuleReturnMockedPreds,
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
        MockModuleReturnMockedPreds,
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
        MockModuleReturnMockedPreds,
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


def test_set_signal_params_numpy(eegneuralnet_cls, preds, Xy):
    X, y = Xy
    net = eegneuralnet_cls(
        MockModuleFinalLayer,
        module__preds=preds,
        cropped=False,
        optimizer=optim.Adam,
        batch_size=32,
        train_split=None,
        max_epochs=1,
    )
    net.fit(X, y=y)
    assert net.module_.n_times == 10
    assert net.module_.n_chans == 3
    assert net.module_.n_outputs == (1 if isinstance(net, EEGRegressor) else 4)


def test_set_signal_params_epochs(eegneuralnet_cls, preds, epochs):
    y = epochs.metadata.target.values
    net = eegneuralnet_cls(
        MockModuleFinalLayer,
        module__preds=preds,
        cropped=False,
        optimizer=optim.Adam,
        batch_size=32,
        train_split=None,
        max_epochs=1,
    )
    net.fit(epochs, y=y)
    assert net.module_.n_times == 10
    assert net.module_.n_chans == 3
    assert net.module_.n_outputs == (1 if isinstance(net, EEGRegressor) else 4)
    assert net.module_.chs_info == epochs.info['chs']
    assert net.module_.input_window_seconds == 10 / 10
    assert net.module_.sfreq == 10


def test_set_signal_params_torch_ds(eegneuralnet_cls, preds):
    n_outputs = (1 if eegneuralnet_cls == EEGRegressor else 4)
    net = eegneuralnet_cls(
        MockModuleFinalLayer,
        module__preds=preds,
        module__n_outputs=n_outputs,
        cropped=False,
        optimizer=optim.Adam,
        batch_size=32,
        train_split=None,
        max_epochs=1,
    )
    net.fit(MockDataset(), y=None)
    assert net.module_.n_times == 10
    assert net.module_.n_chans == 3
    assert net.module_.n_outputs == n_outputs


def test_set_signal_params_windows_ds_metadata(eegneuralnet_cls, preds, windows_dataset_metadata):
    n_outputs = (1 if eegneuralnet_cls == EEGRegressor else 4)
    net = eegneuralnet_cls(
        MockModuleFinalLayer,
        module__preds=preds,
        cropped=False,
        optimizer=optim.Adam,
        batch_size=32,
        train_split=None,
        max_epochs=1,
    )
    net.fit(windows_dataset_metadata, y=None)
    assert net.module_.n_times == 10
    assert net.module_.n_chans == 3
    assert net.module_.n_outputs == n_outputs


def test_set_signal_params_windows_ds_channels(eegneuralnet_cls, preds, windows_dataset_channels):
    n_outputs = (1 if eegneuralnet_cls == EEGRegressor else 4)
    net = eegneuralnet_cls(
        MockModuleFinalLayer,
        module__preds=preds,
        module__n_outputs=n_outputs,
        cropped=False,
        optimizer=optim.Adam,
        batch_size=32,
        train_split=None,
        max_epochs=1,
    )
    net.fit(windows_dataset_channels, y=None)
    assert net.module_.n_times == 10
    assert net.module_.n_chans == 3
    assert net.module_.n_outputs == n_outputs


def test_set_signal_params_concat_ds_metadata(eegneuralnet_cls, preds, concat_dataset_metadata):
    n_outputs = (1 if eegneuralnet_cls == EEGRegressor else 4)
    net = eegneuralnet_cls(
        MockModuleFinalLayer,
        module__preds=preds,
        cropped=False,
        optimizer=optim.Adam,
        batch_size=32,
        train_split=None,
        max_epochs=1,
    )
    net.fit(concat_dataset_metadata, y=None)
    assert net.module_.n_times == 10
    assert net.module_.n_chans == 3
    assert net.module_.n_outputs == n_outputs


def test_set_signal_params_concat_ds_channels(eegneuralnet_cls, preds, concat_dataset_channels):
    n_outputs = (1 if eegneuralnet_cls == EEGRegressor else 4)
    net = eegneuralnet_cls(
        MockModuleFinalLayer,
        module__preds=preds,
        module__n_outputs=n_outputs,
        cropped=False,
        optimizer=optim.Adam,
        batch_size=32,
        train_split=None,
        max_epochs=1,
    )
    net.fit(concat_dataset_channels, y=None)
    assert net.module_.n_times == 10
    assert net.module_.n_chans == 3
    assert net.module_.n_outputs == n_outputs


def test_initialized_module(eegneuralnet_cls, preds, caplog, Xy):
    X, y = Xy
    module = MockModuleReturnMockedPreds(
        preds=preds,
        n_outputs=12,
        n_chans=12,
        n_times=12,
    )
    net = eegneuralnet_cls(
        module,
        cropped=False,
        max_epochs=1,
        train_split=None,
    )
    with caplog.at_level(logging.INFO):
        net.fit(X, y)
    assert "The module passed is already initialized" in caplog.text
    assert net.module_.n_outputs == 12
    assert net.module_.n_chans == 12
    assert net.module_.n_times == 12


# @pytest.mark.parametrize("model_name,model_cls", models_dict.items())
def test_module_name(eegneuralnet_cls):
    net = eegneuralnet_cls(
        "ShallowFBCSPNet",
        module__n_outputs=4,
        module__n_chans=3,
        module__n_times=100,
        cropped=False,
    )
    net.initialize()
    assert isinstance(net.module_, ShallowFBCSPNet)


def test_unknown_module_name(eegneuralnet_cls):
    net = eegneuralnet_cls(
        "InexistentModel",
    )
    with pytest.raises(ValueError) as excinfo:
        net.initialize()
    assert "Unknown model name" in str(excinfo.value)
