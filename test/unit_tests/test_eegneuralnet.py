# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD-3
import logging

import mne
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import softmax
from sklearn.base import clone
from skorch.callbacks import LRScheduler
from skorch.helper import SliceDataset
from skorch.utils import to_tensor
from torch import optim
from torch.nn.functional import nll_loss

from braindecode import EEGClassifier, EEGRegressor
from braindecode.datasets import BaseConcatDataset, WindowsDataset
from braindecode.eegneuralnet import _EEGNeuralNet
from braindecode.models.base import EEGModuleMixin

# from braindecode.models.util import models_dict
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.training import CroppedLoss


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
        self.preds = to_tensor(preds, device="cpu")
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
        columns=["target", "i_window_in_trial", "i_start_in_trial", "i_stop_in_trial"],
    )
    return mne.EpochsArray(
        X,
        info=mne.create_info(
            ch_names=[
                "ch1",
                "ch2",
                "ch3",
            ],
            sfreq=10,
            ch_types="eeg",
        ),
        metadata=metadata,
    )


@pytest.fixture
def windows_dataset_metadata(epochs):
    return WindowsDataset(
        windows=epochs,
        targets_from="metadata",
        description={},
    )


@pytest.fixture
def windows_dataset_channels(epochs):
    return WindowsDataset(
        windows=epochs,
        targets_from="channels",
        description={},
    )

@pytest.fixture
def slice_dataset(windows_dataset_channels):
    X = SliceDataset(windows_dataset_channels)
    return X

@pytest.fixture
def concat_dataset_metadata(windows_dataset_metadata):
    return BaseConcatDataset([windows_dataset_metadata, windows_dataset_metadata])


@pytest.fixture
def concat_dataset_channels(
    windows_dataset_metadata,
    windows_dataset_channels,
):
    return BaseConcatDataset([windows_dataset_metadata, windows_dataset_channels])


def test_trialwise_predict_and_predict_proba(eegneuralnet_cls):
    preds = np.array(
        [
            [0.125, 0.875],
            [1.0, 0.0],
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
        batch_size=32,
    )
    eegneuralnet.initialize()
    target_predict = preds if isinstance(eegneuralnet, EEGRegressor) else preds.argmax(1)
    preds = preds if isinstance(eegneuralnet, EEGRegressor) else softmax(preds, axis=1)
    np.testing.assert_array_equal(target_predict, eegneuralnet.predict(MockDataset()))
    np.testing.assert_allclose(preds, eegneuralnet.predict_proba(MockDataset()))


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
        batch_size=32,
    )
    eegneuralnet.initialize()
    target_predict = (
        preds.mean(-1)
        if isinstance(eegneuralnet, EEGRegressor)
        else preds.mean(-1).argmax(1)
    )
    # for cropped decoding classifier returns one label for each trial (averaged over all crops)
    np.testing.assert_array_equal(target_predict, eegneuralnet.predict(MockDataset()))
    # for cropped decoding classifier returns values for each trial (average over all crops)
    np.testing.assert_array_equal(
        preds.mean(-1), eegneuralnet.predict_proba(MockDataset())
    )


def test_cropped_predict_and_predict_proba_not_aggregate_predictions(
    eegneuralnet_cls, preds
):
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
        aggregate_predictions=False,
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
        batch_size=32,
    )
    eegneuralnet.initialize()
    with pytest.warns(
        UserWarning,
        match="This method was designed to predict " "trials in cropped mode.",
    ):
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
            ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=1)),
        ],
        criterion=CroppedLoss,
        criterion__loss_function=nll_loss,
        optimizer=optim.Adam,
        batch_size=32,
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
    assert net.module_.chs_info == epochs.info["chs"]
    assert net.module_.input_window_seconds == 10 / 10
    assert net.module_.sfreq == 10


def test_set_signal_params_torch_ds(eegneuralnet_cls, preds):
    n_outputs = 1 if eegneuralnet_cls == EEGRegressor else 4
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


def test_set_signal_params_windows_ds_metadata(
    eegneuralnet_cls, preds, windows_dataset_metadata
):
    n_outputs = 1 if eegneuralnet_cls == EEGRegressor else 4
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


def test_set_signal_params_windows_ds_channels(
    eegneuralnet_cls, preds, windows_dataset_channels
):
    n_outputs = 1 if eegneuralnet_cls == EEGRegressor else 4
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


def test_set_signal_params_concat_ds_metadata(
    eegneuralnet_cls, preds, concat_dataset_metadata
):
    n_outputs = 1 if eegneuralnet_cls == EEGRegressor else 4
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


def test_set_signal_params_concat_ds_channels(
    eegneuralnet_cls, preds, concat_dataset_channels
):
    n_outputs = 1 if eegneuralnet_cls == EEGRegressor else 4
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


def test_EEGRegressor_drop_index(Xy):
    # Initialize EEGRegressor with drop_index=False
    X, y = Xy

    net = EEGRegressor(
        MockModuleFinalLayer,
        module__preds=preds,
        cropped=False,
        optimizer=optim.Adam,
        batch_size=32,
        train_split=None,
        max_epochs=1,
    )

    # Test if the iterator is returned when drop_index is False
    iterator = net.get_iterator(X, training=False, drop_index=False)
    assert isinstance(iterator, torch.utils.data.DataLoader)


def test_EEGRegressor_get_n_outputs(preds):
    # Initialize EEGRegressor

    eeg_regressor = EEGRegressor(
        MockModuleFinalLayer,
        module__preds=preds,
        cropped=False,
        optimizer=optim.Adam,
        batch_size=2,
        train_split=None,
        max_epochs=1,
    )

    # Test _get_n_outputs method
    assert eeg_regressor._get_n_outputs(y=None,
                                        classes=None) is None
    assert eeg_regressor._get_n_outputs(y=np.array([0, 1, 2, 3, 4]),
                                        classes=None) == 1
    assert eeg_regressor._get_n_outputs(y=np.array(
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),
        classes=None) == 5


def test_EEGRegressor_predict_trials(Xy, preds):
    X, y = Xy
    # Initialize EEGRegressor
    eeg_regressor = EEGRegressor(
        MockModuleFinalLayer,
        module__preds=preds,
        cropped=False,
        optimizer=optim.Adam,
        batch_size=2,
        train_split=None,
        max_epochs=1,
    )

    eeg_regressor.fit(X, y=y)

    preds, targets = eeg_regressor.predict_trials(X,
                                                  return_targets=True)
    assert preds.shape[0] == len(X)
    assert np.array_equal(targets, np.concatenate([X[i][1]
                                                  for i in range(len(X))]))
from braindecode.eegneuralnet import CroppedTrialEpochScoring


class ConcreteEEGNeuralNet(_EEGNeuralNet):
    def _get_n_outputs(self, y, classes):
        # provide your implementation here
        pass

@pytest.fixture()
def net():
    net = ConcreteEEGNeuralNet(module="EEGNetv4", criterion=CroppedTrialEpochScoring,
                               cropped=False, max_epochs=1, train_split=None,
                               n_times=5)
    return net


def test_cropped_trial_epoch_scoring(net):
    train_scoring = net._parse_str_callback('accuracy')[0][1]
    valid_scoring = net._parse_str_callback('accuracy')[1][1]

    assert train_scoring.on_train is True
    assert train_scoring.name == 'train_accuracy'

    assert valid_scoring.on_train is False
    assert valid_scoring.name == 'valid_accuracy'

def test_get_n_outputs():
    with pytest.raises(TypeError):
        _EEGNeuralNet()._get_n_outputs(None, None)


def test_set_signal_params_slice_dataset(
    eegneuralnet_cls, preds, slice_dataset
):
    if eegneuralnet_cls != EEGClassifier:
        n_outputs = 1
        y_train = np.array([0, 1, 2, 3, 4])
    else:
        n_outputs = 5
        y_train = np.array([0, 1, 2, 3, 4]) # dummy values for y_train

    net = eegneuralnet_cls(
        MockModuleFinalLayer,
        module__preds=preds,
        cropped=False,
        optimizer=optim.Adam,
        batch_size=32,
        train_split=None,
        max_epochs=1,
    )
    net.fit(slice_dataset, y=y_train)
    assert net.module_.n_times == 10
    assert net.module_.n_chans == 3
    assert net.module_.n_outputs == n_outputs
