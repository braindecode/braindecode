# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Sarthak Tayal <sarthaktayal2@gmail.com>
#
# License: BSD-3
import logging
import warnings

import mne
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import softmax
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from skorch.callbacks import LRScheduler
from skorch.dataset import Dataset, ValidSplit
from skorch.exceptions import NotInitializedError
from skorch.helper import SliceDataset, predefined_split
from skorch.utils import to_tensor
from torch import optim
from torch.nn.functional import nll_loss

from braindecode import EEGClassifier, EEGRegressor
from braindecode.augmentation import AugmentedDataLoader, LabelSmoothing, Mixup
from braindecode.datasets import BaseConcatDataset, RawDataset, WindowsDataset
from braindecode.eegneuralnet import _EEGNeuralNet
from braindecode.models.base import EEGModuleMixin

# from braindecode.models.util import models_dict
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.samplers import SequenceSampler
from braindecode.training import CroppedLoss, mixup_criterion
from braindecode.training.scoring import CroppedTrialEpochScoring, predict_trials


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


class MockModuleCroppedPreds(EEGModuleMixin, torch.nn.Module):
    # keeps the time axis so the output looks like a cropped/dense prediction
    def __init__(
        self,
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
        self.final_layer = torch.nn.Conv1d(
            self.n_chans, self.n_outputs, self.n_times - 1
        )

    def forward(self, x):
        return self.final_layer(x)


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
def windows_dataset_channels(epochs, Xy):
    _, y = Xy
    target = mne.EpochsArray(
        np.repeat(y[:, None, None], epochs.get_data().shape[-1], axis=-1).astype(float),
        info=mne.create_info(["target"], sfreq=10, ch_types="misc"),
        metadata=epochs.metadata,
    )
    return WindowsDataset(
        windows=epochs.copy().add_channels([target]),
        targets_from="channels",
        description={},
    )


@pytest.fixture
def slice_dataset(windows_dataset_channels):
    X = SliceDataset(windows_dataset_channels)
    return X


@pytest.fixture
def cropped_windows_dataset():
    # two trials of two overlapping windows each, what predict_trials expects
    X = np.random.RandomState(20200101).rand(4, 3, 250).astype(np.float32)
    metadata = pd.DataFrame(
        [
            (0, 0, 0, 249),
            (0, 1, 2, 251),
            (1, 0, 0, 249),
            (1, 1, 2, 251),
        ],
        columns=["target", "i_window_in_trial", "i_start_in_trial", "i_stop_in_trial"],
    )
    epochs = mne.EpochsArray(
        X,
        info=mne.create_info(ch_names=["ch1", "ch2", "ch3"], sfreq=10, ch_types="eeg"),
        metadata=metadata,
    )
    windows = WindowsDataset(windows=epochs, targets_from="metadata", description={})
    return BaseConcatDataset([windows])


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
    target_predict = (
        preds if isinstance(eegneuralnet, EEGRegressor) else preds.argmax(1)
    )
    preds = preds if isinstance(eegneuralnet, EEGRegressor) else softmax(preds, axis=1)
    np.testing.assert_array_equal(target_predict, eegneuralnet.predict(MockDataset()))
    np.testing.assert_allclose(preds, eegneuralnet.predict_proba(MockDataset()))


@pytest.mark.parametrize("spatial_shape", [(2,), (2, 3)])
def test_cross_entropy_predict_proba_normalizes_classes_at_each_position(spatial_shape):
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(5, 4, *spatial_shape))
    net = EEGClassifier(
        MockModuleReturnMockedPreds,
        module__preds=logits,
        module__n_outputs=4,
        module__n_chans=3,
        module__n_times=10,
    ).initialize()
    np.testing.assert_allclose(
        net.predict_proba(MockDataset()), softmax(logits, axis=1)
    )
    np.testing.assert_array_equal(net.predict(MockDataset()), logits.argmax(axis=1))
    net.set_params(predict_nonlinearity=None)
    np.testing.assert_array_equal(net.predict_proba(MockDataset()), logits)


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
    target_predict = (
        preds if isinstance(eegneuralnet, EEGRegressor) else preds.argmax(1)
    )
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
        match="This method was designed to predict trials in cropped mode.",
    ):
        eegneuralnet.predict_trials(MockDataset(), return_targets=False)


@pytest.mark.parametrize("as_string", [False, True])
def test_predict_trials_module_not_instantiated(
    eegneuralnet_cls, cropped_windows_dataset, as_string
):
    # module given as a name or a class, so only module_ holds the built model
    module = "ShallowFBCSPNet" if as_string else MockModuleCroppedPreds
    kwargs = dict(module__final_conv_length=2) if as_string else dict()
    eegneuralnet = eegneuralnet_cls(
        module,
        module__n_outputs=2,
        module__n_chans=3,
        module__n_times=250,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=nll_loss,
        optimizer=optim.Adam,
        batch_size=4,
        **kwargs,
    )
    eegneuralnet.initialize()
    trial_preds, trial_targets = eegneuralnet.predict_trials(cropped_windows_dataset)
    assert trial_preds.shape[0] == 2
    assert trial_preds.shape[1] == 2
    assert trial_targets.shape[0] == 2
    # same thing the functional helper gives for the built module, same batching
    expected_preds, expected_targets = predict_trials(
        eegneuralnet.module_,
        cropped_windows_dataset,
        batch_size=eegneuralnet.batch_size,
    )
    # float32 convolutions wobble a bit between backends, keep this loose
    np.testing.assert_allclose(trial_preds, expected_preds, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(trial_targets, expected_targets)


def test_predict_with_window_inds_and_ys_module_not_instantiated(
    eegneuralnet_cls, cropped_windows_dataset
):
    # this is the path the cropped train scoring callbacks go through
    eegneuralnet = eegneuralnet_cls(
        MockModuleCroppedPreds,
        module__n_outputs=2,
        module__n_chans=3,
        module__n_times=250,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=nll_loss,
        optimizer=optim.Adam,
        batch_size=4,
    )
    eegneuralnet.initialize()
    results = eegneuralnet.predict_with_window_inds_and_ys(cropped_windows_dataset)
    assert results["preds"].shape[0] == 4
    assert results["i_window_in_trials"].shape[0] == 4
    assert results["i_window_stops"].shape[0] == 4
    assert results["window_ys"].shape[0] == 4


def test_predict_with_window_inds_and_ys_calls_the_module(
    eegneuralnet_cls, cropped_windows_dataset
):
    # module_(X), not module_.forward(X): hooks such as a channel layer run
    eegneuralnet = eegneuralnet_cls(
        MockModuleCroppedPreds,
        module__n_outputs=2,
        module__n_chans=3,
        module__n_times=250,
        cropped=True,
        batch_size=4,
    )
    eegneuralnet.initialize()
    calls = []
    eegneuralnet.module_.register_forward_pre_hook(lambda *_: calls.append(1))
    eegneuralnet.predict_with_window_inds_and_ys(cropped_windows_dataset)
    assert calls


def test_predict_trials_sets_eval_mode(eegneuralnet_cls, cropped_windows_dataset):
    eegneuralnet = eegneuralnet_cls(
        MockModuleCroppedPreds,
        module__n_outputs=2,
        module__n_chans=3,
        module__n_times=250,
        cropped=True,
        batch_size=4,
    )
    eegneuralnet.initialize()
    eegneuralnet.module_.train()
    eegneuralnet.predict_trials(cropped_windows_dataset, return_targets=False)
    assert not eegneuralnet.module_.training


def test_predict_trials_not_initialized(eegneuralnet_cls, cropped_windows_dataset):
    eegneuralnet = eegneuralnet_cls(
        MockModuleCroppedPreds,
        module__n_outputs=2,
        module__n_chans=3,
        module__n_times=250,
        cropped=True,
        batch_size=4,
    )
    with pytest.raises(NotInitializedError):
        eegneuralnet.predict_trials(cropped_windows_dataset)


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


@pytest.mark.filterwarnings("ignore:The training set has")
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


@pytest.mark.filterwarnings("ignore:The training set has")
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


@pytest.mark.filterwarnings("ignore:The training set has")
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


@pytest.mark.filterwarnings("ignore:The training set has")
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


@pytest.mark.filterwarnings("ignore:The training set has")
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


@pytest.mark.filterwarnings("ignore:The training set has")
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


@pytest.mark.filterwarnings("ignore:The training set has")
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


@pytest.mark.filterwarnings("ignore:The training set has")
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

    preds, targets = eeg_regressor.predict_trials(X, return_targets=True)
    assert preds.shape[0] == len(X)
    # X is a plain array here, so X[i][1] is the second row of each example
    assert np.array_equal(targets, np.stack([X[i][1] for i in range(len(X))]))


def test_EEGRegressor_predict_trials_scalar_targets():
    # windows datasets yield scalar targets, which cannot be concatenated
    rng = np.random.RandomState(0)
    info = mne.create_info(4, 100.0, "eeg")
    datasets = [
        RawDataset(
            mne.io.RawArray(rng.randn(4, 1000), info, verbose=False),
            pd.Series({"target": float(i)}),
            target_name="target",
        )
        for i in range(2)
    ]
    windows = create_fixed_length_windows(
        BaseConcatDataset(datasets),
        window_size_samples=200,
        window_stride_samples=200,
        drop_last_window=True,
    )
    regressor = EEGRegressor(
        ShallowFBCSPNet,
        module__n_chans=4,
        module__n_outputs=1,
        module__n_times=200,
        module__final_conv_length="auto",
        cropped=False,
        batch_size=4,
        train_split=None,
        max_epochs=1,
    )
    regressor.fit(windows, y=None)
    with pytest.warns(UserWarning, match="designed to predict trials in cropped"):
        _, targets = regressor.predict_trials(windows)
    np.testing.assert_array_equal(targets, [0.0] * 5 + [1.0] * 5)


class ConcreteEEGNeuralNet(_EEGNeuralNet):
    @property
    def mode(self):
        """"""


@pytest.fixture()
def net():
    net = ConcreteEEGNeuralNet(
        module="EEGNet",
        criterion=CroppedTrialEpochScoring,
        cropped=False,
        max_epochs=1,
        train_split=None,
        n_times=5,
    )
    return net


def test_cropped_trial_epoch_scoring(net):
    train_scoring = net._parse_str_callback("accuracy")[0][1]
    valid_scoring = net._parse_str_callback("accuracy")[1][1]

    assert train_scoring.on_train is True
    assert train_scoring.name == "train_accuracy"

    assert valid_scoring.on_train is False
    assert valid_scoring.name == "valid_accuracy"


def test_concrete_eegneuralnet_get_iterator_accepts_drop_index():
    sub_net = ConcreteEEGNeuralNet(
        module=MockModuleFinalLayer(
            np.zeros((2, 2), dtype=np.float32),
            n_outputs=2,
            n_chans=2,
            n_times=10,
        ),
        criterion=torch.nn.CrossEntropyLoss,
        max_epochs=1,
        batch_size=2,
        train_split=None,
    )
    sub_net.initialize()
    ds = torch.utils.data.TensorDataset(
        torch.zeros(4, 2, 10), torch.zeros(4, dtype=torch.long)
    )
    loader = sub_net.get_iterator(ds, training=False, drop_index=False)
    batches = list(loader)
    assert len(batches) == 2
    res = sub_net.predict_with_window_inds_and_ys(ds)
    assert res["preds"].shape == (4, 2)
    assert res["i_window_in_trials"] is None


@pytest.mark.filterwarnings("ignore:The training set has")
def test_set_signal_params_slice_dataset(eegneuralnet_cls, preds, slice_dataset):
    if eegneuralnet_cls != EEGClassifier:
        n_outputs = 1
        y_train = np.array([0, 1, 2, 3, 4])
    else:
        n_outputs = 5
        y_train = np.array([0, 1, 2, 3, 4])  # dummy values for y_train

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


@pytest.mark.parametrize(
    "batch_size,should_warn",
    [(6, True), (32, True), (2, False), (5, False)],
)
def test_drop_last_small_trainset_warns(
    eegneuralnet_cls, preds, Xy, batch_size, should_warn
):
    # Xy has 5 training examples. iterator_train__drop_last=True is the
    # braindecode default, so a batch_size larger than the training set drops
    # every batch and silently skips training. We expect a UserWarning telling
    # the user to lower batch_size or disable drop_last; batch_size <= 5 keeps
    # at least one batch and must not warn.
    X, y = Xy
    net = eegneuralnet_cls(
        MockModuleFinalLayer,
        module__preds=preds,
        cropped=False,
        optimizer=optim.Adam,
        batch_size=batch_size,
        train_split=None,
        max_epochs=1,
    )
    match = "all training batches are dropped"
    if should_warn:
        with pytest.warns(UserWarning, match=match):
            net.fit(X, y=y)
    else:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            net.fit(X, y=y)
        assert not any(match in str(w.message) for w in records)


def _fit_regressor_on_dataset():
    from braindecode.datasets import create_from_X_y

    rng = np.random.RandomState(0)
    X = rng.randn(8, 4, 250).astype("float32")
    y = rng.randn(8).astype("float32")
    dataset = create_from_X_y(X, y, drop_last_window=False, sfreq=100)
    net = EEGRegressor(
        ShallowFBCSPNet,
        module__final_conv_length="auto",
        train_split=None,
        batch_size=8,
        max_epochs=1,
        verbose=0,
    )
    return net, dataset, net.fit(dataset, y=None)


def test_eegregressor_loss_is_per_trial_mse_for_dataset_targets():
    net, dataset, _ = _fit_regressor_on_dataset()
    X_batch, y_batch = next(iter(net.get_iterator(net.get_dataset(dataset))))
    y_pred = net.infer(X_batch).detach()
    assert y_pred.shape == (8, 1) and y_batch.shape == (8,)
    expected = ((y_pred.squeeze(1) - y_batch) ** 2).mean()
    assert net.get_loss(y_pred, y_batch).item() == pytest.approx(expected.item())


def test_eegregressor_fit_returns_self():
    net, _, fitted = _fit_regressor_on_dataset()
    assert fitted is net


@pytest.mark.parametrize("from_dataset", [False, True], ids=["numpy", "dataset"])
def test_eegregressor_cropped_loss_is_per_trial_mse(from_dataset):
    # fit reshapes a 1-D numpy y to (n, 1), a dataset yields (n,) targets.
    # Either way each time-averaged prediction must meet its own target.
    from braindecode.datasets import create_from_X_y

    n_trials, n_chans, n_times = 8, 4, 300
    rng = np.random.RandomState(0)
    X = rng.randn(n_trials, n_chans, n_times).astype("float32")
    y = rng.randn(n_trials).astype("float32")
    torch.manual_seed(0)
    module = ShallowFBCSPNet(
        n_chans=n_chans, n_outputs=1, n_times=n_times, final_conv_length=2
    )
    module.to_dense_prediction_model()
    net = EEGRegressor(
        module,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.mse_loss,
        train_split=None,
        batch_size=n_trials,
        max_epochs=1,
        verbose=0,
    )
    if from_dataset:
        data = create_from_X_y(
            X,
            y,
            drop_last_window=False,
            sfreq=100,
            window_size_samples=n_times,
            window_stride_samples=module.get_output_shape()[2],
        )
        net.fit(data, y=None)
        dataset = net.get_dataset(data)
    else:
        net.fit(X, y)
        dataset = net.get_dataset(X, y.reshape(-1, 1))
    X_batch, y_batch = next(iter(net.get_iterator(dataset)))[:2]
    with torch.no_grad():
        y_pred = net.infer(X_batch)
        loss = net.get_loss(y_pred, y_batch)
    per_trial_pred = y_pred.mean(dim=2).reshape(-1)
    expected = ((per_trial_pred - y_batch.reshape(-1).float()) ** 2).mean()
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("validation", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cropped_array_grid_search_accuracy(pipeline, validation, dtype):
    X = np.random.RandomState(438).randn(20, 3, 10).astype(dtype)
    y = np.tile([0, 1], 10)
    net = EEGClassifier(
        MockModuleCroppedPreds,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.cross_entropy,
        callbacks=["accuracy"],
        train_split=ValidSplit(2, stratified=True) if validation else None,
        batch_size=3,
        iterator_train__drop_last=False,
        max_epochs=2,
        verbose=0,
    )
    estimator = (
        Pipeline([("identity", FunctionTransformer()), ("net", net)])
        if pipeline
        else net
    )
    grid = GridSearchCV(
        estimator,
        {"net__lr" if pipeline else "lr": [0.01]},
        cv=2,
        scoring="accuracy",
        error_score="raise",
    ).fit(X, y)
    fitted = grid.best_estimator_["net"] if pipeline else grid.best_estimator_
    train, valid = fitted.get_split_datasets(X, y)
    for name, dataset in [("train", train), ("valid", valid)]:
        if dataset is None:
            continue
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
        results = fitted.predict_with_window_inds_and_ys(dataset)
        assert results["i_window_in_trials"] is None
        assert results["i_window_stops"] is None
        predictions = fitted.forward(dataset).mean(-1).argmax(1).numpy()
        assert fitted.history[-1, f"{name}_accuracy"] == accuracy_score(
            targets, predictions
        )
    assert grid.predict(X).shape == y.shape
    assert np.isfinite(grid.cv_results_["mean_test_score"]).all()
    cloned = clone(fitted)
    assert cloned.callbacks == ["accuracy"]
    assert not cloned.initialized_
    assert net.callbacks == ["accuracy"]
    assert not net.initialized_


@pytest.mark.parametrize("validation", ["none", "windows", "array"])
def test_cropped_window_accuracy_preserves_trial_indices(
    cropped_windows_dataset, validation
):
    dataset = cropped_windows_dataset
    valid = dataset
    if validation == "array":
        valid = Dataset(
            np.stack([dataset[i][0] for i in range(len(dataset))]),
            np.array([dataset[i][1] for i in range(len(dataset))]),
        )
    callbacks = ["accuracy"] + [
        CroppedTrialEpochScoring(
            lambda net, X, y: len(y), name=f"{name}_n_trials", on_train=on_train
        )
        for name, on_train in [("train", True), ("valid", False)]
        if on_train or validation != "none"
    ]
    net = EEGClassifier(
        MockModuleCroppedPreds,
        cropped=True,
        classes=[0, 1],
        module__n_times=249,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.cross_entropy,
        callbacks=callbacks,
        train_split=predefined_split(valid) if validation != "none" else None,
        batch_size=3,
        iterator_train__drop_last=False,
        max_epochs=2,
        verbose=0,
    ).fit(dataset, y=None)
    result = net.predict_with_window_inds_and_ys(dataset)
    np.testing.assert_array_equal(result["i_window_in_trials"], [0, 1, 0, 1])
    np.testing.assert_array_equal(result["i_window_stops"], [249, 251, 249, 251])
    preds, targets = net.predict_trials(dataset)
    expected = accuracy_score(targets, preds.mean(-1).argmax(1))
    assert net.history[-1, "train_accuracy"] == expected
    assert preds.shape == (2, 2, 5)  # 3 + 2 predictions, overlap removed
    assert net.history[-1, "train_n_trials"] == 2
    if validation != "none":
        if validation == "array":
            expected = accuracy_score(
                valid.y, net.forward(valid).mean(-1).argmax(1).numpy()
            )
        assert net.history[-1, "valid_accuracy"] == expected
        assert net.history[-1, "valid_n_trials"] == (4 if validation == "array" else 2)


@pytest.mark.parametrize("validation", [False, True])
@pytest.mark.parametrize("kind", ["float32", "float64", "dict"])
def test_cropped_array_callback_input_contract(kind, validation):
    X = np.random.RandomState(438).randn(20, 3, 10)
    if kind == "float32":
        X = X.astype(np.float32)
    if kind == "dict":
        X = {"x": X}
    y = np.tile([0, 1], 10)
    net = EEGClassifier(
        MockModuleCroppedPreds,
        module__n_chans=3,
        module__n_outputs=2,
        module__n_times=10,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.cross_entropy,
        train_split=ValidSplit(2, stratified=True) if validation else None,
        iterator_train__drop_last=False,
        batch_size=3,
        max_epochs=2,
        lr=0.0,
        verbose=0,
    )
    for callbacks in ([], ["accuracy"]):
        torch.manual_seed(438)
        fitted = clone(net).set_params(callbacks=callbacks).fit(X, y)
        predictions = fitted.predict(X)
        if not callbacks:
            control = predictions
        else:
            np.testing.assert_array_equal(predictions, control)
            if not validation:
                assert fitted.history[-1, "train_accuracy"] == accuracy_score(
                    y, predictions
                )
        assert predictions.shape == y.shape


@pytest.mark.parametrize("train_indexed", [False, True])
@pytest.mark.parametrize("valid_indexed", [False, True])
def test_cropped_raw_means_and_cache_reset(train_indexed, valid_indexed):
    X = np.array(
        [
            [[9, 9, 9], [0, 0, 0]],
            [[1000, 0, 0], [-1000, 9, 9]],
            [[1, 2, 3], [5, 4, 3]],
            [[999, 4, 5], [999, 2, 1]],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 1, 1])
    array = Dataset(X, y)

    # The first prediction in each second window overlaps its predecessor.
    class IndexedWindows(Dataset):
        def __getitem__(self, index):
            x, target = super().__getitem__(index)
            i = index % 2
            return x, target, (i, 2 * i, 3 + 2 * i)

    indexed = IndexedWindows(X, y)
    seen = []

    def score(net, dataset, targets):
        is_indexed = dataset is indexed
        expected = [[27 / 5, 18 / 5], [3, 3]] if is_indexed else X.mean(-1)
        np.testing.assert_allclose(net.forward(dataset), expected, rtol=1e-6)
        np.testing.assert_array_equal(targets, [0, 1] if is_indexed else y)
        seen.append(len(targets))
        return len(targets)

    # A fixed identity convolution keeps raw numeric expectations independent
    # of optimization while retaining a real trainable module/backward pass.
    module = torch.nn.Conv1d(2, 2, 1, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.eye(2).unsqueeze(-1))
    net = EEGClassifier(
        module,
        cropped=True,
        classes=[0, 1],
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.cross_entropy,
        callbacks=[
            CroppedTrialEpochScoring(score, name=name, on_train=on_train)
            for name, on_train in [("train_raw", True), ("valid_raw", False)]
        ],
        train_split=predefined_split(indexed if valid_indexed else array),
        batch_size=3,
        iterator_train__shuffle=True,
        iterator_train__drop_last=False,
        max_epochs=2,
        lr=0.0,
        verbose=0,
    ).fit(indexed if train_indexed else array, y=None)
    assert seen == [2 if train_indexed else 4, 2 if valid_indexed else 4] * 2
    assert "forward_iter" not in net.__dict__
    for _, callback in net.callbacks_:
        if isinstance(callback, CroppedTrialEpochScoring):
            assert callback.y_preds_ == callback.y_trues_ == []
            assert not callback.crops_to_trials_computed
            if not callback.on_train:
                assert callback.window_inds_ == []


@pytest.mark.parametrize("estimator", [EEGClassifier, EEGRegressor])
def test_predict_trials_explicit_loader_kwargs(estimator):
    class Windows(torch.utils.data.Dataset):
        def __len__(self):
            return 5

        def __getitem__(self, index):
            return (
                torch.ones(1, 4),
                index // 3,
                [index % 3, 2 * (index % 3), 4 + 2 * (index % 3)],
            )

        def get_metadata(self):
            return pd.DataFrame({"i_window_in_trial": [0, 1, 2, 0, 1]})

    class RecordingModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))
            self.sizes = []

        def forward(self, X):
            self.sizes.append(len(X))
            return X * self.scale

    net = estimator(
        RecordingModule(),
        cropped=True,
        batch_size=3,
        train_split=None,
        iterator_valid__batch_size=4,
        iterator_valid__drop_last=True,
    )
    net.initialize()
    X = Windows()
    net.predict_trials(X, loader_kwargs={"batch_size": 2})
    assert net.module_.sizes == [2, 2, 1]
    net.module_.sizes.clear()
    net.predict_trials(X)
    assert net.module_.sizes == [3, 2]
    net.cropped = False
    with pytest.raises(ValueError, match="cropped"):
        net.predict_trials(X, loader_kwargs={"batch_size": 2})


def _per_trial_logits():
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(16, 3))  # (batch, 3)


def _dense_logits():
    return torch.nn.Conv1d(2, 3, kernel_size=1)  # (batch, 3, 8)


@pytest.mark.parametrize(
    "y, transforms, model, dtype",
    [
        (torch.tensor([0.0, 1.0, 2.0, 1.0]), [], _per_trial_logits, torch.int64),
        # a layout that differs from the logits' is class indices, not probabilities
        (
            torch.arange(32).reshape(4, 8).remainder(3).float(),
            [],
            _dense_logits,
            torch.int64,
        ),
        (
            torch.tensor([0.0, 1.0, 2.0, 1.0]),
            [Mixup(0.5)],
            _per_trial_logits,
            torch.int64,
        ),
        (torch.eye(3)[[0, 1, 2, 1]], [], _per_trial_logits, torch.float32),
        # dense probabilities (batch, classes, time) have the shape of the logits
        (
            torch.arange(96.0).reshape(4, 3, 8).softmax(dim=1),
            [],
            _dense_logits,
            torch.float32,
        ),
        (
            torch.tensor([0, 1, 2, 1]),
            [LabelSmoothing(0.1, 3)],
            _per_trial_logits,
            torch.float32,
        ),
        (
            torch.tensor([0, 1, 2, 1]),
            [LabelSmoothing(0.1, 3), Mixup(0.5)],
            _per_trial_logits,
            torch.float32,
        ),
    ],
    ids=[
        "float_indices",
        "dense_float_indices",
        "float_indices_mixup",
        "one_hot",
        "dense_probabilities",
        "smoothed",
        "smoothed_mixup",
    ],
)
def test_classifier_target_dtypes(y, transforms, model, dtype):
    received = []

    class RecordingLoss(torch.nn.Module):
        def forward(self, logits, target):
            # Mixup gives the targets of both mixed trials and their weights as a list.
            received.append((target[0] if isinstance(target, list) else target).dtype)
            return mixup_criterion(logits, target)

    EEGClassifier(
        model(),
        criterion=RecordingLoss,
        iterator_train=AugmentedDataLoader,
        iterator_train__transforms=transforms,
        train_split=None,
        max_epochs=1,
        batch_size=len(y),
        verbose=0,
    ).fit(torch.randn(len(y), 2, 8), y)
    assert received == [dtype]


class _SequenceLinear(torch.nn.Module):
    """(batch, n_windows, n_chans, n_times) to (batch, n_classes, n_windows)."""

    def __init__(self, n_features, n_classes):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.linear(x.flatten(start_dim=2)).transpose(1, 2)


@pytest.mark.parametrize("layout", ["last_target", "all_targets", "sequences"])
def test_channel_target_classification_trains(layout):
    # Window targets sliced out of the float32 signal are floats holding class
    # indices: (batch, 1) or (batch, 1, n_times) from a misc channel of sleep
    # stages, (batch, 3) for float labels read in sequences of 3 windows.
    n_windows, n_times = 8, 8
    rng = np.random.RandomState(0)
    fit_kwargs = {}
    if layout == "sequences":
        info = mne.create_info(["C3", "C4"], 100.0, "eeg")
        recordings = [
            RawDataset(
                mne.io.RawArray(rng.randn(2, n_windows * n_times), info),
                {"label": float(i % 3)},
                target_name="label",
            )
            for i in range(4)
        ]
        windows = create_fixed_length_windows(
            BaseConcatDataset(recordings),
            window_size_samples=n_times,
            window_stride_samples=n_times,
            on_last_window="drop",
        )
        model = _SequenceLinear(2 * n_times, 3)
        fit_kwargs = dict(
            iterator_train__sampler=SequenceSampler(windows.get_metadata(), 3, 1),
            iterator_train__shuffle=False,
        )
    else:
        # one integer stage per window in the misc channel
        data = rng.randn(3, n_windows * n_times)
        data[2] = np.repeat(np.arange(n_windows) % 3, n_times)
        info = mne.create_info(["C3", "C4", "stage"], 100.0, ["eeg", "eeg", "misc"])
        windows = create_fixed_length_windows(
            BaseConcatDataset([RawDataset(mne.io.RawArray(data, info))]),
            window_size_samples=n_times,
            window_stride_samples=n_times,
            on_last_window="drop",
            targets_from="channels",
            last_target_only=layout == "last_target",
        )
        if layout == "last_target":
            model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(16, 3))
        else:
            model = torch.nn.Conv1d(2, 3, kernel_size=1)

    received = []

    class IndexCrossEntropy(torch.nn.Module):
        def forward(self, logits, target):
            received.append(target.dtype)
            # class indices laid out as the logits without their class axis
            target = target.reshape(logits.shape[:1] + logits.shape[2:])
            return torch.nn.functional.cross_entropy(logits, target)

    net = EEGClassifier(
        model,
        criterion=IndexCrossEntropy,
        train_split=None,
        max_epochs=1,
        batch_size=4,
        verbose=0,
        **fit_kwargs,
    ).fit(windows, y=None)
    assert received and set(received) == {torch.int64}
    assert np.isfinite(net.history[-1, "train_loss"])


class _LinearHead(torch.nn.Module):
    def __init__(self, n_chans, n_times, n_outputs):
        super().__init__()
        self.linear = torch.nn.Linear(n_chans * n_times, n_outputs)

    def forward(self, x):
        return self.linear(x.flatten(start_dim=1))


@pytest.mark.parametrize("data", ["one_hot", "dirichlet", "windows"])
def test_soft_labels_with_n_outputs_and_classes(data):
    # Class probabilities are not class names and do not size the model: the
    # n_outputs the user passes is used as is, and classes names the stored labels.
    rng = np.random.RandomState(0)
    soft = np.eye(3, dtype="float32") * 0.8 + 0.1
    if data != "windows":
        X = rng.randn(6, 2, 8).astype("float32")
        if data == "one_hot":
            y = np.eye(3, dtype="float32")[[0, 1, 2] * 2]
        else:
            y = rng.dirichlet(np.ones(3), size=6).astype("float32")
        classes = [0, 1, 2]
    else:
        info = mne.create_info(["C3", "C4"], 100.0, "eeg")
        raw = mne.io.RawArray(rng.randn(2, 16), info)
        labels = ["left", "right", "rest"]
        recordings = [
            RawDataset(raw, {"label": label}, target_name="label") for label in labels
        ]
        X = create_fixed_length_windows(
            BaseConcatDataset(recordings),
            window_size_samples=8,
            window_stride_samples=8,
            on_last_window="drop",
        )
        X.target_transform = dict(zip(labels, soft)).__getitem__
        y, classes = None, labels
    net = EEGClassifier(
        _LinearHead,
        module__n_outputs=3,
        classes=classes,
        train_split=None,
        max_epochs=1,
        batch_size=2,
        verbose=0,
    ).fit(X, y)
    assert net.module_.linear.out_features == 3


def test_get_split_datasets_works_before_initialize():
    """skorch's public ``get_split_datasets`` needs no initialized module (v2.0.0)."""
    X = np.random.randn(8, 3, 200).astype("float32")
    clf = EEGClassifier(
        ShallowFBCSPNet,
        module__n_chans=3,
        module__n_times=200,
        module__n_outputs=2,
        train_split=None,
    )
    dataset_train, dataset_valid = clf.get_split_datasets(X, np.array([0, 1] * 4))
    assert len(dataset_train) == 8 and dataset_valid is None
