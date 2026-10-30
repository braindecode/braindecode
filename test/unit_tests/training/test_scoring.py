# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Robin Tibor Schirrmeister <robintibor@gmail.com>
#          Sarthak Tayal <sarthaktayal2@gmail.com>
#
# License: BSD-3

import math

import numpy as np
import pandas as pd
import pytest
import sklearn.datasets
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from skorch import History
from skorch.callbacks import Callback
from skorch.utils import to_numpy, to_tensor
from torch import optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

from braindecode.classifier import EEGClassifier
from braindecode.datasets.moabb import MOABBDataset
from braindecode.datasets.xy import create_from_X_y
from braindecode.models import ShallowFBCSPNet
from braindecode.preprocessing import create_windows_from_events
from braindecode.training import CroppedLoss
from braindecode.training.scoring import (
    CroppedTimeSeriesEpochScoring,
    CroppedTrialEpochScoring,
    PostEpochTrainScoring,
    predict_trials,
    trial_preds_from_window_preds,
)
from braindecode.util import set_random_seeds

bnci_kwargs = {
    "n_sessions": 1,
    "n_runs": 1,
    "n_subjects": 1,
    "paradigm": "imagery",
    "duration": 386.9,
    "sfreq": 250,
    "event_list": ("left", "right"),
    "channels": ("C3", "Cz", "C1"),
}


class MockSkorchNet(sklearn.base.BaseEstimator):
    def __init__(self):
        self.device = "cpu"
        self.forward_iter = None
        self.history = History()
        self.history.new_epoch()
        self._default_callbacks = []

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.concatenate([to_numpy(x.argmax(1)) for x in self.forward_iter(X)], 0)

    def get_iterator(self, X_test, training):
        return DataLoader(X_test, batch_size=2)


def test_cropped_trial_epoch_scoring():

    dataset_train = None
    # Definition of test cases
    predictions_cases = [
        # Expected predictions classification results: [1, 0, 0, 0]
        np.array(
            [
                [[0.2, 0.1, 0.1, 0.1], [0.8, 0.9, 0.9, 0.9]],  # trial 0 preds
                [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],  # trial 1 preds
                [[1.0, 1.0, 1.0, 0.2], [0.0, 0.0, 0.0, 0.8]],  # trial 2 preds
                [[0.9, 0.8, 0.9, 1.0], [0.1, 0.2, 0.1, 0.0]],  # trial 3 preds
            ]
        ),
        # Expected predictions classification results: [1, 1, 1, 0]
        np.array(
            [
                [[0.2, 0.1, 0.1, 0.1], [0.8, 0.9, 0.9, 0.9]],
                [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
                [[0.0, 0.0, 0.0, 0.2], [1.0, 1.0, 1.0, 0.8]],
                [[0.9, 0.8, 0.9, 1.0], [0.1, 0.2, 0.1, 0.0]],
            ]
        ),
    ]
    y_true_cases = [
        [torch.tensor([0, 0]), torch.tensor([1, 1])],
        [torch.tensor([1, 1]), torch.tensor([1, 1])],
    ]
    expected_accuracies_cases = [0.25, 0.75]

    window_inds = [
        (
            torch.tensor([0, 0]),  # i_window_in_trials
            [None],  # won't be used
            torch.tensor([4, 4]),  # i_window_stops
        ),
        (
            torch.tensor([0, 0]),  # i_window_in_trials
            [None],  # won't be used
            torch.tensor([4, 4]),  # i_window_stops
        ),
    ]

    for predictions, y_true, accuracy in zip(
        predictions_cases, y_true_cases, expected_accuracies_cases
    ):
        dataset_valid = create_from_X_y(
            np.zeros((4, 1, 10)),
            np.concatenate(y_true),
            sfreq=100,
            window_size_samples=10,
            window_stride_samples=4,
            drop_last_window=False,
        )

        mock_skorch_net = MockSkorchNet()
        cropped_trial_epoch_scoring = CroppedTrialEpochScoring("accuracy", on_train=False)
        mock_skorch_net.callbacks_ = [("", cropped_trial_epoch_scoring)]
        cropped_trial_epoch_scoring.initialize()
        cropped_trial_epoch_scoring.y_preds_ = [
            to_tensor(predictions[:2], device="cpu"),
            to_tensor(predictions[2:], device="cpu"),
        ]
        cropped_trial_epoch_scoring.y_trues_ = y_true
        cropped_trial_epoch_scoring.window_inds_ = window_inds

        cropped_trial_epoch_scoring.on_epoch_end(
            mock_skorch_net, dataset_train, dataset_valid
        )

        np.testing.assert_almost_equal(mock_skorch_net.history[0]["accuracy"], accuracy)


def test_cropped_trial_epoch_scoring_none_x_test():
    dataset_train = None
    dataset_valid = None
    predictions = np.array(
        [
            [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
            [[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
        ]
    )
    y_true = [torch.tensor([0, 0]), torch.tensor([1, 1])]
    window_inds = [
        (
            torch.tensor([0, 0]),  # i_window_in_trials
            [None],  # won't be used
            torch.tensor([4, 4]),  # i_window_stops
        ),
        (
            torch.tensor([0, 0]),  # i_window_in_trials
            [None],  # won't be used
            torch.tensor([4, 4]),  # i_window_stops
        ),
    ]
    cropped_trial_epoch_scoring = CroppedTrialEpochScoring("accuracy")
    cropped_trial_epoch_scoring.initialize()
    cropped_trial_epoch_scoring.y_preds_ = [
        to_tensor(predictions[:2], device="cpu"),
        to_tensor(predictions[2:], device="cpu"),
    ]
    cropped_trial_epoch_scoring.y_trues_ = y_true
    cropped_trial_epoch_scoring.window_inds_ = window_inds

    mock_skorch_net = MockSkorchNet()
    mock_skorch_net.callbacks_ = [("", cropped_trial_epoch_scoring)]
    output = cropped_trial_epoch_scoring.on_epoch_end(
        mock_skorch_net, dataset_train, dataset_valid
    )
    assert output is None


def test_cropped_time_series_trial_epoch_scoring():

    dataset_train = None
    # Definition of test cases
    predictions_cases = [
        # Expected predictions classification results: [1, 0, 0, 0]
        np.array(
            [
                [[0.2, 0.1, 0.1, 0.1], [0.8, 0.9, 0.9, 0.9]],  # trial 0 preds
                [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],  # trial 1 preds
                [[1.0, 1.0, 1.0, 0.2], [0.0, 0.0, 0.0, 0.8]],  # trial 2 preds
                [[0.9, 0.8, 0.9, 1.0], [0.1, 0.2, 0.1, 0.0]],  # trial 3 preds
            ]
        ),
        # Expected predictions classification results: [1, 1, 1, 0]
        np.array(
            [
                [[0.2, 0.1, 0.1, 0.1], [0.8, 0.9, 0.9, 0.9]],
                [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
                [[0.0, 0.0, 0.0, 0.2], [1.0, 1.0, 1.0, 0.8]],
                [[0.9, 0.8, 0.9, 1.0], [0.1, 0.2, 0.1, 0.0]],
            ]
        ),
    ]

    y_true_cases = [
        [np.full((4, 1, 5), fill_value=np.nan)],
        [np.full((4, 1, 5), fill_value=np.nan)],
    ]
    y_true_cases[0][0][:, 0, 1] = [0, 0, 1, 1]
    y_true_cases[0] = torch.tensor(y_true_cases[0][0]).split(2)
    y_true_cases[1][0][:, 0, 1] = [1, 1, 1, 1]
    y_true_cases[1] = torch.tensor(y_true_cases[1][0]).split(2)
    expected_accuracies_cases = [0.25, 0.75]

    window_inds = [
        (
            torch.tensor([0, 0]),  # i_window_in_trials
            [None],  # won't be used
            torch.tensor([4, 4]),  # i_window_stops
        ),
        (
            torch.tensor([0, 0]),  # i_window_in_trials
            [None],  # won't be used
            torch.tensor([4, 4]),  # i_window_stops
        ),
    ]

    for predictions, y_true, accuracy in zip(
        predictions_cases, y_true_cases, expected_accuracies_cases
    ):
        dataset_valid = create_from_X_y(
            np.zeros((4, 1, 10)),
            np.concatenate(y_true),
            sfreq=100,
            window_size_samples=10,
            window_stride_samples=4,
            drop_last_window=False,
        )

        mock_skorch_net = MockSkorchNet()
        cropped_trial_epoch_scoring = CroppedTimeSeriesEpochScoring(
            "accuracy", on_train=False
        )
        mock_skorch_net.callbacks_ = [("", cropped_trial_epoch_scoring)]
        cropped_trial_epoch_scoring.initialize()
        cropped_trial_epoch_scoring.y_preds_ = [
            to_tensor(predictions[:2], device="cpu"),
            to_tensor(predictions[2:], device="cpu"),
        ]
        cropped_trial_epoch_scoring.y_trues_ = y_true
        cropped_trial_epoch_scoring.window_inds_ = window_inds

        cropped_trial_epoch_scoring.on_epoch_end(
            mock_skorch_net, dataset_train, dataset_valid
        )

        np.testing.assert_almost_equal(mock_skorch_net.history[0]["accuracy"], accuracy)


def test_post_epoch_train_scoring():
    cuda = False
    set_random_seeds(seed=20170629, cuda=cuda)

    n_classes = 2
    classes = list(range(n_classes))

    class EEGDataSet(Dataset):
        def __init__(self, X, y):
            self.X = X
            if self.X.ndim == 3:
                self.X = self.X[:, :, :, None]
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    X, y = sklearn.datasets.make_classification(
        40, (3 * 100), n_informative=3 * 50, n_classes=2
    )
    X = X.reshape(40, 3, 100).astype(np.float32)

    n_chans = X.shape[1]

    train_set = EEGDataSet(X, y)

    class TestCallback(Callback):
        def on_epoch_end(self, net, *args, **kwargs):
            preds = net.predict(train_set.X)
            y_true = train_set.y
            np.testing.assert_allclose(
                clf.history[-1]["train_f1"],
                f1_score(y_true, preds),
                rtol=1e-4,
                atol=1e-4,
            )
            np.testing.assert_allclose(
                clf.history[-1]["train_acc"],
                accuracy_score(y_true, preds),
                rtol=1e-4,
                atol=1e-4,
            )

    set_random_seeds(20200114, cuda)

    # final_conv_length = auto ensures
    # we only get a single output in the time dimension
    model = ShallowFBCSPNet(
        n_chans=n_chans,
        n_outputs=n_classes,
        n_times=train_set.X.shape[2],
        pool_time_stride=1,
        pool_time_length=2,
        final_conv_length="auto",
    )
    if cuda:
        model.cuda()

    clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=optim.AdamW,
        train_split=None,
        optimizer__lr=0.0625 * 0.01,
        optimizer__weight_decay=0,
        batch_size=64,
        callbacks=[
            (
                "train_accuracy",
                PostEpochTrainScoring(
                    "accuracy", lower_is_better=False, name="train_acc"
                ),
            ),
            (
                "train_f1_score",
                PostEpochTrainScoring("f1", lower_is_better=False, name="train_f1"),
            ),
            ("test_callback", TestCallback()),
        ],
        classes=classes,
    )

    clf.fit(train_set, y=None, epochs=4)


def _check_preds_windows_trials(preds, window_inds, expected_trial_preds):
    # transform to 2 lists from tuples
    i_window_in_trials = []
    i_stop_in_trials = []
    for i_window_in_trial, _, i_stop_in_trial in window_inds:
        i_window_in_trials.append(i_window_in_trial)
        i_stop_in_trials.append(i_stop_in_trial)
    trial_preds = trial_preds_from_window_preds(
        preds, i_window_in_trials, i_stop_in_trials
    )
    np.testing.assert_equal(
        len(trial_preds),
        len(expected_trial_preds),
    )
    for expected_pred, actual_pred in zip(expected_trial_preds, trial_preds):
        np.testing.assert_array_equal(
            actual_pred,
            expected_pred,
        )


def test_two_windows_same_trial_with_overlap():
    preds = [
        [[4, 5, 6, 7]],
        [[6, 7, 8, 9]],
    ]
    window_inds = ((0, 0, 8), (1, 2, 10))
    expected_trial_preds = [[[4, 5, 6, 7, 8, 9]]]
    _check_preds_windows_trials(preds, window_inds, expected_trial_preds)


def test_three_windows_two_trials_with_overlap():
    preds = [[[4, 5, 6, 7]], [[6, 7, 8, 9]], [[0, 1, 2, 3]]]
    window_inds = (
        (0, 0, 8),
        (1, 2, 10),
        (
            0,
            0,
            6,
        ),
    )
    expected_trial_preds = [[[4, 5, 6, 7, 8, 9]], [[0, 1, 2, 3]]]
    _check_preds_windows_trials(preds, window_inds, expected_trial_preds)


def test_one_window_one_trial():
    preds = [[[4, 5, 6, 7]]]
    window_inds = ((0, 0, 8),)
    expected_trial_preds = [[[4, 5, 6, 7]]]
    _check_preds_windows_trials(preds, window_inds, expected_trial_preds)


def test_three_windows_two_trials_no_overlap():
    preds = [[[4, 5, 6, 7]], [[6, 7, 8, 9]], [[0, 1, 2, 3]]]
    window_inds = (
        (0, 0, 8),
        (1, 4, 12),
        (
            0,
            0,
            6,
        ),
    )
    expected_trial_preds = [[[4, 5, 6, 7, 6, 7, 8, 9]], [[0, 1, 2, 3]]]
    _check_preds_windows_trials(preds, window_inds, expected_trial_preds)


def test_predict_trials():
    ds = MOABBDataset("FakeDataset", subject_ids=1, dataset_kwargs=bnci_kwargs)
    ds1 = ds.split([0])["0"]

    # determine original trial size
    windows_ds1 = create_windows_from_events(
        ds1,
    )
    trial_size = windows_ds1[0][0].shape[1]

    # create two windows per trial, where windows maximally overlap
    window_size_samples = trial_size - 1
    window_stride_samples = 5
    windows_ds1 = create_windows_from_events(
        ds1,
        window_size_samples=window_size_samples,
        window_stride_samples=window_stride_samples,
        drop_last_window=False,
    )

    n_chans = windows_ds1[0][0].shape[0]
    n_classes = len(windows_ds1.get_metadata()["target"].unique())
    model = ShallowFBCSPNet(
        n_chans=n_chans,
        n_outputs=n_classes,
        n_times=window_size_samples,
    )
    model.to_dense_prediction_model()

    output_shape = model.get_output_shape()
    # the number of samples required to get 1 output
    receptive_field_size = window_size_samples - output_shape[-1] + 1

    preds, targets = predict_trials(model, windows_ds1)

    # some model, cropped data
    assert preds.shape[-1] + receptive_field_size - 1 == trial_size
    assert preds.shape[1] == n_classes
    assert preds.shape[0] == targets.shape[0]
    metadata = windows_ds1.get_metadata()
    expected_targets = metadata[metadata["i_window_in_trial"] == 0]["target"].values
    np.testing.assert_array_equal(expected_targets, targets)

    # some model, trialwise data
    windows_ds2 = create_windows_from_events(ds1)
    with pytest.warns(
        UserWarning,
        match="This function was designed to predict" " trials from cropped datasets.",
    ):
        predict_trials(model, windows_ds2)

    # cropped EEGClassifier, cropped data
    clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=optim.AdamW,
        train_split=None,
        optimizer__lr=0.0625 * 0.01,
        optimizer__weight_decay=0,
        batch_size=64,
    )
    clf.initialize()
    clf.predict_trials(windows_ds1, return_targets=True)

    # cropped EEGClassifier, trialwise data
    with pytest.warns(
        UserWarning,
        match="This method was designed to predict "
        "trials in cropped mode. Calling it "
        "when cropped is False will give the "
        "same result as '.predict'.",
    ):
        clf.predict_trials(windows_ds2)


def test_cropped_train_scoring_module_not_instantiated():
    # cropped train scoring reads the built module, so a class must work here
    set_random_seeds(seed=20200114, cuda=False)
    n_trials, n_chans, n_times = 8, 3, 400
    X = np.random.RandomState(20200114).randn(n_trials, n_chans, n_times)
    X = X.astype(np.float32)
    y = np.array([0, 1] * (n_trials // 2))

    probe = ShallowFBCSPNet(
        n_chans=n_chans, n_outputs=2, n_times=n_times, final_conv_length=2
    )
    probe.to_dense_prediction_model()
    n_preds_per_input = probe.get_output_shape()[2]

    windows_ds = create_from_X_y(
        X,
        y,
        drop_last_window=False,
        sfreq=100.0,
        window_size_samples=n_times,
        window_stride_samples=n_preds_per_input,
    )

    clf = EEGClassifier(
        ShallowFBCSPNet,
        module__n_chans=n_chans,
        module__n_outputs=2,
        module__n_times=n_times,
        module__final_conv_length=2,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.nll_loss,
        optimizer=optim.AdamW,
        train_split=None,
        batch_size=2,
        max_epochs=1,
        classes=[0, 1],
        callbacks=["accuracy"],
    )
    clf.fit(windows_ds, y=None)
    assert "train_accuracy" in clf.history[-1]


def test_post_epoch_train_scoring_uses_batch(monkeypatch):
    import braindecode.training.scoring as scoring_module

    if hasattr(scoring_module, "check_version"):
        monkeypatch.setattr(
            scoring_module, "check_version", lambda *args, **kwargs: False
        )

    class DummyNet:
        def __init__(self):
            self.device = "cpu"
            self.history = History()
            self.history.new_epoch()
            self.evaluation_step_called_with_batch = False

        def fit(self, X, y=None):
            return self

        def get_dataset(self, dataset):
            return dataset

        def get_iterator(self, dataset, training=False):
            return DataLoader(dataset, batch_size=2)

        def evaluation_step(self, batch, training=False):
            assert isinstance(batch, (list, tuple))
            self.evaluation_step_called_with_batch = True
            X, _ = batch
            return X

    X = torch.zeros((4, 1))
    y = torch.randint(0, 2, (4,))
    dataset = TensorDataset(X, y)

    pes = PostEpochTrainScoring(
        lambda net, X, y: 0.0, lower_is_better=False, name="train_score"
    )
    pes.initialize()

    net = DummyNet()
    net.callbacks_ = [("", pes)]

    pes.on_epoch_end(net, dataset, None)

    assert net.history[0]["train_score"] == 0.0
    assert net.evaluation_step_called_with_batch


def test_predict_trials_multi_value_targets():
    # 2 trials x 3 overlapping windows, each window with a 2-value target, as
    # given by target channels with last_target_only=True
    class CroppedWindows(torch.utils.data.Dataset):
        def __init__(self):
            self.items = [
                (
                    np.random.randn(2, 20).astype("float32"),
                    np.array([trial, trial + 10], dtype="float32"),
                    [i_window, 10 * i_window, 10 * i_window + 20],
                )
                for trial in range(2)
                for i_window in range(3)
            ]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

        def get_metadata(self):
            return pd.DataFrame({"i_window_in_trial": [it[2][0] for it in self.items]})

    preds, ys = predict_trials(
        torch.nn.Conv1d(2, 2, 10), CroppedWindows(), batch_size=2
    )
    assert preds.shape[:2] == (2, 2)
    np.testing.assert_array_equal(ys, [[0, 10], [1, 11]])


def test_trial_preds_from_window_preds_delegates_to_streaming_iterator(monkeypatch):
    # The public materialized function wraps the single streaming
    # overlap-removal primitive (no second algorithm) and still returns a list.
    import braindecode.training.scoring as scoring_module

    calls = []
    original = scoring_module.iter_trial_preds

    def spy(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(scoring_module, "iter_trial_preds", spy)
    preds = [[[4, 5, 6, 7]], [[6, 7, 8, 9]], [[0, 1, 2, 3]]]
    trial_preds = trial_preds_from_window_preds(preds, [0, 1, 0], [8, 10, 8])
    assert len(calls) == 1
    assert type(trial_preds) is list
    assert len(trial_preds) == 2
    np.testing.assert_array_equal(trial_preds[0], [[4, 5, 6, 7, 8, 9]])
    np.testing.assert_array_equal(trial_preds[1], [[0, 1, 2, 3]])


class _InferenceWindows(Dataset):
    """Five windows, overlapping in two unequal trials; no stored signals."""

    def __init__(self, target_kind="scalar", repeats=1, fail_at=None, ch_pos=False):
        self.target_kind = target_kind
        self.repeats = repeats
        self.reads = 0
        self.fail_at = fail_at
        self.ch_pos = ch_pos

    def __len__(self):
        return 5 * self.repeats

    def __getitem__(self, index):
        self.reads += 1
        if index == self.fail_at:
            raise RuntimeError("dataset failure")
        local = index % 5
        trial = index // 5 * 2 + (local >= 2)
        window = local if local < 2 else local - 2
        start = window * 2
        signal = np.arange(start, start + 4, dtype=np.float64)[None] + 100 * trial
        if self.target_kind == "scalar":
            target = np.int64(trial)
        elif self.target_kind == "multi":
            target = np.array([trial, trial + 10], dtype=np.int64)
        else:
            target = signal.astype(np.int64)
        row = (signal, target, [window, start, start + 4])
        return (*row, np.zeros((1, 3))) if self.ch_pos else row

    def get_metadata(self):
        return pd.DataFrame({"i_window_in_trial": [0, 1, 0, 1, 2] * self.repeats})


class _InferenceModule(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones((), dtype=torch.float64))
        self.batch_sizes = []
        self.fail = fail

    def forward(self, X):
        assert not self.training
        assert not torch.is_grad_enabled()
        self.batch_sizes.append(len(X))
        if self.fail:
            raise RuntimeError("forward failure")
        return X * self.scale


def test_predict_trials_explicit_loader_kwargs():
    module = _InferenceModule()
    dataset = _InferenceWindows()
    preds, targets = predict_trials(module, dataset, loader_kwargs={"batch_size": 2})
    assert module.batch_sizes == [2, 2, 1]
    assert dataset.reads == 5
    assert len(preds) == len(targets) == 2
    np.testing.assert_array_equal(targets, [0, 1])
    module.batch_sizes.clear()
    predict_trials(module, dataset)
    assert module.batch_sizes == [1] * 5
    # The historical materialized API leaves the module in eval mode.
    assert not module.training


@pytest.mark.parametrize(
    "key",
    [
        "shuffle",
        "sampler",
        "batch_sampler",
        "drop_last",
        "dataset",
        "collate_fn",
        "timeout",
        "unknown",
    ],
)
def test_trial_loader_rejects_ordering_and_unknown_keys(key):
    with pytest.raises(ValueError, match=key):
        predict_trials(
            _InferenceModule(), _InferenceWindows(), loader_kwargs={key: None}
        )


@pytest.mark.parametrize(
    "options",
    [
        {"batch_size": 0},
        {"batch_size": None},
        {"num_workers": -1},
        {"persistent_workers": True},
        {"prefetch_factor": 2},
    ],
)
def test_trial_loader_enforces_dataloader_constraints(options):
    with pytest.raises((ValueError, TypeError)):
        predict_trials(_InferenceModule(), _InferenceWindows(), loader_kwargs=options)


@pytest.mark.parametrize("kind", ["scalar", "multi", "time"])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 16])
@pytest.mark.parametrize("return_targets", [True, False])
def test_iter_trial_predictions_matches_materialized_reference(
    kind, batch_size, return_targets
):
    from braindecode.training.scoring import iter_trial_predictions

    dataset = _InferenceWindows(kind)
    module = _InferenceModule()
    options = {"batch_size": batch_size}
    results = list(
        iter_trial_predictions(module, dataset, return_targets, loader_kwargs=options)
    )
    materialized = predict_trials(
        module, dataset, return_targets, loader_kwargs=options
    )
    for trial, result in enumerate(results):
        pred, target = result if return_targets else (result, None)
        expected = np.arange(6 if trial == 0 else 8)[None] + 100 * trial
        np.testing.assert_allclose(pred, expected, rtol=1e-6, atol=1e-7)
        assert pred.dtype == np.float32
        assert pred.flags.owndata
        np.testing.assert_array_equal(
            pred, (materialized[0] if return_targets else materialized)[trial]
        )
        if return_targets:
            target_expected = (
                trial
                if kind == "scalar"
                else ([trial, trial + 10] if kind == "multi" else expected)
            )
            np.testing.assert_array_equal(target, target_expected)
            np.testing.assert_array_equal(target, materialized[1][trial])
            assert target.dtype == np.float32


def test_trial_iterator_bounded_consumption():
    from braindecode.training.scoring import iter_trial_predictions

    dataset = _InferenceWindows(repeats=100)
    module = _InferenceModule()
    iterator = iter_trial_predictions(module, dataset, loader_kwargs={"batch_size": 2})
    assert dataset.reads == 0
    first = next(iterator)
    assert dataset.reads <= 4  # first trial + boundary, rounded to a loader batch
    assert torch.is_grad_enabled()  # no_grad must not leak across yield
    first_copy = first[0].copy()
    second = next(iterator)
    assert not np.shares_memory(first[0], second[0])
    np.testing.assert_array_equal(first[0], first_copy)
    reads = dataset.reads
    iterator.close()
    assert dataset.reads == reads
    assert module.training


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize(
    "failure", ["dataset", "forward", "channel_positions", "exhaustion"]
)
def test_trial_iterator_restores_mode_on_cleanup(training, failure):
    from braindecode.training.scoring import iter_trial_predictions

    module = _InferenceModule(fail=failure == "forward").train(training)
    dataset = _InferenceWindows(
        fail_at=1 if failure == "dataset" else None,
        ch_pos=failure == "channel_positions",
    )
    iterator = iter_trial_predictions(module, dataset)
    if failure == "exhaustion":
        list(iterator)
    else:
        with pytest.raises((RuntimeError, ValueError)):
            list(iterator)
    assert module.training == training
    assert torch.is_grad_enabled()


def test_trial_iterator_does_not_retain_unused_targets(monkeypatch):
    import weakref

    from braindecode.training import scoring

    refs = []
    original = scoring.DataLoader

    class TrackingLoader(original):
        def __iter__(self):
            for batch in super().__iter__():
                refs.append(weakref.ref(batch[1]))
                yield batch

    monkeypatch.setattr(scoring, "DataLoader", TrackingLoader)
    iterator = scoring.iter_trial_predictions(
        _InferenceModule(),
        _InferenceWindows(repeats=20),
        False,
        loader_kwargs={"batch_size": 1},
    )
    next(iterator)
    # The active DataLoader may hold its current batch; earlier targets must die.
    assert all(ref() is None for ref in refs[:-1])
    iterator.close()
    assert all(ref() is None for ref in refs)


def test_trial_iterator_uses_shared_lazy_overlap_primitive(monkeypatch):
    from braindecode.training import scoring

    original = scoring.iter_trial_preds
    calls = []

    def checked(preds, windows, stops):
        assert iter(preds) is preds
        assert iter(windows) is windows
        assert iter(stops) is stops
        calls.append(True)
        return original(preds, windows, stops)

    monkeypatch.setattr(scoring, "iter_trial_preds", checked)
    list(scoring.iter_trial_predictions(_InferenceModule(), _InferenceWindows("time")))
    assert len(calls) == 2


def test_trial_iterator_loader_whitelist_and_checkpoint():
    from braindecode.training.scoring import iter_trial_predictions

    module = _InferenceModule()
    before = {k: v.clone() for k, v in module.state_dict().items()}
    options = {
        "batch_size": 2,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": None,
        "worker_init_fn": None,
        "generator": torch.Generator().manual_seed(2),
    }
    list(iter_trial_predictions(module, _InferenceWindows(), loader_kwargs=options))
    clone = _InferenceModule()
    clone.load_state_dict(module.state_dict(), strict=True)
    module.load_state_dict(clone.state_dict(), strict=True)
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert module.batch_sizes == [2, 2, 1]
    assert options["batch_size"] == 2


def test_trial_materialization_preserves_container_types():
    # Variable trials must stay lists, never tuples; fixed trials stay arrays.
    variable, time_targets = predict_trials(
        _InferenceModule(), _InferenceWindows("time")
    )
    assert isinstance(variable, list)
    assert isinstance(time_targets, list)

    class FixedWindows(_InferenceWindows):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            # Keep two windows of each trial.
            return super().__getitem__(index)

    fixed, targets = predict_trials(_InferenceModule(), FixedWindows())
    assert isinstance(fixed, np.ndarray)
    assert isinstance(targets, np.ndarray)
    assert fixed.shape == (2, 1, 6)


@pytest.mark.parametrize("failure", [False, True])
def test_trial_iterator_closes_persistent_workers(monkeypatch, failure):
    from braindecode.training import scoring

    workers = []
    original = scoring.DataLoader

    class TrackingLoader(original):
        def __iter__(self):
            iterator = super().__iter__()
            workers.extend(iterator._workers)
            return iterator

    monkeypatch.setattr(scoring, "DataLoader", TrackingLoader)
    module = _InferenceModule(fail=failure)
    iterator = scoring.iter_trial_predictions(
        module,
        _InferenceWindows(repeats=100),
        loader_kwargs={
            "batch_size": 2,
            "num_workers": 1,
            "persistent_workers": True,
            "prefetch_factor": 1,
        },
    )
    if failure:
        with pytest.raises(RuntimeError, match="forward failure"):
            next(iterator)
    else:
        next(iterator)
        iterator.close()
    assert workers and all(not worker.is_alive() for worker in workers)
    assert module.training


def test_trial_iterator_restores_mixed_child_modes():
    from braindecode.training.scoring import iter_trial_predictions

    module = _InferenceModule()
    module.child = torch.nn.Dropout().eval()
    iterator = iter_trial_predictions(module, _InferenceWindows())
    next(iterator)
    iterator.close()
    assert module.training
    assert not module.child.training


# Training-metric accounting characterization: the forward/update counts and
# final-weight semantics that an accelerated training path must preserve.
_ACCOUNTING_METRICS = ("accuracy", "balanced_accuracy")
_ACCOUNTING_SCORERS = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
}


def _accounting_data(n_samples, n_chans=3, n_times=100, seed=20261026):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_chans, n_times).astype(np.float32)
    y = (np.arange(n_samples) % 2).astype(np.int64)
    return X, y


class _ForwardStepCounter(Callback):
    """Count module forwards by mode and optimizer updates via public hooks."""

    def initialize(self):
        for handle in getattr(self, "handles_", []):
            handle.remove()
        self.handles_ = []
        self.hooked_ = None
        self.forwards_ = {"train": 0, "eval": 0}
        self.samples_ = {"train": 0, "eval": 0}
        self.optimizer_steps_ = 0
        return self

    def on_train_begin(self, net, **kwargs):
        if self.hooked_ != (id(net.module_), id(net.optimizer_)):
            for handle in self.handles_:
                handle.remove()
            self.handles_ = [
                net.module_.register_forward_pre_hook(self._on_forward),
                net.optimizer_.register_step_post_hook(self._on_step),
            ]
            self.hooked_ = (id(net.module_), id(net.optimizer_))

    def _on_forward(self, module, args):
        mode = "train" if module.training else "eval"
        self.forwards_[mode] += 1
        self.samples_[mode] += len(args[0])

    def _on_step(self, optimizer, args, kwargs):
        self.optimizer_steps_ += 1


class _ScoringCapture(Callback):
    """After the metrics: keep their reevaluation (skorch clears it at train end)."""

    def on_epoch_end(self, net, **kwargs):
        self.y_preds_ = [
            cb.y_preds_
            for _, cb in net.callbacks_
            if isinstance(cb, PostEpochTrainScoring)
        ]


def _accounting_classifier(
    n_metrics,
    *,
    batch_size=4,
    valid_batch_size=3,
    drop_last=True,
    callbacks=(),
    after=(),
):
    set_random_seeds(seed=20261026, cuda=False)
    model = ShallowFBCSPNet(
        n_chans=3,
        n_outputs=2,
        n_times=100,
        pool_time_stride=1,
        pool_time_length=2,
        final_conv_length="auto",
    )
    return EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=optim.AdamW,
        optimizer__lr=1e-2,
        train_split=None,
        batch_size=batch_size,
        iterator_valid__batch_size=valid_batch_size,
        iterator_train__drop_last=drop_last,
        max_epochs=1,
        classes=[0, 1],
        # Extra callbacks run after the counter and before the training metrics.
        callbacks=[
            ("counter", _ForwardStepCounter()),
            *callbacks,
            *_ACCOUNTING_METRICS[:n_metrics],
            *after,
        ],
        verbose=0,
    )


@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize("n_metrics", [0, 1, 2])
def test_multiple_train_metrics_share_final_weight_predictions(n_metrics, drop_last):
    n_samples, batch_size, valid_batch_size = 10, 4, 3
    X, y = _accounting_data(n_samples)
    clf = _accounting_classifier(
        n_metrics,
        batch_size=batch_size,
        valid_batch_size=valid_batch_size,
        drop_last=drop_last,
        after=[("capture", _ScoringCapture())],
    )
    clf.fit(X, y)
    counter = dict(clf.callbacks_)["counter"]

    # drop_last is explicit: 10 samples, batch 4 -> 2 full batches (8 samples)
    # with the braindecode default, 3 batches (all 10 samples) without it.
    if drop_last:
        expected_train_batches = n_samples // batch_size
        expected_train_samples = expected_train_batches * batch_size
        assert (expected_train_batches, expected_train_samples) == (2, 8)
    else:
        expected_train_batches = math.ceil(n_samples / batch_size)
        expected_train_samples = n_samples
        assert expected_train_batches == 3
    optimizer_steps = counter.optimizer_steps_
    assert optimizer_steps == expected_train_batches
    assert counter.forwards_["train"] == expected_train_batches
    assert counter.samples_["train"] == expected_train_samples
    assert clf.history[-1, "train_batch_count"] == expected_train_batches
    assert sum(clf.history[-1, "batches", :, "train_batch_size"]) == (
        expected_train_samples
    )

    eval_forward_count = counter.forwards_["eval"]
    if n_metrics == 0:
        # No training metric: no added scoring forward at all.
        assert eval_forward_count == 0
        assert counter.samples_["eval"] == 0
        assert "train_accuracy" not in clf.history[-1]
        return

    # One eval-mode reevaluation over every training sample (no drop_last),
    # batched by the validation iterator, shared by all training metrics.
    assert eval_forward_count == math.ceil(n_samples / valid_batch_size)
    assert counter.samples_["eval"] == n_samples
    shared = dict(clf.callbacks_)["capture"].y_preds_
    assert len(shared) == n_metrics
    assert all(y_preds is shared[0] for y_preds in shared)

    # Final post-epoch weights in eval mode, not train-mode minibatch outputs.
    final_outputs = clf.forward(X, training=False)
    torch.testing.assert_close(torch.cat(shared[0]), final_outputs)
    preds = final_outputs.argmax(1).numpy()
    for name in _ACCOUNTING_METRICS[:n_metrics]:
        expected = _ACCOUNTING_SCORERS[name](y, preds)
        assert clf.history[-1, f"train_{name}"] == pytest.approx(expected, abs=1e-12)


def test_train_metric_reevaluation_preserves_training_rng():
    X, y = _accounting_data(10)
    results = {}
    for n_metrics in (0, 1, 2):
        clf = _accounting_classifier(n_metrics)
        clf.fit(X, y, epochs=3)
        counter = dict(clf.callbacks_)["counter"]
        assert counter.optimizer_steps_ == 3 * 2
        assert counter.forwards_["eval"] == (3 * math.ceil(10 / 3) if n_metrics else 0)
        results[n_metrics] = (
            {k: v.clone() for k, v in clf.module_.state_dict().items()},
            clf.history[:, "train_loss"],
            clf.history[:, "train_accuracy"] if n_metrics else None,
        )
    reference_state, reference_loss, _ = results[0]
    for n_metrics in (1, 2):
        state, loss, _ = results[n_metrics]
        assert loss == reference_loss
        assert state.keys() == reference_state.keys()
        assert all(torch.equal(state[k], reference_state[k]) for k in state)
    assert results[1][2] == results[2][2]


class _RngProbe(Callback):
    def on_epoch_end(self, net, **kwargs):
        self.rng_state_ = torch.random.get_rng_state()


def test_train_scoring_error_restores_adapter_and_rng():
    X, y = _accounting_data(10)
    seen = {}

    def failing_metric(net, X, y):
        counter = dict(net.callbacks_)["counter"]
        seen["adapter_active"] = "forward_iter" in vars(net)
        seen["eval_forwards"] = counter.forwards_["eval"]
        net.predict(X)  # served by the temporary cached-prediction adapter
        seen["eval_forwards_after_predict"] = counter.forwards_["eval"]
        raise RuntimeError("metric failed")

    clf = _accounting_classifier(
        0,
        callbacks=[
            ("rng_probe", _RngProbe()),
            ("train_failing", PostEpochTrainScoring(failing_metric, name="train_f")),
        ],
    )
    with pytest.raises(RuntimeError, match="metric failed"):
        clf.fit(X, y)
    assert seen["adapter_active"]
    assert seen["eval_forwards"] == seen["eval_forwards_after_predict"] == 4
    assert "forward_iter" not in vars(clf)
    probe = dict(clf.callbacks_)["rng_probe"]
    assert torch.equal(torch.random.get_rng_state(), probe.rng_state_)


class _FailableTrainSet(Dataset):
    def __init__(self, X, y):
        self.X, self.y, self.fail = X, y, False

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.fail:
            raise RuntimeError("load failed")
        return self.X[idx], self.y[idx]


class _FailNextLoad(Callback):
    """Placed before the scorer: arm a loading failure for the reevaluation."""

    def __init__(self, dataset):
        self.dataset = dataset

    def on_epoch_end(self, net, **kwargs):
        self.rng_state_ = torch.random.get_rng_state()
        self.dataset.fail = True


def test_train_scoring_reevaluation_error_restores_rng():
    X, y = _accounting_data(10)
    dataset = _FailableTrainSet(X, y)
    clf = _accounting_classifier(
        1, callbacks=[("fail_next_load", _FailNextLoad(dataset))]
    )
    with pytest.raises(RuntimeError, match="load failed"):
        clf.fit(dataset, y=None)
    assert "forward_iter" not in vars(clf)
    probe = dict(clf.callbacks_)["fail_next_load"]
    assert torch.equal(torch.random.get_rng_state(), probe.rng_state_)


class _ModeProbe(Callback):
    """Before the scorers: record the module mode training left behind."""

    def on_epoch_end(self, net, **kwargs):
        self.training_ = net.module_.training


def _failing_metric(net, X, y):
    raise RuntimeError("metric failed")


@pytest.mark.parametrize(
    "failure, expected_training",
    [
        # The reevaluation ran evaluation_step (eval mode); nothing switches back.
        ("scorer", False),
        # Loading failed before any evaluation_step: the training mode remains.
        ("reevaluation_loading", True),
    ],
)
def test_train_scoring_error_module_mode_is_characterized(failure, expected_training):
    # BASE does not restore train/eval mode after a scoring error (skorch's own
    # validation also leaves eval mode); the next train step sets train mode.
    X, y = _accounting_data(10)
    dataset = _FailableTrainSet(X, y)
    if failure == "scorer":
        extra = [
            ("mode_probe", _ModeProbe()),
            ("train_failing", PostEpochTrainScoring(_failing_metric, name="train_f")),
        ]
        n_metrics, match = 0, "metric failed"
    else:
        extra = [("mode_probe", _ModeProbe()), ("fail", _FailNextLoad(dataset))]
        n_metrics, match = 1, "load failed"
    clf = _accounting_classifier(n_metrics, callbacks=extra)
    with pytest.raises(RuntimeError, match=match):
        clf.fit(dataset, y=None)
    assert dict(clf.callbacks_)["mode_probe"].training_ is True
    assert clf.module_.training is expected_training
    assert "forward_iter" not in vars(clf)
