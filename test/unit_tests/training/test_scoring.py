# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Robin Tibor Schirrmeister <robintibor@gmail.com>
#
# License: BSD-3


import numpy as np
import pytest
import sklearn.datasets
import torch
from sklearn.metrics import f1_score, accuracy_score
from skorch import History
from skorch.callbacks import Callback
from skorch.utils import to_numpy, to_tensor
from torch import optim
from torch.utils.data import Dataset, DataLoader
from braindecode.classifier import EEGClassifier
from braindecode.datasets.xy import create_from_X_y
from braindecode.models import ShallowFBCSPNet, get_output_shape
from braindecode.util import set_random_seeds
from braindecode.training.scoring import (
    CroppedTrialEpochScoring, PostEpochTrainScoring, trial_preds_from_window_preds,
    predict_trials, CroppedTimeSeriesEpochScoring)
from braindecode.datasets.moabb import MOABBDataset
from braindecode.models.util import to_dense_prediction_model
from braindecode.preprocessing import create_windows_from_events


class MockSkorchNet:
    def __init__(self):
        self.device = "cpu"
        self.forward_iter = None
        self.history = History()
        self.history.new_epoch()
        self._default_callbacks = []

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.concatenate(
            [to_numpy(x.argmax(1)) for x in self.forward_iter(X)], 0
        )

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

    window_inds = [(
        torch.tensor([0, 0]),  # i_window_in_trials
        [None],  # won't be used
        torch.tensor([4, 4]),  # i_window_stops
    ), (
        torch.tensor([0, 0]),  # i_window_in_trials
        [None],  # won't be used
        torch.tensor([4, 4]),  # i_window_stops
    )]

    for predictions, y_true, accuracy in zip(
        predictions_cases, y_true_cases, expected_accuracies_cases
    ):
        dataset_valid = create_from_X_y(
            np.zeros((4, 1, 10)), np.concatenate(y_true),
            sfreq=100, window_size_samples=10, window_stride_samples=4,
            drop_last_window=False)

        mock_skorch_net = MockSkorchNet()
        cropped_trial_epoch_scoring = CroppedTrialEpochScoring(
            "accuracy", on_train=False)
        mock_skorch_net.callbacks_ = [(
            "", cropped_trial_epoch_scoring)]
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

        np.testing.assert_almost_equal(
            mock_skorch_net.history[0]["accuracy"], accuracy
        )


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
    window_inds = [(
        torch.tensor([0, 0]),  # i_window_in_trials
        [None],  # won't be used
        torch.tensor([4, 4]),  # i_window_stops
    ),
        (
        torch.tensor([0, 0]),  # i_window_in_trials
        [None],  # won't be used
        torch.tensor([4, 4]),  # i_window_stops
    )]
    cropped_trial_epoch_scoring = CroppedTrialEpochScoring("accuracy")
    cropped_trial_epoch_scoring.initialize()
    cropped_trial_epoch_scoring.y_preds_ = [
        to_tensor(predictions[:2], device="cpu"),
        to_tensor(predictions[2:], device="cpu"),
    ]
    cropped_trial_epoch_scoring.y_trues_ = y_true
    cropped_trial_epoch_scoring.window_inds_ = window_inds

    mock_skorch_net = MockSkorchNet()
    mock_skorch_net.callbacks_ = [(
        "", cropped_trial_epoch_scoring)]
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

    window_inds = [(
        torch.tensor([0, 0]),  # i_window_in_trials
        [None],  # won't be used
        torch.tensor([4, 4]),  # i_window_stops
    ), (
        torch.tensor([0, 0]),  # i_window_in_trials
        [None],  # won't be used
        torch.tensor([4, 4]),  # i_window_stops
    )]

    for predictions, y_true, accuracy in zip(
        predictions_cases, y_true_cases, expected_accuracies_cases
    ):
        dataset_valid = create_from_X_y(
            np.zeros((4, 1, 10)), np.concatenate(y_true), sfreq=100,
            window_size_samples=10, window_stride_samples=4, drop_last_window=False)

        mock_skorch_net = MockSkorchNet()
        cropped_trial_epoch_scoring = CroppedTimeSeriesEpochScoring(
            "accuracy", on_train=False)
        mock_skorch_net.callbacks_ = [(
            "", cropped_trial_epoch_scoring)]
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

        np.testing.assert_almost_equal(
            mock_skorch_net.history[0]["accuracy"], accuracy
        )


def test_post_epoch_train_scoring():
    cuda = False
    set_random_seeds(seed=20170629, cuda=cuda)

    n_classes = 2

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

    in_chans = X.shape[1]

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
        in_chans=in_chans,
        n_classes=n_classes,
        input_window_samples=train_set.X.shape[2],
        pool_time_stride=1,
        pool_time_length=2,
        final_conv_length="auto",
    )
    if cuda:
        model.cuda()

    clf = EEGClassifier(
        model,
        criterion=torch.nn.NLLLoss,
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
                PostEpochTrainScoring(
                    "f1", lower_is_better=False, name="train_f1"
                ),
            ),
            ("test_callback", TestCallback()),
        ],
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
        preds, i_window_in_trials, i_stop_in_trials)
    np.testing.assert_equal(len(trial_preds), len(expected_trial_preds),)
    for expected_pred, actual_pred in zip(expected_trial_preds, trial_preds):
        np.testing.assert_array_equal(actual_pred, expected_pred, )


def test_two_windows_same_trial_with_overlap():
    preds = [[[4, 5, 6, 7]], [[6, 7, 8, 9]], ]
    window_inds = ((0, 0, 8), (1, 2, 10))
    expected_trial_preds = [[[4, 5, 6, 7, 8, 9]]]
    _check_preds_windows_trials(preds, window_inds, expected_trial_preds)


def test_three_windows_two_trials_with_overlap():
    preds = [[[4, 5, 6, 7]], [[6, 7, 8, 9]], [[0, 1, 2, 3]]]
    window_inds = ((0, 0, 8), (1, 2, 10), (0, 0, 6,))
    expected_trial_preds = [[[4, 5, 6, 7, 8, 9]], [[0, 1, 2, 3]]]
    _check_preds_windows_trials(preds, window_inds, expected_trial_preds)


def test_one_window_one_trial():
    preds = [[[4, 5, 6, 7]]]
    window_inds = ((0, 0, 8),)
    expected_trial_preds = [[[4, 5, 6, 7]]]
    _check_preds_windows_trials(preds, window_inds, expected_trial_preds)


def test_three_windows_two_trials_no_overlap():
    preds = [[[4, 5, 6, 7]], [[6, 7, 8, 9]], [[0, 1, 2, 3]]]
    window_inds = ((0, 0, 8), (1, 4, 12), (0, 0, 6,))
    expected_trial_preds = [[[4, 5, 6, 7, 6, 7, 8, 9]], [[0, 1, 2, 3]]]
    _check_preds_windows_trials(preds, window_inds, expected_trial_preds)


def test_predict_trials():
    ds = MOABBDataset('BNCI2014001', subject_ids=1)
    ds1 = ds.split([0])['0']

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

    in_chans = windows_ds1[0][0].shape[0]
    n_classes = len(windows_ds1.get_metadata()['target'].unique())
    model = ShallowFBCSPNet(
        in_chans=in_chans,
        n_classes=n_classes,
    )
    to_dense_prediction_model(model)

    output_shape = get_output_shape(model, in_chans, window_size_samples)
    # the number of samples required to get 1 output
    receptive_field_size = window_size_samples - output_shape[-1] + 1

    preds, targets = predict_trials(model, windows_ds1)

    # some model, cropped data
    assert preds.shape[-1] + receptive_field_size - 1 == trial_size
    assert preds.shape[1] == n_classes
    assert preds.shape[0] == targets.shape[0]
    metadata = windows_ds1.get_metadata()
    expected_targets = metadata[metadata['i_window_in_trial'] == 0][
        'target'].values
    np.testing.assert_array_equal(expected_targets, targets)

    # some model, trialwise data
    windows_ds2 = create_windows_from_events(ds1)
    with pytest.warns(UserWarning, match='This function was designed to predict'
                                         ' trials from cropped datasets.'):
        predict_trials(model, windows_ds2)

    # cropped EEGClassifier, cropped data
    clf = EEGClassifier(
        model,
        criterion=torch.nn.NLLLoss,
        optimizer=optim.AdamW,
        train_split=None,
        optimizer__lr=0.0625 * 0.01,
        optimizer__weight_decay=0,
        batch_size=64,
    )
    clf.initialize()
    clf.predict_trials(windows_ds1, return_targets=True)

    # cropped EEGClassifier, trialwise data
    with pytest.warns(UserWarning, match="This method was designed to predict "
                                         "trials in cropped mode. Calling it "
                                         "when cropped is False will give the "
                                         "same result as '.predict'."):
        clf.predict_trials(windows_ds2)
