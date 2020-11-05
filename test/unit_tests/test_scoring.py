# Authors: Maciej Sliwowski
#          Lukas Gemein
#          Robin Tibor Schirrmeister
#
# License: BSD-3


import numpy as np
import sklearn.datasets
import torch
from sklearn.metrics import f1_score, accuracy_score
from skorch import History
from skorch.callbacks import Callback
from skorch.utils import to_numpy, to_tensor
from torch import optim
from torch.utils.data import Dataset, DataLoader
from braindecode.classifier import EEGClassifier
from braindecode.datautil.xy import create_from_X_y
from braindecode.training.scoring import CroppedTrialEpochScoring
from braindecode.training.scoring import PostEpochTrainScoring
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds
from braindecode.training.scoring import trial_preds_from_window_preds


class MockSkorchNet:
    def __init__(self):
        """
        Initialize the forward history.

        Args:
            self: (todo): write your description
        """
        self.device = "cpu"
        self.forward_iter = None
        self.history = History()
        self.history.new_epoch()
        self._default_callbacks = []

    def fit(self, X, y=None):
        """
        Fit the model.

        Args:
            self: (todo): write your description
            X: (array): write your description
            y: (array): write your description
        """
        return self

    def predict(self, X):
        """
        Predict the model.

        Args:
            self: (todo): write your description
            X: (array): write your description
        """
        return np.concatenate(
            [to_numpy(x.argmax(dim=1)) for x in self.forward_iter(X)], 0
        )

    def get_iterator(self, X_test, training):
        """
        Returns an iterator of a test.

        Args:
            self: (todo): write your description
            X_test: (todo): write your description
            training: (bool): write your description
        """
        return DataLoader(X_test, batch_size=2)


def test_cropped_trial_epoch_scoring():
    """
    Test for the trial.

    Args:
    """

    dataset_train = None
    # Definition of test cases
    predictions_cases = [
        # Exepected predictions classification results: [1, 0, 0, 0]
        np.array(
            [
                [[0.2, 0.1, 0.1, 0.1], [0.8, 0.9, 0.9, 0.9]], # trial 0 preds
                [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]], # trial 1 preds
                [[1.0, 1.0, 1.0, 0.2], [0.0, 0.0, 0.0, 0.8]], # trial 2 preds
                [[0.9, 0.8, 0.9, 1.0], [0.1, 0.2, 0.1, 0.0]], # trial 3 preds
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
            torch.tensor([0,0]), # i_window_in_trials
            [None],# won't be used
            torch.tensor([4,4]), # i_window_stops
    ),(
            torch.tensor([0,0]), # i_window_in_trials
            [None],# won't be used
            torch.tensor([4,4]), # i_window_stops
    ),]

    for predictions, y_true, accuracy in zip(
        predictions_cases, y_true_cases, expected_accuracies_cases
    ):
        dataset_valid = create_from_X_y(
            np.zeros((4, 1, 10)), np.concatenate(y_true),
            window_size_samples=10, window_stride_samples=4, drop_last_window=False)

        mock_skorch_net = MockSkorchNet()
        cropped_trial_epoch_scoring = CroppedTrialEpochScoring(
            "accuracy", on_train=False)
        mock_skorch_net.callbacks = [(
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
    """
    Test for the trial.

    Args:
    """
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
    ), (
            torch.tensor([0,0]), # i_window_in_trials
            [None],# won't be used
            torch.tensor([4,4]), # i_window_stops
    ),]
    cropped_trial_epoch_scoring = CroppedTrialEpochScoring("accuracy")
    cropped_trial_epoch_scoring.initialize()
    cropped_trial_epoch_scoring.y_preds_ = [
        to_tensor(predictions[:2], device="cpu"),
        to_tensor(predictions[2:], device="cpu"),
    ]
    cropped_trial_epoch_scoring.y_trues_ = y_true
    cropped_trial_epoch_scoring.window_inds_ = window_inds

    mock_skorch_net = MockSkorchNet()
    mock_skorch_net.callbacks = [(
        "", cropped_trial_epoch_scoring)]
    output = cropped_trial_epoch_scoring.on_epoch_end(
        mock_skorch_net, dataset_train, dataset_valid
    )
    assert output is None


def test_post_epoch_train_scoring():
    """
    Generate - test test test test test.

    Args:
    """
    cuda = False
    set_random_seeds(seed=20170629, cuda=cuda)

    n_classes = 2

    class EEGDataSet(Dataset):
        def __init__(self, X, y):
            """
            Initialize inputs.

            Args:
                self: (todo): write your description
                X: (int): write your description
                y: (int): write your description
            """
            self.X = X
            if self.X.ndim == 3:
                self.X = self.X[:, :, :, None]
            self.y = y

        def __len__(self):
            """
            The length of the length of the data.

            Args:
                self: (todo): write your description
            """
            return len(self.X)

        def __getitem__(self, idx):
            """
            Return the item at indexx.

            Args:
                self: (todo): write your description
                idx: (list): write your description
            """
            return self.X[idx], self.y[idx]

    X, y = sklearn.datasets.make_classification(
        40, (3 * 100), n_informative=3 * 50, n_classes=2
    )
    X = X.reshape(40, 3, 100).astype(np.float32)

    in_chans = X.shape[1]

    train_set = EEGDataSet(X, y)

    class TestCallback(Callback):
        def on_epoch_end(self, net, *args, **kwargs):
            """
            Perform a model.

            Args:
                self: (todo): write your description
                net: (todo): write your description
            """
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
    """
    Check that all trial_predals.

    Args:
        preds: (todo): write your description
        window_inds: (int): write your description
        expected_trial_preds: (todo): write your description
    """
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
    """
    Test if two iterators are identical.

    Args:
    """
    preds = [[[4,5,6,7]], [[6,7,8,9]],]
    window_inds = ((0,0,8),(1,2,10))
    expected_trial_preds = [[[4,5,6,7,8,9]]]
    _check_preds_windows_trials(preds, window_inds, expected_trial_preds)


def test_three_windows_two_trials_with_overlap():
    """
    Test if all intersections between two windows are identical.

    Args:
    """
    preds = [[[4, 5, 6, 7]], [[6, 7, 8, 9]], [[0, 1, 2, 3]]]
    window_inds = ((0, 0, 8), (1, 2, 10), (0, 0, 6,))
    expected_trial_preds = [[[4, 5, 6, 7, 8, 9]], [[0, 1, 2, 3]]]
    _check_preds_windows_trials(preds, window_inds, expected_trial_preds)


def test_one_window_one_trial():
    """
    Test if one single trial.

    Args:
    """
    preds = [[[4,5,6,7]]]
    window_inds = ((0,0,8),)
    expected_trial_preds = [[[4,5,6,7]]]
    _check_preds_windows_trials(preds, window_inds, expected_trial_preds)


def test_three_windows_two_trials_no_overlap():
    """
    Test if all intersections between two populations.

    Args:
    """
    preds = [[[4, 5, 6, 7]], [[6, 7, 8, 9]], [[0, 1, 2, 3]]]
    window_inds = ((0, 0, 8), (1, 4, 12), (0, 0, 6,))
    expected_trial_preds = [[[4, 5, 6, 7, 6, 7, 8, 9]], [[0, 1, 2, 3]]]
    _check_preds_windows_trials(preds, window_inds, expected_trial_preds)
