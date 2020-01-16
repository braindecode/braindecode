import numpy as np
from skorch import History
from skorch.utils import to_numpy, to_tensor
from torch.utils.data.dataset import Dataset

from braindecode.datautil import CropsDataLoader
from braindecode.experiments.scoring import CroppedTrialEpochScoring


class EEGDataSet(Dataset):
    def __init__(self, X, y):
        self.X = X
        if self.X.ndim == 3:
            self.X = self.X[:, :, :, None]
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        i_trial, start, stop = idx
        return self.X[i_trial, :, start:stop], self.y[i_trial]


class MockSkorchNet:
    def __init__(self):
        self.device = "cpu"
        self.forward_iter = None
        self.history = History()
        self.history.new_epoch()

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.concatenate([to_numpy(x) for x in self.forward_iter(X)], 0)

    def get_iterator(self, X_test, training):
        return CropsDataLoader(
            X_test, input_time_length=10, n_preds_per_input=4, batch_size=2
        )


def test_cropped_trial_epoch_scoring():
    dataset_train = None
    # Definition of test cases
    predictions_cases = [
        # Exepected redictions classification results: [1, 0, 0, 0]
        np.array(
            [
                [[0.2, 0.1, 0.1, 0.1], [0.8, 0.9, 0.9, 0.9]],
                [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
                [[1.0, 1.0, 1.0, 0.2], [0.0, 0.0, 0.0, 0.8]],
                [[0.9, 0.8, 0.9, 1.0], [0.1, 0.2, 0.1, 0.0]],
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
        [np.array([0, 0]), np.array([1, 1])],
        [np.array([1, 1]), np.array([1, 1])],
    ]
    expected_accuracies_cases = [0.25, 0.75]

    for predictions, y_true, accuracy in zip(
        predictions_cases, y_true_cases, expected_accuracies_cases
    ):
        dataset_valid = EEGDataSet(np.zeros((4, 1, 10)), np.concatenate(y_true))
        mock_skorch_net = MockSkorchNet()
        cropped_trial_epoch_scoring = CroppedTrialEpochScoring("accuracy")
        cropped_trial_epoch_scoring.initialize()
        cropped_trial_epoch_scoring.y_preds_ = [
            to_tensor(predictions[:2], device="cpu"),
            to_tensor(predictions[2:], device="cpu"),
        ]
        cropped_trial_epoch_scoring.y_trues_ = y_true

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
    y_true = [np.array([0, 0]), np.array([1, 1])]
    cropped_trial_epoch_scoring = CroppedTrialEpochScoring("accuracy")
    cropped_trial_epoch_scoring.initialize()
    cropped_trial_epoch_scoring.y_preds_ = [
        to_tensor(predictions[:2], device="cpu"),
        to_tensor(predictions[2:], device="cpu"),
    ]
    cropped_trial_epoch_scoring.y_trues_ = y_true

    mock_skorch_net = MockSkorchNet()

    output = cropped_trial_epoch_scoring.on_epoch_end(
        mock_skorch_net, dataset_train, dataset_valid
    )
    assert output is None
