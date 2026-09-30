# Authors: Maciej Sliwowski
#          Robin Tibor Schirrmeister
#
# License: BSD-3
import numpy as np
import pytest
import torch
from scipy.stats import binomtest
from sklearn.model_selection import StratifiedKFold
from skorch.dataset import Dataset
from skorch.helper import predefined_split

from braindecode.classifier import EEGClassifier
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds
from test.acceptance_tests._bnci import load_left_right_trials, shuffled_labels

SEED = 20170629
N_FOLDS = 4
N_EPOCHS = 10

# Pooled held-out accuracy (288 trials) over seed sweeps that varied the
# weight-init/dropout, batch-order and fold seeds separately and together:
# real labels 0.708-0.812 (231 runs), shuffled-label controls 0.413-0.604
# (275 runs). Both thresholds keep about 10 trials of margin.
MIN_ACCURACY = 0.67
MAX_CONTROL_ACCURACY = 0.64
MAX_P_VALUE = 1e-3


@pytest.fixture(scope="module")
def trials():
    # 288 trials (144 per class, both sessions), 22 channels, 0.5-3.5 s at 100 Hz.
    X, y, _ = load_left_right_trials(tmin=0.5, tmax=3.5)
    return X, y


def fit_fold(X, y, train_idx, valid_idx, init_seed, shuffle_seed, lr=1e-3):
    # init_seed drives weight init and dropout, shuffle_seed the batch order.
    set_random_seeds(init_seed, cuda=False)
    model = ShallowFBCSPNet(
        n_chans=X.shape[1],
        n_outputs=2,
        n_times=X.shape[2],
        final_conv_length="auto",
    )
    clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        train_split=predefined_split(Dataset(X[valid_idx], y[valid_idx])),
        batch_size=32,
        iterator_train__shuffle=True,
        iterator_train__generator=torch.Generator().manual_seed(shuffle_seed),
        callbacks=["accuracy"],
        device="cpu",
        classes=[0, 1],
    )
    clf.fit(X[train_idx], y=y[train_idx], epochs=N_EPOCHS)
    return clf


def cross_validate(X, y, init_seed=SEED, shuffle_seed=SEED, fold_seed=SEED, lr=1e-3):
    """Predict every trial once, with a model that did not train on it."""
    folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=fold_seed)
    y_pred = np.full_like(y, -1)
    for i_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        clf = fit_fold(
            X, y, train_idx, valid_idx, init_seed + i_fold, shuffle_seed + i_fold, lr
        )
        y_pred[valid_idx] = clf.predict(X[valid_idx])
        # The last history row scores the final weights on this held-out fold.
        np.testing.assert_allclose(
            np.mean(y_pred[valid_idx] == y[valid_idx]),
            clf.history[-1, "valid_accuracy"],
        )
        assert np.all(np.isfinite(clf.history[:, "train_loss"]))
        assert np.all(np.isfinite(clf.history[:, "valid_loss"]))
    assert np.all(y_pred >= 0)
    return y_pred


def _history_rows(history):
    # Everything but the wall-clock durations must be identical.
    return [{key: val for key, val in row.items() if key != "dur"} for row in history]


@pytest.mark.network
def test_trialwise_decoding_is_replicable(trials, deterministic_algorithms):
    X, y = trials
    train_idx, valid_idx = next(
        StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y)
    )
    first = fit_fold(X, y, train_idx, valid_idx, SEED, SEED)
    second = fit_fold(X, y, train_idx, valid_idx, SEED, SEED)

    first_rows = _history_rows(first.history)
    second_rows = _history_rows(second.history)
    assert len(first_rows) == len(second_rows) == N_EPOCHS
    for epoch, (row, other) in enumerate(zip(first_rows, second_rows), start=1):
        assert row == other, f"histories differ at epoch {epoch}"
    np.testing.assert_array_equal(
        first.predict_proba(X[valid_idx]), second.predict_proba(X[valid_idx])
    )


@pytest.mark.network
def test_trialwise_decoding(trials, deterministic_algorithms):
    X, y = trials
    n_correct = int(np.sum(cross_validate(X, y) == y))
    n_trials = len(y)
    accuracy = n_correct / n_trials

    y_control = shuffled_labels(y, SEED)
    control_accuracy = np.mean(cross_validate(X, y_control) == y_control)

    p_value = binomtest(n_correct, n_trials, 0.5, alternative="greater").pvalue
    print(
        f"held-out accuracy {accuracy:.4f} ({n_correct}/{n_trials}), p={p_value:.1e}; "
        f"shuffled-label control {control_accuracy:.4f}"
    )
    assert accuracy >= MIN_ACCURACY, (accuracy, control_accuracy)
    assert p_value < MAX_P_VALUE, p_value
    # A label leak between train and held-out trials would let the control
    # score far above chance.
    assert control_accuracy <= MAX_CONTROL_ACCURACY, (accuracy, control_accuracy)
