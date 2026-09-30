# Authors: Maciej Sliwowski
#          Robin Tibor Schirrmeister
#
# License: BSD-3
import numpy as np
import pytest
import torch
from scipy.stats import binomtest
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from braindecode import EEGClassifier
from braindecode.datasets.xy import create_from_X_y
from braindecode.models import ShallowFBCSPNet
from braindecode.training.losses import CroppedLoss
from braindecode.util import set_random_seeds
from test.acceptance_tests._bnci import SFREQ, load_left_right_trials, shuffled_labels

SEED = 20170629
N_EPOCHS = 10
INPUT_WINDOW_SAMPLES = 300

# Accuracy on the 144 trials of the second session over seed sweeps that varied
# the weight-init/dropout and batch-order seeds separately and together: real
# labels 0.826-0.882 (151 runs), shuffled-label control 0.382-0.590 (101 runs).
# Both thresholds keep about 10 trials of margin.
MIN_ACCURACY = 0.75
MAX_CONTROL_ACCURACY = 0.66
MAX_P_VALUE = 1e-3


@pytest.fixture(scope="module")
def sessions():
    # 144 trials (72 per class) per session, 22 channels, 0-4 s at 100 Hz.
    X, y, session = load_left_right_trials(tmin=0.0, tmax=4.0)
    assert len(np.unique(session)) == 2
    train = session == np.unique(session)[0]
    return X[train], y[train], X[~train], y[~train]


def fit_and_predict(
    X_train, y_train, X_valid, y_valid, init_seed=SEED, shuffle_seed=SEED, lr=1e-3
):
    """Train on one session and predict the trials of the other one."""
    set_random_seeds(init_seed, cuda=False)
    model = ShallowFBCSPNet(
        n_chans=X_train.shape[1],
        n_outputs=2,
        n_times=INPUT_WINDOW_SAMPLES,
        final_conv_length=12,
    )
    model.to_dense_prediction_model()
    n_preds_per_input = model.get_output_shape()[2]

    def windows(X, y):
        return create_from_X_y(
            X,
            y,
            drop_last_window=False,
            sfreq=SFREQ,
            window_size_samples=INPUT_WINDOW_SAMPLES,
            window_stride_samples=n_preds_per_input,
        )

    train_set, valid_set = windows(X_train, y_train), windows(X_valid, y_valid)
    clf = EEGClassifier(
        model,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.cross_entropy,
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        train_split=predefined_split(valid_set),
        batch_size=32,
        iterator_train__shuffle=True,
        iterator_train__generator=torch.Generator().manual_seed(shuffle_seed),
        callbacks=[
            "accuracy",
            # Annealing the lr keeps the final weights, which are the ones
            # evaluated, from depending much on the last few batches.
            ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=N_EPOCHS - 1)),
        ],
        classes=[0, 1],
    )
    clf.fit(train_set, y=None, epochs=N_EPOCHS)

    # Average the crop predictions of each trial, as the scoring callback does.
    trial_preds, trial_targets = clf.predict_trials(valid_set)
    y_pred = trial_preds.mean(axis=2).argmax(axis=1)
    np.testing.assert_array_equal(trial_targets, y_valid)
    np.testing.assert_allclose(
        np.mean(y_pred == y_valid), clf.history[-1, "valid_accuracy"]
    )
    assert len(clf.history) == N_EPOCHS
    for key in ("train_loss", "valid_loss"):
        # Cross-entropy is non-negative; a negative loss means log-probabilities
        # and logits got mixed up.
        assert np.all(np.asarray(clf.history[:, key]) >= 0)
    return y_pred


@pytest.mark.network
def test_cropped_decoding(sessions, deterministic_algorithms):
    X_train, y_train, X_valid, y_valid = sessions
    y_pred = fit_and_predict(X_train, y_train, X_valid, y_valid)
    n_correct = int(np.sum(y_pred == y_valid))
    n_trials = len(y_valid)
    accuracy = n_correct / n_trials

    control_train = shuffled_labels(y_train, SEED)
    control_valid = shuffled_labels(y_valid, SEED + 1)
    control_accuracy = np.mean(
        fit_and_predict(X_train, control_train, X_valid, control_valid) == control_valid
    )

    p_value = binomtest(n_correct, n_trials, 0.5, alternative="greater").pvalue
    print(
        f"held-out accuracy {accuracy:.4f} ({n_correct}/{n_trials}), p={p_value:.1e}; "
        f"shuffled-label control {control_accuracy:.4f}"
    )
    assert accuracy >= MIN_ACCURACY, (accuracy, control_accuracy)
    assert p_value < MAX_P_VALUE, p_value
    # A label leak between the sessions would let the control score far above
    # chance.
    assert control_accuracy <= MAX_CONTROL_ACCURACY, (accuracy, control_accuracy)
