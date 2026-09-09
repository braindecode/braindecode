import numpy as np


def assert_learning_history(
    history, n_epochs, loss_keys, accuracy_keys=(), improving_accuracy_keys=()
):
    assert len(history) == n_epochs

    for key in loss_keys:
        values = np.asarray(history[:, key], dtype=float)
        assert len(values) == n_epochs
        assert np.all(np.isfinite(values))
        assert values[0] > values[-1]

    for key in accuracy_keys:
        values = np.asarray(history[:, key], dtype=float)
        assert len(values) == n_epochs
        assert np.all(np.isfinite(values))
        assert np.all((0.0 <= values) & (values <= 1.0))

    for key in improving_accuracy_keys:
        values = np.asarray(history[:, key], dtype=float)
        assert values[0] < values[-1]
