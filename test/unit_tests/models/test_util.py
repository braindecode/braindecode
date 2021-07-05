# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import pytest
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from braindecode.models.modules import Expression
from braindecode.models.util import (
    get_output_shape, aggregate_probas, _pad_shift_array)
from torch import nn


def test_get_output_shape_1d_model():
    model = nn.Conv1d(1, 1, 3)
    out_shape = get_output_shape(model, in_chans=1, input_window_samples=5)
    assert out_shape == (1, 1, 3,)


def test_get_output_shape_2d_model():
    model = nn.Sequential(
        Expression(lambda x: x.unsqueeze(-1)),
        nn.Conv2d(1, 1, (3, 1)))
    out_shape = get_output_shape(model, in_chans=1, input_window_samples=5)
    assert out_shape == (1, 1, 3, 1)


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
@pytest.mark.parametrize('n_sequences,n_classes,n_windows,stride',
                         [[10, 3, 2, 1], [3, 3, 2, 5], [3, 3, 1, 2]])
def test_pad_shift_array(n_sequences, n_classes, n_windows, stride, dtype):
    dense_y = np.random.RandomState(33).rand(n_sequences, n_classes, n_windows)

    n_outputs = (n_sequences - 1) * stride + n_windows
    shifted_y = np.concatenate([
        np.concatenate((
            np.zeros((1, n_classes, i * stride)), dense_y[[i]],
            np.zeros((1, n_classes, n_outputs - n_windows - i * stride))),
            axis=2)
        for i in range(n_sequences)], axis=0)
    shifted_y2 = _pad_shift_array(dense_y, stride=stride)

    assert (shifted_y == shifted_y2).all()


def test_pad_shift_array_not_3d():
    with pytest.raises(NotImplementedError):
        _pad_shift_array(np.zeros((2, 2)))


@pytest.mark.parametrize('n_sequences,n_classes,n_windows,stride',
                         [[3, 3, 2, 2], [3, 3, 1, 1], [10, 3, 2, 1]])
def test_aggregate_probas(n_sequences, n_classes, n_windows, stride):
    # Aggregation should be done recording-wise, we don't want overlapping
    # sequences from different recordings...

    n_outputs = (n_sequences - 1) * stride + n_windows
    y_true = np.arange(n_outputs) % n_classes
    logits = OneHotEncoder(sparse=False).fit_transform(y_true.reshape(-1, 1))
    logits = np.lib.stride_tricks.sliding_window_view(
        logits, n_windows, axis=0)[::stride]

    y_pred_probas = aggregate_probas(logits, n_windows_stride=stride)

    assert y_pred_probas.ndim == 2
    assert y_pred_probas.shape[0] == n_outputs
    assert y_pred_probas.shape[1] == n_classes
    assert (y_true == y_pred_probas.argmax(axis=1)).all()
