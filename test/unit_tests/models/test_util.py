# Authors: Robin Schirrmeister <robintibor@gmail.com>
#          Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)

import pytest
import numpy as np
from torch import nn
from sklearn.preprocessing import OneHotEncoder

from braindecode.models.modules import Expression
from braindecode.models.util import (
    get_output_shape, aggregate_probas, _pad_shift_array)


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
    dense_y = np.random.RandomState(33).rand(
        n_sequences, n_classes, n_windows).astype(dtype)
    n_outputs = (n_sequences - 1) * stride + n_windows

    # Align sequences with _pad_shift_array
    shifted_y = _pad_shift_array(dense_y, stride=stride)

    # Align sequences explicitly (to reproduce output of _pad_shift_array)
    shifted_y2 = np.concatenate([
        np.concatenate((
            np.zeros((1, n_classes, i * stride)),
            dense_y[[i]],
            np.zeros((1, n_classes, n_outputs - n_windows - i * stride))),
            axis=2)
        for i in range(n_sequences)], axis=0)

    assert (shifted_y == shifted_y2).all()


def test_pad_shift_array_not_3d():
    with pytest.raises(NotImplementedError):
        _pad_shift_array(np.zeros((2, 2)))


@pytest.mark.parametrize('n_sequences,n_classes,n_windows,stride',
                         [[3, 3, 2, 2], [3, 3, 1, 1], [10, 3, 2, 1]])
def test_aggregate_probas(n_sequences, n_classes, n_windows, stride):
    # Create fake matrix of logits where each example has a logit of 1 for the
    # given class and zeros elsewhere
    n_outputs = (n_sequences - 1) * stride + n_windows
    y_true = np.arange(n_outputs) % n_classes  # fake target for each window
    logits = OneHotEncoder(sparse=False).fit_transform(y_true.reshape(-1, 1))
    logits = np.lib.stride_tricks.sliding_window_view(  # extract sequences
        logits, n_windows, axis=0)[::stride]

    y_pred_probas = aggregate_probas(logits, n_windows_stride=stride)

    # Make sure shape is right
    assert y_pred_probas.ndim == 2
    assert y_pred_probas.shape == (n_outputs, n_classes)

    # Make sure results of aggregation match the original targets
    assert (y_pred_probas.argmax(axis=1) == y_true).all()
