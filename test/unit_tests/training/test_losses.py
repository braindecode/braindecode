# Authors: Simon Brandt <simonbrandt@protonmail.com>
#          Maciej Sliwowski <maciek.sliwowski@gmail.com>
#
# License: BSD-3 (3-clause)

import pytest
import numpy as np
import torch

from braindecode.training.losses import mixup_criterion, TimeSeriesLoss


def test_mixup_criterion():
    n_classes = 2
    n_samples = 5
    y_a = torch.zeros(n_samples, dtype=torch.int64)
    y_b = torch.ones(n_samples, dtype=torch.int64)
    lam = torch.arange(.1, 1, 1 / n_samples)

    preds = torch.Tensor(
        np.random.RandomState(42).randn(n_samples, n_classes)
    )

    target = (y_a, y_b, lam)
    loss = mixup_criterion(preds, target)
    expected = - (lam * preds[:, 0] + (1 - lam) * preds[:, 1]).mean()
    assert loss == pytest.approx(expected)

    target = y_a
    loss = mixup_criterion(preds, target)
    expected = - preds[:, 0].mean()
    assert loss == pytest.approx(expected)


def test_time_series_loss():
    targets = torch.Tensor(
        np.array(
            [
                [[np.nan, 0.2, np.nan, 0.3, np.nan, 0.1, np.nan, 0.9],
                 [np.nan, 0.8, np.nan, 0.2, np.nan, 0.9, np.nan, 0.2]],
                [[np.nan, 0.3, np.nan, 0.2, np.nan, 0.5, np.nan, 0.1],
                 [np.nan, 0.1, np.nan, 0.2, np.nan, 0.3, np.nan, 0.2]]
            ]
        ))
    targets_expected = torch.Tensor(
        np.array(
            [
                [[0.1, 0.9], [0.9, 0.2]],
                [[0.5, 0.1], [0.3, 0.2]]
            ]
        ))

    preds = torch.Tensor(
        np.array(
            [
                [[0.4, 0.9, 0.4], [0.1, 0.3, 0.8]],
                [[0.2, 0.5, 0.3], [0.2, 0.1, 0.5]],
            ]
        ))
    preds_expected = torch.Tensor(
        np.array(
            [
                [[0.4, 0.4], [0.1, 0.8]],
                [[0.2, 0.3], [0.2, 0.5]]
            ]
        ))

    time_series_loss = TimeSeriesLoss(lambda *args: args)

    preds_out, targets_out = time_series_loss(preds, targets)

    torch.testing.assert_allclose(preds_out, preds_expected.flatten())
    torch.testing.assert_allclose(targets_out, targets_expected.flatten())
