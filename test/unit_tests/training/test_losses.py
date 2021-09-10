# Authors: Simon Brandt <simonbrandt@protonmail.com>
#
# License: BSD-3 (3-clause)

import pytest
import numpy as np
import torch

from braindecode.training.losses import mixup_criterion


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
