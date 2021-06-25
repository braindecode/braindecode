# Authors: Simon Brandt <simonbrandt@protonmail.com>
#
# License: BSD-3 (3-clause)

import torch
from braindecode.training.losses import mixup_criterion


def test_mixup_criterion():
    N_classes = 2
    N_samples = 5
    y_a = torch.zeros(N_samples, dtype=torch.int64)
    y_b = torch.ones(N_samples, dtype=torch.int64)
    lam = torch.arange(.1, 1, 1/N_samples)

    preds = torch.rand((N_samples, N_classes))

    target = (y_a, y_b, lam)
    loss = mixup_criterion(preds, target)
    expected = - (lam * preds[:, 0] + (1 - lam) * preds[:, 1]).mean()
    assert loss == expected

    target = y_a
    loss = mixup_criterion(preds, target)
    expected = - preds[:, 0].mean()
    assert loss == expected
