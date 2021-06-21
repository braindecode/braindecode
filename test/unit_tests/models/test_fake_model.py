# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import torch
import numpy as np
import pytest
from braindecode.util import set_random_seeds
from braindecode.models import FakeConvModel


@pytest.fixture(scope="module")
def input_sizes():
    return dict(n_channels=18, n_in_times=600, n_classes=2, n_samples=7)


def check_forward_pass(model, input_sizes, only_check_until_dim=None):
    # Test 4d Input
    set_random_seeds(0, False)
    rng = np.random.RandomState(42)
    X = rng.randn(input_sizes['n_samples'], input_sizes['n_channels'],
                  input_sizes['n_in_times'], 1)
    X = torch.Tensor(X.astype(np.float32))

    # Test 3d input
    set_random_seeds(0, False)
    X = X.squeeze(-1)
    assert len(X.shape) == 3
    y_pred_new = model(X)
    assert y_pred_new.shape[:only_check_until_dim] == (
        input_sizes['n_samples'], input_sizes['n_classes'])


def test_deep4net(input_sizes):
    model = FakeConvModel(
        input_sizes['n_channels'], input_sizes['n_classes'],
    )
    check_forward_pass(model, input_sizes, only_check_until_dim=2)
