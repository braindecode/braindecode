# Authors: Cédric Rommel <cpe.rommel@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import pytest
import torch
from sklearn.utils import check_random_state
from skorch.helper import to_tensor
from torch import optim

from braindecode.augmentation import AugmentedDataLoader
from braindecode.augmentation import TimeReverse
from braindecode.classifier import EEGClassifier


def pytest_addoption(parser):
    parser.addoption("--seed", action="store", type=int, default=29)


@pytest.fixture
def rng_seed(pytestconfig):
    seed = pytestconfig.getoption("seed")
    return seed


@pytest.fixture
def random_batch(rng_seed, batch_size=5):
    """ Generate batch of elements containing feature matrix of size 66x50
    filled with random floats between 0 and 1.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = check_random_state(rng_seed)
    X = torch.from_numpy(rng.random((batch_size, 22, 51))).float().to(device)
    return X, torch.zeros(batch_size)


class MockModule(torch.nn.Module):
    def __init__(self, preds):
        super().__init__()
        self.preds = to_tensor(preds, device='cpu')
        self.linear = torch.nn.Linear(5, 5)

    def forward(self, x):
        return self.preds


@pytest.fixture
def augmented_mock_clf():
    return EEGClassifier(
        MockModule(np.random.rand(4, 2)),
        optimizer=optim.Adam,
        batch_size=32,
        iterator_train=AugmentedDataLoader,
        iterator_train__transforms=TimeReverse(probability=0.5),
    )
