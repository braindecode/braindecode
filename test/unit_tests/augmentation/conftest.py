# Authors: Cédric Rommel <cpe.rommel@gmail.com>
#
# License: BSD (3-clause)

import pytest
import torch
from sklearn.utils import check_random_state


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
    X = torch.from_numpy(rng.random((batch_size, 66, 51))).float().to(device)
    return X, torch.zeros(batch_size)
