# Authors: Cédric Rommel <cpe.rommel@gmail.com>
#
# License: BSD (3-clause)

import pytest
import numpy as np


def pytest_addoption(parser):
    parser.addoption("--seed", action="store", type=int, default=29)


@pytest.fixture
def rng_seed(pytestconfig):
    seed = pytestconfig.getoption("seed")
    return seed
