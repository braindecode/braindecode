# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD-3

import numpy as np
import pytest

from braindecode.datautil.signalproc import (
    exponential_running_demean, exponential_running_standardize)


@pytest.fixture(scope="module")
def mock_data():
    np.random.seed(20200217)
    mock_input = np.random.rand(2, 10).reshape(2, 10)
    expected_standardized = np.array(
        [[ 0.        , -1.41385996, -1.67770482,  1.95328935,  0.61618697,
          -0.55294099, -1.08890304,  1.04546089, -1.368485  , -1.08669994],
         [ 0.        , -1.41385996, -0.41117774,  1.65212819, -0.5392431 ,
          -0.23009334,  0.15087203, -1.45238971,  1.88407553, -0.38583499]])
    expected_demeaned = np.array(
        [[ 0.        , -0.02547392, -0.10004415,  0.47681459,  0.1399319 ,
          -0.11764405, -0.23535964,  0.22749205, -0.3155749 , -0.25316515],
         [ 0.        , -0.29211105, -0.07138808,  0.44137798, -0.13274718,
          -0.0519248 ,  0.03156507, -0.33137195,  0.52134583, -0.1020266 ]])
    return mock_input, expected_standardized, expected_demeaned


def test_exponential_running_standardize(mock_data):
    mock_input, expected_data, _ = mock_data
    standardized_data = exponential_running_standardize(mock_input)
    assert mock_input.shape == standardized_data.shape == expected_data.shape
    np.testing.assert_allclose(
        standardized_data, expected_data, rtol=1e-4, atol=1e-4)


def test_exponential_running_demean(mock_data):
    mock_input, _, expected_data = mock_data
    demeaned_data = exponential_running_demean(mock_input)
    assert mock_input.shape == demeaned_data.shape == expected_data.shape
    np.testing.assert_allclose(
        demeaned_data, expected_data, rtol=1e-4, atol=1e-4)


def test_exponential_running_init_block_size(mock_data):
    mock_input, _, _ = mock_data
    init_block_size = 3
    standardized_data = exponential_running_standardize(
        mock_input, init_block_size=init_block_size)
    np.testing.assert_allclose(
        standardized_data[:, :init_block_size].sum(), [0], rtol=1e-4, atol=1e-4)

    demeaned_data = exponential_running_demean(
        mock_input, init_block_size=init_block_size)
    np.testing.assert_allclose(
        demeaned_data[:, :init_block_size].sum(), [0], rtol=1e-4, atol=1e-4)
