# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD-3

import numpy as np
from braindecode.datasets.xy import create_from_X_y


def test_crops_data_loader_explicit():

    X = np.arange(0, 15)
    y = [0]

    n_time_in = 10
    n_time_out = 4
    sfreq = 100

    expected_crops = [np.arange(0, 10), np.arange(4, 14), np.arange(5, 15)]

    dataset = create_from_X_y(
        X[None, None], y,
        sfreq=sfreq,
        window_size_samples=n_time_in,
        window_stride_samples=n_time_out,
        drop_last_window=False
    )

    Xs, ys, i_s = zip(*list(dataset))

    assert len(Xs) == len(ys) == 3

    for actual, expected in zip(Xs, expected_crops):
        np.testing.assert_array_equal(actual.squeeze(), expected)
