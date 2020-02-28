# Authors: Lukas Gemein
#          Robin Tibor Schirrmeister
#          Alexandre Gramfort
#
# License: BSD-3

import numpy as np
from braindecode.datasets.croppedxy import CroppedXyDataset


def test_crops_data_loader_explicit():

    X = np.arange(0, 15)
    y = [0]

    n_time_in = 10
    n_time_out = 4

    expected_crops = [np.arange(0, 10), np.arange(4, 14), np.arange(5, 15)]

    dataset = CroppedXyDataset(X[None, None], y,
        input_time_length=n_time_in,
        n_preds_per_input=n_time_out,
    )

    Xs, ys, i_s = zip(*list(dataset))
    print(Xs,)

    assert len(Xs) == len(ys) == 3

    for actual, expected,  in zip(Xs, expected_crops):
        np.testing.assert_array_equal(actual.squeeze(), expected)
