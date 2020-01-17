# Authors: Lukas Gemein
#          Robin Tibor Schirrmeister
#          Alexandre Gramfort
#
# License: BSD-3

import numpy as np
from torch.utils.data import Dataset

from braindecode.datautil.loader import CropsDataLoader
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.datautil import SignalAndTarget


class EEGDataSet(Dataset):
    def __init__(self, X, y):
        self.X = X
        if self.X.ndim == 3:
            self.X = self.X[:, :, :, None]
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        i_trial, start, stop = idx
        return self.X[i_trial, :, start:stop], self.y[i_trial]


def test_crops_data_loader_regression():
    """Test CropsDataLoader."""

    # Convert data from volt to millivolt
    # Pytorch expects float32 for input and int64 for labels.
    rng = np.random.RandomState(42)
    X = rng.randn(60, 64, 343).astype(np.float32)
    y = rng.randint(0, 2, size=len(X))

    train_set = SignalAndTarget(X, y=y)
    input_time_length = X.shape[2]
    n_preds_per_input = X.shape[2] // 4

    n_times_input = input_time_length  # size of signal passed to nn
    batch_size = 32

    iterator = CropsFromTrialsIterator(batch_size=batch_size,
                                       input_time_length=n_times_input,
                                       n_preds_per_input=n_preds_per_input)

    ds = EEGDataSet(train_set.X, train_set.y)

    loader = \
        CropsDataLoader(ds, batch_size=batch_size,
                        input_time_length=input_time_length,
                        n_preds_per_input=n_preds_per_input,
                        num_workers=0)

    for (X1b, y1b), (X2b, y2b) in \
            zip(iterator.get_batches(train_set, shuffle=False), loader):
        np.testing.assert_array_equal(y1b, y2b)
        np.testing.assert_array_equal(X1b, X2b)


def test_crops_data_loader_explicit():

    X = np.arange(0, 15)
    y = [0]

    n_time_in = 10
    n_time_out = 4

    expected_crops = [np.arange(0, 10),
                      np.arange(4, 14),
                      np.arange(5, 15)]


    dataset = EEGDataSet(X[None, None], y)
    loader = CropsDataLoader(dataset, n_time_in, n_time_out,
                             batch_size=3)

    Xs, ys = zip(*list(loader))

    assert len(Xs) == len(ys) == 1

    for expected, actual in zip(Xs[0].squeeze(), expected_crops):
        np.testing.assert_array_equal(expected, actual)
