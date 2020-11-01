import numpy as np
from numpy.testing import assert_almost_equal
import torch
import random
from braindecode.augmentation.transforms.masking_along_axis import mask_along_frequency, mask_along_time
from test.unit_tests.augmentation.utils import get_dummy_sample
from braindecode.datasets.transform_classes import Transform
from braindecode.datasets.base import Datum
from braindecode.augmentation.transforms.global_variables import fft_args

def test_augmented_training_manager():
    train_sample, valid_sample, test_sample = get_dummy_sample()

def test_mask_along_axis():
    train_sample, _, _ = get_dummy_sample()

    datum = Datum(X=train_sample[0][0], y=train_sample[0][1])
    aug_datum = Datum(X=train_sample[0][0], y=train_sample[0][1])
    aug_datum = Transform(mask_along_time, 1, 0.1)(aug_datum)
    aug_datum = Transform(mask_along_frequency, 1, 0.1)(aug_datum)
    
    global fft_args
    aug_spec = torch.stft(aug_datum.X, n_fft=fft_args["n_fft"],
                        hop_length=fft_args["hop_length"],
                        win_length=fft_args["win_length"],
                        window=torch.hann_window(fft_args["n_fft"]))
    spec = torch.stft(datum.X, n_fft=fft_args["n_fft"],
                      hop_length=fft_args["hop_length"],
                      win_length=fft_args["win_length"],
                      window=torch.hann_window(fft_args["n_fft"]))
    
    img = spec.X[0, :, :, 0].numpy()
    img_with_zeros = aug_spec.X[0, :, :, 0].numpy()
    # first, asserting masked transform contains at least
    # one row with zeros.
    line_has_zeros = np.all(assert_almost_equal(img_with_zeros, 0.0, decimal=5), axis=0)
    column_has_zeros = np.all(assert_almost_equal(img_with_zeros, 0.0, decimal=5), axis=1)

    lines_with_zeros = [i for i in range(
        len(line_has_zeros)) if line_has_zeros[i]]
    columns_with_zeros = [
        i for i in range(len(column_has_zeros)) if column_has_zeros[i]
    ]
    
    assert(lines_with_zeros == list(range(5, 10)))
    assert(columns_with_zeros == list(range(4, 25)))

    # Second, asserting the equality of img
    # and img_with_zeros on other elements
    where_equal = [(round(img[i, j], 5) == round(img_with_zeros[i, j], 5))
                   for i in range(img.shape[0])
                   for j in range(img.shape[1])
                   if ((j not in lines_with_zeros)
                       and (i not in columns_with_zeros))]

    assert(all(where_equal))
