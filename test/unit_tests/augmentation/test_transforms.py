import numpy as np
import torch
from braindecode.augmentation.transforms.masking_along_axis import mask_along_axis_random
from braindecode.datasets.sleep_physionet import get_dummy_sample

from braindecode.augmentation.global_variables import fft_args
from braindecode.util import set_random_seeds


def test_dummy_sample():
    _, _, _, = get_dummy_sample()


def test_mask_along_axis():
    train_sample, _, _ = get_dummy_sample()
    set_random_seeds(0, cuda=True)
    X = train_sample[0][0]
    spec = torch.stft(X, n_fft=fft_args["n_fft"],
                      hop_length=fft_args["hop_length"],
                      win_length=fft_args["win_length"],
                      window=torch.hann_window(fft_args["n_fft"]))
    aug_spec = torch.stft(X, n_fft=fft_args["n_fft"],
                          hop_length=fft_args["hop_length"],
                          win_length=fft_args["win_length"],
                          window=torch.hann_window(fft_args["n_fft"]))
    aug_spec = mask_along_axis_random(aug_spec, params={"magnitude": 0.2, "axis": 2, "mask_value": 0})
    aug_spec = mask_along_axis_random(aug_spec, params={"magnitude": 0.2, "axis": 1, "mask_value": 0})

    img = spec[0, :, :, 0].numpy()
    img_with_zeros = aug_spec[0, :, :, 0].numpy()

    # first, asserting masked transform contains at least
    # one row with zeros.
    line_has_zeros = [np.allclose(img_with_zeros[i, :], [0.0] * img_with_zeros.shape[1], atol=0.01) for i in range(img_with_zeros.shape[0])]
    column_has_zeros = [np.allclose(img_with_zeros[:, i], [0.0] * img_with_zeros.shape[0], atol=0.01) for i in range(img_with_zeros.shape[1])]

    lines_with_zeros = [i for i in range(
        len(line_has_zeros)) if line_has_zeros[i]]
    columns_with_zeros = [
        i for i in range(len(column_has_zeros)) if column_has_zeros[i]
    ]

    assert(lines_with_zeros == list(range(33, 37)))
    assert(columns_with_zeros == list(range(8, 9)))

    # Second, asserting the equality of img
    # and img_with_zeros on other elements
    where_equal = [(round(img[i, j], 5) == round(img_with_zeros[i, j], 5))
                   for i in range(img.shape[0])
                   for j in range(img.shape[1])
                   if ((i not in lines_with_zeros) and
                       (j not in columns_with_zeros))]
    print(where_equal)
    assert(all(where_equal))
