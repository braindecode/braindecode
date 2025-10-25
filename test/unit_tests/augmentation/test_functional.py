import pytest
import torch

from braindecode.augmentation.functional import (
    _analytic_transform,
    channels_shuffle,
    segmentation_reconstruction,
)


def test_channels_shuffle():
    X = torch.rand((5, 64, 100))
    # Random EEG data for 5 examples, 64 channels, and 100 time points
    y = torch.randint(0, 2, (5,))
    # Random labels for 5 examples
    p_shuffle = 0
    random_state = 42

    transformed_X, transformed_y = channels_shuffle(X, y, p_shuffle,
                                                    random_state)

    # Check if the output is the same as the input
    assert torch.equal(transformed_X, X)
    assert torch.equal(transformed_y, y)


def test_analytic_transform():
    # Create a real tensor
    x = torch.rand((5, 64,
                    100))
    # Random data for 5 examples, 64 channels, and 100 time points

    # Call the _analytic_transform function
    transformed_x = _analytic_transform(x)

    # Check if the output is complex
    assert transformed_x.is_complex()

    # Check if the output has the same shape as the input
    assert transformed_x.shape == x.shape

    # Check if the function raises a ValueError when the input is complex
    with pytest.raises(ValueError):
        _analytic_transform(torch.complex(torch.rand(5), torch.rand(5)))


def test_analytic_transform_even():
    # Create a real tensor with an even length in the last dimension
    x = torch.rand((5, 64, 100))
    # Random data for 5 examples, 64 channels, and 100 time points

    # Call the _analytic_transform function
    transformed_x = _analytic_transform(x)

    # Check if the output is complex
    assert transformed_x.is_complex()

    # Check if the output has the same shape as the input
    assert transformed_x.shape == x.shape


def test_segmentation_reconstruction():
    X = torch.zeros((20, 64, 100))
    # Random EEG data for 20 examples, 64 channels, and 100 time points
    y = torch.randint(0, 4, (20,))
    # Random labels for 5 examples

    n_segments = 5

    from sklearn.utils import check_random_state
    rng = check_random_state(42)

    classes = torch.unique(y)
    data_classes = [(i, X[y == i]) for i in classes]

    rand_idxs = dict()
    for label, X_class in data_classes:
        n_trials = X_class.shape[0]
        rand_idxs[label] = rng.randint(0, n_trials, (n_trials, n_segments))

    idx_shuffle = rng.permutation(X.shape[0])

    transformed_X, transformed_y = segmentation_reconstruction(X, y, n_segments,
                                                               data_classes, rand_idxs,
                                                               idx_shuffle)

    # Check the output
    assert torch.equal(transformed_X, X)
    # preserve time sequence
    assert torch.equal(torch.bincount(transformed_y), torch.bincount(y))
    # preserve number of occurrences of each label
