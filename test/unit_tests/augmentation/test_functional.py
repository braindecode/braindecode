import pytest
import torch

from braindecode.augmentation.functional import (
    _analytic_transform,
    amplitude_scale,
    band_rotation,
    channels_rereference,
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


def test_amplitude_scale():
    X = torch.rand((5, 64, 100))
    # Random EEG data for 5 examples, 64 channels, and 100 time points
    y = torch.randint(0, 2, (5,))
    # Random labels for 5 examples
    random_state = 42

    scale = (1,1) # sanity check
    transformed_X1, _ = amplitude_scale(X, y, scale, random_state)
    scale = (0,0) # sanity check
    transformed_X0, _ = amplitude_scale(X, y, scale, random_state)

    # Check if the output is the same as the input
    assert torch.equal(transformed_X1, X)
    assert torch.equal(transformed_X0, torch.zeros_like(X))


def test_band_rotation_shape_and_seed_reproducibility():
    # 2 bands × 8 electrodes/band; small T for fast assertions.
    X = torch.randn(4, 16, 32)
    y = torch.zeros(4)
    out1, _ = band_rotation(
        X, y, num_bands=2, electrodes_per_band=8,
        band_offsets=(-1, 0, 1), max_temporal_jitter=4,
        random_state=42,
    )
    out2, _ = band_rotation(
        X, y, num_bands=2, electrodes_per_band=8,
        band_offsets=(-1, 0, 1), max_temporal_jitter=4,
        random_state=42,
    )
    assert out1.shape == X.shape
    assert torch.equal(out1, out2)  # same seed → identical
    assert not torch.equal(out1, X)  # something rolled


def test_band_rotation_no_op_when_offsets_zero_and_no_jitter():
    X = torch.randn(2, 32, 64)
    y = torch.zeros(2)
    out, _ = band_rotation(
        X, y, num_bands=2, electrodes_per_band=16,
        band_offsets=(0,), max_temporal_jitter=0,
        random_state=0,
    )
    assert torch.equal(out, X)


def test_band_rotation_rejects_mismatched_channel_count():
    X = torch.randn(2, 30, 64)  # 30 != 2 * 16
    y = torch.zeros(2)
    with pytest.raises(ValueError, match="num_bands \\* electrodes_per_band="):
        band_rotation(
            X, y, num_bands=2, electrodes_per_band=16,
            band_offsets=(-1, 1), max_temporal_jitter=0,
            random_state=0,
        )


def test_band_rotation_circular_roll_preserves_values():
    """Per-band roll is circular: every input value should appear in the
    output (no zero-padding) when offsets are non-trivial."""
    # Use a deterministic-shape input so we can compare value sets.
    X = torch.arange(2 * 16 * 8).float().view(2, 16, 8)
    y = torch.zeros(2)
    out, _ = band_rotation(
        X, y, num_bands=2, electrodes_per_band=8,
        band_offsets=(2,),  # force a +2 roll on every band
        max_temporal_jitter=0,
        random_state=0,
    )
    assert sorted(out.flatten().tolist()) == sorted(X.flatten().tolist())


def test_channel_reref():
    X0 = torch.zeros((5, 64, 100))  # sanity
    X1 = torch.ones((5, 64, 100))  # sanity

    # Random EEG data for 5 examples, 64 channels, and 100 time points
    y = torch.randint(0, 2, (5,))
    # Random labels for 5 examples
    random_state = 42

    transformed_X0, _ = channels_rereference(X0, y, random_state)
    transformed_X1, _ = channels_rereference(X1, y, random_state)
    # Check if the output is the same as the input
    assert torch.equal(transformed_X0, X0)
    assert torch.equal(transformed_X1.sum(dim=(1,2)), -100*torch.ones(5))
