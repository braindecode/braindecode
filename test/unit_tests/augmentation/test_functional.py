import pytest
import torch

from braindecode.augmentation.functional import (
    _analytic_transform,
    amplitude_scale,
    band_rotation,
    channels_rereference,
    channels_shuffle,
    label_smoothing,
    mixup,
    segmentation_reconstruction,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
def test_mixup_preserves_input_dtype(dtype, default_dtype):
    X = torch.arange(24, dtype=dtype).reshape(2, 3, 4).requires_grad_()
    y = torch.tensor([0, 1])
    lam = torch.tensor([0.25, 0.75], dtype=dtype)
    idx_perm = torch.tensor([1, 0])
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        mixed, targets = mixup(X, y, lam, idx_perm)
    finally:
        torch.set_default_dtype(original_dtype)

    assert mixed.dtype == X.dtype
    assert mixed.device == X.device
    expected = torch.arange(9, 21, dtype=dtype).repeat(2, 1).reshape_as(X)
    torch.testing.assert_close(mixed, expected)
    assert torch.equal(targets[0], y)
    assert torch.equal(targets[1], y[idx_perm])
    assert targets[2] is lam
    mixed.sum().backward()
    torch.testing.assert_close(X.grad[0], torch.full_like(X[0], 0.5))
    torch.testing.assert_close(X.grad[1], torch.full_like(X[1], 1.5))


@pytest.mark.parametrize("dtype", [torch.int16, torch.int64, torch.bool])
@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
def test_mixup_integer_inputs_keep_floating_interpolation(dtype, default_dtype):
    X = torch.tensor([[[0, 1]], [[1, 0]]], dtype=dtype)
    y = torch.tensor([0, 1])
    lam = torch.tensor([0.25, 0.75])
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        mixed, _ = mixup(X, y, lam, torch.tensor([1, 0]))
    finally:
        torch.set_default_dtype(original_dtype)

    expected = torch.tensor([[[0.75, 0.25]], [[0.75, 0.25]]], dtype=default_dtype)
    torch.testing.assert_close(mixed, expected)


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


def test_amplitude_scale_random_state_variants():
    # regression for None default and numpy RandomState input
    import numpy as np

    X = torch.rand((3, 4, 50))
    y = torch.zeros(3)
    scale = (0.5, 2.0)

    out_none, _ = amplitude_scale(X, y, scale, random_state=None)
    assert out_none.shape == X.shape

    out_rs, _ = amplitude_scale(X, y, scale, random_state=np.random.RandomState(7))
    assert out_rs.shape == X.shape

    out_a, _ = amplitude_scale(X, y, scale, random_state=123)
    out_b, _ = amplitude_scale(X, y, scale, random_state=123)
    assert torch.equal(out_a, out_b)


def test_band_rotation_shape_and_seed_reproducibility():
    # 2 bands × 8 electrodes/band; small T for fast assertions.  Force
    # ``band_offsets=(2,)`` so we get a deterministic non-zero rotation
    # regardless of the RNG draw — the previous version asserted the
    # output differs from the input under ``random_state=42``, which
    # silently couples the test to the sampler's exact draw sequence.
    X = torch.randn(4, 16, 32)
    y = torch.zeros(4)
    kwargs = dict(
        num_bands=2, electrodes_per_band=8,
        band_offsets=(2,), max_temporal_jitter=0,
        random_state=42,
    )
    out1, _ = band_rotation(X, y, **kwargs)
    out2, _ = band_rotation(X, y, **kwargs)
    assert out1.shape == X.shape
    assert torch.equal(out1, out2)  # same seed → identical
    assert not torch.equal(out1, X)  # offsets=(2,) guarantees a roll


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


def test_band_rotation_rejects_empty_band_offsets():
    X = torch.randn(2, 32, 64)
    y = torch.zeros(2)
    with pytest.raises(ValueError, match="band_offsets must be non-empty"):
        band_rotation(
            X, y, num_bands=2, electrodes_per_band=16,
            band_offsets=(), max_temporal_jitter=0,
            random_state=0,
        )


def test_band_rotation_rejects_negative_jitter():
    X = torch.randn(2, 32, 64)
    y = torch.zeros(2)
    with pytest.raises(ValueError, match="max_temporal_jitter must be >= 0"):
        band_rotation(
            X, y, num_bands=2, electrodes_per_band=16,
            band_offsets=(-1, 1), max_temporal_jitter=-3,
            random_state=0,
        )


@pytest.mark.parametrize(
    ("kwargs", "msg"),
    [
        ({"num_bands": 0}, "num_bands must be >= 1"),
        ({"electrodes_per_band": 0}, "electrodes_per_band must be >= 1"),
        ({"band_offsets": (1.5, 2.0)}, "band_offsets must contain integers"),
    ],
)
def test_band_rotation_rejects_invalid_params(kwargs, msg):
    X = torch.randn(2, 32, 64)
    y = torch.zeros(2)
    base = dict(num_bands=2, electrodes_per_band=16, band_offsets=(-1, 1),
                max_temporal_jitter=0, random_state=0)
    base.update(kwargs)
    with pytest.raises(ValueError, match=msg):
        band_rotation(X, y, **base)


def test_band_rotation_circular_jitter_wraps_vs_zero_pads():
    """``circular_jitter=False`` zero-pads the gap; ``True`` (default)
    wraps end-of-window samples to the start.  We use a deterministic
    band_offsets=(0,) so the rotation step is a no-op and the only
    visible change is the temporal shift on band 1."""
    # Band 0 = constant 1.0, band 1 = ramp 1..T (avoid 0 so we can check
    # "is 0 present?" as a marker of zero-padding).  Channel layout
    # ``(B=1, num_bands*E=4, T=8)`` keeps assertions readable.
    T = 8
    band0 = torch.ones(1, 2, T)
    band1 = torch.arange(1, T + 1, dtype=torch.float).expand(1, 2, T).clone()
    X = torch.cat([band0, band1], dim=1)  # (1, 4, T)
    y = torch.zeros(1)

    common = dict(
        num_bands=2, electrodes_per_band=2, band_offsets=(0,),
        max_temporal_jitter=3,
    )
    # With a fixed seed, both branches sample the same shift; only the
    # boundary handling differs.  ``shift`` is drawn from
    # ``[-max_temporal_jitter, +max_temporal_jitter]`` and may legitimately
    # be 0, so we branch on the observed ``n_diff`` rather than assuming a
    # particular RNG draw — the test stays valid even if the sampler order
    # changes upstream.
    out_circ, _ = band_rotation(X, y, **common, circular_jitter=True, random_state=11)
    out_pad, _ = band_rotation(X, y, **common, circular_jitter=False, random_state=11)

    diff = (out_circ != out_pad).any(dim=1).squeeze(0)  # (T,)
    n_diff = int(diff.sum())
    if n_diff == 0:
        # shift==0: both modes are no-ops on the time axis (band_offsets=(0,)
        # also makes the channel roll a no-op), so outputs must equal X.
        assert torch.equal(out_circ, X)
        assert torch.equal(out_pad, X)
        return
    # shift!=0: differences are confined to the |shift|-wide boundary on
    # band 1; padded variant zeros that boundary while circular wraps it.
    assert n_diff <= common["max_temporal_jitter"]
    band1_pad = out_pad[0, 2:, :]
    band1_circ = out_circ[0, 2:, :]
    assert (band1_pad == 0).any()
    assert not (band1_circ == 0).any()


def test_band_rotation_circular_roll_preserves_values():
    """Per-band roll is circular: every input value should appear in the
    output (no zero-padding) when offsets are non-trivial."""
    # Small, deterministic input so the sorted-list comparison stays
    # ``O(n log n)`` on a few-hundred-element tensor.  Don't scale this up
    # to large shapes — for production-size inputs use a multiset-aware
    # check (e.g. ``torch.unique(..., return_counts=True)`` or histogram
    # comparison); a ``set`` would silently drop duplicates and miss
    # value-multiplicity bugs.
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


def test_label_smoothing():
    X = torch.rand((5, 64, 100))
    y = torch.tensor([0, 1, 2, 1, 0])
    epsilon, num_classes = 0.1, 3

    transformed_X, transformed_y = label_smoothing(X, y, epsilon, num_classes)

    assert torch.equal(transformed_X, X)
    expected = torch.eye(num_classes)[y] * (1 - epsilon) + epsilon / num_classes
    assert torch.allclose(transformed_y, expected)
    assert torch.allclose(transformed_y.sum(-1), torch.ones(5))


def test_label_smoothing_rejects_invalid_targets():
    X = torch.rand((2, 8, 32))
    with pytest.raises(ValueError, match="num_classes is required"):
        label_smoothing(X, torch.tensor([0, 1]), 0.1)
    with pytest.raises(ValueError, match="non-integral"):
        label_smoothing(X, torch.tensor([0.0, 1.5]), 0.1, num_classes=3)
    with pytest.raises(ValueError, match="dtype"):
        label_smoothing(X, torch.tensor([[0, 1], [1, 0]]), 0.1)


def _scalar_mixup_reference(X, y, lam, idx_perm):
    """Independent pre-vectorization loop, including promotion and allocation."""
    batch_size, n_channels, n_times = X.shape
    mixed = X.new_zeros(
        (batch_size, n_channels, n_times),
        dtype=X.dtype if X.is_floating_point() else torch.get_default_dtype(),
    )
    y_a = y.clone()
    y_b = y[idx_perm].clone()
    for idx in range(batch_size):
        mixed[idx] = lam[idx] * X[idx] + (1 - lam[idx]) * X[idx_perm[idx]]
    return mixed, (y_a, y_b, lam)


@pytest.mark.parametrize("batch_size", [0, 1, 2, 17])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32,
                                   torch.float64, torch.int16, torch.int64, torch.bool])
@pytest.mark.parametrize("lam_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("storage", ["contiguous", "transposed", "sliced"])
@pytest.mark.parametrize("target_shape", [(), (1,), (2,)])
def test_mixup_vectorization_matches_scalar_reference(
    batch_size, dtype, lam_dtype, default_dtype, storage, target_shape
):
    # A cast before complement, normal tensor promotion, or changed output
    # allocation would fail this exact-value comparison to the scalar oracle.
    generator = torch.Generator().manual_seed(143)
    base = torch.randn(batch_size, 3, 14, generator=generator).mul(13).to(dtype)
    X = base[:, :, :7].contiguous()
    if storage == "transposed":
        X = X.transpose(1, 2)
    elif storage == "sliced":
        X = base[:, :, ::2]
    lam = torch.linspace(0, 1, batch_size, dtype=lam_dtype)
    if batch_size > 2:
        lam[2] = 0.12345678912345678
    idx_perm = torch.zeros(batch_size, dtype=torch.long)  # duplicates intentionally
    y = torch.arange(batch_size * (2 if target_shape == (2,) else 1),
                     dtype=torch.float64).add(0.25).reshape(batch_size, *target_shape)
    before = [v.clone() for v in (X, y, lam, idx_perm)]
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        expected, expected_targets = _scalar_mixup_reference(X, y, lam, idx_perm)
        actual, targets = mixup(X, y, lam, idx_perm)
    finally:
        torch.set_default_dtype(original_dtype)
    assert torch.equal(actual, expected)
    assert actual.dtype == expected.dtype
    assert actual.stride() == expected.stride()
    assert targets[2] is lam
    for actual_target, expected_target in zip(targets, expected_targets):
        assert torch.equal(actual_target, expected_target)
        assert actual_target.dtype == expected_target.dtype
    if batch_size:
        assert targets[0].data_ptr() != y.data_ptr()
        assert targets[1].data_ptr() != y.data_ptr()
    for original, saved in zip((X, y, lam, idx_perm), before):
        assert torch.equal(original.flatten().contiguous().view(torch.uint8),
                           saved.flatten().contiguous().view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("lam_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
def test_mixup_gradient_and_rng_contract(dtype, lam_dtype, default_dtype):
    generator = torch.Generator().manual_seed(51)
    X = torch.randn(17, 3, 7, generator=generator, dtype=dtype).requires_grad_()
    lam = torch.linspace(0.01, 0.99, 17, dtype=lam_dtype).requires_grad_()
    y = torch.arange(17)
    idx_perm = torch.tensor([0, 0, 3, 2, 2, 5, 4, 3, 1, 6, 7, 8, 9, 10, 10, 11, 12])
    reference_X = X.detach().clone().requires_grad_()
    reference_lam = lam.detach().clone().requires_grad_()
    state = torch.random.get_rng_state().clone()
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        actual, _ = mixup(X, y, lam, idx_perm)
        expected, _ = _scalar_mixup_reference(reference_X, y, reference_lam, idx_perm)
        weight = torch.linspace(-1, 1, actual.numel(), dtype=dtype).reshape_as(actual)
        (actual * weight).sum().backward()
        (expected * weight).sum().backward()
    finally:
        torch.set_default_dtype(original_dtype)
    assert torch.equal(actual, expected)
    torch.testing.assert_close(X.grad, reference_X.grad)
    torch.testing.assert_close(lam.grad, reference_lam.grad)
    assert torch.equal(state, torch.random.get_rng_state())
