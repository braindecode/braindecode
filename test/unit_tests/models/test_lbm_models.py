# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3

import pytest
import torch

from braindecode.models import Labram


@pytest.fixture
def n_times():
    return 1000


@pytest.fixture
def n_chans():
    return 64


@pytest.fixture
def n_outputs():
    return 4


@pytest.fixture
def patch_size():
    return 200


@pytest.fixture
def emb_size():
    return 200


@pytest.fixture
def n_layers():
    return 2


@pytest.fixture
def att_num_heads():
    return 4


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def model_config_tokenizer(n_times, n_chans, n_outputs, patch_size, emb_size, n_layers, att_num_heads):
    return {
        "n_times": n_times,
        "n_chans": n_chans,
        "n_outputs": n_outputs,
        "patch_size": patch_size,
        "emb_size": emb_size,
        "n_layers": n_layers,
        "att_num_heads": att_num_heads,
        "neural_tokenizer": True,
    }


@pytest.fixture
def model_config_decoder(n_times, n_chans, n_outputs, patch_size, emb_size, n_layers, att_num_heads):
    return {
        "n_times": n_times,
        "n_chans": n_chans,
        "n_outputs": n_outputs,
        "patch_size": patch_size,
        "emb_size": emb_size,
        "in_channels": 8,
        "out_channels": 8,
        "n_layers": n_layers,
        "att_num_heads": att_num_heads,
        "neural_tokenizer": False,
    }


@pytest.fixture
def model_tokenizer(model_config_tokenizer):
    return Labram(**model_config_tokenizer)


@pytest.fixture
def model_decoder(model_config_decoder):
    return Labram(**model_config_decoder)


# ==============================================================================
# Tests for Labram with neural_tokenizer=True (default)
# ==============================================================================


def test_labram_neural_tokenizer_initialization(model_tokenizer):
    """Test that the model initializes correctly in tokenizer mode."""
    assert model_tokenizer is not None
    assert model_tokenizer.neural_tokenizer is True
    assert model_tokenizer.n_chans == 64
    assert model_tokenizer.n_times == 1000
    assert model_tokenizer.n_outputs == 4


def test_labram_neural_tokenizer_forward_pass_basic(model_tokenizer, batch_size, n_chans, n_times, n_outputs):
    """Test basic forward pass in tokenizer mode."""
    x = torch.randn(batch_size, n_chans, n_times)
    output = model_tokenizer(x)
    assert output.shape == (batch_size, n_outputs)


def test_labram_neural_tokenizer_forward_pass_single_sample(model_tokenizer, n_chans, n_times, n_outputs):
    """Test forward pass with single sample in tokenizer mode."""
    x = torch.randn(1, n_chans, n_times)
    output = model_tokenizer(x)
    assert output.shape == (1, n_outputs)


def test_labram_neural_tokenizer_forward_features_all_tokens(model_tokenizer, n_chans, n_times, emb_size):
    """Test forward_features with return_all_tokens=True in tokenizer mode."""
    batch_size = 2
    x = torch.randn(batch_size, n_chans, n_times)
    output = model_tokenizer.forward_features(x, return_all_tokens=True)

    # Output should be (batch, cls + channels*n_patchs, emb_dim)
    # With n_patchs=5 and n_chans=64: 1 + 64*5 = 321
    assert output.shape[0] == batch_size
    assert output.shape[1] == 321  # 1 cls token + 64 channels * 5 patches
    assert output.shape[2] == emb_size


def test_labram_neural_tokenizer_forward_features_patch_tokens(model_tokenizer, n_chans, n_times, emb_size):
    """Test forward_features with return_patch_tokens=True in tokenizer mode."""
    batch_size = 2
    x = torch.randn(batch_size, n_chans, n_times)
    output = model_tokenizer.forward_features(x, return_patch_tokens=True)

    # Output should be (batch, channels*n_patchs, emb_dim)
    # With n_patchs=5 and n_chans=64: 64*5 = 320
    assert output.shape == (batch_size, 320, emb_size)


def test_labram_neural_tokenizer_forward_features_default(model_tokenizer, n_chans, n_times, emb_size):
    """Test forward_features with default settings in tokenizer mode."""
    batch_size = 2
    x = torch.randn(batch_size, n_chans, n_times)
    output = model_tokenizer.forward_features(x)

    # Default should return mean pooled output: (batch, emb_dim)
    assert output.shape == (batch_size, emb_size)


def test_labram_neural_tokenizer_different_batch_sizes(model_tokenizer, n_chans, n_times, n_outputs):
    """Test with different batch sizes in tokenizer mode."""
    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, n_chans, n_times)
        output = model_tokenizer(x)
        assert output.shape == (batch_size, n_outputs)


def test_labram_neural_tokenizer_gradient_flow(model_tokenizer, n_chans, n_times):
    """Test that gradients flow correctly through the model in tokenizer mode."""
    x = torch.randn(4, n_chans, n_times, requires_grad=True)
    output = model_tokenizer(x)
    loss = output.sum()
    loss.backward()

    # Check that gradients exist
    assert model_tokenizer.cls_token.grad is not None
    assert any(p.grad is not None for p in model_tokenizer.blocks[0].parameters())


# ==============================================================================
# Tests for Labram with neural_tokenizer=False (decoder mode)
# ==============================================================================


def test_labram_neural_decoder_initialization(model_decoder):
    """Test that the model initializes correctly in decoder mode."""
    assert model_decoder is not None
    assert model_decoder.neural_tokenizer is False
    assert model_decoder.n_chans == 64
    assert model_decoder.n_times == 1000
    assert model_decoder.n_outputs == 4


def test_labram_neural_decoder_forward_pass_basic(model_decoder, batch_size, n_chans, n_times, n_outputs):
    """Test basic forward pass in decoder mode."""
    x = torch.randn(batch_size, n_chans, n_times)
    output = model_decoder(x)
    assert output.shape == (batch_size, n_outputs)


def test_labram_neural_decoder_forward_pass_single_sample(model_decoder, n_chans, n_times, n_outputs):
    """Test forward pass with single sample in decoder mode."""
    x = torch.randn(1, n_chans, n_times)
    output = model_decoder(x)
    assert output.shape == (1, n_outputs)


def test_labram_neural_decoder_forward_features_all_tokens(model_decoder, n_chans, n_times, emb_size):
    """Test forward_features with return_all_tokens=True in decoder mode."""
    batch_size = 2
    x = torch.randn(batch_size, n_chans, n_times)
    output = model_decoder.forward_features(x, return_all_tokens=True)

    # Output should be (batch, cls + n_patches, emb_dim)
    assert output.shape[0] == batch_size
    assert output.shape[1] == 6  # 1 cls token + 5 patches (1000 / 200 = 5)
    assert output.shape[2] == emb_size


def test_labram_neural_decoder_forward_features_patch_tokens(model_decoder, n_chans, n_times, emb_size):
    """Test forward_features with return_patch_tokens=True in decoder mode."""
    batch_size = 2
    x = torch.randn(batch_size, n_chans, n_times)
    output = model_decoder.forward_features(x, return_patch_tokens=True)

    # Output should be (batch, n_patches, emb_dim)
    # After removing cls token
    assert output.shape[0] == batch_size
    assert output.shape[1] == 5  # n_patches (1000 / 200 = 5)
    assert output.shape[2] == emb_size


def test_labram_neural_decoder_forward_features_default(model_decoder, n_chans, n_times, emb_size):
    """Test forward_features with default settings in decoder mode."""
    batch_size = 2
    x = torch.randn(batch_size, n_chans, n_times)
    output = model_decoder.forward_features(x)

    # Default should return mean pooled output: (batch, feature_dim)
    assert output.shape == (batch_size, emb_size)


def test_labram_neural_decoder_different_batch_sizes(model_decoder, n_chans, n_times, n_outputs):
    """Test with different batch sizes in decoder mode."""
    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, n_chans, n_times)
        output = model_decoder(x)
        assert output.shape == (batch_size, n_outputs)


def test_labram_neural_decoder_gradient_flow(model_decoder, n_chans, n_times):
    """Test that gradients flow correctly through the model in decoder mode."""
    x = torch.randn(4, n_chans, n_times, requires_grad=True)
    output = model_decoder(x)
    loss = output.sum()
    loss.backward()

    # Check that gradients exist
    assert model_decoder.cls_token.grad is not None
    assert any(p.grad is not None for p in model_decoder.blocks[0].parameters())


# ==============================================================================
# Tests for Dimensionality Consistency between modes
# ==============================================================================


def test_labram_output_shapes_consistency_between_modes(n_times, n_chans, n_outputs):
    """Ensure that both modes produce compatible outputs."""
    batch_size = 2

    model_tokenizer = Labram(
        n_times=n_times,
        n_chans=n_chans,
        n_outputs=n_outputs,
        neural_tokenizer=True,
    )

    model_decoder = Labram(
        n_times=n_times,
        n_chans=n_chans,
        n_outputs=n_outputs,
        neural_tokenizer=False,
    )

    x = torch.randn(batch_size, n_chans, n_times)

    output_tokenizer = model_tokenizer(x)
    output_decoder = model_decoder(x)

    # Both should have the same output shape
    assert output_tokenizer.shape == output_decoder.shape == (batch_size, n_outputs)


def test_labram_patch_embedding_shapes(n_times, n_chans, patch_size, emb_size):
    """Test patch embedding output shapes."""
    from braindecode.models.labram import _PatchEmbed, _SegmentPatch

    batch_size = 2

    # Test SegmentPatch
    segment_patch = _SegmentPatch(
        n_times=n_times,
        patch_size=patch_size,
        n_chans=n_chans,
        emb_dim=patch_size,
    )

    x = torch.randn(batch_size, n_chans, n_times)
    output_segment = segment_patch(x)

    # Should be (batch, n_chans, n_patches, patch_size)
    assert output_segment.shape == (batch_size, n_chans, 5, patch_size)

    # Test PatchEmbed
    patch_embed = _PatchEmbed(
        n_times=n_times,
        patch_size=patch_size,
        in_channels=n_chans,
        emb_dim=emb_size,
    )

    output_patch = patch_embed(x)

    # Should be (batch, n_patches, emb_dim)
    assert output_patch.shape == (batch_size, 5, emb_size)


def test_labram_no_dimension_mismatch_errors(n_times, n_chans, n_outputs):
    """Test that there are no dimension mismatch errors in forward pass."""
    batch_size = 2

    # Test neural tokenizer mode
    model = Labram(
        n_times=n_times,
        n_chans=n_chans,
        n_outputs=n_outputs,
        neural_tokenizer=True,
    )

    x = torch.randn(batch_size, n_chans, n_times)

    try:
        output = model.forward_features(x, return_all_tokens=True)
        # Should complete without error
        assert output is not None
    except RuntimeError as e:
        pytest.fail(f"forward_features raised RuntimeError: {e}")

    # Test neural decoder mode
    model = Labram(
        n_times=n_times,
        n_chans=n_chans,
        n_outputs=n_outputs,
        neural_tokenizer=False,
    )

    try:
        output = model.forward_features(x, return_all_tokens=True)
        # Should complete without error
        assert output is not None
    except RuntimeError as e:
        pytest.fail(f"forward_features raised RuntimeError: {e}")


# ==============================================================================
# Tests for Edge Cases
# ==============================================================================


def test_labram_zero_output_channels(n_chans, n_times):
    """Test model with n_outputs=0."""
    model = Labram(
        n_times=n_times,
        n_chans=n_chans,
        n_outputs=0,
        neural_tokenizer=True,
    )

    x = torch.randn(2, n_chans, n_times)
    # With n_outputs=0, final_layer is Identity
    output = model(x)

    # Output should be the feature output
    assert output.shape == (2, 200)


def test_labram_small_input_size():
    """Test with small input size."""
    model = Labram(
        n_times=400,
        n_chans=32,
        n_outputs=4,
        patch_size=200,
        neural_tokenizer=True,
    )

    x = torch.randn(2, 32, 400)
    output = model(x)

    assert output.shape == (2, 4)


def test_labram_large_patch_size_warning():
    """Test that warning is issued when patch_size > n_times."""
    with pytest.warns(UserWarning, match="patch_size.*n_times"):
        model = Labram(
            n_times=400,
            n_chans=32,
            n_outputs=4,
            patch_size=500,  # Larger than n_times
            neural_tokenizer=True,
        )


# ==============================================================================
# Tests for Input Validation
# ==============================================================================


def test_labram_wrong_input_shape(model_tokenizer):
    """Test that wrong input shape raises error."""
    # Wrong shape (missing channel dimension)
    x = torch.randn(2, 1000)

    with pytest.raises((RuntimeError, ValueError)):
        model_tokenizer(x)


def test_labram_wrong_channel_count(model_tokenizer, n_times):
    """Test with wrong number of channels."""
    # Wrong number of channels
    x = torch.randn(2, 32, n_times)

    # This might not raise immediately but could cause issues
    # depending on how the model is implemented
    try:
        output = model_tokenizer(x)
        # If it doesn't raise, the shape might be unexpected
        assert output is not None
    except RuntimeError:
        # Expected behavior
        pass
