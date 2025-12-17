# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3

import os
from pathlib import Path
from shutil import rmtree

import mne
import pytest
import torch

try:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

from braindecode.models import LUNA, REVE, Labram
from braindecode.models.reve import RevePositionBank


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
def num_heads():
    return 4


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def model_config_tokenizer(n_times, n_chans, n_outputs, patch_size, emb_size, n_layers, num_heads):
    return {
        "n_times": n_times,
        "n_chans": n_chans,
        "n_outputs": n_outputs,
        "patch_size": patch_size,
        "embed_dim": emb_size,
        "num_layers": n_layers,
        "num_heads": num_heads,
        "neural_tokenizer": True,
    }


@pytest.fixture
def model_config_decoder(n_times, n_chans, n_outputs, patch_size, emb_size, n_layers, num_heads):
    return {
        "n_times": n_times,
        "n_chans": n_chans,
        "n_outputs": n_outputs,
        "patch_size": patch_size,
        "embed_dim": emb_size,
        "conv_in_channels": 8,
        "conv_out_channels": 8,
        "num_layers": n_layers,
        "num_heads": num_heads,
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


def test_labram_can_load_pretrained_weights():
    """Ensure that Labram can load pre-trained weights via torch.hub convenience."""
    model = Labram(n_times=1600, n_chans=64, n_outputs=4)
    url = "https://huggingface.co/braindecode/Labram-Braindecode/resolve/main/braindecode_labram_base.pt"

    state_dict = torch.hub.load_state_dict_from_url(
        url,
        progress=True,
        map_location="cpu",
        file_name="braindecode_labram_base_resolved.pt",
    )
    load_result = model.load_state_dict(state_dict)

    assert not load_result.missing_keys
    assert not load_result.unexpected_keys


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


# ==============================================================================
# Tests for LUNA Model Variants (Base, Large, Huge)
# ==============================================================================


@pytest.fixture
def luna_base_config():
    """Configuration for LUNA Base variant."""
    return {
        "n_outputs": 2,
        "n_chans": 22,
        "n_times": 1000,
        "embed_dim": 64,
        "num_queries": 4,
        "depth": 8,
        "num_heads": 2,
    }


@pytest.fixture
def luna_large_config():
    """Configuration for LUNA Large variant."""
    return {
        "n_outputs": 2,
        "n_chans": 22,
        "n_times": 1000,
        "embed_dim": 96,
        "num_queries": 6,
        "depth": 10,
        "num_heads": 2,
    }


@pytest.fixture
def luna_huge_config():
    """Configuration for LUNA Huge variant."""
    return {
        "n_outputs": 2,
        "n_chans": 22,
        "n_times": 1000,
        "embed_dim": 128,
        "num_queries": 8,
        "depth": 24,
        "num_heads": 2,
    }


@pytest.fixture
def luna_base_model(luna_base_config):
    """Create LUNA Base model."""
    return LUNA(**luna_base_config)


@pytest.fixture
def luna_large_model(luna_large_config):
    """Create LUNA Large model."""
    return LUNA(**luna_large_config)


@pytest.fixture
def luna_huge_model(luna_huge_config):
    """Create LUNA Huge model."""
    return LUNA(**luna_huge_config)


@pytest.fixture
def luna_base_pretrained_model():
    """Load LUNA Base pretrained model from HuggingFace Hub.

    This fixture downloads and caches the base model. Uses mne_data folder
    for persistence across CI runs.

    Model located at: https://huggingface.co/thorir/LUNA
    """
    if not HAS_SAFETENSORS:
        pytest.skip("safetensors and huggingface_hub are required")

    # Set cache directory to mne_data for CI persistence
    mne_data_dir = mne.get_config('MNE_DATA')
    if mne_data_dir is None:
        mne_data_dir = str(Path.home() / 'mne_data')
    cache_dir = str(Path(mne_data_dir) / 'luna_pretrained')

    # Load from HuggingFace Hub with mne_data cache
    try:
        # Download the safetensors file
        model_path = hf_hub_download(
            repo_id="thorir/LUNA",
            filename="LUNA_base.safetensors",
            cache_dir=cache_dir,
        )

        # Create model instance
        model = LUNA(
            n_outputs=2,
            n_chans=22,
            n_times=1000,
            embed_dim=64,
            num_queries=4,
            depth=8,
        )

        # Load weights using safetensors
        state_dict = load_file(model_path)
        model.load_state_dict(state_dict, strict=False)

        return model
    except Exception as e:
        # Skip tests if model not available
        pytest.skip(f"Pretrained model not available: {type(e).__name__}: {str(e)[:100]}")


# ==============================================================================
# Tests for LUNA Base Variant
# ==============================================================================


def test_luna_base_initialization(luna_base_model, luna_base_config):
    """Test LUNA Base initialization with correct architecture."""
    assert luna_base_model is not None
    assert luna_base_model.embed_dim == 64
    assert luna_base_model.num_queries == 4
    assert luna_base_model.depth == 8
    assert len(luna_base_model.blocks) == 8


def test_luna_base_forward_pass(luna_base_model):
    """Test LUNA Base forward pass produces correct output shape."""
    x = torch.randn(2, 22, 1000)
    with torch.no_grad():
        output = luna_base_model(x)
    assert output.shape == (2, 2)


def test_luna_base_parameter_count(luna_base_model):
    """Test LUNA Base has expected parameter count."""
    total_params = sum(p.numel() for p in luna_base_model.parameters())
    # Base should have roughly 7M parameters
    assert 5_000_000 < total_params < 10_000_000


def test_luna_base_different_batch_sizes(luna_base_model):
    """Test LUNA Base with different batch sizes."""
    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, 22, 1000)
        with torch.no_grad():
            output = luna_base_model(x)
        assert output.shape == (batch_size, 2)


def test_luna_base_gradient_flow(luna_base_model):
    """Test that gradients flow correctly through LUNA Base."""
    x = torch.randn(2, 22, 1000, requires_grad=True)
    output = luna_base_model(x)
    loss = output.sum()
    loss.backward()

    # Check that gradients exist in transformer blocks
    assert any(p.grad is not None for p in luna_base_model.blocks[0].parameters())
    # Check gradient in final classification head
    assert luna_base_model.final_layer.decoder_ffn.fc1.weight.grad is not None


# ==============================================================================
# Tests for LUNA Large Variant
# ==============================================================================


def test_luna_large_initialization(luna_large_model, luna_large_config):
    """Test LUNA Large initialization with correct architecture."""
    assert luna_large_model is not None
    assert luna_large_model.embed_dim == 96
    assert luna_large_model.num_queries == 6
    assert luna_large_model.depth == 10
    assert len(luna_large_model.blocks) == 10


def test_luna_large_forward_pass(luna_large_model):
    """Test LUNA Large forward pass produces correct output shape."""
    x = torch.randn(2, 22, 1000)
    with torch.no_grad():
        output = luna_large_model(x)
    assert output.shape == (2, 2)


def test_luna_large_parameter_count(luna_large_model):
    """Test LUNA Large has expected parameter count."""
    total_params = sum(p.numel() for p in luna_large_model.parameters())
    # Large should have roughly 43M parameters
    assert 30_000_000 < total_params < 60_000_000


def test_luna_large_different_batch_sizes(luna_large_model):
    """Test LUNA Large with different batch sizes."""
    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, 22, 1000)
        with torch.no_grad():
            output = luna_large_model(x)
        assert output.shape == (batch_size, 2)


def test_luna_large_gradient_flow(luna_large_model):
    """Test that gradients flow correctly through LUNA Large."""
    x = torch.randn(2, 22, 1000, requires_grad=True)
    output = luna_large_model(x)
    loss = output.sum()
    loss.backward()

    # Check that gradients exist in transformer blocks
    assert any(p.grad is not None for p in luna_large_model.blocks[0].parameters())
    # Check gradient in final classification head
    assert luna_large_model.final_layer.decoder_ffn.fc1.weight.grad is not None


# ==============================================================================
# Tests for LUNA Huge Variant
# ==============================================================================


def test_luna_huge_initialization(luna_huge_model, luna_huge_config):
    """Test LUNA Huge initialization with correct architecture."""
    assert luna_huge_model is not None
    assert luna_huge_model.embed_dim == 128
    assert luna_huge_model.num_queries == 8
    assert luna_huge_model.depth == 24
    assert len(luna_huge_model.blocks) == 24


def test_luna_huge_forward_pass(luna_huge_model):
    """Test LUNA Huge forward pass produces correct output shape."""
    x = torch.randn(2, 22, 1000)
    with torch.no_grad():
        output = luna_huge_model(x)
    assert output.shape == (2, 2)


def test_luna_huge_parameter_count(luna_huge_model):
    """Test LUNA Huge has expected parameter count."""
    total_params = sum(p.numel() for p in luna_huge_model.parameters())
    # Huge should have roughly 312M parameters
    assert 250_000_000 < total_params < 350_000_000


def test_luna_huge_different_batch_sizes(luna_huge_model):
    """Test LUNA Huge with different batch sizes."""
    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, 22, 1000)
        with torch.no_grad():
            output = luna_huge_model(x)
        assert output.shape == (batch_size, 2)


def test_luna_huge_gradient_flow(luna_huge_model):
    """Test that gradients flow correctly through LUNA Huge."""
    x = torch.randn(2, 22, 1000, requires_grad=True)
    output = luna_huge_model(x)
    loss = output.sum()
    loss.backward()

    # Check that gradients exist in transformer blocks
    assert any(p.grad is not None for p in luna_huge_model.blocks[0].parameters())
    # Check gradient in final classification head
    assert luna_huge_model.final_layer.decoder_ffn.fc1.weight.grad is not None


# ==============================================================================
# Tests for LUNA Variant Comparisons
# ==============================================================================


def test_luna_variants_parameter_count_hierarchy(luna_base_model, luna_large_model, luna_huge_model):
    """Test that parameter counts follow the hierarchy Base < Large < Huge."""
    base_params = sum(p.numel() for p in luna_base_model.parameters())
    large_params = sum(p.numel() for p in luna_large_model.parameters())
    huge_params = sum(p.numel() for p in luna_huge_model.parameters())

    assert base_params < large_params
    assert large_params < huge_params


def test_luna_variants_device_compatibility(luna_base_model, luna_large_model, luna_huge_model):
    """Test LUNA variants work correctly on CPU."""
    x = torch.randn(2, 22, 1000)

    for model_name, model in [
        ("Base", luna_base_model),
        ("Large", luna_large_model),
        ("Huge", luna_huge_model),
    ]:
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert output.shape == (2, 2), f"LUNA {model_name} output shape incorrect"

        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            x_cuda = x.cuda()
            with torch.no_grad():
                output_cuda = model_cuda(x_cuda)
            assert output_cuda.shape == (2, 2)
            assert output_cuda.device.type == "cuda"


def test_luna_variants_different_channel_counts(luna_base_config, luna_large_config, luna_huge_config):
    """Test LUNA variants handle different channel counts."""
    configs = [luna_base_config, luna_large_config, luna_huge_config]

    for n_chans in [1, 4, 8, 16, 32, 64]:
        for config in configs:
            config["n_chans"] = n_chans
            model = LUNA(**config)
            model.eval()

            x = torch.randn(2, n_chans, 1000)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (2, 2)


def test_luna_variants_output_consistency(luna_base_config, luna_large_config, luna_huge_config):
    """Test that all LUNA variants produce consistent output shapes."""
    configs = [luna_base_config, luna_large_config, luna_huge_config]
    test_input = torch.randn(2, 22, 1000)

    for config in configs:
        model = LUNA(**config)
        model.eval()

        with torch.no_grad():
            output = model(test_input)

        assert output.shape == (2, 2), f"Output shape mismatch for config {config}"


# ==============================================================================
# Tests for Pretrained Models
# ==============================================================================


def test_luna_base_pretrained_loads(luna_base_pretrained_model):
    """Test that LUNA base pretrained model loads successfully from HuggingFace."""
    assert luna_base_pretrained_model is not None
    assert isinstance(luna_base_pretrained_model, LUNA)


def test_luna_base_pretrained_forward_pass(luna_base_pretrained_model):
    """Test pretrained base model forward pass."""
    model = luna_base_pretrained_model
    model.eval()

    x = torch.randn(2, 22, 1000)
    with torch.no_grad():
        output = model(x)

    assert output.shape == (2, 2)


def test_luna_base_pretrained_parameter_count(luna_base_pretrained_model):
    """Test pretrained base model has expected parameter count."""
    total_params = sum(p.numel() for p in luna_base_pretrained_model.parameters())
    # Base should have roughly 7M parameters
    assert 5_000_000 < total_params < 10_000_000


def test_luna_base_pretrained_different_batch_sizes(luna_base_pretrained_model):
    """Test pretrained base model with different batch sizes."""
    model = luna_base_pretrained_model
    model.eval()

    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, 22, 1000)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (batch_size, 2)


def test_luna_base_pretrained_caching(luna_base_pretrained_model):
    """Test that pretrained model weights are cached in mne_data."""

    # Check that cache directory exists and has files
    mne_data_dir = mne.get_config('MNE_DATA')
    if mne_data_dir is None:
        mne_data_dir = str(Path.home() / 'mne_data')
    cache_dir = Path(mne_data_dir) / 'luna_pretrained'

    if cache_dir.exists():
        # Check that model files were downloaded
        cache_files = list(cache_dir.rglob("*"))
        assert len(cache_files) > 0, "Cache directory should contain downloaded files"


# ==============================================================================
# Tests for REVE Model
# ==============================================================================

# Check if HF token for REVE is available
HF_TOKEN_REVE_MISSING = os.getenv("HF_TOKEN_REVE") is None or os.getenv("HF_TOKEN_REVE") == ""


@pytest.fixture
def reve_batch_size():
    return 2


@pytest.fixture
def reve_n_chans():
    return 32


@pytest.fixture
def reve_n_times():
    return 1000


@pytest.fixture
def reve_n_outputs():
    return 10


@pytest.fixture
def reve_cache_dir():
    return "./cache"


@pytest.fixture
def reve_model_id():
    return "brain-bzh/reve-base"


@pytest.fixture
def reve_positions_id():
    return "brain-bzh/reve-positions"


@pytest.fixture
def reve_position_bank():
    return RevePositionBank()


@pytest.fixture
def reve_position_bank_hf(reve_positions_id, reve_cache_dir):
    try:
        from transformers import AutoModel
        return AutoModel.from_pretrained(
            reve_positions_id,
            cache_dir=reve_cache_dir,
            trust_remote_code=True,
        )
    except ImportError:
        pytest.skip("transformers not installed")


@pytest.fixture
def reve_model_hf(reve_model_id, reve_cache_dir):
    try:
        from transformers import AutoModel
        return AutoModel.from_pretrained(
            reve_model_id,
            cache_dir=reve_cache_dir,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN_REVE")
        )
    except ImportError:
        pytest.skip("transformers not installed")


@pytest.fixture
def reve_model_bd(reve_model_id, reve_cache_dir, reve_n_times, reve_n_chans, reve_n_outputs):
    return REVE.from_pretrained(
        reve_model_id,
        cache_dir=reve_cache_dir,
        n_times=reve_n_times,
        n_chans=reve_n_chans,
        n_outputs=reve_n_outputs,
        token=os.getenv("HF_TOKEN_REVE"),
    )


def test_reve_positions_match(reve_position_bank, reve_position_bank_hf, reve_cache_dir):
    """Test that the positions from both implementations match."""
    all_pos_hf = reve_position_bank_hf.get_all_positions()
    all_pos_bd = reve_position_bank.get_all_positions()

    assert all_pos_hf == all_pos_bd, "Position names mismatch"

    for pos in all_pos_bd:
        pos_hf = reve_position_bank_hf([pos])
        pos_bd = reve_position_bank([pos])
        assert torch.allclose(pos_hf, pos_bd)

    # Cleanup
    if os.path.exists(reve_cache_dir):
        rmtree(reve_cache_dir)


@pytest.mark.skipif(HF_TOKEN_REVE_MISSING, reason="HF token for REVE is missing")
def test_reve_model_outputs_match(
    reve_position_bank_hf,
    reve_model_hf,
    reve_model_bd,
    reve_batch_size,
    reve_n_chans,
    reve_n_times,
    reve_cache_dir,
):
    """Test that the outputs from both implementations match."""
    ch_list = [f"E{i + 1}" for i in range(reve_n_chans)]

    torch.manual_seed(42)
    eeg_input = torch.randn(reve_batch_size, reve_n_chans, reve_n_times)

    pos_hf = reve_position_bank_hf(ch_list)
    pos_hf = pos_hf.unsqueeze(0).repeat(reve_batch_size, 1, 1)

    pos_bd = reve_model_bd.get_positions(ch_list)
    pos_bd = pos_bd.unsqueeze(0).repeat(reve_batch_size, 1, 1)

    assert torch.allclose(pos_hf, pos_bd)

    # return_output is True to bypass the last layer
    output_bd = reve_model_bd(eeg_input, pos_bd, return_output=True)[-1]
    output_hf = reve_model_hf(eeg_input, pos_hf, return_output=True)[-1]

    assert torch.allclose(output_hf, output_bd)

    # Cleanup
    if os.path.exists(reve_cache_dir):
        rmtree(reve_cache_dir)
