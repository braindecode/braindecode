# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3

import hashlib
import json
import os
from pathlib import Path
from urllib.error import URLError

import mne
import numpy as np
import pooch
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

try:
    from huggingface_hub import hf_hub_download

    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

try:
    from safetensors.torch import load_file

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

from braindecode.models import (
    LUNA,
    REVE,
    BrainOmni,
    BrainTokenizer,
    CBraMod,
    CodeBrain,
    Labram,
)
from braindecode.models.base import EEGModuleMixin
from braindecode.models.brainomni import (
    _SEANetDecoder,
    _SEANetEncoder,
    _SensorEmbedding,
    _SpatialTemporalBlock,
    _TokenizerEncoder,
)
from braindecode.models.labram import LABRAM_CHANNEL_ORDER
from braindecode.models.reve import RevePositionBank
from braindecode.models.util import _geometry_from_chs_info
from braindecode.modules import Codebook, ResidualVQ


@pytest.fixture
def n_times():
    return 1000


@pytest.fixture
def n_chans():
    return 128


@pytest.fixture
def chs_info():
    return [{"ch_name": ch_name} for ch_name in LABRAM_CHANNEL_ORDER]


@pytest.fixture
def ch_names(chs_info):
    return [ch["ch_name"] for ch in chs_info]


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
def model_config_tokenizer(
    n_times, n_chans, chs_info, n_outputs, patch_size, emb_size, n_layers, num_heads
):
    return {
        "n_times": n_times,
        "n_chans": n_chans,
        "chs_info": chs_info,
        "n_outputs": n_outputs,
        "patch_size": patch_size,
        "embed_dim": emb_size,
        "num_layers": n_layers,
        "num_heads": num_heads,
        "neural_tokenizer": True,
    }


@pytest.fixture
def model_config_decoder(
    n_times, n_chans, chs_info, n_outputs, patch_size, emb_size, n_layers, num_heads
):
    return {
        "n_times": n_times,
        "n_chans": n_chans,
        "chs_info": chs_info,
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
    assert model_tokenizer.n_chans == 128
    assert model_tokenizer.n_times == 1000
    assert model_tokenizer.n_outputs == 4


def test_labram_neural_tokenizer_forward_pass_basic(
    model_tokenizer, batch_size, n_chans, n_times, n_outputs
):
    """Test basic forward pass in tokenizer mode."""
    x = torch.randn(batch_size, n_chans, n_times)
    output = model_tokenizer(x)
    assert output.shape == (batch_size, n_outputs)


def test_labram_neural_tokenizer_forward_pass_single_sample(
    model_tokenizer, n_chans, n_times, n_outputs
):
    """Test forward pass with single sample in tokenizer mode."""
    x = torch.randn(1, n_chans, n_times)
    output = model_tokenizer(x)
    assert output.shape == (1, n_outputs)


def test_labram_neural_tokenizer_different_batch_sizes(
    model_tokenizer, n_chans, n_times, n_outputs
):
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
    assert model_decoder.n_chans == 128
    assert model_decoder.n_times == 1000
    assert model_decoder.n_outputs == 4


def test_labram_neural_decoder_forward_pass_basic(
    model_decoder, batch_size, n_chans, n_times, n_outputs
):
    """Test basic forward pass in decoder mode."""
    x = torch.randn(batch_size, n_chans, n_times)
    output = model_decoder(x)
    assert output.shape == (batch_size, n_outputs)


def test_labram_neural_decoder_forward_pass_single_sample(
    model_decoder, n_chans, n_times, n_outputs
):
    """Test forward pass with single sample in decoder mode."""
    x = torch.randn(1, n_chans, n_times)
    output = model_decoder(x)
    assert output.shape == (1, n_outputs)


@pytest.mark.network
@pytest.mark.huggingface
def test_labram_can_load_pretrained_weights():
    """Ensure that Labram can load pre-trained weights from HuggingFace Hub."""
    mne_data_dir = mne.get_config("MNE_DATA")
    if mne_data_dir is None:
        mne_data_dir = str(Path.home() / "mne_data")
    cache_dir = Path(mne_data_dir) / "labram_pretrained"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = str(cache_dir)

    try:
        model = Labram.from_pretrained(
            "braindecode/labram-pretrained",
            cache_dir=cache_dir,
        )
    except (URLError, OSError) as err:
        pytest.skip(f"Could not download pretrained Labram checkpoint: {err}")

    # Verify model was loaded and can run a forward pass
    x = torch.randn(1, model.n_chans, model.n_times)
    output = model(x)
    assert output.shape[0] == 1


def test_labram_neural_decoder_different_batch_sizes(
    model_decoder, n_chans, n_times, n_outputs
):
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


def test_labram_output_shapes_consistency_between_modes(n_times, chs_info, n_outputs):
    """Ensure that both modes produce compatible outputs."""
    batch_size = 2

    model_tokenizer = Labram(
        n_times=n_times,
        chs_info=chs_info,
        n_outputs=n_outputs,
        neural_tokenizer=True,
    )

    model_decoder = Labram(
        n_times=n_times,
        chs_info=chs_info,
        n_outputs=n_outputs,
        neural_tokenizer=False,
    )

    x = torch.randn(batch_size, len(chs_info), n_times)

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


# ==============================================================================
# Tests for Edge Cases
# ==============================================================================


def test_labram_small_input_size(chs_info):
    """Test with small input size."""
    model = Labram(
        n_times=400,
        chs_info=chs_info,
        n_outputs=4,
        patch_size=200,
        neural_tokenizer=True,
    )

    x = torch.randn(2, len(chs_info), 400)
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

    with pytest.raises((RuntimeError, ValueError, IndexError)):
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
    except (RuntimeError, IndexError, ValueError):
        # Expected behavior
        pass


# ==============================================================================
# Tests for Labram Channel Reordering
# ==============================================================================


def test_labram_channel_order_constant_exported():
    """Test that LABRAM_CHANNEL_ORDER is exported and has expected format."""
    assert LABRAM_CHANNEL_ORDER is not None
    assert isinstance(LABRAM_CHANNEL_ORDER, (list, tuple))
    assert len(LABRAM_CHANNEL_ORDER) > 100  # Should have 100+ channels
    assert "FP1" in LABRAM_CHANNEL_ORDER
    assert "CZ" in LABRAM_CHANNEL_ORDER
    assert "O2" in LABRAM_CHANNEL_ORDER


# ==============================================================================
# Tests for Labram.forward(ch_names=...) subset / case / error paths
# ==============================================================================


def _small_labram_for_ch_names(chs_info, n_outputs):
    """Build a tiny tokenizer-mode Labram on the full canonical bank."""
    return Labram(
        n_times=400,
        chs_info=chs_info,
        n_outputs=n_outputs,
        patch_size=200,
        embed_dim=64,
        num_layers=1,
        num_heads=4,
        neural_tokenizer=True,
    )


def test_labram_forward_with_ch_names_subset(chs_info, n_outputs):
    """Forward an arbitrary subset of canonical channels via ch_names."""
    model = _small_labram_for_ch_names(chs_info, n_outputs)
    model.eval()

    # Pick 8 canonical channels in non-canonical order
    subset = [LABRAM_CHANNEL_ORDER[i] for i in (10, 0, 30, 5, 60, 15, 90, 20)]
    x = torch.randn(2, len(subset), 400)

    with torch.no_grad():
        out = model(x, ch_names=subset)

    assert out.shape == (2, n_outputs)


def test_labram_forward_ch_names_is_case_insensitive(chs_info, n_outputs):
    """Mixed-case ch_names should match LABRAM_CHANNEL_ORDER case-insensitively."""
    model = _small_labram_for_ch_names(chs_info, n_outputs)
    model.eval()

    upper = [LABRAM_CHANNEL_ORDER[i] for i in (0, 10, 20)]
    mixed = [name.title() for name in upper]  # e.g. "Fp1", "Fpz", ...
    x = torch.randn(1, len(mixed), 400)

    with torch.no_grad():
        out_upper = model(x, ch_names=upper)
        out_mixed = model(x, ch_names=mixed)

    # Same channels under either casing -> identical outputs.
    assert torch.allclose(out_upper, out_mixed)


def test_labram_forward_ch_names_unknown_channel_raises(chs_info, n_outputs):
    """Unknown channel names should produce a clear ValueError."""
    model = _small_labram_for_ch_names(chs_info, n_outputs)
    bad_names = [LABRAM_CHANNEL_ORDER[0], "NOT_A_REAL_CHANNEL"]
    x = torch.randn(1, len(bad_names), 400)

    with pytest.raises(ValueError, match="LABRAM_CHANNEL_ORDER"):
        model(x, ch_names=bad_names)


def test_labram_forward_ch_names_length_mismatch_raises(chs_info, n_outputs):
    """len(ch_names) must equal x.shape[1]."""
    model = _small_labram_for_ch_names(chs_info, n_outputs)
    names = [LABRAM_CHANNEL_ORDER[i] for i in (0, 1, 2)]
    x = torch.randn(1, 4, 400)  # 4 channels, 3 names

    with pytest.raises(ValueError, match="len.ch_names"):
        model(x, ch_names=names)


def test_labram_forward_none_ch_names_wrong_count_raises(chs_info, n_outputs):
    """ch_names=None with a non-canonical channel count must raise early."""
    model = _small_labram_for_ch_names(chs_info, n_outputs)
    x = torch.randn(1, 22, 400)  # not 128

    with pytest.raises(ValueError, match="ch_names is None"):
        model(x)


def test_labram_forward_return_flags_remain_positional(
    chs_info, n_outputs, n_chans
):
    """Back-compat: return_* flags can still be passed positionally."""
    model = _small_labram_for_ch_names(chs_info, n_outputs)
    model.eval()
    x = torch.randn(1, n_chans, 400)

    with torch.no_grad():
        out_default = model(x)
        # Positional: return_patch_tokens=False, return_all_tokens=True.
        # ch_names is keyword-only, so this triggers the all-tokens path
        # without forcing callers to switch to kwargs for the return flags.
        out_all = model(x, False, True)

    assert out_default.shape == (1, n_outputs)
    # all_tokens returns one token per CLS + (n_chans * n_patches) patch
    # tokens; only the trailing dim has to equal n_outputs.
    assert out_all.dim() == 3
    assert out_all.shape[0] == 1
    assert out_all.shape[-1] == n_outputs
    assert out_all.shape[1] > 1  # more than just the CLS token


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

    Available variants:
    - LUNA_base.safetensors (embed_dim=64, num_queries=4, depth=8)
    - LUNA_large.safetensors (embed_dim=96, num_queries=6, depth=10)
    - LUNA_huge.safetensors (embed_dim=128, num_queries=8, depth=24)
    """
    if not HAS_SAFETENSORS:
        pytest.skip("safetensors and huggingface_hub are required")

    # Set cache directory to mne_data for CI persistence
    mne_data_dir = mne.get_config("MNE_DATA")
    if mne_data_dir is None:
        mne_data_dir = str(Path.home() / "mne_data")
    cache_dir = str(Path(mne_data_dir) / "luna_pretrained")

    # Load from HuggingFace Hub with mne_data cache
    try:
        # Download the safetensors file
        model_path = hf_hub_download(
            repo_id="thorir/LUNA",
            filename="LUNA_base.safetensors",
            cache_dir=cache_dir,
        )

        # Create model instance for classification (fine-tuning)
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
        # load_state_dict applies model.mapping automatically
        model.load_state_dict(state_dict, strict=False)

        return model
    except Exception as e:
        # Skip tests if model not available
        pytest.skip(
            f"Pretrained model not available: {type(e).__name__}: {str(e)[:100]}"
        )


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


def test_luna_channel_embed_batch_ordering(luna_base_config):
    # channel embeddings should be consistent within each batch element
    luna_base_config["n_chans"] = 3
    luna_base_config["n_times"] = 80
    luna_base_config["patch_size"] = 20
    model = LUNA(**luna_base_config)
    model.eval()

    B, C, num_patches = 2, 3, 4
    channel_locations = torch.zeros(B, C, 3)
    for c in range(C):
        channel_locations[0, c, 0] = c / (C - 1)
        channel_locations[1, c, 1] = c / (C - 1)

    x_signal = torch.randn(B, C, 80)
    with torch.no_grad():
        x_tok, ch_emb = model.prepare_tokens(x_signal, channel_locations, mask=None)

    # each batch's patches should have identical channel embeddings
    b0 = ch_emb[:num_patches, 1, :]
    b1 = ch_emb[num_patches:, 1, :]
    for i in range(1, num_patches):
        assert torch.allclose(b0[0], b0[i], atol=1e-5)
    for i in range(1, num_patches):
        assert torch.allclose(b1[0], b1[i], atol=1e-5)

    # embeddings for different batches (with different channel_locations)
    # should not be identical
    assert not torch.allclose(b0[0], b1[0], atol=1e-5)


def test_luna_mapping_includes_temperature_typo():
    # pretrained weights have typo key, mapping should handle it
    model = LUNA(n_outputs=2, n_chans=22, n_times=1000, embed_dim=64,
                 num_queries=4, depth=8)
    assert "cross_attn.temparature" in model.mapping


def test_luna_variants_parameter_count_hierarchy(
    luna_base_model, luna_large_model, luna_huge_model
):
    """Test that parameter counts follow the hierarchy Base < Large < Huge."""
    base_params = sum(p.numel() for p in luna_base_model.parameters())
    large_params = sum(p.numel() for p in luna_large_model.parameters())
    huge_params = sum(p.numel() for p in luna_huge_model.parameters())

    assert base_params < large_params
    assert large_params < huge_params


def test_luna_variants_device_compatibility(
    luna_base_model, luna_large_model, luna_huge_model
):
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


def test_luna_variants_different_channel_counts(
    luna_base_config, luna_large_config, luna_huge_config
):
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


def test_luna_variants_output_consistency(
    luna_base_config, luna_large_config, luna_huge_config
):
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


@pytest.mark.network
@pytest.mark.huggingface
def test_luna_base_pretrained_loads(luna_base_pretrained_model):
    """Test that LUNA base pretrained model loads successfully from HuggingFace."""
    assert luna_base_pretrained_model is not None
    assert isinstance(luna_base_pretrained_model, LUNA)


@pytest.mark.network
@pytest.mark.huggingface
def test_luna_base_pretrained_forward_pass(luna_base_pretrained_model):
    """Test pretrained base model forward pass."""
    model = luna_base_pretrained_model
    model.eval()

    x = torch.randn(2, 22, 1000)
    with torch.no_grad():
        output = model(x)

    assert output.shape == (2, 2)


@pytest.mark.network
@pytest.mark.huggingface
def test_luna_base_pretrained_parameter_count(luna_base_pretrained_model):
    """Test pretrained base model has expected parameter count."""
    total_params = sum(p.numel() for p in luna_base_pretrained_model.parameters())
    # Base should have roughly 7M parameters
    assert 5_000_000 < total_params < 10_000_000


@pytest.mark.network
@pytest.mark.huggingface
def test_luna_base_pretrained_different_batch_sizes(luna_base_pretrained_model):
    """Test pretrained base model with different batch sizes."""
    model = luna_base_pretrained_model
    model.eval()

    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, 22, 1000)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (batch_size, 2)


@pytest.mark.network
@pytest.mark.huggingface
def test_luna_base_pretrained_caching(luna_base_pretrained_model):
    """Test that pretrained model weights are cached in mne_data."""

    # Check that cache directory exists and has files
    mne_data_dir = mne.get_config("MNE_DATA")
    if mne_data_dir is None:
        mne_data_dir = str(Path.home() / "mne_data")
    cache_dir = Path(mne_data_dir) / "luna_pretrained"

    if cache_dir.exists():
        # Check that model files were downloaded
        cache_files = list(cache_dir.rglob("*"))
        assert len(cache_files) > 0, "Cache directory should contain downloaded files"


# ==============================================================================
# Tests for REVE Model
# ==============================================================================

# Check if HF token for REVE is available
HF_TOKEN_REVE_MISSING = (
    os.getenv("HF_TOKEN_REVE") is None or os.getenv("HF_TOKEN_REVE") == ""
)

# REVE test constants
REVE_BATCH_SIZE = 2
REVE_N_CHANS = 32
REVE_N_TIMES = 1000
REVE_N_OUTPUTS = 10
REVE_MODEL_ID = "brain-bzh/reve-base"
REVE_POSITIONS_ID = "brain-bzh/reve-positions"


def _get_reve_cache_dir():
    """Get cache directory for REVE pretrained models."""
    mne_data_dir = mne.get_config("MNE_DATA")
    if mne_data_dir is None:
        mne_data_dir = str(Path.home() / "mne_data")
    return str(Path(mne_data_dir) / "reve_pretrained")


@pytest.mark.network
@pytest.mark.huggingface
def test_reve_positions_match():
    """Test that the positions from both implementations match."""
    pytest.skip(
        "TODO: Fix me. The test is broken on the CI but works locally (even after erasing the cache dir)."
    )
    try:
        from transformers import AutoModel
    except ImportError:
        pytest.skip("transformers not installed")

    cache_dir = _get_reve_cache_dir()
    pos_bank_hf = AutoModel.from_pretrained(
        REVE_POSITIONS_ID,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    pos_bank_bd = RevePositionBank()

    all_pos_hf = pos_bank_hf.get_all_positions()
    all_pos_bd = pos_bank_bd.get_all_positions()

    assert all_pos_hf == all_pos_bd, "Position names mismatch"

    for pos in all_pos_bd:
        pos_hf = pos_bank_hf([pos])
        pos_bd = pos_bank_bd([pos])
        assert torch.allclose(pos_hf, pos_bd)


@pytest.mark.skipif(HF_TOKEN_REVE_MISSING, reason="HF token for REVE is missing")
@pytest.mark.network
@pytest.mark.huggingface
def test_reve_model_outputs_match():
    """Test that the outputs from both implementations match."""
    try:
        from transformers import AutoModel
    except ImportError:
        pytest.skip("transformers not installed")

    try:
        import flash_attn  # noqa: F401
    except ImportError:
        pytest.skip("flash_attn not installed - outputs differ without it")

    cache_dir = _get_reve_cache_dir()

    # Load HuggingFace models
    pos_bank_hf = AutoModel.from_pretrained(
        REVE_POSITIONS_ID,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    model_hf = AutoModel.from_pretrained(
        REVE_MODEL_ID,
        cache_dir=cache_dir,
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN_REVE"),
    )

    # Load Braindecode model
    model_bd = REVE.from_pretrained(
        REVE_MODEL_ID,
        cache_dir=cache_dir,
        n_times=REVE_N_TIMES,
        n_chans=REVE_N_CHANS,
        n_outputs=REVE_N_OUTPUTS,
        token=os.getenv("HF_TOKEN_REVE"),
    )

    ch_list = [f"E{i + 1}" for i in range(REVE_N_CHANS)]

    torch.manual_seed(42)
    eeg_input = torch.randn(REVE_BATCH_SIZE, REVE_N_CHANS, REVE_N_TIMES)

    pos_hf = pos_bank_hf(ch_list)
    pos_hf = pos_hf.unsqueeze(0).repeat(REVE_BATCH_SIZE, 1, 1)

    pos_bd = model_bd.get_positions(ch_list)
    pos_bd = pos_bd.unsqueeze(0).repeat(REVE_BATCH_SIZE, 1, 1)

    assert torch.allclose(pos_hf, pos_bd)

    # return_output is True to bypass the last layer
    output_bd = model_bd(eeg_input, pos_bd, return_output=True)[-1]
    output_hf = model_hf(eeg_input, pos_hf, return_output=True)[-1]

    assert torch.allclose(output_hf, output_bd)


# ==============================================================================
# Offline robustness of the REVE position bank (no network required)
# ==============================================================================


def test_reve_position_bank_uses_prefetched_file(tmp_path, monkeypatch):
    """A prefetched positions file is used offline, without any download."""
    config = {"Cz": [0.0, 0.0, 1.0], "Pz": [0.0, -0.5, 0.5]}
    (tmp_path / "reve_positions.json").write_text(json.dumps(config))
    monkeypatch.setattr(
        pooch, "retrieve", lambda *a, **k: pytest.fail("unexpected download")
    )

    bank = RevePositionBank(cache_dir=str(tmp_path))

    assert bank.get_all_positions() == list(config.keys())
    assert bank.forward(["Cz", "Pz"]).shape == (2, 3)


def test_reve_position_bank_download_failure_raises(tmp_path, monkeypatch):
    """On a cache miss, a download failure points the user at offline prefetch."""

    def _fail(*args, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr(pooch, "retrieve", _fail)

    with pytest.raises(RuntimeError, match="prefetch it to"):
        RevePositionBank(cache_dir=str(tmp_path))


def test_reve_position_bank_corrupt_cache_redownloads(tmp_path, monkeypatch):
    """A corrupt/partial cached file triggers a re-download instead of crashing."""
    cache_file = tmp_path / "reve_positions.json"
    cache_file.write_text("{ this is not valid json")
    config = {"Cz": [0.0, 0.0, 1.0]}

    def _fake_retrieve(url, known_hash, fname, path, **kwargs):
        (tmp_path / fname).write_text(json.dumps(config))

    monkeypatch.setattr(pooch, "retrieve", _fake_retrieve)

    bank = RevePositionBank(cache_dir=str(tmp_path))

    assert bank.get_all_positions() == list(config.keys())


# ==============================================================================
# Tests for CBraMod Model
# ==============================================================================


@pytest.mark.network
@pytest.mark.huggingface
def test_cbramod_load_weights():
    model = CBraMod(return_encoder_output=True)
    state_dict = torch.hub.load_state_dict_from_url(
        "https://huggingface.co/braindecode/cbramod-pretrained/resolve/main/pytorch_model.bin",
        map_location="cpu",
    )
    load_result = model.load_state_dict(state_dict)
    assert not load_result.missing_keys
    assert not load_result.unexpected_keys


def test_cbramod_forward_pass():
    model = CBraMod(return_encoder_output=True)
    x = torch.randn(2, 22, 1000)
    output = model(x)
    assert output.shape == (2, 22, 5, 200)


# ==============================================================================
# Tests for CodeBrain Model
# ==============================================================================



def test_codebrain_forward_pass():
    model = CodeBrain(n_chans=19, n_outputs=2, n_times=6000)
    x = torch.randn(2, 19, 6000)
    output = model(x)
    assert output.shape == (2, 2)


def test_codebrain_pretrain_mode():
    model = CodeBrain(n_chans=19, n_outputs=2, n_times=6000, pretrain_mode=True)
    x = torch.randn(2, 19, 6000)
    x_t, x_f = model(x)
    # seq_len = 6000 // 200 = 30, output shape: (batch, n_chans, seq_len, codebook_size)
    assert x_t.shape == (2, 19, 30, 4096)
    assert x_f.shape == (2, 19, 30, 4096)


def test_codebrain_return_features():
    model = CodeBrain(n_chans=19, n_outputs=2, n_times=6000)
    x = torch.randn(2, 19, 6000)
    out = model(x, return_features=True)
    assert isinstance(out, dict)
    assert "features" in out
    assert "cls_token" in out
    # features shape: (batch, n_chans, seq_len, out_channels)
    assert out["features"].shape == (2, 19, 30, 200)
    assert out["cls_token"] is None


# ==============================================================================
# Tests for BrainOmni / BrainTokenizer (unified EEG/MEG foundation model)
# ==============================================================================


# Shared small-model config (keeps every BrainOmni/BrainTokenizer build fast).
_BRAINOMNI_KW = dict(
    emb_dim=16,
    n_neuro=3,
    n_filters=8,
    codebook_dim=16,
    codebook_size=32,
    num_quantizers=2,
    tokenizer_num_heads=4,
)


def _loc(x=0.0, y=0.0, z=0.0, *rest):
    arr = np.zeros(12, dtype=np.float64)
    arr[:3] = (x, y, z)
    for i, v in enumerate(rest):
        arr[3 + i] = v
    return arr


def _eeg_chs_info(n):
    rng = np.random.default_rng(0)
    return [
        {"ch_name": f"C{i}", "kind": "eeg", "loc": _loc(*rng.random(3))}
        for i in range(n)
    ]


def _mixed_chs_info():
    rng = np.random.default_rng(1)
    return [
        {"ch_name": "E1", "kind": "eeg", "loc": _loc(*rng.random(3))},
        {"ch_name": "E2", "kind": "eeg", "loc": _loc(*rng.random(3))},
        {"ch_name": "M1", "kind": "mag", "coil_type": 3022, "loc": _loc(*rng.random(6))},
        {"ch_name": "G1", "kind": "grad", "coil_type": 3012, "loc": _loc(*rng.random(6))},
    ]


def _small_tokenizer(n_chans=4, n_times=512):
    return BrainTokenizer(
        chs_info=_eeg_chs_info(n_chans), n_times=n_times, sfreq=256.0, **_BRAINOMNI_KW
    )


def _small_brainomni(n_chans=4, n_outputs=3, n_times=512, sfreq=256.0, chs_info=None):
    return BrainOmni(
        chs_info=chs_info if chs_info is not None else _eeg_chs_info(n_chans),
        n_outputs=n_outputs,
        n_times=n_times,
        sfreq=sfreq,
        lm_dim=16,
        num_heads=4,
        depth=2,
        **_BRAINOMNI_KW,
    )


def _quantizer(num_quantizers=2):
    return ResidualVQ(
        dim=16,
        codebook_dim=16,
        codebook_size=32,
        num_quantizers=num_quantizers,
        rotation_trick=True,
        quantize_optimize_method="ema",
    )


def _distributed_codebook_update(rank, world_size, init_file, output_dir):
    """Exercise one EMA update with different data on each CPU rank."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        codebook = Codebook(
            dim=2,
            codebook_size=2,
            decay=0.5,
            threshold_ema_dead_code=0,
        )
        codebook.inited.fill_(1)
        codebook.cluster_size.copy_(torch.tensor([4.0, 4.0]))
        codebook.embed.copy_(torch.tensor([[-1.0, 0.0], [1.0, 0.0]]))
        codebook.embed_avg.copy_(torch.tensor([[-4.0, 0.0], [4.0, 0.0]]))
        local_samples = (
            torch.tensor([[[-2.0, 0.0], [-1.0, 0.0]]])
            if rank == 0
            else torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])
        )
        codebook.train()(local_samples)
        torch.manual_seed(rank)
        fresh_codebook = Codebook(
            dim=2,
            codebook_size=2,
            decay=0.5,
            threshold_ema_dead_code=0,
            kmeans_iters=2,
        )
        fresh_codebook.train()(local_samples)
        torch.save(
            {
                "cluster_size": codebook.cluster_size,
                "embed_avg": codebook.embed_avg,
                "embed": codebook.embed,
                "fresh_inited": fresh_codebook.inited,
                "fresh_cluster_size": fresh_codebook.cluster_size,
                "fresh_embed_avg": fresh_codebook.embed_avg,
                "fresh_embed": fresh_codebook.embed,
            },
            Path(output_dir) / f"rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


# ---- geometry derivation -----------------------------------------------------


def _real_chs(ch_names, ch_types):
    """Build MNE channel dicts with a minimal finite loc (identity rotation)."""
    finite_loc = np.array([0.1, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    info = mne.create_info(ch_names, 256.0, ch_types)
    for ch in info["chs"]:
        ch["loc"] = finite_loc.copy()
    return info["chs"]


@pytest.mark.parametrize(
    "chs_info, expected",
    [
        ([{"ch_name": "C1", "kind": "eeg", "loc": _loc(1.0)}], [0]),  # simplified EEG dict
        (_real_chs(["MEG0111"], ["grad"]), [2]),  # real GRAD
        (_real_chs(["MEG0112"], ["mag"]), [1]),   # real MAG
        (_real_chs(["E1"], ["eeg"]), [0]),         # real EEG
    ],
    ids=["eeg_simplified", "grad", "mag", "eeg_real"],
)
def test_geometry_sensor_type(chs_info, expected):
    _, sensor_type = _geometry_from_chs_info(chs_info)
    assert sensor_type.tolist() == expected


def test_geometry_eeg_orientation_and_centering():
    pos, _ = _geometry_from_chs_info(
        [
            {"ch_name": "A", "kind": "eeg", "loc": _loc(1.0, 0.0, 0.0)},
            {"ch_name": "B", "kind": "eeg", "loc": _loc(-1.0, 0.0, 0.0)},
        ]
    )
    assert pos.shape == (2, 6)
    assert np.allclose(pos[:, 3:], 0.0)  # EEG orientation columns are zero
    assert np.allclose(pos[:, :3].mean(axis=0), 0.0, atol=1e-6)  # mean-centered


@pytest.mark.parametrize(
    "kind, coil_type, orientation_slice",
    [
        ("grad", 3012, slice(3, 6)),  # planar gradiometer
        ("grad", 5001, slice(9, 12)),  # axial gradiometer
        ("mag", 3022, slice(9, 12)),
    ],
)
def test_geometry_meg_orientation_uses_mne_loc_axes(
    kind, coil_type, orientation_slice
):
    loc = _loc(0.1, 0.2, 0.3)
    loc[orientation_slice] = (0.4, 0.5, 0.6)
    pos, _ = _geometry_from_chs_info(
        [{"ch_name": "M1", "kind": kind, "coil_type": coil_type, "loc": loc}]
    )
    assert np.allclose(pos[0, 3:], (0.4, 0.5, 0.6))


@pytest.mark.parametrize(
    "kind, coil_type, orientation_slice",
    [
        ("grad", 3012, slice(3, 6)),
        ("grad", 5001, slice(9, 12)),
        ("mag", 3022, slice(9, 12)),
    ],
)
def test_geometry_nonfinite_meg_orientation_raises(
    kind, coil_type, orientation_slice
):
    loc = _loc(0.1, 0.2, 0.3)
    loc[orientation_slice] = np.nan
    with pytest.raises(ValueError, match="finite coil orientation"):
        _geometry_from_chs_info(
            [
                {
                    "ch_name": "M1",
                    "kind": kind,
                    "coil_type": coil_type,
                    "loc": loc,
                }
            ]
        )


@pytest.mark.parametrize("loc", [np.full(12, np.nan), None], ids=["nan", "absent"])
def test_geometry_bad_loc_raises(loc):
    ch = {"ch_name": "A", "kind": "eeg"}
    if loc is not None:
        ch["loc"] = loc
    with pytest.raises(ValueError, match="set_montage"):
        _geometry_from_chs_info([ch])


# ---- submodule shape contracts -----------------------------------------------


@pytest.mark.parametrize(
    "build, make_inputs, exp_shape",
    [
        (lambda: nn.RMSNorm(8, eps=1e-6), lambda: (torch.randn(2, 5, 8),), (2, 5, 8)),
        (
            lambda: _SpatialTemporalBlock(16, 4, 0.0, causal=False),
            lambda: (torch.randn(2, 3, 7, 16),),  # (batch, chans, tokens, dim)
            (2, 3, 7, 16),
        ),
        (
            lambda: _SensorEmbedding(n_dim=16),
            lambda: (torch.randn(2, 5, 6), torch.zeros(2, 5, dtype=torch.long)),
            (2, 5, 16),
        ),
        (
            lambda: _TokenizerEncoder(
                n_filters=8,
                ratios=[8, 4, 2],
                kernel_size=5,
                last_kernel_size=5,
                n_dim=16,
                n_head=4,
                dropout=0.0,
                n_neuro=3,
            ),
            lambda: (torch.randn(2, 5, 1, 512), torch.randn(2, 5, 16)),
            (2, 3, 1, 8, 16),  # channels (5) collapse to n_neuro (3); T = 512/64
        ),
    ],
    ids=["rmsnorm", "st_block", "sensor_module", "tokenizer_encoder"],
)
def test_brainomni_submodule_shapes(build, make_inputs, exp_shape):
    out = build()(*make_inputs())
    assert out.shape == exp_shape
    assert torch.isfinite(out).all()


def test_seanet_roundtrip_downsampling():
    kw = dict(
        channels=1,
        dimension=32,
        n_filters=8,
        ratios=[8, 4, 2],
        kernel_size=5,
        last_kernel_size=5,
    )
    enc, dec = _SEANetEncoder(**kw), _SEANetDecoder(**kw)
    z = enc(torch.randn(4, 1, 512))  # 512 / (8*4*2) = 8
    assert z.shape == (4, 32, 8)
    x_rec = dec(z)
    assert x_rec.shape == (4, 1, 512)
    assert torch.isfinite(x_rec).all()


def test_seanet_roundtrip_supports_odd_ratio():
    kw = dict(
        channels=1,
        dimension=8,
        n_filters=4,
        ratios=[5],
        kernel_size=5,
        last_kernel_size=5,
    )
    encoder, decoder = _SEANetEncoder(**kw), _SEANetDecoder(**kw)
    x = torch.randn(1, 1, 25)

    reconstruction = decoder(encoder(x))

    assert reconstruction.shape == x.shape
    assert torch.isfinite(reconstruction).all()


def test_seanet_reflect_padding_supports_one_sample_input():
    encoder = _SEANetEncoder(
        channels=1,
        dimension=8,
        n_filters=4,
        ratios=[2],
        kernel_size=5,
        last_kernel_size=5,
    )

    encoded = encoder(torch.randn(1, 1, 1))

    assert encoded.shape == (1, 8, 1)
    assert torch.isfinite(encoded).all()


# ---- residual vector quantization --------------------------------------------


def test_brain_quantizer_shapes_and_loss():
    q = _quantizer(num_quantizers=4).eval()
    x_q, indices, loss = q(torch.randn(2, 5, 16))
    assert x_q.shape == (2, 5, 16)
    assert indices.shape == (2, 5, 4)  # num_quantizers
    assert torch.isfinite(loss)


def test_brain_quantizer_initializes_codebook_from_first_batch():
    torch.manual_seed(0)
    quantizer = _quantizer(num_quantizers=1).train()
    codebook = quantizer.layers[0]._codebook
    x = torch.randn(8, 10, 16)

    assert codebook.inited.item() == 0
    assert torch.count_nonzero(codebook.cluster_size) == 0
    assert torch.count_nonzero(codebook.embed) == 0

    quantizer(x)

    assert codebook.inited.item() == 1
    assert codebook.cluster_size.sum() > 0
    assert torch.isfinite(codebook.embed).all()
    assert codebook.embed.norm(dim=-1).max() < 1.1


def test_brain_quantizer_supports_torch_without_compiler_namespace(monkeypatch):
    """PyTorch 2.0 has no public ``torch.compiler`` namespace."""
    codebook = Codebook(
        dim=2,
        codebook_size=2,
        threshold_ema_dead_code=0,
        kmeans_init=False,
    ).eval()
    monkeypatch.delattr(torch, "compiler")

    quantized, indices = codebook(torch.randn(1, 2, 2))

    assert quantized.shape == (1, 2, 2)
    assert indices.shape == (1, 2)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_brain_quantizer_distributed_ema_uses_global_statistics(tmp_path):
    world_size = 2
    mp.spawn(
        _distributed_codebook_update,
        args=(world_size, str(tmp_path / "init"), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )
    states = [
        torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
        for rank in range(world_size)
    ]

    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key])
    torch.testing.assert_close(states[0]["cluster_size"], torch.tensor([3.0, 3.0]))
    torch.testing.assert_close(
        states[0]["embed_avg"], torch.tensor([[-3.5, 0.0], [3.5, 0.0]])
    )
    torch.testing.assert_close(
        states[0]["embed"], torch.tensor([[-7.0 / 6.0, 0.0], [7.0 / 6.0, 0.0]])
    )
    assert states[0]["fresh_inited"].item() == 1


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"dim": 0}, "dim"),
        ({"codebook_dim": 0}, "codebook_dim"),
        ({"codebook_size": 0}, "codebook_size"),
        ({"num_quantizers": 0}, "num_quantizers"),
    ],
)
def test_brain_quantizer_rejects_invalid_arguments(kwargs, match):
    defaults = {
        "dim": 16,
        "codebook_dim": 16,
        "codebook_size": 32,
        "num_quantizers": 2,
    }
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=match):
        ResidualVQ(**defaults)


@pytest.mark.parametrize(
    "train, expect_change",
    [(True, True), (False, False)],
    ids=["train_updates", "eval_frozen"],
)
def test_brain_quantizer_codebook_ema(train, expect_change):
    torch.manual_seed(0)
    q = _quantizer(num_quantizers=2)
    q.train(train)
    codebook = q.layers[0]._codebook
    q(torch.randn(8, 10, 16))  # initialize fresh K-means codebooks
    before = codebook.embed.clone()
    for _ in range(3):
        q(torch.randn(8, 10, 16))
    assert (not torch.allclose(before, codebook.embed)) is expect_change


# ---- public BrainTokenizer ---------------------------------------------------


def test_braintokenizer_is_subclass():
    assert issubclass(BrainTokenizer, EEGModuleMixin)


def test_braintokenizer_has_final_layer():
    last_two = [name for name, _ in _small_tokenizer().named_children()][-2:]
    assert "final_layer" in last_two


@pytest.mark.parametrize("n_times", [300, 512, 600])
def test_braintokenizer_forward_reconstruction_shape(n_times):
    model = _small_tokenizer(n_times=n_times).eval()
    x = torch.randn(2, 4, n_times)
    assert model(x).shape == x.shape


def test_braintokenizer_first_training_forward_keeps_codebooks_bounded():
    torch.manual_seed(0)
    model = _small_tokenizer().train()
    codebook = model.quantizer.layers[0]._codebook

    reconstruction = model(torch.randn(2, 4, 512))

    assert torch.isfinite(reconstruction).all()
    assert codebook.inited.item() == 1
    assert codebook.cluster_size.sum() > 0
    assert codebook.embed.norm(dim=-1).max() < 1.1


@pytest.mark.parametrize(
    "window_length, ratios",
    [(5, (5,)), (1, (1,))],
    ids=["odd_ratio", "one_sample_window"],
)
def test_braintokenizer_supports_documented_positive_ratios_and_windows(
    window_length, ratios
):
    model = BrainTokenizer(
        chs_info=_eeg_chs_info(2),
        n_times=window_length,
        sfreq=256.0,
        window_length=window_length,
        ratios=ratios,
        emb_dim=8,
        n_neuro=2,
        n_filters=4,
        tokenizer_num_heads=2,
        codebook_dim=8,
        codebook_size=8,
        num_quantizers=1,
    ).eval()
    x = torch.randn(1, 2, window_length)

    assert model(x).shape == x.shape


def test_brainomni_constructs_from_official_stage2_config():
    official_config = {
        "window_length": 8,
        "n_filters": 4,
        "ratios": [2],
        "kernel_size": 3,
        "last_kernel_size": 3,
        "n_dim": 8,
        "n_head": 2,
        "n_neuro": 2,
        "dropout": 0.0,
        "codebook_dim": 8,
        "codebook_size": 8,
        "num_quantizers": 1,
        "rotation_trick": True,
        "quantize_optimize_method": "ema",
        "overlap_ratio": 0.0,
        "lm_dim": 8,
        "lm_head": 2,
        "lm_depth": 2,
        "lm_dropout": 0.0,
        "mask_ratio": 0.5,
        "num_quantizers_used": 1,
    }
    original_config = dict(official_config)

    model = BrainOmni.from_opentslab_config(
        official_config,
        chs_info=_eeg_chs_info(2),
        n_outputs=3,
        n_times=8,
        sfreq=256.0,
    )

    assert model.lm_dim == 8
    assert len(model.blocks) == 2
    assert model.tokenizer.emb_dim == 8
    assert official_config == original_config


def test_braintokenizer_reconstruction_zero_fills_dropped_tail():
    model = _small_tokenizer(n_times=600).eval()
    reconstruction = model(torch.randn(1, 4, 600))
    assert torch.count_nonzero(reconstruction[..., 512:]) == 0


def test_braintokenizer_encode_decode_and_tokenize():
    model = _small_tokenizer()
    x = torch.randn(2, 4, 512)
    recon, commit_loss, indices = model.encode_decode(x)
    assert recon.shape == x.shape
    assert torch.isfinite(commit_loss)
    assert indices.shape[-1] == 2  # num_quantizers
    model.eval()
    feat, idx = model.tokenize(x)
    assert feat.shape[:2] == (2, 3) and feat.shape[-1] == 16  # (batch, n_neuro, tokens, emb_dim)
    assert idx.shape[-1] == 2


@pytest.mark.parametrize(
    "n_times, overlap_ratio, expected_starts",
    [
        (300, 0.0, [0]),
        (600, 0.0, [0]),
        (600, 0.25, [0, 384]),
    ],
)
def test_braintokenizer_unfold_matches_released_source(
    n_times, overlap_ratio, expected_starts
):
    model = _small_tokenizer(n_times=n_times)
    x = torch.arange(n_times, dtype=torch.float32).reshape(1, 1, -1)
    windows = model._unfold(x, overlap_ratio=overlap_ratio)
    assert windows[0, 0, :, 0].tolist() == expected_starts


@pytest.mark.parametrize(
    "overlap_ratio", [-0.1, 1.0 - 0.5 / 512, 1.0, 1.1]
)
def test_braintokenizer_rejects_invalid_overlap(overlap_ratio):
    model = _small_tokenizer()
    with pytest.raises(ValueError, match="overlap_ratio"):
        model.tokenize(torch.randn(1, 4, 512), overlap_ratio=overlap_ratio)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"window_length": 0}, "window_length"),
        ({"n_filters": 0}, "n_filters"),
        ({"ratios": ()}, "ratios"),
        ({"ratios": (8, 0, 2)}, "ratios"),
        ({"kernel_size": 0}, "kernel_size"),
        ({"last_kernel_size": 0}, "last_kernel_size"),
        ({"emb_dim": 15}, "emb_dim.*tokenizer_num_heads"),
        ({"tokenizer_num_heads": 0}, "tokenizer_num_heads"),
        ({"n_neuro": 0}, "n_neuro"),
        ({"drop_prob": -0.1}, "drop_prob"),
        ({"drop_prob": 1.1}, "drop_prob"),
    ],
)
def test_braintokenizer_rejects_invalid_constructor_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        BrainTokenizer(
            chs_info=_eeg_chs_info(4),
            n_times=512,
            sfreq=256.0,
            **(_BRAINOMNI_KW | kwargs),
        )


# ---- public BrainOmni classifier ---------------------------------------------


@pytest.mark.parametrize(
    "chs_info, n_times, n_outputs",
    [
        (_eeg_chs_info(4), 512, 3),  # standard EEG
        (_eeg_chs_info(4), 300, 3),  # input shorter than window_length -> padded
        (_mixed_chs_info(), 512, 2),  # mixed EEG + MAG + GRAD
    ],
    ids=["standard", "short_input", "mixed_eeg_meg"],
)
def test_brainomni_forward_shape(chs_info, n_times, n_outputs):
    model = _small_brainomni(
        chs_info=chs_info, n_outputs=n_outputs, n_times=n_times
    ).eval()
    out = model(torch.randn(2, len(chs_info), n_times))
    assert out.shape == (2, n_outputs)


def test_brainomni_encode_shape_and_normalized():
    feat = _small_brainomni().eval().encode(torch.randn(2, 4, 512))
    assert feat.ndim == 4 and feat.shape[1] == 3 and feat.shape[-1] == 16
    norms = feat.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_brainomni_reset_head_changes_only_head():
    model = _small_brainomni(n_outputs=3)
    tokenizer_id = id(model.tokenizer)
    model.reset_head(5)
    assert model(torch.randn(2, 4, 512)).shape == (2, 5)
    assert id(model.tokenizer) == tokenizer_id  # backbone untouched


def test_brainomni_return_features_head_contract():
    model = _small_brainomni().eval()
    x = torch.randn(2, 4, 512)
    bundle = model(x, return_features=True)
    assert set(bundle) == {"features", "cls_token"}
    assert bundle["cls_token"] is None
    assert torch.allclose(model.final_layer(bundle["features"]), model(x))


def test_brainomni_reset_head_preserves_dtype():
    model = _small_brainomni().double()
    model.reset_head(5)
    assert next(model.final_layer.parameters()).dtype == torch.float64


def test_brainomni_released_dropout_configuration():
    """Tokenizer and Stage-2 dropout follow their separate released configs."""
    model = _small_brainomni()
    assert model.tokenizer.drop_prob == 0.0
    assert model.tokenizer.encoder.backwardsolution.dropout == 0.0
    assert model.tokenizer.final_layer.forwardsolution.dropout == 0.0
    assert model.drop_prob == 0.1
    assert model.blocks[0].time_attn.dropout == 0.1
    assert model.final_layer[0].p == 0.1


def test_brainomni_downstream_head_dropout_is_independent():
    """The released downstream head keeps 0.1 dropout when LM dropout changes."""
    model = BrainOmni(
        chs_info=_eeg_chs_info(4),
        n_outputs=3,
        n_times=512,
        sfreq=256.0,
        lm_dim=16,
        num_heads=4,
        depth=2,
        drop_prob=0.0,
        **_BRAINOMNI_KW,
    )
    assert model.blocks[0].time_attn.dropout == 0.0
    assert model.final_layer[0].p == 0.1


@pytest.mark.parametrize("model_factory", [_small_tokenizer, _small_brainomni])
def test_brainomni_native_state_dict_roundtrip(model_factory):
    source = model_factory()
    target = model_factory()
    target.load_state_dict(source.state_dict(), strict=True)
    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], value)


@pytest.mark.parametrize("model_factory", [_small_tokenizer, _small_brainomni])
def test_brainomni_checkpoint_key_remap_rejects_collisions(model_factory):
    model = model_factory()
    state_dict = model.state_dict()
    if isinstance(model, BrainTokenizer):
        native_key = next(key for key in state_dict if ".conv.weight_g" in key)
        official_key = native_key.replace(".conv.weight_g", ".conv.conv.weight_g")
    else:
        native_key = next(
            key for key in state_dict if "tokenizer." in key and ".conv.weight_g" in key
        )
        official_key = native_key.replace(".conv.weight_g", ".conv.conv.weight_g")
    state_dict[official_key] = state_dict[native_key].clone()
    with pytest.raises(ValueError, match="collide"):
        model.load_state_dict(state_dict, strict=True)


def test_brainomni_tokenizer_parameters_are_frozen():
    model = _small_brainomni()
    assert all(not parameter.requires_grad for parameter in model.tokenizer.parameters())
    assert all(parameter.requires_grad for parameter in model.blocks.parameters())


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"overlap_ratio": -0.1}, "overlap_ratio"),
        ({"overlap_ratio": 1.0 - 0.5 / 512}, "overlap_ratio"),
        ({"overlap_ratio": 1.0}, "overlap_ratio"),
        ({"lm_dim": 15}, "lm_dim.*num_heads"),
        ({"num_heads": 3}, "num_heads.*even"),
        ({"depth": 0}, "depth"),
        ({"tokenizer_drop_prob": -0.1}, "tokenizer_drop_prob"),
        ({"tokenizer_drop_prob": 1.1}, "tokenizer_drop_prob"),
        ({"drop_prob": -0.1}, "drop_prob"),
        ({"drop_prob": 1.1}, "drop_prob"),
    ],
)
def test_brainomni_rejects_invalid_constructor_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        model_kwargs = dict(lm_dim=16, num_heads=4, depth=2)
        model_kwargs.update(kwargs)
        BrainOmni(
            chs_info=_eeg_chs_info(4),
            n_outputs=3,
            n_times=512,
            sfreq=256.0,
            **(_BRAINOMNI_KW | model_kwargs),
        )


@pytest.mark.network
@pytest.mark.huggingface
def test_braintokenizer_released_checkpoint_strict_load_and_parity(tmp_path):
    """Gate the exact official tokenizer artifact and public loading path."""
    if not HAS_HF_HUB:
        pytest.skip("huggingface_hub is required")
    revision = "9a4d3c70495370397ccfbfd6d2496f25647545a5"
    path = Path(
        hf_hub_download(
            "OpenTSLab/BrainOmni",
            "braintokenizer/BrainTokenizer.pt",
            revision=revision,
            cache_dir=tmp_path,
        )
    )
    config_path = Path(
        hf_hub_download(
            "OpenTSLab/BrainOmni",
            "braintokenizer/model_cfg.json",
            revision=revision,
            cache_dir=tmp_path,
        )
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "d41c44c14c3f3b11fd0fb660752e356dff4cb4bc5f32a05f470f503ffddc7b1a"
    )
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == (
        "67d99edcfdc54285bed7f37757b6fc9732199c56ac48f86027af1c8a22e95183"
    )
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model = BrainTokenizer.from_opentslab_config(
        json.loads(config_path.read_text()),
        chs_info=_eeg_chs_info(2),
        n_times=512,
        sfreq=256.0,
    ).eval()
    model.load_state_dict(state_dict, strict=True)

    # Values captured from pinned source 340d6b5 with the same released
    # artifact and deterministic input/geometry.
    model.pos.copy_(
        torch.tensor([[0.1, 0.2, 0.3, 0, 0, 0], [-0.2, 0.1, 0.4, 0, 0, 0]])
    )
    torch.manual_seed(123)
    feat, indices = model.tokenize(torch.randn(1, 2, 512))
    assert feat.shape == (1, 16, 8, 256)
    assert indices.shape == (1, 16, 8, 4)
    expected = torch.tensor(
        [
            -0.048095703125,
            0.039093017578125,
            0.0056915283203125,
            -0.0263824462890625,
            0.063201904296875,
            0.07301521301269531,
            -0.024904251098632812,
            -0.02942657470703125,
            -0.081573486328125,
            -0.005706787109375,
            -0.040435791015625,
            0.038818359375,
            -0.12237548828125,
            -0.0460357666015625,
            0.004184722900390625,
            0.00177764892578125,
        ]
    )
    torch.testing.assert_close(feat.flatten()[:16], expected, rtol=1e-5, atol=1e-5)
    assert feat.sum().item() == pytest.approx(-52.497127532958984, abs=1e-4)
    assert indices.sum().item() == 138744


@pytest.mark.network
@pytest.mark.huggingface
def test_brainomni_released_checkpoint_strict_load_and_parity(tmp_path):
    """Gate the exact official tiny artifact and deterministic encoder path."""
    if not HAS_HF_HUB:
        pytest.skip("huggingface_hub is required")
    revision = "9a4d3c70495370397ccfbfd6d2496f25647545a5"
    path = Path(
        hf_hub_download(
            "OpenTSLab/BrainOmni",
            "tiny/BrainOmni.pt",
            revision=revision,
            cache_dir=tmp_path,
        )
    )
    config_path = Path(
        hf_hub_download(
            "OpenTSLab/BrainOmni",
            "tiny/model_cfg.json",
            revision=revision,
            cache_dir=tmp_path,
        )
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "62c67ba6a84ea0625e67a3b5e7463fe3930bfee88a612a225e9062a052542ffc"
    )
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == (
        "4b994b38f1a8dccc8136f2b837a8ad2d3ebb028a1af0c527f9e0902a1a5c252e"
    )
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model = BrainOmni.from_opentslab_config(
        json.loads(config_path.read_text()),
        chs_info=_eeg_chs_info(2),
        n_times=512,
        sfreq=256.0,
        n_outputs=3,
    ).eval()
    original_head = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if key.startswith("final_layer.")
    }
    model.load_state_dict(state_dict, strict=True)
    assert all(
        torch.equal(model.state_dict()[key], value)
        for key, value in original_head.items()
    )
    assert torch.any(
        model.blocks[0].time_attn.rope_embedding_layer.rotate.imag != 0
    )

    # The released DeepSpeed export stores RoPE's derived complex cache as
    # real-only. Both the pinned upstream run and public loader regenerate the
    # mathematically defined cache before computing this signature.
    model.tokenizer.pos.copy_(
        torch.tensor([[0.1, 0.2, 0.3, 0, 0, 0], [-0.2, 0.1, 0.4, 0, 0, 0]])
    )
    torch.manual_seed(123)
    feat = model.encode(torch.randn(1, 2, 512))
    assert feat.shape == (1, 16, 8, 256)
    expected = torch.tensor(
        [
            0.00458572618663311,
            -0.005507787223905325,
            -0.02145325019955635,
            -0.047605402767658234,
            -0.05783329904079437,
            0.04926654323935509,
            -0.00911555252969265,
            -0.03314506262540817,
            -0.006572749465703964,
            -0.22567233443260193,
            -0.12159579992294312,
            0.00981579814106226,
            -0.04502265900373459,
            -0.006218839902430773,
            0.01613355241715908,
            0.06309985369443893,
        ]
    )
    torch.testing.assert_close(feat.flatten()[:16], expected, rtol=1e-5, atol=1e-5)
    assert feat.sum().item() == pytest.approx(62.14745330810547, abs=1e-4)


@pytest.mark.network
@pytest.mark.huggingface
def test_brainomni_base_released_checkpoint_strict_load(tmp_path):
    """Gate the exact official base architecture and public loading path."""
    if not HAS_HF_HUB:
        pytest.skip("huggingface_hub is required")
    revision = "9a4d3c70495370397ccfbfd6d2496f25647545a5"
    path = Path(
        hf_hub_download(
            "OpenTSLab/BrainOmni",
            "base/BrainOmni.pt",
            revision=revision,
            cache_dir=tmp_path,
        )
    )
    config_path = Path(
        hf_hub_download(
            "OpenTSLab/BrainOmni",
            "base/model_cfg.json",
            revision=revision,
            cache_dir=tmp_path,
        )
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "435db24e57a55df05aa7e16355def7b7ecbedb22aa1ec16063e7d14efd2386d0"
    )
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == (
        "492e2229b1fb87d49330b23f482a1641ec7cdc0b41d38f76384f18fdef3696d5"
    )
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model = BrainOmni.from_opentslab_config(
        json.loads(config_path.read_text()),
        chs_info=_eeg_chs_info(2),
        n_times=512,
        sfreq=256.0,
        n_outputs=3,
    )
    original_head = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if key.startswith("final_layer.")
    }
    model.load_state_dict(state_dict, strict=True)
    assert torch.equal(model.projection.weight, state_dict["projection.weight"])
    assert all(
        torch.equal(model.state_dict()[key], value)
        for key, value in original_head.items()
    )


def test_brainomni_vq_frozen_during_train_step():
    torch.manual_seed(0)
    model = _small_brainomni().train()
    x = torch.randn(2, 4, 512)
    model.tokenizer.tokenize(x)  # one-time initialization for a fresh tokenizer
    codebook = model.tokenizer.quantizer.layers[0]._codebook
    before = codebook.embed.clone()
    model(x).sum().backward()
    assert torch.allclose(before, codebook.embed)  # frozen tokenizer


def test_brainomni_sfreq_warning():
    with pytest.warns(UserWarning, match="256"):
        _small_brainomni(sfreq=128.0)
    with pytest.warns(UserWarning, match="256"):
        _small_brainomni(sfreq=255.6)
