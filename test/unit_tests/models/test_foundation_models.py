# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3

import json
import os
from contextlib import nullcontext
from pathlib import Path
from urllib.error import URLError

import mne
import pooch
import pytest
import torch

import braindecode.models.luna as luna_module
import braindecode.models.zuna as zuna_module

try:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

from braindecode.models import (
    DIVER1,
    LUNA,
    REVE,
    ZUNA,
    CBraMod,
    CodeBrain,
    Labram,
    STEEGFormer,
    steegformer,
)
from braindecode.models.diver1 import _STCPE, channel_metadata_from_chs_info
from braindecode.models.labram import LABRAM_CHANNEL_ORDER
from braindecode.models.luna import _RotarySelfAttentionBlock
from braindecode.models.reve import Attention, FourierEmb4D, RevePositionBank
from braindecode.util import resolve_montage_name

_ORIGINAL_TORCH_CAT = torch.cat


def _reject_empty_tensors(tensors, *args, **kwargs):
    dim = kwargs.get("dim", args[0] if args else 0)
    if any(tensor.shape[dim] == 0 for tensor in tensors):
        raise RuntimeError("backend does not support empty tensor concatenation")
    return _ORIGINAL_TORCH_CAT(tensors, *args, **kwargs)


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


@pytest.mark.parametrize("qkv_bias", [False, True])
def test_labram_attention_calls_qkv_module(qkv_bias):
    # Adapters such as LoRA hook or replace ``attn.qkv``; the attention has to call
    # the module, not only read its weight, or they have no effect.
    ch_names = list(LABRAM_CHANNEL_ORDER[:4])
    model = Labram(n_chans=4, n_outputs=2, n_times=800, qkv_bias=qkv_bias).eval()
    attentions = [block.attn for block in model.blocks]
    if qkv_bias:  # non-zero biases, as in the checkpoint
        for attn in attentions:
            torch.nn.init.normal_(attn.q_bias)
            torch.nn.init.normal_(attn.v_bias)
    x = torch.randn(2, 4, 800)
    reference = model(x, ch_names=ch_names)

    calls = []
    handles = [
        attn.qkv.register_forward_hook(lambda *_: calls.append(None))
        for attn in attentions
    ]
    assert torch.equal(model(x, ch_names=ch_names), reference)
    assert len(calls) == len(attentions)
    for handle in handles:
        handle.remove()

    for attn in attentions:  # an adapter that changes the projection
        attn.qkv.register_forward_hook(lambda _m, _i, out: 2 * out)
    assert not torch.allclose(model(x, ch_names=ch_names), reference)


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


def test_luna_full_rotary_attention_avoids_empty_concatenation(monkeypatch):
    """Full-head RoPE does not concatenate zero-width tensor views."""
    monkeypatch.setattr(torch, "cat", _reject_empty_tensors)
    attention = _RotarySelfAttentionBlock(dim=32, num_heads=4)
    signal = torch.randn(2, 10, 32, requires_grad=True)

    output = attention(signal)
    output.sum().backward()

    assert output.shape == signal.shape
    assert signal.grad is not None


def test_luna_rotary_embedding_is_native_and_checkpoint_compatible():
    """LUNA uses Braindecode-native RoPE without changing checkpoint keys."""
    attention = _RotarySelfAttentionBlock(dim=32, num_heads=4)

    assert type(attention.rotary_emb).__module__ == luna_module.__name__
    assert "rotary_emb.freqs" in attention.state_dict()
    expected = 1.0 / (10000 ** (torch.arange(0, 8, 2).float() / 8))
    torch.testing.assert_close(attention.rotary_emb.freqs, expected)


@pytest.mark.parametrize("axis_dim", [2, 4, 8])
def test_zuna_builds_rotary_frequency_table_natively(axis_dim):
    """ZUNA's native table matches rotary-embedding-torch semantics."""
    positions = torch.arange(5, dtype=torch.float32)
    table = zuna_module._build_rotary_frequency_table(
        positions, axis_dim=axis_dim, theta=10000.0
    )

    embedding_dim = max(axis_dim, 4)
    inverse_frequencies = 1.0 / (
        10000.0 ** (torch.arange(0, embedding_dim, 2).float() / embedding_dim)
    )
    expected = torch.outer(positions, inverse_frequencies).repeat_interleave(2, dim=1)
    torch.testing.assert_close(table, expected[:, :axis_dim])


# ==============================================================================
# Tests for ZUNA's on_non_divisible option
# ==============================================================================

_ZUNA_SMALL = dict(n_outputs=2, sfreq=250.0, dim=64, n_layers=1, n_heads=2, head_dim=32)


def _zuna_chs_info():
    info = mne.create_info(["Fz", "Cz", "Pz", "C3", "C4", "O1"], 250.0, "eeg")
    info.set_montage("standard_1020")
    return info["chs"]


def test_zuna_rejects_non_divisible_n_times_by_default():
    """The default ``on_non_divisible="error"`` keeps the previous behavior."""
    with pytest.raises(ValueError, match="divisible"):
        ZUNA(chs_info=_zuna_chs_info(), n_times=1000, **_ZUNA_SMALL)


def test_zuna_rejects_invalid_on_non_divisible():
    """An unknown ``on_non_divisible`` value raises, even for a divisible n_times."""
    with pytest.raises(ValueError, match="on_non_divisible"):
        ZUNA(
            chs_info=_zuna_chs_info(),
            n_times=1024,
            on_non_divisible="bogus",
            **_ZUNA_SMALL,
        )


def test_zuna_pad_equals_explicit_zero_padding():
    """``"pad"`` matches the same weights built with a padded ``n_times``."""
    torch.manual_seed(0)
    padded = ZUNA(
        chs_info=_zuna_chs_info(), n_times=1000, on_non_divisible="pad", **_ZUNA_SMALL
    ).eval()
    reference = ZUNA(chs_info=_zuna_chs_info(), n_times=1024, **_ZUNA_SMALL).eval()
    reference.load_state_dict(padded.state_dict())
    x = torch.randn(2, 6, 1000)
    torch.testing.assert_close(
        padded(x), reference(torch.nn.functional.pad(x, (0, 24))), rtol=0, atol=0
    )


def test_zuna_crop_drops_trailing_samples():
    """``"crop"`` matches the same weights built with a cropped ``n_times``."""
    torch.manual_seed(0)
    cropped = ZUNA(
        chs_info=_zuna_chs_info(), n_times=1000, on_non_divisible="crop", **_ZUNA_SMALL
    ).eval()
    reference = ZUNA(chs_info=_zuna_chs_info(), n_times=992, **_ZUNA_SMALL).eval()
    reference.load_state_dict(cropped.state_dict())
    x = torch.randn(2, 6, 1000)
    torch.testing.assert_close(cropped(x), reference(x[..., :992]), rtol=0, atol=0)


def test_reve_attention_matches_explicit_attention():
    attention = Attention(dim=16, heads=2, head_dim=8)
    x = torch.randn(2, 5, 16, requires_grad=True)
    q, k, v = (
        t.reshape(2, 5, 2, 8).transpose(1, 2)
        for t in attention.to_qkv(attention.norm(x)).chunk(3, dim=-1)
    )
    weights = (q @ k.transpose(-1, -2) / 8**0.5).softmax(dim=-1)
    expected = attention.to_out((weights @ v).transpose(1, 2).reshape(2, 5, 16))
    actual = attention(x)
    torch.testing.assert_close(actual, expected)
    parameters = (x, *attention.parameters())
    actual_grads = torch.autograd.grad(
        actual.square().sum(), parameters, retain_graph=True
    )
    expected_grads = torch.autograd.grad(expected.square().sum(), parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


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


def test_reve_fourier_emb_4d_computes_in_float32():
    """Intel Gaudi (HPU) autocast feeds sin/cos bf16 position x frequency products.

    CPU autocast leaves elementwise ``mul`` alone, so bf16 positions reproduce it.
    ``_embed`` is the unguarded computation.
    """
    torch.manual_seed(0)
    electrodes = torch.randn(2, 16, 3)
    positions = FourierEmb4D.add_time_patch(
        electrodes / electrodes.norm(dim=-1, keepdim=True), 3
    )
    module = FourierEmb4D(dimension=64, freqs=4)
    reference = module._embed(positions)
    assert torch.equal(module(positions), reference)

    def rel_error(out):
        return ((out.float() - reference).norm() / reference.norm()).item()

    bf16 = positions.to(torch.bfloat16)
    out = module(bf16)
    assert out.dtype == torch.bfloat16
    assert rel_error(out) < 0.01 < 0.02 < rel_error(module._embed(bf16))


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


@pytest.fixture
def diver1_model():
    info = mne.create_info([f"A{i}" for i in range(6)], 500.0, "seeg")
    for i, ch in enumerate(info["chs"]):
        ch["loc"][:3] = [0.01 * i, 0.02, -0.03]
    return DIVER1(
        n_outputs=4, chs_info=info["chs"], n_times=1000, sfreq=500.0,
        pooling="mean", d_model=64, n_layers=2,
    ).eval()


@pytest.mark.parametrize(
    "kind,located,slots", [("ecog", True, [1., 0.]), ("eeg", False, [0., -1.])]
)
def test_diver1_channel_metadata(kind, located, slots):
    info = mne.create_info(["A0", "A1"], 500.0, kind)
    if located:
        for i, ch in enumerate(info["chs"]):
            ch["loc"][:3] = [0.01 * i, 0.02, -0.03]
    metadata = channel_metadata_from_chs_info(info["chs"])
    assert metadata.shape == (2, 5)
    torch.testing.assert_close(metadata[:, 3:], torch.tensor([slots, slots]))
    if located:
        torch.testing.assert_close(metadata[1, :3], torch.tensor([10., 20., -30.]))
    else:
        assert torch.isnan(metadata[:, :3]).all()


@pytest.mark.parametrize("kind, slots", [("eeg", [0, -1]), ("ecog", [1, 0]), ("seeg", [1, 2]), ("dbs", [1, 2])])
def test_diver1_channel_metadata_from_chs_info(kind, slots):
    """Standalone metadata retains MNE units and DIVER-1 type slots."""
    info = mne.create_info(["A1", "A2"], 500.0, kind)
    info["chs"][0]["loc"][:3] = [0.01, 0.02, -0.03]
    metadata = channel_metadata_from_chs_info(info["chs"])
    torch.testing.assert_close(metadata[0, :3], torch.tensor([10.0, 20.0, -30.0]))
    assert torch.isnan(metadata[1, :3]).all()
    assert metadata[:, 3:].tolist() == [slots, slots]
    with pytest.raises(ValueError, match="cannot determine"):
        channel_metadata_from_chs_info([dict(kind="unknown")])


def test_diver1_channel_metadata_rejects_unknown_modality():
    info = mne.create_info(["A0", "A1"], 500.0, "misc")
    with pytest.raises(ValueError, match="cannot determine the recording modality"):
        channel_metadata_from_chs_info(info["chs"])


def test_diver1_montage_switching_and_permutation(diver1_model):
    model = diver1_model
    other = mne.create_info([f"B{i}" for i in range(9)], 500.0, "ecog")
    for i, ch in enumerate(other["chs"]):
        ch["loc"][:3] = [0.01 * i, 0.02, -0.03]
    metadata = channel_metadata_from_chs_info(other["chs"])
    xa, xb = torch.randn(2, 6, 1000), torch.randn(2, 9, 1000)
    perm = torch.tensor([4, 0, 3, 1, 5, 2])
    with torch.no_grad():
        first_a, first_b = model(xa), model(xb, metadata)
        assert first_b.shape == (2, 4)
        torch.testing.assert_close(model(xb, metadata), first_b)
        torch.testing.assert_close(model(xa), first_a)
        torch.testing.assert_close(model(xa, model.default_chan_metadata), first_a)
        torch.testing.assert_close(
            model(xa[:, perm], model.default_chan_metadata[perm]), first_a,
            atol=1e-5, rtol=1e-5,
        )


@pytest.mark.parametrize(
    "metadata,pooling,match",
    [
        (None, "mean", "built for 6 channels but got input with 9"),
        (torch.zeros(3, 5), "mean", r"shape \(9, 5\)"),
        (torch.zeros(9, 4), "mean", r"shape \(9, 5\)"),
        (torch.full((9, 5), 7.0), "mean", "modality column"),
        (torch.zeros(9, 5).index_fill_(1, torch.tensor([4]), 3.), "mean", "sub-modality"),
        (torch.zeros(9, 5), "flatten", "pooling='flatten'"),
    ],
)
def test_diver1_rejects_incompatible_montage(diver1_model, metadata, pooling, match):
    model = diver1_model
    if pooling == "flatten":
        model = DIVER1(
            n_outputs=4, chs_info=model.chs_info, n_times=1000, sfreq=500.0,
            pooling=pooling, d_model=64, n_layers=2,
        ).eval()
    with pytest.raises(ValueError, match=match):
        model(torch.randn(1, 9, 1000), metadata)


@pytest.mark.parametrize("n_outputs", [0, 5])
def test_diver1_reset_head_preserves_zero_outputs(diver1_model, n_outputs):
    diver1_model.reset_head(n_outputs)
    assert diver1_model.final_layer.out_features == n_outputs
    assert diver1_model.get_config()["n_outputs"] == n_outputs


def test_diver1_stcpe_preserves_low_precision_overlap(monkeypatch):
    # BF16 fold accumulates 257 overlapping ones to 256, not scalar 257.
    model = _STCPE(8, 4, 257, torch.nn.SiLU, 1).bfloat16().eval()
    original_fold = torch.nn.functional.fold
    calls = []

    def capture_fold(*args, **kwargs):
        out = original_fold(*args, **kwargs)
        calls.append(out)
        return out

    monkeypatch.setattr(torch.nn.functional, "fold", capture_fold)
    x = torch.randn(1, 1, 1, 8, dtype=torch.bfloat16, requires_grad=True)
    actual = model(x)
    assert len(calls) == 2
    folded, overlap = calls
    torch.testing.assert_close(overlap, torch.full_like(overlap, 256))
    expected = model.up(model.unfold_features(folded / overlap))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(actual.sum(), x, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), x)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)



@pytest.mark.parametrize("patch_size", [50, 500])
@pytest.mark.parametrize("out_size", [8, 16, 32])
def test_diver1_cnn_out_size(patch_size, out_size, tmp_path):
    """Output-size HPO preserves token width and survives config/Hub cloning."""
    info = mne.create_info(["A0", "A1"], 500.0, "seeg")
    kwargs = dict(
        chs_info=info["chs"], n_outputs=2, n_times=2 * patch_size - 1,
        patch_size=patch_size, d_model=64, n_layers=1, pooling="mean",
        cnn_out_size=out_size,
    )
    torch.manual_seed(21)
    model = DIVER1(**kwargs).eval()
    init_rng = torch.get_rng_state()
    assert model.cnn_out_size == out_size
    assert model.get_config()["cnn_out_size"] == out_size
    padded = 1 << (patch_size - 1).bit_length()
    assert model.patch_cnn.proj_in[0].stride == (1, padded // out_size)
    assert model.patch_cnn.proj_in[0].out_channels == 64 // out_size
    # The pre-existing explicit-stride path remains exactly equivalent.
    kwargs.pop("cnn_out_size")
    torch.manual_seed(21)
    legacy = DIVER1(**kwargs, cnn_stride=padded // out_size).eval()
    assert torch.equal(init_rng, torch.get_rng_state())
    assert model.state_dict().keys() == legacy.state_dict().keys()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, legacy.state_dict()[key], rtol=0, atol=0)
    x = torch.randn(1, 2, 2 * patch_size - 1, requires_grad=True)
    tokens = model.patch_cnn(model.patch_tokenizer(x))
    assert tokens.shape == (1, 2, 2, 64)
    actual = model(x)
    assert actual.shape == (1, 2)
    torch.testing.assert_close(actual, legacy(x), rtol=0, atol=0)
    actual.sum().backward()
    assert torch.isfinite(x.grad).all()
    # Standard module deepcopy, config reconstruction, and local Hub roundtrip.
    import copy

    cloned = copy.deepcopy(model)
    rebuilt = DIVER1.from_config(model.get_config()).eval()
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    model.save_pretrained(tmp_path)
    restored = DIVER1.from_pretrained(tmp_path).eval()
    scripted = torch.jit.script(model)
    for candidate in (cloned, rebuilt, restored, scripted):
        torch.testing.assert_close(candidate(x), actual, rtol=0, atol=0)
    for candidate in (cloned, rebuilt, restored):
        assert candidate.get_config()["cnn_out_size"] == out_size


@pytest.mark.parametrize(
    "options,match",
    [
        ({"cnn_out_size": 0}, "positive integer divisor"),
        ({"cnn_out_size": -8}, "positive integer divisor"),
        ({"cnn_out_size": 3}, "positive integer divisor"),
        ({"cnn_out_size": 1024}, "positive integer divisor"),
        ({"cnn_out_size": 8.0}, "positive integer divisor"),
        ({"cnn_out_size": True}, "positive integer divisor"),
        ({"cnn_out_size": 32, "d_model": 48, "num_heads": 2}, "d_model.*divisible"),
        ({"cnn_out_size": 8, "cnn_stride": 64}, "mutually exclusive"),
        ({"cnn_out_size": 8, "cnn_stride": 32}, "mutually exclusive"),
    ],
)
def test_diver1_cnn_out_size_validation(options, match):
    info = mne.create_info(["A0", "A1"], 500.0, "seeg")
    kwargs = dict(chs_info=info["chs"], n_outputs=2, n_times=500, d_model=64, n_layers=1)
    with pytest.raises(ValueError, match=match):
        DIVER1(**(kwargs | options))


def test_diver1_mup_attention_scale():
    """The released DIVER-1 checkpoints need attention scaled by 1 / head_dim."""
    info = mne.create_info(["C3", "Cz", "C4"], 500.0, "eeg")
    info.set_montage("standard_1020")
    for mup in (True, False):
        model = DIVER1(
            chs_info=info["chs"], n_outputs=2, n_times=1000, mup_attention=mup
        )
        attention = [m for m in model.modules() if hasattr(m, "head_dim")]
        assert attention
        for module in attention:
            assert module.scale == (1.0 / module.head_dim if mup else None)


@pytest.fixture
def steeg_vocab(monkeypatch):
    """Use an offline vocabulary; keep the model's real 145-slot capacity."""
    names = ["Fp1", "Fp2", "Cz", "Oz", "T7", "T8", "Pz", "Fz"]
    monkeypatch.setattr(steegformer, "_channel_order", lambda: names)
    monkeypatch.setattr(
        steegformer,
        "_channel_index",
        lambda: {n.upper(): i for i, n in enumerate(names)},
    )
    return names


@pytest.mark.parametrize(
    "names, explicit, expected, warning",
    [
        (["oz", "FP1", "Cz"], None, [3, 0, 2], None),
        (["E1", "fp1", "E2"], None, [3, 0, 5], "nearest 10-05 site"),
        (["E1", "fp1", "E2"], [5, 4, 3], [5, 4, 3], None),
    ],
    ids=["known-names", "mixed-positions", "explicit-override"],
)
@pytest.mark.filterwarnings("error")
def test_steegformer_channel_mapping(steeg_vocab, names, explicit, expected, warning):
    info = mne.create_info(["Oz", "Cz", "T8"], 250, "eeg")
    info.set_montage(
        mne.channels.make_standard_montage(resolve_montage_name("standard_1005"))
    )
    chs = [dict(ch, ch_name=name) for ch, name in zip(info["chs"], names)]
    # Unknown electrodes are slightly displaced; fp1 must ignore its Cz position.
    chs[0]["loc"][:3] += [0.003, 0, 0.002]
    chs[2]["loc"][:3] += [0, 0.004, -0.003]
    with pytest.warns(UserWarning, match=warning) if warning else nullcontext():
        model = STEEGFormer(
            n_chans=3,
            n_outputs=2,
            n_times=64,
            chs_info=chs,
            chan_pos_idx=explicit,
            embed_dim=32,
            depth=1,
            num_heads=2,
        )
    assert model.channel_indices.tolist() == expected


@pytest.mark.parametrize(
    "n_chans, fallback, n_chans_pos",
    [
        (3, "unlocated", 145),
        (146, "unlocated", 145),
        (256, "hydrocel", 145),
        (3, "absent", 145),
        (146, "absent", 145),
        (3, "unpublished", 256),
        (257, "unpublished", 256),
        (3, "unavailable", 145),
        (146, "unavailable", 145),
    ],
)
def test_steegformer_montage_fallback(
    steeg_vocab, monkeypatch, n_chans, fallback, n_chans_pos
):
    info = mne.create_info([f"E{i + 1}" for i in range(n_chans)], 250, "eeg")
    if fallback == "hydrocel":
        info.set_montage(mne.channels.make_standard_montage("GSN-HydroCel-256"))
    if fallback == "unavailable":
        def unavailable():
            raise OSError("offline")

        monkeypatch.setattr(steegformer, "_channel_index", unavailable)
    overflow = n_chans > n_chans_pos and fallback != "hydrocel"
    expectation = (
        pytest.raises(ValueError, match="identity mapping.*chan_pos_idx")
        if overflow
        else pytest.warns(
            UserWarning,
            match="nearest 10-05 site" if fallback == "hydrocel" else "identity",
        )
    )
    with expectation:
        model = STEEGFormer(
            n_chans=n_chans,
            n_outputs=2,
            n_times=64,
            chs_info=None if fallback == "absent" else info["chs"],
            n_chans_pos=n_chans_pos,
            embed_dim=32,
            depth=1,
            num_heads=2,
        )
    if not overflow:
        if fallback == "hydrocel":
            slots = model.channel_indices
            assert slots.shape == (256,)
            assert 0 <= int(slots.min()) <= int(slots.max()) < len(steeg_vocab)
        else:
            assert model.channel_indices.tolist() == list(range(n_chans))
