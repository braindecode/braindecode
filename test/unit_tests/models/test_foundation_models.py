# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3

import os
from pathlib import Path
from urllib.error import URLError

import mne
import pytest
import torch

try:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

from braindecode.models import LUNA, REVE, CBraMod, Labram
from braindecode.models.labram import LABRAM_CHANNEL_ORDER
from braindecode.models.reve import RevePositionBank


@pytest.fixture
def n_times():
    return 1000


@pytest.fixture
def n_chans():
    return 64


@pytest.fixture
def chs_info(n_chans):
    return [{"ch_name": ch_name} for ch_name in LABRAM_CHANNEL_ORDER[:n_chans]]


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
    assert model_tokenizer.n_chans == 64
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


def test_labram_neural_tokenizer_forward_features_all_tokens(
    model_tokenizer, n_chans, ch_names, n_times, emb_size
):
    """Test forward_features with return_all_tokens=True in tokenizer mode."""
    batch_size = 2
    x = torch.randn(batch_size, n_chans, n_times)
    x_reorder, input_chans = model_tokenizer._select_channels(x, ch_names)
    output = model_tokenizer.forward_features(
        x_reorder, input_chans=input_chans, return_all_tokens=True
    )

    # Output should be (batch, cls + channels*n_patchs, emb_dim)
    # With n_patchs=5 and n_chans=64: 1 + 64*5 = 321
    assert output.shape[0] == batch_size
    assert output.shape[1] == 321  # 1 cls token + 64 channels * 5 patches
    assert output.shape[2] == emb_size


def test_labram_neural_tokenizer_forward_features_patch_tokens(
    model_tokenizer, n_chans, ch_names, n_times, emb_size
):
    """Test forward_features with return_patch_tokens=True in tokenizer mode."""
    batch_size = 2
    x = torch.randn(batch_size, n_chans, n_times)
    x_reorder, input_chans = model_tokenizer._select_channels(x, ch_names)
    output = model_tokenizer.forward_features(
        x_reorder, input_chans=input_chans, return_patch_tokens=True
    )

    # Output should be (batch, channels*n_patchs, emb_dim)
    # With n_patchs=5 and n_chans=64: 64*5 = 320
    assert output.shape == (batch_size, 320, emb_size)


def test_labram_neural_tokenizer_forward_features_default(
    model_tokenizer, n_chans, ch_names, n_times, emb_size
):
    """Test forward_features with default settings in tokenizer mode."""
    batch_size = 2
    x = torch.randn(batch_size, n_chans, n_times)
    x_reorder, input_chans = model_tokenizer._select_channels(x, ch_names)
    output = model_tokenizer.forward_features(x_reorder, input_chans=input_chans)

    # Default should return mean pooled output: (batch, emb_dim)
    assert output.shape == (batch_size, emb_size)


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
    assert model_decoder.n_chans == 64
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


def test_labram_neural_decoder_forward_features_all_tokens(
    model_decoder, n_chans, n_times, emb_size
):
    """Test forward_features with return_all_tokens=True in decoder mode."""
    batch_size = 2
    x = torch.randn(batch_size, n_chans, n_times)
    x_reorder, input_chans = model_decoder._select_channels(x, ch_names=None)
    output = model_decoder.forward_features(
        x_reorder, input_chans=input_chans, return_all_tokens=True
    )

    # Output should be (batch, cls + n_patches, emb_dim)
    assert output.shape[0] == batch_size
    assert output.shape[1] == 6  # 1 cls token + 5 patches (1000 / 200 = 5)
    assert output.shape[2] == emb_size


def test_labram_neural_decoder_forward_features_patch_tokens(
    model_decoder, n_chans, n_times, emb_size
):
    """Test forward_features with return_patch_tokens=True in decoder mode."""
    batch_size = 2
    x = torch.randn(batch_size, n_chans, n_times)
    x_reorder, input_chans = model_decoder._select_channels(x, ch_names=None)
    output = model_decoder.forward_features(
        x_reorder, input_chans=input_chans, return_patch_tokens=True
    )

    # Output should be (batch, n_patches, emb_dim)
    # After removing cls token
    assert output.shape[0] == batch_size
    assert output.shape[1] == 5  # n_patches (1000 / 200 = 5)
    assert output.shape[2] == emb_size


def test_labram_neural_decoder_forward_features_default(
    model_decoder, n_chans, n_times, emb_size
):
    """Test forward_features with default settings in decoder mode."""
    batch_size = 2
    x = torch.randn(batch_size, n_chans, n_times)
    x_reorder, input_chans = model_decoder._select_channels(x, ch_names=None)
    output = model_decoder.forward_features(x_reorder, input_chans=input_chans)

    # Default should return mean pooled output: (batch, feature_dim)
    assert output.shape == (batch_size, emb_size)


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


def test_labram_no_dimension_mismatch_errors(n_times, chs_info, ch_names, n_outputs):
    """Test that there are no dimension mismatch errors in forward pass."""
    batch_size = 2

    # Test neural tokenizer mode
    model = Labram(
        n_times=n_times,
        chs_info=chs_info,
        n_outputs=n_outputs,
        neural_tokenizer=True,
    )

    x = torch.randn(batch_size, len(chs_info), n_times)

    try:
        x_reorder, input_chans = model._select_channels(x, ch_names)
        output = model.forward_features(
            x_reorder, input_chans=input_chans, return_all_tokens=True
        )
        # Should complete without error
        assert output is not None
    except RuntimeError as e:
        pytest.fail(f"forward_features raised RuntimeError: {e}")

    # Test neural decoder mode
    model = Labram(
        n_times=n_times,
        chs_info=chs_info,
        n_outputs=n_outputs,
        neural_tokenizer=False,
    )

    try:
        x_reorder, input_chans = model._select_channels(x, ch_names)
        output = model.forward_features(
            x_reorder, input_chans=input_chans, return_all_tokens=True
        )
        # Should complete without error
        assert output is not None
    except RuntimeError as e:
        pytest.fail(f"forward_features raised RuntimeError: {e}")


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
    assert isinstance(LABRAM_CHANNEL_ORDER, tuple)
    assert len(LABRAM_CHANNEL_ORDER) > 100  # Should have 100+ channels
    assert "FP1" in LABRAM_CHANNEL_ORDER
    assert "CZ" in LABRAM_CHANNEL_ORDER
    assert "O2" in LABRAM_CHANNEL_ORDER


def test_labram_with_chs_info_initialization():
    """Test Labram initialization with chs_info parameter."""
    chs_info = [
        {"ch_name": "FP1"},
        {"ch_name": "FP2"},
        {"ch_name": "CZ"},
        {"ch_name": "O1"},
        {"ch_name": "O2"},
    ]
    model = Labram(
        n_times=1000,
        n_chans=5,
        n_outputs=4,
        chs_info=chs_info,
        neural_tokenizer=True,
    )
    ch_indices = [LABRAM_CHANNEL_ORDER.index(ch["ch_name"].upper()) for ch in chs_info]

    assert model._input_channels_mask is not None
    assert model._labram_ch_indices is not None
    assert torch.equal(model._input_channels_mask, torch.tensor([1, 1, 1, 1, 1], dtype=torch.bool))
    assert torch.equal(model._labram_ch_indices, torch.tensor(ch_indices))


def test_labram_with_chs_info_forward_pass():
    """Test forward pass with chs_info parameter."""
    chs_info = [
        {"ch_name": "FP1"},
        {"ch_name": "FP2"},
        {"ch_name": "CZ"},
        {"ch_name": "O1"},
        {"ch_name": "O2"},
    ]
    model = Labram(
        n_times=1000,
        n_chans=5,
        n_outputs=4,
        chs_info=chs_info,
        neural_tokenizer=True,
        num_layers=2,
    )

    x = torch.randn(2, 5, 1000)
    output = model(x)
    assert output.shape == (2, 4)


def test_labram_channel_reordering_order():
    """Test that Labram keeps input order while mapping positional indices."""
    # Input channels in non-standard order
    chs_info = [{"ch_name": "O2"}, {"ch_name": "CZ"}, {"ch_name": "FP1"}]
    ch_names = ["O2", "CZ", "FP1"]
    model = Labram(
        n_times=1000,
        n_chans=3,
        n_outputs=4,
        chs_info=chs_info,
        neural_tokenizer=True,
        num_layers=2,
    )

    # FP1 comes before CZ, and CZ comes before O2 in LABRAM order.
    fp1_idx = LABRAM_CHANNEL_ORDER.index("FP1")
    cz_idx = LABRAM_CHANNEL_ORDER.index("CZ")
    o2_idx = LABRAM_CHANNEL_ORDER.index("O2")

    # _labram_ch_indices follow input-channel order.
    labram_indices = model._labram_ch_indices.tolist()
    assert labram_indices == [o2_idx, cz_idx, fp1_idx]
    assert torch.equal(model._input_channels_mask, torch.tensor([1, 1, 1], dtype=torch.bool))

    # Verify positional embedding indices include CLS and LABRAM positions
    x = torch.randn(1, 3, 1000)
    x_selected, input_chans = model._select_channels(x, ch_names=ch_names)
    assert torch.equal(x_selected, x)
    assert input_chans.tolist() == [0, o2_idx + 1, cz_idx + 1, fp1_idx + 1]


@pytest.mark.parametrize("forward_mode", [True, False])
def test_labram_channel_reordering_selects_correct_data(forward_mode):
    """Test that Labram channel selection preserves input order of matched channels."""
    chs_info = [{"ch_name": "O2"}, {"ch_name": "CZ"}, {"ch_name": "FP1"}]
    model = Labram(
        n_times=200,
        n_chans=3,
        n_outputs=4,
        chs_info=chs_info,
        neural_tokenizer=True,
        num_layers=2,
        patch_size=200,
    )

    # Create distinguishable input for each channel
    x = torch.zeros(1, 3, 200)
    x[0, 0, :] = 1.0  # O2 channel
    x[0, 1, :] = 2.0  # CZ channel
    x[0, 2, :] = 3.0  # FP1 channel

    ch_names = None
    if forward_mode:
        ch_names = ["CZ", "O2", "FP1"]

    # Apply reordering
    x_selected, input_chans = model._select_channels(x, ch_names=ch_names)

    # Selected tensor preserves input order for matched channels.
    assert x_selected.shape == (1, 3, 200)
    assert torch.allclose(x_selected[0, 0, :], torch.tensor(1.0))
    assert torch.allclose(x_selected[0, 1, :], torch.tensor(2.0))
    assert torch.allclose(x_selected[0, 2, :], torch.tensor(3.0))

    fp1_idx = LABRAM_CHANNEL_ORDER.index("FP1")
    cz_idx = LABRAM_CHANNEL_ORDER.index("CZ")
    o2_idx = LABRAM_CHANNEL_ORDER.index("O2")
    if forward_mode:
        assert input_chans.tolist() == [0, cz_idx + 1, o2_idx + 1, fp1_idx + 1]
    else:
        assert input_chans.tolist() == [0, o2_idx + 1, cz_idx + 1, fp1_idx + 1]


def test_labram_case_insensitive_channel_matching():
    """Test that channel names are matched case-insensitively."""
    chs_info = [
        {"ch_name": "fp1"},
        {"ch_name": "Fp2"},
        {"ch_name": "CZ"},
        {"ch_name": "o1"},
        {"ch_name": "O2"},
    ]
    model = Labram(
        n_times=1000,
        n_chans=5,
        n_outputs=4,
        chs_info=chs_info,
        neural_tokenizer=True,
        num_layers=2,
    )

    assert model._input_channels_mask is not None
    assert model._input_channels_mask.sum().item() == 5


def test_labram_unmatched_channels_warning():
    """Test that unmatched channel names produce a warning."""
    chs_info = [
        {"ch_name": "FP1"},
        {"ch_name": "UNKNOWN_CHANNEL"},
        {"ch_name": "CZ"},
    ]
    with pytest.warns(UserWarning, match="not in LABRAM_CHANNEL_ORDER"):
        model = Labram(
            n_times=1000,
            n_chans=3,
            n_outputs=4,
            chs_info=chs_info,
            neural_tokenizer=True,
            num_layers=2,
        )
    assert model._input_channels_mask.sum().item() == 2
    assert len(model._labram_ch_indices) == 2


def test_labram_no_matched_channels_error():
    """Test warning when no channels match LABRAM_CHANNEL_ORDER."""
    chs_info = [
        {"ch_name": "UNKNOWN1"},
        {"ch_name": "UNKNOWN2"},
        {"ch_name": "UNKNOWN3"},
    ]
    with pytest.raises(
        ValueError, match="No input channels matched LABRAM_CHANNEL_ORDER"
    ):
        model = Labram(
            n_times=1000,
            n_chans=3,
            n_outputs=4,
            chs_info=chs_info,
            neural_tokenizer=True,
            num_layers=2,
        )


def test_labram_without_chs_info_no_reordering():
    """Test that without chs_info, no reordering is performed."""
    model = Labram(
        n_times=1000,
        n_chans=5,
        n_outputs=4,
        neural_tokenizer=True,
        num_layers=2,
    )

    assert model._input_channels_mask is None
    assert model._labram_ch_indices is None


def test_labram_with_chs_info():
    """Test channel name extraction from chs_info (MNE format)."""
    chs_info = [
        {"ch_name": "FP1"},
        {"ch_name": "FP2"},
        {"ch_name": "CZ"},
        {"ch_name": "O1"},
        {"ch_name": "O2"},
    ]
    model = Labram(
        n_times=1000,
        n_outputs=4,
        chs_info=chs_info,
        neural_tokenizer=True,
        num_layers=2,
    )

    assert model._input_channels_mask is not None
    assert model._input_channels_mask.sum().item() == 5


def test_labram_channel_reordering_gradient_flow():
    """Test that gradients flow correctly through channel reordering."""
    chs_info = [
        {"ch_name": "O2"},
        {"ch_name": "CZ"},
        {"ch_name": "FP1"},
        {"ch_name": "FP2"},
        {"ch_name": "O1"},
    ]
    model = Labram(
        n_times=1000,
        n_chans=5,
        n_outputs=4,
        chs_info=chs_info,
        neural_tokenizer=True,
        num_layers=2,
    )

    x = torch.randn(2, 5, 1000, requires_grad=True)
    output = model(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert model.cls_token.grad is not None


def test_labram_channel_reordering_device_compatibility():
    """Test channel reordering works on GPU if available."""
    chs_info = [{"ch_name": "FP1"}, {"ch_name": "CZ"}, {"ch_name": "O2"}]
    model = Labram(
        n_times=1000,
        n_chans=3,
        n_outputs=4,
        chs_info=chs_info,
        neural_tokenizer=True,
        num_layers=2,
    )

    if torch.cuda.is_available():
        model = model.cuda()
        x = torch.randn(2, 3, 1000).cuda()
        output = model(x)
        assert output.device.type == "cuda"
        assert output.shape == (2, 4)


def test_labram_manual_input_chans_bypasses_reordering():
    """Test that providing input_chans manually bypasses automatic reordering."""
    chs_info = [{"ch_name": "O2"}, {"ch_name": "CZ"}, {"ch_name": "FP1"}]
    model = Labram(
        n_times=1000,
        n_chans=3,
        n_outputs=4,
        chs_info=chs_info,
        neural_tokenizer=True,
        num_layers=2,
    )

    x = torch.randn(2, 3, 1000)

    # Provide manual input_chans - should bypass automatic reordering
    # manual_input_chans = torch.tensor([0, 1, 2, 3])  # CLS + 3 channels
    manual_ch_names = ["FP1", "CZ", "O2"]
    output = model(x, ch_names=manual_ch_names)

    assert output.shape == (2, 4)


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

        # Load weights using safetensors with key mapping
        state_dict = load_file(model_path)
        # Apply key mapping for pretrained weights compatibility
        mapping = model.mapping.copy()
        mapping["cross_attn.temparature"] = "cross_attn.temperature"
        mapped_state_dict = {mapping.get(k, k): v for k, v in state_dict.items()}
        model.load_state_dict(mapped_state_dict, strict=False)

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
# Tests for CBraMod Model
# ==============================================================================


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
