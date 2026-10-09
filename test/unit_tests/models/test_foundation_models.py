# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3

import copy
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from urllib.error import URLError

import mne
import pooch
import pytest
import torch
import torch.nn as nn

import braindecode.models.luna as luna_module
import braindecode.models.zuna as zuna_module

try:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

from braindecode.models import (
    AXON,
    DIVER1,
    LUNA,
    MAPA,
    REVE,
    ZUNA,
    CBraMod,
    CodeBrain,
    Labram,
    PopulationTransformer,
    SleepFM,
    SleepFMStager,
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


def _labram_embedded_tokens(model, x, ch_names):
    """Tokens once the position and time embeddings are added (input of ``pos_drop``)."""
    seen = {}
    handle = model.pos_drop.register_forward_pre_hook(
        lambda _module, args: seen.update(tokens=args[0].detach().clone())
    )
    with torch.no_grad():
        model(x, ch_names=ch_names)
    handle.remove()
    return seen["tokens"]


def _labram_numbered_time_slots(n_slots=16, emb_dim=200):
    """A time embedding whose slot ``i`` holds the value ``i`` in every dimension."""
    slots = torch.arange(n_slots, dtype=torch.float32).view(1, n_slots, 1)
    return slots.expand(1, n_slots, emb_dim).clone()


@pytest.mark.parametrize("n_times", [600, 800, 3000])
def test_labram_uses_pretrained_time_slots_at_any_window(n_times):
    # The original LaBraM keeps 16 absolute time slots whatever the window
    # (``time_embed`` in modeling_finetune.py), and a window of P patches adds
    # slot p to every token of patch p. The released weights hold those 16
    # slots (saved at 15 s), so they must load into a model built for any
    # window, and the slots it uses must be the first P ones.
    names = list(LABRAM_CHANNEL_ORDER[:3])
    released = Labram(n_chans=3, n_times=3000, n_outputs=0).state_dict()
    released["temporal_embedding"] = _labram_numbered_time_slots()
    model = Labram(n_chans=3, n_times=n_times, n_outputs=0).eval()

    model.load_state_dict(released)

    x = torch.randn(2, 3, n_times)
    with_slots = _labram_embedded_tokens(model, x, names)
    with torch.no_grad():
        model.temporal_embedding.zero_()
    without_slots = _labram_embedded_tokens(model, x, names)
    added = with_slots - without_slots
    n_patches = n_times // 200
    # Tokens are channel-major: (channel, patch) -> channel * n_patches + patch.
    per_patch = added[:, 1:].reshape(2, 3, n_patches, 200)
    expected = torch.arange(n_patches, dtype=torch.float32).view(1, 1, n_patches, 1)
    assert torch.allclose(per_patch, expected.expand_as(per_patch), atol=1e-5)
    assert torch.equal(added[:, 0], torch.zeros(2, 200))  # [CLS] has no time slot


def test_labram_long_window_keeps_pretrained_time_slots():
    # A window longer than the 16 released slots still builds; the first 16
    # slots come from the checkpoint, the others keep their initialization,
    # and the user is told that some of the slots in use are not pretrained.
    model = Labram(n_chans=3, n_times=6000, n_outputs=0)  # 30 patches
    own = model.temporal_embedding.detach().clone()
    released = Labram(n_chans=3, n_times=3000, n_outputs=0).state_dict()
    released["temporal_embedding"] = _labram_numbered_time_slots()

    with pytest.warns(UserWarning, match="time slots"):
        model.load_state_dict(released)

    loaded = model.temporal_embedding.detach()
    assert torch.equal(loaded[:, :16], _labram_numbered_time_slots())
    assert torch.equal(loaded[:, 16:30], own[:, 16:30])


def test_labram_loads_time_embedding_saved_with_one_slot_per_patch():
    # braindecode <= 1.8 sized the time embedding to the window (patches + 1
    # slots). Such checkpoints must keep loading and give the same outputs.
    names = list(LABRAM_CHANNEL_ORDER[:3])
    saved = Labram(n_chans=3, n_times=800, n_outputs=2).eval()
    state = saved.state_dict()
    state["temporal_embedding"] = state["temporal_embedding"][:, :5]
    x = torch.randn(2, 3, 800)
    with torch.no_grad():
        expected = saved(x, ch_names=names)

    reloaded = Labram(n_chans=3, n_times=800, n_outputs=2).eval()
    reloaded.load_state_dict(state)

    with torch.no_grad():
        assert torch.allclose(reloaded(x, ch_names=names), expected, atol=1e-6)


def test_labram_default_readout_is_mean_of_patch_tokens():
    # The original LaBraM fine-tunes on LayerNorm(mean of the patch tokens)
    # (use_mean_pooling=True in modeling_finetune.py and in
    # run_class_finetuning.py); its pretraining loss never uses [CLS].
    names = list(LABRAM_CHANNEL_ORDER[:3])
    model = Labram(n_chans=3, n_times=800, n_outputs=0).eval()
    seen = {}
    model.blocks[-1].register_forward_hook(
        lambda _module, _inputs, output: seen.update(tokens=output)
    )
    x = torch.randn(2, 3, 800)

    with torch.no_grad():
        features = model(x, ch_names=names)

    patch_mean = seen["tokens"][:, 1:].mean(1)
    expected = torch.nn.functional.layer_norm(patch_mean, (200,), eps=1e-6)
    assert torch.allclose(features, expected, atol=1e-5)


@pytest.mark.parametrize("with_token_norm", [True, False])
def test_labram_mean_pooling_loads_pretraining_checkpoint(with_token_norm):
    # The released weights come from pretraining: they hold the per-token
    # final ``norm`` and no pooling ``fc_norm``. The original fine-tuning script
    # leaves ``norm`` unused and starts ``fc_norm`` from its initialization.
    # A strict load must do the same, also when the state dict was already
    # filtered to the keys the model has (so without ``norm``).
    pretraining = Labram(
        n_chans=3, n_times=800, n_outputs=0, use_mean_pooling=False
    ).state_dict()
    if not with_token_norm:
        pretraining = {
            k: v for k, v in pretraining.items() if not k.startswith("norm.")
        }
    model = Labram(n_chans=3, n_times=800, n_outputs=0, use_mean_pooling=True)

    model.load_state_dict(pretraining)

    assert torch.equal(model.fc_norm.weight, torch.ones(200))
    assert torch.equal(model.fc_norm.bias, torch.zeros(200))
    assert torch.equal(model.cls_token, pretraining["cls_token"])


def test_labram_mean_pooling_rejects_cls_finetuned_checkpoint():
    # A checkpoint fine-tuned with the [CLS] readout has a head trained on
    # [CLS]; it must not load silently into a mean-pooling model.
    finetuned = Labram(
        n_chans=3, n_times=800, n_outputs=2, use_mean_pooling=False
    ).state_dict()
    model = Labram(n_chans=3, n_times=800, n_outputs=2, use_mean_pooling=True)

    with pytest.raises(RuntimeError, match="fc_norm"):
        model.load_state_dict(finetuned)


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


def test_labram_neural_decoder_temporal_embeddings_match_time_patches(
    chs_info, n_times
):
    """Decoder mode adds one temporal embedding per temporal patch token."""
    model = Labram(
        n_times=n_times,
        chs_info=chs_info,
        n_outputs=4,
        patch_size=200,
        embed_dim=4,
        conv_in_channels=8,
        num_layers=0,
        num_heads=1,
        use_abs_pos_emb=False,
        # The [CLS] readout ends in ``norm``, replaced by Identity below, so
        # ``return_all_tokens`` gives the tokens as they enter the readout.
        use_mean_pooling=False,
        neural_tokenizer=False,
    )
    batch_size = 2
    x = torch.zeros(batch_size, len(chs_info), n_times)
    input_chans = torch.arange(len(LABRAM_CHANNEL_ORDER) + 1)
    model.norm = nn.Identity()
    model.pos_drop = nn.Identity()

    with torch.no_grad():
        model.cls_token.zero_()
        model.patch_embed[0].proj.weight.zero_()
        model.patch_embed[0].proj.bias.zero_()
        model.temporal_embedding.zero_()
        expected_time_embed = torch.arange(
            1,
            model.patch_embed[0].n_patchs * model.embed_dim + 1,
            dtype=model.temporal_embedding.dtype,
        ).reshape(1, model.patch_embed[0].n_patchs, model.embed_dim)
        model.temporal_embedding[:, 1:, :] = expected_time_embed

    features = model.forward_features(
        x, input_chans=input_chans, return_all_tokens=True
    )

    assert features.shape == (
        batch_size,
        model.patch_embed[0].n_patchs + 1,
        model.embed_dim,
    )
    assert torch.equal(features[:, 0], torch.zeros_like(features[:, 0]))
    assert torch.equal(
        features[:, 1:],
        expected_time_embed.expand(batch_size, -1, -1),
    )


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


def test_labram_forward_return_flags_remain_positional(chs_info, n_outputs, n_chans):
    """Back-compat: return_* flags can still be passed positionally."""
    model = _small_labram_for_ch_names(chs_info, n_outputs)
    model.eval()
    x = torch.randn(1, n_chans, 400)

    with torch.no_grad():
        out_default = model(x)
        # Positional: return_patch_tokens=False, return_all_tokens=True.
        # ch_names comes after the return flags, so this triggers the all-tokens path
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


def test_zuna_pads_non_divisible_n_times_by_default():
    """The default ``on_non_divisible="pad"`` accepts any window (with a warning);
    ``"error"`` restores the strict behavior."""
    with pytest.warns(UserWarning, match="not divisible"):
        model = ZUNA(chs_info=_zuna_chs_info(), n_times=1000, **_ZUNA_SMALL)
    out = model(torch.randn(1, len(_zuna_chs_info()), 1000))
    assert torch.isfinite(out).all()
    with pytest.raises(ValueError, match="divisible"):
        ZUNA(
            chs_info=_zuna_chs_info(),
            n_times=1000,
            on_non_divisible="error",
            **_ZUNA_SMALL,
        )


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
    model = LUNA(
        n_outputs=2, n_chans=22, n_times=1000, embed_dim=64, num_queries=4, depth=8
    )
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


def test_luna_variants_different_channel_counts(luna_base_config):
    """Test LUNA handles different channel counts (the variants differ only in
    width and depth; test_luna_variants_output_consistency forwards each)."""
    for n_chans in [1, 4, 8, 16, 32, 64]:
        luna_base_config["n_chans"] = n_chans
        model = LUNA(**luna_base_config)
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


def test_codebrain_trains_after_inference_mode():
    # The lazily initialised kernel norm must not become an inference tensor.
    model = CodeBrain(n_chans=2, n_outputs=2, n_times=400).eval()
    x = torch.randn(2, 2, 400)
    with torch.inference_mode():
        model(x)
    model.train()
    model(x).sum().backward()
    assert all(not b.is_inference() for b in model.buffers())


# ==============================================================================
# Tests for SleepFM and SleepFMStager: masks and the release's two-stage
# pipeline (shapes, features and compilation are in the shared suites)
# ==============================================================================


def _small(cls, **kwargs):
    """Reduced model: 20 patches of 64 samples; stager chunks of 8, 8 and 4."""
    config = dict(n_chans=3, n_times=1280, n_outputs=5, sfreq=128.0, patch_size=64)
    config |= dict(embed_dim=16, drop_prob=0.0, max_seq_length=32)
    if cls is SleepFM:
        config |= dict(num_heads=4, num_layers=1, pooling_heads=4)
    else:
        config |= dict(channel_modalities=["A", "A", "B"], encoder_chunk_patches=8)
        config |= dict(encoder_num_heads=4, encoder_num_layers=1)
        config |= dict(encoder_pooling_heads=4, staging_pooling_heads=4)
    return cls(**(config | kwargs))


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize(
    "cls,masked",
    [(SleepFM, "channel"), (SleepFMStager, "channel"), (SleepFMStager, "patch")],
)
def test_sleepfm_masked_input_is_ignored(cls, masked, training):
    """Masked channels or patches reach neither the output nor BatchNorm."""
    model = _small(cls).train(training)
    x = torch.randn(2, 3, 1280)
    corrupted = x.clone()
    if masked == "channel":
        mask = torch.tensor([[False, False, True], [False, True, False]])
        corrupted[mask] = 1e3
        kwargs = {"channel_mask": mask}
    else:
        mask = torch.zeros(2, 20, dtype=torch.bool)
        mask[1, 12:] = True  # ends inside a chunk
        corrupted[1, :, 12 * 64 :] = 1e3
        kwargs = {"temporal_mask": mask}
    runs = []
    for signal in (x, corrupted):
        net = copy.deepcopy(model)
        stats = [v for k, v in net.state_dict().items() if "running" in k]
        runs.append((net(signal, **kwargs), stats))
    torch.testing.assert_close(runs[0], runs[1])


@pytest.mark.parametrize("n_valid", [16, 12], ids=["chunk_aligned", "inside_chunk"])
def test_sleepfm_stager_matches_the_two_stage_pipeline(n_valid):
    """Stager = SleepFM.encode per modality and chunk, then the staging head."""
    encoder = _small(SleepFM, max_seq_length=128).eval()
    stager = _small(SleepFMStager).eval()
    shared = {
        k: v
        for k, v in encoder.state_dict().items()
        if k in stager.state_dict() and not k.startswith("final_layer.")
    }
    stager.load_state_dict(shared, strict=False)
    x = torch.randn(2, 3, 1280)
    channel_mask = torch.tensor([[False, True, False], [False, False, False]])
    temporal_mask = torch.zeros(2, 20, dtype=torch.bool)
    temporal_mask[1, n_valid:] = True

    def embed(chans):
        out = torch.zeros(2, 20, 16)
        for i, stop in enumerate((20, n_valid)):
            for start in range(0, stop, 8):
                end = min(start + 8, stop)
                signal = x[i : i + 1, chans, start * 64 : end * 64]
                out[i, start:end] = encoder.encode(
                    signal, channel_mask[i : i + 1, chans]
                )[1][0]
        return out

    with torch.no_grad():
        embeddings = torch.stack(
            [embed([0, 1]), embed([2])] + [torch.zeros(2, 20, 16)] * 2, 1
        )
        modality_mask = torch.tensor([[False, False, True, True]] * 2)
        features = stager.staging_head(embeddings, modality_mask, temporal_mask)
        expected = stager.final_layer(features).transpose(1, 2)
        output = stager(x, channel_mask, temporal_mask=temporal_mask)
        permuted = _small(SleepFMStager, channel_modalities=["B", "A", "A"]).eval()
        permuted.load_state_dict(stager.state_dict())
        torch.testing.assert_close(permuted(x[:, [2, 0, 1]]), stager(x))

    assert output.shape == (2, 5, 20)
    torch.testing.assert_close(output[0], expected[0])
    torch.testing.assert_close(output[1, :, :n_valid], expected[1, :, :n_valid])


@pytest.mark.skipif(not HAS_SAFETENSORS, reason="safetensors is required")
def test_sleepfm_stager_completes_a_head_only_checkpoint(tmp_path):
    """Older stager mirror revisions hold the tokenizer and head only."""
    from safetensors.torch import save_file

    stager, encoder = _small(SleepFMStager), _small(SleepFM, max_seq_length=128)
    stager.save_pretrained(tmp_path / "stager")
    encoder.save_pretrained(tmp_path / "encoder")
    head = ("patch_embedding.", "staging_head.", "final_layer.")
    head_only = {k: v for k, v in stager.state_dict().items() if k.startswith(head)}
    save_file(head_only, tmp_path / "stager" / "model.safetensors")

    loaded = SleepFMStager.from_pretrained(
        tmp_path / "stager", encoder_model_name_or_path=tmp_path / "encoder"
    )
    expected = encoder.state_dict() | head_only
    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(value, expected[key], msg=key)


@pytest.mark.network
@pytest.mark.huggingface
def test_sleepfm_pretrained_loads():
    kwargs = dict(n_chans=3, n_times=1280, n_outputs=5, sfreq=128.0)
    encoder = SleepFM.from_pretrained(**kwargs)
    kwargs["channel_modalities"] = ["BAS", "BAS", "EKG"]
    stager, reference = SleepFMStager.from_pretrained(**kwargs), SleepFMStager(**kwargs)
    with torch.no_grad():
        features = encoder.eval()(torch.randn(2, 3, 1280), return_features=True)
        assert features["features"].shape == (2, 128)
        assert stager.eval()(torch.randn(2, 3, 1280)).shape == (2, 5, 2)
    for name in (
        "patch_embedding.tokenizer.0.weight",
        "transformer_encoder.layers.5.linear2.weight",
        "staging_head.lstm.weight_ih_l0",
        "final_layer.weight",
    ):
        assert not torch.equal(
            stager.get_parameter(name), reference.get_parameter(name)
        )


# ==============================================================================
# Tests for AXON Model
# ==============================================================================


@pytest.mark.network
@pytest.mark.huggingface
def test_axon_pretrained_loads():
    chs_info = [{"ch_name": n, "kind": "eeg"} for n in ["Fz", "C3", "Cz", "C4", "Pz"]]
    model = AXON.from_pretrained(
        "MannasAI/axon-eeg", chs_info=chs_info, n_outputs=2, n_times=800
    )
    out = model(torch.randn(2, 5, 800))
    assert out.shape == (2, 2)


_AXON_SMALL = dict(embed_dim=64, depth=2, num_heads=4)
_AXON_NAMES = ["Fp1", "Fp2", "F3", "F4", "C3", "Cz", "C4", "P3", "P4", "O1", "O2"]


def _axon_chs(names=_AXON_NAMES):
    info = mne.create_info(names, sfreq=200.0, ch_types="eeg")
    info.set_montage(resolve_montage_name("standard_1005"), match_case=False)
    return info["chs"]


def _axon_model(chs_info, **kw):
    torch.manual_seed(0)
    return AXON(chs_info=chs_info, n_outputs=3, sfreq=200.0, **{**_AXON_SMALL, **kw}).eval()


def test_axon_channel_order_does_not_matter():
    """Electrodes are identified by position, so permuting channels together
    with chs_info must leave the pooled embedding unchanged."""
    chs = _axon_chs()
    model = _axon_model(chs)
    perm = torch.randperm(len(_AXON_NAMES))
    permuted = _axon_model([chs[i] for i in perm])
    permuted.load_state_dict(model.state_dict())
    x = torch.randn(2, len(_AXON_NAMES), 1000)
    with torch.no_grad():
        a = model(x, return_features=True)["features"]
        b = permuted(x[:, perm], return_features=True)["features"]
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_axon_positions_from_channel_names():
    """Without 'loc', standard channel names resolve to the same positions."""
    with_loc = _axon_model(_axon_chs())
    without_loc = _axon_model([{"ch_name": n, "kind": "eeg"} for n in _AXON_NAMES])
    torch.testing.assert_close(
        with_loc.encoder.channel_positions, without_loc.encoder.channel_positions
    )


def test_axon_unknown_channel_without_position_raises():
    chs = [{"ch_name": "Fp1", "kind": "eeg"}, {"ch_name": "NOT_A_CHANNEL", "kind": "eeg"}]
    with pytest.raises(ValueError, match="NOT_A_CHANNEL"):
        _axon_model(chs)


def test_axon_weights_load_onto_another_montage():
    """Channel positions are not stored in the weights."""
    source = _axon_model(_axon_chs())
    target = _axon_model(_axon_chs(["C3", "Cz", "C4", "FC3", "CP4"]))
    target.load_state_dict(source.state_dict(), strict=True)
    assert "encoder.channel_positions" not in source.state_dict()


def test_axon_input_unit_does_not_matter():
    """Microvolts and volts (MNE's default) give the same output."""
    model = _axon_model(_axon_chs())
    x_uv = 20.0 * torch.randn(2, len(_AXON_NAMES), 600) + 5.0
    with torch.no_grad():
        torch.testing.assert_close(model(x_uv), model(x_uv * 1e-6), atol=1e-4, rtol=1e-4)


def test_axon_too_short_window_raises():
    with pytest.raises(ValueError, match="patch_size"):
        AXON(chs_info=_axon_chs(), n_outputs=2, n_times=100, **_AXON_SMALL)


def test_axon_warns_on_non_200_hz():
    with pytest.warns(UserWarning, match="200 Hz"):
        AXON(chs_info=_axon_chs(), n_outputs=2, sfreq=250.0, **_AXON_SMALL)


@pytest.fixture
def diver1_model():
    info = mne.create_info([f"A{i}" for i in range(6)], 500.0, "seeg")
    for i, ch in enumerate(info["chs"]):
        ch["loc"][:3] = [0.01 * i, 0.02, -0.03]
    return DIVER1(
        n_outputs=4,
        chs_info=info["chs"],
        n_times=1000,
        sfreq=500.0,
        pooling="mean",
        d_model=64,
        n_layers=2,
    ).eval()


@pytest.mark.parametrize(
    "kind,located,slots", [("ecog", True, [1.0, 0.0]), ("eeg", False, [0.0, -1.0])]
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
        torch.testing.assert_close(metadata[1, :3], torch.tensor([10.0, 20.0, -30.0]))
    else:
        assert torch.isnan(metadata[:, :3]).all()


@pytest.mark.parametrize(
    "kind, slots",
    [("eeg", [0, -1]), ("ecog", [1, 0]), ("seeg", [1, 2]), ("dbs", [1, 2])],
)
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
            model(xa[:, perm], model.default_chan_metadata[perm]),
            first_a,
            atol=1e-5,
            rtol=1e-5,
        )


@pytest.mark.parametrize(
    "metadata,pooling,match",
    [
        (None, "mean", "built for 6 channels but got input with 9"),
        (torch.zeros(3, 5), "mean", r"shape \(9, 5\)"),
        (torch.zeros(9, 4), "mean", r"shape \(9, 5\)"),
        (torch.full((9, 5), 7.0), "mean", "modality column"),
        (
            torch.zeros(9, 5).index_fill_(1, torch.tensor([4]), 3.0),
            "mean",
            "sub-modality",
        ),
        (torch.zeros(9, 5), "flatten", "pooling='flatten'"),
    ],
)
def test_diver1_rejects_incompatible_montage(diver1_model, metadata, pooling, match):
    model = diver1_model
    if pooling == "flatten":
        model = DIVER1(
            n_outputs=4,
            chs_info=model.chs_info,
            n_times=1000,
            sfreq=500.0,
            pooling=pooling,
            d_model=64,
            n_layers=2,
        ).eval()
    with pytest.raises(ValueError, match=match):
        model(torch.randn(1, 9, 1000), metadata)


@pytest.mark.parametrize("n_outputs", [0, 5])
def test_diver1_reset_head_preserves_zero_outputs(diver1_model, n_outputs):
    diver1_model.reset_head(n_outputs)
    assert diver1_model.final_layer.out_features == n_outputs
    assert diver1_model.get_config()["n_outputs"] == n_outputs


# The bfloat16 ``fold`` kernel crashes the Windows CI runners' Python process
# with 0xC000001D (illegal instruction); pytest-xdist then reports a lost
# worker. It happens on master too, so it is the runner/CPU path, not DIVER-1.
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="bfloat16 fold crashes Windows CI runners (0xC000001D, illegal instruction)",
)
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
        chs_info=info["chs"],
        n_outputs=2,
        n_times=2 * patch_size - 1,
        patch_size=patch_size,
        d_model=64,
        n_layers=1,
        pooling="mean",
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
    kwargs = dict(
        chs_info=info["chs"], n_outputs=2, n_times=500, d_model=64, n_layers=1
    )
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


# ==============================================================================
# Tests for MAPA Model
# ==============================================================================


MAPA_SUBJECT_A = ["LA1", "LA2", "LA5", "LB3", "LB7"]
MAPA_SUBJECT_B = ["RH2", "RH3", "RH8", "LT1", "LT2", "LT3", "LT4", "RX5"]


@pytest.fixture
def mapa_model():
    return MAPA(
        n_outputs=4,
        n_chans=len(MAPA_SUBJECT_A),
        n_times=2048,
        sfreq=2048,
        contact_labels=MAPA_SUBJECT_A,
        d_model=64,
    ).eval()


def test_mapa_sensor_indices_reads_array_and_contact_number():
    indices = MAPA.sensor_indices(MAPA_SUBJECT_A, ["ctx-lh-insula"] + [None] * 4)
    assert indices.tolist() == [
        [0, 1, 7],
        [0, 2, 74],
        [0, 5, 74],
        [1, 3, 74],
        [1, 7, 74],
    ]


# The region table as released, frozen here only to prove the MNE-derived
# helper reproduces it exactly; this tuple is test-only evidence, not a
# definition braindecode's source code owns (see test below and
# ``dkt_region_slots`` in ``braindecode/models/util.py``).
_RELEASED_MAPA_DKT_REGIONS = (
    "ctx-lh-caudalanteriorcingulate",
    "ctx-lh-caudalmiddlefrontal",
    "ctx-lh-cuneus",
    "ctx-lh-entorhinal",
    "ctx-lh-fusiform",
    "ctx-lh-inferiorparietal",
    "ctx-lh-inferiortemporal",
    "ctx-lh-insula",
    "ctx-lh-isthmuscingulate",
    "ctx-lh-lateraloccipital",
    "ctx-lh-lateralorbitofrontal",
    "ctx-lh-lingual",
    "ctx-lh-medialorbitofrontal",
    "ctx-lh-middletemporal",
    "ctx-lh-paracentral",
    "ctx-lh-parahippocampal",
    "ctx-lh-parsopercularis",
    "ctx-lh-parsorbitalis",
    "ctx-lh-parstriangularis",
    "ctx-lh-pericalcarine",
    "ctx-lh-postcentral",
    "ctx-lh-posteriorcingulate",
    "ctx-lh-precentral",
    "ctx-lh-precuneus",
    "ctx-lh-rostralanteriorcingulate",
    "ctx-lh-rostralmiddlefrontal",
    "ctx-lh-superiorfrontal",
    "ctx-lh-superiorparietal",
    "ctx-lh-superiortemporal",
    "ctx-lh-supramarginal",
    "ctx-lh-transversetemporal",
    "ctx-rh-caudalanteriorcingulate",
    "ctx-rh-caudalmiddlefrontal",
    "ctx-rh-cuneus",
    "ctx-rh-entorhinal",
    "ctx-rh-fusiform",
    "ctx-rh-inferiorparietal",
    "ctx-rh-inferiortemporal",
    "ctx-rh-insula",
    "ctx-rh-isthmuscingulate",
    "ctx-rh-lateraloccipital",
    "ctx-rh-lateralorbitofrontal",
    "ctx-rh-lingual",
    "ctx-rh-medialorbitofrontal",
    "ctx-rh-middletemporal",
    "ctx-rh-paracentral",
    "ctx-rh-parahippocampal",
    "ctx-rh-parsopercularis",
    "ctx-rh-parsorbitalis",
    "ctx-rh-parstriangularis",
    "ctx-rh-pericalcarine",
    "ctx-rh-postcentral",
    "ctx-rh-posteriorcingulate",
    "ctx-rh-precentral",
    "ctx-rh-precuneus",
    "ctx-rh-rostralanteriorcingulate",
    "ctx-rh-rostralmiddlefrontal",
    "ctx-rh-superiorfrontal",
    "ctx-rh-superiorparietal",
    "ctx-rh-superiortemporal",
    "ctx-rh-supramarginal",
    "ctx-rh-transversetemporal",
    "Left-Hippocampus",
    "Left-Amygdala",
    "Left-Caudate",
    "Left-Putamen",
    "Left-Pallidum",
    "Left-Thalamus-Proper",
    "Right-Hippocampus",
    "Right-Amygdala",
    "Right-Caudate",
    "Right-Putamen",
    "Right-Pallidum",
    "Right-Thalamus-Proper",
)


def test_mapa_regions_are_freesurfer_labels_sourced_from_util():
    """The region vocabulary is the shared util helper, anchored to MNE's LUT.

    MAPA owns only the reference and the released slot order; the names are
    standard FreeSurfer labels, so each must be a key of MNE's bundled colour
    table (``mne.read_freesurfer_lut``, no download) rather than hard-coded in
    the model file. ``dkt_region_slots`` must still reproduce the exact
    released slot order (``_RELEASED_MAPA_DKT_REGIONS``, frozen before this
    refactor) even though it now derives the vocabulary from MNE's ids
    instead of typing it out.
    """
    from braindecode.models.mapa import MAPA_DKT_REGIONS
    from braindecode.models.util import dkt_region_slots

    assert MAPA_DKT_REGIONS == dkt_region_slots()
    assert MAPA_DKT_REGIONS == _RELEASED_MAPA_DKT_REGIONS
    assert len(MAPA_DKT_REGIONS) == 74
    lut_names, _ = mne.read_freesurfer_lut()
    assert set(MAPA_DKT_REGIONS) <= set(lut_names)


@pytest.mark.parametrize("n_times", [448, 2048, 4096])
def test_mapa_one_model_reads_another_subject(mapa_model, n_times):
    """A montage and a window the mapa_model was not built for both go through."""
    x = torch.randn(2, len(MAPA_SUBJECT_B), n_times)
    with torch.no_grad():
        y = mapa_model(x, MAPA.sensor_indices(MAPA_SUBJECT_B))
    assert y.shape == (2, 4)


def test_mapa_channel_order_does_not_change_the_output(mapa_model):
    perm = torch.tensor([4, 0, 3, 1, 2])
    x = torch.randn(2, len(MAPA_SUBJECT_A), 2048)
    with torch.no_grad():
        expected = mapa_model(x)
        permuted = mapa_model(x[:, perm], mapa_model.default_sensor_indices[perm])
    torch.testing.assert_close(expected, permuted, atol=1e-5, rtol=1e-5)


def test_mapa_switching_subjects_does_not_leak_between_calls(mapa_model):
    """The cached token layout must not survive a change of montage."""
    xa = torch.randn(2, len(MAPA_SUBJECT_A), 2048)
    xb = torch.randn(2, len(MAPA_SUBJECT_B), 2048)
    indices_b = MAPA.sensor_indices(MAPA_SUBJECT_B)
    with torch.no_grad():
        first_a, first_b = mapa_model(xa), mapa_model(xb, indices_b)
        again_b, again_a = mapa_model(xb, indices_b), mapa_model(xa)
    torch.testing.assert_close(first_a, again_a)
    torch.testing.assert_close(first_b, again_b)


def test_mapa_flatten_pooling_stays_tied_to_its_montage():
    mapa_model = MAPA(
        n_outputs=4,
        n_chans=len(MAPA_SUBJECT_A),
        n_times=2048,
        sfreq=2048,
        contact_labels=MAPA_SUBJECT_A,
        d_model=64,
        pooling="flatten",
    ).eval()
    with pytest.raises(ValueError, match="pooling='flatten'"):
        mapa_model(torch.randn(1, len(MAPA_SUBJECT_B), 2048), MAPA.sensor_indices(MAPA_SUBJECT_B))


@pytest.mark.parametrize(
    "sensor_indices,match",
    [
        (None, "got input with 8 channels"),
        (torch.zeros(3, 3, dtype=torch.long), r"shape \(8, 3\)"),
        (torch.zeros(8, 3), "must hold integers"),
        (torch.full((8, 3), 75), "region slots below 75"),
        (torch.full((8, 3), -1), "must be non-negative"),
    ],
)
def test_mapa_bad_sensor_indices_are_rejected(mapa_model, sensor_indices, match):
    with pytest.raises(ValueError, match=match):
        mapa_model(torch.randn(1, len(MAPA_SUBJECT_B), 2048), sensor_indices)


def test_mapa_window_shorter_than_one_slow_token_is_rejected(mapa_model):
    with pytest.raises(ValueError, match="at least 448 samples"):
        mapa_model(torch.randn(1, len(MAPA_SUBJECT_A), 256))


def _mapa_session_model(**kwargs):
    return MAPA(
        n_outputs=4,
        n_chans=len(MAPA_SUBJECT_A),
        n_times=32,
        sfreq=32,
        contact_labels=MAPA_SUBJECT_A,
        d_model=64,
        normalization="session",
        **kwargs,
    ).eval()


def test_mapa_session_normalization_matches_window_normalization_on_its_bands(mapa_model):
    """Handed the bands window normalization computes, session mode is identical."""
    session = _mapa_session_model()
    session.load_state_dict(mapa_model.state_dict())
    x = torch.randn(2, len(MAPA_SUBJECT_A), 2048)
    frames = torch.cat(mapa_model.frontend._stft_bands(x), dim=2)
    with torch.no_grad():
        torch.testing.assert_close(session(frames), mapa_model(x))


def test_mapa_session_normalization_input_and_output_shape():
    model = _mapa_session_model()
    assert model.input_shape == (1, len(MAPA_SUBJECT_A), 20, 32)
    assert model.get_output_shape() == (1, 4)


def test_mapa_window_normalization_input_shape_is_raw(mapa_model):
    assert mapa_model.input_shape == (1, mapa_model.n_chans, mapa_model.n_times)


def test_mapa_session_normalization_reads_another_subject():
    frames = torch.randn(2, len(MAPA_SUBJECT_B), 20, 64)
    with torch.no_grad():
        y = _mapa_session_model()(frames, MAPA.sensor_indices(MAPA_SUBJECT_B))
    assert y.shape == (2, 4)


@pytest.mark.parametrize(
    "shape,match",
    [
        ((1, 5, 2048), "takes a spectrogram"),
        ((1, 5, 19, 32), "takes a spectrogram"),
        ((1, 5, 20, 4), "at least 8 frames"),
    ],
)
def test_mapa_session_normalization_rejects_bad_input(shape, match):
    with pytest.raises(ValueError, match=match):
        _mapa_session_model()(torch.randn(shape))


def test_mapa_raw_normalization_rejects_a_spectrogram(mapa_model):
    with pytest.raises(ValueError, match="normalization='session'"):
        mapa_model(torch.randn(1, len(MAPA_SUBJECT_A), 20, 32))


def test_mapa_token_layout_tracks_the_montage(mapa_model):
    indices = MAPA.sensor_indices(MAPA_SUBJECT_A)
    first = mapa_model._token_layout(indices, mapa_model.n_frames)
    indices[0, 2] = 3
    changed = mapa_model._token_layout(indices, mapa_model.n_frames)
    assert changed is not first
    assert (changed[2] == 3).any()  # token_region


@pytest.mark.parametrize("kwargs", [{}, {"region_embed": False, "deep_sup": False}])
def test_mapa_scripts_on_both_montage_paths(kwargs):
    """The scripted model reads the construction-time and a foreign montage."""
    model = MAPA(
        n_outputs=4,
        n_chans=len(MAPA_SUBJECT_A),
        n_times=2048,
        sfreq=2048,
        contact_labels=MAPA_SUBJECT_A,
        d_model=64,
        **kwargs,
    ).eval()
    scripted = torch.jit.script(model)
    xa = torch.randn(2, len(MAPA_SUBJECT_A), 2048)
    xb = torch.randn(2, len(MAPA_SUBJECT_B), 4096)
    indices_b = MAPA.sensor_indices(MAPA_SUBJECT_B, ["Left-Hippocampus"] + [None] * 7)
    with torch.no_grad():
        torch.testing.assert_close(scripted(xa), model(xa))
        torch.testing.assert_close(scripted(xb, indices_b), model(xb, indices_b))


def _mapa_reference_windows():
    """Two 1 s windows at 2048 Hz of four amplitude-modulated multi-tone channels."""
    t = torch.arange(2048, dtype=torch.float64) / 2048
    freqs = torch.tensor([3.0, 11.0, 23.0, 47.0, 95.0, 140.0], dtype=torch.float64)
    rates = 0.75 * torch.arange(1, 7, dtype=torch.float64)
    windows = []
    for sample in range(2):
        channels = []
        for channel in range(4):
            phase = 0.7 * channel + 1.3 * sample
            carrier = torch.sin(2 * torch.pi * freqs[:, None] * t + phase)
            envelope = 1 + 0.8 * torch.sin(2 * torch.pi * rates[:, None] * t + phase)
            channels.append((carrier * envelope).sum(0))
        windows.append(torch.stack(channels))
    return torch.stack(windows).float()


@pytest.mark.network
@pytest.mark.huggingface
def test_mapa_released_checkpoint_reproduces_the_reference_features():
    """The re-hosted mapa_vits384 loads and gives the authors' features.

    The expected values were computed with the authors' code (bentang18/MAPA at
    bf2b49e) on the same windows: its STFT and robust z-score fitted on each
    window (the default ``normalization="window"``; the authors fit it on the
    whole session), then ``MapaEncoder.from_checkpoint`` and the mean over every
    token of the four normed deep-supervision taps, which is what
    ``return_features`` pools. CI does not pass ``--run-network`` to the unit
    tests; run it with ``pytest -k mapa_released --run-network``.
    """
    pytest.importorskip("huggingface_hub")
    try:
        model = MAPA.from_pretrained(
            "braindecode/mapa-pretrained",
            n_chans=4,
            contact_labels=["LA1", "LA2", "LA4", "LB1"],
            regions=[
                "ctx-lh-superiortemporal",
                "ctx-lh-superiortemporal",
                "Left-Hippocampus",
                None,
            ],
            strict=True,
        ).eval()
    except (URLError, OSError) as err:
        pytest.skip(f"Could not download the MAPA checkpoint: {err}")

    with torch.no_grad():
        features = model(_mapa_reference_windows(), return_features=True)["features"]
    # The first four dimensions of each of the four taps (blocks 3, 6, 9, 12).
    expected = torch.tensor(
        [
            [-0.097404, -0.035431, -0.036317, -0.00045]
            + [0.000488, -0.010021, -0.031613, -0.012514]
            + [-0.000287, 0.000164, 0.001316, 0.00072]
            + [0.009574, -0.119245, -0.004626, 0.019742],
            [-0.108159, 0.016018, -0.037283, -0.000448]
            + [0.000275, -0.013464, -0.029668, -0.011087]
            + [-0.000288, 0.000152, 0.001267, 0.000396]
            + [-0.000122, -0.127825, -0.006181, 0.038055],
        ]
    )
    taps = features.unflatten(1, (4, -1))[..., :4].flatten(1)
    torch.testing.assert_close(taps, expected, rtol=0, atol=1e-4)
    torch.testing.assert_close(
        features.norm(dim=1), torch.tensor([3.91526, 3.92401]), rtol=1e-4, atol=0
    )


# ==============================================================================
# Tests for PopulationTransformer (PopT)
# ==============================================================================

_POPT_SMALL = dict(hidden_dim=32, ffn_dim=64, n_layers=1, n_heads=4)
# Brain Treebank-like absolute (left, inferior, posterior) integer indices.
_POPT_LIP = [[57, 52, 62], [196, 155, 191], [120, 80, 100]]


def _popt_chs_info(positions):
    info = mne.create_info([f"E{i}" for i in range(len(positions))], 2048.0, "seeg")
    for ch, xyz in zip(info["chs"], positions):
        ch["loc"][:3] = xyz
    return info["chs"]


def test_popt_defaults_are_the_released_config():
    """A default PopT has the shapes of the released ``popt_brainbert_stft``."""
    model = PopulationTransformer(n_chans=4, n_outputs=2, n_times=768)
    layer = model.transformer_encoder.layers[0]
    assert model.hidden_dim == 512
    assert len(model.transformer_encoder.layers) == 6
    assert layer.self_attn.num_heads == 8
    assert layer.linear1.out_features == 2048


@pytest.mark.parametrize(
    "coord_units, scale, shift_coords, expected",
    [
        ("m", 1e-3, False, _POPT_LIP),
        ("raw", 1.0, False, _POPT_LIP),
        ("m", 1e-3, True, [[0, 0, 0], [139, 103, 129], [63, 28, 38]]),
    ],
    ids=["metres", "raw", "opt-in-shift"],
)
def test_popt_chs_info_coords_are_absolute_by_default(
    coord_units, scale, shift_coords, expected
):
    """Upstream feeds absolute indices; the per-axis shift is opt-in."""
    chs_info = _popt_chs_info([[v * scale for v in xyz] for xyz in _POPT_LIP])
    model = PopulationTransformer(
        n_chans=3,
        n_outputs=1,
        n_times=16,
        chs_info=chs_info,
        coord_units=coord_units,
        shift_coords=shift_coords,
        **_POPT_SMALL,
    )
    assert model.electrode_coords.tolist() == expected


def test_popt_out_of_range_coords_warn_and_clamp():
    chs_info = _popt_chs_info([[-0.02, 0.01, 0.03], [0.04, -0.05, 0.06]])
    with pytest.warns(UserWarning, match="clamped"):
        model = PopulationTransformer(
            n_chans=2, n_outputs=1, n_times=16, chs_info=chs_info, **_POPT_SMALL
        )
    assert model.electrode_coords.tolist() == [[0, 10, 30], [40, 0, 60]]


def test_popt_head_is_upstream_linear_and_loads_legacy_head():
    """The head is upstream's single linear layer on the CLS token.

    The ``braindecode/popt-pretrained`` mirror stores an earlier
    LayerNorm + Linear head (``final_layer.norm``/``final_layer.fc``); it still
    loads strictly, onto the linear layer.
    """
    model = PopulationTransformer(n_chans=3, n_outputs=2, n_times=16, **_POPT_SMALL)
    model.eval()
    assert type(model.final_layer) is torch.nn.Linear
    x = torch.randn(2, 3, 16)
    cls_token = model(x, return_features=True)["cls_token"]
    torch.testing.assert_close(model(x), model.final_layer(cls_token), rtol=0, atol=0)

    legacy = {
        k: v for k, v in model.state_dict().items() if not k.startswith("final_layer")
    }
    fc_weight, fc_bias = torch.randn(2, 32), torch.randn(2)
    legacy.update(
        {
            "final_layer.norm.weight": torch.ones(32),
            "final_layer.norm.bias": torch.zeros(32),
            "final_layer.fc.weight": fc_weight,
            "final_layer.fc.bias": fc_bias,
        }
    )
    reloaded = PopulationTransformer(n_chans=3, n_outputs=2, n_times=16, **_POPT_SMALL)
    reloaded.load_state_dict(legacy, strict=True)
    torch.testing.assert_close(reloaded.final_layer.weight, fc_weight)
    torch.testing.assert_close(reloaded.final_layer.bias, fc_bias)


_TEN_TWENTY = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "O2",
]


def _patch_models():
    return [
        (Labram, dict(n_chans=19, n_times=800, sfreq=200), 200),
        (Labram, dict(n_chans=19, n_times=800, sfreq=200, neural_tokenizer=False), 200),
        (CBraMod, dict(n_chans=19, n_times=800, sfreq=200), 200),
        (LUNA, dict(chs_info=_zuna_chs_info(), n_times=800, sfreq=200), 40),
        (ZUNA, dict(chs_info=_zuna_chs_info(), n_times=1024, **_ZUNA_SMALL), 32),
    ]


@pytest.mark.parametrize("cls,kwargs,patch_size", _patch_models())
def test_non_divisible_window_padded_by_default(cls, kwargs, patch_size):
    """A window that is not a multiple of patch_size pads (warning) and only
    raises with on_non_divisible="error"; the tokenizer adds no parameters."""
    kwargs = dict(kwargs, n_times=kwargs["n_times"] + 37, n_outputs=2)
    with pytest.warns(UserWarning, match="not divisible"):
        model = cls(**kwargs).eval()
    n_chans = kwargs.get("n_chans") or len(kwargs["chs_info"])
    x = torch.randn(1, n_chans, kwargs["n_times"])
    with torch.no_grad():
        out = model(x, ch_names=_TEN_TWENTY) if cls is Labram else model(x)
    assert torch.isfinite(out).all()
    assert not any("tokenizer" in k for k in model.state_dict())
    with pytest.raises(ValueError, match="divisible"):
        cls(**kwargs, on_non_divisible="error")


def test_cbramod_head_is_concrete_when_geometry_is_derived(tmp_path):
    """chs_info / input_window_seconds define the geometry as much as n_chans /
    n_times do: the head must be a real Linear, so the model saves and loads
    (a LazyLinear cannot be serialized before a forward pass)."""
    pytest.importorskip("huggingface_hub")
    chs = _zuna_chs_info()
    model = CBraMod(
        chs_info=chs, input_window_seconds=4.0, sfreq=200, n_outputs=2, n_layer=1
    )
    assert type(model.final_layer[1]) is nn.Linear  # LazyLinear subclasses Linear
    model.save_pretrained(tmp_path)
    loaded = CBraMod.from_pretrained(tmp_path)
    x = torch.randn(1, len(chs), 800)
    assert torch.allclose(model.eval()(x), loaded.eval()(x), atol=1e-5)
    # unknown geometry still falls back to a lazy head
    assert isinstance(CBraMod(n_outputs=2, n_layer=1).final_layer[1], nn.LazyLinear)
    # n_times alone (no channels) still reaches the tokenizer's divisibility check
    with pytest.raises(ValueError, match="divisible"):
        CBraMod(n_times=1001, n_outputs=2, n_layer=1, on_non_divisible="error")
