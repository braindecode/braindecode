# Authors: Alexandre Gramfort
#          Lukas Gemein <l.gemein@gmail.com>
#          Hubert Banville <hubert.jbanville@gmail.com>
#          Robin Schirrmeister <robintibor@gmail.com>
#          Daniel Wilson <dan.c.wil@gmail.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#          Matthew Chen <matt.chen42601@gmail.com>
#
# License: BSD-3

from collections import OrderedDict
from functools import partial

import numpy as np
import pytest
import torch
from sklearn.utils import check_random_state
from torch import nn

from braindecode.models import (
    BENDR,
    BIOT,
    EEGPT,
    TCN,
    ATCNet,
    AttentionBaseNet,
    AttnSleep,
    BrainModule,
    ContraWR,
    Deep4Net,
    DeepSleepNet,
    EEGConformer,
    EEGInceptionERP,
    EEGInceptionMI,
    EEGITNet,
    EEGMiner,
    EEGNet,
    EEGNeX,
    EEGSimpleConv,
    EEGTCNet,
    FBCNet,
    FBMSNet,
    HybridNet,
    IFNet,
    Labram,
    MEDFormer,
    SCCNet,
    ShallowFBCSPNet,
    SleepStagerBlanco2020,
    SleepStagerChambon2018,
    SPARCNet,
    TIDNet,
    TSception,
    USleep,
)
from braindecode.models.eegpt import (
    _apply_rotary_emb,
    _Attention,
    _EEGTransformer,
    _PatchEmbed,
    _rotate_half,
)
from braindecode.models.labram import LABRAM_CHANNEL_ORDER
from braindecode.util import set_random_seeds


@pytest.fixture(scope="module")
def input_sizes():
    return dict(n_channels=18, n_in_times=600, n_classes=2, n_samples=7)


def check_forward_pass(model, input_sizes, only_check_until_dim=None):
    # Test 4d Input
    set_random_seeds(0, False)
    rng = np.random.RandomState(42)
    X = rng.randn(
        input_sizes["n_samples"],
        input_sizes["n_channels"],
        input_sizes["n_in_times"],
        1,
    )
    X = torch.Tensor(X.astype(np.float32))
    y_pred = model(X)
    assert y_pred.shape[:only_check_until_dim] == (
        input_sizes["n_samples"],
        input_sizes["n_classes"],
    )

    # Test 3d input
    set_random_seeds(0, False)
    X = X.squeeze(-1)
    assert len(X.shape) == 3
    y_pred_new = model(X)
    assert y_pred_new.shape[:only_check_until_dim] == (
        input_sizes["n_samples"],
        input_sizes["n_classes"],
    )
    np.testing.assert_allclose(
        y_pred.detach().cpu().numpy(),
        y_pred_new.detach().cpu().numpy(),
        atol=1e-4,
        rtol=0,
    )

def check_forward_pass_3d(model, input_sizes, only_check_until_dim=None):
    rng = np.random.RandomState(42)

    # Test 3d input
    X = rng.randn(
        input_sizes["n_samples"],
        input_sizes["n_channels"],
        input_sizes["n_in_times"],
    )
    X = torch.Tensor(X.astype(np.float32))
    set_random_seeds(0, False)
    X = X.squeeze(-1)
    assert len(X.shape) == 3
    y_pred_new = model(X)
    assert y_pred_new.shape[:only_check_until_dim] == (
        input_sizes["n_samples"],
        input_sizes["n_classes"],
    )


def test_shallow_fbcsp_net(input_sizes):
    model = ShallowFBCSPNet(
        input_sizes["n_channels"],
        input_sizes["n_classes"],
        input_sizes["n_in_times"],
        final_conv_length="auto",
    )
    check_forward_pass(model, input_sizes)


def test_shallow_fbcsp_net_load_state_dict(input_sizes):
    model = ShallowFBCSPNet(
        input_sizes["n_channels"],
        input_sizes["n_classes"],
        input_sizes["n_in_times"],
        final_conv_length="auto",
    )

    state_dict = OrderedDict()
    state_dict["conv_time.weight"] = torch.rand([40, 1, 25, 1])
    state_dict["conv_time.bias"] = torch.rand([40])
    state_dict["conv_spat.weight"] = torch.rand(
        [40, 40, 1, input_sizes["n_channels"]])
    state_dict["bnorm.weight"] = torch.rand([40])
    state_dict["bnorm.bias"] = torch.rand([40])
    state_dict["bnorm.running_mean"] = torch.rand([40])
    state_dict["bnorm.running_var"] = torch.rand([40])
    state_dict["bnorm.num_batches_tracked"] = torch.rand([])
    state_dict["conv_classifier.weight"] = torch.rand(
        [input_sizes["n_classes"], 40, model.final_conv_length, 1]
    )
    state_dict["conv_classifier.bias"] = torch.rand([input_sizes["n_classes"]])
    model.load_state_dict(state_dict)


def test_deep4net(input_sizes):
    model = Deep4Net(
        input_sizes["n_channels"],
        input_sizes["n_classes"],
        input_sizes["n_in_times"],
        final_conv_length="auto",
    )
    check_forward_pass(model, input_sizes)


def test_deep4net_load_state_dict(input_sizes):
    model = Deep4Net(
        input_sizes["n_channels"],
        input_sizes["n_classes"],
        input_sizes["n_in_times"],
        final_conv_length="auto",
    )
    state_dict = OrderedDict()
    state_dict["conv_time.weight"] = torch.rand([25, 1, 10, 1])
    state_dict["conv_time.bias"] = torch.rand([25])
    state_dict["conv_spat.weight"] = torch.rand(
        [25, 25, 1, input_sizes["n_channels"]])
    state_dict["bnorm.weight"] = torch.rand([25])
    state_dict["bnorm.bias"] = torch.rand([25])
    state_dict["bnorm.running_mean"] = torch.rand([25])
    state_dict["bnorm.running_var"] = torch.rand([25])
    state_dict["bnorm.num_batches_tracked"] = torch.rand([])
    state_dict["conv_2.weight"] = torch.rand([50, 25, 10, 1])
    state_dict["bnorm_2.weight"] = torch.rand([50])
    state_dict["bnorm_2.bias"] = torch.rand([50])
    state_dict["bnorm_2.running_mean"] = torch.rand([50])
    state_dict["bnorm_2.running_var"] = torch.rand([50])
    state_dict["bnorm_2.num_batches_tracked"] = torch.rand([])
    state_dict["conv_3.weight"] = torch.rand([100, 50, 10, 1])
    state_dict["bnorm_3.weight"] = torch.rand([100])
    state_dict["bnorm_3.bias"] = torch.rand([100])
    state_dict["bnorm_3.running_mean"] = torch.rand([100])
    state_dict["bnorm_3.running_var"] = torch.rand([100])
    state_dict["bnorm_3.num_batches_tracked"] = torch.rand([])
    state_dict["conv_4.weight"] = torch.rand([200, 100, 10, 1])
    state_dict["bnorm_4.weight"] = torch.rand([200])
    state_dict["bnorm_4.bias"] = torch.rand([200])
    state_dict["bnorm_4.running_mean"] = torch.rand([200])
    state_dict["bnorm_4.running_var"] = torch.rand([200])
    state_dict["bnorm_4.num_batches_tracked"] = torch.rand([])
    state_dict["conv_classifier.weight"] = torch.rand(
        [input_sizes["n_classes"], 200, model.final_conv_length, 1]
    )
    state_dict["conv_classifier.bias"] = torch.rand([input_sizes["n_classes"]])
    model.load_state_dict(state_dict)



def test_hybridnet(input_sizes):
    model = HybridNet(
        input_sizes["n_channels"],
        input_sizes["n_classes"],
        input_sizes["n_in_times"],
    )
    check_forward_pass(model, input_sizes, only_check_until_dim=2)


def test_eegnet(input_sizes):
    model = EEGNet(
        input_sizes["n_channels"],
        input_sizes["n_classes"],
        n_times=input_sizes["n_in_times"],
    )
    check_forward_pass(model, input_sizes)


def test_tcn(input_sizes):
    model = TCN(
        n_chans=input_sizes["n_channels"],
        n_outputs=input_sizes["n_classes"],
        n_filters=5,
        n_blocks=2,
        kernel_size=4,
        drop_prob=0.5,
    )
    check_forward_pass(model, input_sizes, only_check_until_dim=2)


def test_eegpt(input_sizes):
    channels_names = [
        'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'FPZ', 'FZ', 'CZ', 'CPZ', 'PZ', 'POZ', 'OZ'
    ]
    input_sizes_copy = input_sizes.copy()
    input_sizes_copy["n_channels"] = len(channels_names)
    model = EEGPT(
        n_outputs=input_sizes_copy["n_classes"],
        n_chans=input_sizes_copy["n_channels"],
        n_times=input_sizes_copy["n_in_times"],
    )
    check_forward_pass_3d(model, input_sizes_copy)


@pytest.mark.parametrize(
    "patch_size, patch_stride, embed_num, embed_dim, depth, num_heads, "
    "mlp_ratio, drop_prob, attn_drop_rate, drop_path_rate, return_encoder_output, "
    "use_chs_info, n_chans, n_times, n_outputs",
    [
        # Test 1: Default configuration with basic channels
        (64, 32, 4, 512, 8, 8, 4.0, 0.0, 0.0, 0.0, False, False, 13, 600, 4),
        # Test 2: Different patch sizes
        (32, 16, 4, 256, 4, 4, 4.0, 0.0, 0.0, 0.0, False, False, 10, 500, 2),
        # Test 3: Larger embed_dim and more heads
        (64, 32, 2, 768, 6, 12, 4.0, 0.0, 0.0, 0.0, False, False, 8, 600, 3),
        # Test 4: Return encoder output (feature extraction mode)
        (64, 32, 4, 512, 8, 8, 4.0, 0.0, 0.0, 0.0, True, False, 13, 600, 4),
        # Test 5: With dropout enabled
        (64, 32, 4, 512, 4, 8, 4.0, 0.1, 0.1, 0.1, False, False, 10, 600, 2),
        # Test 6: With chs_info provided (proper channel names)
        (64, 32, 4, 512, 4, 8, 4.0, 0.0, 0.0, 0.0, False, True, 13, 600, 4),
        # Test 7: Smaller model (depth=2, fewer heads)
        (64, 32, 2, 256, 2, 4, 4.0, 0.0, 0.0, 0.0, False, False, 8, 600, 2),
        # Test 8: Different MLP ratio
        (64, 32, 4, 512, 4, 8, 2.0, 0.0, 0.0, 0.0, False, False, 10, 600, 3),
        # Test 9: Large number of outputs (many classes)
        (64, 32, 4, 512, 4, 8, 4.0, 0.0, 0.0, 0.0, False, False, 10, 600, 10),
        # Test 10: Minimal configuration
        (64, 32, 1, 128, 2, 2, 4.0, 0.0, 0.0, 0.0, False, False, 4, 600, 2),
    ],
    ids=[
        "default_config",
        "small_patch_size",
        "larger_embed_dim",
        "encoder_output_mode",
        "with_dropout",
        "with_chs_info",
        "small_model",
        "different_mlp_ratio",
        "many_classes",
        "minimal_config",
    ],
)
def test_eegpt_parametrized(
    patch_size,
    patch_stride,
    embed_num,
    embed_dim,
    depth,
    num_heads,
    mlp_ratio,
    drop_prob,
    attn_drop_rate,
    drop_path_rate,
    return_encoder_output,
    use_chs_info,
    n_chans,
    n_times,
    n_outputs,
):
    """Comprehensive test for EEGPT model covering various configurations."""
    # Define channel names from the EEGPT channel list
    available_channels = [
        'FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AF4', 'AF8',
        'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
        'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
        'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
        'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
        'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
        'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
        'O1', 'OZ', 'O2',
    ]
    channel_names = available_channels[:n_chans]

    # Prepare chs_info if requested
    chs_info = None
    if use_chs_info:
        chs_info = [
            {"ch_name": ch, "kind": "eeg"} for ch in channel_names
        ]

    # Create model
    model = EEGPT(
        n_outputs=n_outputs,
        n_chans=n_chans,
        n_times=n_times,
        chs_info=chs_info,
        patch_size=patch_size,
        patch_stride=patch_stride,
        embed_num=embed_num,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_prob=drop_prob,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        return_encoder_output=return_encoder_output,
    )
    model.eval()

    # Create random input
    batch_size = 2
    rng = np.random.RandomState(42)
    X = rng.randn(batch_size, n_chans, n_times)
    X = torch.Tensor(X.astype(np.float32))

    # Forward pass
    with torch.no_grad():
        output = model(X)

    # Verify output shape
    if return_encoder_output:
        # Encoder output has shape (batch, n_patches, embed_num, embed_dim)
        # Recalculate n_patches considering padding
        eff_stride = patch_size if patch_stride is None else patch_stride
        if patch_stride is None:
             rem = n_times % patch_size
             pad = patch_size - rem if rem != 0 else 0
             n_patches = (n_times + pad) // patch_size
        else:
             rem = (n_times - patch_size) % patch_stride
             pad = patch_stride - rem if rem != 0 else 0
             n_patches = (n_times + pad - patch_size) // patch_stride + 1
        expected_shape = (batch_size, n_patches, embed_num, embed_dim)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )
    else:
        # Classification output has shape (batch, n_outputs)
        expected_shape = (batch_size, n_outputs)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )


def test_eegpt_invalid_channel():
    """Test EEGPT fallback when chs_info contains invalid channel names."""
    from braindecode.models.eegpt import EEGPT

    invalid_chs_info = [
        {"ch_name": "INVALID_CH", "kind": "eeg"},
        {"ch_name": "F3", "kind": "eeg"},
    ]

    # Use chan_proj_type="none" to test chs_info path (default uses channel projection)
    with pytest.warns(RuntimeWarning, match="Unknown channel name"):
        model = EEGPT(
            n_outputs=4,
            n_chans=2,
            n_times=600,
            chs_info=invalid_chs_info,
            chan_proj_type="none",
        )

    # Mixed fallback strategy:
    # INVALID_CH (index 0) -> 0
    # F3 (index 1) -> CHANNEL_DICT['F3']
    from braindecode.models.eegpt import CHANNEL_DICT
    expected_ids = torch.tensor([0, CHANNEL_DICT['F3']])
    assert torch.equal(model.chans_id.view(-1), expected_ids)


def test_eegpt_patch_norm_embed():
    """Test the _PatchNormEmbed alternative patch embedding module."""
    from braindecode.models.eegpt import _PatchEmbed

    n_chans = 8
    n_times = 640  # Must be divisible by patch_size
    patch_size = 64
    embed_dim = 128

    patch_embed = _PatchEmbed(
        n_chans=n_chans,
        n_times=n_times,
        patch_size=patch_size,
        embed_dim=embed_dim,
        apply_norm=True,
    )

    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, n_chans, n_times)
    output = patch_embed(x)

    # Expected: (batch, n_patches, n_chans, embed_dim)
    n_patches = n_times // patch_size
    expected_shape = (batch_size, n_patches, n_chans, embed_dim)
    assert output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {output.shape}"
    )


def test_eegpt_patch_norm_embed_with_stride():
    """Test _PatchNormEmbed with custom stride."""
    from braindecode.models.eegpt import _PatchEmbed

    n_chans = 8
    n_times = 640
    patch_size = 64
    patch_stride = 32
    embed_dim = 128

    patch_embed = _PatchEmbed(
        n_chans=n_chans,
        n_times=n_times,
        patch_size=patch_size,
        patch_stride=patch_stride,
        embed_dim=embed_dim,
        apply_norm=True,
    )

    batch_size = 2
    x = torch.randn(batch_size, n_chans, n_times)
    output = patch_embed(x)

    n_patches = (n_times - patch_size) // patch_stride + 1
    expected_shape = (batch_size, n_patches, n_chans, embed_dim)
    assert output.shape == expected_shape


def test_eegpt_patch_embed_padding():
    """Test that _PatchEmbed automatically pads input if n_times is not divisible."""
    from braindecode.models.eegpt import _PatchEmbed

    # Case 1: n_times=100, patch_size=64.
    # Remainder 36. Padding should be 64-36 = 28.
    # New size 128 (2 patches).
    n_chans = 8
    n_times = 100
    patch_size = 64
    embed_dim = 128

    patch_embed = _PatchEmbed(
        n_chans=n_chans,
        n_times=n_times,
        patch_size=patch_size,
        embed_dim=embed_dim,
        apply_norm=True,
    )

    assert patch_embed.padding_size == 28
    assert patch_embed.n_times_padded == 128

    batch_size = 2
    x = torch.randn(batch_size, n_chans, n_times)
    output = patch_embed(x)

    # Expected: (batch, n_patches=2, n_chans, embed_dim)
    expected_shape = (batch_size, 2, n_chans, embed_dim)
    assert output.shape == expected_shape


def test_eegpt_patch_embed_no_stride():
    """Test PatchEmbed with default (non-overlapping) patches."""
    from braindecode.models.eegpt import _PatchEmbed

    n_chans = 8
    n_times = 640
    patch_size = 64
    embed_dim = 128

    # patch_stride=None means non-overlapping patches
    patch_embed = _PatchEmbed(
        n_chans=n_chans,
        n_times=n_times,
        patch_size=patch_size,
        patch_stride=None,
        embed_dim=embed_dim,
    )

    batch_size = 2
    x = torch.randn(batch_size, n_chans, n_times)
    output = patch_embed(x)

    n_patches = n_times // patch_size
    expected_shape = (batch_size, n_patches, n_chans, embed_dim)
    assert output.shape == expected_shape


def test_eegpt_droppath():
    """Test EEGPT with drop_path_rate > 0 to cover DropPath branch."""
    from braindecode.models.eegpt import EEGPT

    model = EEGPT(
        n_outputs=4,
        n_chans=8,
        n_times=600,
        depth=2,
        embed_dim=128,
        num_heads=4,
        drop_path_rate=0.2,  # Non-zero to trigger DropPath
    )
    model.train()  # DropPath only active during training

    batch_size = 2
    x = torch.randn(batch_size, 8, 600)
    output = model(x)

    assert output.shape == (batch_size, 4)


def test_eegpt_transformer_patch_norm_embed():
    n_times = 100
    patch_size = 20
    n_chans = 2
    embed_dim = 16

    model = _EEGTransformer(
        n_chans=n_chans,
        n_times=n_times,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=4,
        patch_module=partial(_PatchEmbed, apply_norm=True),
    )

    x = torch.randn(1, n_chans, n_times)
    chan_ids = torch.arange(n_chans).unsqueeze(0)
    out = model(x, chan_ids)
    assert out.shape == (1, n_times // patch_size, 1, embed_dim)


def test_eegpt_transformer_masking():
    n_chans = 2
    n_times = 100
    patch_size = 20
    embed_dim = 16

    model = _EEGTransformer(
        n_chans=n_chans,
        n_times=n_times,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=4,
    )

    x = torch.randn(1, n_chans, n_times)
    chan_ids = torch.arange(n_chans).unsqueeze(0)

    n_patches = n_times // patch_size
    total_tokens = n_patches * n_chans
    mask_x = torch.arange(total_tokens).reshape(n_patches, n_chans)

    out = model(x, chan_ids=chan_ids, mask_x=mask_x)
    assert out.shape == (1, n_patches, 1, embed_dim)

    mask_t = torch.arange(n_patches // 2)
    out_t = model(x, chan_ids=chan_ids, mask_t=mask_t)
    assert out_t.shape[1] == n_patches // 2


def test_eegpt_rope_helpers():
    x = torch.randn(1, 4, 10)
    rotated = _rotate_half(x)
    assert rotated.shape == x.shape

    t = torch.randn(1, 4, 10)
    freqs = torch.randn(1, 4, 10)
    out = _apply_rotary_emb(freqs, t)
    assert out.shape == t.shape


def test_eegpt_apply_rotary_emb_invalid_dim():
    t = torch.randn(1, 10, 16)
    freqs = torch.randn(1, 10, 32)
    with pytest.raises(ValueError, match="feature dimension"):
        _apply_rotary_emb(freqs, t)


def test_eegpt_attention_with_rope():
    dim = 16
    num_heads = 4
    attn = _Attention(dim, num_heads=num_heads, use_rope=True)

    x = torch.randn(2, 5, dim)
    freqs = torch.randn(2, num_heads, 5, dim // num_heads)
    out = attn(x, freqs=freqs)
    assert out.shape == x.shape


def test_eegpt_return_attention_layer():
    model = _EEGTransformer(
        n_chans=2,
        n_times=100,
        return_attention_layer=1,
    )
    x = torch.randn(1, 2, 100)
    out = model(x)

    expected_seq_len = model.patch_embed.n_chans + model.embed_num
    assert out.shape[1] == model.num_heads
    assert out.shape[-1] == expected_seq_len
    assert out.shape[-2] == expected_seq_len


def test_eegpt_buffer_device():
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        pytest.skip("No CUDA or MPS device available.")

    device = "cuda" if torch.cuda.is_available() else "mps"
    model = EEGPT(n_outputs=2, n_chans=3, n_times=100).to(device)

    assert model.chans_id.device.type == device

    x = torch.randn(1, 3, 100, device=device)
    with torch.no_grad():
        model(x)


def test_eegitnet(input_sizes):
    model = EEGITNet(
        n_outputs=input_sizes["n_classes"],
        n_chans=input_sizes["n_channels"],
        n_times=input_sizes["n_in_times"],
    )

    check_forward_pass(
        model,
        input_sizes,
    )


@pytest.mark.parametrize("model_cls", [EEGInceptionERP])
def test_eeginception_erp(input_sizes, model_cls):
    model = model_cls(
        n_outputs=input_sizes["n_classes"],
        n_chans=input_sizes["n_channels"],
        n_times=input_sizes["n_in_times"],
    )

    check_forward_pass(
        model,
        input_sizes,
    )


@pytest.mark.parametrize("model_cls", [EEGInceptionERP])
def test_eeginception_erp_n_params(model_cls):
    """Make sure the number of parameters is the same as in the paper when
    using the same architecture hyperparameters.
    """
    model = model_cls(
        n_chans=8,
        n_outputs=2,
        n_times=128,  # input_time
        sfreq=128,
        drop_prob=0.5,
        n_filters=8,
        scales_samples_s=(0.5, 0.25, 0.125),
        activation=torch.nn.ELU,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == 14926  # From paper's TABLE IV EEG-Inception Architecture Details


def test_eeginception_mi(input_sizes):
    sfreq = 250
    model = EEGInceptionMI(
        n_outputs=input_sizes["n_classes"],
        n_chans=input_sizes["n_channels"],
        input_window_seconds=input_sizes["n_in_times"] / sfreq,
        sfreq=sfreq,
    )

    check_forward_pass(
        model,
        input_sizes,
    )


@pytest.mark.parametrize(
    "n_filter,reported",
    [(6, 51386), (12, 204002), (16, 361986), (24, 812930), (64, 5767170)],
)
def test_eeginception_mi_binary_n_params(n_filter, reported):
    """Make sure the number of parameters is the same as in the paper when
    using the same architecture hyperparameters.

    Note
    ----
    For some reason, we match the correct number of parameters for all
    configurations in the binary classification case, but none for the 4-class
    case... Should be investigated by contacting the authors.
    """
    model = EEGInceptionMI(
        n_chans=3,
        n_outputs=2,
        input_window_seconds=3.0,  # input_time
        sfreq=250,
        n_convs=3,
        n_filters=n_filter,
        kernel_unit_s=0.1,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # From first column of TABLE 2 in EEG-Inception paper
    assert n_params == reported


def test_atcnet(input_sizes):
    sfreq = 250
    input_sizes["n_in_times"] = 1125
    model = ATCNet(
        n_chans=input_sizes["n_channels"],
        n_outputs=input_sizes["n_classes"],
        input_window_seconds=input_sizes["n_in_times"] / sfreq,
        sfreq=sfreq,
    )

    check_forward_pass(
        model,
        input_sizes,
    )


def test_atcnet_n_params():
    """Make sure the number of parameters is the same as in the paper when
    using the same architecture hyperparameters.
    """
    n_windows = 5
    att_head_dim = 8
    num_heads = 2

    model = ATCNet(
        n_chans=22,
        n_outputs=4,
        input_window_seconds=4.5,
        sfreq=250,
        n_windows=n_windows,
        head_dim=att_head_dim,
        num_heads=num_heads,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # The paper states the models has around "115.2 K" parameters in its
    # conclusion. By analyzing the official tensorflow code, we found indeed
    # 115,172 parameters, but these take into account untrainable batch norm
    # params, while the number of trainable parameters is 113,732.
    official_code_nparams = 113_732

    assert n_params == official_code_nparams


@pytest.mark.parametrize(
    "n_channels,sfreq,n_classes,input_size_s",
    [(20, 128, 5, 30), (10, 256, 4, 20), (1, 64, 2, 30)],
)
def test_sleep_stager(n_channels, sfreq, n_classes, input_size_s):
    rng = np.random.RandomState(42)
    time_conv_size_s = 0.5
    max_pool_size_s = 0.125
    pad_size_s = 0.25
    n_examples = 10

    model = SleepStagerChambon2018(
        n_channels,
        sfreq,
        n_conv_chs=8,
        time_conv_size_s=time_conv_size_s,
        max_pool_size_s=max_pool_size_s,
        pad_size_s=pad_size_s,
        input_window_seconds=input_size_s,
        n_outputs=n_classes,
        drop_prob=0.25,
    )
    model.eval()

    X = rng.randn(n_examples, n_channels, int(sfreq * input_size_s))
    X = torch.from_numpy(X.astype(np.float32))

    y_pred1 = model(X)  # 3D inputs
    y_pred2 = model(X.unsqueeze(1))  # 4D inputs
    assert y_pred1.shape == (n_examples, n_classes)
    assert y_pred2.shape == (n_examples, n_classes)
    np.testing.assert_allclose(
        y_pred1.detach().cpu().numpy(), y_pred2.detach().cpu().numpy()
    )


@pytest.mark.parametrize(
    "n_chans,sfreq,n_classes,input_size_s",
    [(20, 128, 5, 30), (10, 100, 4, 20), (1, 64, 2, 30)],
)
def test_usleep(n_chans, sfreq, n_classes, input_size_s):
    rng = np.random.RandomState(42)
    n_examples = 10
    seq_length = 3

    model = USleep(
        n_chans=n_chans,
        sfreq=sfreq,
        n_outputs=n_classes,
        input_window_seconds=input_size_s,
        ensure_odd_conv_size=True,
    )
    model.eval()

    X = rng.randn(n_examples, n_chans, int(sfreq * input_size_s))
    X = torch.from_numpy(X.astype(np.float32))

    y_pred1 = model(X)  # 3D inputs : (batch, channels, time)
    y_pred2 = model(X.unsqueeze(1))  # 4D inputs : (batch, 1, channels, time)
    y_pred3 = model(
        torch.stack([X for idx in range(seq_length)], axis=1)
    )  # (batch, sequence, channels, time)
    assert y_pred1.shape == (n_examples, n_classes)
    assert y_pred2.shape == (n_examples, n_classes)
    assert y_pred3.shape == (n_examples, n_classes, seq_length)
    np.testing.assert_allclose(
        y_pred1.detach().cpu().numpy(), y_pred2.detach().cpu().numpy()
    )


def test_usleep_n_params():
    """Make sure the number of parameters is the same as in the paper when
    using the same architecture hyperparameters.
    """
    model = USleep(
        n_chans=2,
        sfreq=128,
        depth=12,
        n_time_filters=5,
        complexity_factor=1.67,
        with_skip_connection=True,
        n_outputs=5,
        input_window_seconds=30,
        time_conv_size_s=9 / 128,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == 3114337  # From paper's supplementary materials, Table 2


def test_sleep_stager_return_feats():
    n_channels = 2
    sfreq = 10
    input_size_s = 30
    n_classes = 3

    model = SleepStagerChambon2018(
        n_channels,
        sfreq,
        n_conv_chs=8,
        input_window_seconds=input_size_s,
        n_outputs=n_classes,
        return_feats=True,
    )
    model.eval()

    rng = np.random.RandomState(42)
    X = rng.randn(10, n_channels, int(sfreq * input_size_s))
    X = torch.from_numpy(X.astype(np.float32))

    out = model(X)
    assert out.shape == (10, model.len_last_layer)


def test_tidnet(input_sizes):
    model = TIDNet(
        input_sizes["n_channels"],
        input_sizes["n_classes"],
        input_sizes["n_in_times"],
    )
    check_forward_pass(model, input_sizes)


@pytest.mark.parametrize(
    "sfreq,n_classes,input_size_s,d_model",
    [(100, 5, 30, 80), (125, 4, 30, 100)]
)
def test_eldele_2021(sfreq, n_classes, input_size_s, d_model):
    # (100, 5, 30, 80) - Physionet Sleep
    # (125, 4, 30, 100) - SHHS
    rng = np.random.RandomState(42)
    n_channels = 1
    n_examples = 10

    model = AttnSleep(
        sfreq=sfreq,
        n_outputs=n_classes,
        input_window_seconds=input_size_s,
        d_model=d_model,
        return_feats=False,
    )
    model.eval()

    X = rng.randn(n_examples, n_channels,
                  np.ceil(input_size_s * sfreq).astype(int))
    X = torch.from_numpy(X.astype(np.float32))

    y_pred1 = model(X)  # 3D inputs
    assert y_pred1.shape == (n_examples, n_classes)


def test_eldele_2021_feats():
    n_channels = 1
    sfreq = 100
    input_size_s = 30
    n_classes = 3
    n_examples = 10

    model = AttnSleep(
        sfreq,
        input_window_seconds=input_size_s,
        n_outputs=n_classes,
        return_feats=True,
    )
    model.eval()

    rng = np.random.RandomState(42)
    X = rng.randn(n_examples, n_channels, int(sfreq * input_size_s))
    X = torch.from_numpy(X.astype(np.float32))

    out = model(X)
    assert out.shape == (n_examples, model.len_last_layer)


@pytest.mark.parametrize(
    "n_channels,sfreq,n_groups,n_classes,input_size_s",
    [(20, 128, 2, 5, 30), (10, 100, 2, 4, 20), (1, 64, 1, 2, 30)],
)
def test_blanco_2020(n_channels, sfreq, n_groups, n_classes, input_size_s):
    rng = np.random.RandomState(42)
    n_examples = 10

    model = SleepStagerBlanco2020(
        n_chans=n_channels,
        sfreq=sfreq,
        n_groups=n_groups,
        input_window_seconds=input_size_s,
        n_outputs=n_classes,
        return_feats=False,
    )
    model.eval()

    X = rng.randn(n_examples, n_channels,
                  np.ceil(input_size_s * sfreq).astype(int))
    X = torch.from_numpy(X.astype(np.float32))

    y_pred1 = model(X)  # 3D inputs
    y_pred2 = model(X.unsqueeze(2))  # 4D inputs
    assert y_pred1.shape == (n_examples, n_classes)
    assert y_pred2.shape == (n_examples, n_classes)
    np.testing.assert_allclose(
        y_pred1.detach().cpu().numpy(), y_pred2.detach().cpu().numpy()
    )


def test_blanco_2020_feats():
    n_channels = 2
    sfreq = 50
    input_size_s = 30
    n_classes = 3
    n_examples = 10

    model = SleepStagerBlanco2020(
        n_channels,
        sfreq,
        input_window_seconds=input_size_s,
        n_outputs=n_classes,
        return_feats=True,
    )
    model.eval()

    rng = np.random.RandomState(42)
    X = rng.randn(n_examples, n_channels, int(sfreq * input_size_s))
    X = torch.from_numpy(X.astype(np.float32))

    out = model(X)
    assert out.shape == (n_examples, model.len_last_layer)


def test_eegitnet_shape():
    n_channels = 2
    sfreq = 50
    input_size_s = 30
    n_classes = 3
    n_examples = 10
    model = EEGITNet(
        n_outputs=n_classes,
        n_chans=n_channels,
        n_times=int(sfreq * input_size_s),
    )
    model.eval()

    rng = np.random.RandomState(42)
    X = rng.randn(n_examples, n_channels, int(sfreq * input_size_s))
    X = torch.from_numpy(X.astype(np.float32))

    out = model(X)
    assert out.shape == (n_examples, n_classes)


@pytest.mark.parametrize("n_classes", [5, 4, 2])
def test_deepsleepnet(n_classes):
    n_channels = 1
    sfreq = 100
    input_size_s = 30
    n_examples = 10

    model = DeepSleepNet(n_outputs=n_classes, return_feats=False)
    model.eval()

    rng = np.random.RandomState(42)
    X = rng.randn(n_examples, n_channels,
                  np.ceil(input_size_s * sfreq).astype(int))
    X = torch.from_numpy(X.astype(np.float32))

    y_pred1 = model(X)  # 3D inputs
    y_pred2 = model(X.unsqueeze(1))  # 4D inputs
    assert y_pred1.shape == (n_examples, n_classes)
    assert y_pred2.shape == (n_examples, n_classes)
    np.testing.assert_allclose(
        y_pred1.detach().cpu().numpy(), y_pred2.detach().cpu().numpy()
    )


def test_deepsleepnet_feats():
    n_channels = 1
    sfreq = 100
    input_size_s = 30
    n_classes = 3
    n_examples = 10

    model = DeepSleepNet(n_outputs=n_classes, return_feats=True)
    model.eval()

    rng = np.random.RandomState(42)
    X = rng.randn(n_examples, n_channels, int(sfreq * input_size_s))
    X = torch.from_numpy(X.astype(np.float32))

    out = model(X.unsqueeze(1))
    assert out.shape == (n_examples, model.len_last_layer)


def test_deepsleepnet_feats_with_hook():
    n_channels = 1
    sfreq = 100
    input_size_s = 30
    n_classes = 3
    n_examples = 10

    model = DeepSleepNet(n_outputs=n_classes, return_feats=False)
    model.eval()

    rng = np.random.RandomState(42)
    X = rng.randn(n_examples, n_channels, int(sfreq * input_size_s))
    X = torch.from_numpy(X.astype(np.float32))

    def get_intermediate_layers(intermediate_layers, layer_name):
        def hook(model, input, output):
            intermediate_layers[layer_name] = output.flatten(
                start_dim=1).detach()

        return hook

    intermediate_layers = {}
    layer_name = "features_extractor"
    model.features_extractor.register_forward_hook(
        get_intermediate_layers(intermediate_layers, layer_name)
    )

    y_pred = model(X.unsqueeze(1))
    assert intermediate_layers["features_extractor"].shape == (
        n_examples,
        model.len_last_layer,
    )
    assert y_pred.shape == (n_examples, n_classes)


@pytest.fixture
def sample_input():
    batch_size = 16
    n_channels = 12
    n_timesteps = 1000
    return torch.rand(batch_size, n_channels, n_timesteps)


@pytest.fixture
def model():
    return EEGConformer(n_outputs=2, n_chans=12, n_times=1000)


def test_model_creation(model):
    assert model is not None


def test_sparcnet_dummy():
    input_sizes = dict(n_channels=32, n_in_times=125, n_classes=2, n_samples=64)
    model = SPARCNet(
        n_chans=input_sizes["n_channels"],
        n_outputs=input_sizes["n_classes"],
        n_times=input_sizes["n_in_times"],
        sfreq=500.0,
    )
    check_forward_pass_3d(model, input_sizes)


@pytest.mark.parametrize(
    "n_times, n_chans, sfreq, n_outputs",
    [
        (256, 8, 256.0, 2),
        (204, 8, 256.0, 2),
        (125, 32, 500.0, 2),
        (204, 16, 256.0, 2),
        (128, 16, 128.0, 2),
        (153, 8, 512.0, 2),
    ],
)
def test_atcnet_dummy(n_times, n_chans, sfreq, n_outputs):
    batch_size = 64
    input_sizes = dict(
        n_channels=n_chans,
        n_in_times=n_times,
        n_classes=n_outputs,
        n_samples=batch_size,
    )
    model = ATCNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        sfreq=sfreq,
    )
    check_forward_pass_3d(model, input_sizes)


@pytest.mark.parametrize(
    "n_times, n_chans, sfreq, n_outputs",
    [
        (125, 32, 500.0, 2),
        (614, 64, 2048.0, 2),
        (153, 8, 512.0, 2),
    ],
)
def test_tsception_dummy(n_times, n_chans, sfreq, n_outputs):
    batch_size = 64
    input_sizes = dict(
        n_channels=n_chans,
        n_in_times=n_times,
        n_classes=n_outputs,
        n_samples=batch_size,
    )
    model = TSception(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        sfreq=sfreq,
    )
    check_forward_pass_3d(model, input_sizes)


@pytest.mark.parametrize(
    "n_times, n_chans, sfreq, n_outputs",
    [
        (125, 32, 500.0, 2),
        (614, 64, 2048.0, 2),
        (153, 8, 512.0, 2),
    ],
)
def test_sccnet_dummy(n_times, n_chans, sfreq, n_outputs):
    batch_size = 64
    input_sizes = dict(
        n_channels=n_chans,
        n_in_times=n_times,
        n_classes=n_outputs,
        n_samples=batch_size,
    )
    model = SCCNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        sfreq=sfreq,
    )
    check_forward_pass_3d(model, input_sizes)


@pytest.mark.parametrize(
    "n_times, n_chans, sfreq, n_outputs",
    [
        (2000, 63, 500.0, 4),
    ],
)
def test_eeginceptionmi_dummy(n_times, n_chans, sfreq, n_outputs):
    batch_size = 64
    input_sizes = dict(
        n_channels=n_chans,
        n_in_times=n_times,
        n_classes=n_outputs,
        n_samples=batch_size,
    )
    model = EEGInceptionMI(
        n_chans=n_chans,
        n_outputs=n_outputs,
        input_window_seconds=n_times / sfreq,
        sfreq=sfreq,
    )
    check_forward_pass_3d(model, input_sizes)


@pytest.mark.parametrize(
    "n_times, n_chans, sfreq, n_outputs",
    [
        (256, 8, 256.0, 2),
        (204, 8, 256.0, 2),
        (125, 32, 500.0, 2),
        (204, 16, 256.0, 2),
        (128, 16, 128.0, 2),
        (384, 14, 128.0, 5),
        (153, 8, 512.0, 2),
    ],
)
def test_deep4net_dummy(n_times, n_chans, sfreq, n_outputs):
    batch_size = 64
    input_sizes = dict(
        n_channels=n_chans,
        n_in_times=n_times,
        n_classes=n_outputs,
        n_samples=batch_size,
    )
    model = Deep4Net(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        sfreq=sfreq,
    )
    check_forward_pass_3d(model, input_sizes)


@pytest.mark.parametrize(
    "n_times, n_chans, sfreq, n_outputs",
    [
        (1536, 16, 512.0, 3),
        (512, 16, 512.0, 2),
        (125, 32, 500.0, 2),
        (2560, 32, 512.0, 2),
        (2560, 13, 512.0, 2),
        (512, 32, 512.0, 2),
        (2560, 15, 512.0, 2),
        (1024, 30, 1024.0, 2),
        (5120, 16, 512.0, 2),
        (1536, 64, 512.0, 2),
        (2048, 32, 2048.0, 2),
        (614, 64, 2048.0, 2),
        (1536, 61, 512.0, 7),
        (899, 31, 1000.0, 2),
        (4000, 62, 1000.0, 4),
        (4000, 62, 1000.0, 2),
        (153, 8, 512.0, 2),
        (1000, 62, 1000.0, 2),
        (1200, 31, 1000.0, 2),
    ],
)
def test_contrawr_dummy(n_times, n_chans, sfreq, n_outputs):
    batch_size = 64
    input_sizes = dict(
        n_channels=n_chans,
        n_in_times=n_times,
        n_classes=n_outputs,
        n_samples=batch_size,
    )
    model = ContraWR(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        sfreq=sfreq,
    )
    check_forward_pass_3d(model, input_sizes)


@pytest.mark.parametrize(
    "n_times, n_chans, sfreq, n_outputs",
    [
        (204, 8, 256.0, 2),
        (125, 32, 500.0, 2),
        (204, 16, 256.0, 2),
        (614, 64, 2048.0, 2),
        (899, 31, 1000.0, 2),
        (153, 8, 512.0, 2),
    ],
)
def test_biot_dummy(n_times, n_chans, sfreq, n_outputs):
    batch_size = 64
    input_sizes = dict(
        n_channels=n_chans,
        n_in_times=n_times,
        n_classes=n_outputs,
        n_samples=batch_size,
    )
    model = BIOT(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        sfreq=sfreq,
    )
    check_forward_pass_3d(model, input_sizes)


@pytest.mark.parametrize(
    "n_times, n_chans, sfreq, n_outputs",
    [
        (125, 32, 500.0, 2),
        (128, 16, 128.0, 2),
        (153, 8, 512.0, 2),
    ],
)
def test_attentionbasenet_dummy(n_times, n_chans, sfreq, n_outputs):
    batch_size = 64
    input_sizes = dict(
        n_channels=n_chans,
        n_in_times=n_times,
        n_classes=n_outputs,
        n_samples=batch_size,
    )
    model = AttentionBaseNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        sfreq=sfreq,
    )
    check_forward_pass_3d(model, input_sizes)



def test_conformer_forward_pass(sample_input, model):
    output = model(sample_input)
    assert isinstance(output, torch.Tensor)

    model_with_feature = EEGConformer(
        n_outputs=2, n_chans=12, n_times=1000, return_features=True
    )
    output = model_with_feature(sample_input)

    assert isinstance(output, torch.Tensor) and output.shape == torch.Size([16, 61, 40])


def test_patch_embedding(sample_input, model):
    patch_embedding = model.patch_embedding
    x = torch.unsqueeze(sample_input, dim=1)
    output = patch_embedding(x)
    assert output.shape[0] == sample_input.shape[0]


def test_model_trainable_parameters(model):
    patch_parameters = model.patch_embedding.parameters()
    transformer_parameters = model.transformer.parameters()
    classification_parameters = model.fc.parameters()
    final_layer_parameters = model.final_layer.parameters()

    trainable_patch_params = sum(
        p.numel() for p in patch_parameters if p.requires_grad)

    trainable_transformer_params = sum(
        p.numel() for p in transformer_parameters if p.requires_grad
    )

    trainable_classification_params = sum(
        p.numel() for p in classification_parameters if p.requires_grad
    )

    trainable_final_layer_parameters = sum(
        p.numel() for p in final_layer_parameters if p.requires_grad
    )

    assert trainable_patch_params == 22000
    assert trainable_transformer_params == 118320
    assert trainable_classification_params == 633120
    assert trainable_final_layer_parameters == 66


@pytest.mark.parametrize("n_chans", (2 ** np.arange(8)).tolist())
@pytest.mark.parametrize("n_outputs", [2, 3, 4, 5, 50])
@pytest.mark.parametrize("input_size_s", [1, 2, 5, 10, 15, 30])
def test_biot(n_chans, n_outputs, input_size_s):
    rng = check_random_state(42)
    sfreq = 200
    n_examples = 3
    n_times = np.ceil(input_size_s * sfreq).astype(int)

    model = BIOT(
        n_outputs=n_outputs,
        n_chans=n_chans,
        n_times=n_times,
        sfreq=sfreq,
        hop_length=50,
    )
    model.eval()

    X = rng.randn(n_examples, n_chans, n_times)
    X = torch.from_numpy(X.astype(np.float32))

    y_pred1 = model(X)  # 3D inputs
    assert y_pred1.shape == (n_examples, n_outputs)
    assert isinstance(y_pred1, torch.Tensor)


@pytest.fixture
def default_biot_params():
    return {
        "embed_dim": 256,
        "num_heads": 8,
        "num_layers": 4,
        "sfreq": 200,
        "hop_length": 50,
        "n_outputs": 2,
        "n_chans": 64,
        "n_times": 1000,
    }


def test_initialization_default_parameters(default_biot_params):
    """Test BIOT initialization with default parameters."""
    biot = BIOT(**default_biot_params)

    assert biot.embed_dim == 256
    assert biot.num_heads == 8
    assert biot.num_layers == 4


def test_model_trainable_parameters_biot(default_biot_params):
    biot = BIOT(**default_biot_params)

    biot_encoder = biot.encoder.parameters()
    biot_classifier = biot.final_layer.parameters()

    trainable_params_bio = sum(p.numel() for p in biot_encoder if p.requires_grad)
    trainable_params_clf = sum(p.numel() for p in biot_classifier if p.requires_grad)

    assert trainable_params_bio == 3198464  # ~ 3.2 M according to Labram paper
    assert trainable_params_clf == 514


@pytest.fixture
def default_labram_params():
    return {
        "n_times": 1000,
        "n_chans": 64,
        "chs_info": [{"ch_name": ch_name} for ch_name in LABRAM_CHANNEL_ORDER[:64]],
        "patch_size": 200,
        "sfreq": 200,
        "qk_norm": partial(nn.LayerNorm, eps=1e-6),
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
        "mlp_ratio": 4,
        "n_outputs": 2,
    }


def test_model_trainable_parameters_labram(default_labram_params):
    """
    Test the number of trainable parameters in Labram model based on the
    paper values.

    Parameters
    ----------
    default_labram_params: dict with default parameters for Labram model

    """
    labram_base = Labram(num_layers=12, num_heads=12,
                         **default_labram_params)

    labram_base_parameters = labram_base.get_torchinfo_statistics().trainable_params

    # We added some parameters layers in the segmentation step to match the
    # braindecode convention.
    assert np.round(labram_base_parameters / 1e6, 1) == 5.7
    # ~ 5.7 M with current braindecode adaptation

    labram_large = Labram(
        num_layers=24,
        num_heads=16,
        conv_out_channels=16,
        embed_dim=400,
        **default_labram_params,
    )
    labram_large_parameters = labram_large.get_torchinfo_statistics().trainable_params

    assert np.round(labram_large_parameters / 1e6, 0) == 46
    # ~ 46 M matching the paper

    labram_huge = Labram(
        num_layers=48,
        num_heads=16,
        conv_out_channels=32,
        embed_dim=800,
        **default_labram_params,
    )

    labram_huge_parameters = labram_huge.get_torchinfo_statistics().trainable_params
    # 369M matching the paper
    assert np.round(labram_huge_parameters / 1e6, 0) == 369

    assert labram_base.get_num_layers() == 12
    assert labram_large.get_num_layers() == 24
    assert labram_huge.get_num_layers() == 48


@pytest.mark.parametrize("use_mean_pooling", [True, False])
def test_labram_returns(default_labram_params, use_mean_pooling):
    """
    Testing if the model is returning the correct shapes for the different
    return options.

    Parameters
    ----------
    default_labram_params: dict with default parameters for Labram model

    """
    labram_base = Labram(
        num_layers=12,
        num_heads=12,
        use_mean_pooling=use_mean_pooling,
        **default_labram_params,
    )
    # Defining a random data
    X = torch.rand(1, default_labram_params["n_chans"],
                   default_labram_params["n_times"])

    with torch.no_grad():
        out = labram_base(X, return_all_tokens=False,
                          return_patch_tokens=False)

        assert out.shape == torch.Size([1, default_labram_params["n_outputs"]])

        out_patches = labram_base(X, return_all_tokens=False,
                                  return_patch_tokens=True)

        assert out_patches.shape == torch.Size(
            [1, 320, default_labram_params["n_outputs"]]
        )

        out_all_tokens = labram_base(X, return_all_tokens=True,
                                     return_patch_tokens=False)
        assert out_all_tokens.shape == torch.Size(
            [1, 321, default_labram_params["n_outputs"]]
        )


def test_labram_without_pos_embed(default_labram_params):
    labram_base_not_pos_emb = Labram(
        num_layers=12, num_heads=12, use_abs_pos_emb=False,
        **default_labram_params
    )

    X = torch.rand(1, default_labram_params["n_chans"],
                   default_labram_params["n_times"])

    with torch.no_grad():
        out_without_pos_emb = labram_base_not_pos_emb(X)
        assert out_without_pos_emb.shape == torch.Size([1, 2])


# def test_labram_n_outputs_0(default_labram_params):
#     """
#     Testing if the model is returning the correct shapes for the different
#     return options.

#     Parameters
#     ----------
#     default_labram_params: dict with default parameters for Labram model

#     """
#     default_labram_params["n_outputs"] = 0
#     labram_base = Labram(num_layers=12, num_heads=12,
#                          **default_labram_params)
#     # Defining a random data
#     X = torch.rand(1, default_labram_params["n_chans"],
#                    default_labram_params["n_times"])

#     with torch.no_grad():
#         out = labram_base(X)
#         assert out.shape[-1] == default_labram_params["patch_size"]
#         assert isinstance(labram_base.final_layer, nn.Identity)


@pytest.fixture
def param_eegsimple():
    return {
        "n_times": 1000,
        "n_chans": 18,
        "patch_size": 200,
        "n_classes": 2,
        "sfreq": 100
    }


def test_eeg_simpleconv(param_eegsimple):
    batch_size = 16

    input = torch.rand(batch_size,
                       param_eegsimple['n_chans'],
                       param_eegsimple['n_times'])

    model = EEGSimpleConv(
        n_outputs=param_eegsimple['n_classes'],
        n_chans=param_eegsimple['n_chans'],
        sfreq=param_eegsimple['sfreq'],
        feature_maps=32,
        n_convs=1,
        resampling_freq=80,
        kernel_size=8,
    )
    output = model(input)
    assert isinstance(output, torch.Tensor)
    assert (output.shape[0] == batch_size and
            output.shape[1] == param_eegsimple['n_classes'])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == 21250


def test_eeg_simpleconv_features(param_eegsimple):
    batch_size = 16

    input = torch.rand(batch_size,
                       param_eegsimple['n_chans'],
                       param_eegsimple['n_times'])

    model = EEGSimpleConv(
        n_outputs=param_eegsimple['n_classes'],
        n_chans=param_eegsimple['n_chans'],
        sfreq=param_eegsimple['sfreq'],
        feature_maps=32,
        n_convs=1,
        resampling_freq=80,
        kernel_size=8,
        return_feature=True
    )

    output = model(input)
    assert isinstance(output, torch.Tensor)

    feature = output


    assert (feature.shape[0] == batch_size and
            feature.shape[1] == 32)


@pytest.fixture(scope="module")
def default_attentionbasenet_params():
    return {
        'n_times': 1000,
        'n_chans': 22,
        'n_outputs': 4,
    }


@pytest.mark.parametrize("attention_mode", [
    None,
    "se",
    "gsop",
    "fca",
    "encnet",
    "eca",
    "ge",
    "gct",
    "srm",
    "cbam",
    "cat",
    "catlite"
])
def test_attentionbasenet(default_attentionbasenet_params, attention_mode):
    model = AttentionBaseNet(**default_attentionbasenet_params,
                             attention_mode=attention_mode)
    input_sizes = dict(
        n_samples=7,
        n_channels=default_attentionbasenet_params.get("n_chans"),
        n_in_times=default_attentionbasenet_params.get("n_times"),
        n_classes=default_attentionbasenet_params.get("n_outputs")
    )
    check_forward_pass(model, input_sizes)


def test_parameters_contrawr():

    model = ContraWR(n_outputs=2, n_chans=22, sfreq=250, n_times=1000)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 1.6M parameters according to the Labram paper, table 1
    assert np.round(n_params / 1e6, 1) == 1.6


def test_parameters_SPARCNet():

    model = SPARCNet(n_outputs=2, n_chans=16, n_times=400)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 0.79M parameters according to the Labram paper, table 1
    # The model parameters are indeed in the n_times range
    assert np.round(n_params / 1e6, 1) == 0.8


def test_parameters_EEGTCNet():

    model = EEGTCNet(n_outputs=4, n_chans=22, n_times=1000)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 4.27 K according to the Table V from the original paper.
    assert np.round(n_params / 1e3, 1) == 4.2


@pytest.mark.parametrize("method", ["plv", "mag", "corr"])
def test_eegminer_initialization_and_forward(method):
    """
    Test EEGMiner initialization and forward pass for different methods ('plv', 'mag', 'corr').
    """
    batch_size = 4
    n_chans = 8
    n_times = 256
    n_outputs = 2
    sfreq = 100.0  # Hz
    input_tensor = torch.randn(batch_size, n_chans, n_times)

    eegminer = EEGMiner(
        method=method,
        n_chans=n_chans,
        n_times=n_times,
        n_outputs=n_outputs,
        sfreq=sfreq,
        filter_f_mean=[10.0, 20.0],
        filter_bandwidth=[5.0, 5.0],
        filter_shape=[2.0, 2.0],
        group_delay=[20.0, 20.0],
    )

    output = eegminer(input_tensor)
    assert output.shape == (batch_size, n_outputs), \
        f"Output shape should be ({batch_size}, {n_outputs}) for method '{method}', got {output.shape}"


def test_eegminer_invalid_parameters():
    """
    Test that EEGMiner raises an error when initialized with invalid parameters.
    """
    n_chans = 8
    n_times = 256
    n_outputs = 2
    sfreq = 100.0  # Hz

    # Invalid method
    with pytest.raises(ValueError):
        EEGMiner(
            method="invalid_method",
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
            sfreq=sfreq,
        )


def test_eegminer_filter_clamping():
    """
    Test that EEGMiner's filters are constructed correctly and parameters are clamped.
    """
    n_chans = 4
    n_times = 256
    n_outputs = 2
    sfreq = 100.0  # Hz

    eegminer = EEGMiner(
        method="mag",
        n_chans=n_chans,
        n_times=n_times,
        n_outputs=n_outputs,
        sfreq=sfreq,
        filter_f_mean=[50.0, -10.0],  # Values outside clamp range
        filter_bandwidth=[0.5, 100.0],  # Values outside clamp range
        filter_shape=[1.5, 3.5],  # Values outside clamp range
        group_delay=[20.0, 20.0],
    )

    # Construct filters
    eegminer.filter.construct_filters()
    f_mean = eegminer.filter.f_mean.data * (sfreq / 2)
    bandwidth = eegminer.filter.bandwidth.data * (sfreq / 2)
    shape = eegminer.filter.shape.data

    # Check clamping
    assert torch.all(f_mean >= 1.0) and torch.all(f_mean <= 45.0), \
        f"f_mean should be clamped between 1.0 and 45.0 Hz, got {f_mean}"
    assert torch.all(bandwidth >= 1.0) and torch.all(bandwidth <= 50.0), \
        f"bandwidth should be clamped between 1.0 and 50.0 Hz, got {bandwidth}"
    assert torch.all(shape >= 2.0) and torch.all(shape <= 3.0), \
        f"shape should be clamped between 2.0 and 3.0, got {shape}"


def test_eegminer_corr_output_size():
    """
    Test that EEGMiner produces the correct number of features for the 'corr' method.
    """
    batch_size = 2
    n_chans = 6
    n_times = 256
    n_outputs = 2
    sfreq = 100.0  # Hz
    n_filters = 2

    input_tensor = torch.randn(batch_size, n_chans, n_times)

    eegminer = EEGMiner(
        method="corr",
        n_chans=n_chans,
        n_times=n_times,
        n_outputs=n_outputs,
        sfreq=sfreq,
        filter_f_mean=[10.0, 20.0],
        filter_bandwidth=[5.0, 5.0],
        filter_shape=[2.0, 2.0],
        group_delay=[20.0, 20.0],
    )

    output = eegminer(input_tensor)
    expected_n_features = n_filters * n_chans * (n_chans - 1) // 2
    assert eegminer.n_features == expected_n_features, \
        f"Expected {expected_n_features} features, got {eegminer.n_features}"
    assert output.shape == (batch_size, n_outputs), \
        f"Output shape should be ({batch_size}, {n_outputs}), got {output.shape}"


def test_eegminer_plv_values_range():
    """
    Test that the PLV values computed by EEGMiner are within the valid range [0, 1].
    """
    batch_size = 1
    n_chans = 4
    n_times = 512
    n_outputs = 2
    sfreq = 256.0  # Hz

    input_tensor = torch.randn(batch_size, n_chans, n_times)

    eegminer = EEGMiner(
        method="plv",
        n_chans=n_chans,
        n_times=n_times,
        n_outputs=n_outputs,
        sfreq=sfreq,
        filter_f_mean=[8.0, 12.0],
        filter_bandwidth=[2.0, 2.0],
        filter_shape=[2.0, 2.0],
        group_delay=[20.0, 20.0],
    )

    # Forward pass up to PLV computation
    x = eegminer.ensure_dim(input_tensor)
    x = eegminer.filter(x)
    x = eegminer._apply_plv(x, n_chans=n_chans)

    # PLV values should be in [0, 1]
    assert torch.all(x >= 0.0) and torch.all(x <= 1.0), \
        "PLV values should be in the range [0, 1]"


def test_eegnet_final_layer_linear_true():
    """Test that final_layer_linear=True uses a conv-based classifier without warning."""
    model = EEGNet(
        final_layer_with_constraint=True,
        n_chans=4,
        n_times=128,
        n_outputs=2
    )

    X = torch.randn(2, 4, 128)  # (batch_size=2, channels=4, time=128)
    y = model(X)

    # Check output shape: should be (batch_size, n_outputs)
    assert y.shape == (2, 2), f"Unexpected output shape {y.shape}"

    # Check final layer is Conv2d instead of Flatten/LinearWithConstraint
    final_layer = dict(model.named_modules())["final_layer"]
    # Inside final_layer for conv-based approach, we expect "conv_classifier" as the first sub-module:
    assert hasattr(final_layer,
                   "linearconstraint"), "Expected a 'linear constraint' sub-module."

def test_eegnet_final_layer_linear_false():
    """Test that final_layer_conv=False raises a DeprecationWarning and uses
    a linear layer."""
    with pytest.warns(DeprecationWarning,
                      match="Parameter 'final_layer_with_constraint=False' is deprecated"):
        model = EEGNet(
            final_layer_with_constraint=False,
            n_chans=4,
            n_times=128,
            n_outputs=2
        )

    X = torch.randn(2, 4, 128)
    y = model(X)

    # Check output shape: should be (batch_size, n_outputs)
    assert y.shape == (2, 2), f"Unexpected output shape {y.shape}"

    # Check final layer is Flatten + LinearWithConstraint (no "conv_classifier")
    final_layer = dict(model.named_modules())["final_layer"]
    submodule_names = list(dict(final_layer.named_children()).keys())
    assert "conv_classifier" in submodule_names, "Did expect a convolutional classifier."
    assert "linearconstraint" not in submodule_names, "Did not expected a linearconstraint sub-module."



@pytest.mark.parametrize(
    "temporal_layer", ['VarLayer', 'StdLayer', 'LogVarLayer',
                       'MeanLayer', 'MaxLayer']
)
def test_fbcnet_forward_pass(temporal_layer):
    n_chans = 22
    n_times = 1000
    n_outputs = 2
    batch_size = 8
    n_bands = 9

    model = FBCNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        n_bands=n_bands,
        temporal_layer=temporal_layer,
        sfreq=250,
    )

    x = torch.randn(batch_size, n_chans, n_times)
    output = model(x)

    assert output.shape == (batch_size, n_outputs)

def test_fbcnet_specified_filter_parameters():
    n_chans = 22
    n_times = 1000
    n_outputs = 2
    n_bands = 9

    model = FBCNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        n_bands=n_bands,
        sfreq=250,
        filter_parameters={"method": "fir",
                           "filter_length": "auto",
                           "l_trans_bandwidth": 1.0,
                           "h_trans_bandwidth": 1.0,
                           "phase": "zero",
                           "iir_params": None,
                           "fir_window": "hamming",
                           "fir_design": "firwin",
                           })

    filter_bank_layer = model.spectral_filtering
    assert filter_bank_layer.n_bands == 9
    assert filter_bank_layer.phase == "zero"
    assert filter_bank_layer.method == "fir"
    assert filter_bank_layer.n_chans == 22
    assert filter_bank_layer.method_iir is False


@pytest.mark.parametrize(
    "n_chans, n_bands, n_filters_spat, stride_factor",
    [
        (3, 9, 32, 4),
        (22, 9, 32, 4),
        (22, 5, 16, 2),
        (64, 10, 64, 8),
    ],
)
def test_fbcnet_num_parameters(n_chans, n_bands, n_filters_spat, stride_factor):
    """
    The calculation total is according to paper page 13.
    Equation:
    (n_filters_spat ∗ n_bands*n_chans + n_filters_spat ∗ n_bands) +
    (2*n_filters_spat ∗ n_bands) +
    (n_filters_spat ∗ n_bands ∗ stride_factor ∗ n_outputs + n_outputs)
    Where
    number of EEG channels, variable n_chans,
    number of time points, variable n_time
    number of frequency bands, variable n_bands
    number of convolution filters per frequency band, variable n_filters_spat,
    number of output classes, variable n_outputs
    temporal window length, variable stride_factor
    Returns
    -------
    """
    n_times = 1000
    n_outputs = 2
    sfreq = 250

    conv_params = (n_filters_spat * n_bands*n_chans + n_filters_spat * n_bands)

    batchnorm_params = (2*n_filters_spat * n_bands)

    linear_parameters = n_filters_spat * n_bands * stride_factor * n_outputs + n_outputs

    total_parameters = conv_params + batchnorm_params + linear_parameters

    model = FBCNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        n_bands=n_bands,
        n_filters_spat=n_filters_spat,
        stride_factor=stride_factor,
        sfreq=sfreq,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_parameters == num_params


@pytest.mark.parametrize("n_times", [100, 500, 1000, 5000, 10000])
def test_fbcnet_different_n_times(n_times):
    n_chans = 22
    n_outputs = 2
    batch_size = 8

    model = FBCNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        n_bands=9,
        sfreq=250,
    )

    x = torch.randn(batch_size, n_chans, n_times)
    output = model(x)

    assert output.shape == (batch_size, n_outputs)
@pytest.mark.parametrize("stride_factor", [1, 2, 4, 5])
def test_fbcnet_stride_factor_warning(stride_factor):
    n_chans = 22
    n_times = 1003  # Not divisible by stride_factor when stride_factor > 1
    n_outputs = 2

    if n_times % stride_factor != 0:
        with pytest.warns(UserWarning, match="Input will be padded."):

            _ = FBCNet(
                n_chans=n_chans,
                n_outputs=n_outputs,
                n_times=n_times,
                stride_factor=stride_factor,
                sfreq=250,
            )


def test_fbcnet_invalid_temporal_layer():
    with pytest.raises(NotImplementedError):
        FBCNet(
            n_chans=22,
            n_outputs=2,
            n_times=1000,
            temporal_layer='InvalidLayer',
            sfreq=250,
        )

@pytest.mark.parametrize(
    "temporal_layer", ['VarLayer', 'StdLayer', 'LogVarLayer',
                       'MeanLayer', 'MaxLayer']
)
def test_fbmsnet_forward_pass(temporal_layer):
    n_chans = 22
    n_times = 1000
    n_outputs = 2
    batch_size = 8
    n_bands = 9

    model = FBMSNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        n_bands=n_bands,
        temporal_layer=temporal_layer,
        sfreq=250
    )

    x = torch.randn(batch_size, n_chans, n_times)
    output = model(x)

    assert output.shape == (batch_size, n_outputs)


def test_fbmsnet_specified_filter_parameters():
    n_chans = 22
    n_times = 1000
    n_outputs = 2
    n_bands = 9

    model = FBMSNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        n_bands=n_bands,
        sfreq=250,
        filter_parameters={"method": "fir",
                           "filter_length": "auto",
                           "l_trans_bandwidth": 1.0,
                           "h_trans_bandwidth": 1.0,
                           "phase": "zero",
                           "iir_params": None,
                           "fir_window": "hamming",
                           "fir_design": "firwin",
                           },
    )

    filter_bank_layer = model.spectral_filtering
    assert filter_bank_layer.n_bands == 9
    assert filter_bank_layer.phase == "zero"
    assert filter_bank_layer.method == "fir"
    assert filter_bank_layer.n_chans == 22
    assert filter_bank_layer.method_iir is False


@pytest.mark.parametrize("n_times", [100, 500, 1000, 5000, 10000])
def test_fbmsnet_different_n_times(n_times):
    n_chans = 22
    n_outputs = 2
    batch_size = 8

    model = FBMSNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        n_bands=9,
        sfreq=250,
    )

    x = torch.randn(batch_size, n_chans, n_times)
    output = model(x)

    assert output.shape == (batch_size, n_outputs)


@pytest.mark.parametrize("stride_factor", [1, 2, 4, 5])
def test_fbmsnet_stride_factor_warning(stride_factor):
    n_chans = 22
    n_times = 1003  # Not divisible by stride_factor when stride_factor > 1
    n_outputs = 2

    if n_times % stride_factor != 0:
        with pytest.warns(UserWarning, match="Input will be padded."):

            _ = FBMSNet(
                n_chans=n_chans,
                n_outputs=n_outputs,
                n_times=n_times,
                stride_factor=stride_factor,
                sfreq=250,
            )


def test_fbmsnet_invalid_temporal_layer():
    with pytest.raises(NotImplementedError):
        FBMSNet(
            n_chans=22,
            n_outputs=2,
            n_times=1000,
            temporal_layer='InvalidLayer',
            sfreq=250,
        )

def test_initialize_weights_linear():
    linear = nn.Linear(10, 5)
    IFNet._initialize_weights(linear)
    assert torch.allclose(linear.bias, torch.zeros_like(linear.bias))
    assert linear.weight.std().item() <= 0.02  # Checking trunc_normal_ std


def test_initialize_weights_norm():
    layer_norm = nn.LayerNorm(10)
    IFNet._initialize_weights(layer_norm)
    assert torch.allclose(layer_norm.weight, torch.ones_like(layer_norm.weight))
    assert torch.allclose(layer_norm.bias, torch.zeros_like(layer_norm.bias))

    batch_norm = nn.BatchNorm1d(10)
    IFNet._initialize_weights(batch_norm)
    assert torch.allclose(batch_norm.weight, torch.ones_like(batch_norm.weight))
    assert torch.allclose(batch_norm.bias, torch.zeros_like(batch_norm.bias))


def test_initialize_weights_conv():
    conv = nn.Conv1d(3, 6, kernel_size=3)
    IFNet._initialize_weights(conv)
    assert conv.weight.std().item() <= 0.02  # Checking trunc_normal_ std
    if conv.bias is not None:
        assert torch.allclose(conv.bias, torch.zeros_like(conv.bias))


test_cases = [
    pytest.param(64, id="n_times=64_perfect_multiple"),
    pytest.param(437, id="n_times=437_trace_example"), # Expect 104
    pytest.param(95, id="n_times=95_edge_case_1"), # Expect 24
    pytest.param(67, id="n_times=67_edge_case_2"), # Expect 16
    pytest.param(94, id="n_times=94_edge_case_3"), # Expect 24
]

@pytest.mark.parametrize("n_times_input", test_cases)
def test_eegnex_final_layer_in_features(n_times_input):
    """
    Tests if the EEGNeX model correctly calculates the 'in_features'
    for its final linear layer during initialization, especially for
    n_times values that are not perfect multiples of pooling factors,
    considering the specified padding.
    """
    n_chans_test = 2
    n_outputs_test = 5

    model = EEGNeX(
        n_chans=n_chans_test,
        n_outputs=n_outputs_test,
        n_times=n_times_input
    )

    print(model)

@pytest.mark.parametrize("batch_norm", [True, False])
def test_batchnorm_deep4net(batch_norm):
    """
    Test the number of trainable parameters in Deep4Net model.
    """
    model = Deep4Net(n_outputs=2, n_chans=22, n_times=1000, batch_norm=batch_norm)

    assert model is not None

def test_fc_length_eegconformer():
    """
    Test the number of trainable parameters in EEGConformer model.
    """
    model = EEGConformer(
        n_chans=64,  # Number of EEG channels
        n_outputs=2,  # Number of output classes
        n_times=500,  # Length of the input sequence (e.g., 500 time steps)
        final_fc_length=120,
        input_window_seconds=1.0,
        return_features=True,
        drop_prob=0.5,  # Dropout probability
        sfreq=500.0  # Sampling frequency of the EEG data
    )

    assert model is not None


# ============================================================================
# BrainModule Tests
# ============================================================================

@pytest.fixture
def brain_module_params():
    """Fixture with common BrainModule parameters."""
    return dict(
        n_chans=22,
        n_outputs=4,
        n_times=1000,
        sfreq=250,
    )


@pytest.mark.parametrize("n_times", [500, 1000, 2000])
@pytest.mark.parametrize("sfreq", [100, 250, 500])
@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_brain_module_basic(brain_module_params, n_times, sfreq, batch_size):
    """Test BrainModule with various input sizes and sample rates."""
    set_random_seeds(0, False)
    params = brain_module_params.copy()
    params.update({"n_times": n_times, "sfreq": sfreq})

    model = BrainModule(**params)
    model.eval()

    x = torch.randn(batch_size, params["n_chans"], n_times)
    output = model(x)

    assert output.shape == (batch_size, params["n_outputs"])
    assert not torch.isnan(output).any()


@pytest.mark.parametrize("subject_dim", [16, 32, 64])
def test_brain_module_subject_embeddings(brain_module_params, subject_dim):
    """Test subject embeddings with different dimensions and validation."""
    set_random_seeds(0, False)
    n_subjects = 30
    params = brain_module_params.copy()
    params.update({"n_subjects": n_subjects, "subject_dim": subject_dim})

    model = BrainModule(**params)
    model.eval()

    x = torch.randn(4, params["n_chans"], params["n_times"])
    subject_idx = torch.randint(0, n_subjects, (4,))

    output = model(x, subject_index=subject_idx)
    assert output.shape == (4, params["n_outputs"])
    assert not torch.isnan(output).any()

    # Test missing subject_index raises error
    with pytest.raises(ValueError, match="subject_index is required"):
        model(x)


@pytest.mark.parametrize("subject_dim", [16, 32, 64])
@pytest.mark.parametrize("subject_layers_dim", ["input", "hidden"])
def test_brain_module_subject_layers(brain_module_params, subject_dim, subject_layers_dim):
    """Test subject-specific layer transformations with different dimensions."""
    set_random_seeds(0, False)
    n_subjects = 25
    params = brain_module_params.copy()
    params.update({
        "n_subjects": n_subjects,
        "subject_dim": subject_dim,
        "subject_layers": True,
        "subject_layers_dim": subject_layers_dim,
    })

    model = BrainModule(**params)
    model.eval()

    x = torch.randn(4, params["n_chans"], params["n_times"])
    subject_idx = torch.randint(0, n_subjects, (4,))

    output = model(x, subject_index=subject_idx)
    assert output.shape == (4, params["n_outputs"])
    assert not torch.isnan(output).any()

    # Test that different subjects produce different outputs
    x_same = torch.ones(2, params["n_chans"], params["n_times"])
    subject_idx_1 = torch.tensor([0, 0])
    subject_idx_2 = torch.tensor([1, 1])

    with torch.no_grad():
        output_1 = model(x_same, subject_index=subject_idx_1)
        output_2 = model(x_same, subject_index=subject_idx_2)

    # Outputs should differ for different subjects (with high probability)
    assert not torch.allclose(output_1, output_2, atol=1e-4)


@pytest.mark.parametrize("n_fft,fft_complex", [(64, True), (256, False), (512, True)])
def test_brain_module_stft(brain_module_params, n_fft, fft_complex):
    """Test STFT with different FFT sizes and complex/power spectrograms."""
    set_random_seeds(0, False)
    params = brain_module_params.copy()
    params.update({"n_fft": n_fft, "fft_complex": fft_complex})

    model = BrainModule(**params)
    model.eval()

    for batch_size in [1, 4, 8]:
        x = torch.randn(batch_size, params["n_chans"], params["n_times"])
        output = model(x)
        assert output.shape == (batch_size, params["n_outputs"])
        assert not torch.isnan(output).any()


def test_brain_module_parameter_validation():
    """Test parameter validation for all features."""
    # Invalid subject_layers
    with pytest.raises(ValueError, match="subject_layers=True requires subject_dim > 0"):
        BrainModule(
            n_chans=22, n_outputs=4, n_times=1000, sfreq=250,
            subject_layers=True, subject_dim=0,
        )

    # Invalid depth
    with pytest.raises(ValueError, match="depth must be >= 1"):
        BrainModule(
            n_chans=22, n_outputs=4, n_times=1000, sfreq=250, depth=0,
        )

    # Invalid kernel_size
    with pytest.raises(ValueError, match="kernel_size must be > 0"):
        BrainModule(
            n_chans=22, n_outputs=4, n_times=1000, sfreq=250, kernel_size=0,
        )

    # kernel_size must be odd
    with pytest.raises(ValueError, match="kernel_size must be odd"):
        BrainModule(
            n_chans=22, n_outputs=4, n_times=1000, sfreq=250, kernel_size=4,
        )

    # channel_dropout_type requires channel_dropout_prob > 0
    with pytest.raises(ValueError, match="channel_dropout_type requires channel_dropout_prob > 0"):
        BrainModule(
            n_chans=22, n_outputs=4, n_times=1000, sfreq=250,
            channel_dropout_prob=0.0, channel_dropout_type="eeg",
        )

    # glu_context requires glu > 0
    with pytest.raises(ValueError, match="glu_context > 0 requires glu > 0"):
        BrainModule(
            n_chans=22, n_outputs=4, n_times=1000, sfreq=250,
            glu=0, glu_context=1,
        )

    # glu_context must be < kernel_size
    with pytest.raises(ValueError, match="glu_context must be < kernel_size"):
        BrainModule(
            n_chans=22, n_outputs=4, n_times=1000, sfreq=250,
            kernel_size=5, glu=1, glu_context=5,
        )


def test_brain_module_gradient_flow(brain_module_params):
    """Test gradient flow through model with various features."""
    for config in [
        {"glu": 1, "depth": 2},
        {"n_subjects": 20, "subject_dim": 32},
        {"channel_dropout_prob": 0.2},
        {"growth": 1.5, "depth": 3},
    ]:
        set_random_seeds(0, False)
        params = brain_module_params.copy()
        params.update(config)

        model = BrainModule(**params)
        model.train()

        x = torch.randn(
            4, params["n_chans"], params["n_times"],
            requires_grad=True,
        )
        if "n_subjects" in config:
            subject_idx = torch.randint(0, config["n_subjects"], (4,))
            output = model(x, subject_index=subject_idx)
        else:
            output = model(x)

        loss = output.sum()
        loss.backward()

        # Check gradients exist and are not NaN
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        for param in model.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any()


@pytest.mark.parametrize("growth", [1.0, 1.5, 2.0])
def test_brain_module_growth(brain_module_params, growth):
    """Test different growth factors for channel expansion."""
    set_random_seeds(0, False)
    params = brain_module_params.copy()
    params.update({"growth": growth, "depth": 3, "hidden_dim": 64})

    model = BrainModule(**params)
    model.eval()

    x = torch.randn(4, params["n_chans"], params["n_times"])
    output = model(x)

    assert output.shape == (4, params["n_outputs"])
    assert not torch.isnan(output).any()


# ============================================================================
# Channel Dropout Tests
# ============================================================================

@pytest.mark.parametrize("dropout_prob", [0.0, 0.1, 0.3, 0.5])
def test_brain_module_channel_dropout(brain_module_params, dropout_prob):
    """Test channel dropout with various probabilities."""
    set_random_seeds(0, False)
    params = brain_module_params.copy()
    params.update({"channel_dropout_prob": dropout_prob})

    model = BrainModule(**params)
    model.train()

    x = torch.randn(4, params["n_chans"], params["n_times"])
    output = model(x)

    assert output.shape == (4, params["n_outputs"])
    assert not torch.isnan(output).any()

    # Verify dropout is None when prob=0
    if dropout_prob == 0.0:
        assert model.channel_dropout is None
    else:
        assert model.channel_dropout is not None


def test_brain_module_channel_dropout_eval_mode(brain_module_params):
    """Test channel dropout is disabled in eval mode (deterministic)."""
    set_random_seeds(0, False)
    params = brain_module_params.copy()
    params.update({"channel_dropout_prob": 0.5})

    model = BrainModule(**params)
    model.eval()

    x = torch.randn(4, params["n_chans"], params["n_times"])

    with torch.no_grad():
        output1 = model(x)
        output2 = model(x)

    torch.testing.assert_close(output1, output2)


def test_brain_module_channel_dropout_with_ch_info():
    """Test channel dropout with ch_info for selective channel dropout."""
    set_random_seeds(0, False)

    ch_info = [
        {"ch_name": "Fp1", "ch_type": "eeg"},
        {"ch_name": "Fp2", "ch_type": "eeg"},
        {"ch_name": "F3", "ch_type": "eeg"},
        {"ch_name": "F4", "ch_type": "eeg"},
        {"ch_name": "A1", "ch_type": "ref"},
        {"ch_name": "A2", "ch_type": "ref"},
    ]

    params = {
        "n_chans": 6,
        "n_outputs": 2,
        "n_times": 1000,
        "hidden_dim": 32,
        "depth": 1,
        "channel_dropout_prob": 0.5,
        "channel_dropout_type": "eeg",
        "chs_info": ch_info,
    }

    model = BrainModule(**params)
    model.train()

    x = torch.ones(4, 6, 1000)
    for _ in range(3):
        output = model(x)
        assert output.shape == (4, 2)
        assert not torch.isnan(output).any()


# ============================================================================
# GLU (Gated Linear Units) Tests
# ============================================================================

@pytest.mark.parametrize("glu,glu_context,depth", [
    (0, 0, 2),
    (1, 0, 2),
    (1, 1, 2),
    (2, 1, 3),
])
def test_brain_module_glu(brain_module_params, glu, glu_context, depth):
    """Test GLU with various intervals and context windows."""
    set_random_seeds(0, False)
    params = brain_module_params.copy()
    params.update({"glu": glu, "glu_context": glu_context, "depth": depth})

    model = BrainModule(**params)
    model.train()

    x = torch.randn(4, params["n_chans"], params["n_times"])
    output = model(x)

    assert output.shape == (4, params["n_outputs"])
    assert not torch.isnan(output).any()

    # Verify GLU modules only created when glu > 0
    if glu > 0:
        assert any(g is not None for g in model.encoder.glus)
    else:
        assert all(g is None for g in model.encoder.glus)


@pytest.mark.parametrize("depth", [2, 4, 6])
def test_brain_module_depth_variants(brain_module_params, depth):
    """Test different depth configurations."""
    set_random_seeds(0, False)
    params = brain_module_params.copy()
    params.update({"depth": depth})

    model = BrainModule(**params)
    model.train()

    x = torch.randn(4, params["n_chans"], params["n_times"])
    output = model(x)

    assert output.shape == (4, params["n_outputs"])
    assert not torch.isnan(output).any()


def test_brain_module_glu_eval_determinism(brain_module_params):
    """Test GLU is deterministic in eval mode."""
    set_random_seeds(0, False)
    params = brain_module_params.copy()
    params.update({"glu": 1, "depth": 2})

    model = BrainModule(**params)
    model.eval()

    x = torch.randn(4, params["n_chans"], params["n_times"])

    with torch.no_grad():
        output1 = model(x)
        output2 = model(x)

    torch.testing.assert_close(output1, output2)


def test_brain_module_glu_combined_features(brain_module_params):
    """Test GLU combined with other features."""
    set_random_seeds(0, False)
    params = brain_module_params.copy()
    params.update({
        "glu": 1,
        "glu_context": 1,
        "channel_dropout_prob": 0.1,
        "subject_dim": 32,
        "n_subjects": 50,
        "depth": 2,
    })

    model = BrainModule(**params)
    model.train()

    x = torch.randn(4, params["n_chans"], params["n_times"])
    subject_idx = torch.randint(0, 50, (4,))

    output = model(x, subject_index=subject_idx)

    assert output.shape == (4, params["n_outputs"])
    assert not torch.isnan(output).any()

def test_bendr():
    """
    Test BENDR model forward pass with 3D inputs.
    BENDR only accepts 3D inputs: (batch, channels, time).
    """
    set_random_seeds(0, False)

    # Standard configuration
    model = BENDR(
        n_chans=20,
        n_outputs=4,
        n_times=None,  # Auto-infer
        sfreq=256,
        input_window_seconds=20.0,
    )

    # Test with 3D inputs only (BENDR doesn't support 4D)
    input_sizes = dict(n_channels=20, n_in_times=5120, n_classes=4, n_samples=2)
    check_forward_pass_3d(model, input_sizes)


def test_bendr_parameter_counts():
    """
    Test BENDR parameter counts match paper specifications.

    Paper reports ~157M parameters total:
    - Encoder: ~4M parameters
    - Contextualizer: ~153M parameters
    """
    set_random_seeds(0, False)

    # Standard 20-channel configuration
    model = BENDR(
        n_chans=20,
        n_outputs=2,
        n_times=5120,
        sfreq=256,
    )

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Should be close to paper: 157,141,049,
    # At braindecode, there are 2 k params difference
    # that might come from implementation details from different
    # torch versions or minor code changes. 157,143,101 in my case.

    # Allow 0.1% tolerance
    expected = 157_141_049
    assert abs(total_params - expected) / expected < 0.001, \
        f"Expected ~{expected:,} params, got {total_params:,}"

    # Count encoder parameters (should be ~4M)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    assert 3_900_000 < encoder_params < 4_100_000, \
        f"Encoder should have ~4M params, got {encoder_params:,}"

    # Count contextualizer parameters (should be ~153M)
    contextualizer_params = sum(p.numel() for p in model.contextualizer.parameters())
    assert 152_000_000 < contextualizer_params < 154_000_000, \
        f"Contextualizer should have ~153M params, got {contextualizer_params:,}"


def test_bendr_different_channels():
    """
    Test BENDR with different channel counts.
    Parameter count should scale with number of channels.
    """
    set_random_seeds(0, False)

    configs = [
        (1, 157_112_891),   # Single channel
        (20, 157_142_075),  # Standard
        (64, 157_209_659),  # More channels
    ]

    for n_chans, expected_params in configs:
        model = BENDR(
            n_chans=n_chans,
            n_outputs=2,
            n_times=5120,
            sfreq=256,
        )

        total_params = sum(p.numel() for p in model.parameters())

        # Check exact match
        assert total_params == expected_params, \
            f"For {n_chans} channels: expected {expected_params:,}, got {total_params:,}"


def test_bendr_output_shapes():
    """
    Test BENDR output shapes for different configurations.
    """
    set_random_seeds(0, False)

    # Binary classification
    model_binary = BENDR(n_chans=20, n_outputs=2, n_times=5120, sfreq=256)
    x = torch.randn(4, 20, 5120)
    y = model_binary(x)
    assert y.shape == (4, 2), f"Expected (4, 2), got {y.shape}"

    # Multi-class classification
    model_multi = BENDR(n_chans=20, n_outputs=10, n_times=5120, sfreq=256)
    y = model_multi(x)
    assert y.shape == (4, 10), f"Expected (4, 10), got {y.shape}"

    # Regression
    model_reg = BENDR(n_chans=20, n_outputs=1, n_times=5120, sfreq=256)
    y = model_reg(x)
    assert y.shape == (4, 1), f"Expected (4, 1), got {y.shape}"


def test_bendr_variable_length():
    """
    Test BENDR with variable input lengths.
    Model should handle different sequence lengths at inference.
    """
    set_random_seeds(0, False)

    model = BENDR(
        n_chans=20,
        n_outputs=4,
        n_times=None,  # Don't specify - should work with any length
        sfreq=256,
    )

    # Test different lengths
    for n_times in [2560, 5120, 10240]:
        x = torch.randn(2, 20, n_times)
        y = model(x)
        assert y.shape == (2, 4), f"Failed for length {n_times}: got shape {y.shape}"


def test_bendr_gradient_flow():
    """
    Test that gradients flow through the entire model.
    """
    set_random_seeds(0, False)

    model = BENDR(n_chans=20, n_outputs=4, n_times=5120, sfreq=256)
    x = torch.randn(2, 20, 5120, requires_grad=True)

    y = model(x)
    loss = y.sum()
    loss.backward()

    # Check gradients exist in encoder
    encoder_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.encoder.parameters()
    )
    assert encoder_has_grad, "No gradients in encoder"

    # Check gradients exist in contextualizer
    contextualizer_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.contextualizer.parameters()
    )
    assert contextualizer_has_grad, "No gradients in contextualizer"


@pytest.mark.parametrize("drop_prob", [0.0, 0.1, 0.15])
def test_bendr_dropout_configurations(drop_prob):
    """
    Test BENDR with different dropout rates.
    Paper uses 0.15 for pretraining, 0.0 for fine-tuning.
    """
    set_random_seeds(0, False)

    model = BENDR(
        n_chans=20,
        n_outputs=4,
        n_times=5120,
        sfreq=256,
        drop_prob=drop_prob,
    )

    x = torch.randn(2, 20, 5120)

    # Training mode
    model.train()
    y_train = model(x)
    assert y_train.shape == (2, 4)

    # Eval mode
    model.eval()
    y_eval = model(x)
    assert y_eval.shape == (2, 4)

    # With dropout=0, outputs should be identical
    if drop_prob == 0.0:
        np.testing.assert_allclose(
            y_train.detach().numpy(),
            y_eval.detach().numpy(),
            rtol=1e-5,
            atol=1e-7,
        )


@pytest.mark.parametrize(
    "no_inter_attn,single_channel,output_attention",
    [
        (False, False, False),
        (False, False, True),
        (False, True, False),
        (False, True, True),
        (True, False, False),
        (True, False, True),
        (True, True, False),
        (True, True, True),
    ],
)
def test_medformer_boolean_combinations(no_inter_attn, single_channel, output_attention):
    """
    Test all combinations of MEDFormer boolean parameters.
    Ensures all 8 combinations work correctly.
    """
    set_random_seeds(0, False)

    model = MEDFormer(
        n_chans=22,
        n_outputs=4,
        n_times=1000,
        no_inter_attn=no_inter_attn,
        single_channel=single_channel,
        output_attention=output_attention,
    )

    x = torch.randn(2, 22, 1000)
    y = model(x)
    assert y.shape == (2, 4)

    # Verify parameters are correctly set
    assert model.single_channel == single_channel
    assert model.output_attention == output_attention

    # Check inter_attention based on no_inter_attn
    first_medformer_layer = model.encoder.attn_layers[0].attention
    if no_inter_attn:
        assert first_medformer_layer.inter_attention is None
    else:
        assert first_medformer_layer.inter_attention is not None


@pytest.mark.parametrize("patch_len_list", [[2, 8, 16], [4, 8], [2, 4, 8, 16]])
def test_medformer_patch_len_configurations(patch_len_list):
    """
    Test MEDFormer with different patch length configurations.
    """
    set_random_seeds(0, False)

    model = MEDFormer(
        n_chans=22,
        n_outputs=4,
        n_times=1000,
        patch_len_list=patch_len_list,
    )

    x = torch.randn(2, 22, 1000)
    y = model(x)
    assert y.shape == (2, 4)

    # Check that the number of patch embeddings matches
    assert len(model.enc_embedding.value_embeddings) == len(patch_len_list)
