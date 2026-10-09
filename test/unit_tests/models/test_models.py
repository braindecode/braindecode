# Authors: Alexandre Gramfort
#          Lukas Gemein <l.gemein@gmail.com>
#          Hubert Banville <hubert.jbanville@gmail.com>
#          Robin Schirrmeister <robintibor@gmail.com>
#          Daniel Wilson <dan.c.wil@gmail.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#          Matthew Chen <matt.chen42601@gmail.com>
#          Sarthak Tayal <sarthaktayal2@gmail.com>
#
# License: BSD-3

import inspect
import re
import warnings
from collections import OrderedDict
from functools import partial
from unittest import mock

import mne
import numpy as np
import pytest
import torch
from scipy.signal import stft
from sklearn.utils import check_random_state
from torch import nn

from braindecode.models import (
    BDTCN,
    BENDR,
    BIOT,
    DGCNN,
    EEGCLIP,
    EEGPT,
    SSTDPN,
    TCN,
    ATCNet,
    AttentionBaseNet,
    AttnSleep,
    BaRISTA,
    BrainBERT,
    BrainModule,
    Brant,
    CBraMod,
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
    EMG2QwertyNet,
    FBCNet,
    FBLightConvNet,
    FBMSNet,
    HybridNet,
    IFNet,
    Labram,
    MEDFormer,
    MetaNeuromotorHand,
    NeuroRVQ,
    NeuroRVQTokenizer,
    SCCNet,
    ShallowFBCSPNet,
    SleepStagerBlanco2020,
    SleepStagerChambon2018,
    SPARCNet,
    SyncNet,
    TFMTokenizer,
    TIDNet,
    TMSANet,
    TSception,
    USleep,
)
from braindecode.models.brainbert import _STFTSpectrogram
from braindecode.models.csbrain import (
    REGION_CENTRAL,
    REGION_FRONTAL,
    REGION_OCCIPITAL,
    REGION_PARIETAL,
    REGION_TEMPORAL,
    CSBrain,
    build_region_attention_mask,
    derive_brain_regions,
    make_area_config,
    region_of_electrode,
)
from braindecode.models.eegpt import (
    _apply_rotary_emb,
    _Attention,
    _EEGTransformer,
    _PatchEmbed,
    _rotate_half,
)
from braindecode.models.labram import LABRAM_CHANNEL_ORDER
from braindecode.models.neurorvq_tokenizer import _EMAVectorQuantizer
from braindecode.models.usleep import _DecoderBlock
from braindecode.models.util import (
    _get_possible_signal_params,
    _get_signal_params,
    models_dict,
    models_mandatory_parameters,
)
from braindecode.util import set_random_seeds

all_models_dict = dict(models_dict)


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



def test_shallow_fbcsp_net_split_first_layer_without_batch_norm(input_sizes):
    model = ShallowFBCSPNet(
        input_sizes["n_channels"],
        input_sizes["n_classes"],
        input_sizes["n_in_times"],
        final_conv_length="auto",
        split_first_layer=True,
        batch_norm=False,
    )

    assert torch.count_nonzero(model.conv_time_spat.conv_time.bias) == 0
    assert torch.count_nonzero(model.conv_time_spat.conv_spat.bias) == 0


def test_shallow_fbcsp_net_without_split_first_layer(input_sizes):
    model = ShallowFBCSPNet(
        input_sizes["n_channels"],
        input_sizes["n_classes"],
        input_sizes["n_in_times"],
        final_conv_length="auto",
        split_first_layer=False,
    )

    assert hasattr(model, "conv_time")
    assert not hasattr(model, "conv_time_spat")
    check_forward_pass(model, input_sizes)


def test_shallow_fbcsp_net_without_split_first_layer_without_batch_norm(
    input_sizes,
):
    model = ShallowFBCSPNet(
        input_sizes["n_channels"],
        input_sizes["n_classes"],
        input_sizes["n_in_times"],
        final_conv_length="auto",
        split_first_layer=False,
        batch_norm=False,
    )

    assert hasattr(model, "conv_time")
    assert model.conv_time.bias is not None
    assert torch.count_nonzero(model.conv_time.bias) == 0
    check_forward_pass(model, input_sizes)


def test_shallow_fbcsp_net_without_split_first_layer_loads_legacy_state_dict(
    input_sizes,
):
    model = ShallowFBCSPNet(
        input_sizes["n_channels"],
        input_sizes["n_classes"],
        input_sizes["n_in_times"],
        final_conv_length="auto",
        split_first_layer=False,
    )
    state_dict = OrderedDict()
    for key, value in model.state_dict().items():
        legacy_key = key.replace("final_layer.conv_classifier", "conv_classifier")
        state_dict[legacy_key] = value.clone()

    model.load_state_dict(state_dict)

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


def test_deep4net_short_input_scales_split_temporal_kernel():
    with pytest.warns(UserWarning, match="smaller than the minimum required"):
        model = Deep4Net(
            n_chans=18,
            n_outputs=2,
            n_times=300,
            final_conv_length=1,
        )

    assert model.filter_time_length < 10
    assert model.conv_time_spat.conv_time.kernel_size[0] == model.filter_time_length

    with torch.no_grad():
        y = model(torch.randn(2, 18, 300))


def test_deep4net_explicit_final_conv_length_without_n_times():
    model = Deep4Net(
        n_chans=18,
        n_outputs=2,
        n_times=None,
        final_conv_length=1,
    )
    x = torch.randn(2, 18, 600)

    with torch.no_grad():
        y = model(x)

    assert y.shape[:2] == (2, 2)




def test_deep4net_without_split_first_layer(input_sizes):
    model = Deep4Net(
        input_sizes["n_channels"],
        input_sizes["n_classes"],
        input_sizes["n_in_times"],
        final_conv_length="auto",
        split_first_layer=False,
    )

    assert hasattr(model, "conv_time")
    assert not hasattr(model, "conv_time_spat")
    check_forward_pass(model, input_sizes)


def test_deep4net_without_split_first_layer_loads_legacy_state_dict(input_sizes):
    model = Deep4Net(
        input_sizes["n_channels"],
        input_sizes["n_classes"],
        input_sizes["n_in_times"],
        final_conv_length="auto",
        split_first_layer=False,
    )
    state_dict = OrderedDict()
    for key, value in model.state_dict().items():
        legacy_key = key.replace("final_layer.conv_classifier", "conv_classifier")
        state_dict[legacy_key] = value.clone()

    model.load_state_dict(state_dict)


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




def test_deep4net_stride_before_pool_dense_geometry():
    """Keep the pre-CombinedConv stride semantics used by old checkpoints."""
    model = Deep4Net(
        n_chans=21,
        n_outputs=128,
        n_times=1000,
        final_conv_length=2,
        stride_before_pool=True,
    ).eval()

    assert model.conv_time_spat.conv_spat.stride == (3, 1)
    assert model.pool.stride == (1, 1)

    model.to_dense_prediction_model()
    out = model(torch.randn(2, 21, 1000))

    assert out.shape == (2, 128, 319)


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


def test_tcn_with_unit_kernel(input_sizes):
    model = TCN(
        n_chans=input_sizes["n_channels"],
        n_outputs=input_sizes["n_classes"],
        n_filters=5,
        n_blocks=2,
        kernel_size=1,
        drop_prob=0.0,
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


def _eegpt(names, **kwargs):
    from braindecode.models.eegpt import EEGPT

    return EEGPT(
        n_outputs=2,
        n_times=600,
        chs_info=[{"ch_name": name} for name in names],
        chan_proj_type="none",
        **kwargs,
    ).eval()


def test_eegpt_chans_id_is_not_saved():
    # The channel IDs are derived from the montage at construction.
    model = _eegpt(["C3", "CZ", "C4"])
    assert "chans_id" not in model.state_dict()


@pytest.mark.parametrize(
    "target_names",
    [
        ["C3", "CZ", "C4"],  # fewer channels than the checkpoint
        ["P4", "PZ", "CP2", "C4", "CZ", "C3"],  # same channels, other order
    ],
)
def test_eegpt_checkpoint_keeps_the_model_channel_ids(target_names):
    # Checkpoints saved before chans_id became non-persistent, like the released
    # EEGPT weights, still store it. Loading one must neither fail on another
    # montage nor replace the IDs of the model's channels by the checkpoint's.
    source = _eegpt(["C3", "CZ", "C4", "CP2", "PZ", "P4"])
    state_dict = dict(source.state_dict(), chans_id=source.chans_id.clone())
    target = _eegpt(target_names)
    expected = target.chans_id.clone()

    target.load_state_dict(state_dict)

    assert torch.equal(target.chans_id, expected)
    assert torch.equal(
        target.target_encoder.chan_embed.weight,
        source.target_encoder.chan_embed.weight,
    )


def test_eegpt_checkpoint_loads_with_channel_projection():
    # The default channel projection gives 19 channel IDs; the released
    # checkpoint was saved without a projection and stores 62.
    from braindecode.models.eegpt import EEGPT

    source = _eegpt(["C3", "CZ", "C4", "CP2", "PZ", "P4"])
    state_dict = dict(source.state_dict(), chans_id=source.chans_id.clone())
    target = EEGPT(n_outputs=2, n_chans=6, n_times=600).eval()

    incompatible = target.load_state_dict(state_dict, strict=False)

    assert not incompatible.unexpected_keys
    assert all(key.startswith("chan_proj.") for key in incompatible.missing_keys)
    assert target.chans_id.shape == (1, 19)


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


@pytest.mark.parametrize("dtype, rtol", [(torch.float32, 1e-5), (torch.float64, 1e-10)])
def test_eeginception_mi_fft_conv(dtype, rtol):
    """The FFT path equals the direct convolution up to rounding (outputs and
    gradients); on CPU the default takes the direct one on a small input and
    the FFT one past the crossover."""
    kw = dict(n_chans=3, n_outputs=2, n_times=256, sfreq=128, n_filters=8)
    models = {}
    for mode in (False, True, None):
        torch.manual_seed(0)
        models[mode] = EEGInceptionMI(fft_conv=mode, **kw).to(dtype).eval()
    x = torch.randn(8, 3, 256, dtype=dtype)
    outs, grads = {}, {}
    for mode in (False, True):
        outs[mode] = models[mode](x)
        outs[mode].square().sum().backward()
        grads[mode] = torch.cat([p.grad.flatten() for p in models[mode].parameters()])
    for got, ref in ((outs[True], outs[False]), (grads[True], grads[False])):
        assert (got - ref).abs().max() <= rtol * ref.abs().max()
    with torch.no_grad():  # largest kernel: 108 samples
        assert torch.equal(models[None](x[:1]), models[False](x[:1]))
        assert torch.equal(models[None](x), models[True](x))


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


def test_usleep_decoder_crop_returns_prefix_views():
    """Decoder alignment avoids allocating device-specific index tensors."""
    longer = torch.arange(30).reshape(2, 3, 5)
    shorter = torch.arange(24).reshape(2, 3, 4)

    cropped_longer, cropped_shorter = _DecoderBlock._crop_tensors_to_match(
        longer, shorter
    )

    torch.testing.assert_close(cropped_longer, longer[..., :4])
    torch.testing.assert_close(cropped_shorter, shorter)
    assert (
        cropped_longer.untyped_storage().data_ptr()
        == longer.untyped_storage().data_ptr()
    )
    assert (
        cropped_shorter.untyped_storage().data_ptr()
        == shorter.untyped_storage().data_ptr()
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


def test_attn_sleep_requires_signal_geometry():
    with pytest.raises(ValueError, match="at least two of"):
        AttnSleep(sfreq=100, n_outputs=5)


def test_attn_sleep_reports_required_d_model():
    with pytest.raises(ValueError, match="d_model=54"):
        AttnSleep(sfreq=100, n_outputs=5, n_times=2000)


def test_attn_sleep_rejects_invalid_attention_heads():
    with pytest.raises(ValueError, match="positive integer that divides d_model"):
        AttnSleep(sfreq=100, n_outputs=5, n_times=3000, n_attn_heads=7)


@pytest.mark.parametrize("return_feats", [False, True])
def test_attn_sleep_supports_other_window_lengths(return_feats):
    model = AttnSleep(
        sfreq=100,
        n_outputs=5,
        n_times=2000,
        d_model=54,
        n_attn_heads=6,
        return_feats=return_feats,
    )
    output = model(torch.randn(2, 1, 2000))
    expected_width = model.len_last_layer if return_feats else 5
    assert output.shape == (2, expected_width)


def test_attn_sleep_activation_reaches_afr():
    model = AttnSleep(
        sfreq=100, n_outputs=5, n_times=3000, activation=nn.GELU
    )
    activations = [
        module
        for module in model.feature_extractor[0].AFR.modules()
        if isinstance(module, (nn.ReLU, nn.GELU))
    ]
    assert activations
    assert all(isinstance(module, nn.GELU) for module in activations)


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

    model = DeepSleepNet(
        n_outputs=n_classes, return_feats=False,
        n_chans=n_channels, n_times=int(np.ceil(input_size_s * sfreq)),
    )
    model.eval()

    rng = np.random.RandomState(42)
    X = rng.randn(n_examples, n_channels,
                  int(np.ceil(input_size_s * sfreq)))
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

    model = DeepSleepNet(
        n_outputs=n_classes, return_feats=True,
        n_chans=n_channels, n_times=int(sfreq * input_size_s),
    )
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

    model = DeepSleepNet(
        n_outputs=n_classes, return_feats=False,
        n_chans=n_channels, n_times=int(sfreq * input_size_s),
    )
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


@pytest.mark.parametrize(
    "n_chans, n_times, n_outputs",
    [
        (64, 500, 1),
        (2, 3000, 5),
        (22, 1000, 3),
    ],
)
def test_deepsleepnet_variable_input(n_chans, n_times, n_outputs):
    # deepsleepnet should work with different input shapes not just 1ch 3000t
    model = DeepSleepNet(
        n_chans=n_chans, n_outputs=n_outputs, n_times=n_times,
    )
    model.eval()
    x = torch.randn(2, n_chans, n_times)
    out = model(x)
    assert out.shape == (2, n_outputs)


@pytest.mark.parametrize(
    "bilstm_hidden_size, bilstm_num_layers, drop_prob, return_feats",
    [
        (256, 1, 0.3, True),
        (512, 2, 0.5, False),
        (128, 3, 0.0, False),
    ],
)
def test_deepsleepnet_custom_params(
    bilstm_hidden_size, bilstm_num_layers, drop_prob, return_feats
):
    model = DeepSleepNet(
        n_chans=1, n_outputs=5, n_times=3000,
        bilstm_hidden_size=bilstm_hidden_size,
        bilstm_num_layers=bilstm_num_layers,
        drop_prob=drop_prob,
        return_feats=return_feats,
    )
    model.eval()
    out = model(torch.randn(2, 1, 3000))
    expected_feats = bilstm_hidden_size * 2
    assert model.len_last_layer == expected_feats
    if return_feats:
        assert out.shape == (2, expected_feats)
    else:
        assert out.shape == (2, 5)


def test_deepsleepnet_custom_cnn_params():
    model = DeepSleepNet(
        n_chans=1, n_outputs=5, n_times=3000,
        small_n_filters_1=32, small_n_filters_2=64,
        large_n_filters_1=32, large_n_filters_2=64,
    )
    model.eval()
    assert model(torch.randn(2, 1, 3000)).shape == (2, 5)
    assert model.cnn1.conv1[0].out_channels == 32
    assert model.cnn1.conv2[0].out_channels == 64
    assert model.cnn2.conv1[0].out_channels == 32
    assert model.cnn2.conv2[0].out_channels == 64


def test_deepsleepnet_too_small_ntimes():
    with pytest.raises(ValueError, match="n_times=10 is too small"):
        DeepSleepNet(n_chans=1, n_outputs=5, n_times=10)


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
    "n_chans,n_times,n_outputs,embed_dim,att_drop_prob",
    [
        (22, 1000, 4, 19, 0.5),  # BCI Competition IV 2a
        (3, 1000, 2, 6, 0.5),  # BCI Competition IV 2b
        (44, 1125, 4, 10, 0.7),  # HGD
    ],
)
def test_tmsanet_released_configurations(
    n_chans, n_times, n_outputs, embed_dim, att_drop_prob
):
    model = TMSANet(
        n_chans=n_chans,
        n_times=n_times,
        n_outputs=n_outputs,
        embed_dim=embed_dim,
        att_drop_prob=att_drop_prob,
    ).eval()
    # Released head width embed_dim // num_heads: 19 -> 16 -> 19 for 2a.
    assert model.transformer[1].attention.w_q.out_features == embed_dim // 4 * 4
    assert model(torch.randn(2, n_chans, n_times)).shape == (2, n_outputs)


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
    # 64 windows held 6.3 GB of activations; a macOS runner has 7 GB for 3 workers.
    batch_size = 2
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

def _make_chs_info(n_chans):
    """Create synthetic chs_info with 3-D positions for testing."""
    # Use a standard montage and pick the first n_chans channels
    montage = mne.channels.make_standard_montage("standard_1005")
    ch_names = montage.ch_names[:n_chans]
    info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types="eeg")
    info.set_montage(montage)
    return info["chs"]


@pytest.mark.parametrize(
    "n_times, n_chans, sfreq, n_outputs",
    [
        (128, 62, 256.0, 4),
        (256, 32, 512.0, 2),
    ],
)
def test_dgcnn_dummy(n_times, n_chans, sfreq, n_outputs):
    batch_size = 8
    input_sizes = dict(
        n_channels=n_chans,
        n_in_times=n_times,
        n_classes=n_outputs,
        n_samples=batch_size,
    )
    chs_info = _make_chs_info(n_chans)
    model = DGCNN(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        sfreq=sfreq,
        chs_info=chs_info,
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


# Every channel count, output size and window length once, not their 240-case product.
@pytest.mark.parametrize(
    "n_chans, n_outputs, input_size_s",
    [(1, 2, 1), (2, 3, 2), (4, 4, 5), (8, 5, 10), (16, 50, 15), (32, 2, 30),
     (64, 3, 1), (128, 4, 2)],
)
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


def test_biot_encoder_index_is_buffer(default_biot_params):
    biot = BIOT(**default_biot_params)

    assert "index" in dict(biot.encoder.named_buffers())
    assert "index" not in dict(biot.encoder.named_parameters())
    assert biot.encoder.index.dtype == torch.long


@pytest.fixture
def default_labram_params():
    return {
        "n_times": 1000,
        "n_chans": 128,
        "chs_info": [{"ch_name": ch_name} for ch_name in LABRAM_CHANNEL_ORDER],
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

        # 128 channels * 5 patches (1000 / 200) = 640 patch tokens
        assert out_patches.shape == torch.Size(
            [1, 640, default_labram_params["n_outputs"]]
        )

        out_all_tokens = labram_base(X, return_all_tokens=True,
                                     return_patch_tokens=False)
        # 1 cls token + 640 patch tokens = 641
        assert out_all_tokens.shape == torch.Size(
            [1, 641, default_labram_params["n_outputs"]]
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
    # 4.27 K according to Table V from the original paper. With the
    # source-faithful TCN BatchNorm (default), braindecode reports 4.3 K, which
    # is closer to the paper than the 4.2 K of the previous (BN-less) variant.
    assert np.round(n_params / 1e3, 1) == 4.3


def test_eegtcnet_tcn_batchnorm():
    """TCN residual blocks carry source-faithful BatchNorm by default, and the
    legacy (BN-less) architecture is recoverable with ``tcn_batch_norm=False``."""
    model = EEGTCNet(n_outputs=4, n_chans=22, n_times=1125)
    n_bn = sum(
        isinstance(m, nn.BatchNorm1d) for m in model.tcn_block.modules()
    )
    # one BatchNorm per conv, two convs per residual block.
    assert n_bn == 2 * model.depth
    # the source uses a bias on the residual 1x1 downsample.
    assert model.tcn_block.downsample.bias is not None

    legacy = EEGTCNet(n_outputs=4, n_chans=22, n_times=1125, tcn_batch_norm=False)
    assert not any(
        isinstance(m, nn.BatchNorm1d) for m in legacy.tcn_block.modules()
    )
    assert legacy.tcn_block.downsample.bias is None
    # both variants still produce valid logits.
    x = torch.randn(2, 22, 1125)
    assert model(x).shape == (2, 4)
    assert legacy(x).shape == (2, 4)


def test_eegtcnet_separate_dropout():
    """EEGNet and TCN dropout rates can be set independently (source uses
    p_eeg=0.2, p_tcn=0.3)."""
    model = EEGTCNet(
        n_outputs=4,
        n_chans=22,
        n_times=1125,
        drop_prob_eeg=0.2,
        drop_prob_tcn=0.3,
    )
    assert model.eegnet_tc.drop1.p == 0.2
    assert model.eegnet_tc.drop2.p == 0.2
    tcn_drops = [
        m.p for m in model.tcn_block.modules() if isinstance(m, nn.Dropout)
    ]
    assert tcn_drops == [0.3] * (2 * model.depth)

    # When unset, both fall back to the single ``drop_prob``.
    default = EEGTCNet(n_outputs=4, n_chans=22, n_times=1125, drop_prob=0.4)
    assert default.eegnet_tc.drop1.p == 0.4
    assert all(
        m.p == 0.4
        for m in default.tcn_block.modules()
        if isinstance(m, nn.Dropout)
    )


def test_eegtcnet_batch_size_one_train_mode():
    """Batch-size-1 forward in train mode must not raise even when the TCN
    sequence collapses to length 1 (small n_times), now that BatchNorm is on by
    default. With n_times=64 the EEGNet front-end reduces time to a single TCN
    step, so the (1, filters, 1) tensor would break BatchNorm1d without the
    batch-size-one guard."""
    model = EEGTCNet(n_outputs=4, n_chans=22, n_times=64).train()
    out = model(torch.randn(1, 22, 64))
    assert out.shape == (1, 4)


def test_sstdpn_proto_sep_constrains_class_rows():
    """``proto_sep`` is renormalized per class-row (dim=0), so each prototype
    vector has L2 norm <= ``proto_sep_maxnorm`` after a forward pass."""
    model = SSTDPN(n_outputs=4, n_chans=22, n_times=1000)
    model.proto_sep.data.fill_(10.0)
    model(torch.randn(2, 22, 1000))
    row_norms = model.proto_sep.detach().norm(p=2, dim=1)
    # float32 renorm leaves a ~1e-6 residual; use a 1e-5 tolerance.
    assert torch.all(row_norms <= model.proto_sep_maxnorm + 1e-5)


def test_sstdpn_proto_cpt_std_default():
    """ICP prototypes default to the source ``torch.randn`` std (1.0)."""
    assert SSTDPN(n_outputs=4, n_chans=22, n_times=1000).proto_cpt_std == 1.0


def test_atcnet_conv_max_norm():
    """``conv_max_norm_const`` clamps the conv/TCN kernels per output filter,
    while the default (``None``) adds no parameters and no constraint."""
    max_norm = 0.6
    model = ATCNet(n_chans=22, n_outputs=4, n_times=1125, conv_max_norm_const=max_norm)
    model(torch.randn(2, 22, 1125))  # apply the parametrization

    def max_filter_norm(weight):
        return weight.reshape(weight.shape[0], -1).norm(p=2, dim=1).max().item()

    assert max_filter_norm(model.conv_block.conv1.weight) <= max_norm + 1e-5
    assert max_filter_norm(model.conv_block.conv2.weight) <= max_norm + 1e-5
    assert (
        max_filter_norm(model.temporal_conv_nets[0][0].conv1.weight)
        <= max_norm + 1e-5
    )

    # Default leaves the architecture (and parameter count) untouched.
    default = ATCNet(n_chans=22, n_outputs=4, n_times=1125)
    assert sum(p.numel() for p in default.parameters()) == sum(
        p.numel() for p in model.parameters()
    )


@pytest.mark.parametrize("conv_max_norm", [None, 0.6])
def test_atcnet_source_optimizer_param_groups(conv_max_norm):
    """The helper returns conv/dense/other groups with the source weight
    decays, covering every parameter exactly once -- including when the conv
    kernels are max-norm parametrized (``conv_max_norm_const`` set), in which
    case the grouped entries must be the leaf ``.original`` parameters."""
    model = ATCNet(
        n_chans=22, n_outputs=4, n_times=1125, conv_max_norm_const=conv_max_norm
    )
    groups = model.source_optimizer_param_groups()

    assert [g["weight_decay"] for g in groups] == [0.009, 0.5, 0.0]
    grouped = [p for g in groups for p in g["params"]]
    # every parameter is covered exactly once (compare by identity)...
    assert {id(p) for p in grouped} == {id(p) for p in model.parameters()}
    assert len(grouped) == len(list(model.parameters()))
    # ...and each entry is a real trainable leaf (not a computed parametrized weight).
    assert all(p.is_leaf and p.requires_grad for p in grouped)
    assert all(len(g["params"]) > 0 for g in groups)
    # groups are accepted by a real optimizer.
    torch.optim.Adam(groups, lr=1e-3)


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


@pytest.mark.parametrize("method", ["mag", "corr", "plv"])
def test_eegminer_legacy_state_dict_compatibility(method):
    model_kwargs = {
        "method": method,
        "n_chans": 4,
        "n_outputs": 2,
        "n_times": 128,
        "sfreq": 100.0,
    }
    model = EEGMiner(**model_kwargs)
    state_dict = model.state_dict()
    legacy_keys = {
        "filter.n_range",
        "filter.f_mean",
        "filter.bandwidth",
        "filter.shape",
        "filter.group_delay",
        "batch_layer.running_mean",
        "batch_layer.running_var",
        "batch_layer.num_batches_tracked",
        "final_layer.weight",
        "final_layer.bias",
    }

    assert set(state_dict) == legacy_keys
    EEGMiner(**model_kwargs).load_state_dict(state_dict, strict=True)


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
    x = eegminer.feature_layer(x)

    # PLV values should be in [0, 1]
    assert torch.all(x >= 0.0) and torch.all(x <= 1.0), \
        "PLV values should be in the range [0, 1]"


_BATCH_SIZE_ONE_TRAIN_MODE_PARAMS = [
    pytest.param(model_name, required_params, signal_params, id=model_name)
    for model_name, required_params, signal_params in models_mandatory_parameters
]


@pytest.mark.parametrize(
    "model_name, required_params, signal_params", _BATCH_SIZE_ONE_TRAIN_MODE_PARAMS
)
def test_models_batch1_train_mode(
    model_name, required_params, signal_params
):
    """Models must accept batch_size=1 even in train mode.

    BatchNorm layers, when present, must also be restored to train mode
    after temporarily using running statistics for single-sample inputs.
    Multi-output models (e.g. tokenizers returning ``(target,
    reconstruction)``) must keep the batch dimension on every output.
    """

    def _assert_batch_one(out):
        outputs = out if isinstance(out, (tuple, list)) else (out,)
        assert all(o.shape[0] == 1 for o in outputs)

    sp = _get_signal_params(signal_params)
    model_kwargs = _get_possible_signal_params(sp, required_params)[0]
    model = all_models_dict[model_name](**model_kwargs)
    batch_norms = [
        module
        for module in model.modules()
        if isinstance(
            module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
        )
    ]
    x = torch.randn(1, sp["n_chans"], sp["n_times"])

    # In train mode with batch_size=1, this must not raise.
    model.train()
    assert model.training
    with torch.no_grad():
        out = model(x)
    _assert_batch_one(out)
    # Model and BatchNorm layers must be restored to train mode after forward.
    assert model.training
    assert all(batch_norm.training for batch_norm in batch_norms)

    # In eval mode with batch_size=1, this must also work.
    model.eval()
    with torch.no_grad():
        out = model(x)
    _assert_batch_one(out)


def test_batchnorm_decorator_preserves_forward_input_keyword():
    model = ContraWR(n_chans=3, n_outputs=2, n_times=1000, sfreq=200.0)
    x = torch.randn(1, model.n_chans, model.n_times)

    model.train()
    with torch.no_grad():
        out = model(X=x)

    assert out.shape == (1, model.n_outputs)


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


def test_fbmsnet_return_features():
    n_chans = 22
    n_times = 1000
    n_outputs = 4
    batch_size = 2
    default_n_filters_spat = 36
    default_dilatability = 8
    default_stride_factor = 4

    model = FBMSNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        sfreq=250,
        return_features=True,
    )
    model.eval()

    with torch.no_grad():
        logits, features = model(torch.randn(batch_size, n_chans, n_times))

    expected_feature_dim = model.out_channels_spatial * model.stride_factor
    assert logits.shape == (batch_size, n_outputs)
    assert features.shape == (batch_size, expected_feature_dim)
    assert expected_feature_dim == (
        default_n_filters_spat * default_dilatability * default_stride_factor
    )


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


@pytest.mark.parametrize("win_len, n_windows", [(100, 10), (250, 4), (500, 2)])
def test_fblightconvnet_win_len_sets_number_of_windows(win_len, n_windows):
    model = FBLightConvNet(
        n_chans=8,
        n_outputs=3,
        n_times=1000,
        sfreq=250,
        win_len=win_len,
    )
    assert model.attn_conv.kernel_size == n_windows


def test_fblightconvnet_stride_factor_is_deprecated_and_ignored():
    kwargs = dict(n_chans=8, n_outputs=3, n_times=1000, sfreq=250)

    set_random_seeds(2025, cuda=False)
    default = FBLightConvNet(**kwargs).eval()

    with pytest.warns(DeprecationWarning, match="stride_factor"):
        set_random_seeds(2025, cuda=False)
        passed = FBLightConvNet(stride_factor=17, **kwargs).eval()

    # stride_factor never reached a layer, so the two models agree exactly
    assert passed.attn_conv.kernel_size == default.attn_conv.kernel_size
    x = torch.randn(2, 8, 1000)
    with torch.no_grad():
        assert torch.equal(passed(x), default(x))


def test_fblightconvnet_default_build_is_not_deprecated():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FBLightConvNet(n_chans=8, n_outputs=3, n_times=1000, sfreq=250)
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]


@pytest.mark.parametrize("n_times", [200, 249])
def test_fblightconvnet_window_shorter_than_win_len(n_times):
    # used to reach xavier_uniform_ with an empty kernel and die on a
    # division by zero
    with pytest.raises(ValueError, match="shorter than win_len"):
        FBLightConvNet(n_chans=8, n_outputs=3, n_times=n_times, sfreq=250)


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
        assert not all(isinstance(g, nn.Identity) for g in model.encoder.glus)
    else:
        assert all(isinstance(g, nn.Identity) for g in model.encoder.glus)


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


# ============================================================================
# BrainModule Spatial ChannelMerger Tests
# ============================================================================


def _chs_info_with_loc(loc_array):
    """Build chs_info dicts with a 12-entry ``loc`` per channel."""
    chs_info = []
    for i, loc in enumerate(loc_array):
        chs_info.append(
            {
                "ch_name": f"ch{i}",
                "ch_type": "eeg",
                "kind": 2,  # FIFFV_EEG_CH
                "loc": np.asarray(loc, dtype=float),
            }
        )
    return chs_info


def _rng_chs_info(n_chans, seed=0):
    """chs_info with distinct random loc per channel (positions span [0, 1])."""
    return _chs_info_with_loc(np.random.default_rng(seed).random((n_chans, 12)))


def _run_forward(model, n_chans, n_outputs, n_times=512, batch=2, with_subject=False):
    """Eval, forward a random batch, assert output shape and no NaNs."""
    model.eval()
    kwargs = (
        {"subject_index": torch.zeros(batch, dtype=torch.long)} if with_subject else {}
    )
    out = model(torch.randn(batch, n_chans, n_times), **kwargs)
    assert out.shape == (batch, n_outputs)
    assert not torch.isnan(out).any()


@pytest.mark.parametrize(
    "kwargs, n_chans, n_outputs, with_subject, check",
    [
        pytest.param(
            dict(n_chans=19, chs_info=_rng_chs_info(19), use_merger=True),
            19,
            4,
            False,
            lambda m: m.merger is not None
            and m.use_merger
            and tuple(m.channel_positions.shape) == (19, 2),
            id="merger",
        ),
        pytest.param(
            dict(
                n_chans=19,
                chs_info=_rng_chs_info(19),
                use_merger=True,
                n_virtual_channels=32,
                subject_layers=True,
                subject_dim=8,
                n_subjects=5,
            ),
            19,
            4,
            True,
            lambda m: m.subject_layers_module is not None,
            id="merger_subject_layers",
        ),
        pytest.param(
            dict(n_chans=8, subject_layers=True, subject_dim=4, n_subjects=5, n_fft=64),
            8,
            2,
            True,
            None,
            id="subject_layers_stft",
        ),
        pytest.param(
            dict(n_chans=8, subject_dim=4, n_subjects=5, n_fft=64),
            8,
            2,
            True,
            None,
            id="stft_subject_embedding",
        ),
        pytest.param(
            dict(n_chans=8, dilation_growth=2.5),
            8,
            2,
            False,
            None,
            id="float_dilation_growth",
        ),
    ],
)
def test_brainmodule_forward_runs(kwargs, n_chans, n_outputs, with_subject, check):
    """Each config constructs, forwards, and yields a clean (B, n_outputs)."""
    set_random_seeds(0, False)
    m = BrainModule(n_outputs=n_outputs, n_times=512, sfreq=128, **kwargs)
    if check is not None:
        assert check(m)
    _run_forward(m, n_chans, n_outputs, with_subject=with_subject)


@pytest.mark.parametrize(
    "chs_info",
    [
        pytest.param(_chs_info_with_loc(np.zeros((8, 12))), id="all_zero_loc"),
        pytest.param(None, id="no_chs_info"),
    ],
)
def test_brainmodule_merger_autodisable(chs_info):
    """use_merger auto-disables (with warning) when chs_info lacks locations."""
    set_random_seeds(0, False)
    with pytest.warns(UserWarning):
        m = BrainModule(
            n_chans=8,
            n_outputs=2,
            n_times=512,
            sfreq=128,
            chs_info=chs_info,
            use_merger=True,
        )
    assert m.merger is None
    assert m.use_merger is False
    _run_forward(m, 8, 2)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        pytest.param(dict(n_virtual_channels=0), "n_virtual_channels", id="nvc_0"),
        pytest.param(dict(n_virtual_channels=-1), "n_virtual_channels", id="nvc_neg"),
        pytest.param(dict(merger_drop_prob=1.0), "merger_drop_prob", id="drop_prob"),
    ],
)
def test_brainmodule_merger_invalid_params(kwargs, match):
    """Invalid merger params are rejected up front with a clear error."""
    with pytest.raises(ValueError, match=match):
        BrainModule(
            n_chans=4,
            n_outputs=2,
            n_times=256,
            sfreq=128,
            chs_info=_rng_chs_info(4),
            use_merger=True,
            **kwargs,
        )


def test_brainmodule_merger_stft_warns():
    """use_merger + n_fft warns about the large input_projection (memory)."""
    with pytest.warns(UserWarning, match="STFT"):
        BrainModule(
            n_chans=8,
            n_outputs=2,
            n_times=512,
            sfreq=128,
            chs_info=_rng_chs_info(8),
            use_merger=True,
            n_virtual_channels=16,
            n_fft=64,
        )


def test_brainmodule_default_unchanged():
    """Default behavior is preserved: no merger unless opted in."""
    set_random_seeds(0, False)
    m = BrainModule(n_chans=8, n_outputs=2, n_times=512, sfreq=128)
    assert m.merger is None
    assert m.use_merger is False


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
    "n_chans,n_outputs,final_layer,expected",
    [
        (20, 4, True, (2, 4)),       # encoder-only basic
        (20, 2, True, (2, 2)),        # encoder-only binary
        (20, 10, True, (2, 10)),      # encoder-only multi-class
        (20, 4, False, (2, 2048)),    # encoder-only no final layer
    ],
)
def test_bendr_encoder_only(n_chans, n_outputs, final_layer, expected):
    """Test output shapes for encoder-only configs."""
    set_random_seeds(0, False)

    model = BENDR(
        n_chans=n_chans, n_outputs=n_outputs, n_times=5120, sfreq=256,
        encoder_only=True, final_layer=final_layer,
    )
    x = torch.randn(2, n_chans, 5120)
    y = model(x)
    assert y.shape == expected, f"Expected {expected}, got {y.shape}"


@pytest.mark.parametrize("n_times", [2560, 5120, 10240])
def test_bendr_encoder_only_variable_length(n_times):
    """Test encoder-only mode with variable input lengths."""
    set_random_seeds(0, False)

    model = BENDR(
        n_chans=20, n_outputs=4, n_times=None, sfreq=256, encoder_only=True,
    )
    x = torch.randn(2, 20, n_times)
    y = model(x)
    assert y.shape == (2, 4), f"Failed for length {n_times}: got {y.shape}"


def test_bendr_encoder_only_gradient_flow():
    """Encoder-only mode: encoder has grads, contextualizer does not."""
    set_random_seeds(0, False)

    model = BENDR(
        n_chans=20, n_outputs=4, n_times=5120, sfreq=256, encoder_only=True,
    )
    x = torch.randn(2, 20, 5120, requires_grad=True)
    model(x).sum().backward()

    encoder_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.encoder.parameters()
    )
    assert encoder_has_grad, "No gradients in encoder"

    ctx_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.contextualizer.parameters()
    )
    assert not ctx_has_grad, "Contextualizer should have no grads"


def test_bendr_encoder_only_parameter_counts():
    """Encoder/contextualizer params match full model; final_layer is larger."""
    set_random_seeds(0, False)

    model_full = BENDR(
        n_chans=20, n_outputs=4, n_times=5120, sfreq=256, encoder_only=False,
    )
    model_enc = BENDR(
        n_chans=20, n_outputs=4, n_times=5120, sfreq=256, encoder_only=True,
    )

    for name in ("encoder", "contextualizer"):
        p_full = sum(p.numel() for p in getattr(model_full, name).parameters())
        p_enc = sum(p.numel() for p in getattr(model_enc, name).parameters())
        assert p_full == p_enc, f"{name} params differ: {p_full} vs {p_enc}"

    fl_full = sum(p.numel() for p in model_full.final_layer.parameters())
    fl_enc = sum(p.numel() for p in model_enc.final_layer.parameters())
    assert fl_enc > fl_full


def test_bendr_backward_compatibility():
    """encoder_only=False (default) produces identical output to original."""
    set_random_seeds(0, False)

    model_default = BENDR(n_chans=20, n_outputs=4, n_times=5120, sfreq=256)
    model_explicit = BENDR(
        n_chans=20, n_outputs=4, n_times=5120, sfreq=256, encoder_only=False,
    )
    model_explicit.load_state_dict(model_default.state_dict())
    model_default.eval()
    model_explicit.eval()

    x = torch.randn(2, 20, 5120)
    np.testing.assert_allclose(
        model_default(x).detach().numpy(),
        model_explicit(x).detach().numpy(),
        rtol=1e-5, atol=1e-7,
    )


def test_bendr_encoder_only_short_input_raises():
    """RuntimeError when input is too short for 4-chunk pooling."""
    model = BENDR(
        n_chans=20, n_outputs=4, n_times=None, sfreq=256, encoder_only=True,
    )
    with pytest.raises(RuntimeError, match="too few"):
        model(torch.randn(2, 20, 96))


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

    # 200 samples: single_channel attends over time patches of every channel.
    model = MEDFormer(
        n_chans=22,
        n_outputs=4,
        n_times=200,
        no_inter_attn=no_inter_attn,
        single_channel=single_channel,
        output_attention=output_attention,
    )

    x = torch.randn(2, 22, 200)
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


def test_eegitnet_mapping_targets():
    # mapping values should point to real keys in the model state dict
    model = EEGITNet(
        n_outputs=4, n_chans=22, n_times=1000,
    )
    sd_keys = set(model.state_dict().keys())
    for old_key, new_key in model.mapping.items():
        assert new_key in sd_keys, f"{new_key} not in state_dict"
    # bias and weight should map separately
    targets = list(model.mapping.values())
    assert len(targets) == len(set(targets)), "mapping has duplicate targets"


def test_eegitnet_inception_kernel_scales():
    # third inception branch kernel should be 4x the base kernel length
    klen = 16
    model = EEGITNet(
        n_outputs=4, n_chans=22, n_times=1000,
        kernel_length=klen,
    )
    inc = model.inception_block
    # branches order: kernel_length, kernel_length*2, kernel_length*4
    k1 = inc.branches[0][0].kernel_size[1]
    k2 = inc.branches[1][0].kernel_size[1]
    k3 = inc.branches[2][0].kernel_size[1]
    assert k1 == klen
    assert k2 == klen * 2
    assert k3 == klen * 4


def test_eeginceptionmi_mapping_targets():
    # mapping keys should match old param names, values should exist in state dict
    model = EEGInceptionMI(
        n_outputs=4, n_chans=22, sfreq=250,
        input_window_seconds=4.5,
    )
    sd_keys = set(model.state_dict().keys())
    for old_key, new_key in model.mapping.items():
        assert new_key in sd_keys, f"{new_key} not in state_dict"
    # old keys should not have typos
    assert "fc.bias" in model.mapping


def test_syncnet_param_init_uses_correct_ranges():
    # Verify phi_ini uses phase_init_values and beta uses beta_init_values
    # (not swapped). Use a valid uniform range for beta and normal_(mean, 0.0)
    # for phi_ini to make phi_ini deterministic.
    beta_low, beta_high = 0.04, 0.06
    phase_value = 0.25
    model = SyncNet(
        n_chans=3, n_times=100, n_outputs=2,
        beta_init_values=(beta_low, beta_high),
        phase_init_values=(phase_value, 0.0),
    )
    # beta should be initialized from the requested uniform range
    assert torch.all(model.beta.data >= beta_low)
    assert torch.all(model.beta.data < beta_high)
    # phi_ini should equal the exact normal mean (with zero std)
    assert torch.all(model.phi_ini.data == phase_value)


def test_syncnet_filter_weight_shape():
    # conv2d weight must be (num_filters, n_chans, 1, filter_width).
    # Verify the permute index mapping with a deterministic sentinel W:
    # permute(3, 2, 0, 1) should map W[0, t, c, o] -> W_permuted[o, c, 0, t].
    num_filters, n_chans, filter_width = 3, 4, 20
    sentinel = torch.arange(filter_width * n_chans * num_filters).reshape(
        1, filter_width, n_chans, num_filters
    )

    W_permuted = sentinel.permute(3, 2, 0, 1).contiguous()
    assert W_permuted.shape == (num_filters, n_chans, 1, filter_width)
    for o in range(num_filters):
        for c in range(n_chans):
            for t in range(filter_width):
                assert W_permuted[o, c, 0, t] == sentinel[0, t, c, o]

    # reshape/view would produce the same shape but a different (wrong) mapping
    W_viewed = sentinel.reshape(num_filters, n_chans, 1, filter_width)
    assert not torch.equal(W_permuted, W_viewed)
    # Concrete mismatch: permute walks outer dim first, reshape walks in memory order
    assert W_permuted[0, 0, 0, 1] == sentinel[0, 1, 0, 0]
    assert W_viewed[0, 0, 0, 1] == sentinel[0, 0, 0, 1]
    assert W_permuted[0, 0, 0, 1] != W_viewed[0, 0, 0, 1]

    # Model forward still works end-to-end with the real permute-based kernel
    model = SyncNet(
        n_chans=n_chans, n_times=200, n_outputs=2,
        num_filters=num_filters, filter_width=filter_width,
    )
    out = model(torch.randn(2, n_chans, 200))
    assert out.shape == (2, 2)


# ---------------------------------------------------------------------------
# MetaNeuromotorHand
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gni_default():
    """Build the default 15-layer handwriting conformer once per test module."""
    return MetaNeuromotorHand(n_times=32000).eval()


def _small_gni(**kwargs):
    defaults = dict(
        n_times=800,
        conformer_num_layers=1,
        conformer_attn_window_size=0,
        conformer_kernel_size=3,
        conformer_stride=1,
        time_reduction_stride=1,
        drop_prob=0.0,
    )
    defaults.update(kwargs)
    return MetaNeuromotorHand(**defaults)


def test_gni_default_contract(gni_default):
    x = torch.randn(1, 16, 32000)
    with torch.no_grad():
        y = gni_default(x)

    n_params = sum(p.numel() for p in gni_default.parameters() if p.requires_grad)
    lengths = gni_default.compute_output_lengths(torch.tensor([32000, 40000]))

    assert n_params == 1_021_284
    assert y.shape == (1, 38, 100)
    assert lengths[0].item() == y.shape[1]
    assert lengths[1] > lengths[0]


def test_gni_head_log_softmax_and_config():
    m = _small_gni(log_softmax=True).eval()
    with torch.no_grad():
        y = m(torch.randn(1, 16, 800))

    assert torch.allclose(y.exp().sum(dim=-1), torch.ones_like(y[..., 0]), atol=1e-5)
    assert MetaNeuromotorHand.from_config(m.get_config()).n_outputs == 100
    m.reset_head(30)
    assert m.n_outputs == 30
    assert m.final_layer.out_features == 30


def test_gni_ctc_backward():
    torch.manual_seed(0)
    m = _small_gni().train()
    x = torch.randn(2, 16, 800)
    emissions = m(x)  # (N, T, V)
    log_probs = torch.log_softmax(emissions, dim=-1).transpose(0, 1)  # (T, N, V)
    input_lengths = m.compute_output_lengths(torch.tensor([800, 800]))
    target_lengths = torch.tensor([2, 2])
    targets = torch.randint(1, 100, (2, 2))
    loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)(
        log_probs, targets, input_lengths, target_lengths
    )
    loss.backward()

    assert input_lengths.min() >= target_lengths.max()
    assert m.final_layer.weight.grad is not None
    assert any(p.grad is not None for p in m.conformer.parameters())


# ---------------------------------------------------------------------------
# EMG2QwertyNet
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def emg2qwerty_default():
    """Build the default 4-block TDS-Conv-CTC encoder once per test module."""
    return EMG2QwertyNet(n_times=8000).eval()


def _small_emg2qwerty(**kwargs):
    """Tiny config for fast unit tests.

    Receptive field is ``n_fft + n_conv_blocks * (kernel_width - 1) *
    hop_length = 64 + 2 * 7 * 16 = 288`` samples; ``n_times=500`` keeps
    the forward small while leaving headroom for the encoder.
    """
    defaults = dict(
        n_times=500,
        mlp_features=(48,),
        block_channels=(12, 12),
        kernel_width=8,
    )
    defaults.update(kwargs)
    return EMG2QwertyNet(**defaults)


def test_emg2qwerty_default_contract(emg2qwerty_default):
    x = torch.randn(1, 32, 8000)
    with torch.no_grad():
        y = emg2qwerty_default(x)

    n_params = sum(
        p.numel() for p in emg2qwerty_default.parameters() if p.requires_grad
    )
    lengths = emg2qwerty_default.compute_output_lengths(
        torch.tensor([8000, 12000])
    )

    assert n_params == 5_293_315
    assert y.shape == (1, 373, 99)
    assert lengths[0].item() == y.shape[1]
    assert lengths[1] > lengths[0]


def test_emg2qwerty_train_eval_and_state_dict():
    """Smoke-test small config: log_softmax head, CTC backward, key layout.

    Covers the full small-model contract in one pass: log-softmax
    normalization, ``from_config`` round-trip, ``reset_head``, a CTC
    backward step, and the upstream-compatible ``state_dict`` layout
    (encoder under ``model.{0,1,3}.*``, head at ``final_layer.*``,
    non-persistent STFT window absent).
    """
    torch.manual_seed(0)
    m = _small_emg2qwerty(log_softmax=True).train()

    x = torch.randn(2, 32, 500)
    log_probs = m(x).transpose(0, 1)  # already log-softmaxed
    assert torch.allclose(
        log_probs.exp().sum(dim=-1),
        torch.ones_like(log_probs[..., 0]),
        atol=1e-5,
    )

    input_lengths = m.compute_output_lengths(torch.tensor([500, 500]))
    target_lengths = torch.tensor([2, 2])
    targets = torch.randint(0, 98, (2, 2))
    loss = torch.nn.CTCLoss(blank=98, zero_infinity=True)(
        log_probs, targets, input_lengths, target_lengths
    )
    loss.backward()
    assert input_lengths.min() >= target_lengths.max()
    assert m.final_layer.weight.grad is not None
    assert any(p.grad is not None for p in m.model.parameters())

    assert EMG2QwertyNet.from_config(m.get_config()).n_outputs == 99

    # ``reset_head`` must propagate to ``get_config`` so save/restore
    # round-trips rebuild the head with the new vocab size, and must
    # inherit dtype/device so post-``.double()``/``.to(...)`` calls keep
    # the model usable.
    m_dbl = _small_emg2qwerty().double().train()
    m_dbl.reset_head(30)
    assert m_dbl.n_outputs == 30 and m_dbl.final_layer.out_features == 30
    assert m_dbl.final_layer.weight.dtype == torch.float64
    assert m_dbl.get_config()["n_outputs"] == 30
    assert EMG2QwertyNet.from_config(m_dbl.get_config()).n_outputs == 30
    # Forward must still work after dtype change + reset_head.
    with torch.no_grad():
        m_dbl(torch.randn(1, 32, 500, dtype=torch.float64))

    keys = list(m.state_dict().keys())
    allowed = ("model.0.", "model.1.", "model.3.", "final_layer.")
    assert not [k for k in keys if not k.startswith(allowed)]
    assert not any(k.startswith("spectrogram.") for k in keys)
    # Mapping values must point at the actual top-level head keys, not
    # just any string — guards against typos in upstream-checkpoint
    # head remap.
    assert EMG2QwertyNet.mapping == {
        "model.4.weight": "final_layer.weight",
        "model.4.bias": "final_layer.bias",
    }
    for new_key in EMG2QwertyNet.mapping.values():
        assert new_key in keys, f"mapping target {new_key!r} missing from state_dict"


def test_emg2qwerty_flexible_band_geometry():
    """num_bands and electrodes_per_band can deviate from the wristband default.

    Smoke-checks that a non-default ``(num_bands=3, electrodes_per_band=8)``
    config (24 channels) builds and forwards. Validates that ``n_chans``
    inconsistent with the geometry raises.
    """
    m_no_rot = _small_emg2qwerty(
        n_chans=24,
        num_bands=3,
        electrodes_per_band=8,
        rotation_offsets=(0,),
        pooling="max",
        log_eps=1e-5,
    ).eval()
    with torch.no_grad():
        y_no_rot = m_no_rot(torch.randn(1, 24, 500))
    assert y_no_rot.shape[0] == 1 and y_no_rot.shape[2] == 99

    # ``rotation_offsets=(0,)`` must produce identical output for
    # rolled-and-unrolled inputs along the electrode axis. Confirms the
    # caller-provided offsets are actually honored (and not silently
    # replaced by the default ``(-1, 0, +1)``).
    m_default_rot = _small_emg2qwerty(
        n_chans=24,
        num_bands=3,
        electrodes_per_band=8,
        rotation_offsets=(-2, 0, +2),
        pooling="mean",
    ).eval()
    assert m_default_rot.model[1].mlps[0].offsets == (-2, 0, 2)
    assert m_no_rot.model[1].mlps[0].offsets == (0,)
    assert m_no_rot.model[1].mlps[0].pooling == "max"

    with pytest.raises(ValueError, match="n_chans == num_bands"):
        EMG2QwertyNet(n_chans=33, num_bands=2, electrodes_per_band=16, n_times=500)
    with pytest.raises(ValueError, match="log_eps"):
        _small_emg2qwerty(log_eps=0.0)
    with pytest.raises(ValueError, match="pooling"):
        _small_emg2qwerty(pooling="sum")


def test_emg2qwerty_input_validation():
    """forward rejects short inputs; compute_output_lengths floors to zero.

    Two related contracts. (1) ``forward`` must enforce the full encoder
    receptive field, not just ``n_fft`` — otherwise ``torch.stft``
    succeeds for ``n_times`` in ``[n_fft, receptive_field)`` and the
    model crashes deep inside :class:`~torch.nn.Conv2d`. (2)
    ``compute_output_lengths`` must use ``rounding_mode="floor"`` so
    that ``T < n_fft`` reports zero emission frames; ``trunc`` would
    silently return ``1`` when ``kernel_width == 1`` (no encoder shrink
    to mask the off-by-one via ``clamp_min``).
    """
    m = _small_emg2qwerty(n_fft=64, hop_length=16, kernel_width=8,
                          block_channels=(12, 12))
    # Receptive field = 64 + 2 * 7 * 16 = 288.
    with pytest.raises(ValueError, match="n_fft"):
        m(torch.randn(1, 32, 50))
    with pytest.raises(ValueError, match="receptive field"):
        m(torch.randn(1, 32, 200))
    # Boundary: exactly the receptive-field length must be accepted and
    # produce a single output frame from each conv block.
    with torch.no_grad():
        y_min = m(torch.randn(1, 32, 288))
    assert y_min.shape[0] == 1 and y_min.shape[2] == 99

    m_floor = _small_emg2qwerty(kernel_width=1, block_channels=(12, 12))
    # T < n_fft -> 0; T == n_fft -> 1 (boundary); T > n_fft increments
    # by ``floor((T - n_fft) / hop_length) + 1``.
    out = m_floor.compute_output_lengths(torch.tensor([10, 0, 50, 64, 80]))
    assert out.tolist() == [0, 0, 0, 1, 2]


def _force_two_randint():
    """Monkeypatch helper: force every size-`()` ``torch.randint`` to draw 2.

    ``_SpecAugment.forward`` samples its mask counts via two
    ``torch.randint(n+1, ())`` calls (time then frequency); freezing
    them at 2 makes the augmentation deterministic without depending
    on the global RNG or torchaudio's internal RNG sequence (which
    can shift across versions and silently turn the test into a flake).

    Returns a ``mock.patch`` context manager rebinding ``torch.randint``;
    captures the real ``torch.randint`` in a closure so the patched
    function can still delegate non-`()` calls.
    """
    real = torch.randint

    def patched(*args, **kwargs):
        size = kwargs.get("size", args[1] if len(args) >= 2 else None)
        if size == ():
            return torch.tensor(2, dtype=torch.long)
        return real(*args, **kwargs)

    return mock.patch("torch.randint", side_effect=patched)


def test_emg2qwerty_spec_augment_contract():
    """Built-in SpecAugment: train-only, per-electrode iid, parameter-free, round-trips."""
    # ``spec_augment=False`` is the back-compat default and must wire an
    # ``nn.Identity`` so existing checkpoints keep loading bit-for-bit.
    assert isinstance(_small_emg2qwerty().spec_augment, torch.nn.Identity)

    # Bad knobs rejected at construction time.
    for bad in (
        {"n_time_masks": -1},
        {"time_mask_param": -1},
        {"spec_augment_prob": 1.5},
    ):
        with pytest.raises(ValueError):
            _small_emg2qwerty(spec_augment=True, **bad)

    torch.manual_seed(0)
    m = _small_emg2qwerty(
        spec_augment=True, n_time_masks=3, time_mask_param=8,
        n_freq_masks=2, freq_mask_param=4, spec_augment_prob=1.0,
    )
    x = torch.randn(2, 32, 500)

    # Eval is deterministic (no SpecAugment sampling).
    m.eval()
    with torch.no_grad():
        assert torch.equal(m(x), m(x))

    # Train mutates the spectrogram and is stochastic between calls.
    m.train()
    spec = m.spectrogram(x)
    with _force_two_randint():
        aug_a, aug_b = m.spec_augment(spec), m.spec_augment(spec)
    assert not torch.equal(aug_a, spec) and not torch.equal(aug_a, aug_b)

    # Per-(B, band, electrode) iid masking: on a uniquely-valued tensor
    # at least one (B, band) row must show distinct mask patterns across
    # its electrodes — pinning the upstream ``emg2qwerty.transforms``
    # recipe and guarding against any future band-shared regression.
    distinct = torch.arange(spec.numel(), dtype=spec.dtype).reshape(spec.shape)
    with _force_two_randint():
        aug = m.spec_augment(distinct)
    # ``changed`` shape: (B, num_bands, electrodes, freq*T_spec) after
    # moving time-axis to the trailing flatten. Compare each electrode's
    # bool grid against electrode-0 within the same (B, band) row.
    changed = (aug != distinct).movedim(0, -1).flatten(start_dim=3)
    same_as_e0 = (changed == changed[:, :, :1]).all(dim=-1)
    assert not same_as_e0.all(), "SpecAugment masking must not be band-shared"

    # Parameter-free (no new state-dict keys) and round-trips via
    # ``from_config`` so reloaded models reproduce the augmentation recipe.
    allowed = ("model.0.", "model.1.", "model.3.", "final_layer.")
    assert all(k.startswith(allowed) for k in m.state_dict())
    cfg = m.get_config()
    assert cfg["spec_augment"] and cfg["time_mask_param"] == 8
    assert isinstance(
        EMG2QwertyNet.from_config(cfg).spec_augment, type(m.spec_augment)
    )


def test_emg2qwerty_feature_flags():
    """Three forward paths: default emissions tensor, runtime dict, init tuple."""
    mlp_features = (48,)
    num_features = 2 * mlp_features[-1]  # num_bands × mlp_features[-1]
    x = torch.randn(2, 32, 500)

    # Default: emissions tensor, ``(B, T_out, n_outputs)``.
    m = _small_emg2qwerty(mlp_features=mlp_features).eval()
    with torch.no_grad():
        emissions = m(x)
        bundle = m(x, return_features=True)
    assert isinstance(emissions, torch.Tensor) and emissions.shape[2] == 99

    # Runtime ``return_features=True`` → dict (BIOT / signal-JEPA convention).
    # ``features`` must be the pre-classifier representation, so applying
    # ``final_layer`` to it reproduces the emissions tensor.
    assert set(bundle) == {"features", "cls_token"} and bundle["cls_token"] is None
    feats = bundle["features"]
    assert feats.shape == (2, emissions.shape[1], num_features)
    assert torch.allclose(m.final_layer(feats), emissions, atol=1e-5)

    # Init ``return_feature=True`` → ``(emissions, features)`` tuple — the
    # config-driven path neuroai's ``DownstreamWrapperModel`` uses with
    # ``model_output_key=1``. Runtime dict flag still wins, ``get_config``
    # round-trips, and ``get_output_shape`` keeps reporting the emissions
    # shape regardless of the flag.
    m_t = _small_emg2qwerty(mlp_features=mlp_features, return_feature=True).eval()
    with torch.no_grad():
        out_t = m_t(x)
        bundle_t = m_t(x, return_features=True)
    assert isinstance(out_t, tuple) and out_t[1].shape == feats.shape
    assert isinstance(bundle_t, dict) and torch.equal(bundle_t["features"], out_t[1])
    assert m_t.get_config()["return_feature"] is True
    assert m_t.get_output_shape() == (1, emissions.shape[1], 99)


@pytest.mark.parametrize("model_cls", [BDTCN, BENDR])
def test_channel_dropout_on_1d_activations(model_cls):
    """BDTCN and BENDR drop whole channels of a ``(batch, channels, times)``
    tensor, so the dropout modules must be ``nn.Dropout1d``. ``nn.Dropout2d``
    routes 3D input to the channel-wise path only through a deprecated
    fallback that warns on every forward pass."""
    model = model_cls(
        n_chans=8, n_outputs=2, n_times=256, sfreq=100.0, drop_prob=0.5
    ).train()

    assert any(isinstance(m, nn.Dropout1d) for m in model.modules())
    assert not any(isinstance(m, nn.Dropout2d) for m in model.modules())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model(torch.randn(4, 8, 256))
    assert not [w for w in caught if "dropout2d" in str(w.message)]


def test_tcn_bai_variant_channel_dropout():
    """The Bai et al. ``TCN`` shares the residual block with ``BDTCN``, so it
    gets the same channel-wise dropout module."""
    model = TCN(n_chans=8, n_outputs=2, n_blocks=2, n_filters=5, drop_prob=0.5)
    assert any(isinstance(m, nn.Dropout1d) for m in model.modules())
    assert not any(isinstance(m, nn.Dropout2d) for m in model.modules())


def test_dropout1d_masks_channels_not_batch_items():
    """The channel-wise dropout used by BDTCN and BENDR zeroes rows of the
    channel axis, independently per batch item. Under the announced
    ``nn.Dropout2d`` semantics for 3D input the same tensor would be read as
    unbatched and whole batch items would be dropped instead."""
    set_random_seeds(2024, cuda=False)
    out = nn.Dropout1d(0.5).train()(torch.ones(64, 16, 32))

    # A dropped entry covers the full time axis of one (batch item, channel)
    # pair.
    zeroed = out.abs().sum(dim=-1) == 0
    assert zeroed.any() and not zeroed.all()
    # At least one batch item keeps some channels and loses others, which
    # cannot happen when the mask is drawn over the batch axis.
    assert (zeroed.any(dim=1) & ~zeroed.all(dim=1)).any()


_PARAM_LINE = re.compile(
    r"^(?P<names>\*{0,2}\w+(?:\s*,\s*\*{0,2}\w+)*)\s*:"
)


def _documented_parameters(model_class):
    """Names listed in the ``Parameters`` section of a model docstring.

    ``EEGModuleMixin.__init_subclass__`` appends the Hugging Face Hub notes to
    every model docstring, so drop them before parsing. After
    :func:`inspect.getdoc` a parameter entry sits at column zero and its
    description is indented, which is what tells them apart from prose lines
    such as ``Note: ...`` inside a description.
    """
    doc = inspect.getdoc(model_class) or ""
    doc = doc.split(".. rubric:: Hugging Face Hub integration")[0]
    lines = doc.split("\n")

    names = []
    inside = False
    for position, line in enumerate(lines):
        stripped = line.strip()
        next_line = lines[position + 1].strip() if position + 1 < len(lines) else ""
        if next_line and set(next_line) == {"-"}:
            inside = stripped == "Parameters"
            continue
        if not inside:
            continue
        match = _PARAM_LINE.match(line)
        if match is None:
            continue
        following = next((nxt for nxt in lines[position + 1 :] if nxt.strip()), "")
        if following.startswith(" "):
            names.extend(
                name.strip().lstrip("*") for name in match.group("names").split(",")
            )
    return names


@pytest.mark.parametrize("model_name", sorted(all_models_dict))
def test_documented_parameters_exist_in_signature(model_name):
    """Every documented parameter must be accepted by the constructor.

    Renamed or removed parameters used to survive in the docstrings, so users
    following the documentation got a ``TypeError`` instead of a model.
    """
    model_class = all_models_dict[model_name]
    parameters = inspect.signature(model_class.__init__).parameters
    if any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
        pytest.skip(f"{model_name} forwards **kwargs, any name is accepted")

    accepted = set(parameters) - {"self"}
    documented = set(_documented_parameters(model_class))
    unknown = sorted(documented - accepted)

    assert not unknown, (
        f"{model_name} documents parameters its constructor does not accept: "
        f"{unknown}"
    )


@pytest.mark.parametrize(
    "model_name",
    [
        "ATCNet",
        "AttnSleep",
        "CTNet",
        "EEGSimpleConv",
        "IFNet",
        "SPARCNet",
        "SleepStagerBlanco2020",
        "SleepStagerChambon2018",
        "TIDNet",
    ],
)
def test_audited_model_parameter_headers_follow_numpydoc(model_name):
    """Changed parameter headers must be parsed as names, not name-and-type."""
    model_class = all_models_dict[model_name]
    doc = inspect.getdoc(model_class) or ""
    doc = doc.split(".. rubric:: Hugging Face Hub integration")[0]
    lines = doc.split("\n")

    malformed = []
    inside = False
    for position, line in enumerate(lines):
        stripped = line.strip()
        next_line = lines[position + 1].strip() if position + 1 < len(lines) else ""
        if next_line and set(next_line) == {"-"}:
            inside = stripped == "Parameters"
            continue
        if not inside:
            continue
        match = _PARAM_LINE.match(line)
        if match is None:
            continue
        following = next((nxt for nxt in lines[position + 1 :] if nxt.strip()), "")
        if following.startswith(" ") and not line.startswith(
            f"{match.group('names')} : "
        ):
            malformed.append(line)

    assert not malformed, (
        f"{model_class.__name__} has parameter headers numpydoc misparses: "
        f"{malformed}"
    )


# ---------------------------------------------------------------------------
# Brant
# ---------------------------------------------------------------------------


def test_brant_band_power_rate_is_independent_of_sfreq():
    from braindecode.models.brant import BRANT_FREQ_BANDS, _BandPowerFeatures

    x = torch.randn(2, 3, 1, 1500)
    at_256 = _BandPowerFeatures(256.0, BRANT_FREQ_BANDS, 1500)(x)
    model_250 = Brant(n_chans=3, n_outputs=2, n_times=1500, patch_size=1500, sfreq=250)
    # sfreq describes the data; the band edges follow band_power_sfreq (upstream fs=256).
    assert torch.equal(model_250.band_power(x), at_256)
    assert model_250.band_power.sfreq == 256.0


def test_brant_head_is_a_bare_linear_layer():
    model = Brant(n_chans=2, n_outputs=3, n_times=1500, patch_size=1500)
    assert isinstance(model.final_layer, torch.nn.Linear)
    model.reset_head(5)
    assert model.n_outputs == 5 and model.final_layer.out_features == 5
    assert model.get_config()["n_outputs"] == 5


def test_brant_rejects_channel_count_mismatch():
    model = Brant(n_chans=2, n_outputs=3, n_times=1500, patch_size=1500).eval()
    with pytest.raises(ValueError, match="channels"):
        model(torch.randn(1, 5, 1500))


def test_brant_rejects_invalid_construction():
    with pytest.raises(ValueError, match="patch_size"):
        Brant(n_chans=2, n_outputs=2, n_times=1499, patch_size=1500)
    with pytest.raises(ValueError, match="n_freq_bands"):
        Brant(n_chans=2, n_outputs=2, n_times=1500, patch_size=1500, n_freq_bands=5)


def test_brant_scripts_and_matches_eager():
    model = Brant(
        n_chans=2,
        n_outputs=3,
        n_times=3000,
        patch_size=1500,
        embed_dim=32,
        ffn_dim=64,
        temporal_n_layers=1,
        spatial_n_layers=1,
        n_heads=2,
    ).eval()
    x = torch.randn(2, 2, 3000)
    scripted = torch.jit.script(model)
    torch.testing.assert_close(scripted(x), model(x))
    # Under scripting, return_features=True yields the logits (is_scripting guard).
    torch.testing.assert_close(scripted(x, return_features=True), model(x))


def test_brant_band_power_matches_scipy_periodogram():
    """Upstream computes log10(sum of periodogram density in each band + 1) at fs=256."""
    from scipy.signal import periodogram

    from braindecode.models.brant import BRANT_FREQ_BANDS, _BandPowerFeatures

    patches = torch.randn(2, 3, 2, 1500)
    ours = _BandPowerFeatures(256.0, BRANT_FREQ_BANDS, 1500)(patches).numpy()
    freqs, psd = periodogram(patches.numpy(), fs=256.0, axis=-1)
    ref = np.stack(
        [
            np.log10(psd[..., (freqs > lo) & (freqs <= hi)].sum(-1) + 1)
            for lo, hi in BRANT_FREQ_BANDS
        ],
        -1,
    )
    assert np.abs(ours - ref).max() < 1e-4


def test_brant_input_length_must_match():
    model = Brant(n_chans=2, n_outputs=2, n_times=3000, patch_size=1500).eval()
    with pytest.raises(ValueError, match="time samples"):
        model(torch.randn(1, 2, 1500))


def test_brant_channel_tokens_keep_channel_order():
    """Pin the per-channel token layout downstream code relies on.

    ``merge_time`` outputs ``(batch, n_chans, seq_len, embed_dim)`` with the
    channel axis in the same order as the input; the spatial encoder has no
    channel-specific parameters, so permuting input channels permutes the
    output channel tokens the same way.
    """
    model = Brant(
        n_chans=3,
        n_outputs=2,
        n_times=3000,
        patch_size=1500,
        embed_dim=32,
        ffn_dim=64,
        temporal_n_layers=1,
        spatial_n_layers=1,
        n_heads=2,
    ).eval()

    captured = {}

    def _hook(module, inputs, output):
        captured["out"] = output

    handle = model.merge_time.register_forward_hook(_hook)
    try:
        x = torch.randn(2, 3, 3000)
        model(x)
        first = captured["out"].clone()
        assert first.shape == (2, 3, 2, 32)  # (batch, n_chans, seq_len, embed_dim)

        perm = [2, 0, 1]
        model(x[:, perm])
        permuted = captured["out"]
        torch.testing.assert_close(permuted, first[:, perm])
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# BrainBERT
# ---------------------------------------------------------------------------


def _brainbert_upstream_spectrogram(wav, clip, zscore_before_clip):
    """Upstream scipy recipe: magnitude of the first 40 bins, per-bin z-score
    (ddof=0, zero std replaced by one), a fully flat window replaced by ones,
    and boundary trimming in either order."""

    def zscore(a):
        std = a.std(axis=-1, ddof=0, keepdims=True)
        std[std == 0] = 1.0
        return (a - a.mean(axis=-1, keepdims=True)) / std

    _, _, zxx = stft(wav, 2048, nperseg=400, noverlap=350)
    mag = np.abs(zxx[:40])
    if zscore_before_clip:  # preprocessors/stft.py, behind the released checkpoint
        mag = zscore(mag)
        if (mag.std() == 0).any():
            mag = np.ones_like(mag)
        mag = mag[:, clip:-clip]
    else:  # notebooks/demo.ipynb
        mag = zscore(mag[:, clip:-clip])
    return np.nan_to_num(mag).T  # (n_frames, 40)


@pytest.mark.parametrize("clip, zscore_before_clip", [(10, True), (5, False)])
@pytest.mark.parametrize("n_times", [2048, 6000])  # 2048 needs scipy's right padding
def test_brainbert_stft_front_end_matches_upstream_scipy(
    clip, zscore_before_clip, n_times
):
    module = _STFTSpectrogram(clip=clip, zscore_before_clip=zscore_before_clip)
    for wav in (np.random.RandomState(0).randn(n_times), np.zeros(n_times)):
        ref = _brainbert_upstream_spectrogram(wav, clip, zscore_before_clip)
        ours = module(torch.from_numpy(wav).view(1, 1, -1))[0, 0].numpy()
        assert ours.shape == ref.shape == (module.n_frames(n_times), 40)
        # The residual is the float32 Hann window (scipy builds it in float64).
        assert np.abs(ref - ours).max() < 5e-6


def test_brainbert_mapping_targets():
    """The authors' ``input_encoding.*`` keys map onto real port parameters."""
    model = BrainBERT(n_chans=1, n_outputs=2, n_times=2048)
    state = model.state_dict()
    assert model.mapping and all(target in state for target in model.mapping.values())


_BARISTA_SMALL = dict(patch_size=32, d_model=16, n_layers=1, num_heads=2, cnn_depth=1)


def _barista_model(spatial_scale, n_chans=4, pooling="mean", **kwargs):
    return BaRISTA(
        n_outputs=2,
        n_chans=n_chans,
        n_times=128,
        spatial_scale=spatial_scale,
        pooling=pooling,
        **_BARISTA_SMALL,
        **kwargs,
    ).eval()


@pytest.mark.parametrize(
    "spatial_scale, indices",
    [("parcels", [1, 5, 120, 0]), ("lobes", [0, 3, 20, 7]), ("none", None)],
)
def test_barista_region_scales(spatial_scale, indices):
    model = _barista_model(spatial_scale, spatial_indices=indices)
    assert model(torch.randn(3, 4, 128)).shape == (3, 2)


def test_barista_forward_indices_other_montage():
    """Mean pooling lets one model read recordings with other montages."""
    model = _barista_model("parcels", spatial_indices=[1, 2, 3, 4])
    out = model(torch.randn(2, 6, 128), spatial_indices=torch.tensor([1, 2, 3, 4, 5, 6]))
    assert out.shape == (2, 2)


@pytest.mark.parametrize("bad", [[1, 2, 3, 121], [1, 2, 3, -1]])
def test_barista_forward_rejects_out_of_range_indices(bad):
    model = _barista_model("parcels", spatial_indices=[1, 2, 3, 4])
    with pytest.raises(ValueError, match=r"\[0, 121\)"):
        model(torch.randn(1, 4, 128), spatial_indices=torch.tensor(bad))


def test_barista_learned_pooling_needs_construction_grid():
    model = _barista_model("parcels", pooling="learned", spatial_indices=[1, 2, 3, 4])
    with pytest.raises(ValueError, match="pooling='mean'"):
        model(torch.randn(1, 6, 128), spatial_indices=torch.tensor([1, 2, 3, 4, 5, 6]))


def test_barista_coords_fallback_uses_left_inferior_posterior_order():
    info = mne.create_info(["a"], 256.0, "seeg")
    info["chs"][0]["loc"][:3] = [0.010, 0.020, 0.030]  # RAS metres
    model = BaRISTA(
        n_outputs=2, chs_info=info["chs"], n_times=128, pooling="mean", **_BARISTA_SMALL
    )
    left, inferior, posterior = model.spatial_emb.default_indices[0].tolist()
    centre = model.spatial_emb.tables[0].num_embeddings // 2
    assert (left, inferior, posterior) == (centre - 10, centre - 30, centre - 20)


def test_mscformer_default_attention_scale_matches_original_source():
    """MSCFormer's default attention scale must reproduce the released
    source's ``embed_dim ** -0.5`` logit scaling.

    The original MSCFormer code divides attention logits by
    ``sqrt(emb_size)`` for every head, not by ``sqrt(head_dim)`` (braindecode's
    more common default for :class:`~braindecode.modules.MultiHeadAttention`).
    The two only coincide when ``num_heads == 1``; with the paper's defaults
    (``num_heads=8``, ``emb_size=48``) they differ by ``sqrt(num_heads)``. A
    state-dict-matched parity check against the original implementation gives
    a max abs logit diff of about 2e-7 with this default, versus about 0.035
    with ``head_dim ** -0.5``.
    """
    from braindecode.models.mscformer import MSCFormer

    model = MSCFormer(n_outputs=4, n_chans=22, n_times=1000)
    expected_scale = model.embed_dim**-0.5
    assert model.embed_dim == 48
    for block in model.trans.layers:
        assert block.attention.module.scale == pytest.approx(expected_scale)

    custom_scale = 0.25
    model_custom = MSCFormer(
        n_outputs=4, n_chans=22, n_times=1000, attention_scale=custom_scale
    )
    for block in model_custom.trans.layers:
        assert block.attention.module.scale == pytest.approx(custom_scale)


@pytest.mark.parametrize("bad_scale", [0.0, -1.0])
def test_mscformer_rejects_non_positive_attention_scale(bad_scale):
    from braindecode.models.mscformer import MSCFormer

    with pytest.raises(ValueError, match="attention_scale"):
        MSCFormer(n_outputs=4, n_chans=22, n_times=1000, attention_scale=bad_scale)


# ---------------------------------------------------------------------------
# NeuroRVQ
# ---------------------------------------------------------------------------


@pytest.fixture
def neurorvq_model_kwargs():
    return {
        "n_chans": 3,
        "n_outputs": 4,
        "n_times": 600,
        "sfreq": 200,
        "channel_names": ("f3", "f4", "cz"),
        "depth": 2,
        "num_heads": 4,
        "out_chans": 4,
        "max_patches": 8,
    }


def test_neurorvq_output_and_features(neurorvq_model_kwargs):
    model = NeuroRVQ(**neurorvq_model_kwargs)
    x = torch.randn(2, model.n_chans, model.n_times)

    logits = model(x)
    features = model(x, return_features=True)

    assert logits.shape == (2, 4)
    assert features["features"].shape == (2, 100 * 4 * 3 * 3)
    assert features["cls_token"] is None


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_times": 601}, "n_times must be divisible by patch_size"),
        ({"sfreq": 250}, "trained at 200 Hz"),
        (
            {"channel_names": ("f3", "f4", "not-an-eeg-channel")},
            "Unsupported NeuroRVQ channel",
        ),
        ({"channel_names": ("f3", "f3", "cz")}, "channel_names must be unique"),
        ({"n_times": 1800, "max_patches": 8}, "supports at most 8 patches"),
        ({"modality": "eog"}, "modality must be one of"),
        ({"init_values": None}, "init_values must be a number"),
    ],
)
def test_neurorvq_invalid_configuration(neurorvq_model_kwargs, kwargs, message):
    with pytest.raises(ValueError, match=message):
        NeuroRVQ(**(neurorvq_model_kwargs | kwargs))


def test_neurorvq_reset_head(neurorvq_model_kwargs):
    model = NeuroRVQ(**neurorvq_model_kwargs)
    model.reset_head(2)

    assert model(torch.randn(1, 3, 600)).shape == (1, 2)


def test_neurorvq_channel_slots_from_chs_info(neurorvq_model_kwargs):
    kwargs = neurorvq_model_kwargs | {
        "channel_names": None,
        "chs_info": [
            {"ch_name": "F3"},
            {"ch_name": "F4"},
            {"ch_name": "Cz"},
        ],
    }
    model = NeuroRVQ(**kwargs)

    assert model.channel_names == ("f3", "f4", "cz")
    assert model.spatial_embedding_ix.tolist() == [41, 42, 36]


def test_neurorvq_spatial_slots_preserve_upstream_zero_based_mapping(
    neurorvq_model_kwargs,
):
    # Upstream create_embedding_ix uses zero-based electrode positions and then
    # pads CLS with another 0. Keep that unusual checkpoint contract unchanged.
    kwargs = neurorvq_model_kwargs | {"channel_names": ("a1", "a2", "f3")}
    model = NeuroRVQ(**kwargs)

    assert model.spatial_embedding_ix.tolist() == [0, 1, 41]


def test_neurorvq_default_channel_names_follow_reference_order(neurorvq_model_kwargs):
    kwargs = neurorvq_model_kwargs | {"channel_names": None}
    model = NeuroRVQ(**kwargs)

    from braindecode.models.neurorvq import NEURORVQ_CHANNELS

    assert model.channel_names == NEURORVQ_CHANNELS[:3]


@pytest.mark.parametrize(
    "modality, sfreq, n_times, names, kernels, n_slots, num_quantizers",
    [
        ("eeg", 200, 400, ("f3", "cz"), (21, 9), 105, 8),
        ("ecg", 200, 80, ("i", "v1"), (21, 9), 16, 8),
        ("emg", 1000, 400, ("c1", "c9"), (51, 25), 17, 16),
        ("ppg", 100, 160, ("ppg_c1",), (41, 17), 2, 8),
    ],
)
def test_neurorvq_modality_presets(
    modality, sfreq, n_times, names, kernels, n_slots, num_quantizers
):
    geometry = dict(
        n_chans=len(names),
        n_times=n_times,
        sfreq=sfreq,
        channel_names=names,
        modality=modality,
    )
    model = NeuroRVQ(n_outputs=2, depth=1, **geometry)
    tokenizer = NeuroRVQTokenizer(
        encoder_depth=1, decoder_depth=1, n_code=16, **geometry
    ).eval()
    x = torch.randn(2, len(names), n_times)

    for m in (model, tokenizer.encoder):
        conv = m.patch_embed
        assert (conv.conv1_1.kernel_size[1], conv.conv2_1.kernel_size[1]) == kernels
        assert m.pos_embed.shape[0] == n_slots
    assert len(tokenizer.quantize_1.layers) == num_quantizers
    width = 4 * model.embed_dim
    if modality != "eeg":  # mean-pooled head
        assert model(x, return_features=True)["features"].shape == (2, width)
    assert model(x).shape == (2, 2)
    target, reconstruction = tokenizer(x)
    assert target.shape == reconstruction.shape == (2, n_times // model.patch_size * len(names), model.patch_size)
    assert tokenizer.tokenize(x).shape[:2] == (4, num_quantizers)


def test_neurorvq_transformer_block_uses_sequential_residuals():
    from braindecode.models.neurorvq import _Block

    block = _Block(
        dim=16,
        num_heads=4,
        mlp_ratio=2,
        qkv_bias=True,
        qk_norm=torch.nn.LayerNorm,
        drop=0,
        attn_drop=0,
        drop_path=0,
        init_values=1e-5,
    ).eval()
    x = torch.randn(2, 5, 16)

    expected = x + block.gamma_1 * block.attn(block.norm1(x))
    expected = expected + block.gamma_2 * block.mlp(block.norm2(expected))

    torch.testing.assert_close(block(x), expected)

# ---------------------------------------------------------------------------
# NeuroRVQTokenizer
# ---------------------------------------------------------------------------


def _small_neurorvq_tokenizer(**kwargs):
    params = dict(
        n_chans=3,
        n_times=400,
        sfreq=200,
        channel_names=("f3", "f4", "cz"),
        max_patches=4,
        out_chans=4,
        num_heads=4,
        encoder_depth=1,
        decoder_depth=1,
        n_code=16,
        code_dim=16,
        num_quantizers=2,
    )
    return NeuroRVQTokenizer(**{**params, **kwargs})


def test_neurorvq_tokenizer_codes_and_cold_codebooks():
    torch.manual_seed(0)
    model = _small_neurorvq_tokenizer().eval()
    signal = torch.randn(2, 3, 400)
    assert not model.quantize_1.layers[0].embedding.initted.item()

    codes = model.tokenize(signal)  # initializes the cold codebooks once
    assert codes.shape == (4, 2, 2, 6) and codes.dtype == torch.long
    state = {k: v.clone() for k, v in model.state_dict().items()}
    torch.testing.assert_close(model.tokenize(signal), codes)
    for name, value in state.items():
        torch.testing.assert_close(model.state_dict()[name], value)

    time, spatial = model._embedding_indices(signal.device)
    _, forward_codes = model._encode(model._patches(signal), time, spatial)
    # Later residual stages subtract the straight-through ``z + (q - z)`` instead
    # of ``q``; the rounding can flip near-ties, so compare the first stage.
    torch.testing.assert_close(forward_codes[:, 0], codes[:, 0])

    model.train()
    _, reconstruction = model(signal)
    assert model.quantize_1.layers[0].cluster_size.sum() > 0
    reconstruction.square().mean().backward()
    assert model.encode_task_layer_1[0].weight.grad is not None


def test_neurorvq_tokenizer_standardizes_each_window():
    model = _small_neurorvq_tokenizer().eval()
    target, reconstruction = model(5.0 * torch.randn(2, 3, 400) + 3.0)
    assert target.shape == reconstruction.shape == (2, 6, 200)
    for output in (target, reconstruction):
        torch.testing.assert_close(output.mean(dim=(1, 2)), torch.zeros(2), atol=1e-5, rtol=0)
        torch.testing.assert_close(output.std(dim=(1, 2)), torch.ones(2), atol=1e-4, rtol=0)


def test_neurorvq_ema_quantizer_matches_normalized_ema_update():
    quantizer = _EMAVectorQuantizer(n_codes=2, code_dim=2).train()
    quantizer.decay = 0.5
    with torch.no_grad():
        quantizer.embedding.weight.copy_(torch.eye(2))
        quantizer.embedding.initted.fill_(True)

    _, indices = quantizer(torch.tensor([[[[0.8, -0.6]], [[0.6, 0.8]]]]))

    expected = torch.nn.functional.normalize(torch.tensor([[0.9, 0.3], [-0.3, 0.9]]))
    assert indices.tolist() == [0, 1]
    torch.testing.assert_close(quantizer.embedding.weight, expected)
    torch.testing.assert_close(quantizer.cluster_size, torch.tensor([0.5, 0.5]))


@pytest.mark.parametrize("statistic_code_usage", [True, False])
def test_neurorvq_ema_quantizer_eval_code_usage(statistic_code_usage):
    quantizer = _EMAVectorQuantizer(2, 2, statistic_code_usage).eval()
    quantizer.decay = 0.5
    with torch.no_grad():
        quantizer.embedding.weight.copy_(torch.eye(2))
        quantizer.embedding.initted.fill_(True)

    # (batch, code_dim, 1, 3): three vectors, two nearest to code 0.
    quantizer(torch.tensor([[[[1.0, 0.9, 0.1]], [[0.1, -0.2, 1.0]]]]))

    expected = [1.0, 0.5] if statistic_code_usage else [0.0, 0.0]
    torch.testing.assert_close(quantizer.cluster_size, torch.tensor(expected))
    torch.testing.assert_close(quantizer.embedding.weight, torch.eye(2))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sfreq": 250}, "200 Hz"),
        ({"n_times": 401}, "divisible by patch_size"),
        ({"channel_names": ("f3", "f4", "x")}, "Unsupported NeuroRVQ channel"),
    ],
)
def test_neurorvq_tokenizer_rejects_incompatible_signal_metadata(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _small_neurorvq_tokenizer(**kwargs)


# ---------------------------------------------------------------------------

def test_neurorvq_block_qk_norm_factory_receives_eps():
    """``_Block.qk_norm`` is called as ``qk_norm(head_dim, eps=1e-6)``."""
    from braindecode.models.neurorvq import _Block

    calls = []

    def factory(head_dim, eps):
        calls.append((head_dim, eps))
        return torch.nn.LayerNorm(head_dim, eps=eps)

    _Block(
        dim=16,
        num_heads=4,
        mlp_ratio=2,
        qkv_bias=True,
        qk_norm=factory,
        drop=0,
        attn_drop=0,
        drop_path=0,
        init_values=1e-5,
    )
    assert calls == [(4, 1e-6), (4, 1e-6)]

# SeizureTransformer
# ---------------------------------------------------------------------------


def _small_seizure_transformer(n_times):
    from braindecode.models import SeizureTransformer

    return SeizureTransformer(
        n_chans=4, n_outputs=2, n_times=n_times, num_layers=1, dim_feedforward=64
    ).eval()


@pytest.mark.parametrize(
    "n_times, input_times",
    [
        (1024, 1024),
        # 1001 is odd at four of the five pooling levels.
        (1001, 1001),
        # Inputs shorter than n_times are accepted.
        (1024, 999),
    ],
)
def test_seizure_transformer_predicts_every_input_sample(n_times, input_times):
    model = _small_seizure_transformer(n_times)
    with torch.no_grad():
        out = model(torch.randn(2, 4, input_times))
    assert out.shape == (2, 2, input_times)


def test_seizure_transformer_rejects_inputs_longer_than_n_times():
    model = _small_seizure_transformer(256)
    with pytest.raises(ValueError, match="at most 256"):
        model(torch.randn(1, 4, 512))


def test_seizure_transformer_rejects_invalid_construction():
    from braindecode.models import SeizureTransformer

    with pytest.raises(ValueError, match="same length"):
        SeizureTransformer(n_chans=4, n_outputs=1, n_times=256, n_filters=(8, 16))
    with pytest.raises(ValueError, match="num_heads"):
        SeizureTransformer(n_chans=4, n_outputs=1, n_times=256, num_heads=3)

# ---------------------------------------------------------------------------
# TFMTokenizer
# ---------------------------------------------------------------------------


def _small_tfm_tokenizer(**kwargs):
    params = dict(
        sfreq=200,
        embed_dim=16,
        codebook_size=32,
        freq_encoder_depth=1,
        temporal_encoder_depth=1,
        decoder_depth=1,
        max_seq_len=32,
    )
    return TFMTokenizer(**{**params, **kwargs})


def test_tfm_tokenizer_tokenize_outputs_and_masks():
    model = _small_tfm_tokenizer().eval()
    x = torch.randn(2, 3, 500)
    target = model.compute_spectrogram(x)
    mask_a, mask_b = model.make_complementary_masks(target)

    # One mask for every trial and channel, and its exact complement.
    assert torch.equal(mask_b, ~mask_a)
    assert torch.equal(mask_a[0, 0], mask_a[-1, -1])
    assert not mask_a.all() and mask_a.any()

    out = model.tokenize(x, spectrogram_mask=mask_a)
    assert out.reconstruction.shape == target.shape == (2, 3, 100, 4)
    assert out.token_ids.shape == (2, 3, 4)
    assert 0 <= out.token_ids.min() and out.token_ids.max() < 32
    assert out.quantized.shape == out.embeddings.shape == (6, 4, 16)
    torch.testing.assert_close(out.target_spectrogram, target)
    torch.testing.assert_close(model(x, spectrogram_mask=mask_a), out.reconstruction)


def test_tfm_tokenizer_codebook_is_ema_only():
    torch.manual_seed(7)
    model = _small_tfm_tokenizer(codebook_size=64)
    before = model.quantizer.embed.clone()

    out = model.tokenize(torch.randn(1, 1, 200))
    out.quantization_loss.backward()

    # The VQ loss alone trains both encoder paths; the codebook moves by EMA.
    assert model.frequency_patch_embedding[0].weight.grad.norm() > 0
    assert model.temporal_patch_embedding[0].weight.grad.norm() > 0
    assert not torch.equal(model.quantizer.embed, before)
    # No EMA update in eval mode.
    state = {k: v.clone() for k, v in model.quantizer.state_dict().items()}
    model.eval().tokenize(torch.randn(1, 1, 200))
    for k, v in model.quantizer.state_dict().items():
        torch.testing.assert_close(v, state[k])
    assert "stft_window" not in model.state_dict()


# ---------------------------------------------------------------------------
# CSBrain
# ---------------------------------------------------------------------------

_CSBRAIN_2A_NAMES = (
    "Fz FC3 FC1 FCz FC2 FC4 C5 C3 C1 Cz C2 C4 C6 CP3 CP1 CPz CP2 CP4 P1 Pz P2 POz"
).split()


def test_csbrain_region_of_electrode_prefixes():
    assert region_of_electrode("Fpz") == REGION_FRONTAL
    assert region_of_electrode("AF7") == REGION_FRONTAL
    assert region_of_electrode("Fz") == REGION_FRONTAL
    assert region_of_electrode("FC3") == REGION_FRONTAL
    assert region_of_electrode("C3") == REGION_CENTRAL
    assert region_of_electrode("CPz") == REGION_CENTRAL
    assert region_of_electrode("P8") == REGION_PARIETAL
    assert region_of_electrode("PO7") == REGION_OCCIPITAL
    assert region_of_electrode("Oz") == REGION_OCCIPITAL
    assert region_of_electrode("T7") == REGION_TEMPORAL
    # Generic labels fall back to the central region instead of failing.
    assert region_of_electrode("EEG 021") == REGION_CENTRAL
    assert region_of_electrode("") == REGION_CENTRAL


def test_csbrain_name_rule_matches_reference_physionet_layout():
    """The name rule reproduces the reference PhysioNet-MI 64-channel layout
    (``models/model_for_physio.py`` of yuchen2199/CSBrain)."""
    names = (
        "FC5 FC3 FC1 FCZ FC2 FC4 FC6 C5 C3 C1 CZ C2 C4 C6 CP5 CP3 CP1 CPZ CP2 "
        "CP4 CP6 FP1 FPZ FP2 AF7 AF3 AFZ AF4 AF8 F7 F5 F3 F1 FZ F2 F4 F6 F8 "
        "FT7 FT8 T7 T8 T9 T10 TP7 TP8 P7 P5 P3 P1 PZ P2 P4 P6 P8 PO7 PO3 POZ "
        "PO4 PO8 O1 OZ O2 IZ"
    ).split()
    reference = [0] * 7 + [4] * 14 + [0] * 17 + [2] * 8 + [1] * 9 + [3] * 9
    assert [region_of_electrode(n) for n in names] == reference


def test_csbrain_derive_brain_regions_sorts_regions_contiguous():
    chs_info = [{"ch_name": n, "kind": "eeg"} for n in _CSBRAIN_2A_NAMES]
    ordered, sorted_indices = derive_brain_regions(chs_info)

    # Regions are contiguous and ordered by identifier: frontal(6),
    # parietal(3), occipital(1, POz), central(12).
    area_config = make_area_config(ordered)
    assert [area_config[k]["channels"] for k in sorted(area_config)] == [6, 3, 1, 12]
    # The permutation keeps every channel once.
    assert sorted(sorted_indices) == list(range(len(_CSBRAIN_2A_NAMES)))
    # Inside a region the original order is preserved.
    central = [_CSBRAIN_2A_NAMES[i] for i, r in zip(sorted_indices, ordered) if r == 4]
    assert central == _CSBRAIN_2A_NAMES[6:18]


def test_csbrain_brain_regions_reproduce_reference_2a_layout():
    """``brain_regions`` reproduces the reference BCI-IV-2a layout and order."""
    regions_2a = [0] + [4] * 17 + [1] * 4
    model = CSBrain(
        n_outputs=4,
        n_chans=22,
        n_times=800,
        brain_regions=regions_2a,
        n_layer=1,
    )
    # The reference topology keeps the input order inside each region.
    assert model.sorted_indices.tolist() == [0, *range(18, 22), *range(1, 18)]
    assert {k: v["channels"] for k, v in model.area_config.items()} == {
        "region_0": 1,
        "region_1": 4,
        "region_4": 17,
    }


def test_csbrain_region_attention_mask_groups_electrodes():
    # Two regions of 2 electrodes each -> 2 groups of 2.
    area_config = {
        "region_0": {"channels": 2, "slice": slice(0, 2)},
        "region_4": {"channels": 2, "slice": slice(2, 4)},
    }
    mask = build_region_attention_mask(area_config, n_channels=4)
    assert mask.shape == (4, 4)
    # Each electrode attends to itself and exactly one electrode per region.
    assert torch.all((mask == 0).sum(dim=1) == 2)
    # Groups are symmetric (either both allowed or both blocked).
    assert torch.equal(mask == 0, (mask == 0).T)


def test_csbrain_without_channel_names_skips_region_structure():
    model = CSBrain(n_outputs=2, n_chans=4, n_times=400, sfreq=200.0, n_layer=1)
    assert model.sorted_indices is None
    assert model.area_config == {}
    assert model.encoder[0].region_attn_mask is None
    assert model(torch.randn(2, 4, 400)).shape == (2, 2)


def test_csbrain_forward_with_region_structure():
    chs_info = [{"ch_name": n, "kind": "eeg"} for n in _CSBRAIN_2A_NAMES]
    model = CSBrain(n_outputs=4, chs_info=chs_info, n_times=800, sfreq=200.0, n_layer=2)
    assert model(torch.randn(2, 22, 800)).shape == (2, 4)
    feats = model(torch.randn(1, 22, 800), return_features=True)
    assert feats["features"].shape == (1, 22, 4, 200)


def test_csbrain_masked_forward_replaces_patches():
    model = CSBrain(n_outputs=2, n_chans=3, n_times=400, sfreq=200.0, n_layer=1)
    mask = torch.zeros(1, 3, 2, dtype=torch.bool)
    mask[:, :, 0] = True
    assert model(torch.randn(1, 3, 400), mask=mask).shape == (1, 2)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_csbrain_train_mode_small_batches(batch_size):
    model = CSBrain(n_outputs=2, n_chans=3, n_times=400, sfreq=200.0, n_layer=1)
    model.train()
    assert model(torch.randn(batch_size, 3, 400)).shape == (batch_size, 2)


def test_csbrain_head_drop_prob_sets_only_the_head():
    model = CSBrain(
        n_outputs=2, n_chans=3, n_times=400, n_layer=1, drop_prob=0.1, head_drop_prob=0.3
    )
    ps = {
        name.startswith("final_layer"): m.p
        for name, m in model.named_modules()
        if isinstance(m, nn.Dropout)
    }
    assert ps == {False: 0.1, True: 0.3}


def test_csbrain_init_keeps_residual_stream_bounded():
    """Only Linear layers are re-initialised, as in the reference.

    A fan-out Kaiming init on the Conv2d embeddings grew the residual stream
    ~3x per layer (features ~1e6 at 12 layers), which made
    ``test_model_compiled[CSBrain]`` flaky. With the reference init the
    features stay around 1e2-1e3.
    """
    set_random_seeds(0, cuda=False)
    model = CSBrain(n_outputs=2, n_chans=22, n_times=1000).eval()
    with torch.no_grad():
        feats = model(torch.randn(1, 22, 1000), return_features=True)["features"]
    assert feats.abs().max() < 1e4


@pytest.mark.parametrize("n_times, hidden", [(800, 800), (2000, 2000)])
def test_csbrain_head_hidden_width_follows_patches(n_times, hidden):
    """The reference head is ``n_chans * n_patch * 200 -> n_patch * 200 -> 200``."""
    model = CSBrain(n_outputs=3, n_chans=4, n_times=n_times, sfreq=200.0, n_layer=1)
    head = model.final_layer
    assert head[1].in_features == 4 * hidden
    assert head[1].out_features == hidden
    assert head[4].out_features == 200
    model.reset_head(5)
    assert model.final_layer[1].out_features == hidden
    assert model.final_layer[-1].out_features == 5


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(chs_info=[{"ch_name": n} for n in ("F3", "C3", "P3", "O1")], n_times=2000),
        dict(n_chans=4, input_window_seconds=10.0, sfreq=200.0),
    ],
)
def test_csbrain_head_width_follows_derived_shapes(kwargs):
    """n_chans from chs_info and n_times from the window give the same head."""
    model = CSBrain(n_outputs=2, n_layer=1, **kwargs)
    assert model.final_layer[1].in_features == 4 * 2000
    assert model.final_layer[1].out_features == 2000


def test_csbrain_head_hidden_dim_overrides_the_reference_width():
    """SEED-V's reference head is 62 * 1 * 200 -> 800 -> 200 -> 5."""
    model = CSBrain(n_outputs=5, n_chans=62, n_times=200, head_hidden_dim=800, n_layer=1)
    shapes = [tuple(m.weight.shape) for m in model.final_layer if hasattr(m, "weight")]
    assert shapes == [(800, 62 * 200), (200, 800), (5, 200)]
    model.reset_head(3)
    assert model.final_layer[1].out_features == 800
    assert model.n_outputs == 3
    clone = CSBrain.from_config(model.get_config())
    assert tuple(clone.final_layer[1].weight.shape) == (800, 62 * 200)
    assert tuple(clone.final_layer[-1].weight.shape) == (3, 200)


def test_csbrain_rejects_brain_regions_of_wrong_length():
    with pytest.raises(ValueError, match="brain_regions has 3 entries for 4"):
        CSBrain(n_outputs=2, n_chans=4, n_times=400, brain_regions=[0, 1, 2], n_layer=1)


@pytest.mark.parametrize("model_class", [CBraMod, CSBrain])
def test_cbramod_patch_embedding_patch_size_not_200(model_class):
    """The shared patch embedding sizes its rFFT bins from ``patch_size``."""
    model = model_class(n_outputs=2, n_chans=3, n_times=800, patch_size=400, n_layer=1)
    assert model(torch.randn(2, 3, 800)).shape == (2, 2)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="needs CUDA"
            ),
        ),
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="needs MPS"
            ),
        ),
    ],
)
def test_csbrain_patch_embedding_same_seed_init_is_stem_first(device):
    """Same seed gives CSBrain's former stem-first patch-embedding init.

    The Linear weight is skipped: ``_weights_init`` redraws it afterwards.
    """
    with torch.device(device):
        torch.manual_seed(0)
        emb = CSBrain(n_outputs=2, n_chans=4, n_times=400, n_layer=1).patch_embedding
        torch.manual_seed(0)
        d_model = emb.d_model
        reference = [
            nn.Conv2d(1, 25, (1, 49), (1, 25), (0, 24)),
            nn.Conv2d(25, 25, (1, 3), (1, 1), (0, 1)),
            nn.Conv2d(25, 25, (1, 3), (1, 1), (0, 1)),
            nn.Conv2d(d_model, d_model, (19, 7), padding=(9, 3), groups=d_model),
            nn.Linear(101, d_model),
        ]
    built = [*emb.proj_in[::3], emb.positional_encoding[0], emb.spectral_proj[0]]
    reference[-1].weight = built[-1].weight
    for ref, layer in zip(reference, built):
        for ref_param, param in zip(ref.parameters(), layer.parameters()):
            torch.testing.assert_close(param, ref_param, rtol=0, atol=0)


def test_csbrain_channel_order_reproduces_reference_topology():
    """``channel_order`` takes the reference's ``sorted_indices`` (CHB-MIT)."""
    regions = [0, 0, 2, 1, 0, 0, 2, 1, 0, 0, 4, 1, 0, 0, 4, 1]
    order = [1, 0, 8, 9, 13, 12, 4, 5, 3, 11, 15, 7, 2, 6, 10, 14]
    model = CSBrain(
        n_outputs=1,
        n_chans=16,
        n_times=400,
        brain_regions=regions,
        channel_order=order,
        n_layer=1,
    )
    assert model.sorted_indices.tolist() == order
    assert [v["channels"] for v in model.area_config.values()] == [8, 4, 2, 2]
    assert model(torch.randn(2, 16, 400)).shape == (2, 1)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(brain_regions=[0, 4, 1], channel_order=[0, 1, 1]), "permutation"),
        (dict(brain_regions=[0, 4, 1], channel_order=[0, 1, 2]), "ascending"),
        (dict(channel_order=[0, 1, 2]), "needs brain_regions"),
    ],
)
def test_csbrain_rejects_invalid_channel_order(kwargs, match):
    with pytest.raises(ValueError, match=match):
        CSBrain(n_outputs=2, n_chans=3, n_times=400, n_layer=1, **kwargs)


# ----------------------------------------------------------------------------
# EEGCLIP


def test_eegclip_matches_authors_projection_and_clip_loss():
    """Reference: ``EEGClip/clip_models.py`` and ``loss_methods.py`` @1d6b89b."""
    torch.manual_seed(0)
    model = EEGCLIP(
        n_chans=21, n_times=1200, n_outputs=64, text_embedding_dim=768, drop_prob=0
    )
    X, text = torch.randn(4, 21, 1200), torch.randn(4, 768)
    torch.manual_seed(1)  # same Deep4Net dropout masks in both passes (train mode)
    paired = model.forward_paired(X, text)
    torch.manual_seed(1)
    features = model.eeg_encoder(X)
    # Authors' Deep4Net ends with a log-softmax over its 128 outputs.
    torch.testing.assert_close(
        features.exp().sum(dim=1), torch.ones(4, 519), rtol=0, atol=1e-4
    )
    # Authors' ProjectionHead(transpose=True) on [B, N_pred, 128], mean over time.
    x = features.transpose(1, 2)
    for layer in model.final_layer:
        if isinstance(layer, nn.BatchNorm1d):
            x = layer(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = layer(x)
    eeg = x.mean(dim=1)
    torch.testing.assert_close(paired["eeg_embeds"], eeg)
    # ClipLoss: raw (not exponentiated) logit_scale, no L2 normalization.
    t = model.text_projection(text)
    labels = torch.arange(4)
    logits = model.logit_scale * eeg @ t.T
    expected = (
        nn.functional.cross_entropy(logits, labels)
        + nn.functional.cross_entropy(model.logit_scale * t @ eeg.T, labels)
    ) / 2
    loss = model.contrastive_loss(paired["eeg_embeds"], paired["text_embeds"])
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert model.logit_scale.grad is not None


def test_eegclip_custom_encoders_and_masked_mean_pooling():
    class TextEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 8)

        def forward(self, input_ids, attention_mask=None):
            return (self.embedding(input_ids),)  # tuple, like return_dict=False

    model = EEGCLIP(
        n_chans=3,
        n_times=20,
        n_outputs=4,
        eeg_encoder=nn.Flatten(),  # (batch, features) output
        eeg_embedding_dim=60,
        text_encoder=TextEncoder(),
        text_embedding_dim=8,
        text_pooling="mean",
    ).eval()
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    tok = model.text_encoder.embedding(tokens)
    expected = torch.stack([tok[0, :2].mean(dim=0), tok[1, :1].mean(dim=0)])
    torch.testing.assert_close(
        model.encode_text(tokens, attention_mask=mask),
        model.text_projection(expected),
    )
    paired = model.forward_paired(torch.randn(2, 3, 20), tokens, attention_mask=mask)
    assert paired["logits_per_eeg"].shape == (2, 2)

    model.reset_head(6)
    assert model.text_projection[-1].out_features == 6
    assert model.text_projection.training is False
    with pytest.raises(ValueError, match="custom encoder"):
        model.get_config()
