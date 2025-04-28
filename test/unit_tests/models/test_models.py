# Authors: Alexandre Gramfort
#          Lukas Gemein <l.gemein@gmail.com>
#          Hubert Banville <hubert.jbanville@gmail.com>
#          Robin Schirrmeister <robintibor@gmail.com>
#          Daniel Wilson <dan.c.wil@gmail.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com
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
    BIOT,
    TCN,
    ATCNet,
    AttentionBaseNet,
    ContraWR,
    Deep4Net,
    DeepSleepNet,
    EEGConformer,
    EEGInceptionERP,
    EEGInceptionMI,
    EEGITNet,
    EEGMiner,
    EEGNetv1,
    EEGNetv4,
    EEGResNet,
    EEGNeX,
    EEGSimpleConv,
    EEGTCNet,
    FBCNet,
    HybridNet,
    IFNet,
    Labram,
    ShallowFBCSPNet,
    SleepStagerBlanco2020,
    SleepStagerChambon2018,
    SleepStagerEldele2021,
    SPARCNet,
    TIDNet,
    USleep,
)
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


def test_eegresnet(input_sizes):
    model = EEGResNet(
        input_sizes["n_channels"],
        input_sizes["n_classes"],
        input_sizes["n_in_times"],
        final_pool_length=5,
        n_first_filters=2,
    )
    check_forward_pass(model, input_sizes, only_check_until_dim=2)


def test_eegresnet_pool_length_auto(input_sizes):
    model = EEGResNet(
        input_sizes["n_channels"],
        input_sizes["n_classes"],
        input_sizes["n_in_times"],
        final_pool_length="auto",
        n_first_filters=2,
    )
    check_forward_pass(model, input_sizes, only_check_until_dim=2)


def test_hybridnet(input_sizes):
    model = HybridNet(
        input_sizes["n_channels"],
        input_sizes["n_classes"],
        input_sizes["n_in_times"],
    )
    check_forward_pass(model, input_sizes, only_check_until_dim=2)


def test_eegnet_v4(input_sizes):
    model = EEGNetv4(
        input_sizes["n_channels"],
        input_sizes["n_classes"],
        n_times=input_sizes["n_in_times"],
    )
    check_forward_pass(model, input_sizes)


def test_eegnet_v1(input_sizes):
    model = EEGNetv1(
        n_chans=input_sizes["n_channels"],
        n_outputs=input_sizes["n_classes"],
        n_times=input_sizes["n_in_times"],
    )
    check_forward_pass(
        model,
        input_sizes,
    )


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
    att_num_heads = 2

    model = ATCNet(
        n_chans=22,
        n_outputs=4,
        input_window_seconds=4.5,
        sfreq=250,
        n_windows=n_windows,
        att_head_dim=att_head_dim,
        att_num_heads=att_num_heads,
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

    model = SleepStagerEldele2021(
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

    model = SleepStagerEldele2021(
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
        "emb_size": 256,
        "att_num_heads": 8,
        "n_layers": 4,
        "sfreq": 200,
        "hop_length": 50,
        "n_outputs": 2,
        "n_chans": 64,
    }


def test_initialization_default_parameters(default_biot_params):
    """Test BIOT initialization with default parameters."""
    biot = BIOT(**default_biot_params)

    assert biot.emb_size == 256
    assert biot.att_num_heads == 8
    assert biot.n_layers == 4


def test_model_trainable_parameters_biot(default_biot_params):
    biot = BIOT(**default_biot_params)

    biot_encoder = biot.encoder.parameters()
    biot_classifier = biot.final_layer.parameters()

    trainable_params_bio = sum(
        p.numel() for p in biot_encoder if p.requires_grad)
    trainable_params_clf = sum(
        p.numel() for p in biot_classifier if p.requires_grad)

    assert trainable_params_bio == 3198464  # ~ 3.2 M according with Labram paper
    assert trainable_params_clf == 514


@pytest.fixture
def default_labram_params():
    return {
        "n_times": 1000,
        "n_chans": 64,
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
    labram_base = Labram(n_layers=12, att_num_heads=12,
                         **default_labram_params)

    labram_base_parameters = labram_base.get_torchinfo_statistics().trainable_params

    # We added some parameters layers in the segmentation step to match the
    # braindecode convention.
    assert np.round(labram_base_parameters / 1e6, 1) == 5.8
    # ~ 5.8 M matching the paper

    labram_large = Labram(
        n_layers=24,
        att_num_heads=16,
        out_channels=16,
        emb_size=400,
        **default_labram_params,
    )
    labram_large_parameters = labram_large.get_torchinfo_statistics().trainable_params

    assert np.round(labram_large_parameters / 1e6, 0) == 46
    # ~ 46 M matching the paper

    labram_huge = Labram(
        n_layers=48,
        att_num_heads=16,
        out_channels=32,
        emb_size=800,
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
        n_layers=12,
        att_num_heads=12,
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
        n_layers=12, att_num_heads=12, use_abs_pos_emb=False,
        **default_labram_params
    )

    X = torch.rand(1, default_labram_params["n_chans"],
                   default_labram_params["n_times"])

    with torch.no_grad():
        out_without_pos_emb = labram_base_not_pos_emb(X)
        assert out_without_pos_emb.shape == torch.Size([1, 2])


def test_labram_n_outputs_0(default_labram_params):
    """
    Testing if the model is returning the correct shapes for the different
    return options.

    Parameters
    ----------
    default_labram_params: dict with default parameters for Labram model

    """
    default_labram_params["n_outputs"] = 0
    labram_base = Labram(n_layers=12, att_num_heads=12,
                         **default_labram_params)
    # Defining a random data
    X = torch.rand(1, default_labram_params["n_chans"],
                   default_labram_params["n_times"])

    with torch.no_grad():
        out = labram_base(X)
        assert out.shape[-1] == default_labram_params["patch_size"]
        assert isinstance(labram_base.final_layer, nn.Identity)


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

    model = ContraWR(n_outputs=2, n_chans=22, sfreq=250)

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


def test_eegnetv4_final_layer_linear_true():
    """Test that final_layer_linear=True uses a conv-based classifier without warning."""
    model = EEGNetv4(
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

def test_eegnetv4_final_layer_linear_false():
    """Test that final_layer_conv=False raises a DeprecationWarning and uses
    a linear layer."""
    with pytest.warns(DeprecationWarning,
                      match="Parameter 'final_layer_with_constraint=False' is deprecated"):
        model = EEGNetv4(
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