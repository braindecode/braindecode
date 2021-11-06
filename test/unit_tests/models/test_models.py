# Authors: Alexandre Gramfort
#          Lukas Gemein <l.gemein@gmail.com>
#          Hubert Banville <hubert.jbanville@gmail.com>
#          Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD-3


import numpy as np
import torch
import pytest

from braindecode.models import (
    Deep4Net, EEGNetv4, EEGNetv1, HybridNet, ShallowFBCSPNet, EEGResNet, TCN,
    SleepStagerChambon2018, SleepStagerBlanco2020, SleepStagerEldele2021, USleep, TIDNet)
from braindecode.util import set_random_seeds


@pytest.fixture(scope="module")
def input_sizes():
    return dict(n_channels=18, n_in_times=600, n_classes=2, n_samples=7)


def check_forward_pass(model, input_sizes, only_check_until_dim=None):
    # Test 4d Input
    set_random_seeds(0, False)
    rng = np.random.RandomState(42)
    X = rng.randn(input_sizes['n_samples'], input_sizes['n_channels'],
                  input_sizes['n_in_times'], 1)
    X = torch.Tensor(X.astype(np.float32))
    y_pred = model(X)
    assert y_pred.shape[:only_check_until_dim] == (
        input_sizes['n_samples'], input_sizes['n_classes'])

    # Test 3d input
    set_random_seeds(0, False)
    X = X.squeeze(-1)
    assert len(X.shape) == 3
    y_pred_new = model(X)
    assert y_pred_new.shape[:only_check_until_dim] == (
        input_sizes['n_samples'], input_sizes['n_classes'])
    np.testing.assert_allclose(y_pred.detach().cpu().numpy(),
                               y_pred_new.detach().cpu().numpy())


def test_shallow_fbcsp_net(input_sizes):
    model = ShallowFBCSPNet(
        input_sizes['n_channels'], input_sizes['n_classes'],
        input_sizes['n_in_times'], final_conv_length="auto"
    )
    check_forward_pass(model, input_sizes)


def test_deep4net(input_sizes):
    model = Deep4Net(
        input_sizes['n_channels'], input_sizes['n_classes'],
        input_sizes['n_in_times'], final_conv_length="auto"
    )
    check_forward_pass(model, input_sizes)


def test_eegresnet(input_sizes):
    model = EEGResNet(
        input_sizes['n_channels'],
        input_sizes['n_classes'],
        input_sizes['n_in_times'],
        final_pool_length=5,
        n_first_filters=2,
    )
    check_forward_pass(model, input_sizes, only_check_until_dim=2)


def test_eegresnet_pool_length_auto(input_sizes):
    model = EEGResNet(
        input_sizes['n_channels'],
        input_sizes['n_classes'],
        input_sizes['n_in_times'],
        final_pool_length='auto',
        n_first_filters=2,
    )
    check_forward_pass(model, input_sizes, only_check_until_dim=2)


def test_hybridnet(input_sizes):
    model = HybridNet(
        input_sizes['n_channels'], input_sizes['n_classes'],
        input_sizes['n_in_times'],)
    check_forward_pass(model, input_sizes, only_check_until_dim=2)


def test_eegnet_v4(input_sizes):
    model = EEGNetv4(
        input_sizes['n_channels'], input_sizes['n_classes'],
        input_window_samples=input_sizes['n_in_times'])
    check_forward_pass(model, input_sizes)


def test_eegnet_v1(input_sizes):
    model = EEGNetv1(
        input_sizes['n_channels'], input_sizes['n_classes'],
        input_window_samples=input_sizes['n_in_times'])
    check_forward_pass(model, input_sizes,)


def test_tcn(input_sizes):
    model = TCN(
        input_sizes['n_channels'], input_sizes['n_classes'],
        n_filters=5, n_blocks=2, kernel_size=4, drop_prob=.5,
        add_log_softmax=True)
    check_forward_pass(model, input_sizes, only_check_until_dim=2)


@pytest.mark.parametrize('n_channels,sfreq,n_classes,input_size_s',
                         [(20, 128, 5, 30), (10, 256, 4, 20), (1, 64, 2, 30)])
def test_sleep_stager(n_channels, sfreq, n_classes, input_size_s):
    rng = np.random.RandomState(42)
    time_conv_size_s = 0.5
    max_pool_size_s = 0.125
    pad_size_s = 0.25
    n_examples = 10

    model = SleepStagerChambon2018(
        n_channels, sfreq, n_conv_chs=8, time_conv_size_s=time_conv_size_s,
        max_pool_size_s=max_pool_size_s, pad_size_s=pad_size_s,
        input_size_s=input_size_s, n_classes=n_classes, dropout=0.25)
    model.eval()

    X = rng.randn(n_examples, n_channels, int(sfreq * input_size_s))
    X = torch.from_numpy(X.astype(np.float32))

    y_pred1 = model(X)  # 3D inputs
    y_pred2 = model(X.unsqueeze(1))  # 4D inputs
    assert y_pred1.shape == (n_examples, n_classes)
    assert y_pred2.shape == (n_examples, n_classes)
    np.testing.assert_allclose(y_pred1.detach().cpu().numpy(),
                               y_pred2.detach().cpu().numpy())


@pytest.mark.parametrize('in_chans,sfreq,n_classes,input_size_s',
                         [(20, 128, 5, 30), (10, 100, 4, 20), (1, 64, 2, 30)])
def test_usleep(in_chans, sfreq, n_classes, input_size_s):
    rng = np.random.RandomState(42)
    n_examples = 10
    seq_length = 3

    model = USleep(
        in_chans=in_chans, sfreq=sfreq, n_classes=n_classes,
        input_size_s=input_size_s, ensure_odd_conv_size=True)
    model.eval()

    X = rng.randn(n_examples, in_chans, int(sfreq * input_size_s))
    X = torch.from_numpy(X.astype(np.float32))

    y_pred1 = model(X)  # 3D inputs : (batch, channels, time)
    y_pred2 = model(X.unsqueeze(1))  # 4D inputs : (batch, 1, channels, time)
    y_pred3 = model(torch.stack([X for idx in range(seq_length)],
                                axis=1))  # (batch, sequence, channels, time)
    assert y_pred1.shape == (n_examples, n_classes)
    assert y_pred2.shape == (n_examples, n_classes)
    assert y_pred3.shape == (n_examples, n_classes, seq_length)
    np.testing.assert_allclose(y_pred1.detach().cpu().numpy(),
                               y_pred2.detach().cpu().numpy())


def test_usleep_n_params():
    """Make sure the number of parameters is the same as in the paper when
    using the same architecture hyperparameters.
    """
    model = USleep(
        in_chans=2, sfreq=128, depth=12, n_time_filters=5,
        complexity_factor=1.67, with_skip_connection=True, n_classes=5,
        input_size_s=30, time_conv_size_s=9 / 128)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == 3114337  # From paper's supplementary materials, Table 2


def test_sleep_stager_return_feats():
    n_channels = 2
    sfreq = 10
    input_size_s = 30
    n_classes = 3

    model = SleepStagerChambon2018(
        n_channels, sfreq, n_conv_chs=8, input_size_s=input_size_s,
        n_classes=n_classes, return_feats=True)
    model.eval()

    rng = np.random.RandomState(42)
    X = rng.randn(10, n_channels, int(sfreq * input_size_s))
    X = torch.from_numpy(X.astype(np.float32))

    out = model(X)
    assert out.shape == (10, model.len_last_layer)


def test_tidnet(input_sizes):
    model = TIDNet(
        input_sizes['n_channels'], input_sizes['n_classes'],
        input_sizes['n_in_times'],)
    check_forward_pass(model, input_sizes)


@pytest.mark.parametrize('sfreq,n_classes,input_size_s,d_model',
                         [(100, 5, 30, 80), (125, 4, 30, 100)])
def test_eldele_2021(sfreq, n_classes, input_size_s, d_model):
    # (100, 5, 30, 80) - Physionet Sleep
    # (125, 4, 30, 100) - SHHS
    rng = np.random.RandomState(42)
    n_channels = 1
    n_examples = 10

    model = SleepStagerEldele2021(sfreq=sfreq, n_classes=n_classes, input_size_s=input_size_s,
                                  d_model=d_model, return_feats=False)
    model.eval()

    X = rng.randn(n_examples, n_channels, np.ceil(input_size_s * sfreq).astype(int))
    X = torch.from_numpy(X.astype(np.float32))

    y_pred1 = model(X)  # 3D inputs
    assert y_pred1.shape == (n_examples, n_classes)


def test_eldele_2021_feats():
    n_channels = 1
    sfreq = 100
    input_size_s = 30
    n_classes = 3
    n_examples = 10

    model = SleepStagerEldele2021(sfreq, input_size_s=input_size_s, n_classes=n_classes,
                                  return_feats=True)
    model.eval()

    rng = np.random.RandomState(42)
    X = rng.randn(n_examples, n_channels, int(sfreq * input_size_s))
    X = torch.from_numpy(X.astype(np.float32))

    out = model(X)
    assert out.shape == (n_examples, model.len_last_layer)


@pytest.mark.parametrize('n_channels,sfreq,n_groups,n_classes,input_size_s',
                         [(20, 128, 2, 5, 30), (10, 100, 2, 4, 20), (1, 64, 1, 2, 30)])
def test_blanco_2020(n_channels, sfreq, n_groups, n_classes, input_size_s):
    rng = np.random.RandomState(42)
    n_examples = 10

    model = SleepStagerBlanco2020(n_channels=n_channels, sfreq=sfreq, n_groups=n_groups,
                                  input_size_s=input_size_s, n_classes=n_classes,
                                  return_feats=False)
    model.eval()

    X = rng.randn(n_examples, n_channels, np.ceil(input_size_s * sfreq).astype(int))
    X = torch.from_numpy(X.astype(np.float32))

    y_pred1 = model(X)  # 3D inputs
    y_pred2 = model(X.unsqueeze(2))  # 4D inputs
    assert y_pred1.shape == (n_examples, n_classes)
    assert y_pred2.shape == (n_examples, n_classes)
    np.testing.assert_allclose(y_pred1.detach().cpu().numpy(),
                               y_pred2.detach().cpu().numpy())


def test_blanco_2020_feats():
    n_channels = 2
    sfreq = 50
    input_size_s = 30
    n_classes = 3
    n_examples = 10

    model = SleepStagerBlanco2020(n_channels, sfreq, input_size_s=input_size_s,
                                  n_classes=n_classes, return_feats=True)
    model.eval()

    rng = np.random.RandomState(42)
    X = rng.randn(n_examples, n_channels, int(sfreq * input_size_s))
    X = torch.from_numpy(X.astype(np.float32))

    out = model(X)
    assert out.shape == (n_examples, model.len_last_layer)
