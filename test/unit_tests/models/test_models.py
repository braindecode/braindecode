# Authors: Alexandre Gramfort
#          Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD-3


import numpy as np
import torch
import pytest

from braindecode.models import (
    Deep4Net, EEGNetv4, EEGNetv1, HybridNet, ShallowFBCSPNet, EEGResNet, TCN,
    SleepStager)


def test_shallow_fbcsp_net():
    rng = np.random.RandomState(42)
    n_channels = 18
    n_in_times = 200
    n_classes = 2
    n_samples = 7
    X = rng.randn(n_samples, n_channels, n_in_times, 1)
    X = torch.Tensor(X.astype(np.float32))
    model = ShallowFBCSPNet(
        n_channels, n_classes, n_in_times, final_conv_length="auto"
    )
    y_pred = model(X)
    assert y_pred.shape == (n_samples, n_classes)


def test_deep4net():
    rng = np.random.RandomState(42)
    n_channels = 18
    n_in_times = 600
    n_classes = 2
    n_samples = 7
    X = rng.randn(n_samples, n_channels, n_in_times, 1)
    X = torch.Tensor(X.astype(np.float32))
    model = Deep4Net(
        n_channels, n_classes, n_in_times, final_conv_length="auto"
    )
    y_pred = model(X)
    assert y_pred.shape == (n_samples, n_classes)


def test_eegresnet():
    rng = np.random.RandomState(42)
    n_channels = 18
    n_in_times = 600
    n_classes = 2
    n_samples = 7
    X = rng.randn(n_samples, n_channels, n_in_times, 1)
    X = torch.Tensor(X.astype(np.float32))
    model = EEGResNet(
        n_channels,
        n_classes,
        n_in_times,
        final_pool_length=5,
        n_first_filters=2,
    )
    y_pred = model(X)
    assert y_pred.shape[:2] == (n_samples, n_classes)


def test_hybridnet():
    rng = np.random.RandomState(42)
    n_channels = 18
    n_in_times = 600
    n_classes = 2
    n_samples = 7
    X = rng.randn(n_samples, n_channels, n_in_times, 1)
    X = torch.Tensor(X.astype(np.float32))
    model = HybridNet(n_channels, n_classes, n_in_times)
    y_pred = model(X)
    assert y_pred.shape[:2] == (n_samples, n_classes)


def test_eegnet_v4():
    rng = np.random.RandomState(42)
    n_channels = 18
    n_in_times = 500
    n_classes = 2
    n_samples = 7
    X = rng.randn(n_samples, n_channels, n_in_times, 1)
    X = torch.Tensor(X.astype(np.float32))
    model = EEGNetv4(n_channels, n_classes, input_window_samples=n_in_times)
    y_pred = model(X)
    assert y_pred.shape == (n_samples, n_classes)


def test_eegnet_v1():
    rng = np.random.RandomState(42)
    n_channels = 18
    n_in_times = 500
    n_classes = 2
    n_samples = 7
    X = rng.randn(n_samples, n_channels, n_in_times, 1)
    X = torch.Tensor(X.astype(np.float32))
    model = EEGNetv1(n_channels, n_classes, input_window_samples=n_in_times)
    y_pred = model(X)
    assert y_pred.shape == (n_samples, n_classes)


def test_tcn():
    rng = np.random.RandomState(42)
    n_channels = 18
    n_classes = 2
    n_samples = 7
    model = TCN(
        n_channels, n_classes,
        n_filters=5, n_blocks=2, kernel_size=4, drop_prob=.5,
        add_log_softmax=True)
    n_in_times = model.min_len
    X = rng.randn(n_samples, n_channels, n_in_times)
    X = torch.Tensor(X.astype(np.float32))
    y_pred = model(X)
    assert y_pred.shape == (n_samples, n_classes)


@pytest.mark.parametrize('n_channels,sfreq,n_classes,input_size_s',
                         [(20, 128, 5, 30), (10, 256, 4, 20), (1, 64, 2, 30)])
def test_sleep_stager(n_channels, sfreq, n_classes, input_size_s):
    rng = np.random.RandomState(42)
    time_conv_size_s = 0.5
    max_pool_size_s = 0.125
    n_examples = 10

    model = ChambonSleepStager(
        n_channels, sfreq, n_conv_chs=8, time_conv_size_s=time_conv_size_s,
        max_pool_size_s=max_pool_size_s, n_classes=n_classes,
        input_size_s=input_size_s, dropout=0.25)

    X = rng.randn(n_examples, n_channels, int(sfreq * input_size_s))
    X = torch.from_numpy(X.astype(np.float32))

    y_pred = model(X)
    assert y_pred.shape == (n_examples, n_classes)
