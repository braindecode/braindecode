# Authors: Alexandre Gramfort
#
# License: BSD-3

import numpy as np

import torch as th

from braindecode.models import Deep4Net
from braindecode.models import EEGNetv4, EEGNetv1
from braindecode.models import HybridNet
from braindecode.models import ShallowFBCSPNet
from braindecode.models import EEGResNet


def test_shallow_fbcsp_net():
    rng = np.random.RandomState(42)
    n_channels = 18
    n_in_times = 200
    n_classes = 2
    n_samples = 7
    X = rng.randn(n_samples, n_channels, n_in_times, 1)
    X = th.Tensor(X.astype(np.float32))
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
    X = th.Tensor(X.astype(np.float32))
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
    X = th.Tensor(X.astype(np.float32))
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
    X = th.Tensor(X.astype(np.float32))
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
    X = th.Tensor(X.astype(np.float32))
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
    X = th.Tensor(X.astype(np.float32))
    model = EEGNetv1(n_channels, n_classes, input_window_samples=n_in_times)
    y_pred = model(X)
    assert y_pred.shape == (n_samples, n_classes)
