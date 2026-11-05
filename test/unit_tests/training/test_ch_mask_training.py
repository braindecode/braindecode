"""The ``ch_mask`` contract when a model is trained through braindecode.

Every model of ``test_ch_mask._CASES`` is fit on recordings with different
channel counts, batched by :func:`braindecode.datasets.pad_channels_collate`.
Each recording must get the output, and the gradients, it gets alone.
"""

import inspect

import mne
import numpy as np
import pytest
import torch
import torch.nn.functional as F

import braindecode.models
from braindecode import EEGClassifier
from braindecode.datasets import BaseConcatDataset, RawDataset, pad_channels_collate
from braindecode.models import ShallowFBCSPNet
from braindecode.preprocessing import create_fixed_length_windows
from test.unit_tests.models.test_ch_mask import _CASES, _FIXED_HEADS, _padded_case


def _windows(n_chans_per_recording, n_times, n_windows=3, pos=False):
    rng = np.random.default_rng(0)
    montage = mne.channels.make_standard_montage("biosemi32")
    recordings = []
    for target, n_chans in enumerate(n_chans_per_recording):
        info = mne.create_info(montage.ch_names[:n_chans], 100.0, "eeg")
        data = rng.standard_normal((n_chans, n_windows * n_times))
        raw = mne.io.RawArray(data, info, verbose="error").set_montage(montage)
        recordings.append(RawDataset(raw, description={"label": target}))
    windows = create_fixed_length_windows(
        BaseConcatDataset(recordings), window_size_samples=n_times,
        window_stride_samples=n_times, on_last_window="drop", preload=True,
    )
    windows.set_target("label")
    windows.set_return_ch_pos(pos)
    return windows


def _net(module, **kwargs):
    return EEGClassifier(
        module, max_epochs=1, batch_size=4, train_split=None,
        iterator_train__shuffle=False, iterator_train__drop_last=False,
        iterator_train__collate_fn=pad_channels_collate,
        iterator_valid__collate_fn=pad_channels_collate, **kwargs,
    )


# The loader routes ``pos`` only: these models need their own per-recording input.
_NEEDS_PER_RECORDING_INPUT = {"PopulationTransformer", "BaRISTA", "DIVER1"}
_XFAIL = pytest.mark.xfail(
    strict=True, raises=ValueError,
    reason="the loader routes no per-recording coords, spatial_indices or "
    "chan_metadata, and these models need them to tell recordings apart",
)


@pytest.mark.parametrize("name", [
    pytest.param(name, marks=_XFAIL) if name in _NEEDS_PER_RECORDING_INPUT else name
    for name in _CASES
])
def test_fit_predict_with_pad_channels_collate(name):
    torch.manual_seed(0)
    model, n_times, per_recording = _CASES[name]()
    n_chans = model.n_chans
    pos = per_recording is not None and per_recording[0] == "pos"
    windows = _windows([n_chans - 1, n_chans], n_times, pos=pos)
    seen = []
    model.register_forward_pre_hook(
        lambda m, args, kwargs: seen.append(kwargs["ch_mask"].sum(1).tolist()),
        with_kwargs=True,
    )
    net = _net(model).fit(windows, y=None)
    # 3 windows of each recording; the first batch pads the first recording's.
    assert seen == [[n_chans - 1] * 3 + [n_chans], [n_chans] * 2]
    mixed = net.predict_proba(windows)
    alone = [net.predict_proba(part) for part in windows.split([[0], [1]]).values()]
    np.testing.assert_allclose(mixed, np.concatenate(alone), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", list(_CASES))
def test_training_step_gradients_match_each_recording_alone(name):
    # In eval mode (no dropout), so both losses see the same network.
    model, x, mask, xs, batch_kwargs, alone_kwargs = _padded_case(name)
    params = [p for p in model.parameters() if p.requires_grad]
    y = torch.arange(len(xs)) % 3
    loss = F.cross_entropy(model(x, ch_mask=mask, **batch_kwargs), y)
    alone_loss = torch.stack([
        F.cross_entropy(model(xi[None], **kwargs), y[i : i + 1])
        for i, (xi, kwargs) in enumerate(zip(xs, alone_kwargs))
    ]).mean()
    grads = torch.autograd.grad(loss, params, materialize_grads=True)
    alone_grads = torch.autograd.grad(alone_loss, params, materialize_grads=True)
    for grad, alone_grad in zip(grads, alone_grads):
        torch.testing.assert_close(grad, alone_grad)


def test_every_ch_mask_model_has_a_case():
    takes_ch_mask = {
        name for name in braindecode.models.__all__
        if isinstance(cls := getattr(braindecode.models, name), type)
        and issubclass(cls, torch.nn.Module)
        and "ch_mask" in inspect.signature(cls.forward).parameters
    }
    covered = {case.split("-")[0] for case in (*_CASES, *_FIXED_HEADS)}
    assert takes_ch_mask and takes_ch_mask <= covered, takes_ch_mask - covered


def test_model_without_ch_mask_refuses_padded_batches():
    net = _net(ShallowFBCSPNet(n_chans=2, n_outputs=2, n_times=200))
    with pytest.raises(ValueError, match="ShallowFBCSPNet.forward takes no ch_mask"):
        net.fit(_windows([1, 2], 200), y=None)
