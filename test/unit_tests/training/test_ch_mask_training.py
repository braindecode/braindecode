"""The ``ch_mask`` contract when a model is trained through braindecode.

Every registered model whose ``forward`` takes ``ch_mask`` is fit on recordings
with different channel counts, batched by
:func:`braindecode.datasets.pad_channels_collate`. Each recording must get the
output, and the gradients, it gets alone.
"""

import inspect

import mne
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from braindecode import EEGClassifier
from braindecode.datasets import BaseConcatDataset, RawDataset, pad_channels_collate
from braindecode.models import ShallowFBCSPNet
from braindecode.models.util import models_mandatory_parameters
from braindecode.preprocessing import create_fixed_length_windows
from test.unit_tests.models._fixtures import non_classification_models
from test.unit_tests.models.test_model_contract import (
    _ch_mask_case,
    _forward_or_refuse,
)


def _windows(n_chans_per_recording, n_times, n_windows=3, pos=False):
    rng = np.random.default_rng(0)
    montage = mne.channels.make_standard_montage("standard_1005")
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


# Known misses, not fixed here.
_FIT_XFAIL = {
    "PopulationTransformer": "the loader routes no per-recording coords, which "
    "PopulationTransformer needs on fewer channels than it was built with",
    "Labram": "the loader routes no ch_names, which Labram needs on fewer than "
    "its 128 channels",
    "MAPA": "the loader routes no per-sample sensor_indices, which MAPA needs with "
    "ch_mask",
}


@pytest.mark.parametrize(
    "model_name,required_params,signal_params",
    [
        pytest.param(
            *case,
            marks=[pytest.mark.xfail(reason=_FIT_XFAIL[case[0]], strict=True)]
            if case[0] in _FIT_XFAIL
            else [],
            id=case[0],
        )
        for case in models_mandatory_parameters
        if case[0] not in non_classification_models
    ],
)
def test_registered_model_ch_mask_training(model_name, required_params, signal_params):
    """One cross-entropy step on the padded batch of ``_ch_mask_case`` has the
    gradients of the per-recording losses, and a fit through
    :class:`braindecode.EEGClassifier` predicts on the mixed set what it predicts
    on each recording alone. A model that refuses ``ch_mask`` passes."""
    model, x, mask, xs, batch_kwargs, alone_kwargs = _ch_mask_case(
        model_name, required_params, signal_params
    )
    logits = _forward_or_refuse(model, x, mask, batch_kwargs)
    if logits is None:
        return
    # In eval mode (no dropout), so both losses see the same network.
    params = [p for p in model.parameters() if p.requires_grad]
    y = torch.arange(len(xs)) % logits.shape[1]
    alone_loss = torch.stack([
        F.cross_entropy(model(xi, **kwargs), y[i : i + 1])
        for i, (xi, kwargs) in enumerate(zip(xs, alone_kwargs))
    ]).mean()
    grads = torch.autograd.grad(F.cross_entropy(logits, y), params, materialize_grads=True)
    alone_grads = torch.autograd.grad(alone_loss, params, materialize_grads=True)
    for grad, alone_grad in zip(grads, alone_grads):
        torch.testing.assert_close(grad, alone_grad)

    n_chans = x.shape[1]
    pos = "pos" in inspect.signature(type(model).forward).parameters
    windows = _windows([n_chans - 1, n_chans], x.shape[2], pos=pos)
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


def test_model_without_ch_mask_refuses_padded_batches():
    net = _net(ShallowFBCSPNet(n_chans=2, n_outputs=2, n_times=200))
    with pytest.raises(ValueError, match="ShallowFBCSPNet.forward takes no ch_mask"):
        net.fit(_windows([1, 2], 200), y=None)
