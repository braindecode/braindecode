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
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, StackDataset

from braindecode import EEGClassifier
from braindecode.datasets import BaseConcatDataset, RawDataset, pad_channels_collate
from braindecode.models import BIOT, ShallowFBCSPNet
from braindecode.models.labram import LABRAM_CHANNEL_ORDER
from braindecode.models.util import models_dict, models_mandatory_parameters
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.util import resolve_montage_name
from test.unit_tests.models._fixtures import non_classification_models
from test.unit_tests.models.test_model_contract import (
    _ch_mask_case,
    _forward_or_refuse,
)

pytestmark = pytest.mark.integration


def _windows(n_chans_per_recording, n_times, n_windows=3, pos=False, names=None):
    rng = np.random.default_rng(0)
    montage = mne.channels.make_standard_montage(resolve_montage_name("standard_1005"))
    recordings = []
    for target, n_chans in enumerate(n_chans_per_recording):
        ch_names = montage.ch_names if names is None else names
        info = mne.create_info(ch_names[:n_chans], 100.0, "eeg")
        data = rng.standard_normal((n_chans, n_windows * n_times))
        raw = mne.io.RawArray(data, info, verbose="error")
        if pos:
            raw.set_montage(montage)
        recordings.append(RawDataset(raw, description={"label": target}))
    windows = create_fixed_length_windows(
        BaseConcatDataset(recordings),
        window_size_samples=n_times,
        window_stride_samples=n_times,
        on_last_window="drop",
        preload=True,
    )
    windows.set_target("label")
    windows.set_return_ch_pos(pos)
    return windows


def _net(module, collate_fn=pad_channels_collate, **kwargs):
    return EEGClassifier(
        module,
        max_epochs=1,
        batch_size=4,
        train_split=None,
        iterator_train__shuffle=False,
        iterator_train__drop_last=False,
        iterator_train__collate_fn=collate_fn,
        iterator_valid__collate_fn=collate_fn,
        **kwargs,
    )


# Each of these backbones requires a different kind of recording metadata.
_MISSING_METADATA = {
    "PopulationTransformer": "pass `coords`",
    "Labram": "ch_names is None",
    "MAPA": "ch_mask needs per-sample sensor_indices",
}


def _collate_native_metadata(batch):
    windows, metadata = zip(*batch)
    x, y, crop, mask = pad_channels_collate(windows)
    (key,) = metadata[0]
    values = [entry[key] for entry in metadata]
    if key == "ch_names":
        # LaBraM's default input uses its canonical channel order. Here both
        # recordings are known prefixes, so only the trailing slots need padding.
        assert all(names == LABRAM_CHANNEL_ORDER[: len(names)] for names in values)
        extra = len(LABRAM_CHANNEL_ORDER) - x.shape[1]
        return (
            {"x": F.pad(x, (0, 0, 0, extra)), "ch_mask": F.pad(mask, (0, extra))},
            y,
            crop,
        )
    # Coordinate bins and contact indices are integer triplets per electrode.
    channels = pad_sequence([value.squeeze(0) for value in values], batch_first=True)
    return {"x": x, "ch_mask": mask, key: channels}, y, crop


@pytest.mark.parametrize(
    "model_name,required_params,signal_params",
    [
        pytest.param(*case, id=case[0])
        for case in models_mandatory_parameters
        if case[0] not in non_classification_models
        and "ch_mask" in inspect.signature(models_dict[case[0]].forward).parameters
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
    if model_name in _MISSING_METADATA:
        logits = model(x, ch_mask=mask, **batch_kwargs)
    else:
        logits = _forward_or_refuse(model, x, mask, batch_kwargs)
        if logits is None:
            return
    # In eval mode (no dropout), so both losses see the same network.
    params = [p for p in model.parameters() if p.requires_grad]
    y = torch.arange(len(xs)) % logits.shape[1]
    alone_loss = torch.stack(
        [
            F.cross_entropy(model(xi, **kwargs), y[i : i + 1])
            for i, (xi, kwargs) in enumerate(zip(xs, alone_kwargs))
        ]
    ).mean()
    grads = torch.autograd.grad(
        F.cross_entropy(logits, y), params, materialize_grads=True
    )
    alone_grads = torch.autograd.grad(alone_loss, params, materialize_grads=True)
    for grad, alone_grad in zip(grads, alone_grads):
        torch.testing.assert_close(grad, alone_grad)

    if model_name in _MISSING_METADATA:
        weights = min(
            (p for p, g in zip(params, grads) if g.count_nonzero()), key=torch.numel
        )
        before_fit = weights.detach().clone()

    n_chans = x.shape[1]
    pos = "pos" in inspect.signature(type(model).forward).parameters
    windows = _windows(
        [n_chans - 1, n_chans],
        x.shape[2],
        pos=pos,
        names=LABRAM_CHANNEL_ORDER if model_name == "Labram" else None,
    )
    parts = list(windows.split([[0], [1]]).values())
    net = _net(model)
    if model_name in _MISSING_METADATA:
        # The shortest recording needs its own metadata even when unpadded.
        with pytest.raises(ValueError, match=_MISSING_METADATA[model_name]):
            net.initialize().predict_proba(parts[0])
        # Preserve the attachment between each window and its recording.
        parts = [
            StackDataset(part, [metadata] * len(part))
            for part, metadata in zip(parts, alone_kwargs)
        ]
        windows = ConcatDataset(parts)
        net = _net(model, collate_fn=_collate_native_metadata)

    seen = []
    model.register_forward_pre_hook(
        lambda m, args, kwargs: seen.append(kwargs["ch_mask"].sum(1).tolist()),
        with_kwargs=True,
    )
    net.fit(windows, y=None)
    if model_name in _MISSING_METADATA:
        assert not torch.equal(weights, before_fit)
    # 3 windows of each recording; the first batch pads the first recording's.
    assert seen == [[n_chans - 1] * 3 + [n_chans], [n_chans] * 2]
    mixed = net.predict_proba(windows)
    alone = [net.predict_proba(part) for part in parts]
    np.testing.assert_allclose(mixed, np.concatenate(alone), rtol=1e-5, atol=1e-6)


def test_channel_strategy_model_trains_on_padded_batches():
    # The channel layer maps each recording, from its positions, onto BIOT's
    # bipolar pretraining channels; the backbone never sees ch_mask.
    windows = _windows([8, 12], n_times=200, pos=True)  # spline: a fitted head
    net = _net(
        BIOT,
        module__n_outputs=2,
        module__n_times=200,
        module__sfreq=100,
        module__embed_dim=32,
        module__num_heads=2,
        module__num_layers=1,
        module__channel_strategy="spline",
    )
    net.fit(windows, y=None)
    proba = net.predict_proba(windows)
    assert proba.shape == (len(windows), 2) and np.isfinite(proba).all()


def test_model_without_ch_mask_refuses_padded_batches():
    net = _net(ShallowFBCSPNet(n_chans=2, n_outputs=2, n_times=200))
    with pytest.raises(ValueError, match="ShallowFBCSPNet.forward takes no ch_mask"):
        net.fit(_windows([1, 2], 200), y=None)
