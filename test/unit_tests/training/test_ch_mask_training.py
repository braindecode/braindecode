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
from braindecode.models.biot import BIOT_CHANNEL_ORDER
from braindecode.models.labram import LABRAM_CHANNEL_ORDER
from braindecode.models.util import models_mandatory_parameters
from braindecode.preprocessing import create_fixed_length_windows
from test.unit_tests.models._fixtures import non_classification_models
from test.unit_tests.models.test_model_contract import (
    _ch_mask_case,
    _forward_or_refuse,
)


def _windows(channel_sets, n_times, n_windows=3, pos=False):
    rng = np.random.default_rng(0)
    montage = mne.channels.make_standard_montage("colin27_1005")
    recordings = []
    for target, channels in enumerate(channel_sets):
        names = montage.ch_names[:channels] if isinstance(channels, int) else channels
        info = mne.create_info(names, 100.0, "eeg")
        data = rng.standard_normal((len(names), n_windows * n_times))
        raw = mne.io.RawArray(data, info, verbose="error")
        if pos:  # LaBraM's names are not all in the montage
            raw.set_montage(montage)
        recordings.append(RawDataset(raw, description={"label": target}))
    windows = create_fixed_length_windows(
        BaseConcatDataset(recordings), window_size_samples=n_times,
        window_stride_samples=n_times, on_last_window="drop", preload=True,
    )
    windows.set_target("label")
    windows.set_return_ch_pos(pos)
    return windows


def _net(module, collate_fn=pad_channels_collate, **kwargs):
    return EEGClassifier(
        module, max_epochs=1, batch_size=4, train_split=None,
        iterator_train__shuffle=False, iterator_train__drop_last=False,
        iterator_train__collate_fn=collate_fn,
        iterator_valid__collate_fn=collate_fn, **kwargs,
    )


# Backbones that need metadata of each recording to run on fewer channels than
# they were built with; without it the input contract refuses the channel count.
_MISSING_METADATA = {"PopulationTransformer", "Labram", "MAPA"}


def _collate_native_metadata(batch):
    """Pad ``(window, metadata)`` pairs like ``pad_channels_collate`` and route the
    metadata of each recording to the forward: its ``(n_chans, 3)`` coordinates or
    sensor indices, padded like the signals. skorch turns every batch entry into a
    tensor, which cannot hold LaBraM's ``ch_names``: its signals fill the canonical
    channels instead, as the recordings are prefixes of them."""
    windows, metadata = zip(*batch)
    x, y, crop, mask = pad_channels_collate(windows)  # x: (batch, max_chans, n_times)
    (key,) = metadata[0]
    values = [entry[key] for entry in metadata]
    if key == "ch_names":
        assert all(names == LABRAM_CHANNEL_ORDER[: len(names)] for names in values)
        extra = len(LABRAM_CHANNEL_ORDER) - x.shape[1]
        x, mask = F.pad(x, (0, 0, 0, extra)), F.pad(mask, (0, extra))
        return {"x": x, "ch_mask": mask}, y, crop
    # (n_chans, 3) per recording -> (batch, max_chans, 3), zero-padded like x
    channels = pad_sequence([value.squeeze(0) for value in values], batch_first=True)
    return {"x": x, "ch_mask": mask, key: channels}, y, crop


@pytest.mark.parametrize(
    "model_name,required_params,signal_params",
    [
        pytest.param(*case, id=case[0])
        for case in models_mandatory_parameters
        if case[0] not in non_classification_models
    ],
)
def test_registered_model_ch_mask_training(model_name, required_params, signal_params):
    """One cross-entropy step on the padded batch of ``_ch_mask_case`` has the
    gradients of the per-recording losses, and a fit through
    :class:`braindecode.EEGClassifier` predicts on the mixed set what it predicts
    on each recording alone. A model that refuses ``ch_mask`` passes; one of
    ``_MISSING_METADATA`` does not refuse it, and raises on a recording without
    its metadata."""
    model, x, mask, xs, batch_kwargs, alone_kwargs = _ch_mask_case(
        model_name, required_params, signal_params
    )
    if model_name in _MISSING_METADATA:  # they take ch_mask: refusing is a failure
        logits = model(x, ch_mask=mask, **batch_kwargs)
    else:
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
    channel_sets = [n_chans - 1, n_chans]
    if model_name == "Labram":
        channel_sets = [LABRAM_CHANNEL_ORDER[:n] for n in channel_sets]
    windows = _windows(channel_sets, x.shape[2], pos=pos)
    parts = list(windows.split([[0], [1]]).values())
    net = _net(model)
    if model_name in _MISSING_METADATA:
        # Without its metadata the shorter recording is refused, even unpadded.
        with pytest.raises(ValueError, match=f"{model_name} was built for n_chans="):
            net.initialize().predict_proba(parts[0])
        # Each window carries the metadata of its recording.
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
    # 3 windows of each recording; the first batch pads the first recording's.
    assert seen == [[n_chans - 1] * 3 + [n_chans], [n_chans] * 2]
    mixed = net.predict_proba(windows)
    alone = [net.predict_proba(part) for part in parts]
    np.testing.assert_allclose(mixed, np.concatenate(alone), rtol=1e-5, atol=1e-6)


def test_channel_strategy_model_trains_on_padded_batches():
    # The channel layer maps each recording, from its positions, onto BIOT's
    # bipolar pretraining channels; the backbone never sees ch_mask.
    names = [
        ["F3", "C3", "F7", "P7", "O1", "C4", "P3", "F4"],
        ["O1", "T7", "P7", "C3", "F3", "F7", "F8", "P3", "O2", "P4", "C4", "F4"],
    ]
    windows = _windows(names, n_times=200, pos=True)
    net = _net(BIOT, module__n_outputs=2, module__n_times=200, module__sfreq=100,
               module__embed_dim=32, module__num_heads=2, module__num_layers=1,
               module__channel_strategy="spline")
    net.fit(windows, y=None)
    proba = net.predict_proba(windows)
    assert proba.shape == (len(windows), 2) and np.isfinite(proba).all()
    row = BIOT_CHANNEL_ORDER.index("F3-C3")
    for channels, recording in zip(names, windows.datasets):
        signal = torch.from_numpy(recording[0][0])[None]
        observed_signal, observed = net.module_.channel_layer(signal, recording.raw.info["chs"])
        assert observed[row]
        torch.testing.assert_close(
            observed_signal[:, row], signal[:, channels.index("F3")] - signal[:, channels.index("C3")]
        )
    alone = [net.predict_proba(part) for part in windows.split([[0], [1]]).values()]
    np.testing.assert_allclose(proba, np.concatenate(alone), rtol=1e-5, atol=1e-6)


def test_model_without_ch_mask_refuses_padded_batches():
    net = _net(ShallowFBCSPNet(n_chans=2, n_outputs=2, n_times=200))
    with pytest.raises(ValueError, match="ShallowFBCSPNet.forward takes no ch_mask"):
        net.fit(_windows([1, 2], 200), y=None)
