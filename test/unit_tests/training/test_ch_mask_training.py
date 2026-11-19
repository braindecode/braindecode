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
from sklearn.base import clone

from braindecode import EEGClassifier
from braindecode.datasets import BaseConcatDataset, RawDataset, pad_channels_collate
from braindecode.models import BBL, BIOT, ShallowFBCSPNet
from braindecode.models.biot import BIOT_CHANNEL_ORDER
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


def test_channel_strategy_model_trains_on_padded_batches(tmp_path):
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

    checkpoint = tmp_path / "biot.pt"
    net.save_params(f_params=checkpoint)
    restored = clone(net).initialize()
    restored.load_params(f_params=checkpoint)
    np.testing.assert_allclose(restored.predict_proba(windows), proba, rtol=1e-5, atol=1e-6)


def test_model_without_ch_mask_refuses_padded_batches():
    net = _net(ShallowFBCSPNet(n_chans=2, n_outputs=2, n_times=200))
    with pytest.raises(ValueError, match="ShallowFBCSPNet.forward takes no ch_mask"):
        net.fit(_windows([1, 2], 200), y=None)


def test_bbl_raw_time_mask_training_before_context_and_channel_padding():
    """A downstream clean-latent matching smoke, not an author pretext loss.
    Corruption happens before conv/context; padding and hidden inputs send no
    gradient, while the real backbone receives finite nonzero gradients."""
    torch.manual_seed(0)
    model = BBL(
        n_chans=3, n_outputs=2, n_times=39, n_sensors=5, n_datasets=2,
        n_subjects=4, model_dim=8, subject_dim=2, n_filters=2,
    ).train()
    clean = torch.randn(2, 3, 39)
    real = torch.tensor([[True, True, False], [True, True, True]])
    hide = torch.zeros(2, 39, dtype=torch.bool)
    hide[0, 8:19] = True
    hide[1, 20:33] = True
    targets = model(clean, ch_mask=real, return_features=True, return_temporal_features=True)["temporal_features"].detach().clone()
    selected = ~real[:, :, None] | hide[:, None, :]
    corrupt = clean.masked_fill(selected, float("nan")).requires_grad_(True)
    projector_inputs, encoder_inputs = [], []
    h1 = model.projector.register_forward_pre_hook(lambda m, a: projector_inputs.append(a[0].detach().clone()))
    h2 = model.encoder.register_forward_pre_hook(lambda m, a: encoder_inputs.append(a[0].detach().clone()))
    out = model(corrupt, ch_mask=real, temporal_mask=hide, return_features=True, return_temporal_features=True)
    h1.remove()
    h2.remove()
    expected_input = F.pad(clean.masked_fill(selected, 0.0), (0, 0, 0, 2))
    assert torch.equal(projector_inputs[0], expected_input)
    assert torch.isfinite(encoder_inputs[0]).all()
    assert torch.isnan(corrupt.detach()[selected.expand_as(corrupt)]).all()
    reference = model(clean.masked_fill(selected, 0.0), return_features=True, return_temporal_features=True)
    assert torch.equal(out["temporal_features"], reference["temporal_features"])
    # Padded values cannot affect the temporal representation.
    other_padding = corrupt.detach().masked_fill(~real[:, :, None], 12345.0)
    other = model(other_padding, ch_mask=real, temporal_mask=hide, return_features=True, return_temporal_features=True)
    assert torch.equal(out["temporal_features"], other["temporal_features"])
    loss = F.mse_loss(out["temporal_features"], targets)
    assert torch.isfinite(loss) and loss > 0
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    loss.backward()
    ignored = selected.expand_as(corrupt)
    assert torch.equal(corrupt.grad[ignored], torch.zeros_like(corrupt.grad[ignored]))
    assert torch.isfinite(corrupt.grad).all()
    backbone = [p for p in [*model.projector.parameters(), *model.encoder.parameters()] if p.grad is not None]
    assert backbone and all(torch.isfinite(p.grad).all() for p in backbone)
    assert any(p.grad.abs().sum() > 0 for p in backbone)
    before = model.projector[0].weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.projector[0].weight)
