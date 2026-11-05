"""``ch_mask`` contract for every registered model whose ``forward`` takes it.

A model opts in by accepting ``ch_mask``; no test needs an edit. Built as in
``test_model_contract.py`` (C channels), it gets two recordings with C - 1 and C
real channels, NaN-padded to C. Each recording must get the outputs and
gradients it gets alone, or ``forward`` must refuse ``ch_mask`` with a
``ValueError`` that names it (a head tied to the channels).
"""

import inspect

import mne
import numpy as np
import pytest
import torch

from braindecode import EEGClassifier
from braindecode.datasets import BaseConcatDataset, RawDataset, pad_channels_collate
from braindecode.models import ShallowFBCSPNet
from braindecode.models.labram import LABRAM_CHANNEL_ORDER
from braindecode.models.util import models_dict, models_mandatory_parameters
from braindecode.preprocessing import create_fixed_length_windows
from test.unit_tests.models._fixtures import (
    _get_signal_params,
    non_classification_models,
)
from test.unit_tests.models.test_model_contract import (
    _batched_tensor_leaves,
    _build_case,
    _materialize,
)


def _positions(i, n_chans):
    return 0.1 * torch.rand((n_chans, 3), generator=torch.Generator().manual_seed(i))


# The only per-model data: inputs a model needs to run alone on fewer channels,
# {model: {forward argument: build(i, n_chans) -> one row per channel}}; a
# non-tensor (channel names) is one value for the whole batch.
_CH_MASK_EXTRAS = {
    "Labram": {"ch_names": lambda i, n_chans: LABRAM_CHANNEL_ORDER[:n_chans]},
    "PopulationTransformer": {
        "coords": lambda i, n_chans: torch.randint(
            0, 200, (n_chans, 3), generator=torch.Generator().manual_seed(i)
        )
    },
    "LUNA": {"channel_locations": _positions},
    "REVE": {"pos": _positions},
}
# Known breaks when trained through EEGClassifier, which routes ch_mask and pos.
_CH_MASK_FIT_XFAIL = {
    "Labram": "the loader routes no ch_names",
    "PopulationTransformer": "the loader routes no per-recording coords",
}


def _takes(model_name, argument):
    forward = models_dict[model_name].forward
    return argument in inspect.signature(forward).parameters


_CH_MASK_CASES = [c for c in models_mandatory_parameters if _takes(c[0], "ch_mask")]


def _padded(model_name, x):
    """The recordings alone, NaN-padded into one batch, the mask and the extras."""
    torch.manual_seed(0)
    n_chans, n_times = x.shape[1:]
    n_real = [n_chans - 1, n_chans]
    xs = [torch.randn(n, n_times) for n in n_real]
    mask = torch.arange(n_chans) < torch.tensor(n_real)[:, None]
    padded = torch.full((2, n_chans, n_times), float("nan"))
    for i, xi in enumerate(xs):
        padded[i, : len(xi)] = xi
    batch, alone = {}, [{}, {}]
    for name, build in _CH_MASK_EXTRAS.get(model_name, {}).items():
        values = [build(i, n) for i, n in enumerate(n_real)]
        if not torch.is_tensor(values[0]):
            batch[name] = values[1]
            for kwargs, value in zip(alone, values):
                kwargs[name] = value
            continue
        batch[name] = torch.zeros((2, n_chans) + values[0].shape[1:], dtype=values[0].dtype)
        for i, value in enumerate(values):
            batch[name][i, : len(value)] = value
            alone[i][name] = value[None]
    return xs, padded, mask, batch, alone


@pytest.mark.parametrize(
    "model_name,required_params,signal_params",
    _CH_MASK_CASES,
    ids=[case[0] for case in _CH_MASK_CASES],
)
def test_registered_model_ch_mask_contract(model_name, required_params, signal_params):
    """Each recording of a padded batch gets its outputs alone; padding gets no
    gradient and its tokens are zero. Or forward refuses ``ch_mask``."""
    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    xs, padded, mask, batch, alone = _padded(model_name, x)
    padded.requires_grad_(True)
    try:
        out = model(padded, ch_mask=mask, **batch)
    except ValueError as error:
        if "ch_mask" not in str(error):
            raise
        return  # a head tied to the construction-time channels
    leaves = _batched_tensor_leaves(out, 2)
    sum(leaf.float().sum() for leaf in leaves if leaf.requires_grad).backward()
    assert torch.equal(padded.grad[~mask], torch.zeros_like(padded.grad[~mask]))
    with torch.no_grad():
        for i, (xi, kwargs) in enumerate(zip(xs, alone)):
            expected = _batched_tensor_leaves(model(xi[None], **kwargs), 1)
            assert len(expected) == len(leaves)
            for leaf, alone_leaf in zip(leaves, expected):
                torch.testing.assert_close(leaf[i : i + 1].detach(), alone_leaf)
        if _takes(model_name, "return_features"):
            features = model(padded, ch_mask=mask, return_features=True, **batch)
            tokens = features.get("tokens") if isinstance(features, dict) else None
            if torch.is_tensor(tokens) and tokens.shape[:2] == mask.shape:
                assert torch.equal(tokens[~mask], torch.zeros_like(tokens[~mask]))


def _windows(chs_info, n_times, sfreq, n_real, pos=False):
    """One recording per channel count, two windows each, labelled 0, 1, ..."""
    rng = np.random.default_rng(0)
    locs = _positions(0, max(n_real)).numpy()
    recordings = []
    for label, n_chans in enumerate(n_real):
        info = mne.create_info([ch["ch_name"] for ch in chs_info[:n_chans]], sfreq, "eeg")
        for ch, loc in zip(info["chs"], locs):
            ch["loc"][:3] = loc
        data = rng.standard_normal((n_chans, 2 * n_times))
        raw = mne.io.RawArray(data, info, verbose="error")
        recordings.append(RawDataset(raw, description={"label": label}))
    windows = create_fixed_length_windows(
        BaseConcatDataset(recordings), window_size_samples=n_times,
        window_stride_samples=n_times, on_last_window="drop", preload=True,
    )
    windows.set_target("label")
    windows.set_return_ch_pos(pos)
    return windows


def _net(module):
    return EEGClassifier(
        module, max_epochs=1, batch_size=4, train_split=None,
        iterator_train__shuffle=False, iterator_train__drop_last=False,
        iterator_train__collate_fn=pad_channels_collate,
        iterator_valid__collate_fn=pad_channels_collate,
    )


@pytest.mark.parametrize(
    "model_name,required_params,signal_params",
    [
        pytest.param(
            *case,
            marks=[pytest.mark.xfail(
                reason=_CH_MASK_FIT_XFAIL[case[0]], strict=True, raises=ValueError
            )] if case[0] in _CH_MASK_FIT_XFAIL else [],
            id=case[0],
        )
        for case in _CH_MASK_CASES
        if case[0] not in non_classification_models
    ],
)
def test_registered_model_ch_mask_training(model_name, required_params, signal_params):
    """Fit through EEGClassifier with pad_channels_collate: each recording keeps
    the predictions and the loss gradient it gets alone."""
    torch.manual_seed(0)
    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    n_chans, n_times = x.shape[1:]
    signal = _get_signal_params(signal_params)
    windows = _windows(
        signal["chs_info"], n_times, signal["sfreq"], [n_chans - 1, n_chans],
        pos=_takes(model_name, "pos"),
    )
    seen = []
    model.register_forward_pre_hook(
        lambda m, args, kwargs: seen.append(kwargs["ch_mask"].sum(1).tolist()),
        with_kwargs=True,
    )
    net = _net(model)
    try:
        net.fit(windows, y=None)
    except ValueError as error:
        if "ch_mask" not in str(error):
            raise
        return  # refused, as in test_registered_model_ch_mask_contract
    assert seen[0] == [n_chans - 1] * 2 + [n_chans] * 2
    parts = list(windows.split([[0], [1]]).values())
    np.testing.assert_allclose(
        net.predict_proba(windows),
        np.concatenate([net.predict_proba(part) for part in parts]),
        rtol=1e-5, atol=1e-6,
    )
    # One step's gradient: the mixed batch, and each recording's batch alone.
    net.module_.eval()  # no dropout, so every batch sees the same network
    params = [p for p in net.module_.parameters() if p.requires_grad]

    def gradients(dataset):
        (X, y), = list(net.get_iterator(dataset, training=False))
        loss = net.get_loss(net.infer(X), y, X=X, training=False)
        return torch.autograd.grad(loss, params, materialize_grads=True)

    first, second = (gradients(part) for part in parts)
    for grad, grad_a, grad_b in zip(gradients(windows), first, second):
        torch.testing.assert_close(grad, (grad_a + grad_b) / 2)


def test_model_without_ch_mask_refuses_padded_batches():
    signal = _get_signal_params(None)
    net = _net(ShallowFBCSPNet(n_chans=3, n_outputs=2, n_times=200))
    windows = _windows(signal["chs_info"], 200, signal["sfreq"], [2, 3])
    with pytest.raises(ValueError, match="ShallowFBCSPNet.forward takes no ch_mask"):
        net.fit(windows, y=None)
