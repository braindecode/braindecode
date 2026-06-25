# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: MIT
from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from torch import nn

from braindecode.models import DANCE

_RNG = np.random.default_rng(0)


def _chs_info(n=19, with_loc=True):
    return [
        {
            "ch_name": f"E{i + 1}",
            "kind": "eeg",
            "loc": (_RNG.random(12) if with_loc else np.zeros(12)),
        }
        for i in range(n)
    ]


_DEFAULTS = dict(
    n_outputs=4, n_chans=19, n_times=6400, sfreq=200.0, input_window_seconds=32.0
)


def _model(**kw):
    return DANCE(**{**_DEFAULTS, "chs_info": _chs_info(), **kw})


def test_dance_init_builds():
    m = _model()
    assert isinstance(m, nn.Module)
    assert m.final_layer.out_features == 4
    assert m.final_layer.in_features == 128  # embed_dim


def test_dance_final_layer_is_last_child():
    assert "final_layer" in [n for n, _ in _model().named_children()][-2:]


def test_dance_activation_default_is_class():
    sig = inspect.signature(DANCE.__init__)
    assert sig.parameters["activation"].default is nn.GELU
    assert sig.parameters["drop_prob"].default == 0.1


def test_dance_positions_buffer_registered():
    m = _model()
    assert m.use_channel_merger is True
    assert m.channel_positions.shape == (19, 2)
    # normalized to [0, 1] per-axis
    assert m.channel_positions.min() >= 0.0
    assert m.channel_positions.max() <= 1.0 + 1e-6
    assert "channel_positions" not in m.state_dict()  # non-persistent buffer
    # merger is NESTED inside the conv, not a top-level attribute
    assert not hasattr(m, "channel_merger")
    assert m.conv.merger is not None
    assert m.conv.merger.heads.shape == (270, 2048)


def test_dance_fallback_when_no_locations():
    with pytest.warns(UserWarning, match="ChannelMerger"):
        m = _model(chs_info=_chs_info(with_loc=False))
    assert m.use_channel_merger is False
    assert m.conv.merger is None


def test_dance_no_merger_consumes_raw_channels():
    m = _model(use_channel_merger=False)
    assert m.conv.merger is None
    assert not hasattr(m, "channel_positions")
    assert m.eval()(torch.randn(2, 19, 6400)).shape == (2, 256, 4)


@pytest.mark.parametrize(
    "shape",
    [(2, 19, 6400), (1, 19, 6400), (1, 19, 9000)],
    ids=["dense", "len6400", "len9000-agnostic"],
)
def test_dance_forward_shape(shape):
    # (B, num_latents, n_outputs); output length is agnostic to input n_times.
    out = _model().eval()(torch.randn(*shape))
    assert out.shape == (shape[0], 256, 4)


def test_dance_forward_batch_one_train_mode():
    # BN at batch=1 must not crash in train mode.
    assert _model().train()(torch.randn(1, 19, 6400)).shape == (1, 256, 4)


def test_dance_forward_min_length_guard():
    with pytest.raises(ValueError, match="receptive field|shorter"):
        _model().eval()(torch.randn(2, 19, 4))


def test_dance_detect_dict_shapes():
    out = _model().eval().detect(torch.randn(2, 19, 6400))
    assert out["class"].shape == (2, 100, 4)
    assert out["start"].shape == out["end"].shape == (2, 100)
    assert out["dense"].shape == (2, 256, 4)


def test_dance_reset_head_rebuilds_both_heads():
    m = _model()
    m.reset_head(7)
    assert m.final_layer.out_features == 7
    assert m.decoder.class_head.out_features == 7
    assert m.n_outputs == 7
    out = m.eval().detect(torch.randn(1, 19, 6400))
    assert out["class"].shape == (1, 100, 7)
    assert out["dense"].shape == (1, 256, 7)
