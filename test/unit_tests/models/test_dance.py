# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: MIT
from __future__ import annotations

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


def _model(**kw):
    defaults = dict(
        n_outputs=4, n_chans=19, chs_info=_chs_info(), n_times=6400,
        sfreq=200.0, input_window_seconds=32.0,
    )
    defaults.update(kw)
    return DANCE(**defaults)


def test_dance_init_builds():
    m = _model()
    assert isinstance(m, nn.Module)
    assert m.final_layer.out_features == 4
    assert m.final_layer.in_features == 128  # embed_dim


def test_dance_final_layer_is_last_child():
    m = _model()
    last_two = [name for name, _ in m.named_children()][-2:]
    assert "final_layer" in last_two


def test_dance_activation_default_is_class():
    import inspect

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
    # buffer is non-persistent -> not in state_dict
    assert "channel_positions" not in m.state_dict()
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
    out = m.eval()(torch.randn(2, 19, 6400))
    assert out.shape == (2, 256, 4)


def test_dance_forward_dense_shape():
    m = _model().eval()
    x = torch.randn(2, 19, 6400)
    out = m(x)
    assert out.shape == (2, 256, 4)  # (B, num_latents, n_outputs)


def test_dance_forward_length_agnostic():
    m = _model().eval()
    out_a = m(torch.randn(1, 19, 6400))
    out_b = m(torch.randn(1, 19, 9000))
    assert out_a.shape == out_b.shape == (1, 256, 4)


def test_dance_forward_batch_one_train_mode():
    m = _model().train()
    out = m(torch.randn(1, 19, 6400))  # must not crash at batch=1
    assert out.shape == (1, 256, 4)


def test_dance_forward_min_length_guard():
    m = _model().eval()
    with pytest.raises(ValueError, match="receptive field|shorter"):
        m(torch.randn(2, 19, 4))
