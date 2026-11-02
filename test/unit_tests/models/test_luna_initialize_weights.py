"""LUNA.initialize_weights() was public in 1.8.1; 2.0 keeps it as a deprecated alias."""

import pytest
import torch

from braindecode.models import LUNA


def _luna():
    return LUNA(n_outputs=2, n_chans=4, n_times=400, sfreq=200.0, depth=2)


def test_initialize_weights_is_deprecated_alias_of_init_parameters():
    torch.manual_seed(0)
    model = _luna()
    torch.manual_seed(1)
    model._init_parameters()
    expected = {k: v.clone() for k, v in model.state_dict().items()}

    torch.manual_seed(0)
    model = _luna()
    torch.manual_seed(1)
    with pytest.warns(DeprecationWarning, match="reset_parameters"):
        model.initialize_weights()
    got = model.state_dict()
    assert expected.keys() == got.keys()
    for k in expected:
        torch.testing.assert_close(got[k], expected[k], rtol=0, atol=0, msg=k)


def test_initialize_weights_changes_weights():
    torch.manual_seed(0)
    model = _luna()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    torch.manual_seed(123)
    with pytest.warns(DeprecationWarning):
        model.initialize_weights()
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
