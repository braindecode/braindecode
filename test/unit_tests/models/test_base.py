# Authors: Pierre Guetschel
#
# License: BSD-3


import json
import subprocess
import sys
from operator import attrgetter
from unittest.mock import patch

import pytest
import torch
from torch import nn

from braindecode.models.base import EEGModuleMixin


class DummyModule(EEGModuleMixin, nn.Sequential):
    """Dummy module for testing EEGModuleMixin"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DummyModuleNTime(EEGModuleMixin, nn.Sequential):
    """Dummy module using one of the properties of EEGModuleMixin
    in its __init__"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_module("dummy", nn.Linear(self.n_times, 1))


class DummyModuleConfigRoundTrip(EEGModuleMixin, nn.Sequential):
    """Dummy module exercising config round-trips."""

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        drop_prob=0.5,
        activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        self.activation = activation
        self.add_module("drop", nn.Dropout(drop_prob))
        self.add_module("activation_module", activation())
        self.add_module("linear", nn.Linear(self.n_times, self.n_outputs))


class DummyModuleWithOverriddenSfreq(DummyModule):
    """Extension model with a custom inherited signal property."""

    @property
    def sfreq(self):
        return 321.0


class DummyModuleWithInheritedOverride(DummyModuleWithOverriddenSfreq):
    """Second-level extension that must retain the custom property."""


class DummyModuleWithExtraIgnoredProperty(DummyModule):
    """Extension model with its own runtime-only property."""

    __jit_ignored_attributes__ = [
        *DummyModule.__jit_ignored_attributes__,
        "runtime_only",
    ]

    @property
    def runtime_only(self):
        raise ValueError("runtime_only must not be inspected while scripting")


class DummyModuleWithInheritedExtraIgnoredProperty(DummyModuleWithExtraIgnoredProperty):
    """Leaf extension inheriting the custom ignored property."""


def test_init_subclass_preserves_inherited_signal_property_override():
    assert (
        DummyModuleWithInheritedOverride.__dict__["sfreq"]
        is DummyModuleWithOverriddenSfreq.__dict__["sfreq"]
    )
    descriptor = DummyModuleWithInheritedOverride.__dict__["sfreq"]
    assert descriptor.__get__(object(), DummyModuleWithInheritedOverride) == 321.0


def test_init_subclass_propagates_extended_jit_ignored_property():
    module = DummyModuleWithInheritedExtraIgnoredProperty(
        n_outputs=1,
        n_chans=1,
        n_times=4,
    ).eval()
    input_tensor = torch.randn(2, 1, 4)

    scripted = torch.jit.script(module)

    assert (
        DummyModuleWithInheritedExtraIgnoredProperty.__dict__["runtime_only"]
        is DummyModuleWithExtraIgnoredProperty.__dict__["runtime_only"]
    )
    torch.testing.assert_close(scripted(input_tensor), module(input_tensor))


def test_init_subclass_with_license_without_hf_hub():
    """A model license can be declared without the optional Hub dependency."""
    code = """
import sys

sys.modules["huggingface_hub"] = None

from braindecode.models import base
from torch import nn

assert not base.HAS_HF_HUB

class LicensedTestModel(
    base.EEGModuleMixin,
    nn.Module,
    license="cc-by-nc-sa-4.0",
):
    pass

assert issubclass(LicensedTestModel, base.EEGModuleMixin)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.fixture(scope="function")
def dummy_module():
    return DummyModule(
        n_outputs=1,
        n_chans=1,
        chs_info=[{"ch_name": "ch1"}],
        n_times=200,
        input_window_seconds=2.0,
        sfreq=100.0,
    )


@pytest.mark.parametrize(
    "n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq",
    [
        (None, 1, [{"ch_name": "ch1"}], 200, 2.0, 100.0),
        (1, None, None, 200, 2.0, 100.0),
        (1, 1, None, 200, 2.0, 100.0),
        (1, 1, [{"ch_name": "ch1"}], None, None, None),
        (1, 1, [{"ch_name": "ch1"}], None, None, 100.0),
        (1, 1, [{"ch_name": "ch1"}], None, 2.0, None),
        (1, 1, [{"ch_name": "ch1"}], 200, None, None),
    ],
)
def test_missing_params(
    n_outputs,
    n_chans,
    chs_info,
    n_times,
    input_window_seconds,
    sfreq,
):
    module = DummyModule(
        n_outputs=n_outputs,
        n_chans=n_chans,
        chs_info=chs_info,
        n_times=n_times,
        input_window_seconds=input_window_seconds,
        sfreq=sfreq,
    )
    with pytest.raises(ValueError):
        assert module.n_outputs == 1
        assert module.n_chans == 1
        assert module.chs_info == [{"ch_name": "ch1"}]
        assert module.n_times == 200
        assert module.input_window_seconds == 2.0
        assert module.sfreq == 100.0


@pytest.mark.parametrize(
    "n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq",
    [
        (1, 1, [{"ch_name": "ch1"}], 200, 2.0, 100.0),
        (1, None, [{"ch_name": "ch1"}], 200, 2.0, 100.0),
        (1, None, [{"ch_name": "ch1"}], None, 2.0, 100.0),
        (1, None, [{"ch_name": "ch1"}], 200, None, 100.0),
        (1, None, [{"ch_name": "ch1"}], 200, 2.0, None),
    ],
)
def test_all_params(
    n_outputs,
    n_chans,
    chs_info,
    n_times,
    input_window_seconds,
    sfreq,
):
    module = DummyModule(
        n_outputs=n_outputs,
        n_chans=n_chans,
        chs_info=chs_info,
        n_times=n_times,
        input_window_seconds=input_window_seconds,
        sfreq=sfreq,
    )
    assert module.n_outputs == 1
    assert module.n_chans == 1
    assert module.chs_info == [{"ch_name": "ch1"}]
    assert module.n_times == 200
    assert module.input_window_seconds == 2.0
    assert module.sfreq == 100.0


@pytest.mark.parametrize(
    "n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq",
    [
        (1, 2, [{"ch_name": "ch1"}], 200, 2.0, 100.0),
        (1, 1, [{"ch_name": "ch1"}], 200, 3.0, 100.0),
    ],
)
def test_incorrect_params(
    n_outputs,
    n_chans,
    chs_info,
    n_times,
    input_window_seconds,
    sfreq,
):
    with pytest.raises(ValueError):
        _ = DummyModule(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )


def test_inexistent_param():
    with pytest.raises(TypeError):
        _ = DummyModule(
            inexistant_param=1,
        )


@pytest.mark.parametrize(
    "n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq",
    [
        (1, 1, [{"ch_name": "ch1"}], 200, 2.0, 100.0),
        (1, 1, [{"ch_name": "ch1"}], 200, 2.0, None),
        (1, 1, [{"ch_name": "ch1"}], 200, None, 100.0),
        (1, 1, [{"ch_name": "ch1"}], None, 2.0, 100.0),
    ],
)
def test_init_submodule(
    n_outputs,
    n_chans,
    chs_info,
    n_times,
    input_window_seconds,
    sfreq,
):
    _ = DummyModuleNTime(
        n_outputs=n_outputs,
        n_chans=n_chans,
        chs_info=chs_info,
        n_times=n_times,
        input_window_seconds=input_window_seconds,
        sfreq=sfreq,
    )


def test_get_torchinfo_statistics():
    n_chans = 1
    n_times = 200
    model = DummyModule(
        n_outputs=1,
        n_chans=n_chans,
        chs_info=[{"ch_name": "ch1"}],
        n_times=n_times,
        input_window_seconds=2.0,
        sfreq=100.0,
    )
    with patch("braindecode.models.base.ModelStatistics") as patch_stats:
        with patch(
            "braindecode.models.base.summary", return_value=patch_stats
        ) as patch_summary:
            result = model.get_torchinfo_statistics()
    patch_summary.assert_called_once_with(
        model,
        input_size=(1, n_chans, n_times),
        col_names=(
            "input_size",
            "output_size",
            "num_params",
            "kernel_size",
        ),
        row_settings=("var_names", "depth"),
        verbose=0,
    )
    assert result == patch_stats


def test__str__():
    n_chans = 1
    n_times = 200
    model = DummyModule(
        n_outputs=1,
        n_chans=n_chans,
        chs_info=[{"ch_name": "ch1"}],
        n_times=n_times,
        input_window_seconds=2.0,
        sfreq=100.0,
    )
    with patch("braindecode.models.base.ModelStatistics") as patch_stats:
        with patch.object(
            model, "get_torchinfo_statistics", return_value=patch_stats
        ) as patch_method_stats:
            result = str(model)

    patch_method_stats.assert_called_once()
    patch_stats.__str__.assert_called_once()
    assert result == str(patch_stats)


def test_get_output_shape():
    n_outputs = 1
    n_chans = 1
    chs_info = ([{"ch_name": "ch1"}],)
    n_times = 200
    input_window_seconds = 2.0
    sfreq = 100.0

    dummy_module = DummyModuleNTime(
        n_outputs=n_outputs,
        n_chans=n_chans,
        chs_info=chs_info,
        n_times=n_times,
        input_window_seconds=input_window_seconds,
        sfreq=sfreq,
    )
    assert dummy_module.get_output_shape() == (1, 1, 1)

    dummy_module.add_module("linear2", nn.Linear(1, 2))
    assert dummy_module.get_output_shape() == (1, 1, 2)


def test_raised_runtimeerror_kernel_size_get_output_shape(dummy_module: DummyModule):
    dummy_module.add_module("too_big_conv", nn.Conv2d(1, 1, kernel_size=(1, 201)))
    err_msg = (
        r"During model prediction RuntimeError was thrown showing that at some "
        r"layer ` Kernel size can't be greater than actual input size` \(see above "
        r"in the stacktrace\). This could be caused by providing too small "
        r"`n_times`\/`input_window_seconds`. Model may require longer chunks of signal "
        r"in the input than \(1, 1, 200\)."
    )
    with pytest.raises(ValueError, match=err_msg):
        dummy_module.get_output_shape()


@pytest.fixture(scope="function")
def config_roundtrip_without_hub_config():
    model = DummyModuleConfigRoundTrip(
        n_outputs=2,
        n_chans=3,
        chs_info=[{"ch_name": f"ch{i}"} for i in range(3)],
        n_times=32,
        input_window_seconds=0.32,
        sfreq=100.0,
        drop_prob=0.25,
        activation=nn.ELU,
    )
    model._hub_mixin_config = None

    config = model.get_config()

    restored = DummyModuleConfigRoundTrip.from_config(json.loads(json.dumps(config)))
    return model, config, restored


@pytest.mark.parametrize(
    "config_key, value_getter, use_identity",
    [
        pytest.param("drop_prob", attrgetter("drop.p"), False, id="drop-prob"),
        pytest.param("activation", attrgetter("activation"), True, id="activation"),
        pytest.param("n_outputs", attrgetter("n_outputs"), False, id="n-outputs"),
        pytest.param("n_chans", attrgetter("n_chans"), False, id="n-chans"),
        pytest.param("n_times", attrgetter("n_times"), False, id="n-times"),
    ],
)
def test_get_config_roundtrip_without_hub_config(
    config_roundtrip_without_hub_config, config_key, value_getter, use_identity
):
    model, config, restored = config_roundtrip_without_hub_config
    expected = value_getter(model)
    expected_config = (
        f"{expected.__module__}.{expected.__qualname__}"
        if isinstance(expected, type)
        else expected
    )

    assert config[config_key] == expected_config

    restored_value = value_getter(restored)
    if use_identity:
        assert restored_value is expected
    else:
        assert restored_value == expected


def test_raised_runtimeerror_output_size_get_output_shape(dummy_module: DummyModule):
    dummy_module.add_module("good_conv", nn.Conv2d(1, 1, kernel_size=(1, 100)))
    dummy_module.add_module("too_big_pool", nn.AvgPool2d(kernel_size=(1, 200)))

    err_msg = (
        r"During model prediction RuntimeError was thrown showing that at some "
        r"layer ` Output size is too small` \(see above "
        r"in the stacktrace\). This could be caused by providing too small "
        r"`n_times`\/`input_window_seconds`. Model may require longer chunks of signal "
        r"in the input than \(1, 1, 200\)."
    )
    with pytest.raises(ValueError, match=err_msg):
        dummy_module.get_output_shape()


@pytest.mark.parametrize(
    "n_times, input_window_seconds, sfreq",
    [
        (1001, 4.004, 250.0),  # Issue example: 4.004 * 250.0 = 1001.0
        (751, 3.004, 250.0),  # 3.004 * 250.0 = 751.0
        (501, 2.004, 250.0),  # 2.004 * 250.0 = 501.0
        (101, 0.404, 250.0),  # 0.404 * 250.0 = 101.0
    ],
)
def test_fractional_input_window_seconds_consistency(
    n_times, input_window_seconds, sfreq
):
    """Test that fractional input_window_seconds values are accepted when consistent.

    This test validates the fix for the bug where int() truncation rejected
    valid configurations. With round(), these values should be accepted.
    """
    # Should not raise ValueError
    module = DummyModule(
        n_outputs=1,
        n_chans=1,
        n_times=n_times,
        input_window_seconds=input_window_seconds,
        sfreq=sfreq,
    )
    assert module.n_times == n_times
    assert module.input_window_seconds == input_window_seconds
    assert module.sfreq == sfreq


@pytest.mark.parametrize(
    "n_times, input_window_seconds, sfreq",
    [
        (1001, None, 250.0),  # Infer input_window_seconds
        (751, None, 250.0),  # Infer input_window_seconds
        (None, 4.004, 250.0),  # Infer n_times
        (None, 3.004, 250.0),  # Infer n_times
    ],
)
def test_fractional_input_window_seconds_inference(
    n_times, input_window_seconds, sfreq
):
    """Test that fractional input_window_seconds can be inferred correctly.

    This test validates that inference uses round() instead of int().
    """
    module = DummyModule(
        n_outputs=1,
        n_chans=1,
        n_times=n_times,
        input_window_seconds=input_window_seconds,
        sfreq=sfreq,
    )
    # Verify the inferred values are correct
    if n_times is None:
        assert module.n_times == round(input_window_seconds * sfreq)
    if input_window_seconds is None:
        assert module.input_window_seconds == n_times / sfreq


class _DummyModelWithParams(EEGModuleMixin, nn.Sequential):
    """A model with its own documented parameters.

    Parameters
    ----------
    hidden_size : int
        Size of hidden layer.
    drop_prob : float
        Dropout probability.
    """

    def __init__(
        self,
        n_chans=None,
        n_outputs=None,
        n_times=None,
        hidden_size=64,
        drop_prob=0.5,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob


def test_docstring_inheritance_preserves_child_description():
    """Regression test: child class description must not be replaced by parent's.

    When NumpyDocstringInheritanceInitMeta is active the child class
    docstring should keep its own description and Parameters section
    while inheriting missing sections (Raises, Notes) from the parent.

    A previous bug caused ``@wraps`` in ``track_model_init_kwargs`` to
    run before the metaclass, making ``inspect.unwrap()`` bypass the
    wrapper and read ``__doc__ = None`` from the original function.
    This caused the parent description to overwrite the child's.
    """
    import os

    doc = _DummyModelWithParams.__doc__
    assert doc is not None, "Class docstring should not be None"

    # Child description must always be present, regardless of env var
    assert "A model with its own documented parameters" in doc, (
        f"Child description was replaced by parent's. Got:\n{doc[:200]}"
    )

    is_enabled = os.environ.get("DOCSTRING_INHERITANCE_ENABLE") == "1"
    if not is_enabled:
        return

    # --- Checks below only apply when docstring inheritance is active ---

    # Parent description must not leak into the description section
    assert "Mixin class for all EEG models" not in doc.split("Parameters")[0], (
        "Parent description leaked into the child's description section"
    )

    # Inherited params should appear in the Parameters section (not just
    # in Hub notes which also mention these names in prose)
    params_section = doc[doc.index("Parameters") :]
    assert "\nn_chans" in params_section or "n_chans :" in params_section, (
        "Inherited parameter 'n_chans' missing from Parameters section"
    )
    assert "\nn_outputs" in params_section or "n_outputs :" in params_section, (
        "Inherited parameter 'n_outputs' missing from Parameters section"
    )

    # Child-specific params must NOT show "The description is missing"
    hidden_idx = params_section.index("hidden_size")
    next_param_candidates = ["drop_prob", "n_chans", "n_outputs"]
    end_idx = len(params_section)
    for candidate in next_param_candidates:
        try:
            idx = params_section.index(candidate, hidden_idx + 1)
            end_idx = min(end_idx, idx)
        except ValueError:
            pass
    hidden_desc = params_section[hidden_idx:end_idx]
    assert "description is missing" not in hidden_desc, (
        f"Child parameter 'hidden_size' lost its description:\n{hidden_desc}"
    )


def test_init_kwargs_tracked_for_subclass():
    """Verify that track_model_init_kwargs captures all constructor args."""
    model = _DummyModelWithParams(
        n_chans=8, n_outputs=2, n_times=100, hidden_size=32, drop_prob=0.3
    )
    kwargs = model._braindecode_init_kwargs
    assert kwargs["hidden_size"] == 32
    assert kwargs["drop_prob"] == 0.3
    assert kwargs["n_chans"] == 8
    assert kwargs["n_outputs"] == 2
    assert kwargs["n_times"] == 100


def test_hub_method_install_hint_no_hf():
    """Stubs raise ImportError with install hint when huggingface_hub is absent."""
    from braindecode.models.base import _BaseHubMixinStub

    class M(_BaseHubMixinStub):
        pass

    with pytest.raises(ImportError, match=r"M\.from_pretrained.*braindecode\[hub\]"):
        M.from_pretrained("any/repo")
    with pytest.raises(ImportError, match=r"M\.push_to_hub.*braindecode\[hub\]"):
        M().push_to_hub("any/repo")


# ---- channel layer (EEGModuleMixin.channel_strategy) -------------------------

from braindecode.models.base import HAS_HF_HUB  # noqa: E402
from braindecode.modules import ChannelTarget, register_channel_strategy  # noqa: E402
from braindecode.modules.channels import ChannelStrategy  # noqa: E402


def _chs(names):
    return [{"ch_name": n, "kind": "eeg"} for n in names]


class _TrainableScale(ChannelStrategy):
    """Exact copies times a learned gain (test-only trainable strategy)."""

    trainable = True
    reconstructs = False

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(()))

    def apply(self, x, m):
        return self.gain * (m.weights @ x)


@pytest.fixture
def trainable_strategy():
    """Register ``_test_trainable`` for one test only (no import-time registry change)."""
    from braindecode.modules.channels.strategies.base import _REGISTRY

    register_channel_strategy("_test_trainable")(_TrainableScale)
    yield "_test_trainable"
    _REGISTRY.pop("_test_trainable", None)


def test_test_strategy_is_not_registered_at_import():
    from braindecode.modules.channels.strategies.base import _REGISTRY

    assert "_test_trainable" not in _REGISTRY


class _ChannelModel(EEGModuleMixin, nn.Module):
    """Backbone over a fixed 4-electrode montage."""

    _channel_target = ChannelTarget("montage", chs_info=_chs(["Fz", "Cz", "Pz", "Oz"]))

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        channel_strategy="native",
        channel_strategy_kwargs=None,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        self._init_channel_tokenizer(channel_strategy, channel_strategy_kwargs)
        self.head = nn.Linear(4, self.n_outputs)

    def forward(self, x, chs_info=None):
        return self.head(self._encode_channels(x, chs_info).x.mean(-1))


def _channel_model(**kw):
    return _ChannelModel(
        chs_info=_chs(["Oz", "Cz", "C3", "Fz", "Pz"]), n_outputs=2, n_times=8, **kw
    )


def test_channel_strategy_roundtrips_through_config():
    model = _channel_model(channel_strategy="idw", channel_strategy_kwargs={"p": 1.0})
    config = json.loads(json.dumps(model.get_config()))
    assert config["channel_strategy"] == "idw"
    clone = _ChannelModel.from_config(config)
    assert clone.channel_tokenizer.strategy.p == 1.0
    clone.load_state_dict(model.state_dict())
    # Pz and Oz are missing: the fitted covariance reconstructs them.
    chs = _chs(["Cz", "C3", "C4", "Fz"])
    x = torch.randn(3, 4, 8)
    torch.testing.assert_close(clone(x, chs), model(x, chs), rtol=0, atol=0)


def test_native_channel_strategy_keeps_the_input():
    model = _ChannelModel(n_chans=4, n_outputs=2, n_times=8)
    x = torch.randn(1, 4, 8)
    assert model._encode_channels(x).x is x
    assert not any(k.startswith("channel_tokenizer") for k in model.state_dict())


def test_model_without_channel_contract_rejects_a_strategy():
    model = DummyModule(n_outputs=2, n_chans=3, n_times=8)
    model._init_channel_tokenizer()  # native is always fine
    with pytest.raises(ValueError, match="no channel contract"):
        model._init_channel_tokenizer("spline")


def test_trainable_strategy_state_saves_and_reloads(trainable_strategy):
    model = _channel_model(channel_strategy="_test_trainable")
    with torch.no_grad():
        model.channel_tokenizer.strategy.gain.fill_(3.0)
    assert "channel_tokenizer.strategy.gain" in model.state_dict()
    clone = _ChannelModel.from_config(model.get_config())
    clone.load_state_dict(model.state_dict(), strict=True)
    assert clone.channel_tokenizer.strategy.gain.item() == 3.0


def test_backbone_checkpoint_into_trainable_strategy_warns_fresh_keys(
    trainable_strategy,
):
    backbone = {
        k: v
        for k, v in _channel_model().state_dict().items()
        if not k.startswith("channel_tokenizer")
    }
    model = _channel_model(channel_strategy="_test_trainable")
    with pytest.warns(UserWarning, match=r"channel_tokenizer\.strategy\.gain"):
        model.load_state_dict(backbone, strict=True)
    assert model.channel_tokenizer.strategy.gain.item() == 1.0
    # Strict loading still guards the backbone.
    del backbone["head.bias"]
    with pytest.raises(RuntimeError, match="head.bias"):
        model.load_state_dict(backbone, strict=True)


@pytest.mark.skipif(not HAS_HF_HUB, reason="requires huggingface_hub")
def test_from_pretrained_with_trainable_strategy_warns_once(
    tmp_path, trainable_strategy
):
    _channel_model().save_pretrained(tmp_path)
    with pytest.warns(UserWarning, match="freshly initialised") as record:
        model = _ChannelModel.from_pretrained(
            tmp_path, channel_strategy="_test_trainable"
        )
    fresh = [w for w in record if "freshly initialised" in str(w.message)]
    assert len(fresh) == 1
    assert model.channel_tokenizer.strategy.gain.item() == 1.0


def _fitted_wiener_model():
    import numpy as np

    model = _channel_model(channel_strategy="wiener")
    dense = _chs(["Fz", "Cz", "Pz", "Oz", "C3", "C4", "FCz", "CPz"])
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8)) @ rng.normal(size=(8, 8))
    model.channel_tokenizer.fit(X, dense)
    return model


def test_backbone_checkpoint_into_wiener_says_fit():
    backbone = {
        k: v
        for k, v in _channel_model().state_dict().items()
        if not k.startswith("channel_tokenizer")
    }
    model = _channel_model(channel_strategy="wiener")
    with pytest.warns(UserWarning, match=r"freshly initialised.*fit\(\)"):
        model.load_state_dict(backbone, strict=True)


@pytest.mark.skipif(not HAS_HF_HUB, reason="requires huggingface_hub")
def test_fitted_wiener_model_saves_and_reloads(tmp_path):
    model = _fitted_wiener_model()
    model.save_pretrained(tmp_path)
    clone = _ChannelModel.from_pretrained(tmp_path)
    assert clone.channel_tokenizer.strategy.cov.shape == (8, 8)
    # Pz and Oz are missing: the fitted covariance reconstructs them.
    chs = _chs(["Cz", "C3", "C4", "Fz"])
    x = torch.randn(3, 4, 8)
    torch.testing.assert_close(clone(x, chs), model(x, chs), rtol=0, atol=0)
