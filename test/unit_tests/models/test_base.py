# Authors: Pierre Guetschel
#
# License: BSD-3


import ast
import inspect
import json
import subprocess
import sys
import warnings
import weakref
from operator import attrgetter
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch
from torch import nn

from braindecode.models import (
    EEGCLIP,
    LUNA,
    PBT,
    AttnSleep,
    CBraMod,
    Deep4Net,
    EEGNet,
    ShallowFBCSPNet,
)
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import models_dict, models_mandatory_parameters
from test.unit_tests.models._fixtures import (
    _get_all_possible_signal_params,
    _get_possible_signal_params,
    _get_signal_params,
)

if TYPE_CHECKING:
    from typing import Any as RuntimeOnlyMetadata


class DummyTrustedIdentity(nn.Identity):
    """Application-owned type, not in the serialized module lookup."""


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


class DummyModuleWithInheritedExtraIgnoredProperty(
    DummyModuleWithExtraIgnoredProperty
):
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
    "required_params, expected_len",
    [
        ([], 42),
        (["n_outputs"], 21),
        (["chs_info"], 14),
        (["n_chans", "n_outputs"], 14),
        (["n_chans", "n_outputs", "n_times"], 8),
        (["n_chans", "n_outputs", "n_times", "sfreq"], 6),
        (["chs_info", "n_outputs", "n_times", "sfreq"], 3),
    ],
)
def test_all_42_signal_param_combinations_on_dummy_module(
    required_params, expected_len
):
    """Exhaustively test all valid signal-parameter combinations on DummyModule."""
    sp = _get_signal_params(None)
    all_combos = _get_all_possible_signal_params(sp, required_params)
    minimal_combos = _get_possible_signal_params(sp, required_params)

    assert len(all_combos) == expected_len
    assert minimal_combos[0] == all_combos[0]
    assert 2 <= len(minimal_combos) <= 3
    for mc in minimal_combos:
        assert mc in all_combos

    for kw in all_combos:
        module = DummyModule(**kw)
        if kw["n_outputs"] is not None:
            assert module.n_outputs == sp["n_outputs"]
        else:
            with pytest.raises(ValueError):
                _ = module.n_outputs

        if kw["chs_info"] is not None or kw["n_chans"] is not None:
            assert module.n_chans == sp["n_chans"]
        else:
            with pytest.raises(ValueError):
                _ = module.n_chans

        n_time_given = sum(
            kw[k] is not None for k in ("n_times", "sfreq", "input_window_seconds")
        )
        if n_time_given >= 2 or kw["n_times"] is not None:
            assert module.n_times == sp["n_times"]
        else:
            with pytest.raises(ValueError):
                _ = module.n_times


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


def test_printing_a_model_does_not_keep_it_alive():
    # No parametrization: torchinfo's records of a parametrized module form
    # cycles that only the cycle collector frees.
    model = ShallowFBCSPNet(n_chans=4, n_outputs=2, n_times=500)
    str(model)
    ref = weakref.ref(model)
    del model
    assert ref() is None


# GCNsNet is not listed: it keeps the bound-method hook of v2.0.0, which the
# whole-model pickles of v2.0.0 store by name.
@pytest.mark.parametrize("name", ["LUNA"])
def test_model_with_a_load_hook_is_freed_by_reference_counting(name):
    required, signal_params = next(
        (r, s) for n, r, s in models_mandatory_parameters if n == name
    )
    model = models_dict[name](**_get_signal_params(signal_params, required))
    ref = weakref.ref(model)
    del model
    assert ref() is None


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
    with pytest.raises(
        RuntimeError, match=r"Kernel size can't be greater than actual input size"
    ):
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
    with pytest.raises(RuntimeError, match=r"Output size is too small"):
        dummy_module.get_output_shape()


def test_declared_subclass_checks_min_n_times_and_fixed_n_times():
    class DeclaredFree(DummyModule):
        _time_axis = "free"
        _channel_axis = "fixed"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._min_n_times = 201

    free_mod = DeclaredFree(n_outputs=2, n_chans=1, n_times=200)
    with pytest.raises(
        ValueError,
        match=r"DeclaredFree needs at least 201 time samples and got 200",
    ):
        free_mod.get_output_shape()
    with pytest.raises(
        ValueError,
        match=r"DeclaredFree needs at least 201 time samples and got 50",
    ):
        free_mod(torch.randn(2, 1, 50))
    assert free_mod(torch.randn(2, 1, 250)).shape == (2, 1, 250)

    class DeclaredFixed(DummyModule):
        _time_axis = "fixed"
        _channel_axis = "fixed"

    fixed_mod = DeclaredFixed(n_outputs=2, n_chans=1, n_times=200)
    with pytest.raises(
        ValueError,
        match=r"DeclaredFixed was built for n_times=200 and got 250 samples",
    ):
        fixed_mod(torch.randn(2, 1, 250))


@pytest.mark.parametrize("name", ["CTNet", "SleepStagerChambon2018", "SignalJEPA_PreLocal"])
def test_window_shorter_than_the_layers_raises_value_error(name):
    from braindecode.models.util import models_dict

    model = models_dict[name](n_chans=2, n_outputs=2, n_times=50, sfreq=100.0)
    with pytest.raises(
        ValueError, match=r"needs at least \d+ time samples and got 50 \(input shape \(2, 2, 50\)\)"
    ):
        model(torch.randn(2, 2, 50))


@pytest.mark.parametrize(
    "name, n_times, kwargs",
    [
        ("SleepStagerBlanco2020", 50, {}),
        ("EEGConformer", 50, {}),
        ("MSVTNet", 50, {}),
        ("ShallowFBCSPNet", 50, {}),
        ("EEGNet", 8, {}),
        ("EEGITNet", 8, {}),
        ("TSception", 50, {"pool_size": 64}),
        ("TSception", 3, {}),
        ("AttnSleep", 1, {"n_chans": 1}),
    ],
)
def test_window_shorter_than_the_layers_raises_value_error_at_init(
    name, n_times, kwargs
):
    from braindecode.models.util import models_dict

    n_chans = kwargs.get("n_chans", 2)
    with pytest.raises(
        ValueError, match=rf"too small `n_times`.*\(1, {n_chans}, {n_times}\)"
    ):
        models_dict[name](
            n_outputs=2, n_times=n_times, sfreq=100.0, **{"n_chans": 2, **kwargs}
        )


@pytest.mark.parametrize(
    "n_times, input_window_seconds, sfreq",
    [
        (1001, 4.004, 250.0),  # Issue example: 4.004 * 250.0 = 1001.0
        (751, 3.004, 250.0),   # 3.004 * 250.0 = 751.0
        (501, 2.004, 250.0),   # 2.004 * 250.0 = 501.0
        (101, 0.404, 250.0),   # 0.404 * 250.0 = 101.0
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
        (1001, None, 250.0),   # Infer input_window_seconds
        (751, None, 250.0),    # Infer input_window_seconds
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


def test_signal_jepa_subclass_config_without_hub():
    """A subclass keeps its own constructor arguments in ``get_config`` without the Hub."""
    code = """
import sys
sys.modules["huggingface_hub"] = None
import mne
from braindecode.models import SignalJEPA_Contextual
from braindecode.models.base import HAS_HF_HUB
assert not HAS_HF_HUB
info = mne.create_info(["Cz", "Pz"], 128, "eeg").set_montage("standard_1020")
model = SignalJEPA_Contextual(chs_info=info["chs"], n_times=256, sfreq=128,
                             n_outputs=2, n_spat_filters=2,
                             transformer__num_encoder_layers=1,
                             transformer__num_decoder_layers=1)
assert model.get_config()["n_spat_filters"] == 2
"""
    subprocess.run([sys.executable, "-c", code], check=True)  # nosec B603 - fixed test


def test_hub_method_install_hint_no_hf():
    """Stubs raise ImportError with install hint when huggingface_hub is absent."""
    from braindecode.models.base import _BaseHubMixinStub

    class M(_BaseHubMixinStub):
        pass

    with pytest.raises(ImportError, match=r"M\.from_pretrained.*braindecode\[hub\]"):
        M.from_pretrained("any/repo")
    with pytest.raises(ImportError, match=r"M\.push_to_hub.*braindecode\[hub\]"):
        M().push_to_hub("any/repo")


def test_get_output_shape_leaves_batchnorm_stats_and_modes_unchanged():
    from braindecode.models import Deep4Net, EEGNet

    model = Deep4Net(n_chans=4, n_outputs=2, n_times=500)
    model.train()
    frozen = model.conv_time_spat
    frozen.eval()
    bn = next(m for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    with torch.no_grad():
        bn.running_mean.fill_(5.0)
        bn.running_var.fill_(4.0)
    tracked = int(bn.num_batches_tracked)

    model.get_output_shape()

    assert torch.all(bn.running_mean == 5.0) and torch.all(bn.running_var == 4.0)
    assert int(bn.num_batches_tracked) == tracked
    assert model.training and not frozen.training
    # models that call get_output_shape in __init__ start with untouched stats
    for module in EEGNet(n_chans=4, n_outputs=2, n_times=256).modules():
        if isinstance(module, nn.BatchNorm2d):
            assert int(module.num_batches_tracked) == 0


@pytest.mark.parametrize("route", ["from_config", "from_pretrained", "pydantic", "coder"])
@pytest.mark.parametrize("value", ["builtins.dict", "hub_test_untrusted.Layer", "torch.nn.functional.relu"])
def test_serialized_types_rejected_before_import(tmp_path, monkeypatch, route, value):
    """All config decoding routes reject unknown paths before import or call."""
    import importlib.abc

    from braindecode.models.config import make_model_config

    attempted = []

    class ImportRecorder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "hub_test_untrusted":
                attempted.append(fullname)
                raise AssertionError("untrusted module import attempted")

    monkeypatch.setattr(sys, "meta_path", [ImportRecorder(), *sys.meta_path])
    config = dict(n_chans=2, n_times=8, n_outputs=2, activation=value)
    if route in ("from_pretrained", "coder"):
        pytest.importorskip("huggingface_hub")
    with pytest.raises(ValueError, match="Unsupported serialized module type"):
        if route == "from_config":
            DummyModuleConfigRoundTrip.from_config(config)
        elif route == "from_pretrained":
            (tmp_path / "config.json").write_text(json.dumps(config))
            DummyModuleConfigRoundTrip.from_pretrained(tmp_path, local_files_only=True)
        elif route == "coder":
            DummyModuleConfigRoundTrip._hub_mixin_coders[type][1](value)
        else:
            make_model_config(DummyModuleConfigRoundTrip, [])(**config)
    assert attempted == []


@pytest.mark.parametrize("path", ["torch.nn.ELU", "torch.nn.modules.activation.ELU", "braindecode.modules.Square", "braindecode.modules.activation.Square", "braindecode.modules.activation.SafeLog", "braindecode.modules.activation.LogActivation", "braindecode.modules.activation.SmoothMaximumUnit"])
def test_supported_serialized_types_roundtrip(tmp_path, path):
    pytest.importorskip("huggingface_hub")
    from braindecode import modules
    from braindecode.models.config import make_model_config

    expected = nn.ELU if path.endswith("ELU") else getattr(modules, path.rsplit(".", 1)[-1])
    config = dict(n_chans=2, n_times=8, n_outputs=2, activation=path)
    model = DummyModuleConfigRoundTrip.from_config(config).eval()
    assert model.activation is expected
    cfg_cls = make_model_config(DummyModuleConfigRoundTrip, [])
    cfg = cfg_cls(**config)
    assert cfg_cls.model_validate_json(cfg.model_dump_json()).activation is expected
    model.save_pretrained(tmp_path)
    restored = DummyModuleConfigRoundTrip.from_pretrained(tmp_path, strict=True).eval()
    x = torch.rand(2, 2, 8) + 0.1  # LogActivation expects positive inputs.
    torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)
    assert restored.activation is expected


@pytest.mark.parametrize("annotation,optional_annotation", [(type[nn.Module], type[nn.Module] | None), ("type[nn.Module]", "type[nn.Module] | None")])
def test_optional_and_required_type_annotations_are_resolved(annotation, optional_annotation):
    """Type args without class defaults must not miss the shared resolver."""
    from braindecode.models.util import resolve_type_kwargs

    class Config:
        def __init__(
            self, required: annotation, optional: optional_annotation = None,
            metadata: "RuntimeOnlyMetadata | None" = None,
        ):
            pass

    kwargs = dict(required="torch.nn.ELU", optional="torch.nn.Identity")
    assert resolve_type_kwargs(Config, kwargs) == dict(required=nn.ELU, optional=nn.Identity)
    with pytest.raises(ValueError, match="Unsupported serialized module type"):
        resolve_type_kwargs(Config, dict(optional="builtins.dict"))


@pytest.mark.parametrize("route", ["from_config", "from_pretrained"])
@pytest.mark.parametrize("activation", [nn.Identity, "torch.nn.Identity", "builtins.dict"])
def test_unrelated_type_annotations_do_not_block_loading(tmp_path, route, activation):
    class AnnotationModel(EEGModuleMixin, nn.Sequential):
        def __init__(self, activation=nn.Identity, metadata: "RuntimeOnlyMetadata | None" = None):
            super().__init__(n_chans=2, n_times=8, n_outputs=2)
            self.activation = activation
            self.metadata = metadata
            self.add_module("act", activation())
            self.add_module("linear", nn.Linear(8, 2))

    if route == "from_pretrained":
        pytest.importorskip("huggingface_hub")
        AnnotationModel().save_pretrained(tmp_path)

    def load():
        if route == "from_config":
            return AnnotationModel.from_config({"activation": activation})
        return AnnotationModel.from_pretrained(
            tmp_path, activation=activation, local_files_only=True, strict=True
        )

    if activation == "builtins.dict":
        with pytest.raises(ValueError, match="Unsupported serialized module type"):
            load()
    else:
        restored = load()
        assert restored.activation is nn.Identity
        assert restored.metadata is None


@pytest.mark.parametrize("annotation", [type, type[nn.Module], type | None, "type[nn.Module] | None"])
def test_trusted_type_override_discards_unsupported_saved_type(tmp_path, annotation):
    pytest.importorskip("huggingface_hub")

    class ExplicitTypeModel(EEGModuleMixin, nn.Sequential):
        def __init__(self, activation: type = nn.Identity):
            super().__init__(n_chans=2, n_times=8, n_outputs=2)
            self.activation = activation
            self.add_module("act", activation())
            self.add_module("linear", nn.Linear(8, 2))

        # Vary the annotation before Hub captures the constructor signature.
        __init__.__annotations__["activation"] = annotation

    model = ExplicitTypeModel(activation=DummyTrustedIdentity)
    model.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["activation"].endswith(".DummyTrustedIdentity")
    with pytest.raises(ValueError, match="Unsupported serialized module type"):
        ExplicitTypeModel.from_pretrained(tmp_path, local_files_only=True)
    restored = ExplicitTypeModel.from_pretrained(
        tmp_path, activation=DummyTrustedIdentity, local_files_only=True, strict=True
    )
    assert restored.activation is DummyTrustedIdentity
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("explicit_override", [False, True])
@pytest.mark.parametrize("annotation_kind", ["plain", "optional", "callable", "callable_type"])
def test_application_registered_type_coder_roundtrip(tmp_path, explicit_override, annotation_kind):
    from collections.abc import Callable

    pytest.importorskip("huggingface_hub")
    decoded = []

    def encode(cls):
        assert cls is nn.Identity
        return "identity-v1"

    def decode(value):
        decoded.append(value)
        return {"identity-v1": nn.Identity}[value]

    annotation = {"plain": type, "optional": type | None, "callable": Callable, "callable_type": type}[annotation_kind]
    coder_type = Callable if annotation_kind.startswith("callable") else type

    class AliasCodedModel(EEGModuleMixin, nn.Sequential, coders={coder_type: (encode, decode)}):
        def __init__(self, activation: type = nn.Identity):
            super().__init__(n_chans=2, n_times=8, n_outputs=2)
            self.activation = activation
            self.add_module("act", activation())
            self.add_module("linear", nn.Linear(8, 2))

        __init__.__annotations__["activation"] = annotation

    model = AliasCodedModel()
    model.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["activation"] == "identity-v1"
    overrides = {"activation": nn.Identity} if explicit_override else {}
    restored = AliasCodedModel.from_pretrained(
        tmp_path, local_files_only=True, strict=True, **overrides
    )
    assert restored.activation is nn.Identity
    assert decoded == ["identity-v1"]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


def test_load_state_dict_from_before_max_norm_parametrization():
    # Released checkpoints (e.g. PierreGtch/EEGNetv4) store the max-norm
    # spatial filters as ``conv_spatial.weight`` and the classifier under its
    # old name. load_state_dict maps both to the current keys.
    torch.manual_seed(0)
    model = EEGNet(n_chans=3, n_outputs=5, n_times=385).eval()
    legacy = {
        k.replace("parametrizations.weight.original", "weight").replace(
            "final_layer.conv_classifier", "conv_classifier"
        ): v
        for k, v in model.state_dict().items()
    }
    assert "conv_spatial.weight" in legacy

    loaded = EEGNet(n_chans=3, n_outputs=5, n_times=385).eval()
    loaded.load_state_dict(legacy)

    x = torch.randn(2, 3, 385)
    torch.testing.assert_close(loaded(x), model(x))


def test_load_state_dict_preserves_batch_norm_version():
    model = EEGNet(n_chans=3, n_outputs=2, n_times=128)
    state = model.state_dict()
    del state["bnorm_1.num_batches_tracked"]
    with pytest.raises(RuntimeError, match=r"bnorm_1\.num_batches_tracked"):
        model.load_state_dict(state)

    # PyTorch added this buffer in BatchNorm version 2.
    state._metadata["bnorm_1"]["version"] = 1
    model.load_state_dict(state)


# Saved with braindecode 2.0.0 (git tag v2.0.0) and torch 2.6: each model below built
# after ``torch.manual_seed(0)`` (``proj_out`` of CBraMod set to the identity, as 2.1
# does not apply it), ``y = model.eval()(x)`` for an ``x`` drawn with seed 1.
# ``v2_pickles.pth`` holds ``torch.save(model)`` after that forward pass, for the models
# that can be pickled whole, and a bare ``tfm_tokenizer._GroupNorm(2, 4)`` with random
# affine parameters. EEGNet, whose weights are parametrized, cannot be pickled: its
# ``state_dict()`` is in ``v2_state_dicts.pth``.
_V2_DATA = Path(__file__).parent / "data"
_V2_MODELS = {
    "EEGNet": lambda: EEGNet(
        n_chans=3, n_outputs=2, n_times=64, F1=2, D=2, kernel_length=8,
        depthwise_kernel_length=4,
    ),
    "CBraMod": lambda: CBraMod(
        n_chans=3, n_outputs=2, n_times=32, patch_size=16, emb_dim=16, nhead=2,
        n_layer=1, dim_feedforward=16,
        channels_kernel_stride_padding_norm=(
            (4, 9, 4, 4, (2, 4)), (4, 3, 1, 1, (2, 4)), (4, 3, 1, 1, (2, 4))
        ),
    ),
    "EEGCLIP": lambda: EEGCLIP(
        n_chans=3, n_outputs=2, n_times=700, text_embedding_dim=8, eeg_embedding_dim=4,
        projection_layers=1,
        eeg_encoder=Deep4Net(
            n_chans=3, n_times=700, n_outputs=4, final_conv_length=2,
            stride_before_pool=True, n_filters_time=2, n_filters_spat=2,
            n_filters_2=2, n_filters_3=2, n_filters_4=2,
        ),
    ),
    "LUNA": lambda: LUNA(
        n_chans=3, n_times=40, n_outputs=2, patch_size=20, num_queries=1,
        embed_dim=16, depth=1, num_heads=1, mlp_ratio=1.0,
    ),
    "PBT": lambda: PBT(
        n_chans=3, n_times=32, n_outputs=2, d_input=8, embed_dim=16, num_heads=2,
        num_layers=1,
    ),
}


def _v2_pickle(name):
    # Unpickling runs code: this file is part of the repository.
    pickles = torch.load(_V2_DATA / "v2_pickles.pth", weights_only=False)  # nosec B614
    return pickles[name]


@pytest.mark.parametrize("name", _V2_MODELS)
def test_v2_checkpoints_load(name):
    if name == "EEGNet":
        saved = torch.load(_V2_DATA / "v2_state_dicts.pth", weights_only=True)[name]
        state = saved["state_dict"]
    else:
        saved = _v2_pickle(name)
        state = saved["model"].state_dict()
    model = _V2_MODELS[name]().eval()
    if name == "LUNA":
        # v2.0.0 omitted hidden LayerNorm in TimmMlp (#1588); verify those are the only
        # missing keys against the 2.0.0 snapshot and that Identity preserves outputs.
        incompatible = model.load_state_dict(state, strict=False)
        assert not incompatible.unexpected_keys
        assert set(incompatible.missing_keys) == {
            "channel_location_embedder.norm.weight",
            "channel_location_embedder.norm.bias",
            "cross_attn.ffn.norm.weight",
            "cross_attn.ffn.norm.bias",
        }
        model.channel_location_embedder.norm = nn.Identity()
        model.cross_attn.ffn.norm = nn.Identity()
    model.load_state_dict(state, strict=True)
    with torch.no_grad():
        torch.testing.assert_close(model(saved["x"]), saved["y"])


@pytest.mark.parametrize(
    "name", ["CBraMod", "EEGCLIP", "LUNA", "PBT", "tfm_tokenizer._GroupNorm"]
)
def test_v2_pickles_run(name):
    saved = _v2_pickle(name)
    with torch.no_grad():
        torch.testing.assert_close(saved["model"](saved["x"]), saved["y"])


def test_reset_head_on_v2_cbramod_pickle():
    saved = _v2_pickle("CBraMod")
    saved["model"].reset_head(3)
    assert saved["model"](saved["x"]).shape == (2, 3)


def _small_attn_sleep():
    # PReLU has weights, so they are among those of the layers that braindecode
    # 2.0.0 registered twice, as ``features.<index>``.
    return AttnSleep(
        sfreq=100, n_outputs=2, input_window_seconds=30, n_tce=1, d_ff=8,
        after_reduced_cnn_size=16, n_attn_heads=1, activation=nn.PReLU,
    )


def _v2_layout(name):
    """Zeros with the keys and shapes braindecode 2.0.0 saved for the small model."""
    pairs = (line.split(" ", 1) for line in (_V2_DATA / name).read_text().splitlines())
    return {key: torch.zeros(json.loads(shape)) for key, shape in pairs}


def test_attn_sleep_reloads_its_checkpoint_strictly(tmp_path):
    pytest.importorskip("huggingface_hub")
    model = _small_attn_sleep().eval()
    model.save_pretrained(tmp_path)
    restored = AttnSleep.from_pretrained(tmp_path, strict=True).eval()
    x = torch.randn(2, 1, 3000)
    torch.testing.assert_close(restored(x), model(x))


def test_attn_sleep_loads_a_v2_state_dict():
    # 2.0.0 state dicts hold both names of the layers it registered twice.
    state = _v2_layout("attn_sleep_v2.0.0_state_dict.txt")
    _small_attn_sleep().load_state_dict(state, strict=True)


def test_attn_sleep_loads_a_v2_safetensors_file(tmp_path):
    pytest.importorskip("huggingface_hub")
    from safetensors.torch import save_file

    # safetensors keeps one name per tied tensor: the SE and PReLU weights of the
    # AFR block are saved as ``features.<index>``, the other layers under their names.
    _small_attn_sleep().save_pretrained(tmp_path)
    state = _v2_layout("attn_sleep_v2.0.0_safetensors.txt")
    save_file(state, tmp_path / "model.safetensors")
    AttnSleep.from_pretrained(tmp_path, strict=True)


# Old forward arguments of v2.0.0 keep working until their removal release
# (deprecated_forward_kwargs); forward itself only takes the new names.
def _aliases(cls):
    return (cls._forward_deprecation or (None, {}, None))[1]


_V2_FORWARD = [
    (name, set(args))
    for name, *args in (
        line.split()
        for line in (_V2_DATA / "v2_forward_arguments.txt").read_text().splitlines()
    )
]


@pytest.mark.parametrize("name,v2_args", _V2_FORWARD)
def test_v2_forward_arguments_are_still_accepted(name, v2_args):
    cls = models_dict[name]
    accepted = set(inspect.signature(cls.forward).parameters) | set(_aliases(cls))
    assert v2_args <= accepted


@pytest.mark.parametrize(
    "name",
    sorted(n for n, c in models_dict.items() if "_deprecated_kwargs" in vars(c.forward)),
)
def test_deprecated_forward_keywords_rename_to_forward_parameters(name):
    params = inspect.signature(models_dict[name].forward).parameters
    for old, new in _aliases(models_dict[name]).items():
        new = new[0] if isinstance(new, tuple) else new
        assert new in params and old not in params


def test_old_forward_keyword_warns_and_matches_the_new_one():
    model = EEGNet(n_chans=3, n_outputs=2, n_times=128).eval()

    class Wrapped(EEGNet):  # overrides forward without the decorator
        def forward(self, x):
            return super().forward(x)

    class Old(EEGNet):  # a 2.0 subclass whose forward still says input
        def forward(self, input):
            return super().forward(input)

    wrapped = Wrapped(n_chans=3, n_outputs=2, n_times=128).eval()
    wrapped.load_state_dict(model.state_dict())
    old = Old(n_chans=3, n_outputs=2, n_times=128).eval()
    old.load_state_dict(model.state_dict())
    x = torch.randn(2, 3, 128)
    with torch.no_grad():
        expected = model(x)
        with pytest.warns(FutureWarning, match="Calling EEGNet with input="):
            assert torch.equal(model(input=x), expected)
        with pytest.warns(FutureWarning, match="Calling Wrapped with input="):
            assert torch.equal(wrapped(input=x), expected)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            assert torch.equal(old(input=x), expected)
        with pytest.raises(TypeError, match="got both 'input'"):
            model(input=x, x=x)
    assert "input" not in str(torch.jit.script(model).forward.schema)


def test_old_forward_arguments_compile_with_fullgraph():
    # __call__ reads a plain class attribute, so torch.compile traces it whole.
    model = EEGNet(n_chans=3, n_outputs=2, n_times=128).eval()
    x = torch.randn(2, 3, 128)
    compiled = torch.compile(model, fullgraph=True, backend="eager")
    with torch.no_grad():
        assert torch.equal(compiled(input=x), model(x))


def test_old_sleepfm_channel_mask_is_the_inverse_of_ch_mask():
    required, signal_params = next(
        (r, s) for n, r, s in models_mandatory_parameters if n == "SleepFM"
    )
    signal = _get_signal_params(signal_params)
    model = models_dict["SleepFM"](**_get_signal_params(signal_params, required))
    model.eval()
    x = torch.randn(2, len(signal["chs_info"]), signal["n_times"])
    missing = torch.zeros(2, x.shape[1], dtype=torch.bool)
    missing[:, -1] = True  # the 2.0 channel_mask is True for missing channels
    with torch.no_grad():
        expected = model(x, ch_mask=~missing)
        assert not torch.equal(expected, model(x))
        with pytest.warns(FutureWarning, match="channel_mask"):
            assert torch.equal(model(x, channel_mask=missing), expected)
        with pytest.warns(FutureWarning, match="channel_mask"):
            assert torch.equal(model(x, missing), expected)  # 2.0 positional order
        with pytest.warns(FutureWarning, match="channel_mask"):
            assert torch.equal(model(x, channel_mask=None), model(x))
        with pytest.raises(TypeError, match="multiple values for argument"):
            model(x, missing, channel_mask=missing)
        with pytest.raises(TypeError, match="at most 3 positional"):
            model(x, missing, False, None)
        with pytest.warns(FutureWarning, match="second positional argument is ch_mask"):
            old = model.encode(x, missing)[0]
        assert torch.equal(old, model.encode(x, ch_mask=~missing)[0])


# Constructor feature flags of v2.0.0 keep their 2.0 output until 2.3
# (deprecated_init_flags). Shapes recorded with v2.0.0 for the test fixtures.
_V2_FLAG_OUTPUTS = [
    (name, flag, ast.literal_eval(output))
    for name, flag, output in (
        line.split(" ", 2)
        for line in (_V2_DATA / "v2_flag_outputs.txt").read_text().splitlines()
    )
]


def _structure(out):
    if torch.is_tensor(out):
        return tuple(out.shape)
    if isinstance(out, dict):
        return {k: _structure(v) for k, v in out.items()}
    if isinstance(out, tuple):
        return [_structure(v) for v in out]
    return out


def test_every_deprecated_flag_has_its_v2_output_recorded():
    decorated = {
        (name, flag)
        for name, cls in models_dict.items()
        for flag in vars(cls).get("_deprecated_init_flags", (None, {}))[1]
    }
    assert decorated == {(name, flag) for name, flag, _ in _V2_FLAG_OUTPUTS}


@pytest.mark.parametrize("name,flag,v2_output", _V2_FLAG_OUTPUTS)
def test_deprecated_constructor_flag_keeps_its_v2_output(name, flag, v2_output):
    required, signal_params = next(
        (r, s) for n, r, s in models_mandatory_parameters if n == name
    )
    signal = _get_signal_params(signal_params)
    kwargs = {**_get_signal_params(signal_params, required), flag: True}
    with pytest.warns(FutureWarning, match=rf"{flag}=\.\.\.\) is deprecated"):
        model = models_dict[name](**kwargs).eval()
    x = torch.randn(2, len(signal["chs_info"]), signal["n_times"])
    with torch.no_grad():
        assert _structure(model(x)) == v2_output


def test_deprecated_constructor_flag_config_and_torchscript():
    kwargs = dict(sfreq=100, n_outputs=2, input_window_seconds=30)
    with pytest.warns(FutureWarning):  # False is passed, so it warns too
        plain = AttnSleep(**kwargs, return_feats=False).eval()
    with pytest.warns(FutureWarning):
        legacy = AttnSleep(**kwargs, return_feats=True).eval()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        AttnSleep.from_config(plain.get_config())  # a false flag loads silently
    with pytest.warns(FutureWarning):
        reloaded = AttnSleep.from_config(legacy.get_config()).eval()
    x = torch.randn(2, 1, 3000)
    with torch.no_grad():
        assert plain(x).shape == (2, 2)
        assert reloaded(x).shape == legacy(x).shape == (2, 2400)
    with pytest.raises(RuntimeError, match="deprecated constructor feature flag"):
        torch.jit.script(legacy)
    # As in 2.0: no head weights with the flag, and asking for a head clears it.
    assert not any(k.startswith("final_layer.") for k in legacy.state_dict())
    legacy.reset_head(3)
    with torch.no_grad():
        assert legacy(x).shape == (2, 3)
    assert legacy.get_config()["return_feats"] is False

