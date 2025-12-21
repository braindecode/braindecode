# Authors: Pierre Guetschel
#
# License: BSD-3


from unittest.mock import patch

import pytest
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
