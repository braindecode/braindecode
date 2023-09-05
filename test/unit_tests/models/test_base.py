# Authors: Pierre Guetschel
#
# License: BSD-3


import pytest
from torch import nn

from braindecode.models.base import EEGModuleMixin


class DummyModule(EEGModuleMixin, nn.Sequential):
    ''' Dummy module for testing EEGModuleMixin '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DummyModuleNTime(EEGModuleMixin, nn.Sequential):
    ''' Dummy module using one of the properties of EEGModuleMixin
     in its __init__ '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_module('dummy', nn.Linear(self.n_times, 1))


@pytest.mark.parametrize(
    'n_outputs, n_chans, ch_names, n_times, input_window_seconds, sfreq',
    [
        (None, 1, ['ch1'], 200, 2., 100.),
        (1, None, None, 200, 2., 100.),
        (1, 1, None, 200, 2., 100.),
        (1, 1, ['ch1'], None, None, None),
        (1, 1, ['ch1'], None, None, 100.),
        (1, 1, ['ch1'], None, 2., None),
        (1, 1, ['ch1'], 200, None, None),
    ]
)
def test_missing_params(
        n_outputs,
        n_chans,
        ch_names,
        n_times,
        input_window_seconds,
        sfreq,
):
    module = DummyModule(
        n_outputs=n_outputs,
        n_chans=n_chans,
        ch_names=ch_names,
        n_times=n_times,
        input_window_seconds=input_window_seconds,
        sfreq=sfreq,
    )
    with pytest.raises(ValueError):
        assert module.n_outputs == 1
        assert module.n_chans == 1
        assert module.ch_names == ['ch1']
        assert module.n_times == 200
        assert module.input_window_seconds == 2.
        assert module.sfreq == 100.


@pytest.mark.parametrize(
    'n_outputs, n_chans, ch_names, n_times, input_window_seconds, sfreq',
    [
        (1, 1, ['ch1'], 200, 2., 100.),
        (1, None, ['ch1'], 200, 2., 100.),
        (1, None, ['ch1'], None, 2., 100.),
        (1, None, ['ch1'], 200, None, 100.),
        (1, None, ['ch1'], 200, 2., None),
    ]
)
def test_all_params(
        n_outputs,
        n_chans,
        ch_names,
        n_times,
        input_window_seconds,
        sfreq,
):
    module = DummyModule(
        n_outputs=n_outputs,
        n_chans=n_chans,
        ch_names=ch_names,
        n_times=n_times,
        input_window_seconds=input_window_seconds,
        sfreq=sfreq,
    )
    assert module.n_outputs == 1
    assert module.n_chans == 1
    assert module.ch_names == ['ch1']
    assert module.n_times == 200
    assert module.input_window_seconds == 2.
    assert module.sfreq == 100.


@pytest.mark.parametrize(
    'n_outputs, n_chans, ch_names, n_times, input_window_seconds, sfreq',
    [
        (1, 2, ['ch1'], 200, 2., 100.),
        (1, 1, ['ch1'], 200, 3., 100.),
    ]
)
def test_incorrect_params(
        n_outputs,
        n_chans,
        ch_names,
        n_times,
        input_window_seconds,
        sfreq,
):
    with pytest.raises(ValueError):
        _ = DummyModule(
            n_outputs=n_outputs,
            n_chans=n_chans,
            ch_names=ch_names,
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
    'n_outputs, n_chans, ch_names, n_times, input_window_seconds, sfreq',
    [
        (1, 1, ['ch1'], 200, 2., 100.),
        (1, 1, ['ch1'], 200, 2., None),
        (1, 1, ['ch1'], 200, None, 100.),
        (1, 1, ['ch1'], None, 2., 100.),
    ]
)
def test_init_submodule(
        n_outputs,
        n_chans,
        ch_names,
        n_times,
        input_window_seconds,
        sfreq,
):
    _ = DummyModuleNTime(
        n_outputs=n_outputs,
        n_chans=n_chans,
        ch_names=ch_names,
        n_times=n_times,
        input_window_seconds=input_window_seconds,
        sfreq=sfreq,
    )
