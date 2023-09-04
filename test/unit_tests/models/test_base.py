# Authors: Pierre Guetschel
#
# License: BSD-3


import pytest
from torch import nn

from braindecode.models.base import EEGModuleMixin


class DummyModule(EEGModuleMixin, nn.Sequential):
    ''' Dummy module for testing EEGModuleMixin

    Parameters
    ----------
    a: int
        Dummy parameter.
    '''
    pass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_module('dummy', nn.Linear(1, 1))


class TestEEGModuleMixin:
    @pytest.mark.parametrize(
        'n_channels, ch_names, n_times, input_window_seconds, sfreq',
        [
            (None, None, 200, 2., 100.),
            (1, None, 200, 2., 100.),
            (1, ['ch1'], None, None, None),
            (1, ['ch1'], None, None, 100.),
            (1, ['ch1'], None, 2., None),
            (1, ['ch1'], 200, None, None),
        ]
    )
    def test_missing_params(
            self,
            n_channels,
            ch_names,
            n_times,
            input_window_seconds,
            sfreq,
    ):
        module = DummyModule(
            n_channels=n_channels,
            ch_names=ch_names,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        with pytest.raises(AttributeError):
            assert module.n_channels == 1
            assert module.ch_names == ['ch1']
            assert module.n_times == 200
            assert module.input_window_seconds == 2.
            assert module.sfreq == 100.

    @pytest.mark.parametrize(
        'n_channels, ch_names, n_times, input_window_seconds, sfreq',
        [
            (1, ['ch1'], 200, 2., 100.),
            (None, ['ch1'], 200, 2., 100.),
            (None, ['ch1'], None, 2., 100.),
            (None, ['ch1'], 200, None, 100.),
            (None, ['ch1'], 200, 2., None),
        ]
    )
    def test_all_params(
            self,
            n_channels,
            ch_names,
            n_times,
            input_window_seconds,
            sfreq,
    ):
        module = DummyModule(
            n_channels=n_channels,
            ch_names=ch_names,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        assert module.n_channels == 1
        assert module.ch_names == ['ch1']
        assert module.n_times == 200
        assert module.input_window_seconds == 2.
        assert module.sfreq == 100.

    @pytest.mark.parametrize(
        'n_channels, ch_names, n_times, input_window_seconds, sfreq',
        [
            (2, ['ch1'], 200, 2., 100.),
            (1, ['ch1'], 200, 3., 100.),
        ]
    )
    def test_incorrect_params(
            self,
            n_channels,
            ch_names,
            n_times,
            input_window_seconds,
            sfreq,
    ):
        with pytest.raises(ValueError):
            _ = DummyModule(
                n_channels=n_channels,
                ch_names=ch_names,
                n_times=n_times,
                input_window_seconds=input_window_seconds,
                sfreq=sfreq,
            )

    def test_inexistent_param(self):
        with pytest.raises(TypeError):
            _ = DummyModule(
                inexistant_param=1,
            )
