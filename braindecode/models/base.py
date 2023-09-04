# Authors: Pierre Guetschel
#
# License: BSD-3

from typing import List, Optional

from docstring_inheritance import NumpyDocstringInheritanceMeta


class EEGModuleMixin(metaclass=NumpyDocstringInheritanceMeta):
    """
    Mixin class for EEG models.

    Parameters
    ----------
    n_channels: int
        Number of EEG channels.
    ch_names: list of str
        Names of the EEG channels.
    input_window_samples: int
        Number of time samples of the input window.
    input_window_seconds: float
        Length of the input window in seconds.
    sfreq: float
        Sampling frequency of the EEG recordings.

    .. note::

       If some input signal-related parameters are not specified,
       there will be an attempt to infer them from the other parameters.

    Raises
    ------
    AttributeError: If some input signal-related parameters are not specified
    and can not be inferred.
    """

    def __init__(
            self,
            n_channels: Optional[int] = None,
            ch_names: Optional[List[str]] = None,
            input_window_samples: Optional[int] = None,
            input_window_seconds: Optional[float] = None,
            sfreq: Optional[float] = None,
    ):
        if (
                n_channels is not None and
                ch_names is not None and
                len(ch_names) != n_channels
        ):
            raise ValueError(f'{n_channels=} different from {ch_names=} length')
        if (
                input_window_samples is not None and
                input_window_seconds is not None and
                sfreq is not None and
                input_window_samples != int(input_window_seconds * sfreq)
        ):
            raise ValueError(
                f'{input_window_samples=} different from '
                f'{input_window_seconds=} * {sfreq=}'
            )
        self._n_channels = n_channels
        self._ch_names = ch_names
        self._input_window_samples = input_window_samples
        self._input_window_seconds = input_window_seconds
        self._sfreq = sfreq
        super().__init__()

    @property
    def n_channels(self):
        if self._n_channels is None and self._ch_names is not None:
            return len(self._ch_names)
        elif self._n_channels is None:
            raise AttributeError(
                'n_channels could not be inferred. Either specify n_channels or ch_names.'
            )
        return self._n_channels

    @property
    def ch_names(self):
        if self._ch_names is None:
            raise AttributeError('ch_names not specified.')
        return self._ch_names

    @property
    def input_window_samples(self):
        if (
                self._input_window_samples is None and
                self._input_window_seconds is not None and
                self._sfreq is not None
        ):
            return int(self._input_window_seconds * self._sfreq)
        elif self._input_window_samples is None:
            raise AttributeError(
                'input_window_samples could not be inferred. '
                'Either specify input_window_samples or input_window_seconds and sfreq.'
            )
        return self._input_window_samples

    @property
    def input_window_seconds(self):
        if (
                self._input_window_seconds is None and
                self._input_window_samples is not None and
                self._sfreq is not None
        ):
            return self._input_window_samples / self._sfreq
        elif self._input_window_seconds is None:
            raise AttributeError(
                'input_window_seconds could not be inferred. '
                'Either specify input_window_seconds or input_window_samples and sfreq.'
            )
        return self._input_window_seconds

    @property
    def sfreq(self):
        if (
                self._sfreq is None and
                self._input_window_seconds is not None and
                self._input_window_samples is not None
        ):
            return self._input_window_samples / self._input_window_seconds
        elif self._sfreq is None:
            raise AttributeError(
                'sfreq could not be inferred. '
                'Either specify sfreq or input_window_seconds and input_window_samples.'
            )
        return self._sfreq
