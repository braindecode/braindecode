# Authors: Pierre Guetschel
#
# License: BSD-3

import warnings
from typing import Iterable, List, Optional, Dict

from docstring_inheritance import NumpyDocstringInheritanceInitMeta
from torchinfo import ModelStatistics, summary


def deprecated_args(obj, *old_new_args):
    out_args = []
    for old_name, new_name, old_val, new_val in old_new_args:
        if old_val is None:
            out_args.append(new_val)
        else:
            warnings.warn(
                f'{obj.__class__.__name__}: {old_name!r} is depreciated. Use {new_name!r} instead.'
            )
            if new_val is not None:
                raise ValueError(
                    f'{obj.__class__.__name__}: Both {old_name!r} and {new_name!r} were specified.'
                )
            out_args.append(old_val)
    return out_args


class EEGModuleMixin(metaclass=NumpyDocstringInheritanceInitMeta):
    """
    Mixin class for all EEG models in braindecode.

    Parameters
    ----------
    n_outputs : int
        Number of outputs of the model. This is the number of classes
        in the case of classification.
    n_chans : int
        Number of EEG channels.
    chs_info : list of dict
        Information about each individual EEG channel. This should be filled with
        ``info["chs"]``. Refer to :class:`mne.Info` for more details.
    n_times : int
        Number of time samples of the input window.
    input_window_seconds : float
        Length of the input window in seconds.
    sfreq : float
        Sampling frequency of the EEG recordings.

    Raises
    ------
    ValueError: If some input signal-related parameters are not specified
                and can not be inferred.

    Notes
    -----
    If some input signal-related parameters are not specified,
    there will be an attempt to infer them from the other parameters.
    """

    def __init__(
            self,
            n_outputs: Optional[int] = None,
            n_chans: Optional[int] = None,
            chs_info: Optional[List[Dict]] = None,
            n_times: Optional[int] = None,
            input_window_seconds: Optional[float] = None,
            sfreq: Optional[float] = None,
    ):
        if (
                n_chans is not None and
                chs_info is not None and
                len(chs_info) != n_chans
        ):
            raise ValueError(f'{n_chans=} different from {chs_info=} length')
        if (
                n_times is not None and
                input_window_seconds is not None and
                sfreq is not None and
                n_times != int(input_window_seconds * sfreq)
        ):
            raise ValueError(
                f'{n_times=} different from '
                f'{input_window_seconds=} * {sfreq=}'
            )
        self._n_outputs = n_outputs
        self._n_chans = n_chans
        self._chs_info = chs_info
        self._n_times = n_times
        self._input_window_seconds = input_window_seconds
        self._sfreq = sfreq
        super().__init__()

    @property
    def n_outputs(self):
        if self._n_outputs is None:
            raise ValueError('n_outputs not specified.')
        return self._n_outputs

    @property
    def n_chans(self):
        if self._n_chans is None and self._chs_info is not None:
            return len(self._chs_info)
        elif self._n_chans is None:
            raise ValueError(
                'n_chans could not be inferred. Either specify n_chans or chs_info.'
            )
        return self._n_chans

    @property
    def chs_info(self):
        if self._chs_info is None:
            raise ValueError('chs_info not specified.')
        return self._chs_info

    @property
    def n_times(self):
        if (
                self._n_times is None and
                self._input_window_seconds is not None and
                self._sfreq is not None
        ):
            return int(self._input_window_seconds * self._sfreq)
        elif self._n_times is None:
            raise ValueError(
                'n_times could not be inferred. '
                'Either specify n_times or input_window_seconds and sfreq.'
            )
        return self._n_times

    @property
    def input_window_seconds(self):
        if (
                self._input_window_seconds is None and
                self._n_times is not None and
                self._sfreq is not None
        ):
            return self._n_times / self._sfreq
        elif self._input_window_seconds is None:
            raise ValueError(
                'input_window_seconds could not be inferred. '
                'Either specify input_window_seconds or n_times and sfreq.'
            )
        return self._input_window_seconds

    @property
    def sfreq(self):
        if (
                self._sfreq is None and
                self._input_window_seconds is not None and
                self._n_times is not None
        ):
            return self._n_times / self._input_window_seconds
        elif self._sfreq is None:
            raise ValueError(
                'sfreq could not be inferred. '
                'Either specify sfreq or input_window_seconds and n_times.'
            )
        return self._sfreq

    def get_torchinfo_statistics(
            self,
            col_names: Optional[Iterable[str]] = (
                    "input_size",
                    "output_size",
                    "num_params",
                    "kernel_size",
            ),
            row_settings: Optional[Iterable[str]] = ("var_names", "depth"),
    ) -> ModelStatistics:
        """Generate table describing the model using torchinfo.summary.

        Parameters
        ----------
        col_names : tuple, optional
            Specify which columns to show in the output, see torchinfo for details, by default
            ("input_size", "output_size", "num_params", "kernel_size")
        row_settings : tuple, optional
             Specify which features to show in a row, see torchinfo for details, by default
             ("var_names", "depth")

        Returns
        -------
        torchinfo.ModelStatistics
            ModelStatistics generated by torchinfo.summary.
        """
        return summary(
            self,
            input_size=(1, self.n_chans, self.n_times),
            col_names=col_names,
            row_settings=row_settings,
            verbose=0,
        )

    def __str__(self) -> str:
        return str(self.get_torchinfo_statistics())
