"""Get epochs from mne.Raw"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#          David Sabbagh <dav.sabbagh@gmail.com>
#          Henrik Bonsmann <henrikbons@gmail.com>
#          Ann-Kathrin Kiessner <ann-kathrin.kiessner@gmx.de>
#          Vytautas Jankauskas <vytauto.jankausko@gmail.com>
#          Dan Wilson <dan.c.wil@gmail.com>
#          Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Mohammed Fattouh <mo.fattouh@gmail.com>
#          Robin Schirrmeister <robintibor@gmail.com>
#          Matthew Chen <matt.chen42601@gmail.com>
#          Sarthak Tayal <sarthaktayal2@gmail.com>
#
# License: BSD (3-clause)

from __future__ import annotations

import warnings
from typing import Any, Callable, Literal

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import ArrayLike

from ..datasets.base import (
    BaseConcatDataset,
    EEGWindowsDataset,
    RawDataset,
    WindowsDataset,
)


class _LazyDataFrame:
    """
    DataFrame-like object that lazily computes values (experimental).

    This class emulates some features of a pandas DataFrame, but computes
    the values on-the-fly when they are accessed. This is useful for
    very long DataFrames with repetitive values.
    Only the methods used by EEGWindowsDataset on its metadata are implemented.

    Parameters:
    -----------
    length: int
        The length of the dataframe.
    functions: dict[str, Callable[[int], Any]]
        A dictionary mapping column names to functions that take an index and
        return the value of the column at that index.
    columns: list[str]
        The names of the columns in the dataframe.
    series: bool
        Whether the object should emulate a series or a dataframe.
    """

    def __init__(
        self,
        length: int,
        functions: dict[str, Callable[[int], Any]],
        columns: list[str],
        series: bool = False,
    ):
        if not (isinstance(length, int) and length >= 0):
            raise ValueError("Length must be a positive integer.")
        if not all(c in functions for c in columns):
            raise ValueError("All columns must have a corresponding function.")
        if series and len(columns) != 1:
            raise ValueError("Series must have exactly one column.")
        self.length = length
        self.functions = functions
        self.columns = columns
        self.series = series

    @property
    def loc(self):
        return self

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key, self.columns)
        if len(key) == 1:
            key = (key[0], self.columns)
        if not len(key) == 2:
            raise IndexError(
                f"index must be either [row] or [row, column], got [{', '.join(map(str, key))}]."
            )
        row, col = key
        if col == slice(None):  # all columns (i.e., call to df[row, :])
            col = self.columns
        one_col = False
        if isinstance(col, str):  # one column
            one_col = True
            col = [col]
        else:  # multiple columns
            col = list(col)
        if not all(c in self.columns for c in col):
            raise IndexError(
                f"All columns must be present in the dataframe with columns {self.columns}. Got {col}."
            )
        if row == slice(None):  # all rows (i.e., call to df[:] or df[:, col])
            return _LazyDataFrame(self.length, self.functions, col)
        if not isinstance(row, int):
            raise NotImplementedError(
                "Row indexing only supports either a single integer or a null slice (i.e., df[:])."
            )
        if not (0 <= row < self.length):
            raise IndexError(f"Row index {row} is out of bounds.")
        if self.series or one_col:
            return self.functions[col[0]](row)
        return pd.Series({c: self.functions[c](row) for c in col})

    def to_numpy(self):
        return _LazyDataFrame(
            length=self.length,
            functions=self.functions,
            columns=self.columns,
            series=len(self.columns) == 1,
        )

    def to_list(self):
        return self.to_numpy()


class _FixedLengthWindowFunctions:
    """Class defining functions for lazy metadata generation in fixed length windowing
    to be used in combination with _LazyDataFrame (experimental)."""

    def __init__(
        self,
        start_offset_samples: int,
        last_potential_start: int,
        window_stride_samples: int,
        window_size_samples: int,
        target: Any,
    ):
        self.start_offset_samples = start_offset_samples
        self.last_potential_start = last_potential_start
        self.window_stride_samples = window_stride_samples
        self.window_size_samples = window_size_samples
        self.target_val = target

    @property
    def length(self) -> int:
        return int(
            np.ceil(
                (self.last_potential_start + 1 - self.start_offset_samples)
                / self.window_stride_samples
            )
        )

    def i_window_in_trial(self, i: int) -> int:
        return i

    def i_start_in_trial(self, i: int) -> int:
        return self.start_offset_samples + i * self.window_stride_samples

    def i_stop_in_trial(self, i: int) -> int:
        return (
            self.start_offset_samples
            + i * self.window_stride_samples
            + self.window_size_samples
        )

    def target(self, i: int) -> Any:
        return self.target_val


def _get_use_mne_epochs(use_mne_epochs, reject, picks, flat, drop_bad_windows):
    should_use_mne_epochs = (
        (reject is not None)
        or (picks is not None)
        or (flat is not None)
        or (drop_bad_windows is True)
    )
    if use_mne_epochs is None:
        if should_use_mne_epochs:
            warnings.warn(
                "Using reject or picks or flat or dropping bad windows means "
                "mne Epochs are created, "
                "which will be substantially slower and may be deprecated in the future."
            )
        return should_use_mne_epochs
    if not use_mne_epochs and should_use_mne_epochs:
        raise ValueError(
            "Cannot set use_mne_epochs=False when using reject, picks, flat, or dropping bad windows."
        )
    return use_mne_epochs


def _normalize_on_missing(on_missing: str) -> str:
    """Translate Braindecode's public spellings to the MNE vocabulary."""
    aliases = {
        "error": "raise",
        "warning": "warn",
        "ignore": "ignore",
        "raise": "raise",
        "warn": "warn",
    }
    try:
        return aliases[on_missing]
    except (KeyError, TypeError):
        raise ValueError(
            "on_missing must be one of 'error', 'warning', 'ignore', "
            f"'raise', or 'warn', got {on_missing!r}."
        ) from None


def _validate_on_last_window(on_last_window: str | None) -> None:
    """Validate a trailing-window strategy shared by public windowers."""
    if on_last_window not in (None, "overlap", "drop", "keep"):
        raise ValueError(
            "on_last_window must be one of 'overlap', 'drop', 'keep', "
            f"got {on_last_window!r}."
        )


# XXX it's called concat_ds...
def create_windows_from_events(
    concat_ds: BaseConcatDataset[RawDataset],
    trial_start_offset_samples: int | dict[str, int] = 0,
    trial_stop_offset_samples: int | dict[str, int] = 0,
    window_size_samples: int | None = None,
    window_stride_samples: int | dict[str, int] | None = None,
    drop_last_window: bool | None = None,
    mapping: dict[str, int] | None = None,
    preload: bool = False,
    drop_bad_windows: bool | None = None,
    picks: str | ArrayLike | slice | None = None,
    reject: dict[str, float] | None = None,
    flat: dict[str, float] | None = None,
    on_missing: str = "error",
    accepted_bads_ratio: float = 0.0,
    use_mne_epochs: bool | None = None,
    on_overlapping_events: Literal["raise", "warn", "ignore"] = "raise",
    n_jobs: int = 1,
    verbose: bool | str | int | None = "error",
    *,
    on_last_window: Literal["overlap", "drop", "keep"] | None = None,
) -> BaseConcatDataset[WindowsDataset | EEGWindowsDataset]:
    """Create windows based on events in mne.Raw.

    This function extracts windows of size window_size_samples in the interval
    [trial onset + trial_start_offset_samples, trial onset + trial duration +
    trial_stop_offset_samples] around each trial, with a separation of
    window_stride_samples between consecutive windows. ``on_last_window``
    controls how a trailing incomplete window is handled.

    Windows are extracted from the interval defined by the following::

                                                trial onset +
                        trial onset                duration
        |--------------------|------------------------|-----------------------|
        trial onset -                                             trial onset +
        trial_start_offset_samples                                   duration +
                                                    trial_stop_offset_samples

    Parameters
    ----------
    concat_ds: BaseConcatDataset[RawDataset]
        A concat of base datasets each holding raw and description.
    trial_start_offset_samples: int | dict[str, int]
        Start offset from original trial onsets, in samples. Defaults to zero.
        If a dict, keys must match the keys of ``mapping`` and different
        offsets are applied per event type.
    trial_stop_offset_samples: int | dict[str, int]
        Stop offset from original trial stop, in samples. Defaults to zero.
        If a dict, keys must match the keys of ``mapping`` and different
        offsets are applied per event type.
    window_size_samples: int | None
        Window size. If None, the window size is inferred from the original
        trial size of the first trial and trial_start_offset_samples and
        trial_stop_offset_samples.
    window_stride_samples: int | dict[str, int] | None
        Stride between windows, in samples. If None, the window stride is
        inferred from the original trial size of the first trial and
        trial_start_offset_samples and trial_stop_offset_samples.
        If a dict, keys must match the keys of ``mapping`` and different
        strides are applied per event type.
    drop_last_window: bool | None
        Deprecated and scheduled for removal in version 2.0. Use
        ``on_last_window`` instead. Passing this parameter emits a
        ``DeprecationWarning``. If True, maps to ``on_last_window='drop'``;
        if False, maps to ``on_last_window='overlap'``.
    on_last_window: {'overlap', 'drop', 'keep'} | None
        How to handle the last incomplete window when the trial duration is not
        evenly divisible by window_size_samples and window_stride_samples.

        - ``'overlap'``: create a final window flush to the trial end, which
          will overlap with the previous window. This is the default and
          preserves the original behavior of ``drop_last_window=False``.
        - ``'drop'``: discard the remainder of the trial. Equivalent to the
          original ``drop_last_window=True``.
        - ``'keep'``: retain the last incomplete window at its natural shorter
          size instead of moving its start backwards. Not compatible with
          ``use_mne_epochs=True``.
    mapping: dict(str: int)
        Mapping from event description to numerical target value. Must be
        provided when any of ``trial_start_offset_samples``,
        ``trial_stop_offset_samples``, or ``window_stride_samples`` is a dict.
    preload: bool
        If True, preload the data of the Epochs objects. This is useful to
        reduce disk reading overhead when returning windows in a training
        scenario, however very large data might not fit into memory.
    drop_bad_windows: bool
        If True, call `.drop_bad()` on the resulting mne.Epochs object. This
        step allows identifying e.g., windows that fall outside of the
        continuous recording. It is suggested to run this step here as otherwise
        the BaseConcatDataset has to be updated as well.
    picks: str | list | slice | None
        Channels to include. If None, all available channels are used. See
        mne.Epochs.
    reject: dict | None
        Epoch rejection parameters based on peak-to-peak amplitude. If None, no
        rejection is done based on peak-to-peak amplitude. See mne.Epochs.
    flat: dict | None
        Epoch rejection parameters based on flatness of signals. If None, no
        rejection based on flatness is done. See mne.Epochs.
    on_missing: str
        What to do if one or several event ids are not found in the recording.
        Valid keys are ``'error'`` | ``'warning'`` | ``'ignore'``. The MNE
        spellings ``'raise'`` and ``'warn'`` are accepted as aliases. See
        :class:`mne.Epochs`.
    accepted_bads_ratio: float, optional
        Acceptable proportion of trials with inconsistent length in a raw. If
        the proportion of trials whose length is exceeded by the window size is
        no greater than this, only the corresponding trials are dropped and the
        computation continues. Otherwise, an error is raised. Defaults to 0.0
        (raise an error). If all trials are accepted for dropping because they
        are too short, a ``ValueError`` is raised because no windows can be
        created. If one event type is completely dropped but other windows
        remain, that type is omitted from the MNE ``event_id`` and windowing
        continues. With per-event dictionary parameters, the ratio is computed
        once across all mapped trials in each raw, not separately by event type.
    use_mne_epochs: bool
        If False, return EEGWindowsDataset objects.
        If True, return mne.Epochs objects encapsulated in WindowsDataset objects,
        which is substantially slower that EEGWindowsDataset.
    on_overlapping_events: Literal['raise', 'warn', 'ignore']
        Behavior when overlapping events are detected. Valid keys are
        'raise' | 'warn' | 'ignore'. 'raise' (default) raises
        NotImplementedError; 'warn' issues a warning and drops non-increasing
        window starts; 'ignore' keeps overlapping starts.
    n_jobs: int
        Number of jobs to use to parallelize the windowing.
    verbose: bool | str | int | None
        Control verbosity of the logging output when calling mne.Epochs.

    Returns
    -------
    windows_datasets: BaseConcatDataset[WindowsDataset | EEGWindowsDataset]
        Concatenated datasets of WindowsDataset containing the extracted windows.
    """

    _check_windowing_arguments(
        trial_start_offset_samples,
        trial_stop_offset_samples,
        window_size_samples,
        window_stride_samples,
    )
    if on_overlapping_events not in ["raise", "warn", "ignore"]:
        raise ValueError(
            f"Invalid value {on_overlapping_events} for on_overlapping_events."
        )

    if drop_last_window is not None:
        if on_last_window is not None:
            raise ValueError(
                "Cannot specify both `drop_last_window` and `on_last_window`. "
                "Use `on_last_window` only, as `drop_last_window` is deprecated."
            )
        warnings.warn(
            "`drop_last_window` is deprecated and will be removed in version 2.0. "
            "Use `on_last_window='drop'` if True, `on_last_window='overlap'` if False. "
            "See https://github.com/braindecode/braindecode/pull/1058 for feedback.",
            DeprecationWarning,
            stacklevel=2,
        )
        on_last_window = "drop" if drop_last_window else "overlap"

    if on_last_window is None:
        on_last_window = "overlap"

    _validate_on_last_window(on_last_window)
    # Validate per-event-type dict parameters
    has_dict_params = any(
        isinstance(p, dict)
        for p in [
            trial_start_offset_samples,
            trial_stop_offset_samples,
            window_stride_samples,
        ]
    )
    if has_dict_params:
        if mapping is None:
            raise ValueError(
                "mapping must be provided when any of "
                "trial_start_offset_samples, trial_stop_offset_samples, "
                "or window_stride_samples is a dict."
            )
        if window_size_samples is None:
            raise ValueError(
                "window_size_samples must be provided (not None) when any of "
                "trial_start_offset_samples, trial_stop_offset_samples, "
                "or window_stride_samples is a dict."
            )
        mapping_keys = set(mapping.keys())
        for param_name, param_val in [
            ("trial_start_offset_samples", trial_start_offset_samples),
            ("trial_stop_offset_samples", trial_stop_offset_samples),
            ("window_stride_samples", window_stride_samples),
        ]:
            if isinstance(param_val, dict) and set(param_val.keys()) != mapping_keys:
                raise ValueError(
                    f"Keys of {param_name} ({set(param_val.keys())}) must "
                    f"match keys of mapping ({mapping_keys})."
                )
        # Normalize int params to dicts so downstream always gets dicts
        if not isinstance(trial_start_offset_samples, dict):
            trial_start_offset_samples = {
                k: trial_start_offset_samples for k in mapping
            }
        if not isinstance(trial_stop_offset_samples, dict):
            trial_stop_offset_samples = {k: trial_stop_offset_samples for k in mapping}
        if not isinstance(window_stride_samples, dict):
            window_stride_samples = {k: window_stride_samples for k in mapping}  # type: ignore

    # If user did not specify mapping, we extract all events from all datasets
    # and map them to increasing integers starting from 0
    infer_mapping = mapping is None
    mapping = dict() if infer_mapping else mapping
    infer_window_size_stride = window_size_samples is None

    if drop_bad_windows is not None:
        warnings.warn(
            "Drop bad windows only has an effect if mne epochs are created, "
            "and this argument may be removed in the future."
        )

    use_mne_epochs = _get_use_mne_epochs(
        use_mne_epochs, reject, picks, flat, drop_bad_windows
    )
    if on_last_window == "keep" and use_mne_epochs:
        raise ValueError(
            "on_last_window='keep' requires use_mne_epochs=False because "
            "mne.Epochs cannot contain variable-length windows."
        )
    if use_mne_epochs and drop_bad_windows is None:
        drop_bad_windows = True

    list_of_windows_ds = Parallel(n_jobs=n_jobs)(
        delayed(_create_windows_from_events)(
            ds,
            infer_mapping,
            infer_window_size_stride,
            trial_start_offset_samples,
            trial_stop_offset_samples,
            window_size_samples,
            window_stride_samples,
            on_last_window,
            mapping,
            preload,
            drop_bad_windows,
            picks,
            reject,
            flat,
            on_missing,
            accepted_bads_ratio,
            verbose,
            use_mne_epochs,
            on_overlapping_events,
        )
        for ds in concat_ds.datasets
    )
    return BaseConcatDataset(list_of_windows_ds)


def create_fixed_length_windows(
    concat_ds: BaseConcatDataset[RawDataset],
    start_offset_samples: int = 0,
    stop_offset_samples: int | None = None,
    window_size_samples: int | None = None,
    window_stride_samples: int | None = None,
    drop_last_window: bool | None = None,
    mapping: dict[str, int] | None = None,
    preload: bool = False,
    drop_bad_windows: bool | None = None,
    picks: str | ArrayLike | slice | None = None,
    reject: dict[str, float] | None = None,
    flat: dict[str, float] | None = None,
    targets_from: str = "metadata",
    last_target_only: bool = True,
    lazy_metadata: bool = False,
    on_missing: str = "error",
    use_mne_epochs: bool | None = None,
    n_jobs: int = 1,
    verbose: bool | str | int | None = "error",
    *,
    on_last_window: Literal["overlap", "drop", "keep"] | None = None,
) -> BaseConcatDataset[WindowsDataset | EEGWindowsDataset]:
    """Windower that creates sliding windows.

    Parameters
    ----------
    concat_ds: ConcatDataset[RawDataset]
        A concat of base datasets each holding raw and description.
    start_offset_samples: int
        Start offset from beginning of recording in samples.
    stop_offset_samples: int | None
        Stop offset from beginning of recording in samples. If None, set to be
        the end of the recording.
    window_size_samples: int | None
        Window size in samples. If None, set to be the maximum possible window size, ie length of
        the recording, once offsets are accounted for.
    window_stride_samples: int | None
        Stride between windows in samples. If None, set to be equal to winddow_size_samples, so
        windows will not overlap.
    drop_last_window: bool | None
        Deprecated and scheduled for removal in version 2.0. Use
        ``on_last_window`` instead. Passing this parameter emits a
        ``DeprecationWarning``. If True, maps to ``on_last_window='drop'``;
        if False, maps to ``on_last_window='overlap'``.
    on_last_window: {'overlap', 'drop', 'keep'} | None
        How to handle the last incomplete window when the recording duration is
        not evenly divisible by window_size_samples and window_stride_samples.
        Must be set if both window_size_samples and window_stride_samples are
        provided. With explicit window sizing, only ``'drop'`` is compatible
        with ``lazy_metadata=True``.

        - ``'overlap'``: create a final window flush to the recording end,
          which may overlap with the previous window.
        - ``'drop'``: discard the remainder. Equivalent to the original
          ``drop_last_window=True``.
        - ``'keep'``: retain the last incomplete window at its natural shorter
          size. Not compatible with ``lazy_metadata=True`` or
          ``use_mne_epochs=True``.
    mapping: dict(str: int)
        Mapping from event description to target value.
    preload: bool
        If True, preload the data of the Epochs objects.
    drop_bad_windows: bool | None
        If True, call `.drop_bad()` on the resulting mne.Epochs object. This
        step allows identifying e.g., windows that fall outside of the
        continuous recording. It is suggested to run this step here as otherwise
        the BaseConcatDataset has to be updated as well. Only has an effect if
        mne Epochs are created (i.e. ``use_mne_epochs=True``).
    picks: str | list | slice | None
        Channels to include. If None, all available channels are used. See
        mne.Epochs.
    reject: dict | None
        Epoch rejection parameters based on peak-to-peak amplitude. If None, no
        rejection is done based on peak-to-peak amplitude. See mne.Epochs.
    flat: dict | None
        Epoch rejection parameters based on flatness of signals. If None, no
        rejection based on flatness is done. See mne.Epochs.
    lazy_metadata: bool
        If True, metadata is not computed immediately, but only when accessed
        by using the _LazyDataFrame (experimental). With explicit window
        sizing, requires ``on_last_window='drop'``. Cannot be used together
        with ``use_mne_epochs=True``.
    on_missing: str
        What to do if one or several event ids are not found in the recording.
        Valid keys are ``'error'`` | ``'warning'`` | ``'ignore'``. The MNE
        spellings ``'raise'`` and ``'warn'`` are accepted as aliases. See
        :class:`mne.Epochs`.
    use_mne_epochs: bool | None
        If False, return EEGWindowsDataset objects.
        If True, return mne.Epochs objects encapsulated in WindowsDataset
        objects, which is substantially slower than EEGWindowsDataset.
        If None, it will be inferred from the other parameters: True if any
        of ``reject``, ``picks``, or ``flat`` is set, or if
        ``drop_bad_windows`` is True; False otherwise. If ``use_mne_epochs``
        is inferred as True and ``drop_bad_windows`` is None, it is treated
        as True.
    n_jobs: int
        Number of jobs to use to parallelize the windowing.
    verbose: bool | str | int | None
        Control verbosity of the logging output when calling mne.Epochs.

    Returns
    -------
    windows_datasets: BaseConcatDataset[WindowsDataset | EEGWindowsDataset]
        Concatenated dataset containing either WindowsDataset or
        EEGWindowsDataset objects with the extracted windows, depending on
        the value of ``use_mne_epochs``.
    """
    if on_last_window is not None and drop_last_window is not None:
        raise ValueError(
            "Cannot specify both `drop_last_window` and `on_last_window`. "
            "Use `on_last_window` only, as `drop_last_window` is deprecated."
        )
    if drop_last_window is not None:
        warnings.warn(
            "`drop_last_window` is deprecated and will be removed in version 2.0. "
            "Use `on_last_window='drop'` if True, `on_last_window='overlap'` if False. "
            "See https://github.com/braindecode/braindecode/pull/1058 for feedback.",
            DeprecationWarning,
            stacklevel=2,
        )
        on_last_window = "drop" if drop_last_window else "overlap"

    stop_offset_samples, window_stride_samples, on_last_window = (
        _check_and_set_fixed_length_window_arguments(
            start_offset_samples,
            stop_offset_samples,
            window_size_samples,
            window_stride_samples,
            on_last_window,
            lazy_metadata,
        )
    )

    if drop_bad_windows is not None:
        warnings.warn(
            "Drop bad windows only has an effect if mne epochs are created, "
            "and this argument may be removed in the future."
        )

    use_mne_epochs = _get_use_mne_epochs(
        use_mne_epochs, reject, picks, flat, drop_bad_windows
    )
    if on_last_window == "keep" and use_mne_epochs:
        raise ValueError(
            "on_last_window='keep' requires use_mne_epochs=False because "
            "mne.Epochs cannot contain variable-length windows."
        )
    if use_mne_epochs and drop_bad_windows is None:
        drop_bad_windows = True
    if use_mne_epochs and lazy_metadata:
        raise ValueError("Cannot use lazy_metadata=True with use_mne_epochs=True.")

    # check if recordings are of different lengths
    lengths = np.array([ds.raw.n_times for ds in concat_ds.datasets])
    if (np.diff(lengths) != 0).any() and window_size_samples is None:
        warnings.warn("Recordings have different lengths, they will not be batch-able!")
    if (window_size_samples is not None) and any(window_size_samples > lengths):
        raise ValueError(
            f"Window size {window_size_samples} exceeds trial duration {lengths.min()}."
        )

    list_of_windows_ds = Parallel(n_jobs=n_jobs)(
        delayed(_create_fixed_length_windows)(
            ds,
            start_offset_samples,
            stop_offset_samples,
            window_size_samples,
            window_stride_samples,
            on_last_window,
            mapping,
            preload,
            drop_bad_windows,
            picks,
            reject,
            flat,
            targets_from,
            last_target_only,
            lazy_metadata,
            on_missing,
            use_mne_epochs,
            verbose,
        )
        for ds in concat_ds.datasets
    )
    return BaseConcatDataset(list_of_windows_ds)


def _create_windows_from_events(
    ds,
    infer_mapping,
    infer_window_size_stride,
    trial_start_offset_samples,
    trial_stop_offset_samples,
    window_size_samples=None,
    window_stride_samples=None,
    on_last_window="overlap",
    mapping=None,
    preload=False,
    drop_bad_windows=True,
    picks=None,
    reject=None,
    flat=None,
    on_missing="error",
    accepted_bads_ratio=0.0,
    verbose="error",
    use_mne_epochs=False,
    on_overlapping_events: Literal["raise", "warn", "ignore"] = "raise",
):
    """Create WindowsDataset from RawDataset based on events.

    Parameters
    ----------
    ds : RawDataset
        Dataset containing continuous data and description.
    infer_mapping : bool
        If True, extract all events from all datasets and map them to
        increasing integers starting from 0.
    infer_window_size_stride : bool
        If True, infer the stride from the original trial size of the first
        trial and trial_start_offset_samples and trial_stop_offset_samples.

    See `create_windows_from_events` for description of other parameters.

    Returns
    -------
    EEGWindowsDataset :
        Windowed dataset.
    """
    # catch window_kwargs to store to dataset
    window_kwargs = [
        (create_windows_from_events.__name__, _get_windowing_kwargs(locals())),
    ]
    if infer_mapping:
        unique_events = np.unique(ds.raw.annotations.description)
        new_unique_events = [x for x in unique_events if x not in mapping]
        # mapping event descriptions to integers from 0 on
        max_id_existing_mapping = len(mapping)
        mapping.update(
            {
                event_name: i_event_type + max_id_existing_mapping
                for i_event_type, event_name in enumerate(new_unique_events)
            }
        )

    events, events_id = mne.events_from_annotations(ds.raw, mapping, verbose=verbose)
    onsets = events[:, 0]
    ann = ds.raw.annotations
    # Onsets are relative to the beginning of the recording
    filtered_durations = np.array(
        [a["duration"] for a in ann if a["description"] in events_id]
    )

    extras = None
    if hasattr(ann, "extras"):
        extras = [a["extras"] for a in ann if a["description"] in events_id]
        if not any(extras):
            extras = None

    stops = onsets + (filtered_durations * ds.raw.info["sfreq"]).astype(int)
    # XXX This could probably be simplified by using chunk_duration in
    #     `events_from_annotations`

    description = events[:, -1]
    event_names_by_code = {
        event_code: event_name for event_name, event_code in events_id.items()
    }
    if isinstance(trial_stop_offset_samples, dict):
        event_names = [event_names_by_code[event_code] for event_code in description]
        stop_offsets = np.array(
            [trial_stop_offset_samples[event_name] for event_name in event_names]
        )
    else:
        stop_offsets = trial_stop_offset_samples

    last_samp = ds.raw.first_samp + ds.raw.n_times - 1
    # `stops` is used exclusively (i.e. `start:stop`), so add back 1
    overflowing_trials = np.flatnonzero(stops + stop_offsets > last_samp + 1)
    if len(overflowing_trials) > 0:
        i_trial = overflowing_trials[-1]
        if isinstance(trial_stop_offset_samples, dict):
            event_name = event_names[i_trial]
            raise ValueError(
                '"trial_stop_offset_samples" too large. Stop of trial '
                f'{i_trial} ({stops[i_trial]}) + "trial_stop_offset_samples" '
                f"for event {event_name!r} ({stop_offsets[i_trial]}) must be "
                "smaller than length of"
                f" recording ({len(ds)})."
            )
        trial_label = "last trial" if i_trial == len(stops) - 1 else f"trial {i_trial}"
        raise ValueError(
            f'"trial_stop_offset_samples" too large. Stop of {trial_label} '
            f'({stops[i_trial]}) + "trial_stop_offset_samples" '
            f"({trial_stop_offset_samples}) must be smaller than length of"
            f" recording ({len(ds)})."
        )

    if isinstance(trial_start_offset_samples, dict):
        # Per-event-type windowing: skip inference, group by event type
        start_offsets = np.array(
            [
                trial_start_offset_samples[event_names_by_code[event_code]]
                for event_code in description
            ]
        )
        bads_mask = _check_bad_trial_ratio(
            (stops + stop_offsets) - (onsets + start_offsets),
            window_size_samples,
            accepted_bads_ratio,
        )

        if not use_mne_epochs:
            onsets = onsets - ds.raw.first_samp
            stops = stops - ds.raw.first_samp

        all_i_trials = []
        all_i_window_in_trials = []
        all_starts: list[int] = []
        all_stops: list[int] = []

        for event_name, event_code in events_id.items():
            mask = (description == event_code) & ~bads_mask
            if not np.any(mask):
                continue
            type_onsets = onsets[mask]
            type_stops = stops[mask]
            orig_indices = np.where(mask)[0]

            start_off = trial_start_offset_samples[event_name]
            stop_off = trial_stop_offset_samples[event_name]
            stride = window_stride_samples[event_name]

            type_i_trials, type_i_win, type_starts, type_stops = _compute_window_inds(
                type_onsets.copy(),
                type_stops.copy(),
                start_off,
                stop_off,
                window_size_samples,
                stride,
                on_last_window,
                0.0,
            )
            # Map local trial indices back to global event indices.
            mapped_i_trials = [orig_indices[i] for i in type_i_trials]
            all_i_trials.extend(mapped_i_trials)
            all_i_window_in_trials.extend(type_i_win)
            all_starts.extend(
                type_starts if isinstance(type_starts, list) else type_starts.tolist()
            )
            all_stops.extend(
                type_stops if isinstance(type_stops, list) else type_stops.tolist()
            )

        # Sort chronologically by start sample
        sort_order = np.argsort(all_starts)
        i_trials = [all_i_trials[i] for i in sort_order]
        i_window_in_trials = [all_i_window_in_trials[i] for i in sort_order]
        starts = [all_starts[i] for i in sort_order]
        stops = np.array([all_stops[i] for i in sort_order])
    else:
        if infer_window_size_stride:
            # window size is trial size
            if window_size_samples is None:
                window_size_samples = (
                    stops[0]
                    + trial_stop_offset_samples
                    - (onsets[0] + trial_start_offset_samples)
                )
                window_stride_samples = window_size_samples
            this_trial_sizes = (stops + trial_stop_offset_samples) - (
                onsets + trial_start_offset_samples
            )
            # Maybe actually this is not necessary?
            # We could also just say we just assume window size=trial size
            # in case not given, without this condition...
            # but then would have to change functions overall
            checker_trials_size = this_trial_sizes == window_size_samples

            if not np.all(checker_trials_size):
                trials_drops = int(len(this_trial_sizes) - sum(checker_trials_size))
                warnings.warn(
                    f"Dropping trials with different windows size {trials_drops}",
                )
                events = events[checker_trials_size]
                onsets = onsets[checker_trials_size]
                stops = stops[checker_trials_size]
                if extras is not None:
                    extras = [e for i, e in enumerate(extras) if checker_trials_size[i]]
        if not use_mne_epochs:
            onsets = onsets - ds.raw.first_samp
            stops = stops - ds.raw.first_samp
        i_trials, i_window_in_trials, starts, stops = _compute_window_inds(
            onsets,
            stops,
            trial_start_offset_samples,
            trial_stop_offset_samples,
            window_size_samples,
            window_stride_samples,
            on_last_window,
            accepted_bads_ratio,
        )

    if len(starts) == 0:
        raise ValueError(
            "No windows can be created because all trials are shorter than "
            "window_size_samples after applying the trial offsets."
        )

    if (on_overlapping_events != "ignore") and any(np.diff(starts) <= 0):
        msg = "Overlapping trials detected. You can ignore, warn, or raise an error, using the on_overlapping_events argument."
        if on_overlapping_events == "raise":
            raise ValueError(msg)
        if on_overlapping_events == "warn":
            warnings.warn(msg)

    events = [
        [start, window_size_samples, description[i_trials[i_start]]]
        for i_start, start in enumerate(starts)
    ]
    events = np.array(events)

    description = events[:, -1]

    if extras is not None:
        extras = [extras[i_trials[i_start]] for i_start in range(len(starts))]

    metadata = pd.DataFrame(
        {
            "i_window_in_trial": i_window_in_trials,
            "i_start_in_trial": starts,
            "i_stop_in_trial": stops,
            "target": description,
        }
    )
    if extras is not None:
        extras_df = pd.DataFrame(extras)
        if forbidden_cols := set(metadata.columns).intersection(extras_df.columns):
            warnings.warn(
                f"Dropping extra columns that conflict with windowing metadata: {forbidden_cols}"
            )
            extras_df = extras_df.drop(columns=forbidden_cols)
        metadata = pd.concat([metadata, extras_df.reset_index(drop=True)], axis=1)

    if use_mne_epochs:
        surviving_event_codes = set(description)
        events_id = {
            event_name: event_code
            for event_name, event_code in events_id.items()
            if event_code in surviving_event_codes
        }
        # window size - 1, since tmax is inclusive
        mne_epochs = mne.Epochs(
            ds.raw,
            events,
            events_id,
            baseline=None,
            tmin=0,
            tmax=(window_size_samples - 1) / ds.raw.info["sfreq"],
            metadata=metadata,
            preload=preload,
            picks=picks,
            reject=reject,
            flat=flat,
            on_missing=_normalize_on_missing(on_missing),
            verbose=verbose,
        )
        if drop_bad_windows:
            mne_epochs.drop_bad()
        windows_ds = WindowsDataset(
            mne_epochs,
            ds.description,
        )
    else:
        windows_ds = EEGWindowsDataset(
            ds.raw,
            metadata=metadata,
            description=ds.description,
        )
    # add window_kwargs and raw_preproc_kwargs to windows dataset
    setattr(windows_ds, "window_kwargs", window_kwargs)
    kwargs_name = "raw_preproc_kwargs"
    if hasattr(ds, kwargs_name):
        setattr(windows_ds, kwargs_name, getattr(ds, kwargs_name))
    return windows_ds


def _create_fixed_length_windows(
    ds,
    start_offset_samples,
    stop_offset_samples,
    window_size_samples,
    window_stride_samples,
    on_last_window,
    mapping=None,
    preload=False,
    drop_bad_windows=True,
    picks=None,
    reject=None,
    flat=None,
    targets_from="metadata",
    last_target_only=True,
    lazy_metadata=False,
    on_missing="error",
    use_mne_epochs=False,
    verbose="error",
):
    """Create WindowsDataset from RawDataset with sliding windows.

    Parameters
    ----------
    ds : RawDataset
        Dataset containing continuous data and description.

    See `create_fixed_length_windows` for description of other parameters.

    Returns
    -------
    WindowsDataset :
        Windowed dataset.
    """
    # catch window_kwargs to store to dataset
    window_kwargs = [
        (create_fixed_length_windows.__name__, _get_windowing_kwargs(locals())),
    ]
    stop = ds.raw.n_times if stop_offset_samples is None else stop_offset_samples

    # assume window should be whole recording
    if window_size_samples is None:
        window_size_samples = stop - start_offset_samples
    if window_stride_samples is None:
        window_stride_samples = window_size_samples

    last_potential_start = stop - window_size_samples

    # get targets from dataset description if they exist
    target = -1 if ds.target_name is None else ds.description[ds.target_name]
    if mapping is not None:
        # in case of multiple targets
        if isinstance(target, pd.Series):
            # Plain comprehension instead of Series.replace(mapping):
            # replace() emits a pandas FutureWarning about silent downcasting
            # and the result is immediately list-ified anyway.
            target = [mapping.get(v, v) for v in target]
        # in case of single value target
        else:
            target = mapping[target]

    if lazy_metadata:
        factory = _FixedLengthWindowFunctions(
            start_offset_samples,
            last_potential_start,
            window_stride_samples,
            window_size_samples,
            target,
        )
        metadata = _LazyDataFrame(
            length=factory.length,
            functions={
                "i_window_in_trial": factory.i_window_in_trial,
                "i_start_in_trial": factory.i_start_in_trial,
                "i_stop_in_trial": factory.i_stop_in_trial,
                "target": factory.target,
            },
            columns=[
                "i_window_in_trial",
                "i_start_in_trial",
                "i_stop_in_trial",
                "target",
            ],
        )
    else:
        # already includes last incomplete window start
        starts = np.arange(
            start_offset_samples, last_potential_start + 1, window_stride_samples
        )

        if len(starts) == 0:
            raise ValueError(
                "No windows can be created: window_size_samples is larger than the available samples after applying offsets."
            )

        if on_last_window == "overlap" and starts[-1] < last_potential_start:
            starts = np.append(starts, last_potential_start)
        elif on_last_window == "keep" and starts[-1] < last_potential_start:
            # the true last incomplete window starts right after last full window
            last_incomplete_start = starts[-1] + window_stride_samples
            if last_incomplete_start < stop:
                starts = np.append(starts, last_incomplete_start)

        stop_values = starts + window_size_samples
        if on_last_window == "keep" and len(starts) > 0:
            stop_values[-1] = min(stop_values[-1], stop)

        metadata = pd.DataFrame(
            {
                "i_window_in_trial": np.arange(len(starts)),
                "i_start_in_trial": starts,
                "i_stop_in_trial": stop_values,
                "target": len(starts) * [target],
            }
        )

    if use_mne_epochs:
        # Construct synthetic events for mne.Epochs
        events = np.column_stack(
            [
                starts + ds.raw.first_samp,
                np.zeros(len(starts), dtype=int),
                np.ones(len(starts), dtype=int),
            ]
        )
        events_id = {"window": 1}
        # window size - 1, since tmax is inclusive
        mne_epochs = mne.Epochs(
            ds.raw,
            events,
            events_id,
            baseline=None,
            tmin=0,
            tmax=(window_size_samples - 1) / ds.raw.info["sfreq"],
            metadata=metadata,
            preload=preload,
            picks=picks,
            reject=reject,
            flat=flat,
            on_missing=_normalize_on_missing(on_missing),
            verbose=verbose,
        )
        if drop_bad_windows:
            mne_epochs.drop_bad()
        windows_ds = WindowsDataset(
            mne_epochs,
            ds.description,
        )
    else:
        window_kwargs.append(
            (
                EEGWindowsDataset.__name__,
                {"targets_from": targets_from, "last_target_only": last_target_only},
            )
        )
        windows_ds = EEGWindowsDataset(
            ds.raw,
            metadata=metadata,
            description=ds.description,
            targets_from=targets_from,
            last_target_only=last_target_only,
        )
    # add window_kwargs and raw_preproc_kwargs to windows dataset
    setattr(windows_ds, "window_kwargs", window_kwargs)
    kwargs_name = "raw_preproc_kwargs"
    if hasattr(ds, kwargs_name):
        setattr(windows_ds, kwargs_name, getattr(ds, kwargs_name))
    return windows_ds


def create_windows_from_target_channels(
    concat_ds: BaseConcatDataset[RawDataset],
    window_size_samples=None,
    preload=False,
    picks=None,
    reject=None,
    flat=None,
    n_jobs=1,
    last_target_only=True,
    verbose="error",
) -> BaseConcatDataset[EEGWindowsDataset]:
    list_of_windows_ds = Parallel(n_jobs=n_jobs)(
        delayed(_create_windows_from_target_channels)(
            ds,
            window_size_samples,
            preload,
            picks,
            reject,
            flat,
            last_target_only,
            "error",
            verbose,
        )
        for ds in concat_ds.datasets
    )
    return BaseConcatDataset(list_of_windows_ds)


def _create_windows_from_target_channels(
    ds,
    window_size_samples,
    preload=False,
    picks=None,
    reject=None,
    flat=None,
    last_target_only=True,
    on_missing="error",
    verbose="error",
):
    """Create WindowsDataset from RawDataset using targets `misc` channels from mne.Raw.

    Parameters
    ----------
    ds : RawDataset
        Dataset containing continuous data and description.

    See `create_fixed_length_windows` for description of other parameters.

    Returns
    -------
    WindowsDataset :
        Windowed dataset.
    """
    window_kwargs = [
        (create_windows_from_target_channels.__name__, _get_windowing_kwargs(locals())),
    ]
    stop = ds.raw.n_times + ds.raw.first_samp

    target = ds.raw.get_data(picks="misc")

    # check all misc channels for valid targets, not just the first one.
    # when multiple target channels exist, some may have values at timepoints
    # where others are nan. using any() across channels catches all of them.
    has_target = np.any(~np.isnan(target), axis=0)
    stops = np.nonzero(has_target)[0] + 1
    stops = stops[(stops < stop) & (stops >= window_size_samples)]
    stops = stops.astype(int)
    metadata = pd.DataFrame(
        {
            "i_window_in_trial": np.arange(len(stops)),
            "i_start_in_trial": stops - window_size_samples,
            "i_stop_in_trial": stops,
            "target": len(stops) * [target],
        }
    )

    targets_from = "channels"
    window_kwargs.append(
        (
            EEGWindowsDataset.__name__,
            {"targets_from": targets_from, "last_target_only": last_target_only},
        )
    )
    windows_ds = EEGWindowsDataset(
        ds.raw,
        metadata=metadata,
        description=ds.description,
        targets_from=targets_from,
        last_target_only=last_target_only,
    )
    setattr(windows_ds, "window_kwargs", window_kwargs)
    kwargs_name = "raw_preproc_kwargs"
    if hasattr(ds, kwargs_name):
        setattr(windows_ds, kwargs_name, getattr(ds, kwargs_name))
    return windows_ds


def _compute_window_inds(
    starts,
    stops,
    start_offset,
    stop_offset,
    size,
    stride,
    on_last_window,
    accepted_bads_ratio,
):
    """Compute window start and stop indices.

    Create window starts from trial onsets (shifted by start_offset) to trial
    end (shifted by stop_offset) separated by stride, as long as window size
    fits into trial.

    Parameters
    ----------
    starts: array-like
        Trial starts in samples.
    stops: array-like
        Trial stops in samples.
    start_offset: int
        Start offset from original trial onsets in samples.
    stop_offset: int
        Stop offset from original trial stop in samples.
    size: int
        Window size.
    stride: int
        Stride between windows.
    on_last_window: str
        How to handle the last incomplete window. One of 'overlap' (create an
        additional window flush to the trial end), 'drop' (discard the remainder),
        or 'keep' (keep the shorter window as-is).
    accepted_bads_ratio: float
        Acceptable proportion of bad trials within a raw. If the proportion of
        trials whose length is exceeded by the window size is no greater than
        this, only the corresponding trials are dropped and the computation
        continues. Otherwise, an error is raised.

    Returns
    -------
    result_lists: (list, list, list, list)
        Trial, i_window_in_trial, start sample and stop sample of windows.
    """
    starts = np.array([starts]) if isinstance(starts, int) else starts
    stops = np.array([stops]) if isinstance(stops, int) else stops
    source_trial_indices = np.arange(len(starts))

    starts += start_offset
    stops += stop_offset
    bads_mask = _check_bad_trial_ratio(
        stops - starts,
        size,
        accepted_bads_ratio,
    )
    starts = starts[~bads_mask]
    stops = stops[~bads_mask]
    source_trial_indices = source_trial_indices[~bads_mask]

    i_window_in_trials, i_trials, window_starts, window_stops = [], [], [], []
    for i_trial, start, stop in zip(source_trial_indices, starts, stops):
        trial_starts = np.arange(start, stop - size + 1, stride).tolist()

        if on_last_window == "overlap" and trial_starts[-1] + size != stop:
            trial_starts.append(stop - size)
        elif on_last_window == "keep" and trial_starts[-1] + size != stop:
            next_start = trial_starts[-1] + stride
            if next_start < stop:
                trial_starts.append(next_start)

        i_window_in_trials.extend(range(len(trial_starts)))
        i_trials.extend([i_trial] * len(trial_starts))
        window_starts.extend(trial_starts)
        window_stops.extend(
            min(window_start + size, stop) for window_start in trial_starts
        )

    if not (len(i_window_in_trials) == len(window_starts) == len(window_stops)):
        raise ValueError(
            f"{len(i_window_in_trials)} == {len(window_starts)} == {len(window_stops)}"
        )

    return i_trials, i_window_in_trials, window_starts, window_stops


def _check_bad_trial_ratio(durations, size, accepted_bads_ratio):
    """Validate short trials against the whole-raw acceptance ratio."""
    bads_mask = size > durations
    if not np.any(bads_mask):
        return bads_mask

    min_duration = durations.min()
    n_bad_trials = np.count_nonzero(bads_mask)
    current_ratio = n_bad_trials / len(durations)
    if current_ratio <= accepted_bads_ratio:
        warnings.warn(
            f"Trials {np.where(bads_mask)[0]} are being dropped as the "
            f"window size ({size}) exceeds their duration {min_duration}."
        )
        return bads_mask

    raise ValueError(
        f"Window size {size} exceeds trial duration "
        f"({min_duration}) for too many trials "
        f"({current_ratio * 100}%). Set "
        f"accepted_bads_ratio to at least {current_ratio} "
        "and restart training to be able to continue."
    )


def _check_windowing_arguments(
    trial_start_offset_samples,
    trial_stop_offset_samples,
    window_size_samples,
    window_stride_samples,
):
    def _is_int_or_none(v, allow_none=False):
        if allow_none and v is None:
            return True
        return isinstance(v, (int, np.integer))

    def _is_int_or_dict(v, allow_none=False):
        if isinstance(v, dict):
            if not all(_is_int_or_none(val) for val in v.values()):
                raise ValueError(f"All values in dict must be integers, got {v}.")
            return True
        return _is_int_or_none(v, allow_none=allow_none)

    assert _is_int_or_dict(trial_start_offset_samples), (
        "trial_start_offset_samples must be an int or a dict[str, int]"
    )
    assert _is_int_or_dict(trial_stop_offset_samples, allow_none=True), (
        "trial_stop_offset_samples must be an int, None, or a dict[str, int]"
    )
    assert isinstance(window_size_samples, (int, np.integer, type(None)))

    assert _is_int_or_dict(window_stride_samples, allow_none=True), (
        "window_stride_samples must be an int, None, or a dict[str, int]"
    )

    # When stride is a dict, window_size_samples must be provided
    stride_is_none = (
        window_stride_samples is None
        if not isinstance(window_stride_samples, dict)
        else False
    )
    assert (window_size_samples is None) == stride_is_none, (
        "window_size_samples and window_stride_samples must both be None or both be set"
    )

    if window_size_samples is not None:
        assert window_size_samples > 0, "window size has to be larger than 0"
        if isinstance(window_stride_samples, dict):
            assert all(v > 0 for v in window_stride_samples.values()), (
                "all window stride values have to be larger than 0"
            )
        else:
            assert window_stride_samples > 0, "window stride has to be larger than 0"


def _check_and_set_fixed_length_window_arguments(
    start_offset_samples,
    stop_offset_samples,
    window_size_samples,
    window_stride_samples,
    on_last_window,
    lazy_metadata,
):
    """Validate arguments and fill defaults for fixed-length windowing."""

    if window_size_samples is not None and window_stride_samples is None:
        window_stride_samples = window_size_samples
        if on_last_window is None:
            on_last_window = "drop"

    _check_windowing_arguments(
        start_offset_samples,
        stop_offset_samples,
        window_size_samples,
        window_stride_samples,
    )

    if stop_offset_samples == 0:
        warnings.warn(
            "Meaning of `trial_stop_offset_samples`=0 has changed, use `None` "
            "to indicate end of trial/recording. Using `None`."
        )
        stop_offset_samples = None

    if start_offset_samples != 0 or stop_offset_samples is not None:
        warnings.warn(
            "Usage of offset_sample args in create_fixed_length_windows is deprecated and"
            " will be removed in future versions. Please use "
            'braindecode.preprocessing.preprocess.Preprocessor("crop", tmin, tmax)'
            " instead."
        )

    _validate_on_last_window(on_last_window)

    if (
        window_size_samples is not None
        and window_stride_samples is not None
        and on_last_window is None
    ):
        raise ValueError(
            "on_last_window must be set if both window_size_samples & "
            "window_stride_samples have also been set. "
            "Use 'drop', 'overlap', or 'keep'."
        )
    elif (
        window_size_samples is None
        and window_stride_samples is None
        and on_last_window is not None
    ):
        on_last_window = None

    if on_last_window in ("keep", "overlap") and lazy_metadata:
        raise ValueError(
            f"Cannot use on_last_window={on_last_window!r} with lazy_metadata=True. "
            "Only on_last_window='drop' is supported with lazy_metadata."
        )

    assert (
        (window_size_samples is None)
        == (window_stride_samples is None)
        == (on_last_window is None)
    )

    return stop_offset_samples, window_stride_samples, on_last_window


def _get_windowing_kwargs(windowing_func_locals):
    input_kwargs = windowing_func_locals
    input_kwargs.pop("ds")
    windowing_kwargs = {k: v for k, v in input_kwargs.items()}
    return windowing_kwargs
