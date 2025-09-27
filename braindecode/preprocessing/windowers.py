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
#
# License: BSD (3-clause)

from __future__ import annotations

import warnings
from typing import Any, Callable

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import ArrayLike

from ..datasets.base import BaseConcatDataset, EEGWindowsDataset, WindowsDataset


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


# XXX it's called concat_ds...
def create_windows_from_events(
    concat_ds: BaseConcatDataset,
    trial_start_offset_samples: int = 0,
    trial_stop_offset_samples: int = 0,
    window_size_samples: int | None = None,
    window_stride_samples: int | None = None,
    drop_last_window: bool = False,
    mapping: dict[str, int] | None = None,
    preload: bool = False,
    drop_bad_windows: bool | None = None,
    picks: str | ArrayLike | slice | None = None,
    reject: dict[str, float] | None = None,
    flat: dict[str, float] | None = None,
    on_missing: str = "error",
    accepted_bads_ratio: float = 0.0,
    use_mne_epochs: bool | None = None,
    n_jobs: int = 1,
    verbose: bool | str | int | None = "error",
):
    """Create windows based on events in mne.Raw.

    This function extracts windows of size window_size_samples in the interval
    [trial onset + trial_start_offset_samples, trial onset + trial duration +
    trial_stop_offset_samples] around each trial, with a separation of
    window_stride_samples between consecutive windows. If the last window
    around an event does not end at trial_stop_offset_samples and
    drop_last_window is set to False, an additional overlapping window that
    ends at trial_stop_offset_samples is created.

    Windows are extracted from the interval defined by the following::

                                                trial onset +
                        trial onset                duration
        |--------------------|------------------------|-----------------------|
        trial onset -                                             trial onset +
        trial_start_offset_samples                                   duration +
                                                    trial_stop_offset_samples

    Parameters
    ----------
    concat_ds: BaseConcatDataset
        A concat of base datasets each holding raw and description.
    trial_start_offset_samples: int
        Start offset from original trial onsets, in samples. Defaults to zero.
    trial_stop_offset_samples: int
        Stop offset from original trial stop, in samples. Defaults to zero.
    window_size_samples: int | None
        Window size. If None, the window size is inferred from the original
        trial size of the first trial and trial_start_offset_samples and
        trial_stop_offset_samples.
    window_stride_samples: int | None
        Stride between windows, in samples. If None, the window stride is
        inferred from the original trial size of the first trial and
        trial_start_offset_samples and trial_stop_offset_samples.
    drop_last_window: bool
        If False, an additional overlapping window that ends at
        trial_stop_offset_samples will be extracted around each event when the
        last window does not end exactly at trial_stop_offset_samples.
    mapping: dict(str: int)
        Mapping from event description to numerical target value.
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
        Valid keys are ‘error’ | ‘warning’ | ‘ignore’. See mne.Epochs.
    accepted_bads_ratio: float, optional
        Acceptable proportion of trials with inconsistent length in a raw. If
        the number of trials whose length is exceeded by the window size is
        smaller than this, then only the corresponding trials are dropped, but
        the computation continues. Otherwise, an error is raised. Defaults to
        0.0 (raise an error).
    use_mne_epochs: bool
        If False, return EEGWindowsDataset objects.
        If True, return mne.Epochs objects encapsulated in WindowsDataset objects,
        which is substantially slower that EEGWindowsDataset.
    n_jobs: int
        Number of jobs to use to parallelize the windowing.
    verbose: bool | str | int | None
        Control verbosity of the logging output when calling mne.Epochs.

    Returns
    -------
    windows_datasets: BaseConcatDataset
        Concatenated datasets of WindowsDataset containing the extracted windows.
    """
    _check_windowing_arguments(
        trial_start_offset_samples,
        trial_stop_offset_samples,
        window_size_samples,
        window_stride_samples,
    )

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
            drop_last_window,
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
        )
        for ds in concat_ds.datasets
    )
    return BaseConcatDataset(list_of_windows_ds)


def create_fixed_length_windows(
    concat_ds: BaseConcatDataset,
    start_offset_samples: int = 0,
    stop_offset_samples: int | None = None,
    window_size_samples: int | None = None,
    window_stride_samples: int | None = None,
    drop_last_window: bool | None = None,
    mapping: dict[str, int] | None = None,
    preload: bool = False,
    picks: str | ArrayLike | slice | None = None,
    reject: dict[str, float] | None = None,
    flat: dict[str, float] | None = None,
    targets_from: str = "metadata",
    last_target_only: bool = True,
    lazy_metadata: bool = False,
    on_missing: str = "error",
    n_jobs: int = 1,
    verbose: bool | str | int | None = "error",
):
    """Windower that creates sliding windows.

    Parameters
    ----------
    concat_ds: ConcatDataset
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
        Whether or not have a last overlapping window, when windows do not
        equally divide the continuous signal. Must be set to a bool if window size and stride are
        not None.
    mapping: dict(str: int)
        Mapping from event description to target value.
    preload: bool
        If True, preload the data of the Epochs objects.
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
        by using the _LazyDataFrame (experimental).
    on_missing: str
        What to do if one or several event ids are not found in the recording.
        Valid keys are ‘error’ | ‘warning’ | ‘ignore’. See mne.Epochs.
    n_jobs: int
        Number of jobs to use to parallelize the windowing.
    verbose: bool | str | int | None
        Control verbosity of the logging output when calling mne.Epochs.

    Returns
    -------
    windows_datasets: BaseConcatDataset
        Concatenated datasets of WindowsDataset containing the extracted windows.
    """
    stop_offset_samples, drop_last_window = (
        _check_and_set_fixed_length_window_arguments(
            start_offset_samples,
            stop_offset_samples,
            window_size_samples,
            window_stride_samples,
            drop_last_window,
            lazy_metadata,
        )
    )

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
            drop_last_window,
            mapping,
            preload,
            picks,
            reject,
            flat,
            targets_from,
            last_target_only,
            lazy_metadata,
            on_missing,
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
    drop_last_window=False,
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
):
    """Create WindowsDataset from BaseDataset based on events.

    Parameters
    ----------
    ds : BaseDataset
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

    events, events_id = mne.events_from_annotations(ds.raw, mapping)
    onsets = events[:, 0]
    # Onsets are relative to the beginning of the recording
    filtered_durations = np.array(
        [a["duration"] for a in ds.raw.annotations if a["description"] in events_id]
    )

    stops = onsets + (filtered_durations * ds.raw.info["sfreq"]).astype(int)
    # XXX This could probably be simplified by using chunk_duration in
    #     `events_from_annotations`

    last_samp = ds.raw.first_samp + ds.raw.n_times - 1
    # `stops` is used exclusively (i.e. `start:stop`), so add back 1
    if stops[-1] + trial_stop_offset_samples > last_samp + 1:
        raise ValueError(
            '"trial_stop_offset_samples" too large. Stop of last trial '
            f'({stops[-1]}) + "trial_stop_offset_samples" '
            f"({trial_stop_offset_samples}) must be smaller than length of"
            f" recording ({len(ds)})."
        )

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
            bads_size_trials = checker_trials_size
            events = events[checker_trials_size]
            onsets = onsets[checker_trials_size]
            stops = stops[checker_trials_size]

    description = events[:, -1]

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
        drop_last_window,
        accepted_bads_ratio,
    )

    if any(np.diff(starts) <= 0):
        raise NotImplementedError("Trial overlap not implemented.")

    events = [
        [start, window_size_samples, description[i_trials[i_start]]]
        for i_start, start in enumerate(starts)
    ]
    events = np.array(events)

    description = events[:, -1]

    metadata = pd.DataFrame(
        {
            "i_window_in_trial": i_window_in_trials,
            "i_start_in_trial": starts,
            "i_stop_in_trial": stops,
            "target": description,
        }
    )
    if use_mne_epochs:
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
            on_missing=on_missing,
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
    drop_last_window,
    mapping=None,
    preload=False,
    picks=None,
    reject=None,
    flat=None,
    targets_from="metadata",
    last_target_only=True,
    lazy_metadata=False,
    on_missing="error",
    verbose="error",
):
    """Create WindowsDataset from BaseDataset with sliding windows.

    Parameters
    ----------
    ds : BaseDataset
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
            target = target.replace(mapping).to_list()
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

        if not drop_last_window and starts[-1] < last_potential_start:
            # if last window does not end at trial stop, make it stop there
            starts = np.append(starts, last_potential_start)

        metadata = pd.DataFrame(
            {
                "i_window_in_trial": np.arange(len(starts)),
                "i_start_in_trial": starts,
                "i_stop_in_trial": starts + window_size_samples,
                "target": len(starts) * [target],
            }
        )

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
    concat_ds,
    window_size_samples=None,
    preload=False,
    picks=None,
    reject=None,
    flat=None,
    n_jobs=1,
    last_target_only=True,
    verbose="error",
):
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
    """Create WindowsDataset from BaseDataset using targets `misc` channels from mne.Raw.

    Parameters
    ----------
    ds : BaseDataset
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
    # TODO: handle multi targets present only for some events
    stops = np.nonzero((~np.isnan(target[0, :])))[0] + 1
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
    drop_last_window,
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
    drop_last_window: bool
        Toggles of shifting last window within range or dropping last samples.
    accepted_bads_ratio: float
        Acceptable proportion of bad trials within a raw. If the number of
        trials whose length is exceeded by the window size is smaller than
        this, then only the corresponding trials are dropped, but the
        computation continues. Otherwise, an error is raised.

    Returns
    -------
    result_lists: (list, list, list, list)
        Trial, i_window_in_trial, start sample and stop sample of windows.
    """
    starts = np.array([starts]) if isinstance(starts, int) else starts
    stops = np.array([stops]) if isinstance(stops, int) else stops

    starts += start_offset
    stops += stop_offset
    if any(size > (stops - starts)):
        bads_mask = size > (stops - starts)
        min_duration = (stops - starts).min()
        if sum(bads_mask) <= accepted_bads_ratio * len(starts):
            starts = starts[np.logical_not(bads_mask)]
            stops = stops[np.logical_not(bads_mask)]
            warnings.warn(
                f"Trials {np.where(bads_mask)[0]} are being dropped as the "
                f"window size ({size}) exceeds their duration {min_duration}."
            )
        else:
            current_ratio = sum(bads_mask) / len(starts)
            raise ValueError(
                f"Window size {size} exceeds trial duration "
                f"({min_duration}) for too many trials "
                f"({current_ratio * 100}%). Set "
                f"accepted_bads_ratio to at least {current_ratio}"
                "and restart training to be able to continue."
            )

    i_window_in_trials, i_trials, window_starts = [], [], []
    for start_i, (start, stop) in enumerate(zip(starts, stops)):
        # Generate possible window starts, with given stride, between original
        # trial onsets and stops (shifted by start_offset and stop_offset,
        # respectively)
        possible_starts = np.arange(start, stop, stride)

        # Possible window start is actually a start, if window size fits in
        # trial start and stop
        for i_window, s in enumerate(possible_starts):
            if (s + size) <= stop:
                window_starts.append(s)
                i_window_in_trials.append(i_window)
                i_trials.append(start_i)

        # If the last window start + window size is not the same as
        # stop + stop_offset, create another window that overlaps and stops
        # at onset + stop_offset
        if not drop_last_window:
            if window_starts[-1] + size != stop:
                window_starts.append(stop - size)
                i_window_in_trials.append(i_window_in_trials[-1] + 1)
                i_trials.append(start_i)

    # Set window stops to be event stops (rather than trial stops)
    window_stops = np.array(window_starts) + size
    if not (len(i_window_in_trials) == len(window_starts) == len(window_stops)):
        raise ValueError(
            f"{len(i_window_in_trials)} == {len(window_starts)} == {len(window_stops)}"
        )

    return i_trials, i_window_in_trials, window_starts, window_stops


def _check_windowing_arguments(
    trial_start_offset_samples,
    trial_stop_offset_samples,
    window_size_samples,
    window_stride_samples,
):
    assert isinstance(trial_start_offset_samples, (int, np.integer))
    assert isinstance(trial_stop_offset_samples, (int, np.integer)) or (
        trial_stop_offset_samples is None
    )
    assert isinstance(window_size_samples, (int, np.integer, type(None)))
    assert isinstance(window_stride_samples, (int, np.integer, type(None)))
    assert (window_size_samples is None) == (window_stride_samples is None)
    if window_size_samples is not None:
        assert window_size_samples > 0, "window size has to be larger than 0"
        assert window_stride_samples > 0, "window stride has to be larger than 0"


def _check_and_set_fixed_length_window_arguments(
    start_offset_samples,
    stop_offset_samples,
    window_size_samples,
    window_stride_samples,
    drop_last_window,
    lazy_metadata,
):
    """Raises warnings for incorrect input arguments and will set correct default values for
    stop_offset_samples & drop_last_window, if necessary.
    """
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

    if (
        window_size_samples is not None
        and window_stride_samples is not None
        and drop_last_window is None
    ):
        raise ValueError(
            "drop_last_window must be set if both window_size_samples &"
            " window_stride_samples have also been set"
        )
    elif (
        window_size_samples is None
        and window_stride_samples is None
        and drop_last_window is False
    ):
        # necessary for following assertion
        drop_last_window = None

    assert (
        (window_size_samples is None)
        == (window_stride_samples is None)
        == (drop_last_window is None)
    )
    if not drop_last_window and lazy_metadata:
        raise ValueError(
            "Cannot have drop_last_window=False and lazy_metadata=True at the same time."
        )
    return stop_offset_samples, drop_last_window


def _get_windowing_kwargs(windowing_func_locals):
    input_kwargs = windowing_func_locals
    input_kwargs.pop("ds")
    windowing_kwargs = {k: v for k, v in input_kwargs.items()}
    return windowing_kwargs
