"""Dataset classes."""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#          David Sabbagh <dav.sabbagh@gmail.com>
#          Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

from __future__ import annotations

import bisect
import html as _html
import json
import os
import shutil
import warnings
from abc import abstractmethod
from collections import Counter
from collections.abc import Callable
from glob import glob
from typing import Any, Generic, Iterable, no_type_check

import mne.io
import numpy as np
import pandas as pd
from mne.utils.docs import deprecated
from torch.utils.data import ConcatDataset, Dataset, IterableDataset
from typing_extensions import TypeVar

from .bids.hub import HubDatasetMixin
from .registry import register_dataset


def _create_description(description) -> pd.Series:
    if description is not None:
        if not isinstance(description, pd.Series) and not isinstance(description, dict):
            raise ValueError(
                f"'{description}' has to be either a pandas.Series or a dict."
            )
        if isinstance(description, dict):
            description = pd.Series(description)
    return description


def _html_row(label, value):
    """Generate a single HTML table row."""
    label = _html.escape(str(label))
    value = _html.escape(str(value))
    return f"<tr><td><b>{label}</b></td><td>{value}</td></tr>"


_METADATA_INTERNAL_COLS = {
    "i_window_in_trial",
    "i_start_in_trial",
    "i_stop_in_trial",
    "target",
}


def _metadata_summary(metadata):
    """Summarize window metadata into a dict.

    Returns a dict with keys: n_windows, target_info, extra_columns,
    window_info, is_lazy.
    """
    is_lazy = not isinstance(metadata, pd.DataFrame)

    if is_lazy:
        n_windows = len(metadata)
        columns = list(metadata.columns) if hasattr(metadata, "columns") else []
        extra_columns = [str(c) for c in columns if c not in _METADATA_INTERNAL_COLS]
        return {
            "n_windows": n_windows,
            "target_info": None,
            "extra_columns": extra_columns,
            "window_info": None,
            "is_lazy": True,
        }

    n_windows = len(metadata)
    extra_columns = [
        str(c) for c in metadata.columns if c not in _METADATA_INTERNAL_COLS
    ]

    target_info = None
    if "target" in metadata.columns:
        targets = metadata["target"]
        n_unique = targets.nunique()
        if n_unique <= 10:
            counts = targets.value_counts().sort_index().to_dict()
            target_info = f"{n_unique} unique ({counts})"
        else:
            target_info = f"{n_unique} unique targets"

    window_info = None
    if "i_start_in_trial" in metadata.columns and "i_stop_in_trial" in metadata.columns:
        sizes = metadata["i_stop_in_trial"] - metadata["i_start_in_trial"]
        if len(sizes) == 0:
            pass
        elif (min_s := int(sizes.min())) == (max_s := int(sizes.max())):
            window_info = {"min": min_s, "max": max_s, "uniform": True}
        else:
            window_info = {"min": min_s, "max": max_s, "uniform": False}

    return {
        "n_windows": n_windows,
        "target_info": target_info,
        "extra_columns": extra_columns,
        "window_info": window_info,
        "is_lazy": False,
    }


def _concat_metadata_summary(datasets):
    """Aggregate metadata summary across datasets without full concatenation.

    Returns a dict with keys: window_info, target_info, extra_columns.
    """
    overall_min = None
    overall_max = None
    target_counts = Counter()
    extra_cols = set()

    for ds in datasets:
        if hasattr(ds, "windows"):
            md = ds.windows.metadata
        elif hasattr(ds, "metadata"):
            md = ds.metadata
        else:
            continue
        if md is None or len(md) == 0:
            continue

        extra_cols.update(
            str(c) for c in md.columns if c not in _METADATA_INTERNAL_COLS
        )

        if "i_start_in_trial" in md.columns and "i_stop_in_trial" in md.columns:
            sizes = md["i_stop_in_trial"] - md["i_start_in_trial"]
            if len(sizes) > 0:
                ds_min, ds_max = int(sizes.min()), int(sizes.max())
                overall_min = (
                    ds_min if overall_min is None else min(overall_min, ds_min)
                )
                overall_max = (
                    ds_max if overall_max is None else max(overall_max, ds_max)
                )

        if "target" in md.columns:
            target_counts.update(md["target"].value_counts().to_dict())

    window_info = None
    if overall_min is not None:
        window_info = {
            "min": overall_min,
            "max": overall_max,
            "uniform": overall_min == overall_max,
        }

    target_info = None
    if target_counts:
        n_unique = len(target_counts)
        if n_unique <= 10:
            sorted_counts = dict(sorted(target_counts.items()))
            target_info = f"{n_unique} unique ({sorted_counts})"
        else:
            target_info = f"{n_unique} unique targets"

    return {
        "window_info": window_info,
        "target_info": target_info,
        "extra_columns": sorted(extra_cols),
    }


def _channel_info(mne_obj):
    """Extract channel summary from an mne object.

    Returns (n_ch, type_str, sfreq).
    """
    info = mne_obj.info
    n_ch = info["nchan"]
    sfreq = info["sfreq"]
    ch_types = mne_obj.get_channel_types()
    type_counts = Counter(ch_types)
    type_str = ", ".join(f"{cnt} {t.upper()}" for t, cnt in sorted(type_counts.items()))
    return n_ch, type_str, sfreq


def _window_info(crop_inds, sfreq):
    """Extract window size from crop indices.

    Returns (win_samples, win_secs) or None if crop_inds is empty.
    """
    if len(crop_inds) == 0:
        return None
    first = crop_inds[0]
    win_samples = int(first[2] - first[1])
    win_secs = win_samples / sfreq
    return win_samples, win_secs


class _ReprBuilder:
    """Lightweight builder that renders both text and HTML from the same data."""

    def __init__(self, cls_name, type_display=None):
        self._cls_name = cls_name
        self._type_display = type_display or cls_name
        self._header_parts = []
        self._header_rows = []
        self._items = []

    def add_header(self, header_text, label, value):
        """Add to both the compact ``<ClassName | ...>`` line and an HTML row."""
        self._header_parts.append(header_text)
        self._header_rows.append((label, value))
        return self

    def add_row(self, label, value):
        """Add a detail row (text: ``  label: value``, HTML: ``<tr>``)."""
        self._items.append(("row", label, value))
        return self

    def add_footnote(self, text):
        """Add a footnote (text: ``  (text)``, HTML: italic ``<td colspan=2>``)."""
        self._items.append(("footnote", text, None))
        return self

    def to_repr(self):
        if self._header_parts:
            parts = " | ".join(self._header_parts)
            lines = [f"<{self._cls_name} | {parts}>"]
        else:
            lines = [f"<{self._cls_name}>"]
        for kind, label, value in self._items:
            if kind == "row":
                lines.append(f"  {label}: {value}")
            else:
                lines.append(f"  ({label})")
        return "\n".join(lines)

    def to_html(self):
        rows = [_html_row("Type", self._type_display)]
        for label, value in self._header_rows:
            rows.append(_html_row(label, value))
        for kind, label, value in self._items:
            if kind == "row":
                html_label = label[0].upper() + label[1:]
                rows.append(_html_row(html_label, value))
            else:
                rows.append(
                    f"<tr><td colspan='2'><i>{_html.escape(str(label))}</i></td></tr>"
                )
        table_rows = "\n".join(rows)
        esc_name = _html.escape(str(self._cls_name))
        return (
            f"<table border='1' class='dataframe'>\n"
            f"  <thead><tr><th colspan='2'>{esc_name}</th></tr></thead>\n"
            f"  <tbody>\n{table_rows}\n  </tbody>\n</table>"
        )


def _build_windowed_repr(
    cls_name, n_windows, mne_obj, crop_inds, description, metadata
):
    """Build repr for EEGWindowsDataset and WindowsDataset."""
    b = _ReprBuilder(cls_name)
    n_ch, type_str, sfreq = _channel_info(mne_obj)
    b.add_header(f"{n_windows} windows", "Windows", n_windows)
    b.add_header(f"{n_ch} ch ({type_str})", "Channels", f"{n_ch} ({type_str})")
    wi = _window_info(crop_inds, sfreq)
    if wi is not None:
        win_samples, win_secs = wi
        b.add_header(
            f"{win_samples} samples/win ({win_secs:.3f} s)",
            "Window size",
            f"{win_samples} samples ({win_secs:.3f} s)",
        )
    b.add_header(f"{sfreq:.1f} Hz", "Sfreq", f"{sfreq:.1f} Hz")
    if description is not None:
        desc_items = ", ".join(f"{k}={v}" for k, v in description.items())
        b.add_row("description", desc_items)
    if metadata is not None:
        summary = _metadata_summary(metadata)
        if summary["is_lazy"]:
            b.add_row("metadata", f"lazy ({summary['n_windows']} windows)")
        else:
            if summary["target_info"]:
                b.add_row("targets", summary["target_info"])
            if summary["extra_columns"]:
                b.add_row("extra metadata", ", ".join(summary["extra_columns"]))
    return b


class RecordDataset(Dataset[tuple[np.ndarray, int | str, tuple[int, int, int]]]):
    def __init__(
        self,
        description: dict | pd.Series | None = None,
        transform: Callable | None = None,
    ):
        self._description = _create_description(description)
        self.transform = transform

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    def description(self) -> pd.Series:
        return self._description

    def set_description(self, description: dict | pd.Series, overwrite: bool = False):
        """Update (add or overwrite) the dataset description.

        Parameters
        ----------
        description : dict | pd.Series
            Description in the form key: value.
        overwrite : bool
            Has to be True if a key in description already exists in the
            dataset description.
        """
        description = _create_description(description)
        if self.description is None:
            self._description = description
        else:
            for key, value in description.items():
                # if the key is already in the existing description, drop it
                if key in self._description:
                    assert overwrite, (
                        f"'{key}' already in description. Please "
                        f"rename or set overwrite to True."
                    )
                    self._description.pop(key)
            self._description = pd.concat([self.description, description])

    @property
    def transform(self) -> Callable | None:
        return self._transform

    @transform.setter
    def transform(self, value: Callable | None):
        if value is not None and not callable(value):
            raise ValueError("Transform needs to be a callable.")
        self._transform = value

    def _build_repr(self):
        return _ReprBuilder(type(self).__name__)

    def __repr__(self):
        return self._build_repr().to_repr()

    def _repr_html_(self):
        return self._build_repr().to_html()


# Type of the datasets contained in BaseConcatDataset
T = TypeVar("T", bound=RecordDataset)


@register_dataset
class RawDataset(RecordDataset):
    """Returns samples from an mne.io.Raw object along with a target.

    Dataset which serves samples from an mne.io.Raw object along with a target.
    The target is unique for the dataset, and is obtained through the
    `description` attribute.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous data.
    description : dict | pandas.Series | None
        Holds additional description about the continuous signal / subject.
    target_name : str | tuple | None
        Name(s) of the index in `description` that should be used to provide the
        target (e.g., to be used in a prediction task later on).
    transform : callable | None
        On-the-fly transform applied to the example before it is returned.
    """

    def __init__(
        self,
        raw: mne.io.BaseRaw,
        description: dict | pd.Series | None = None,
        target_name: str | tuple[str, ...] | None = None,
        transform: Callable | None = None,
    ):
        super().__init__(description, transform)
        self.raw = raw

        # save target name for load/save later
        self.target_name = self._target_name(target_name)
        self.raw_preproc_kwargs: list[dict[str, Any]] = []

    def __getitem__(self, index):
        X = self.raw[:, index][0]
        y = None
        if self.target_name is not None:
            y = self.description[self.target_name]
        if isinstance(y, pd.Series):
            y = y.to_list()
        if self.transform is not None:
            X = self.transform(X)
        return X, y

    def __len__(self):
        return len(self.raw)

    def _build_repr(self):
        b = _ReprBuilder(type(self).__name__)
        n_ch, type_str, sfreq = _channel_info(self.raw)
        n_times = len(self.raw.times)
        duration = n_times / sfreq
        b.add_header(f"{n_ch} ch ({type_str})", "Channels", f"{n_ch} ({type_str})")
        b.add_header(f"{sfreq:.1f} Hz", "Sfreq", f"{sfreq:.1f} Hz")
        b.add_header(
            f"{n_times} samples ({duration:.1f} s)",
            "Samples",
            f"{n_times} ({duration:.1f} s)",
        )
        if self.description is not None:
            desc_items = ", ".join(f"{k}={v}" for k, v in self.description.items())
            b.add_row("description", desc_items)
        return b

    def _target_name(self, target_name):
        if target_name is not None and not isinstance(target_name, (str, tuple, list)):
            raise ValueError("target_name has to be None, str, tuple or list")
        if target_name is None:
            return target_name
        else:
            # convert tuple of names or single name to list
            if isinstance(target_name, tuple):
                target_name = [name for name in target_name]
            elif not isinstance(target_name, list):
                assert isinstance(target_name, str)
                target_name = [target_name]
            assert isinstance(target_name, list)
            # check if target name(s) can be read from description
            for name in target_name:
                if self.description is None or name not in self.description:
                    warnings.warn(
                        f"'{name}' not in description. '__getitem__'"
                        f"will fail unless an appropriate target is"
                        f" added to description.",
                        UserWarning,
                    )
        # return a list of str if there are multiple targets and a str otherwise
        return target_name if len(target_name) > 1 else target_name[0]


@deprecated(
    "The BaseDataset class is deprecated. "
    "If you want to instantiate a dataset containing raws, use RawDataset instead. "
    "If you want to type a Braindecode dataset (i.e. RawDataset|EEGWindowsDataset|WindowsDataset), "
    "use the RecordDataset class instead."
)
@register_dataset
class BaseDataset(RawDataset):
    pass


@register_dataset
class EEGWindowsDataset(RecordDataset):
    """Returns windows from an mne.Raw object, its window indices, along with a target.

    Dataset which serves windows from an mne.Epochs object along with their
    target and additional information. The `metadata` attribute of the Epochs
    object must contain a column called `target`, which will be used to return
    the target that corresponds to a window. Additional columns
    `i_window_in_trial`, `i_start_in_trial`, `i_stop_in_trial` are also
    required to serve information about the windowing (e.g., useful for cropped
    training).
    See `braindecode.datautil.windowers` to directly create a `WindowsDataset`
    from a `RawDataset` object.

    Parameters
    ----------
    windows : mne.Raw or mne.Epochs (Epochs is outdated)
        Windows obtained through the application of a windower to a ``RawDataset``
        (see `braindecode.datautil.windowers`).
    description : dict | pandas.Series | None
        Holds additional info about the windows.
    transform : callable | None
        On-the-fly transform applied to a window before it is returned.
    targets_from : str
        Defines whether targets will be extracted from  metadata or from `misc`
        channels (time series targets). It can be `metadata` (default) or `channels`.
    last_target_only : bool
        If targets are obtained from misc channels whether all targets if the entire
        (compute) window will be returned or only the last target in the window.
    metadata : pandas.DataFrame
        Dataframe with crop indices, so `i_window_in_trial`, `i_start_in_trial`, `i_stop_in_trial`
        as well as `targets`.
    """

    def __init__(
        self,
        raw: mne.io.BaseRaw,
        metadata: pd.DataFrame,
        description: dict | pd.Series | None = None,
        transform: Callable | None = None,
        targets_from: str = "metadata",
        last_target_only: bool = True,
    ):
        super().__init__(description, transform)
        self.raw = raw
        self.metadata = metadata

        self.last_target_only = last_target_only
        if targets_from not in ("metadata", "channels"):
            raise ValueError("Wrong value for parameter `targets_from`.")
        self.targets_from = targets_from
        self.crop_inds = metadata.loc[
            :, ["i_window_in_trial", "i_start_in_trial", "i_stop_in_trial"]
        ].to_numpy()
        if self.targets_from == "metadata":
            self.y = metadata.loc[:, "target"].to_list()
        self.raw_preproc_kwargs: list[dict[str, Any]] = []

    def __getitem__(self, index: int):
        """Get a window and its target.

        Parameters
        ----------
        index : int
            Index to the window (and target) to return.

        Returns
        -------
        np.ndarray
            Window of shape (n_channels, n_times).
        int
            Target for the windows.
        np.ndarray
            Crop indices.
        """

        # necessary to cast as list to get list of three tensors from batch,
        # otherwise get single 2d-tensor...
        crop_inds = self.crop_inds[index].tolist()

        i_window_in_trial, i_start, i_stop = crop_inds
        X = self.raw._getitem((slice(None), slice(i_start, i_stop)), return_times=False)
        X = X.astype("float32")
        # ensure we don't give the user the option
        # to accidentally modify the underlying array
        X = X.copy()
        if self.transform is not None:
            X = self.transform(X)
        if self.targets_from == "metadata":
            y = self.y[index]
        else:
            misc_mask = np.array(self.raw.get_channel_types()) == "misc"
            if self.last_target_only:
                y = X[misc_mask, -1]
            else:
                y = X[misc_mask, :]
            # ensure we don't give the user the option
            # to accidentally modify the underlying array
            y = y.copy()
            # remove the target channels from raw
            X = X[~misc_mask, :]
        return X, y, crop_inds

    def __len__(self):
        return len(self.crop_inds)

    def _build_repr(self):
        return _build_windowed_repr(
            type(self).__name__,
            len(self),
            self.raw,
            self.crop_inds,
            self.description,
            self.metadata,
        )


@register_dataset
class WindowsDataset(RecordDataset):
    """Returns windows from an mne.Epochs object along with a target.

    Dataset which serves windows from an mne.Epochs object along with their
    target and additional information. The `metadata` attribute of the Epochs
    object must contain a column called `target`, which will be used to return
    the target that corresponds to a window. Additional columns
    `i_window_in_trial`, `i_start_in_trial`, `i_stop_in_trial` are also
    required to serve information about the windowing (e.g., useful for cropped
    training).
    See `braindecode.datautil.windowers` to directly create a `WindowsDataset`
    from a ``RawDataset`` object.

    Parameters
    ----------
    windows : mne.Epochs
        Windows obtained through the application of a windower to a RawDataset
        (see `braindecode.datautil.windowers`).
    description : dict | pandas.Series | None
        Holds additional info about the windows.
    transform : callable | None
        On-the-fly transform applied to a window before it is returned.
    targets_from : str
        Defines whether targets will be extracted from mne.Epochs metadata or mne.Epochs `misc`
        channels (time series targets). It can be `metadata` (default) or `channels`.
    """

    def __init__(
        self,
        windows: mne.BaseEpochs,
        description: dict | pd.Series | None = None,
        transform: Callable | None = None,
        targets_from: str = "metadata",
        last_target_only: bool = True,
    ):
        super().__init__(description, transform)
        self.windows = windows
        self.last_target_only = last_target_only
        if targets_from not in ("metadata", "channels"):
            raise ValueError("Wrong value for parameter `targets_from`.")
        self.targets_from = targets_from

        metadata = self.windows.metadata
        assert metadata is not None, "WindowsDataset requires windows with metadata."
        self.crop_inds = metadata.loc[
            :, ["i_window_in_trial", "i_start_in_trial", "i_stop_in_trial"]
        ].to_numpy()
        if self.targets_from == "metadata":
            self.y = metadata.loc[:, "target"].to_list()
        self.raw_preproc_kwargs: list[dict[str, Any]] = []
        self.window_preproc_kwargs: list[dict[str, Any]] = []

    def __getitem__(self, index: int):
        """Get a window and its target.

        Parameters
        ----------
        index : int
            Index to the window (and target) to return.

        Returns
        -------
        np.ndarray
            Window of shape (n_channels, n_times).
        int
            Target for the windows.
        np.ndarray
            Crop indices.
        """
        X = self.windows.get_data(item=index)[0].astype("float32")
        if self.transform is not None:
            X = self.transform(X)
        if self.targets_from == "metadata":
            y = self.y[index]
        else:
            misc_mask = np.array(self.windows.get_channel_types()) == "misc"
            if self.last_target_only:
                y = X[misc_mask, -1]
            else:
                y = X[misc_mask, :]
            # remove the target channels from raw
            X = X[~misc_mask, :]
        # necessary to cast as list to get list of three tensors from batch,
        # otherwise get single 2d-tensor...
        crop_inds = self.crop_inds[index].tolist()
        return X, y, crop_inds

    def __len__(self) -> int:
        return len(self.windows.events)

    def _build_repr(self):
        return _build_windowed_repr(
            type(self).__name__,
            len(self),
            self.windows,
            self.crop_inds,
            self.description,
            self.windows.metadata,
        )


@register_dataset
class BaseConcatDataset(ConcatDataset, HubDatasetMixin, Generic[T]):
    """A base class for concatenated datasets.

    Holds either mne.Raw or mne.Epoch in self.datasets and has
    a pandas DataFrame with additional description.

    Includes Hugging Face Hub integration via HubDatasetMixin for
    uploading and downloading datasets.

    Parameters
    ----------
    list_of_ds : list
        list of RecordDataset
    target_transform : callable | None
        Optional function to call on targets before returning them.
    lazy : bool, default False
        If True, defer computing cumulative sizes until length or item access.
    """

    datasets: list[T]

    def __init__(
        self,
        list_of_ds: list[T | BaseConcatDataset[T]],
        target_transform: Callable | None = None,
        *,
        lazy: bool = False,
    ):
        # Adapted from torch.utils.data.ConcatDataset:
        # added lazy init to defer cumulative size computation until access,
        # plus _ensure_cumulative_sizes/_get_single_item helpers and __len__ changes.
        # if we get a list of BaseConcatDataset, get all the individual datasets
        flattened_list_of_ds: list[T] = []
        for ds in list_of_ds:
            if isinstance(ds, BaseConcatDataset):
                flattened_list_of_ds.extend(ds.datasets)
            else:
                flattened_list_of_ds.append(ds)

        # Validate inputs (same validation for both lazy and non-lazy)
        if len(flattened_list_of_ds) == 0:
            raise ValueError("datasets should not be an empty iterable")
        for ds in flattened_list_of_ds:
            if isinstance(ds, IterableDataset):
                raise TypeError("ConcatDataset does not support IterableDataset")

        if lazy:
            Dataset.__init__(self)
            self.datasets = flattened_list_of_ds
            # Defer cumulative size computation until first access.
            self._cumulative_sizes: list[int] | None = None
        else:
            super().__init__(flattened_list_of_ds)

        self.target_transform = target_transform
        self._lazy = lazy

    @property
    def cumulative_sizes(self) -> list[int]:
        """Cumulative sizes of the underlying datasets.

        When the dataset is created with ``lazy=True``, the cumulative sizes
        are computed on first access and then cached.
        """
        return self._ensure_cumulative_sizes()

    @cumulative_sizes.setter
    def cumulative_sizes(self, value: list[int] | None) -> None:
        # Keep compatibility with torch.utils.data.ConcatDataset, which assigns
        # to ``self.cumulative_sizes`` in its __init__.
        self._cumulative_sizes = value

    def _ensure_cumulative_sizes(self) -> list[int]:
        if self._cumulative_sizes is None:
            self._cumulative_sizes = ConcatDataset.cumsum(self.datasets)
        return self._cumulative_sizes

    def _get_sequence(self, indices):
        X, y = list(), list()
        for ind in indices:
            out_i = self._get_single_item(ind)
            X.append(out_i[0])
            y.append(out_i[1])

        X = np.stack(X, axis=0)
        y = np.array(y)

        return X, y

    def __len__(self):
        return self._ensure_cumulative_sizes()[-1]

    def _get_single_item(self, idx: int):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx

        cumulative_sizes = self._ensure_cumulative_sizes()
        dataset_idx = bisect.bisect_right(cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def __getitem__(self, idx: int | list):
        """
        ---

        idx : int | list
            Index of window and target to return. If provided as a list of
            ints, multiple windows and targets will be extracted and
            concatenated. The target output can be modified on the
            fly by the ``traget_transform`` parameter.
        """
        if isinstance(idx, Iterable):  # Sample multiple windows
            item = self._get_sequence(idx)
        else:
            item = self._get_single_item(idx)
        if self.target_transform is not None:
            item = item[:1] + (self.target_transform(item[1]),) + item[2:]
        return item

    @no_type_check  # TODO, it's a mess
    def split(
        self,
        by: str | list[int] | list[list[int]] | dict[str, list[int]] | None = None,
        property: str | None = None,
        split_ids: list[int] | list[list[int]] | dict[str, list[int]] | None = None,
    ) -> dict[str, BaseConcatDataset]:
        """Split the dataset based on information listed in its description.

        The format could be based on a DataFrame or based on indices.

        Parameters
        ----------
        by : str | list | dict
            If ``by`` is a string, splitting is performed based on the
            description DataFrame column with this name.
            If ``by`` is a (list of) list of integers, the position in the first
            list corresponds to the split id and the integers to the
            datapoints of that split.
            If a dict then each key will be used in the returned
            splits dict and each value should be a list of int.
        property : str
            :bdg-warning:`Deprecated`

            Some property which is listed in the info DataFrame.
        split_ids : list | dict
            :bdg-warning:`Deprecated`

            List of indices to be combined in a subset.
            It can be a list of int or a list of list of int.

        Returns
        -------
        splits : dict
            A dictionary with the name of the split (a string) as key and the
            dataset as value.
        """

        args_not_none = [by is not None, property is not None, split_ids is not None]
        if sum(args_not_none) != 1:
            raise ValueError("Splitting requires exactly one argument.")

        if property is not None or split_ids is not None:
            warnings.warn(
                "Keyword arguments `property` and `split_ids` "
                "are deprecated and will be removed in the future. "
                "Use `by` instead.",
                DeprecationWarning,
            )
            by = property if property is not None else split_ids
        if isinstance(by, str):
            split_ids = {
                k: list(v) for k, v in self.description.groupby(by).groups.items()
            }
        elif isinstance(by, dict):
            split_ids = by
        else:
            # assume list(int)
            if not isinstance(by[0], list):
                by = [by]
            # assume list(list(int))
            split_ids = {split_i: split for split_i, split in enumerate(by)}

        return {
            str(split_name): BaseConcatDataset(
                [self.datasets[ds_ind] for ds_ind in ds_inds],
                target_transform=self.target_transform,
            )
            for split_name, ds_inds in split_ids.items()
        }

    def get_metadata(self) -> pd.DataFrame:
        """Concatenate the metadata and description of the wrapped Epochs.

        Returns
        -------
        metadata : pd.DataFrame
            DataFrame containing as many rows as there are windows in the
            BaseConcatDataset, with the metadata and description information
            for each window.
        """
        if not all(
            [
                isinstance(ds, (WindowsDataset, EEGWindowsDataset))
                for ds in self.datasets
            ]
        ):
            raise TypeError(
                "Metadata dataframe can only be computed when all "
                "datasets are WindowsDataset."
            )

        all_dfs = list()
        for ds in self.datasets:
            if hasattr(ds, "windows"):
                df = ds.windows.metadata
            else:
                df = ds.metadata
            for k, v in ds.description.items():
                df[k] = v
            all_dfs.append(df)

        return pd.concat(all_dfs)

    @property
    def transform(self):
        return [ds.transform for ds in self.datasets]

    @transform.setter
    def transform(self, fn):
        for i in range(len(self.datasets)):
            self.datasets[i].transform = fn

    @property
    def target_transform(self):
        return self._target_transform

    @target_transform.setter
    def target_transform(self, fn):
        if not (callable(fn) or fn is None):
            raise TypeError("target_transform must be a callable.")
        self._target_transform = fn

    def _outdated_save(self, path, overwrite=False):
        """This is a copy of the old saving function, that had inconsistent.

        functionality for BaseDataset and WindowsDataset. It only exists to
        assure backwards compatibility by still being able to run the old tests.

        Save dataset to files.

        Parameters
        ----------
        path : str
            Directory to which .fif / -epo.fif and .json files are stored.
        overwrite : bool
            Whether to delete old files (.json, .fif, -epo.fif) in specified
            directory prior to saving.
        """
        warnings.warn(
            "This function only exists for backwards compatibility "
            "purposes. DO NOT USE!",
            UserWarning,
        )
        if isinstance(self.datasets[0], EEGWindowsDataset):
            raise NotImplementedError(
                "Outdated save not implemented for new window datasets."
            )
        if len(self.datasets) == 0:
            raise ValueError("Expect at least one dataset")
        if not (
            hasattr(self.datasets[0], "raw") or hasattr(self.datasets[0], "windows")
        ):
            raise ValueError("dataset should have either raw or windows attribute")
        file_name_templates = ["{}-raw.fif", "{}-epo.fif"]
        description_file_name = os.path.join(path, "description.json")
        target_file_name = os.path.join(path, "target_name.json")
        if not overwrite:
            from braindecode.datautil.serialization import (  # Import here to avoid circular import
                _check_save_dir_empty,
            )

            _check_save_dir_empty(path)
        else:
            for file_name_template in file_name_templates:
                file_names = glob(
                    os.path.join(path, f"*{file_name_template.lstrip('{}')}")
                )
                _ = [os.remove(f) for f in file_names]
            if os.path.isfile(target_file_name):
                os.remove(target_file_name)
            if os.path.isfile(description_file_name):
                os.remove(description_file_name)
            for kwarg_name in [
                "raw_preproc_kwargs",
                "window_kwargs",
                "window_preproc_kwargs",
            ]:
                kwarg_path = os.path.join(path, ".".join([kwarg_name, "json"]))
                if os.path.exists(kwarg_path):
                    os.remove(kwarg_path)

        is_raw = hasattr(self.datasets[0], "raw")

        if is_raw:
            file_name_template = file_name_templates[0]
        else:
            file_name_template = file_name_templates[1]

        for i_ds, ds in enumerate(self.datasets):
            full_file_path = os.path.join(path, file_name_template.format(i_ds))
            if is_raw:
                ds.raw.save(full_file_path, overwrite=overwrite)
            else:
                ds.windows.save(full_file_path, overwrite=overwrite)

        self.description.to_json(description_file_name)
        for kwarg_name in [
            "raw_preproc_kwargs",
            "window_kwargs",
            "window_preproc_kwargs",
        ]:
            if hasattr(self, kwarg_name):
                kwargs_path = os.path.join(path, ".".join([kwarg_name, "json"]))
                kwargs = getattr(self, kwarg_name)
                if kwargs is not None:
                    json.dump(kwargs, open(kwargs_path, "w"))

    @property
    def description(self) -> pd.DataFrame:
        df = pd.DataFrame([ds.description for ds in self.datasets])
        df.reset_index(inplace=True, drop=True)
        return df

    def set_description(
        self, description: dict | pd.DataFrame, overwrite: bool = False
    ):
        """Update (add or overwrite) the dataset description.

        Parameters
        ----------
        description : dict | pd.DataFrame
            Description in the form key: value where the length of the value
            has to match the number of datasets.
        overwrite : bool
            Has to be True if a key in description already exists in the
            dataset description.
        """
        description = pd.DataFrame(description)
        for key, value in description.items():
            for ds, value_ in zip(self.datasets, value):
                ds.set_description({key: value_}, overwrite=overwrite)

    def save(self, path: str, overwrite: bool = False, offset: int = 0):
        """Save datasets to files by creating one subdirectory for each dataset::

            path/
                0/
                    0-raw.fif | 0-epo.fif
                    description.json
                    raw_preproc_kwargs.json (if raws were preprocessed)
                    window_kwargs.json (if this is a windowed dataset)
                    window_preproc_kwargs.json  (if windows were preprocessed)
                    target_name.json (if target_name is not None and dataset is raw)
                1/
                    1-raw.fif | 1-epo.fif
                    description.json
                    raw_preproc_kwargs.json (if raws were preprocessed)
                    window_kwargs.json (if this is a windowed dataset)
                    window_preproc_kwargs.json  (if windows were preprocessed)
                    target_name.json (if target_name is not None and dataset is raw)

        Parameters
        ----------
        path : str
            Directory in which subdirectories are created to store
             -raw.fif | -epo.fif and .json files to.
        overwrite : bool
            Whether to delete old subdirectories that will be saved to in this
            call.
        offset : int
            If provided, the integer is added to the id of the dataset in the
            concat. This is useful in the setting of very large datasets, where
            one dataset has to be processed and saved at a time to account for
            its original position.
        """
        if len(self.datasets) == 0:
            raise ValueError("Expect at least one dataset")
        if not (
            hasattr(self.datasets[0], "raw") or hasattr(self.datasets[0], "windows")
        ):
            raise ValueError("dataset should have either raw or windows attribute")

        # Create path if it doesn't exist
        os.makedirs(path, exist_ok=True)

        path_contents = os.listdir(path)
        n_sub_dirs = len(
            [e for e in path_contents if os.path.isdir(os.path.join(path, e))]
        )
        for i_ds, ds in enumerate(self.datasets):
            # remove subdirectory from list of untouched files / subdirectories
            if str(i_ds + offset) in path_contents:
                path_contents.remove(str(i_ds + offset))
            # save_dir/i_ds/
            sub_dir = os.path.join(path, str(i_ds + offset))
            if os.path.exists(sub_dir):
                if overwrite:
                    shutil.rmtree(sub_dir)
                else:
                    raise FileExistsError(
                        f"Subdirectory {sub_dir} already exists. Please select"
                        f" a different directory, set overwrite=True, or "
                        f"resolve manually."
                    )
            # save_dir/{i_ds+offset}/
            os.makedirs(sub_dir)
            # save_dir/{i_ds+offset}/{i_ds+offset}-{raw_or_epo}.fif
            self._save_signals(sub_dir, ds, i_ds, offset)
            # save_dir/{i_ds+offset}/metadata_df.pkl
            self._save_metadata(sub_dir, ds)
            # save_dir/{i_ds+offset}/description.json
            self._save_description(sub_dir, ds.description)
            # save_dir/{i_ds+offset}/raw_preproc_kwargs.json
            # save_dir/{i_ds+offset}/window_kwargs.json
            # save_dir/{i_ds+offset}/window_preproc_kwargs.json
            self._save_kwargs(sub_dir, ds)
            # save_dir/{i_ds+offset}/target_name.json
            self._save_target_name(sub_dir, ds)
        if overwrite:
            # the following will be True for all datasets preprocessed and
            # stored in parallel with braindecode.preprocessing.preprocess
            if i_ds + 1 + offset < n_sub_dirs:
                warnings.warn(
                    f"The number of saved datasets ({i_ds + 1 + offset}) "
                    f"does not match the number of existing "
                    f"subdirectories ({n_sub_dirs}). You may now "
                    f"encounter a mix of differently preprocessed "
                    f"datasets!",
                    UserWarning,
                )
        # if path contains files or directories that were not touched, raise
        # warning
        if path_contents:
            warnings.warn(
                f"Chosen directory {path} contains other "
                f"subdirectories or files {path_contents}."
            )

    @staticmethod
    def _save_signals(sub_dir, ds, i_ds, offset):
        raw_or_epo = "raw" if hasattr(ds, "raw") else "epo"
        fif_file_name = f"{i_ds + offset}-{raw_or_epo}.fif"
        fif_file_path = os.path.join(sub_dir, fif_file_name)
        raw_or_windows = "raw" if raw_or_epo == "raw" else "windows"

        # The following appears to be necessary to avoid a CI failure when
        # preprocessing WindowsDatasets with serialization enabled. The failure
        # comes from `mne.epochs._check_consistency` which ensures the Epochs's
        # object `times` attribute is not writeable.
        getattr(ds, raw_or_windows).times.flags["WRITEABLE"] = False

        getattr(ds, raw_or_windows).save(fif_file_path)

    @staticmethod
    def _save_metadata(sub_dir, ds):
        if hasattr(ds, "metadata"):
            metadata_file_path = os.path.join(sub_dir, "metadata_df.pkl")
            ds.metadata.to_pickle(metadata_file_path)

    @staticmethod
    def _save_description(sub_dir, description):
        description_file_path = os.path.join(sub_dir, "description.json")
        description.to_json(description_file_path, default_handler=str)

    @staticmethod
    def _save_kwargs(sub_dir, ds):
        for kwargs_name in [
            "raw_preproc_kwargs",
            "window_kwargs",
            "window_preproc_kwargs",
        ]:
            if hasattr(ds, kwargs_name):
                kwargs_file_name = ".".join([kwargs_name, "json"])
                kwargs_file_path = os.path.join(sub_dir, kwargs_file_name)
                kwargs = getattr(ds, kwargs_name)
                if kwargs is not None:
                    with open(kwargs_file_path, "w") as f:
                        json.dump(kwargs, f, indent=2)

    @staticmethod
    def _save_target_name(sub_dir, ds):
        if hasattr(ds, "target_name"):
            target_file_path = os.path.join(sub_dir, "target_name.json")
            with open(target_file_path, "w") as f:
                json.dump({"target_name": ds.target_name}, f)

    @staticmethod
    def _signal_summary(ds):
        """Return (mne_obj, is_windowed) from a dataset."""
        if hasattr(ds, "windows"):
            mne_obj = ds.windows
        elif hasattr(ds, "raw"):
            mne_obj = ds.raw
        else:
            return None, False
        is_windowed = hasattr(ds, "crop_inds")
        return mne_obj, is_windowed

    def _build_repr(self):
        n_ds = len(self.datasets)
        b = _ReprBuilder("BaseConcatDataset")
        if n_ds == 0:
            b.add_header("0 datasets", "Recordings", 0)
            return b

        n_total = len(self)
        ds_type = type(self.datasets[0]).__name__
        first_ds = self.datasets[0]
        mne_obj, is_windowed = self._signal_summary(first_ds)

        b._type_display = f"BaseConcatDataset of {ds_type}"
        b.add_header(f"{n_ds} {ds_type}(s)", "Recordings", n_ds)
        b.add_header(f"{n_total} total samples", "Total samples", n_total)

        if mne_obj is not None:
            n_ch, type_str, sfreq = _channel_info(mne_obj)
            b.add_row("Sfreq*", f"{sfreq:.1f} Hz")
            b.add_row("Channels*", f"{n_ch} ({type_str})")

            ch_names = mne_obj.info["ch_names"]
            if len(ch_names) <= 10:
                ch_str = ", ".join(ch_names)
            else:
                ch_str = (
                    ", ".join(ch_names[:10]) + f", ... (+{len(ch_names) - 10} more)"
                )
            b.add_row("Ch. names*", ch_str)

            montage = mne_obj.get_montage()
            if montage is not None:
                b.add_row("Montage*", montage.get_positions()["coord_frame"])

            if not is_windowed:
                n_times = len(mne_obj.times)
                duration = n_times / sfreq
                b.add_row("Duration*", f"{duration:.1f} s")

            b.add_footnote("* from first recording")

        desc = self.description
        if desc is not None and not desc.empty:
            col_str = ", ".join(str(c) for c in desc.columns)
            b.add_row(
                "Description",
                f"{desc.shape[0]} recordings × {desc.shape[1]} columns [{col_str}]",
            )

        if is_windowed:
            try:
                summary = _concat_metadata_summary(self.datasets)
                if summary["window_info"] is not None and mne_obj is not None:
                    wi = summary["window_info"]
                    if wi["uniform"]:
                        win_secs = wi["min"] / sfreq
                        b.add_row(
                            "Window",
                            f"{wi['min']} samples ({win_secs:.3f} s)",
                        )
                    else:
                        min_secs = wi["min"] / sfreq
                        max_secs = wi["max"] / sfreq
                        b.add_row(
                            "Window",
                            f"{wi['min']}-{wi['max']} samples"
                            f" ({min_secs:.3f}-{max_secs:.3f} s)",
                        )
                if summary["target_info"]:
                    b.add_row("Targets", summary["target_info"])
                if summary["extra_columns"]:
                    b.add_row(
                        "Extra meta",
                        ", ".join(summary["extra_columns"]),
                    )
            except (TypeError, AttributeError):
                pass

        return b

    def __repr__(self):
        return self._build_repr().to_repr()

    def _repr_html_(self):
        return self._build_repr().to_html()
