import os
import re
import glob

import numpy as np
import pandas as pd
import mne

from .base import BaseDataset, BaseConcatDataset, WindowsDataset
from ..datautil.windowers import (
    create_fixed_length_windows, create_windows_from_events,
    _compute_supercrop_inds)


def create_from_mne_raw(
        raws, trial_start_offset_samples, trial_stop_offset_samples,
        supercrop_size_samples, supercrop_stride_samples, drop_samples,
        descriptions=None, mapping=None, preload=False, drop_bad_windows=True):
    """Create WindowsDatasets from mne.RawArrays

    Parameters
    ----------
    raws: array-like
        list of mne.RawArrays
    trial_start_offset_samples: int
        start offset from original trial onsets in samples
    trial_stop_offset_samples: int
        stop offset from original trial stop in samples
    supercrop_size_samples: int
        supercrop size
    supercrop_stride_samples: int
        stride between supercrops
    drop_samples: bool
        whether or not have a last overlapping supercrop/window, when
        supercrops/windows do not equally divide the continuous signal
    descriptions: array-like
        list of dicts or pandas.Series with additional information about the raws
    mapping: dict(str: int)
        mapping from event description to target value
    preload: bool
        if True, preload the data of the Epochs objects.
    drop_bad_windows: bool
        If True, call `.drop_bad()` on the resulting mne.Epochs object. This
        step allows identifying e.g., windows that fall outside of the
        continuous recording. It is suggested to run this step here as otherwise
        the BaseConcatDataset has to be updated as well.

    Returns
    -------
    windows_datasets: BaseConcatDataset
        X and y transformed to a dataset format that is compativle with skorch
        and braindecode
    """
    if descriptions is not None:
        if len(descriptions) != len(raws):
            raise ValueError(
                f"length of 'raws' ({len(raws)}) and 'description' "
                f"({len(description)}) has to match")
        base_datasets = [BaseDataset(raw, desc) for raw, desc in
                         zip(raws, descriptions)]
    else:
        base_datasets = [BaseDataset(raw) for raw in raws]

    base_datasets = BaseConcatDataset(base_datasets)
    windows_datasets = create_windows_from_events(
        base_datasets,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=trial_stop_offset_samples,
        supercrop_size_samples=supercrop_size_samples,
        supercrop_stride_samples=supercrop_stride_samples,
        drop_samples=drop_samples,
        mapping=mapping,
        drop_bad_windows=drop_bad_windows,
        preload=preload
    )
    return windows_datasets


def create_from_mne_epochs(list_of_epochs, supercrop_size_samples,
                           supercrop_stride_samples, drop_samples):
    """Create WindowsDatasets from mne.Epochs

    Parameters
    ----------
    list_of_epochs: array-like
        list of mne.Epochs
    supercrop_size_samples: int
        supercrop size
    supercrop_stride_samples: int
        stride between supercrops
    drop_samples: bool
        whether or not have a last overlapping supercrop/window, when
        supercrops/windows do not equally divide the continuous signal

    Returns
    -------
    windows_datasets: BaseConcatDataset
        X and y transformed to a dataset format that is compativle with skorch
        and braindecode
    """
    windows_datasets = []
    for epochs in list_of_epochs:
        starts = epochs.events[:, 0]
        targets = epochs.events[:, 2]
        stops = starts + len(epochs.times)

        i_trials, i_supercrops_in_trial, starts, stops = _compute_supercrop_inds(
            starts,
            stops,
            start_offset=0,
            stop_offset=0,
            size=supercrop_size_samples,
            stride=supercrop_stride_samples,
            drop_samples=drop_samples
        )

        # repeat trial targets corresponding to number of windows per trial
        window_targets = []
        for y, count in zip(targets, np.bincount(i_trials)):
            window_targets.extend([y] * count)

        df = pd.DataFrame(dict(
            i_trial=i_trials,
            i_supercrop_in_trial=i_supercrops_in_trial,
            i_start_in_trial=starts,
            i_stop_in_trial=stops,
            target=window_targets
        ))

        epochs.events = df[["i_start_in_trial", "i_trial", "target"]].to_numpy()
        # mne requires epochs.selection to be of same length as metadata
        epochs.selection = list(df.index)
        if epochs.metadata is None:
            epochs.metadata = df
        else:
            epochs.metadata = pd.concat([epochs.metadata, df], axis=1)
        windows_datasets.append(WindowsDataset(epochs))
    return BaseConcatDataset(windows_datasets)

