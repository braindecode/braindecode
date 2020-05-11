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


def create_from_X_y(
        X, y, sfreq, ch_names, drop_samples, supercrop_size_samples=None,
        supercrop_stride_samples=None):
    """Create a BaseConcatDataset of WindowsDatasets from X and y to be used for
    decoding with skorch and braindecode, where X is a list of pre-cut trials
    and y are corresponding targets.

    Parameters
    ----------
    X: array-like
        list of pre-cut trials as n_trials x n_channels x n_times
    y: array-like
        targets corresponding to the trials
    sfreq: common sampling frequency of all trials
    ch_names: array-like
        channel names of the trials
    drop_samples: bool
        whether or not have a last overlapping supercrop/window, when
        supercrops/windows do not equally divide the continuous signal
    supercrop_size_samples: int
        supercrop size
    supercrop_stride_samples: int
        stride between supercrops

    Returns
    -------
    windows_datasets: BaseConcatDataset
        X and y transformed to a dataset format that is compatible with skorch
        and braindecode
    """
    x_times = []
    base_datasets = []
    for x, target in zip(X, y):
        x_times.append(x.shape[1])
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq)
        raw = mne.io.RawArray(x, info)
        base_dataset = BaseDataset(raw, pd.Series({"target": target}),
                                   target_name="target")
        base_datasets.append(base_dataset)
    base_datasets = BaseConcatDataset(base_datasets)

    if supercrop_size_samples is None and supercrop_stride_samples is None:
        if not len(np.unique(x_times)) == 1:
            raise ValueError(f"if 'supercrop_size_samples' and "
                             f"'supercrop_stride_samples' are None, "
                             f"all trials have to have the same length")
        supercrop_size_samples = x_times[0]
        supercrop_stride_samples = x_times[0]
    windows_datasets = create_fixed_length_windows(
        base_datasets,
        start_offset_samples=0,
        stop_offset_samples=0,
        supercrop_size_samples=supercrop_size_samples,
        supercrop_stride_samples=supercrop_stride_samples,
        drop_samples=drop_samples
    )
    return windows_datasets