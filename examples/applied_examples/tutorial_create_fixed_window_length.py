
""".. _fixed-length-windows:

Fixed-Length Windows Extraction
===================================

.. contents:: This example covers:
   :local:
   :depth: 2

"""

######################################################################
# Introduction to Fixed-Length Windows Function
# -------------------------------------
#
# In many EEG decoding tasks, it is useful to split long continuous recordings
# into **fixed-length, overlapping or non-overlapping windows**
#
# The function :func:`braindecode.preprocessing.create_fixed_length_windows`
# provides an easy way to slice a continuous EEG recording into such windows.
#
# This tutorial explains how to use it, what its parameters mean, and how
# it can be applied to EEG datasets.
#

######################################################################
# Overview of create_fixed_length_windows
# ----------------------------------------
#
# The function:
#
# .. code-block:: python
#
#    create_fixed_length_windows(
#        concat_ds,
#        start_offset_samples=0,
#        stop_offset_samples=None,
#        window_size_samples=None,
#        window_stride_samples=None,
#        drop_last_window=None,
#        mapping=None,
#        preload=False,
#        picks=None,
#        reject=None,
#        flat=None,
#        targets_from='metadata',
#        last_target_only=True,
#        lazy_metadata=False,
#        on_missing='error',
#        n_jobs=1,
#        verbose='error',
#    )
#
#

######################################################################
# Parameters
# ----------

# **concat_ds** : `ConcatDataset`
#     - A concat of base datasets each holding raw and description.
#
# **start_offset_samples** : int (default=0)
#     - Start offset from beginning of recording in samples.
#
# **stop_offset_samples** : int or None (default=None)
#     - Stop offset from beginning of recording in samples. If None, set to be the end of the recording.
#
# **window_size_samples** : int or None
#     - Window size in samples. If None, set to be the maximum possible window size, ie length of the recording, once offsets are accounted for.
#
# **window_stride_samples** : int or None
#     - Stride between windows in samples. If None, set to be equal to winddow_size_samples, so windows will not overlap.
#
# **drop_last_window** : bool or None
#     - Whether or not have a last overlapping window, when windows do not equally divide the continuous signal. Must be set to a bool if window size and stride are not None.
#
# **mapping** : dict(str: int) or None
#     - Mapping from event description to target value.
#
# **preload** : bool (default=False)
#     - If True, preload the data of the Epochs objects.
#
# **picks** : str | list | slice | None
#     - Channels to include. If None, all available channels are used. See mne.Epochs.
#
# **reject** : dict or None
#     - Epoch rejection parameters based on peak-to-peak amplitude. If None, no rejection is done based on peak-to-peak amplitude. See mne.Epochs.
#
# **flat** : dict or None
#     - Epoch rejection parameters based on flatness of signals. If None, no rejection based on flatness is done. See mne.Epochs.
#
# **targets_from** : str (default='metadata')
#     - Choose where to get targets from: either 'metadata' or 'events'
#
# **last_target_only** : bool (default=True)
#     - If `True`, only use the last target in the window.
#
# **lazy_metadata** : bool (default=False)
#     - If True, metadata is not computed immediately, but only when accessed by using the _LazyDataFrame (experimental).
#
# **on_missing** : str (default='error')
#     - What to do if one or several event ids are not found in the recording. Valid keys are ‘error’ | ‘warning’ | ‘ignore’. See mne.Epochs.
#
# **n_jobs** : int (default=1)
#     - Number of jobs to use to parallelize the windowing.
#
# **verbose** : bool | str | int | None
#     - Control verbosity of the logging output when calling mne.Epochs.


######################################################################
# Example Usage
# -------------
#
from numpy import multiply

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_fixed_length_windows,
    preprocess
)

# Load the EEG dataset
dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[1])

# Preprocessing
preprocessors = [
    Preprocessor("pick_types", eeg=True, meg=False, stim=False),
    Preprocessor(lambda data: multiply(data, 1e6)),
    Preprocessor("filter", l_freq=4.0, h_freq=38.0),
]
preprocess(dataset, preprocessors)

# Sampling frequency
sfreq = dataset.datasets[0].raw.info["sfreq"]

# Create windows
window_size_samples = int(sfreq * 2)  # 2-second windows
window_stride_samples = int(window_size_samples * 0.5)  # 50% overlap

windows_dataset = create_fixed_length_windows(
    concat_ds=dataset,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=window_size_samples,
    window_stride_samples=window_stride_samples,
    drop_last_window=True,
    mapping=None,
    preload=True,
    picks="eeg",  # Only EEG channels
    reject=dict(eeg=150e-6),  # Reject windows where EEG p2p > 150 µV
    flat=None,
    targets_from="metadata",
    last_target_only=True,
    on_missing="warning",
    n_jobs=1,
    verbose="error",
)

print(
    f"Created {len(windows_dataset)} windows with shape {windows_dataset[0][0].shape}"
)

