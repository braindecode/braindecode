
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
# Example 1: Basic 2-Second, 50% Overlapping Windows
# -------------
#
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_fixed_length_windows,
    preprocess,
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

# Let's inspect the output to better understand what we created.

# Check how many windows were created
print(
    f"Number of windows: {len(windows_dataset)}"
)

# Each window contains EEG data of fixed size
X, y = windows_dataset[0]
print(
    f"Window data shape: {X.shape}"
)
print(
    f"Window label: {y}"
)

######################################################################
# Working with Targets
# --------------------
# In `create_fixed_length_windows`, targets can be derived in two ways:
#
# 1. From the recording metadata (e.g., "session", "condition"). This is the default when `targets_from='metadata'`.
# 2. From signal channels themselves when `targets_from='channels'`, useful for cases like sleep staging where
#    annotations are stored in auxiliary channels.
#
#
# Additionally:
#
# - **mapping**: Optionally map target values (e.g. from "0train" / "1test" to 0 / 1).
# - **last_target_only=True** (default): If multiple targets are present within a window (as in time-varying labels),
#   use only the final target value in that window.

# Example: mapping session names ("0train" and "1test") to integers

mapping = {"0train": 0, "1test": 1}

windows_dataset = create_fixed_length_windows(
    dataset,
    window_size_samples=window_size_samples,
    window_stride_samples=window_stride_samples,
    drop_last_window=True,
    mapping=mapping,
    preload=True,
)

# View first few targets
print(
    "Targets for first 10 windows:"
)
print(
    windows_dataset.datasets[0].windows.get_metadata()['target'][:10]
)


######################################################################
# Example: Rejecting Windows Based on Amplitude
# ------------------------------------
#
# You can set rejection criteria to exclude windows with extreme values:

reject_criteria = dict(eeg=150e-6)  # 150 µV max peak-to-peak allowed

windows_with_rejection = create_fixed_length_windows(
    concat_ds=dataset,
    window_size_samples=200,
    window_stride_samples=100,
    reject=reject_criteria,
    drop_last_window=True
)

print(
    windows_with_rejection
)

######################################################################
# Example: Using lazy metadata generation
# ---------------------------------------
#
# For large datasets, it can be faster to generate metadata on-demand:

lazy_windows = create_fixed_length_windows(
    concat_ds=dataset,
    window_size_samples=200,
    window_stride_samples=100,
    drop_last_window=True,
    lazy_metadata=True
)

print(
    lazy_windows
)

######################################################################
#Example: Shifted Windows
# ---------------------------------------
#
# You can also create shifted windows by using ``start_offset_samples`` or ``stop_offset_samples``.
# For example, start windowing 500 ms later into the recording.

start_offset_seconds = 0.5
start_offset_samples = int(start_offset_seconds * sfreq)

shifted_windows_dataset = create_fixed_length_windows(
    dataset,
    start_offset_samples=start_offset_samples,
    window_size_samples=window_size_samples,
    window_stride_samples=window_stride_samples,
    drop_last_window=True,
    preload=True,
)

print(
    f"Number of shifted windows: {len(shifted_windows_dataset)}"
)