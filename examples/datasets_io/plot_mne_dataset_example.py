""".. _mne-dataset-example:

MNE Dataset Example
===================
"""
##############################################################################
# This example shows how to convert data from mne.Raws or mne.Epochs to a
# braindecode compatible data format.

# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
import mne
import numpy as np

from braindecode.datasets import create_from_mne_epochs, create_from_mne_raw

###############################################################################
# First, fetch some data using mne:

# 5, 6, 7, 10, 13, 14 are codes for executed and imagined hands/feet
subject_id = 22
event_codes = [5, 6, 9, 10, 13, 14]
# event_codes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

# This will download the files if you don't have them yet,
# and then return the paths to the files.
physionet_paths = mne.datasets.eegbci.load_data(
    subject_id, event_codes, update_path=False
)

# Load each of the files
parts = [
    mne.io.read_raw_edf(path, preload=True, stim_channel="auto")
    for path in physionet_paths
]

###############################################################################
# Convert Raw objects to a compatible data format:
descriptions = [{"event_code": code, "subject": subject_id} for code in event_codes]
raw_windows_dataset = create_from_mne_raw(
    parts,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=500,
    window_stride_samples=500,
    drop_last_window=False,
    descriptions=descriptions,
)

###############################################################################
# If trials were already cut beforehand and are available as :class:`mne.Epochs`:
list_of_epochs = [mne.Epochs(raw, [[0, 0, 0]], tmin=0, baseline=None) for raw in parts]
epochs_windows_dataset = create_from_mne_epochs(
    list_of_epochs,
    window_size_samples=50,
    window_stride_samples=50,
    drop_last_window=False,
)

epochs_windows_dataset.description

###############################################################################
# Finally, we draw where the windows of the first recording fall on channel C3.
# The ``metadata`` of each windows dataset gives the start and stop sample of
# every window. Because ``drop_last_window=False``, a trial or epoch that is
# not a multiple of the window size gets one more window, shifted back so that
# it ends at the end of the trial: here the two windows of a trial overlap.

# Importing braindecode.datasets imports MOABB, which applies a global seaborn
# theme (grid, serif font, no tick marks): restore matplotlib's defaults first.
plt.style.use("default")
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)
BLUE, VERMILION, GREY, INK = "#0072B2", "#D55E00", "#7F7F7F", "#14181F"
sfreq = parts[0].info["sfreq"]


def plot_windows(ax, signal_uv, metadata, color, label):
    """Plot a C3 trace with one bar per window below it."""
    times = np.arange(signal_uv.size) / sfreq
    ax.plot(times, signal_uv, color=INK, linewidth=0.6)
    bar_top = signal_uv.min() - 0.1 * np.ptp(signal_uv)
    bar_step = 0.08 * np.ptp(signal_uv)
    for _, window in metadata.iterrows():
        # alternate rows so that overlapping windows stay visible
        y = bar_top - window["i_window_in_trial"] * bar_step
        start_s = window["i_start_in_trial"] / sfreq
        stop_s = window["i_stop_in_trial"] / sfreq
        ax.plot([start_s, stop_s], [y, y], color=color, linewidth=3.5)
    ax.text(
        times[-1],
        bar_top - 1.6 * bar_step,
        label,
        color=color,
        ha="right",
        va="top",
        fontsize=9,
    )
    ax.set_ylim(bar_top - 3 * bar_step, signal_uv.max() + 0.05 * np.ptp(signal_uv))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_xlabel("Time (s)")


fig, (ax_raw, ax_epo) = plt.subplots(
    1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [2.2, 1]}
)

# A: the first three trials of the continuous recording
raw_metadata = raw_windows_dataset.datasets[0].metadata
raw_metadata = raw_metadata[raw_metadata["i_trial_in_dataset"] < 3]
n_shown = raw_metadata["i_stop_in_trial"].max()
raw_uv = parts[0].get_data(picks="C3..")[0, :n_shown] * 1e6  # volts to µV
plot_windows(ax_raw, raw_uv, raw_metadata, BLUE, "500-sample windows")
ax_raw.set_ylabel("C3 amplitude (µV)")
ax_raw.set_title("A  Windows cut from mne.io.Raw", loc="left", fontweight="bold")

# B: the single 0.5 s epoch of the first recording
epoch_uv = list_of_epochs[0].get_data(picks="C3..")[0, 0] * 1e6
epochs_metadata = epochs_windows_dataset.datasets[0].metadata
plot_windows(ax_epo, epoch_uv, epochs_metadata, VERMILION, "50-sample windows")
ax_epo.set_title("B  Windows cut from mne.Epochs", loc="left", fontweight="bold")

fig.tight_layout()
plt.show()
