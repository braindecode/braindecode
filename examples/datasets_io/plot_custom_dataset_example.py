""".. _custom-dataset-example:

Custom Dataset Example
======================

This example shows how to convert data X and y as numpy arrays to a braindecode
compatible data format.
"""

# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

import mne

from braindecode.datasets import create_from_X_y

###############################################################################
# To set up the example, we first fetch some data using `MNE <MNE-Python_>`_:

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
# We take the required data, targets and additional information sampling
# frequency and channel names from the loaded data. Note that this data and
# information can originate from any source.
X = [raw.get_data() for raw in parts]
y = event_codes
sfreq = parts[0].info["sfreq"]
ch_names = parts[0].info["ch_names"]

###############################################################################
# Convert to data format compatible with skorch and braindecode:
windows_dataset = create_from_X_y(
    X,
    y,
    drop_last_window=False,
    sfreq=sfreq,
    ch_names=ch_names,
    window_stride_samples=500,
    window_size_samples=500,
)

windows_dataset.description  # look as dataset description

###############################################################################
# You can manipulate the dataset
print(len(windows_dataset))  # get the number of samples

###############################################################################
# You can now index the data
i = 0
x_i, y_i, window_ind = windows_dataset[0]
n_channels, n_times = x_i.shape  # the EEG data
_, start_ind, stop_ind = window_ind
print(f"n_channels={n_channels}  -- n_times={n_times} -- y_i={y_i}")
print(f"start_ind={start_ind} -- stop_ind={stop_ind}")

###############################################################################
# Finally, we plot this first window on the three motor channels C3, Cz and C4
# to check that ``create_from_X_y`` cut ``window_size_samples=500`` samples,
# that is 500 / 160 Hz = 3.125 s, from the start of the first recording.

import matplotlib.pyplot as plt
import numpy as np

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
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)
BLUE, GREY, INK = "#0072B2", "#7F7F7F", "#14181F"

motor_channels = ["C3..", "Cz..", "C4.."]  # EEGBCI names carry trailing dots
picks = [ch_names.index(ch) for ch in motor_channels]
times = np.arange(start_ind, stop_ind) / sfreq
window_uv = x_i[picks] * 1e6  # volts to microvolts
offset_uv = 1.2 * np.ptp(window_uv)  # vertical spacing between channels

fig, ax = plt.subplots(figsize=(6.4, 3.2))
for i_ch, (ch, trace) in enumerate(zip(motor_channels, window_uv)):
    y_offset = -i_ch * offset_uv
    ax.plot(times, trace + y_offset, color=BLUE, linewidth=0.9)
    ax.text(
        times[0] - 0.05,
        y_offset,
        ch.rstrip("."),
        color=INK,
        ha="right",
        va="center",
    )

# 50 µV vertical scale bar at the right end of the window
scale_uv = 50
x_bar = times[-1] + 0.1
y_bar = -(len(motor_channels) - 1) * offset_uv
ax.plot([x_bar, x_bar], [y_bar, y_bar + scale_uv], color=INK, linewidth=1.5)
ax.text(x_bar + 0.05, y_bar + scale_uv / 2, f"{scale_uv} µV", va="center")

ax.spines["left"].set_visible(False)
ax.set_yticks([])
ax.set_xticks([0, 1, 2, 3])
ax.set_xlabel("Time (s)")
fig.tight_layout()
plt.show()

###############################################################################
# .. include:: /links.inc
