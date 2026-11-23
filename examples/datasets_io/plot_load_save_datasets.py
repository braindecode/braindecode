""".. _load-save-dataset:

Load and save dataset example
=============================

In this example, we show how to load and save braindecode datasets.
"""

# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

import tempfile

import matplotlib.pyplot as plt
import numpy as np

from braindecode.datasets import MOABBDataset
from braindecode.datautil import load_concat_dataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)

###############################################################################
# First, we load some dataset using MOABB.
dataset = MOABBDataset(
    dataset_name="BNCI2014_001",
    subject_ids=[1],
)

###############################################################################
# We can apply preprocessing steps to the dataset. It is also possible to skip
# this step and not apply any preprocessing.
preprocess(concat_ds=dataset, preprocessors=[Preprocessor(fn="resample", sfreq=10)])

###############################################################################
# We save the dataset to a an existing directory. It will create a '.fif' file
# for every dataset in the concat dataset. Additionally it will create two
# JSON files, the first holding the description of the dataset, the second
# holding the name of the target. If you want to store to the same directory
# several times, for example due to trying different preprocessing, you can
# choose to overwrite the existing files.

tmpdir = tempfile.mkdtemp()  # write in a temporary directory
dataset.save(
    path=tmpdir,
    overwrite=False,
)

##############################################################################
# We load the saved dataset from a directory. Signals can be preloaded in
# compliance with mne. Optionally, only specific '.fif' files can be loaded
# by specifying their ids. The target name can be changed, if the dataset
# supports it (TUHAbnormal for example supports 'pathological', 'age', and
# 'gender'. If you stored a preprocessed version with target 'pathological'
# it is possible to change the target upon loading).
dataset_loaded = load_concat_dataset(
    path=tmpdir,
    preload=True,
    ids_to_load=[1, 3],
    target_name=None,
)

##############################################################################
# The serialization utility also supports WindowsDatasets, so we create
# compute windows next.
windows_dataset = create_windows_from_events(
    concat_ds=dataset_loaded,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
)

windows_dataset.description

##############################################################################
# Again, we save the dataset to an existing directory. It will create a
# '-epo.fif' file for every dataset in the concat dataset. Additionally it
# will create a JSON file holding the description of the dataset. If you
# want to store to the same directory several times, for example due to
# trying different windowing parameters, you can choose to overwrite the
# existing files.
windows_dataset.save(
    path=tmpdir,
    overwrite=True,
)

##############################################################################
# Load the saved dataset from a directory. Signals can be preloaded in
# compliance with mne. Optionally, only specific '-epo.fif' files can be
# loaded by specifying their ids.
windows_dataset_loaded = load_concat_dataset(
    path=tmpdir,
    preload=False,
    ids_to_load=[0],
    target_name=None,
)

windows_dataset_loaded.description

##############################################################################
# To check the round trip, we overlay channel C3 as it was saved and as it was
# reloaded, for the continuous recording of run 1 (**A**) and for its first
# window (**B**). The reloaded signals lie on top of the saved ones; the
# printed maximum differences are far below the signal amplitude.

c3 = dataset_loaded.datasets[0].raw.ch_names.index("C3")
sfreq = dataset_loaded.datasets[0].raw.info["sfreq"]
saved_raw = dataset.datasets[1].raw.get_data(picks="C3")[0]  # run 1 = row 1
loaded_raw = dataset_loaded.datasets[0].raw.get_data(picks="C3")[0]
saved_window = windows_dataset[0][0][c3]
loaded_window = windows_dataset_loaded[0][0][c3]
print(f"Max difference, recording: {np.abs(saved_raw - loaded_raw).max():.1e} V")
print(f"Max difference, window: {np.abs(saved_window - loaded_window).max():.1e} V")

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
BLUE, GREY = "#0072B2", "#7F7F7F"

fig, (ax_raw, ax_win) = plt.subplots(
    1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [2, 1]}
)
n_shown = int(20 * sfreq)  # first 20 s of the recording
pairs = [
    (ax_raw, saved_raw[:n_shown], loaded_raw[:n_shown]),
    (ax_win, saved_window, loaded_window),
]
for ax, saved, loaded in pairs:
    times = np.arange(saved.size) / sfreq
    ax.plot(times, saved * 1e6, color=GREY, linewidth=3.0)
    ax.plot(times, loaded * 1e6, color=BLUE, linewidth=1.0)
    ax.text(0.01, 0.95, "Saved", color=GREY, transform=ax.transAxes, weight="bold")
    ax.text(0.01, 0.86, "Reloaded", color=BLUE, transform=ax.transAxes, weight="bold")
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + 0.3 * (y_max - y_min))  # room for the labels
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_xlabel("Time (s)")
ax_raw.set_ylabel("C3 amplitude (µV)")
ax_raw.set_title("A  Recording (run 1)", loc="left", fontweight="bold")
ax_win.set_title("B  First window", loc="left", fontweight="bold")
fig.tight_layout()
plt.show()

###############################################################################
# .. include:: /links.inc
