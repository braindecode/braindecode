""".. _split-dataset-example:

Split Dataset Example
=====================

In this example, we aim to show multiple ways of how you can split your datasets for
training, testing, and evaluating your models.

.. contents:: This example covers:
   :local:
   :depth: 2

"""

# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events

###############################################################################
# Loading the dataset
# -------------------------------------
#
# Firstly, we create a dataset using the braindecode
# :class:`MOABBDataset <braindecode.datasets.MOABBDataset>` to load
# it fetched from MOABB. In this example, we're using Dataset 2a from BCI
# Competition IV.

dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[1])

###############################################################################
# Splitting
# -------------------------------------
#
# By description information
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The class :class:`MOABBDataset <braindecode.datasets.MOABBDataset>` has a pandas
# DataFrame containing additional description of its internal datasets,
# which can be used to help splitting the data
# based on recording information, such as subject, session, and run of each trial.

dataset.description

###############################################################################
# Here, we're splitting the data based on different runs. The method split returns
# a dictionary with string keys corresponding to unique entries in the description
# DataFrame column.

splits = dataset.split("run")
print(splits)
splits["4"].description

###############################################################################
# By row index
# ~~~~~~~~~~~~~
#
# Another way we can split the dataset is based on a list of integers corresponding to
# rows in the description. In this case, the returned dictionary will have
# '0' as the only key.

splits = dataset.split([0, 1, 5])
print(splits)
splits["0"].description

###############################################################################
# However, if we want multiple splits based on indices, we can also define a list
# containing lists of integers. In this case, the dictionary will have string keys
# representing the index of the dataset split in the order of the given list of
# integers.

splits = dataset.split([[0, 1, 5], [2, 3, 4], [6, 7, 8, 9, 10, 11]])
print(splits)
splits["2"].description

###############################################################################
# You can also name each split in the output dictionary by specifying the keys
# of each list of indexes in the input dictionary:

splits = dataset.split(
    {"train": [0, 1, 5], "valid": [2, 3, 4], "test": [6, 7, 8, 9, 10, 11]}
)
print(splits)
splits["test"].description

###############################################################################
# The figure below shows, for each of these four calls, the split key that
# every recording (row of ``dataset.description``) ends up in. Grey cells are
# recordings that the call leaves out.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

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
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)
OKABE_ITO = ["#0072B2", "#009E73", "#D55E00", "#E69F00", "#CC79A7", "#56B4E9"]
EXCLUDED = "#E5E5E5"

split_calls = {
    "By run": "run",
    "By row index": [0, 1, 5],
    "By lists of indices": [[0, 1, 5], [2, 3, 4], [6, 7, 8, 9, 10, 11]],
    "By named lists": {
        "train": [0, 1, 5],
        "valid": [2, 3, 4],
        "test": [6, 7, 8, 9, 10, 11],
    },
}
n_recordings = len(dataset.datasets)

fig, ax = plt.subplots(figsize=(8, 2.8))
for i_row, (name, by) in enumerate(split_calls.items()):
    # split reuses the recordings, so we can find each one by identity
    key_of = {}
    for i_key, (key, subset) in enumerate(dataset.split(by).items()):
        for ds in subset.datasets:
            i_rec = next(i for i, d in enumerate(dataset.datasets) if d is ds)
            key_of[i_rec] = (key, OKABE_ITO[i_key])
    for i_rec in range(n_recordings):
        key, color = key_of.get(i_rec, ("", EXCLUDED))
        ax.add_patch(
            plt.Rectangle((i_rec, i_row), 0.92, 0.85, color=color, linewidth=0)
        )
        # white text on dark fills, black text on light ones
        text_color = "white" if np.mean(to_rgb(color)) < 0.5 else "black"
        ax.text(
            i_rec + 0.46,
            i_row + 0.43,
            key,
            color=text_color,
            ha="center",
            va="center",
            fontsize=8,
        )

ax.set_xlim(-0.1, n_recordings)
ax.set_ylim(len(split_calls), -0.15)
ax.set_xticks(np.arange(n_recordings) + 0.46, labels=range(n_recordings))
ax.set_yticks(np.arange(len(split_calls)) + 0.43, labels=list(split_calls))
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xlabel("Recording (row of dataset.description)")
fig.tight_layout()
plt.show()

###############################################################################
# Observation
# -------------------------------------
#
# Similarly, we can split datasets after creating windows using the same methods.

windows = create_windows_from_events(
    dataset, trial_start_offset_samples=0, trial_stop_offset_samples=0
)

###############################################################################

# Splitting by different runs
print("Using description info")
splits = windows.split("run")
print(splits)
print()

# Splitting by row index
print("Splitting by row index")
splits = windows.split([4, 8])
print(splits)
print()

print("Multiple row index split")
splits = windows.split([[4, 8], [5, 9, 11]])
print(splits)
print()

# Specifying output's keys
print("Specifying keys")
splits = windows.split(dict(train=[4, 8], test=[5, 9, 11]))
print(splits)
