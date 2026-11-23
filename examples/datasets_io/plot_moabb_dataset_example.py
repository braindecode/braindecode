""".. _moabb-dataset-example:

MOABB Dataset Example
========================

In this example, we show how to fetch and prepare a MOABB dataset for usage
with Braindecode.
"""

# Authors: Lukas Gemein <l.gemein@gmail.com>
#          Hubert Banville <hubert.jbanville@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#          Daniel Wilson <dan.c.wil@gmail.com>
#
# License: BSD (3-clause)

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import preprocess

###############################################################################
# First, we create a dataset based on BCIC IV 2a fetched with MOABB,
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[1])

###############################################################################
# The dataset has a pandas DataFrame with additional description of its internal datasets
dataset.description

##############################################################################
# We can iterate through dataset which yields one time point of a continuous signal x,
# and a target y (which can be None if targets are not defined for the entire
# continuous signal).
for x, y in dataset:
    print(x.shape, y)
    break

##############################################################################
# We can apply preprocessing transforms that are defined in mne and work
# in-place. Braindecode provides dedicated preprocessing classes for common
# operations, or you can use the generic Preprocessor class.

# Using the new dedicated preprocessing classes (recommended):
from braindecode.preprocessing import Pick, Resample

preprocessors = [
    Pick(picks="eeg", exclude=()),  # Keep only EEG channels
    Resample(sfreq=100),  # Resample to 100 Hz
]

# Alternative: using the generic Preprocessor class (legacy approach):
# preprocessors = [
#     Preprocessor("pick_types", eeg=True, meg=False, stim=True),
#     Preprocessor("resample", sfreq=100),
# ]

# keep a 2 s excerpt of C3 from the first run to compare before and after
raw_before = dataset.datasets[0].raw.copy().pick(["C3"]).crop(tmin=100, tmax=102)

print(dataset.datasets[0].raw.info["sfreq"])
preprocess(dataset, preprocessors)
print(dataset.datasets[0].raw.info["sfreq"])

raw_after = dataset.datasets[0].raw.copy().pick(["C3"]).crop(tmin=100, tmax=102)

###############################################################################
# We can easily split the dataset based on a criteria applied to the description
# DataFrame:
subsets = dataset.split("session")
print({subset_name: len(subset) for subset_name, subset in subsets.items()})

###############################################################################
# We plot the C3 excerpt before and after resampling (**A**), and the duration
# of each run in the two sessions returned by ``split`` (**B**).

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
BLUE, VERMILION, GREY = "#0072B2", "#D55E00", "#7F7F7F"

fig, (ax_sig, ax_dur) = plt.subplots(
    1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.6, 1]}
)

# A: the same 2 s of C3 at 250 Hz and at 100 Hz
for raw, color, lw, y_label in [
    (raw_before, GREY, 1.6, 0.95),
    (raw_after, BLUE, 1.0, 0.87),
]:
    times = raw.times + 100  # crop restarts times at 0
    ax_sig.plot(times, raw.get_data()[0] * 1e6, color=color, linewidth=lw)
    ax_sig.text(
        0.01,
        y_label,
        f"{raw.info['sfreq']:.0f} Hz",
        color=color,
        transform=ax_sig.transAxes,
        fontweight="bold",
    )
ax_sig.set_xlabel("Time in run (s)")
ax_sig.set_ylabel("C3 amplitude (µV)")
ax_sig.set_xticks([100, 101, 102])
ax_sig.yaxis.set_major_locator(plt.MaxNLocator(4))
ax_sig.set_title("A  Resampling to 100 Hz", loc="left", fontweight="bold")

# B: one horizontal bar per run, grouped by session
colors = {"0train": BLUE, "1test": VERMILION}
y = 0
for session, subset in subsets.items():
    durations = [ds.raw.times[-1] for ds in subset.datasets]
    y_runs = np.arange(y, y + len(durations))
    ax_dur.barh(y_runs, durations, color=colors[session], height=0.7)
    ax_dur.text(
        max(durations) + 5,
        y_runs.mean(),
        f"Session {session}",
        color=colors[session],
        va="center",
    )
    y += len(durations) + 1
ax_dur.invert_yaxis()
ax_dur.set_yticks([])
ax_dur.spines["left"].set_visible(False)
ax_dur.set_xlabel("Run duration (s)")
ax_dur.set_title("B  Runs per session", loc="left", fontweight="bold")

fig.tight_layout()
plt.show()

##############################################################################
# See our :ref:`Trialwise Decoding <bcic-iv-2a-moabb-trial>` and
# :ref:`Cropped Decoding <bcic-iv-2a-moabb-cropped>` examples for
# training with this dataset.
