""".. _process-big-dataset-TUH:

Process a big data EEG resource (TUH EEG Corpus)
================================================

In this example, we showcase usage of the Temple University Hospital EEG Corpus
(https://isip.piconepress.com/projects/nedc/html/tuh_eeg/)
including simple preprocessing steps as well as cutting of compute windows.

.. contents:: This example covers:
   :local:
   :depth: 2


"""

# Authors: Lukas Gemein <l.gemein@gmail.com>
#          Sarthak Tayal <sarthaktayal2@gmail.com>
#
# License: BSD (3-clause)

import tempfile

import matplotlib.pyplot as plt
import mne
import numpy as np
from numpy import multiply

from braindecode.datasets import TUH
from braindecode.preprocessing import (
    Preprocessor,
    create_fixed_length_windows,
    preprocess,
)

mne.set_log_level("ERROR")  # avoid messages every time a window is extracted

###############################################################################
# Creating the dataset using TUHMock
# -------------------------------------
#
# Since the data is not available at the time of the creation of this example,
# we are required to mock some of the dataset functionality. Therefore, if you
# want to try this code with the actual data, please disconsider this section.

from braindecode.datasets.tuh import _TUHMock as TUH  # noqa F811

###############################################################################
# Firstly, we start by creating a TUH mock dataset using braindecode's _TUHMock class.
# The complete code can be found at :func:`braindecode.datasets.TUH`, but we will give
# a small description of how it works.
# This class is able to read the recordings from TUH_PATH and generate a description
# by parsing information from file paths, such as patient id and recording data.
# THis description can later be accessed by the object's .description method.
# After that, the files are sorted chronologically by year, month, day,
# patient id, recording session and segment, and then use the description corresponding
# to the specified by recording ids.
# FInally, additional information regarding age and gender of the subjects are parsed
# directly to the description.

TUH_PATH = "please insert actual path to data here"
# specify the number of jobs for loading and windowing
N_JOBS = 2
tuh = TUH(
    path=TUH_PATH,
    # the mock's v1.1.0 corpus has male and female recordings; its default
    # v1.2.0 corpus has a single female recording
    version="v1.1.0",
    recording_ids=None,
    target_name=None,
    preload=False,
    add_physician_reports=False,
    rename_channels=True,
    set_montage=True,
    n_jobs=1 if TUH.__name__ == "_TUHMock" else N_JOBS,
)


###############################################################################
# We can visualize the age of the subjects, split by sex, using the
# ``description`` DataFrame of the dataset. Each dot is one recording, and the
# vertical bar marks the mean age of each group.

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


def plot_age_by_sex(description):
    groups = {"Male": ("M", "#0072B2"), "Female": ("F", "#D55E00")}
    fig, ax = plt.subplots(figsize=(7, 2.6))
    for i_group, (name, (code, color)) in enumerate(groups.items()):
        ages = description.loc[description["gender"] == code, "age"]
        ax.scatter(ages, np.full(len(ages), i_group), color=color, s=40, alpha=0.8)
        ax.plot([ages.mean()] * 2, [i_group - 0.25, i_group + 0.25], color=color)
        ax.text(
            ages.mean(),
            i_group + 0.3,
            f"mean {ages.mean():.1f} years (n = {len(ages)})",
            color=color,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_yticks(range(len(groups)), list(groups))
    ax.set_ylim(-0.5, len(groups) - 0.1)
    ax.set_xlim(-3, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Age (years)")
    fig.tight_layout()
    plt.show()


plot_age_by_sex(tuh.description)


###############################################################################
# Preprocessing
# -------------------------------------
#
# Selecting recordings
# ~~~~~~~~~~~~~~~~~~~~
#
# First, we will do some selection of available recordings based on the duration.
# We will select those recordings that have at least five minutes duration.
#


def select_by_duration(ds, tmin=0, tmax=None):
    if tmax is None:
        tmax = np.inf
    # determine length of the recordings and select based on tmin and tmax
    split_ids = []
    for d_i, d in enumerate(ds.datasets):
        duration = d.raw.n_times / d.raw.info["sfreq"]
        # select the ones in the required duration range
        if tmin <= duration <= tmax:
            split_ids.append(d_i)
    splits = ds.split(split_ids)
    split = splits["0"]
    return split


tmin = 5 * 60
tmax = None
tuh = select_by_duration(tuh, tmin, tmax)

###############################################################################
# Next, we will discard all recordings that have an incomplete channel
# configuration on the channels that we are interested.

short_ch_names = sorted(
    [
        "A1",
        "A2",
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "F7",
        "F8",
        "T3",
        "T4",
        "T5",
        "T6",
        "Fz",
        "Cz",
        "Pz",
    ]
)


def select_by_channels(ds, ch_names):
    # these are the channels we are looking for
    seta = set(ch_names)
    split_ids = []
    for i, d in enumerate(ds.datasets):
        # these are the channels of the recoding
        setb = set(d.raw.ch_names)
        # if recording contains all channels we are looking for, include it
        if seta.issubset(setb):
            split_ids.append(i)
    return ds.split(split_ids)["0"]


tuh = select_by_channels(tuh, short_ch_names)


###############################################################################
# Combining preprocessing steps
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Next, we use braindecode's preprocess to combine and execute several preprocessing
# steps that are executed through 'mne':
#
# - Crop the recordings to a region of interest
# - Pick channels of interest
# - Re-reference all recordings to average reference (CAR) (requires load)
# - Scale signals to micro volts (requires load)
# - Clip outlier values to +/- 800 micro volts (requires load)
# - Resample recordings to a common frequency (requires load)


def custom_crop(raw, tmin=0.0, tmax=None, include_tmax=True):
    # crop recordings to tmin – tmax. can be incomplete if recording
    # has lower duration than tmax
    # by default mne fails if tmax is bigger than duration
    tmax = min((raw.n_times - 1) / raw.info["sfreq"], tmax)
    raw.crop(tmin=tmin, tmax=tmax, include_tmax=include_tmax)


tmin = 1 * 60
tmax = 6 * 60
sfreq = 100
factor = 1e6

preprocessors = [
    Preprocessor(
        custom_crop, tmin=tmin, tmax=tmax, include_tmax=False, apply_on_array=False
    ),
    # pick first so the average reference does not pull in artifacts from dropped channels
    Preprocessor("pick_channels", ch_names=short_ch_names, ordered=True),
    Preprocessor("set_eeg_reference", ref_channels="average", ch_type="eeg"),
    Preprocessor(
        lambda data: multiply(data, factor), apply_on_array=True
    ),  # Convert from V to uV
    Preprocessor(lambda x: np.clip(x, a_min=-800, a_max=800), apply_on_array=True),
    Preprocessor("resample", sfreq=sfreq),
]

###############################################################################
# Next, we can apply the defined preprocessors on the selected recordings in parallel.
# We additionally use the serialization functionality of
# :func:`braindecode.preprocessing.preprocess` to limit memory usage during
# preprocessing, as each file must be loaded into memory for some of the
# preprocessing steps to work.
# This also makes it possible to use the lazy
# loading capabilities of :class:`braindecode.datasets.BaseConcatDataset`, as
# the preprocessed data is automatically reloaded with ``preload=False``.
#
# .. note::
#    Here we use ``n_jobs=2`` as the machines the documentation is build on
#    only have two cores. This number should be modified based on the machine
#    that is available for preprocessing.

OUT_PATH = tempfile.mkdtemp()  # please insert actual output directory here
tuh_preproc = preprocess(
    concat_ds=tuh, preprocessors=preprocessors, n_jobs=N_JOBS, save_dir=OUT_PATH
)

###############################################################################
# Cut Compute Windows
# ~~~~~~~~~~~~~~~~~~~
# We can finally generate compute windows. The resulting dataset is now ready
# to be used for model training.

window_size_samples = 1000
window_stride_samples = 1000
# Generate compute windows here and store them to disk
tuh_windows = create_fixed_length_windows(
    tuh_preproc,
    window_size_samples=window_size_samples,
    window_stride_samples=window_stride_samples,
    on_last_window="overlap",
    n_jobs=N_JOBS,
)
