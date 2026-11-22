""".. _preprocessing-classes:

Comprehensive Preprocessing with MNE-based Classes
===================================================

This example demonstrates the various preprocessing classes available in
Braindecode that wrap MNE-Python functionality. These classes provide a
convenient and type-safe way to preprocess EEG data.
"""

# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
import mne
import numpy as np

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Anonymize,
    ApplyHilbert,
    Crop,
    Filter,
    Pick,
    Resample,
    SetEEGReference,
    SetMontage,
    preprocess,
)

###############################################################################
# Load a sample dataset
# ---------------------
# We'll use a small MOABB dataset for demonstration

dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[1])

###############################################################################
# Signal Processing
# -----------------
# Apply common signal processing operations

# 1. Resample to reduce computational load
print(f"Original sampling frequency: {dataset.datasets[0].raw.info['sfreq']} Hz")
preprocessors_signal = [
    Resample(sfreq=100),  # Downsample to 100 Hz
]
preprocess(dataset, preprocessors_signal)
print(f"After resampling: {dataset.datasets[0].raw.info['sfreq']} Hz")

# 2. Remove power line noise and apply bandpass filter
preprocessors_filtering = [
    Filter(l_freq=4, h_freq=30),  # Bandpass filter 4-30 Hz
]
preprocess(dataset, preprocessors_filtering)
print("Applied bandpass filter 4-30 Hz")

###############################################################################
# Channel Management
# ------------------
# Select and manipulate channels

# 3. Pick only EEG channels
preprocessors_channels = [
    Pick(picks="eeg"),  # Select only EEG channels
]
print(f"Channels before pick: {len(dataset.datasets[0].raw.ch_names)}")
preprocess(dataset, preprocessors_channels)
print(f"Channels after pick: {len(dataset.datasets[0].raw.ch_names)}")

# 4. Rename channels (example - just for demonstration)
original_names = dataset.datasets[0].raw.ch_names[:3]
print(f"Original channel names (first 3): {original_names}")
# Note: We won't actually rename to avoid breaking the example,
# but this is how you would do it:
# preprocessors_rename = [
#     RenameChannels(mapping={'C3': 'C3_renamed', 'C4': 'C4_renamed'}),
# ]
# preprocess(dataset, preprocessors_rename)

###############################################################################
# Reference & Montage
# -------------------
# Set reference and channel positions

# 5. Set EEG reference to average
preprocessors_reference = [
    SetEEGReference(ref_channels="average"),
]
preprocess(dataset, preprocessors_reference)
print("Set EEG reference to average")

# 6. Set montage for proper channel positions
montage = mne.channels.make_standard_montage("colin27_1020")
preprocessors_montage = [
    SetMontage(montage=montage, match_case=False, on_missing="ignore"),
]
preprocess(dataset, preprocessors_montage)
print(
    f"Set montage, number of positions: {len(dataset.datasets[0].raw.get_montage().get_positions()['ch_pos'])}"
)

###############################################################################
# Data Transformation
# -------------------
# Apply transformations to the data

# 7. Crop data to specific time range
preprocessors_crop = [
    Crop(tmin=0, tmax=60),  # Keep only first 60 seconds
]
print(f"Data duration before crop: {dataset.datasets[0].raw.times[-1]:.1f} s")
preprocess(dataset, preprocessors_crop)
print(f"Data duration after crop: {dataset.datasets[0].raw.times[-1]:.1f} s")

###############################################################################
# Metadata & Configuration
# -------------------------
# Modify metadata and configuration

# 8. Anonymize measurement information
preprocessors_anonymize = [
    Anonymize(),
]
preprocess(dataset, preprocessors_anonymize)
print("Anonymized measurement information")

###############################################################################
# Advanced: Envelope Extraction
# ------------------------------
# Extract signal envelope using Hilbert transform

# 9. Compute envelope (useful for some analyses)
# Note: This modifies the data, so use carefully
preprocessors_envelope = [
    ApplyHilbert(envelope=True),
]
preprocess(dataset, preprocessors_envelope)
print("Computed signal envelope")

###############################################################################
# Combining Multiple Preprocessing Steps
# ---------------------------------------
# You can combine multiple preprocessing steps in a single pipeline

print("\n" + "=" * 60)
print("Complete Preprocessing Pipeline Example")
print("=" * 60)

# Reload dataset for complete pipeline demonstration
dataset_complete = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[1])

# Set montage first (needed for interpolation)
montage = mne.channels.make_standard_montage("colin27_1020")

complete_pipeline = [
    # 1. Set montage
    SetMontage(montage=montage, match_case=False, on_missing="ignore"),
    # 2. Set reference
    SetEEGReference(ref_channels="average"),
    # 3. Bandpass filter
    Filter(l_freq=4, h_freq=30),
    # 4. Downsample
    Resample(sfreq=100),
    # 5. Select only EEG channels
    Pick(picks="eeg"),
    # 6. Crop to region of interest
    Crop(tmin=0, tmax=60),
    # 7. Anonymize
    Anonymize(),
]

print(
    f"Original: {dataset_complete.datasets[0].raw.info['sfreq']} Hz, "
    f"{dataset_complete.datasets[0].raw.times[-1]:.1f} s, "
    f"{len(dataset_complete.datasets[0].raw.ch_names)} channels"
)

# Keep an untouched copy of the first run to compare with after preprocessing
raw_before = dataset_complete.datasets[0].raw.copy().load_data()

preprocess(dataset_complete, complete_pipeline)

print(
    f"After preprocessing: {dataset_complete.datasets[0].raw.info['sfreq']} Hz, "
    f"{dataset_complete.datasets[0].raw.times[-1]:.1f} s, "
    f"{len(dataset_complete.datasets[0].raw.ch_names)} channels"
)

print("\nPreprocessing complete!")

###############################################################################
# Before and after the pipeline
# -----------------------------
# Panel A shows two seconds of channel C3 before the pipeline, after it, and
# the Hilbert envelope computed in the first part of this example. Panel B
# compares the power spectra up to 50 Hz, the highest frequency left at
# the new 100 Hz sampling rate, averaged over the EEG channels. The shaded
# band is the 4-30 Hz pass band of the filter.

BLUE, VERMILION, INK = "#0072B2", "#D55E00", "#14181F"
GREY, LIGHT_GREY = "#7F7F7F", "#B0B0B0"
plt.rcdefaults()  # importing MOABB applies seaborn's "whitegrid" style globally
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)

raw_after = dataset_complete.datasets[0].raw
raw_envelope = dataset.datasets[0].raw
t_start, t_stop = 10.0, 12.0  # seconds into the run


def segment_uv(raw, channel="C3"):
    """Return times and microvolt values of one channel, t_start to t_stop."""
    data, times = (
        raw.copy()
        .pick([channel])
        .get_data(tmin=t_start, tmax=t_stop, return_times=True)
    )
    return times, 1e6 * data[0]


fig, (ax_time, ax_psd) = plt.subplots(
    1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1.4, 1]}
)
times, before = segment_uv(raw_before)
ax_time.plot(times, before - before.mean(), color=LIGHT_GREY, lw=0.8)
for raw, color in [(raw_after, BLUE), (raw_envelope, VERMILION)]:
    ax_time.plot(*segment_uv(raw), color=color, lw=1.2)
for text, y, color in [
    ("Raw (mean removed)", 0.98, GREY),
    ("After pipeline (4-30 Hz, 100 Hz)", 0.90, BLUE),
    ("Hilbert envelope", 0.82, VERMILION),
]:
    ax_time.text(0.01, y, text, color=color, transform=ax_time.transAxes, va="top")
y_low, y_high = ax_time.get_ylim()
ax_time.set_ylim(y_low, y_high + 0.45 * (y_high - y_low))  # room for the names
ax_time.xaxis.set_major_locator(plt.MaxNLocator(5))
ax_time.set_xlabel("Time (s)")
ax_time.set_ylabel("C3 amplitude (µV)")
ax_time.set_title("A  Channel C3", loc="left", fontweight="bold", color=INK)

for raw, color, label in [
    (raw_before, GREY, "Raw"),
    (raw_after, BLUE, "After pipeline"),
]:
    psd = raw.compute_psd(method="welch", picks="eeg", fmax=50, verbose=False)
    power, freqs = psd.get_data(return_freqs=True)
    power_db = 10 * np.log10(1e12 * power.mean(axis=0))  # V²/Hz to dB re 1 µV²/Hz
    ax_psd.plot(freqs, power_db, color=color, lw=1.5)
    ax_psd.text(freqs[-1], power_db[-1], f" {label}", color=color, va="center")
ax_psd.axvspan(4, 30, color="#EAEAEA", lw=0, zorder=0)
ax_psd.yaxis.set_major_locator(plt.MaxNLocator(4))
ax_psd.set_xlabel("Frequency (Hz)")
ax_psd.set_ylabel("Power (dB re 1 µV²/Hz)")
ax_psd.set_title("B  Welch power spectrum", loc="left", fontweight="bold", color=INK)
fig.tight_layout()

###############################################################################
# Summary
# -------
# Braindecode provides 45 preprocessing classes that wrap MNE-Python
# functionality:
#
# **Signal Processing**: Resample, Filter, NotchFilter, SavgolFilter,
# ApplyHilbert, Rescale, OversampledTemporalProjection
#
# **Channel Management**: Pick, PickChannels, PickTypes, DropChannels,
# AddChannels, CombineChannels, RenameChannels, ReorderChannels,
# SetChannelTypes, InterpolateBads, InterpolateTo, InterpolateBridgedElectrodes,
# ComputeBridgedElectrodes, EqualizeChannels
#
# **Reference & Montage**: SetEEGReference, AddReferenceChannels, SetMontage
#
# **SSP Projections**: AddProj, ApplyProj, DelProj
#
# **Data Transformation**: Crop, CropByAnnotations, ComputeCurrentSourceDensity,
# FixStimArtifact, MaxwellFilter, RealignRaw, RegressArtifact
#
# **Artifact Detection & Annotation**: AnnotateAmplitude, AnnotateBreak,
# AnnotateMovement, AnnotateMuscleZscore, AnnotateNan
#
# **Metadata & Configuration**: Anonymize, SetAnnotations, SetMeasDate,
# AddEvents, FixMagCoilTypes, ApplyGradientCompensation
#
# See the API documentation for details on each class and their parameters.
