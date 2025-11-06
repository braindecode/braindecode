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

import mne

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

dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[1])

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
montage = mne.channels.make_standard_montage("standard_1020")
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
dataset_complete = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[1])

# Set montage first (needed for interpolation)
montage = mne.channels.make_standard_montage("standard_1020")

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

preprocess(dataset_complete, complete_pipeline)

print(
    f"After preprocessing: {dataset_complete.datasets[0].raw.info['sfreq']} Hz, "
    f"{dataset_complete.datasets[0].raw.times[-1]:.1f} s, "
    f"{len(dataset_complete.datasets[0].raw.ch_names)} channels"
)

print("\nPreprocessing complete!")

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
