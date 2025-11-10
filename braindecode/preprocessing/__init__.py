"""Preprocessing module for EEG data.

This module provides preprocessing functionality for EEG data through:

1. **MNE-based preprocessing classes**: Automatically generated classes that wrap
   MNE-Python functions for easy integration with Braindecode workflows.
2. **Custom preprocessing functions**: Specialized functions for EEG preprocessing.
3. **Windowing functions**: Tools for creating epochs/windows from continuous data.

MNE Preprocessing Classes
--------------------------
The following preprocessing classes are automatically generated from MNE functions.
Each class inherits from :class:`~braindecode.preprocessing.Preprocessor` and can be
used with :func:`~braindecode.preprocessing.preprocess`.

**Signal Processing**

- :class:`~braindecode.preprocessing.Resample` : Resample data to different sampling frequency
- :class:`~braindecode.preprocessing.Filter` : Apply bandpass, highpass, or lowpass filter
- :class:`~braindecode.preprocessing.FilterData` : Low-level filter function for data arrays
- :class:`~braindecode.preprocessing.NotchFilter` : Remove specific frequencies (e.g., 50/60 Hz power line noise)
- :class:`~braindecode.preprocessing.SavgolFilter` : Apply Savitzky-Golay polynomial filter
- :class:`~braindecode.preprocessing.ApplyHilbert` : Compute analytic signal or envelope
- :class:`~braindecode.preprocessing.Rescale` : Rescale channel amplitudes
- :class:`~braindecode.preprocessing.OversampledTemporalProjection` : Apply oversampled temporal projection

**Channel Management**

- :class:`~braindecode.preprocessing.Pick` : Select specific channels or channel types
- :class:`~braindecode.preprocessing.PickChannels` : Pick channels by name
- :class:`~braindecode.preprocessing.PickTypes` : Pick channels by type (EEG, MEG, etc.)
- :class:`~braindecode.preprocessing.DropChannels` : Remove specific channels
- :class:`~braindecode.preprocessing.AddChannels` : Append new channels from other MNE objects
- :class:`~braindecode.preprocessing.CombineChannels` : Combine data from multiple channels
- :class:`~braindecode.preprocessing.RenameChannels` : Rename channels
- :class:`~braindecode.preprocessing.ReorderChannels` : Reorder channels
- :class:`~braindecode.preprocessing.SetChannelTypes` : Specify sensor types of channels
- :class:`~braindecode.preprocessing.InterpolateBads` : Interpolate bad channels
- :class:`~braindecode.preprocessing.InterpolateTo` : Interpolate EEG data onto new montage
- :class:`~braindecode.preprocessing.InterpolateBridgedElectrodes` : Interpolate bridged electrodes
- :class:`~braindecode.preprocessing.ComputeBridgedElectrodes` : Identify bridged electrodes
- :class:`~braindecode.preprocessing.EqualizeChannels` : Make channel sets identical across datasets
- :class:`~braindecode.preprocessing.EqualizeBads` : Equalize bad channels across instances
- :class:`~braindecode.preprocessing.FindBadChannelsLof` : Find bad channels using LOF algorithm

**Reference & Montage**

- :class:`~braindecode.preprocessing.SetEEGReference` : Specify EEG reference (Raw method)
- :class:`~braindecode.preprocessing.SetBipolarReference` : Set bipolar reference
- :class:`~braindecode.preprocessing.AddReferenceChannels` : Add zero-filled reference channels
- :class:`~braindecode.preprocessing.SetMontage` : Set channel positions/montage

**SSP Projections**

- :class:`~braindecode.preprocessing.AddProj` : Add SSP projection vectors
- :class:`~braindecode.preprocessing.ApplyProj` : Apply SSP operators
- :class:`~braindecode.preprocessing.DelProj` : Remove SSP projection vector

**Data Transformation**

- :class:`~braindecode.preprocessing.Crop` : Crop data to specific time range
- :class:`~braindecode.preprocessing.CropByAnnotations` : Crop data by annotations
- :class:`~braindecode.preprocessing.ComputeCurrentSourceDensity` : Apply CSD transformation
- :class:`~braindecode.preprocessing.FixStimArtifact` : Remove stimulation artifacts
- :class:`~braindecode.preprocessing.MaxwellFilter` : Apply Maxwell filtering (for MEG data)
- :class:`~braindecode.preprocessing.RealignRaw` : Realign raw data
- :class:`~braindecode.preprocessing.RegressArtifact` : Regress out artifacts

**Artifact Detection & Annotation**

- :class:`~braindecode.preprocessing.AnnotateAmplitude` : Annotate periods based on amplitude
- :class:`~braindecode.preprocessing.AnnotateBreak` : Annotate breaks in the data
- :class:`~braindecode.preprocessing.AnnotateMovement` : Annotate movement artifacts
- :class:`~braindecode.preprocessing.AnnotateMuscleZscore` : Annotate muscle artifacts using z-score
- :class:`~braindecode.preprocessing.AnnotateNan` : Annotate NaN values in data

**Metadata & Configuration**

- :class:`~braindecode.preprocessing.Anonymize` : Anonymize measurement information
- :class:`~braindecode.preprocessing.SetAnnotations` : Set annotations
- :class:`~braindecode.preprocessing.SetMeasDate` : Set measurement start date
- :class:`~braindecode.preprocessing.AddEvents` : Add events to stim channel
- :class:`~braindecode.preprocessing.FixMagCoilTypes` : Fix Elekta magnetometer coil types
- :class:`~braindecode.preprocessing.ApplyGradientCompensation` : Apply CTF gradient compensation

Usage Examples
--------------
Using the new preprocessing classes::

    from braindecode.preprocessing import (
        Resample, Filter, NotchFilter, SetEEGReference, preprocess
    )

    preprocessors = [
        Resample(sfreq=100),
        NotchFilter(freqs=[50]),  # Remove 50 Hz power line noise
        Filter(l_freq=4, h_freq=30),  # Bandpass filter
        SetEEGReference(ref_channels='average'),
    ]
    preprocess(dataset, preprocessors)

Using the generic Preprocessor class (legacy approach)::

    from braindecode.preprocessing import Preprocessor, preprocess

    preprocessors = [
        Preprocessor('resample', sfreq=100),
        Preprocessor('filter', l_freq=4, h_freq=30),
    ]
    preprocess(dataset, preprocessors)

See Also
--------
:class:`braindecode.preprocessing.Preprocessor` : Base class for all preprocessors
:func:`braindecode.preprocessing.preprocess` : Apply preprocessors to datasets
:func:`braindecode.preprocessing.create_windows_from_events` : Create epochs from events
:func:`braindecode.preprocessing.create_fixed_length_windows` : Create fixed-length epochs
"""

from .eegprep_preprocess import (
    EEGPrep,
    ReinterpolateRemovedChannels,
    RemoveBadChannels,
    RemoveBadChannelsNoLocs,
    RemoveBadWindows,
    RemoveBursts,
    RemoveCommonAverageReference,
    RemoveDCOffset,
    RemoveDrifts,
    RemoveFlatChannels,
    Resampling,
)
from .mne_preprocess import (  # type: ignore[attr-defined]
    AddChannels,
    AddEvents,
    AddProj,
    AddReferenceChannels,
    AnnotateAmplitude,
    AnnotateBreak,
    AnnotateMovement,
    AnnotateMuscleZscore,
    AnnotateNan,
    Anonymize,
    ApplyGradientCompensation,
    ApplyHilbert,
    ApplyProj,
    CombineChannels,
    ComputeBridgedElectrodes,
    ComputeCurrentSourceDensity,
    Crop,
    CropByAnnotations,
    DelProj,
    DropChannels,
    EqualizeBads,
    EqualizeChannels,
    Filter,
    FilterData,
    FindBadChannelsLof,
    FixMagCoilTypes,
    FixStimArtifact,
    InterpolateBads,
    InterpolateBridgedElectrodes,
    InterpolateTo,
    MaxwellFilter,
    NotchFilter,
    OversampledTemporalProjection,
    Pick,
    PickChannels,
    PickTypes,
    RealignRaw,
    RegressArtifact,
    RenameChannels,
    ReorderChannels,
    Resample,
    Rescale,
    SavgolFilter,
    SetAnnotations,
    SetBipolarReference,
    SetChannelTypes,
    SetEEGReference,
    SetMeasDate,
    SetMontage,
)
from .preprocess import (
    Preprocessor,
    exponential_moving_demean,
    exponential_moving_standardize,
    filterbank,
    preprocess,
)
from .util import _init_preprocessor_dict
from .windowers import (
    create_fixed_length_windows,
    create_windows_from_events,
    create_windows_from_target_channels,
)

# Call this last in order to make sure the list is populated with
# the preprocessors imported in this file.
_init_preprocessor_dict()

__all__ = [
    "exponential_moving_demean",
    "exponential_moving_standardize",
    "filterbank",
    "preprocess",
    "Preprocessor",
    "AddChannels",
    "AddEvents",
    "AddProj",
    "AddReferenceChannels",
    "Anonymize",
    "AnnotateAmplitude",
    "AnnotateBreak",
    "AnnotateMovement",
    "AnnotateMuscleZscore",
    "AnnotateNan",
    "ApplyGradientCompensation",
    "ApplyHilbert",
    "ApplyProj",
    "CombineChannels",
    "ComputeBridgedElectrodes",
    "ComputeCurrentSourceDensity",
    "Crop",
    "CropByAnnotations",
    "DelProj",
    "DropChannels",
    "EqualizeBads",
    "EqualizeChannels",
    "Filter",
    "FilterData",
    "FindBadChannelsLof",
    "FixMagCoilTypes",
    "FixStimArtifact",
    "InterpolateBads",
    "InterpolateBridgedElectrodes",
    "InterpolateTo",
    "MaxwellFilter",
    "NotchFilter",
    "OversampledTemporalProjection",
    "Pick",
    "PickChannels",
    "PickTypes",
    "RealignRaw",
    "RegressArtifact",
    "RenameChannels",
    "ReorderChannels",
    "Resample",
    "Rescale",
    "SavgolFilter",
    "SetAnnotations",
    "SetBipolarReference",
    "SetChannelTypes",
    "SetEEGReference",
    "SetMeasDate",
    "SetMontage",
    "EEGPrep",
    "RemoveDCOffset",
    "Resampling",
    "RemoveFlatChannels",
    "RemoveDrifts",
    "RemoveBadChannels",
    "RemoveBadChannelsNoLocs",
    "RemoveBursts",
    "RemoveBadWindows",
    "ReinterpolateRemovedChannels",
    "RemoveCommonAverageReference",
    "create_windows_from_events",
    "create_fixed_length_windows",
    "create_windows_from_target_channels",
]
