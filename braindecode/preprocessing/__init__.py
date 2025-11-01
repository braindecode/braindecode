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
- :class:`~braindecode.preprocessing.NotchFilter` : Remove specific frequencies (e.g., 50/60 Hz power line noise)
- :class:`~braindecode.preprocessing.SavgolFilter` : Apply Savitzky-Golay polynomial filter
- :class:`~braindecode.preprocessing.ApplyHilbert` : Compute analytic signal or envelope
- :class:`~braindecode.preprocessing.Rescale` : Rescale channel amplitudes

**Channel Management**

- :class:`~braindecode.preprocessing.Pick` : Select specific channels or channel types
- :class:`~braindecode.preprocessing.DropChannels` : Remove specific channels
- :class:`~braindecode.preprocessing.AddChannels` : Append new channels from other MNE objects
- :class:`~braindecode.preprocessing.RenameChannels` : Rename channels
- :class:`~braindecode.preprocessing.ReorderChannels` : Reorder channels
- :class:`~braindecode.preprocessing.SetChannelTypes` : Specify sensor types of channels
- :class:`~braindecode.preprocessing.InterpolateBads` : Interpolate bad channels
- :class:`~braindecode.preprocessing.InterpolateTo` : Interpolate EEG data onto new montage
- :class:`~braindecode.preprocessing.EqualizeChannels` : Make channel sets identical across datasets

**Reference & Montage**

- :class:`~braindecode.preprocessing.SetEEGReference` : Specify EEG reference
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
    Anonymize,
    ApplyGradientCompensation,
    ApplyHilbert,
    ApplyProj,
    ComputeCurrentSourceDensity,
    Crop,
    CropByAnnotations,
    DelProj,
    DropChannels,
    EqualizeChannels,
    Filter,
    FixMagCoilTypes,
    FixStimArtifact,
    InterpolateBads,
    InterpolateTo,
    NotchFilter,
    Pick,
    RenameChannels,
    ReorderChannels,
    Resample,
    Rescale,
    SavgolFilter,
    SetAnnotations,
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
from .windowers import (
    create_fixed_length_windows,
    create_windows_from_events,
    create_windows_from_target_channels,
)

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
    "ApplyGradientCompensation",
    "ApplyHilbert",
    "ApplyProj",
    "ComputeCurrentSourceDensity",
    "Crop",
    "CropByAnnotations",
    "DelProj",
    "DropChannels",
    "EqualizeChannels",
    "Filter",
    "FixMagCoilTypes",
    "FixStimArtifact",
    "InterpolateBads",
    "InterpolateTo",
    "NotchFilter",
    "Pick",
    "RenameChannels",
    "ReorderChannels",
    "Resample",
    "Rescale",
    "SavgolFilter",
    "SetAnnotations",
    "SetChannelTypes",
    "SetEEGReference",
    "SetMeasDate",
    "SetMontage",
    "Crop",
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
