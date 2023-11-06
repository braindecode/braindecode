"""
Utilities for data augmentation.
"""
from .base import Transform, IdentityTransform, Compose, AugmentedDataLoader
from .transforms import (
    TimeReverse,
    SignFlip,
    FTSurrogate,
    ChannelsShuffle,
    ChannelsDropout,
    GaussianNoise,
    ChannelsSymmetry,
    SmoothTimeMask,
    BandstopFilter,
    FrequencyShift,
    SensorsRotation,
    SensorsZRotation,
    SensorsYRotation,
    SensorsXRotation,
    Mixup,
)

from . import functional

__all__ = ["Transform", "IdentityTransform", "Compose", "AugmentedDataLoader",
           "TimeReverse", "SignFlip", "FTSurrogate", "ChannelsShuffle",
           "ChannelsDropout", "GaussianNoise", "ChannelsSymmetry",
           "SmoothTimeMask", "BandstopFilter", "FrequencyShift",
           "SensorsRotation", "SensorsZRotation", "SensorsYRotation",
           "SensorsXRotation", "Mixup", "functional"]
