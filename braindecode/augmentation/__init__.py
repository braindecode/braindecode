"""
Utilities for data augmentation.
"""

from . import functional
from .base import AugmentedDataLoader, Compose, IdentityTransform, Transform
from .transforms import (
    BandstopFilter,
    ChannelsDropout,
    ChannelsShuffle,
    ChannelsSymmetry,
    FrequencyShift,
    FTSurrogate,
    GaussianNoise,
    MaskEncoding,
    Mixup,
    SegmentationReconstruction,
    SensorsRotation,
    SensorsXRotation,
    SensorsYRotation,
    SensorsZRotation,
    SignFlip,
    SmoothTimeMask,
    TimeReverse,
)

__all__ = [
    "Transform",
    "IdentityTransform",
    "Compose",
    "AugmentedDataLoader",
    "TimeReverse",
    "SignFlip",
    "FTSurrogate",
    "ChannelsShuffle",
    "ChannelsDropout",
    "GaussianNoise",
    "ChannelsSymmetry",
    "SmoothTimeMask",
    "BandstopFilter",
    "FrequencyShift",
    "SensorsRotation",
    "SensorsZRotation",
    "SensorsYRotation",
    "SensorsXRotation",
    "Mixup",
    "SegmentationReconstruction",
    "MaskEncoding",
    "functional",
]
