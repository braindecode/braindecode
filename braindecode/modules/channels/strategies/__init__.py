"""Registered channel strategies (importing a module registers its strategies)."""

from .base import (
    ChannelStrategy,
    SpatialMap,
    get_channel_strategy,
    register_channel_strategy,
)
from .interp import FieldStrategy, SplineStrategy
from .learned import LatentStrategy, RegionStrategy, WienerStrategy
from .sensor import ExactStrategy, IDWStrategy, NearestStrategy, ZeroStrategy
from .source import SourceStrategy

__all__ = [
    "ChannelStrategy",
    "ExactStrategy",
    "FieldStrategy",
    "IDWStrategy",
    "LatentStrategy",
    "NearestStrategy",
    "RegionStrategy",
    "SourceStrategy",
    "SpatialMap",
    "SplineStrategy",
    "WienerStrategy",
    "ZeroStrategy",
    "get_channel_strategy",
    "register_channel_strategy",
]
