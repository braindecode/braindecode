"""Registered channel strategies (importing a module registers its strategies)."""

from . import interp, sensor  # noqa: F401  (registers the strategies)
from .base import (
    ChannelStrategy,
    SpatialMap,
    get_channel_strategy,
    register_channel_strategy,
)

__all__ = [
    "ChannelStrategy",
    "SpatialMap",
    "get_channel_strategy",
    "register_channel_strategy",
]
