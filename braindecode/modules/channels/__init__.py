"""One channel layer for every model: resolve a montage, map it with a strategy."""

from .resolve import (
    CHANNEL_NAME_ALIASES,
    ELECTRODE_KINDS,
    ChannelTarget,
    ResolvedMontage,
    resolve_montage,
)
from .source import SourceStrategy
from .strategies import (
    ChannelStrategy,
    ExactStrategy,
    FieldStrategy,
    IDWStrategy,
    LatentStrategy,
    NearestStrategy,
    RegionStrategy,
    SpatialMap,
    SplineStrategy,
    WienerStrategy,
    ZeroStrategy,
    get_channel_strategy,
    register_channel_strategy,
)
from .tokenizer import ChannelEncoding, ChannelTokenizer

__all__ = [
    "CHANNEL_NAME_ALIASES",
    "ELECTRODE_KINDS",
    "ChannelEncoding",
    "ChannelStrategy",
    "ChannelTarget",
    "ChannelTokenizer",
    "ExactStrategy",
    "FieldStrategy",
    "IDWStrategy",
    "LatentStrategy",
    "NearestStrategy",
    "RegionStrategy",
    "ResolvedMontage",
    "SourceStrategy",
    "SpatialMap",
    "SplineStrategy",
    "WienerStrategy",
    "ZeroStrategy",
    "get_channel_strategy",
    "register_channel_strategy",
    "resolve_montage",
]
