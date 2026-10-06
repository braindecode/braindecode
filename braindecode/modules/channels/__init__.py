"""One channel layer for every model: resolve a montage, map it with a strategy."""

from .resolve import (
    CHANNEL_NAME_ALIASES,
    ELECTRODE_KINDS,
    ResolvedMontage,
    match_names,
    nearest_vocabulary,
    resolve_montage,
)
from .strategies import (
    ChannelStrategy,
    SpatialMap,
    get_channel_strategy,
    register_channel_strategy,
)
from .target import ChannelTarget
from .tokenizer import ChannelEncoding, ChannelTokenizer

__all__ = [
    "CHANNEL_NAME_ALIASES",
    "ELECTRODE_KINDS",
    "ChannelEncoding",
    "ChannelStrategy",
    "ChannelTarget",
    "ChannelTokenizer",
    "ResolvedMontage",
    "SpatialMap",
    "get_channel_strategy",
    "match_names",
    "nearest_vocabulary",
    "register_channel_strategy",
    "resolve_montage",
]
