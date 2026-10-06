"""One channel layer for every model: resolve a montage, map it with a strategy."""

from .resolve import (
    CHANNEL_NAME_ALIASES,
    ResolvedMontage,
    match_names,
    nearest_vocabulary,
    resolve_montage,
)

__all__ = [
    "CHANNEL_NAME_ALIASES",
    "ResolvedMontage",
    "match_names",
    "nearest_vocabulary",
    "resolve_montage",
]
