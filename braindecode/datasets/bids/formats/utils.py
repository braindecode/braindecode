# mypy: ignore-errors
"""
Shared utilities for Hub format backends.

This module contains:
- JSON sanitization helpers (NaN/Inf handling)
- Dataclass-based backend parameter definitions with dict auto-cast
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _sanitize_for_json(obj):
    """Replace NaN/Inf with None for valid JSON."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    return obj


def _restore_nan_from_json(obj):
    """Restore NaN values from None in JSON-loaded data."""
    if isinstance(obj, dict):
        return {k: _restore_nan_from_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > 0 and all(isinstance(x, (int, float, type(None))) for x in obj):
            return [np.nan if x is None else x for x in obj]
        return [_restore_nan_from_json(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Backend parameter dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ZarrParams:
    """Parameters for the Zarr backend.

    Parameters
    ----------
    compression : str
        Compression algorithm. Default ``"blosc"``.
    compression_level : int
        Compression level 0-9. Default ``5``.
    chunk_size : int
        Samples per chunk. Default ``5_000_000``.
    """
    compression: str = "blosc"
    compression_level: int = 5
    chunk_size: int = 5_000_000

    @property
    def format(self):
        return "zarr"


@dataclass
class MNEParams:
    """Parameters for the MNE/FIF backend.

    Parameters
    ----------
    split_size : str
        Max file size before splitting (e.g. "2GB", "500MB").
        Default ``"2GB"``.
    """
    split_size: str = "2GB"

    @property
    def format(self):
        return "mne"


_PARAMS_CLS = {"zarr": ZarrParams, "mne": MNEParams}


def resolve_backend_params(backend_params):
    """Resolve backend_params to a dataclass instance.

    Accepts:
    - ``None`` -> ``ZarrParams()`` (default)
    - ``ZarrParams`` / ``MNEParams`` instance -> pass through
    - ``dict`` with ``"format"`` key -> auto-cast to correct dataclass

    Parameters
    ----------
    backend_params : dict | ZarrParams | MNEParams | None
        Backend parameters.

    Returns
    -------
    ZarrParams | MNEParams
        Resolved parameters.
    """
    if backend_params is None:
        return ZarrParams()
    if isinstance(backend_params, (ZarrParams, MNEParams)):
        return backend_params
    if isinstance(backend_params, dict):
        p = dict(backend_params)  # don't mutate caller's dict
        fmt = p.pop("format", "zarr")
        cls = _PARAMS_CLS.get(fmt)
        if cls is None:
            raise ValueError(
                f"Unknown format '{fmt}'. Available: {list(_PARAMS_CLS)}"
            )
        return cls(**p)
    raise TypeError(
        f"backend_params must be dict, ZarrParams, MNEParams, or None; "
        f"got {type(backend_params)}"
    )
