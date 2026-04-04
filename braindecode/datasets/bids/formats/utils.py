# mypy: ignore-errors
"""
Shared utilities for Hub format backends.

This module contains JSON sanitization helpers for NaN/Inf handling,
used by both Zarr and MNE backends.
"""

from __future__ import annotations

import numpy as np


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
