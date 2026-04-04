# mypy: ignore-errors
"""
Backward-compatible re-exports of low-level I/O helpers.

The canonical implementations now live in:
- JSON helpers: ``formats.utils``
- Zarr I/O: ``formats.zarr_backend``

Import from there for new code; this module is kept so that existing
imports continue to work.
"""

from .formats.utils import (  # noqa: F401
    _restore_nan_from_json,
    _sanitize_for_json,
)

from .formats.zarr_backend import (  # noqa: F401
    _create_compressor,
    _load_eegwindows_from_zarr,
    _load_raw_from_zarr,
    _load_windows_from_zarr,
    _save_eegwindows_to_zarr,
    _save_raw_to_zarr,
    _save_windows_to_zarr,
)
