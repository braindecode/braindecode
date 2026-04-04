# mypy: ignore-errors
"""
Format backend protocol and registry for Hub integration.

Available backends:
- "zarr": Compressed Zarr format (supports compression, no lazy loading)
- "mne": MNE FIF format (no compression, supports lazy loading via preload=False)
"""

from .mne_backend import MneBackend  # noqa: F401
from .registry import (  # noqa: F401
    get_format_backend,
    register_format,
    resolve_backend_params,
)
from .utils import _restore_nan_from_json, _sanitize_for_json  # noqa: F401
from .zarr_backend import ZarrBackend  # noqa: F401
