# mypy: ignore-errors
"""
Format backend protocol and registry for Hub integration.

This module defines the interface that storage format backends must implement,
and provides a registry for looking up backends by name.

Available backends:
- "zarr": Compressed Zarr format (supports compression, no lazy loading)
- "mne": MNE FIF format (no compression, supports lazy loading via preload=False)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ...base import BaseDataset, BaseConcatDataset


@runtime_checkable
class FormatBackend(Protocol):
    """Protocol defining the interface every storage format must implement."""

    name: str

    def validate_dependencies(self) -> None:
        """Raise ImportError if required packages are missing."""
        ...

    def get_data_filename(self) -> str:
        """Return the data directory/file name (e.g. 'dataset.zarr')."""
        ...

    def build_format_info(self, format_params: dict) -> dict:
        """Return format-specific fields for the lock file.

        Must include at minimum ``{"format": self.name}``.
        """
        ...

    def convert_datasets(
        self,
        datasets: list[BaseDataset],
        output_path: Path,
        format_params: dict,
    ) -> None:
        """Serialize datasets to *output_path*.

        *format_params* contains backend-specific keys (e.g. compression).
        """
        ...

    def load_datasets(
        self,
        input_path: Path,
        preload: bool,
    ) -> BaseConcatDataset:
        """Deserialize datasets from *input_path*."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_FORMAT_REGISTRY: dict[str, FormatBackend] = {}


def register_format(backend: FormatBackend) -> FormatBackend:
    """Register a format backend."""
    _FORMAT_REGISTRY[backend.name] = backend
    return backend


def get_format_backend(name: str) -> FormatBackend:
    """Look up a registered format backend by name."""
    # Ensure built-in backends are imported (they self-register on import)
    _ensure_builtins_loaded()
    if name not in _FORMAT_REGISTRY:
        available = ", ".join(sorted(_FORMAT_REGISTRY.keys()))
        raise ValueError(
            f"Unknown format '{name}'. Available formats: {available}"
        )
    return _FORMAT_REGISTRY[name]


def _ensure_builtins_loaded():
    """Import built-in backends so they register themselves."""
    if _FORMAT_REGISTRY:
        return
    from . import zarr_backend as _zarr  # noqa: F401
    from . import mne_backend as _mne  # noqa: F401
