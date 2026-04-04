# mypy: ignore-errors
"""
Format backend protocol, registry, and parameter resolution.

This module defines the interface that storage format backends must implement,
provides a registry for looking up backends by name, and handles dict-to-backend
auto-casting via ``resolve_backend_params``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ...base import BaseDataset, BaseConcatDataset


@runtime_checkable
class FormatBackend(Protocol):
    """Protocol defining the interface every storage format must implement."""

    name: str

    def validate_dependencies(self) -> None:
        """Raise ImportError if required packages are missing."""
        ...

    def get_data_filename(self) -> Optional[str]:
        """Return the data directory/file name, or None if per-subject."""
        ...

    def build_format_info(self) -> dict:
        """Return format-specific fields for the lock file.

        Must include at minimum ``{"format": self.name}``.
        """
        ...

    def convert_datasets(
        self,
        datasets: list[BaseDataset],
        output_path: Path,
    ) -> None:
        """Serialize datasets to *output_path*."""
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

_FORMAT_REGISTRY: dict[str, type] = {}


def register_format(backend_cls):
    """Register a format backend class."""
    _FORMAT_REGISTRY[backend_cls.name] = backend_cls
    return backend_cls


def get_format_backend(name: str) -> type:
    """Look up a registered format backend class by name."""
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


# ---------------------------------------------------------------------------
# Parameter resolution
# ---------------------------------------------------------------------------

def resolve_backend_params(backend_params):
    """Resolve ``backend_params`` to a backend instance.

    Accepts:
    - ``None`` -> ``ZarrBackend()`` with defaults
    - Backend instance (``ZarrBackend`` / ``MneBackend``) -> pass through
    - ``dict`` with ``"format"`` key -> auto-cast to correct backend class
      (extra keys not recognized by the backend are silently ignored)

    Parameters
    ----------
    backend_params : dict | ZarrBackend | MneBackend | None
        Backend parameters.

    Returns
    -------
    ZarrBackend | MneBackend
        Resolved backend instance.
    """
    import dataclasses

    _ensure_builtins_loaded()
    if backend_params is None:
        return _FORMAT_REGISTRY["zarr"]()
    if isinstance(backend_params, dict):
        p = dict(backend_params)  # don't mutate caller's dict
        fmt = p.pop("format", "zarr")
        cls = _FORMAT_REGISTRY.get(fmt)
        if cls is None:
            available = ", ".join(sorted(_FORMAT_REGISTRY.keys()))
            raise ValueError(
                f"Unknown format '{fmt}'. Available: {available}"
            )
        # Filter to only fields accepted by the dataclass
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        p = {k: v for k, v in p.items() if k in valid_keys}
        return cls(**p)
    # Already an instance
    return backend_params
