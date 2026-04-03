# mypy: ignore-errors
"""
MNE/FIF storage backend for Hub integration (placeholder).

This backend will store EEG data in MNE's native FIF format. It does not
support compression but enables true lazy loading via memory-mapped files
when ``preload=False``.

.. note::
    This backend is not yet implemented. Using ``format="mne"`` in
    ``push_to_hub`` or loading an MNE-formatted dataset will raise
    ``NotImplementedError``.
"""

from __future__ import annotations

from pathlib import Path

from . import register_format


class MneBackend:
    """MNE/FIF storage backend — no compression, lazy-loadable (placeholder)."""

    name = "mne"

    def validate_dependencies(self) -> None:
        # MNE is always available in braindecode
        pass

    def get_data_filename(self) -> str:
        return "dataset"

    def build_format_info(self, format_params: dict) -> dict:
        return {"format": "mne"}

    def convert_datasets(self, datasets, output_path: Path, format_params: dict):
        raise NotImplementedError(
            "MNE/FIF format push is not yet implemented. Use format='zarr'."
        )

    def load_datasets(self, input_path: Path, preload: bool):
        raise NotImplementedError(
            "MNE/FIF format pull is not yet implemented."
        )


register_format(MneBackend())
