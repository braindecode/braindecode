# mypy: ignore-errors
"""
MNE/FIF storage backend for Hub integration.

This backend stores EEG data in MNE's native FIF format, enabling true lazy
loading via memory-mapped files when ``preload=False``.

FIF files are placed alongside their BIDS sidecar files in per-subject
directories under ``sourcedata/braindecode/``.

Supported dataset types:
- **RawDataset**: saves via ``raw.save()``
- **EEGWindowsDataset**: saves continuous raw via ``raw.save()`` + metadata TSV
- **WindowsDataset**: saves via ``epochs.save()``
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import mne
import pandas as pd

import braindecode

from ...registry import get_dataset_class
from .. import hub_format, hub_validation
from .registry import register_format
from .utils import _restore_nan_from_json, _sanitize_for_json

log = logging.getLogger(__name__)


@register_format
@dataclass
class MneBackend:
    """MNE/FIF storage backend — no compression, lazy-loadable.

    Parameters
    ----------
    split_size : str
        Max file size before splitting (e.g. "2GB", "500MB").
        Default ``"2GB"``.
    """

    split_size: str = "2GB"

    name: ClassVar[str] = "mne"

    @property
    def format(self):
        return "mne"

    def validate_dependencies(self) -> None:
        # MNE is always available in braindecode
        pass

    def get_data_filename(self):
        # MNE backend writes per-subject FIF files into the BIDS tree,
        # so there is no single data directory. Returning None signals
        # that the backend writes directly into sourcedata_dir.
        return None

    def build_format_info(self) -> dict:
        return {
            "format": "mne",
            "split_size": self.split_size,
        }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def convert_datasets(self, datasets, output_path: Path):
        """Serialize datasets as FIF files in per-subject BIDS directories.

        Parameters
        ----------
        datasets : list of BaseDataset
            Datasets to serialize.
        output_path : Path
            The ``sourcedata/braindecode/`` directory.
        """
        dataset_type, _, _ = hub_validation.validate_dataset_uniformity(datasets)

        first_ds = datasets[0]

        # Build manifest
        manifest = {
            "n_datasets": len(datasets),
            "dataset_type": dataset_type,
            "braindecode_version": braindecode.__version__,
            "mne_version": mne.__version__,
            "split_size": self.split_size,
            "recordings": [],
        }

        # Save preprocessing kwargs
        for kwarg_name in [
            "raw_preproc_kwargs",
            "window_kwargs",
            "window_preproc_kwargs",
        ]:
            if hasattr(first_ds, kwarg_name):
                kwargs = getattr(first_ds, kwarg_name)
                if kwargs:
                    manifest[kwarg_name] = kwargs

        for i_ds, ds in enumerate(datasets):
            description = ds.description if ds.description is not None else pd.Series()

            # Build BIDS path for FIF file
            bids_path = hub_format.description_to_bids_path(
                description,
                root=output_path,
                suffix="eeg",
                extension=".fif",
            )
            bids_path.mkdir(exist_ok=True)
            fif_path = bids_path.fpath

            recording_info = {
                "description": _sanitize_for_json(
                    json.loads(description.to_json(date_format="iso"))
                ),
            }

            if dataset_type == "RawDataset":
                ds.raw.save(fif_path, split_size=self.split_size, overwrite=True)
                target_name = ds.target_name if hasattr(ds, "target_name") else None
                if target_name is not None:
                    recording_info["target_name"] = target_name

            elif dataset_type == "EEGWindowsDataset":
                ds.raw.save(fif_path, split_size=self.split_size, overwrite=True)
                # Save metadata alongside FIF
                meta_path = bids_path.copy().update(suffix="metadata", extension=".tsv")
                ds.metadata.to_csv(
                    meta_path.fpath, sep="\t", index=True, encoding="utf-8"
                )
                recording_info["metadata_relpath"] = str(
                    meta_path.fpath.relative_to(output_path)
                )
                recording_info["targets_from"] = ds.targets_from
                recording_info["last_target_only"] = ds.last_target_only

            elif dataset_type == "WindowsDataset":
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*does not conform to MNE naming conventions.*",
                        category=RuntimeWarning,
                    )
                    ds.windows.save(
                        fif_path, split_size=self.split_size, overwrite=True
                    )
                target_name = ds.target_name if hasattr(ds, "target_name") else None
                if target_name is not None:
                    recording_info["target_name"] = target_name

            else:
                raise ValueError(f"Unsupported dataset_type: {dataset_type}")

            recording_info["fif_relpath"] = str(fif_path.relative_to(output_path))
            manifest["recordings"].append(recording_info)

            log.debug(f"Saved recording {i_ds} to {fif_path}")

        # Write manifest
        manifest_path = output_path / "dataset_info.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        log.info(f"Saved {len(datasets)} recordings as FIF to {output_path}")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_datasets(self, input_path: Path, preload: bool):
        """Load datasets from FIF files using the manifest.

        Parameters
        ----------
        input_path : Path
            The ``sourcedata/braindecode/`` directory containing
            ``dataset_info.json``.
        preload : bool
            If False, data is memory-mapped (lazy loading).
        """
        manifest_path = input_path / "dataset_info.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"MNE backend manifest not found at {manifest_path}."
            )

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        dataset_type = manifest["dataset_type"]
        recordings = manifest["recordings"]

        WindowsDataset = get_dataset_class("WindowsDataset")
        EEGWindowsDataset = get_dataset_class("EEGWindowsDataset")
        RawDataset = get_dataset_class("RawDataset")
        BaseConcatDataset = get_dataset_class("BaseConcatDataset")

        datasets = []
        for rec in recordings:
            fif_path = input_path / rec["fif_relpath"]
            description = pd.Series(_restore_nan_from_json(rec["description"]))

            if dataset_type == "RawDataset":
                raw = mne.io.read_raw_fif(fif_path, preload=preload)
                ds = RawDataset(raw, description)
                target_name = rec.get("target_name")
                if target_name is not None:
                    ds.target_name = target_name

            elif dataset_type == "EEGWindowsDataset":
                raw = mne.io.read_raw_fif(fif_path, preload=preload)
                meta_path = input_path / rec["metadata_relpath"]
                metadata = pd.read_csv(meta_path, sep="\t", index_col=0)
                targets_from = rec.get("targets_from", "metadata")
                last_target_only = rec.get("last_target_only", True)
                ds = EEGWindowsDataset(
                    raw=raw,
                    metadata=metadata,
                    description=description,
                    targets_from=targets_from,
                    last_target_only=last_target_only,
                )

            elif dataset_type == "WindowsDataset":
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*does not conform to MNE naming conventions.*",
                        category=RuntimeWarning,
                    )
                    epochs = mne.read_epochs(fif_path, preload=preload)
                ds = WindowsDataset(epochs, description)
                target_name = rec.get("target_name")
                if target_name is not None:
                    ds.target_name = target_name

            else:
                raise ValueError(f"Unsupported dataset_type: {dataset_type}")

            datasets.append(ds)

        concat_ds = BaseConcatDataset(datasets)

        # Restore preprocessing kwargs
        for kwarg_name in [
            "raw_preproc_kwargs",
            "window_kwargs",
            "window_preproc_kwargs",
        ]:
            if kwarg_name in manifest:
                kwargs = manifest[kwarg_name]
                for ds in datasets:
                    setattr(ds, kwarg_name, kwargs)

        return concat_ds
