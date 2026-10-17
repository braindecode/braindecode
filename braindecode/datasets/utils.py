"""Utility functions for dataset handling."""

# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

from __future__ import annotations

import os
from pathlib import Path

from mne.datasets import fetch_dataset

from .bids import BIDSDataset


def _correct_dataset_path(
    path: str, archive_name: str, subfolder_name: str | None = None
) -> str:
    """
    Correct the dataset path after download and extraction.

    This function handles two common post-download scenarios:
    1. Renames '.unzip' suffixed directories created by some extraction tools
    2. Navigates into a subfolder if the archive extracts to a nested directory

    Parameters
    ----------
    path : str
        Expected path to the dataset directory.
    archive_name : str
        Name of the downloaded archive file without extension
        (e.g., "chb_mit_bids", "NMT").
    subfolder_name : str | None
        Name of the subfolder within the extracted archive that contains the
        actual data. If provided and the subfolder exists, the path will be
        updated to point to it. If None, only renaming is attempted.
        Default is None.

    Returns
    -------
    str
        The corrected path to the dataset directory.

    Raises
    ------
    PermissionError
        If the '.unzip' directory exists but cannot be renamed due to
        insufficient permissions.
    """
    if not Path(path).exists():
        unzip_file_name = f"{archive_name}.unzip"
        if (Path(path).parent / unzip_file_name).exists():
            try:
                os.rename(
                    src=Path(path).parent / unzip_file_name,
                    dst=Path(path),
                )
            except PermissionError:
                raise PermissionError(
                    f"Please rename {Path(path).parent / unzip_file_name} "
                    f"manually to {path} and try again."
                )

    # Check if the subfolder exists inside the path
    if subfolder_name is not None:
        subfolder_path = os.path.join(path, subfolder_name)
        if Path(subfolder_path).exists():
            path = subfolder_path

    return path


class _ZenodoBIDSDataset(BIDSDataset):
    """A BIDS dataset in EDF files, downloaded from a Zenodo archive when no
    ``root`` is given. Subclasses set the three class attributes below."""

    _dataset_params: dict
    _archive_name: str  # the downloaded archive
    _unzipped_dir: str  # the folder the archive extracts to

    def __init__(self, root=None, *args, **kwargs):
        if root is None:
            path_root = fetch_dataset(
                dataset_params=self._dataset_params,
                path=None,
                processor="unzip",
                force_update=False,
            )
            # The first fetch extracts into a nested folder: point at the data.
            path_root = _correct_dataset_path(
                path_root, self._archive_name, self._unzipped_dir
            )
        elif not Path(f"{root}/participants.tsv").exists():
            raise ValueError(
                f"The provided root directory {root} does not contain a valid "
                "BIDS dataset (missing participants.tsv). Please ensure the "
                "root points directly to the BIDS dataset directory."
            )
        else:
            path_root = root
        super().__init__(
            *args, root=path_root, extensions=".edf", check=False, **kwargs
        )
