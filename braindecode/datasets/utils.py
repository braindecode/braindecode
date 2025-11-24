"""Utility functions for dataset handling."""

# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

from __future__ import annotations

import os
from pathlib import Path


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
