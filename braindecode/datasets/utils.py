"""Utility functions for dataset handling."""

# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

from __future__ import annotations

import os
from pathlib import Path


def correct_dataset_path(
    path: str, archive_name: str, subfolder_name: str | None = None
) -> str:
    """
    Check if the path is correct and rename the file if needed.

    Parameters
    ----------
    path : str
        Path to the dataset.
    archive_name : str
        Name of the archive file (e.g., "chb_mit_bids.zip", "NMT.zip").
    subfolder_name : str | None
        Name of the subfolder inside the archive. If provided, the function will
        check if this subfolder exists and use it as the path. If None, the path
        will be used as-is after potential renaming. Default is None.

    Returns
    -------
    str
        Corrected path.
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
