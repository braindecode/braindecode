"""
This dataset is a BIDS compatible version of the Siena Scalp EEG Database.

It reorganizes the file structure to comply with the BIDS specification. To this effect:

-   Metadata was organized according to BIDS.
-   Data in the EEG edf files was modified to keep only the 19 channels from a 10-20 EEG system.
-   Annotations were formatted as BIDS-score compatible tsv files.


"""

# Authors: Dan, Jonathan
#          Detti, Paolo
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
from __future__ import annotations

import glob
import os
from pathlib import Path

from mne.datasets import fetch_dataset

from braindecode.datasets import BIDSDataset

SIENA_URL = "https://zenodo.org/records/10640762/files/BIDS_Siena.zip"
SIENA_archive_name = "SIENA.zip"
SIENA_folder_name = "MNE-SIENA-eeg-dataset"
SIENA_dataset_name = "SIENA-EEG-Corpus"

SIENA_dataset_params = {
    "dataset_name": SIENA_dataset_name,
    "url": SIENA_URL,
    "archive_name": SIENA_archive_name,
    "folder_name": SIENA_folder_name,
    "hash": "77b3ce12bcaf6c6cce4e6690ea89cb22bed55af10c525077b430f6e1d2e3c6bf",
    "config_key": SIENA_dataset_name,
}


class SIENA(BIDSDataset):
    """The Siena EEG Dataset.

    The database consists of EEG recordings of 14 patients acquired at the Unit of Neurology
    and Neurophysiology of the University of Siena.

    Subjects include 9 males (ages 25-71) and 5 females (ages 20-58).
    Subjects were monitored with a Video-EEG with a sampling rate of 512 Hz,
    with electrodes arranged on the basis of the international 10-20 System.

    Most of the recordings also contain 1 or 2 EKG signals.
    The diagnosis of epilepsy and the classification of seizures according to the
    criteria of the International League Against Epilepsy were performed by an expert
    clinician after a careful review of the clinical and electrophysiological
    data of each patient.


    This BIDS-compatible version of the dataset was published by Jonathan Dan [Dan2025]_
    and is based on the original Siena Scalp EEG Database [Detti2020a]_, [Detti2020b]_.

    .. versionadded:: 1.3

    Parameters
    ----------
    path: str
        Parent directory of the dataset.
    preload: bool
        If True, preload the data of the Raw objects.

    References
    ----------
    .. [Detti2020a] Detti, P. (2020). Siena Scalp EEG Database (version 1.0.0).
        PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/5d4a-j060
    .. [Detti2020b] Detti, P., Vatti, G., Zabalo Manrique de Lara, G.
        EEG Synchronization Analysis for Seizure Prediction:
        A Study on Data of Noninvasive Recordings.
        Processes 2020, 8(7), 846; https://doi.org/10.3390/pr8070846
    .. [Dan2025] Dan, J., Pale, U., Amirshahi, A., Cappelletti, W.,
        Ingolfsson, T. M., Wang, X., ... & Ryvlin, P. (2025).
        SzCORE: seizure community open-source research evaluatio
        framework for the validation of electroencephalography-based
        automated seizure detection algorithms. Epilepsia, 66, 14-24.
    """

    def __init__(self, path_root=None, *args, **kwargs):
        # correct the path if needed
        if path_root is not None:
            list_tsv = glob.glob(f"{path_root}/**/participants.tsv", recursive=True)
            if isinstance(list_tsv, list) and len(list_tsv) > 0:
                path_root = Path(list_tsv[0]).parent

        # Download dataset if not present
        if path_root is None or len(list_tsv) == 0:
            path_root = fetch_dataset(
                dataset_params=SIENA_dataset_params,
                path=Path(path_root) if path_root is not None else None,
                processor="unzip",
                force_update=False,
            )
            # First time we fetch the dataset, we need to move the files to the
            # correct directory.
            path_root = _correct_path(path_root)

        kwargs["root"] = path_root

        super().__init__(
            *args,
            extensions=".edf",
            check=False,
            **kwargs,
        )


def _correct_path(path: str):
    """
    Check if the path is correct and rename the file if needed.

    Parameters
    ----------
    path: basestring
        Path to the file.

    Returns
    -------
    path: basestring
        Corrected path.
    """
    if not Path(path).exists():
        unzip_file_name = f"{SIENA_archive_name}.unzip"
        if (Path(path).parent / unzip_file_name).exists():
            try:
                os.rename(
                    src=Path(path).parent / unzip_file_name,
                    dst=Path(path),
                )

            except PermissionError:
                raise PermissionError(
                    f"Please rename {Path(path).parent / unzip_file_name}"
                    + f"manually to {path} and try again."
                )
        path = os.path.join(path, "nmt_scalp_eeg_dataset")

    return path
