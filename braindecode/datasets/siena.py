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

from pathlib import Path

from mne.datasets import fetch_dataset

from braindecode.datasets import BIDSDataset
from braindecode.datasets.utils import _correct_dataset_path

SIENA_URL = "https://zenodo.org/records/10640762/files/BIDS_Siena.zip"
SIENA_archive_name = "SIENA.zip"
SIENA_folder_name = "SIENA-BIDS-eeg-dataset"
SIENA_dataset_name = "SIENA-EEG-Corpus"

SIENA_dataset_params = {
    "dataset_name": SIENA_dataset_name,
    "url": SIENA_URL,
    "archive_name": SIENA_archive_name,
    "folder_name": SIENA_folder_name,
    "hash": "126e71e18570cf359a440ba5227494ecffca4b0b0057c733f90ec29ba5e15ff8",  # sha256
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
    root : pathlib.Path | str
        The root of the BIDS path.
    subjects : str | array-like of str | None
        The subject ID. Corresponds to "sub".
    sessions : str | array-like of str | None
        The acquisition session. Corresponds to "ses".
    tasks : str | array-like of str | None
        The experimental task. Corresponds to "task".
    acquisitions: str | array-like of str | None
        The acquisition parameters. Corresponds to "acq".
    runs : str | array-like of str | None
        The run number. Corresponds to "run".
    processings : str | array-like of str | None
        The processing label. Corresponds to "proc".
    recordings : str | array-like of str | None
        The recording name. Corresponds to "rec".
    spaces : str | array-like of str | None
        The coordinate space for anatomical and sensor location
        files (e.g., ``*_electrodes.tsv``, ``*_markers.mrk``).
        Corresponds to "space".
        Note that valid values for ``space`` must come from a list
        of BIDS keywords as described in the BIDS specification.
    splits : str | array-like of str | None
        The split of the continuous recording file for ``.fif`` data.
        Corresponds to "split".
    descriptions : str | array-like of str | None
        This corresponds to the BIDS entity ``desc``. It is used to provide
        additional information for derivative data, e.g., preprocessed data
        may be assigned ``description='cleaned'``.
    suffixes : str | array-like of str | None
        The filename suffix. This is the entity after the
        last ``_`` before the extension. E.g., ``'channels'``.
        The following filename suffix's are accepted:
        'meg', 'markers', 'eeg', 'ieeg', 'T1w',
        'participants', 'scans', 'electrodes', 'coordsystem',
        'channels', 'events', 'headshape', 'digitizer',
        'beh', 'physio', 'stim'
    extensions : str | array-like of str | None
        The extension of the filename. E.g., ``'.json'``.
        By default, uses the ones accepted by :func:`mne_bids.read_raw_bids`.
    datatypes : str | array-like of str | None
        The BIDS data type, e.g., ``'anat'``, ``'func'``, ``'eeg'``, ``'meg'``,
        ``'ieeg'``.
    check : bool
        If ``True``, only returns paths that conform to BIDS. If ``False``
        (default), the ``.check`` attribute of the returned
        :class:`mne_bids.BIDSPath` object will be set to ``True`` for paths that
        do conform to BIDS, and to ``False`` for those that don't.
    preload : bool
        If True, preload the data. Defaults to False.
    n_jobs : int
        Number of jobs to run in parallel. Defaults to 1.

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

    def __init__(self, root=None, *args, **kwargs):
        # Download dataset if not present
        if root is None:
            path_root = fetch_dataset(
                dataset_params=SIENA_dataset_params,
                path=None,
                processor="unzip",
                force_update=False,
            )
            # First time we fetch the dataset, we need to move the files to the
            # correct directory.
            path_root = _correct_dataset_path(
                path_root, SIENA_archive_name, "BIDS_Siena"
            )
        else:
            # Validate that the provided root is a valid BIDS dataset
            if not Path(f"{root}/participants.tsv").exists():
                raise ValueError(
                    f"The provided root directory {root} does not contain a valid "
                    "BIDS dataset (missing participants.tsv). Please ensure the "
                    "root points directly to the BIDS dataset directory."
                )
            path_root = root

        kwargs["root"] = path_root

        super().__init__(
            *args,
            extensions=".edf",
            check=False,
            **kwargs,
        )
