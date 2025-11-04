"""
This dataset is a BIDS-compatible version of the CHB-MIT Scalp EEG Database.

It reorganizes the file structure to comply with the BIDS specification. To this effect:

    The data from subject chb21 was moved to sub-01/ses-02.
    Metadata was organized according to BIDS.
    Data in the EEG edf files was modified to keep only the 18 channels from a double banana bipolar montage.
    Annotations were formatted as BIDS-score compatible `tsv` files.
"""

# Authors: Dan, Jonathan
#          Shoeb, Ali (Data collector)
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
from __future__ import annotations

import glob
import os
from pathlib import Path

from mne.datasets import fetch_dataset

from braindecode.datasets import BIDSDataset

CHB_MIT_URL = "https://zenodo.org/records/10259996/files/BIDS_CHB-MIT.zip"
CHB_MIT_archive_name = "chb_mit_bids.zip"
CHB_MIT_folder_name = "CHB-MIT-BIDS-eeg-dataset"
CHB_MIT_dataset_name = "CHB-MIT-EEG-Corpus"

CHB_MIT_dataset_params = {
    "dataset_name": CHB_MIT_dataset_name,
    "url": CHB_MIT_URL,
    "archive_name": CHB_MIT_archive_name,
    "folder_name": CHB_MIT_folder_name,
    "hash": "126e71e18570cf359a440ba5227494ecffca4b0b0057c733f90ec29ba5e15ff8",  # sha256
    "config_key": CHB_MIT_dataset_name,
}


class CHBMIT(BIDSDataset):
    """The Children's Hospital Boston EEG Dataset.

     This database, collected at the Children's Hospital Boston, consists of EEG recordings
     from pediatric subjects with intractable seizures. Subjects were monitored for up to
     several days following withdrawal of anti-seizure medication in order to characterize
     their seizures and assess their candidacy for surgical intervention.

     Description of the contents of the dataset

     Each folder (sub-01, sub-01, etc.) contains between 9 and 42 continuous .edf
     files from a single subject. Hardware limitations resulted in gaps between
     consecutively-numbered .edf files, during which the signals were not recorded;
     in most cases, the gaps are 10 seconds or less, but occasionally there are much
     longer gaps. In order to protect the privacy of the subjects, all protected health
     information (PHI) in the original .edf files has been replaced with surrogate information
     in the files provided here. Dates in the original .edf files have been replaced by
     surrogate dates, but the time relationships between the individual files belonging
     to each case have been preserved. In most cases, the .edf files contain exactly one
     hour of digitized EEG signals, although those belonging to case sub-10 are two hours
     long, and those belonging to cases sub-04, sub-06, sub-07, sub-09, and sub-23 are
     four hours long; occasionally, files in which seizures are recorded are shorter.

     The EEG is recorded at 256 Hz with a 16-bit resolution. The recordings are
    referenced in a double banana bipolar montage with 18 channels from the 10-20 electrode system.

     This BIDS-compatible version of the dataset was published by Jonathan Dan [Dan2025]_
     and is based on the original CHB MIT EEG Database [Guttag2010]_, [Shoeb2009]_.

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
     .. [Guttag2010] Guttag, J. (2010). CHB-MIT Scalp EEG Database (version 1.0.0).
         PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/C2K01R
     .. [Shoeb2009] Ali Shoeb. Application of Machine Learning to Epileptic
         Seizure Onset Detection and Treatment. PhD Thesis,
         Massachusetts Institute of Technology, September 2009.
     .. [Dan2025] Dan, J., Pale, U., Amirshahi, A., Cappelletti, W.,
         Ingolfsson, T. M., Wang, X., ... & Ryvlin, P. (2025).
         SzCORE: seizure community open-source research evaluatio
         framework for the validation of electroencephalography-based
         automated seizure detection algorithms. Epilepsia, 66, 14-24.
    """

    def __init__(self, root=None, *args, **kwargs):
        # correct the path if needed
        path_root = root
        if path_root is not None:
            list_tsv = glob.glob(f"{path_root}/**/participants.tsv", recursive=True)
            if isinstance(list_tsv, list) and len(list_tsv) > 0:
                path_root = Path(list_tsv[0]).parent

        # Download dataset if not present
        if path_root is None or len(list_tsv) == 0:
            path_root = fetch_dataset(
                dataset_params=CHB_MIT_dataset_params,
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
        unzip_file_name = f"{CHB_MIT_archive_name}.unzip"
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

    # Check if the BIDS_Siena folder exists inside the path
    bids_siena_path = os.path.join(path, "BIDS_CHB_MIT")
    if Path(bids_siena_path).exists():
        path = bids_siena_path

    return path
