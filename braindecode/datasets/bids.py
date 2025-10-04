"""Dataset for loading BIDS.

More information on BIDS (Brain Imaging Data Structure) can be found at https://bids.neuroimaging.io
"""

# Authors: Pierre Guetschel <pierre.guetschel@gmail.com>
#
# License: BSD (3-clause)

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mne
import mne_bids
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .base import BaseConcatDataset, BaseDataset, WindowsDataset


def _description_from_bids_path(bids_path: mne_bids.BIDSPath) -> dict[str, Any]:
    return {
        "path": bids_path.fpath,
        "subject": bids_path.subject,
        "session": bids_path.session,
        "task": bids_path.task,
        "acquisition": bids_path.acquisition,
        "run": bids_path.run,
        "processing": bids_path.processing,
        "recording": bids_path.recording,
        "space": bids_path.space,
        "split": bids_path.split,
        "description": bids_path.description,
        "suffix": bids_path.suffix,
        "extension": bids_path.extension,
        "datatype": bids_path.datatype,
    }


@dataclass
class BIDSDataset(BaseConcatDataset):
    """Dataset for loading BIDS.

    This class has the same parameters as the :func:`mne_bids.find_matching_paths` function
    as it will be used to find the files to load. The default ``extensions`` parameter was changed.

    More information on BIDS (Brain Imaging Data Structure)
    can be found at https://bids.neuroimaging.io

    .. Note::
        For loading "unofficial" BIDS datasets containing epoched data,
        you can use :class:`BIDSEpochsDataset`.

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
    """

    root: Path | str
    subjects: str | list[str] | None = None
    sessions: str | list[str] | None = None
    tasks: str | list[str] | None = None
    acquisitions: str | list[str] | None = None
    runs: str | list[str] | None = None
    processings: str | list[str] | None = None
    recordings: str | list[str] | None = None
    spaces: str | list[str] | None = None
    splits: str | list[str] | None = None
    descriptions: str | list[str] | None = None
    suffixes: str | list[str] | None = None
    extensions: str | list[str] | None = field(
        default_factory=lambda: [
            ".con",
            ".sqd",
            ".pdf",
            ".fif",
            ".ds",
            ".vhdr",
            ".set",
            ".edf",
            ".bdf",
            ".EDF",
            ".snirf",
            ".cdt",
            ".mef",
            ".nwb",
        ]
    )
    datatypes: str | list[str] | None = None
    check: bool = False
    preload: bool = False
    n_jobs: int = 1

    @property
    def _filter_out_epochs(self):
        return True

    def __post_init__(self):
        bids_paths = mne_bids.find_matching_paths(
            root=self.root,
            subjects=self.subjects,
            sessions=self.sessions,
            tasks=self.tasks,
            acquisitions=self.acquisitions,
            runs=self.runs,
            processings=self.processings,
            recordings=self.recordings,
            spaces=self.spaces,
            splits=self.splits,
            descriptions=self.descriptions,
            suffixes=self.suffixes,
            extensions=self.extensions,
            datatypes=self.datatypes,
            check=self.check,
        )
        # Filter out .json files files:
        # (argument ignore_json only available in mne-bids>=0.16)
        bids_paths = [
            bids_path for bids_path in bids_paths if bids_path.extension != ".json"
        ]
        # Filter out _epo.fif files:
        if self._filter_out_epochs:
            bids_paths = [
                bids_path
                for bids_path in bids_paths
                if not (bids_path.suffix == "epo" and bids_path.extension == ".fif")
            ]

        all_base_ds = Parallel(n_jobs=self.n_jobs)(
            delayed(self._get_dataset)(bids_path) for bids_path in bids_paths
        )
        super().__init__(all_base_ds)

    def _get_dataset(self, bids_path: mne_bids.BIDSPath) -> BaseDataset:
        description = _description_from_bids_path(bids_path)
        raw = mne_bids.read_raw_bids(bids_path, verbose=False)
        if self.preload:
            raw.load_data()
        return BaseDataset(raw, description)


class BIDSEpochsDataset(BIDSDataset):
    """**Experimental** dataset for loading :class:`mne.Epochs` organised in BIDS.

    The files must end with ``_epo.fif``.

    .. Warning::
        Epoched data is not officially supported in BIDS.

    .. Note::
        **Parameters:** This class has the same parameters as :class:`BIDSDataset` except
        for arguments ``datatypes``, ``extensions`` and ``check`` which are fixed.
    """

    @property
    def _filter_out_epochs(self):
        return False

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            extensions=".fif",
            suffixes="epo",
            check=False,
            **kwargs,
        )

    def _set_metadata(self, epochs: mne.BaseEpochs) -> None:
        # events = mne.events_from_annotations(epochs
        n_times = epochs.times.shape[0]
        # id_event = {v: k for k, v in epochs.event_id.items()}
        annotations = epochs.annotations
        if annotations is not None:
            target = annotations.description
        else:
            id_events = {v: k for k, v in epochs.event_id.items()}
            target = [id_events[event_id] for event_id in epochs.events[:, -1]]
        metadata_dict = {
            "i_window_in_trial": np.zeros(len(epochs)),
            "i_start_in_trial": np.zeros(len(epochs)),
            "i_stop_in_trial": np.zeros(len(epochs)) + n_times,
            "target": target,
        }
        epochs.metadata = pd.DataFrame(metadata_dict)

    def _get_dataset(self, bids_path):
        description = _description_from_bids_path(bids_path)
        epochs = mne.read_epochs(bids_path.fpath)
        self._set_metadata(epochs)
        return WindowsDataset(epochs, description=description, targets_from="metadata")
