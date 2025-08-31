from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Sequence

import mne_bids
from torch.utils.data import IterableDataset, get_worker_info


class BIDSIterableDataset(IterableDataset):
    """Dataset for loading BIDS.

    .. warning::
        This class is experimental and may change in the future.

    .. warning::
        This dataset is not consistent with the Braindecode API.

    This class has the same parameters as the :func:`mne_bids.find_matching_paths` function
    as it will be used to find the files to load. The default ``extensions`` parameter was changed.

    More information on BIDS (Brain Imaging Data Structure)
    can be found at https://bids.neuroimaging.io

    Examples
    --------
    >>> from braindecode.datasets import BaseDataset, BaseConcatDataset
    >>> from braindecode.datasets.bids import BIDSIterableDataset, _description_from_bids_path
    >>> from braindecode.preprocessing import create_fixed_length_windows
    >>>
    >>> def my_reader_fn(path):
    ...     raw = mne_bids.read_raw_bids(path)
    ...     desc = _description_from_bids_path(path)
    ...     ds = BaseDataset(raw, description=desc)
    ...     windows_ds = create_fixed_length_windows(
    ...         BaseConcatDataset([ds]),
    ...         window_size_samples=400,
    ...         window_stride_samples=200,
    ...     )
    ...     return windows_ds
    >>>
    >>> dataset = BIDSIterableDataset(
    ...     reader_fn=my_reader_fn,
    ...     root="root/of/my/bids/dataset/",
    ... )

    Parameters
    ----------
    reader_fn : Callable[[mne_bids.BIDSPath], Sequence]
        A function that takes a BIDSPath and returns a dataset.
    pool_size : int
        The number of recordings to read and sample from.
    bids_paths : list[mne_bids.BIDSPath] | None
        A list of BIDSPaths to load. If None, will use the paths found by
        :func:`mne_bids.find_matching_paths` and the arguments below.
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

    def __init__(
        self,
        reader_fn: Callable[[mne_bids.BIDSPath], Sequence],
        pool_size: int = 4,
        bids_paths: list[mne_bids.BIDSPath] | None = None,
        root: Path | str | None = None,
        subjects: str | list[str] | None = None,
        sessions: str | list[str] | None = None,
        tasks: str | list[str] | None = None,
        acquisitions: str | list[str] | None = None,
        runs: str | list[str] | None = None,
        processings: str | list[str] | None = None,
        recordings: str | list[str] | None = None,
        spaces: str | list[str] | None = None,
        splits: str | list[str] | None = None,
        descriptions: str | list[str] | None = None,
        suffixes: str | list[str] | None = None,
        extensions: str | list[str] | None = [
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
        ],
        datatypes: str | list[str] | None = None,
        check: bool = False,
    ):
        if bids_paths is None:
            bids_paths = mne_bids.find_matching_paths(
                root=root,
                subjects=subjects,
                sessions=sessions,
                tasks=tasks,
                acquisitions=acquisitions,
                runs=runs,
                processings=processings,
                recordings=recordings,
                spaces=spaces,
                splits=splits,
                descriptions=descriptions,
                suffixes=suffixes,
                extensions=extensions,
                datatypes=datatypes,
                check=check,
                ignore_json=True,
            )
            # Filter out _epo.fif files:
            bids_paths = [
                bids_path
                for bids_path in bids_paths
                if not (bids_path.suffix == "epo" and bids_path.extension == ".fif")
            ]
        self.bids_paths = bids_paths
        self.reader_fn = reader_fn
        self.pool_size = pool_size

    def __add__(self, other):
        assert isinstance(other, BIDSIterableDataset)
        return BIDSIterableDataset(
            reader_fn=self.reader_fn,
            bids_paths=self.bids_paths + other.bids_paths,
            pool_size=self.pool_size,
        )

    def __iadd__(self, other):
        assert isinstance(other, BIDSIterableDataset)
        self.bids_paths += other.bids_paths
        return self

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            bids_paths = self.bids_paths
        else:  # in a worker process
            # split workload
            bids_paths = self.bids_paths[worker_info.id :: worker_info.num_workers]

        pool = []
        end = False
        paths_it = iter(random.sample(bids_paths, k=len(bids_paths)))
        while not (end and len(pool) == 0):
            while not end and len(pool) < self.pool_size:
                try:
                    bids_path = next(paths_it)
                    ds = self.reader_fn(bids_path)
                    if ds is None:
                        print(f"Skipping {bids_path} as it is too short.")
                        continue
                    idx = iter(random.sample(range(len(ds)), k=len(ds)))
                    pool.append((ds, idx))
                except StopIteration:
                    end = True
            i_pool = random.randint(0, len(pool) - 1)
            ds, idx = pool[i_pool]
            try:
                i_ds = next(idx)
                yield ds[i_ds]
            except StopIteration:
                pool.pop(i_pool)
