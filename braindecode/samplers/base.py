"""
Sampler classes.
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Theo Gnassounou <>
#          Young Truong <dt.young112@gmail.com>
#          Sarthak Tayal <sarthaktayal2@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler


def _group_windows_by_recording(metadata, required_keys=None):
    """Return one row per recording, with the positions (``index``) and start
    samples (``i_start_in_trial``) of its windows, sorted by subject, session
    and run.

    A recording is a dataset of the BaseConcatDataset, marked by the
    ``i_dataset`` column of ``get_metadata()``; an integer index starting
    again also starts one, for metadata concatenated from several
    ``get_metadata()`` calls. Without ``i_dataset``, the windows sharing
    subject, session and run form a recording. With ``required_keys``, rows
    are indexed by recording position and those keys.
    """
    keys = [k for k in ["subject", "session", "run"] if k in metadata.columns]
    windows = pd.DataFrame({k: metadata[k].to_numpy() for k in keys})
    if "i_dataset" in metadata.columns:
        new = metadata["i_dataset"].diff().ne(0).to_numpy()
        if pd.api.types.is_integer_dtype(metadata.index):
            new = new | (metadata.index.to_series().diff() <= 0).to_numpy()
        windows["recording"] = new.cumsum()
        keys.append("recording")
    elif not keys:
        raise ValueError(
            "metadata must contain the i_dataset column of "
            "BaseConcatDataset.get_metadata() or at least one of the "
            "columns subject, session or run."
        )
    windows["index"] = np.arange(len(metadata))
    # every window is kept, so that positions and start samples stay aligned
    windows["i_start_in_trial"] = metadata["i_start_in_trial"].to_numpy()
    if required_keys is not None:
        missing_keys = [k for k in required_keys if k not in metadata.columns]
        if len(missing_keys) > 0:
            raise ValueError(f"Columns {missing_keys} were not found in metadata.")
        # recordings by position: label lookups fail on a missing session or run
        windows["i_recording"] = windows.groupby(keys, dropna=False).ngroup()
        for k in required_keys:
            windows[k] = metadata[k].to_numpy()
        keys = ["i_recording", *required_keys]
    info = windows.groupby(keys, dropna=False)[["index", "i_start_in_trial"]].agg(list)
    for col in info:
        info[col] = info[col].map(np.array)
    return info


class RecordingSampler(Sampler):
    """Base sampler simplifying sampling from recordings.

    Parameters
    ----------
    metadata : pd.DataFrame
        DataFrame with at least one of {subject, session, run} columns for each
        window in the BaseConcatDataset to sample examples from. Normally
        obtained with `BaseConcatDataset.get_metadata()`, whose ``i_dataset``
        column makes each dataset one recording; without it, the windows
        sharing subject, session and run form one recording. For instance,
        `metadata.head()` might look like this:
        +-------------------+-----------------+-----------------+--------+----------+-----------+-------+
        | i_window_in_trial | i_start_in_trial| i_stop_in_trial | target | subject  | session   | run   |
        +===================+=================+=================+========+==========+===========+=======+
        | 0                 | 0               | 500             | -1     | 4        | session_T | run_0 |
        +-------------------+-----------------+-----------------+--------+----------+-----------+-------+
        | 1                 | 500             | 1000            | -1     | 4        | session_T | run_0 |
        +-------------------+-----------------+-----------------+--------+----------+-----------+-------+
        | 2                 | 1000            | 1500            | -1     | 4        | session_T | run_0 |
        +-------------------+-----------------+-----------------+--------+----------+-----------+-------+
        | 3                 | 1500            | 2000            | -1     | 4        | session_T | run_0 |
        +-------------------+-----------------+-----------------+--------+----------+-----------+-------+
        | 4                 | 2000            | 2500            | -1     | 4        | session_T | run_0 |
        +-------------------+-----------------+-----------------+--------+----------+-----------+-------+

    random_state : np.RandomState | int | None
        Random state.

    Attributes
    ----------
    info : pd.DataFrame
        Series with MultiIndex index which contains the subject, session, run,
        recording and window indices information in an easily accessible
        structure for quick sampling of windows.
    n_recordings : int
        Number of recordings available.
    """

    def __init__(self, metadata, random_state=None):
        self.metadata = metadata
        self.info = self._init_info(metadata)
        self.rng = check_random_state(random_state)

    def _init_info(self, metadata, required_keys=None):
        """Initialize ``info`` DataFrame.

        Parameters
        ----------
        required_keys : list(str) | None
            List of additional columns of the metadata DataFrame that we should
            groupby when creating ``info``.

        Returns
        -------
            See class attributes.
        """
        return _group_windows_by_recording(metadata, required_keys)

    def sample_recording(self):
        """Return a random recording index."""
        # XXX docstring missing
        return self.rng.choice(self.n_recordings)

    def sample_window(self, rec_ind=None):
        """Return a specific window."""
        # XXX docstring missing
        if rec_ind is None:
            rec_ind = self.sample_recording()
        win_ind = self.rng.choice(self.info.iloc[rec_ind]["index"])
        return win_ind, rec_ind

    def __iter__(self):
        raise NotImplementedError

    @property
    def n_recordings(self):
        return self.info.shape[0]


class DistributedRecordingSampler(DistributedSampler):
    """Base sampler simplifying sampling from recordings in distributed setting.

    Parameters
    ----------
    metadata : pd.DataFrame
        See RecordingSampler. For instance,
        `metadata.head()` might look like this::

            i_window_in_trial  i_start_in_trial  i_stop_in_trial  target  subject    session    run
            0                  0                 0              500      -1        4  session_T  run_0
            1                  1               500             1000      -1        4  session_T  run_0
            2                  2              1000             1500      -1        4  session_T  run_0
            3                  3              1500             2000      -1        4  session_T  run_0
            4                  4              2000             2500      -1        4  session_T  run_0

    random_state : np.RandomState | int | None
        Random state. An int is also used as the ``seed`` with which
        :class:`torch.utils.data.distributed.DistributedSampler` shuffles the
        recordings among processes; otherwise that seed is 0 (torch's default),
        so that every process splits the recordings the same way.

    Attributes
    ----------
    info : pd.DataFrame
        See RecordingSampler.
    n_recordings : int
        Number of recordings available.
    kwargs : dict
        Additional keyword arguments to pass to torch DistributedSampler.
        See https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
    """

    def __init__(
        self,
        metadata,
        random_state=None,
        **kwargs,
    ):
        self.metadata = metadata
        self.info = self._init_info(metadata)
        self.rng = check_random_state(random_state)
        # DistributedSampler needs an int seed (it shuffles with seed + epoch),
        # and the same one on every process: None or a RandomState would give
        # each process its own draw, so fall back to torch's default seed
        seed = kwargs.pop("seed", None)
        if seed is None:
            seed = (
                int(random_state) if isinstance(random_state, (int, np.integer)) else 0
            )
        # send information to DistributedSampler parent to handle data splitting among workers
        super().__init__(self.info, seed=seed, **kwargs)

    def _init_info(self, metadata, required_keys=None):
        """Initialize ``info`` DataFrame.

        Parameters
        ----------
        required_keys : list(str) | None
            List of additional columns of the metadata DataFrame that we should
            groupby when creating ``info``.

        Returns
        -------
            See class attributes.
        """
        return _group_windows_by_recording(metadata, required_keys)

    def sample_recording(self):
        """Return a random recording index.
        super().__iter__() contains indices of datasets specific to the current process
        determined by the DistributedSampler
        """
        # XXX docstring missing
        return self.rng.choice(list(super().__iter__()))

    def sample_window(self, rec_ind=None):
        """Return a specific window."""
        # XXX docstring missing
        if rec_ind is None:
            rec_ind = self.sample_recording()
        win_ind = self.rng.choice(self.info.iloc[rec_ind]["index"])
        return win_ind, rec_ind

    @property
    def n_recordings(self):
        return super().__len__()


class SequenceSampler(RecordingSampler):
    """Sample sequences of consecutive windows.

    Recordings holding fewer than ``n_windows`` windows cannot contain a full
    sequence and contribute none.

    Parameters
    ----------
    metadata : pd.DataFrame
        See RecordingSampler.
    n_windows : int
        Number of consecutive windows in a sequence.
    n_windows_stride : int
        Number of windows between two consecutive sequences.
    randomize : bool
        If True, sample sequences randomly. If False, sample sequences in
        order.
    random_state : np.random.RandomState | int | None
        Random state.

    Attributes
    ----------
    info : pd.DataFrame
        See RecordingSampler.
    file_ids : np.ndarray of ints
        Array of shape (n_sequences,) that indicates from which file each
        sequence comes from. Useful e.g. to do self-ensembling.
    """

    def __init__(
        self, metadata, n_windows, n_windows_stride, randomize=False, random_state=None
    ):
        super().__init__(metadata, random_state=random_state)
        self.randomize = randomize
        self.n_windows = n_windows
        self.n_windows_stride = n_windows_stride
        self.start_inds, self.file_ids = self._compute_seq_start_inds()

    def _compute_seq_start_inds(self):
        """Compute sequence start indices.

        Returns
        -------
        np.ndarray :
            Array of shape (n_sequences,) containing the indices of the first
            windows of possible sequences.
        np.ndarray :
            Array of shape (n_sequences,) containing the unique file number of
            each sequence. Useful e.g. to do self-ensembling.
        """
        # the windows of a recording need not be contiguous in the metadata
        # (no i_dataset column): sequences are read from each recording's
        # windows, by recording and position of their first window
        self._rec_inds = list(self.info["index"])
        starts = [
            np.arange(0, len(inds) - self.n_windows + 1, self.n_windows_stride)
            for inds in self._rec_inds
        ]
        # a typed array keeps the ids integer when a recording yields no sequence
        file_ids = [
            np.full(len(pos), i, dtype=np.int64) for i, pos in enumerate(starts)
        ]
        self._seq_starts = np.concatenate(starts).astype(np.int64)
        file_ids = np.concatenate(file_ids)
        start_inds = np.array(
            [self._rec_inds[i][p] for i, p in zip(file_ids, self._seq_starts)],
            dtype=np.int64,
        )
        return start_inds, file_ids

    def __len__(self):
        return len(self.start_inds)

    def __iter__(self):
        order = np.arange(len(self.start_inds))
        if self.randomize:
            self.rng.shuffle(order)
        for i in order:
            start = self._seq_starts[i]
            inds = self._rec_inds[self.file_ids[i]]
            yield tuple(inds[start : start + self.n_windows].tolist())


class BalancedSequenceSampler(RecordingSampler):
    """Balanced sampling of sequences of consecutive windows with categorical
    targets.

    Balanced sampling of sequences inspired by the approach of [Perslev2021]_:
    1. Uniformly sample a recording out of the available ones.
    2. Uniformly sample one of the classes.
    3. Sample a window of the corresponding class in the selected recording.
    4. Extract a sequence of windows around the sampled window.

    Recordings holding fewer than ``n_windows`` windows cannot contain a full
    sequence and are left out of step 1, as they are in
    :class:`SequenceSampler`.

    Parameters
    ----------
    metadata : pd.DataFrame
        See RecordingSampler.
        Must contain a column `target` with categorical targets.
    n_windows : int
        Number of consecutive windows in a sequence.
    n_sequences : int
        Number of sequences to sample.
    random_state : np.random.RandomState | int | None
        Random state.

    References
    ----------
    .. [Perslev2021] Perslev M, Darkner S, Kempfner L, Nikolic M, Jennum PJ,
           Igel C. U-Sleep: resilient high-frequency sleep staging. npj Digit.
           Med. 4, 72 (2021).
           https://github.com/perslev/U-Time/blob/master/utime/models/usleep.py
    """

    def __init__(self, metadata, n_windows, n_sequences=10, random_state=None):
        super().__init__(metadata, random_state=random_state)

        self.n_windows = n_windows
        self.n_sequences = n_sequences
        self.info_class = self._init_info(metadata, required_keys=["target"])
        self.long_enough_recordings = self._find_long_enough_recordings()

    def _find_long_enough_recordings(self):
        """Return the indices of the recordings that can hold a full sequence.

        Returns
        -------
        np.ndarray :
            Array of recording indices holding at least ``n_windows`` windows.
        """
        n_windows_per_rec = self.info["index"].apply(len).to_numpy()
        if n_windows_per_rec.size == 0:
            raise ValueError("Cannot build sequences from empty metadata.")

        long_enough = np.flatnonzero(n_windows_per_rec >= self.n_windows)
        if len(long_enough) == 0:
            raise ValueError(
                f"No recording holds enough windows to build a sequence of "
                f"{self.n_windows} windows. The longest recording has "
                f"{n_windows_per_rec.max()} windows. Reduce n_windows to at "
                f"most that value, or, if each dataset holds one trial of a "
                f"recording, pass the metadata without its i_dataset column to "
                f"group the trials by subject, session and run."
            )
        return long_enough

    def sample_recording(self):
        """Return a random recording index among the ones long enough."""
        return self.rng.choice(self.long_enough_recordings)

    def sample_class(self, rec_ind=None):
        """Return a random class.

        Parameters
        ----------
        rec_ind : int | None
            Index to the recording to sample from. If None, the recording will
            be uniformly sampled across available recordings.

        Returns
        -------
        int
            Sampled class.
        int
            Index to the recording the class was sampled from.
        """
        if rec_ind is None:
            rec_ind = self.sample_recording()
        available_classes = self.info_class.loc[rec_ind].index
        return self.rng.choice(available_classes), rec_ind

    def _sample_seq_start_ind(self, rec_ind=None, class_ind=None):
        """Sample a sequence and return its start index.

        Sample a window associated with a random recording and a random class
        and randomly sample a sequence with it inside. The function returns the
        index of the beginning of the sequence.

        Parameters
        ----------
        rec_ind : int | None
            Index to the recording to sample from. If None, the recording will
            be uniformly sampled across available recordings.
        class_ind : int | None
            If provided as int, sample a window of the corresponding class. If
            None, the class will be uniformly sampled across available classes.

        Returns
        -------
        int
            Index of the first window of the sequence.
        int
            Corresponding recording index.
        int
            Class of the sampled window.
        """
        if class_ind is None:
            class_ind, rec_ind = self.sample_class(rec_ind)

        rec_inds = self.info.iloc[rec_ind]["index"]
        len_rec_inds = len(rec_inds)

        available_indices = self.info_class.loc[(rec_ind, class_ind), "index"]
        win_ind = self.rng.choice(available_indices)
        win_ind_in_rec = np.where(rec_inds == win_ind)[0][0]

        # Minimum and maximum start indices in the sequence
        min_pos = max(0, win_ind_in_rec - self.n_windows + 1)
        max_pos = min(len_rec_inds - self.n_windows, win_ind_in_rec)
        start_ind = rec_inds[self.rng.randint(min_pos, max_pos + 1)]

        return start_ind, rec_ind, class_ind

    def __len__(self):
        return self.n_sequences

    def __iter__(self):
        for _ in range(self.n_sequences):
            start_ind, rec_ind, _ = self._sample_seq_start_ind()
            # the windows of a recording need not be contiguous in the metadata
            rec_inds = self.info.iloc[rec_ind]["index"]
            start = np.flatnonzero(rec_inds == start_ind)[0]
            yield tuple(rec_inds[start : start + self.n_windows].tolist())
