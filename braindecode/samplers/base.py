"""
Sampler classes.
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from torch.utils.data.sampler import Sampler
from sklearn.utils import check_random_state


class RecordingSampler(Sampler):
    """Base sampler simplifying sampling from recordings.

    Parameters
    ----------
    metadata : pd.DataFrame
        DataFrame with at least one of {subject, session, run} columns for each
        window in the BaseConcatDataset to sample examples from. Normally
        obtained with `BaseConcatDataset.get_metadata()`. For instance,
        `metadata.head()` might look like this:

           i_window_in_trial  i_start_in_trial  i_stop_in_trial  target  subject    session    run
        0                  0                 0              500      -1        4  session_T  run_0
        1                  1               500             1000      -1        4  session_T  run_0
        2                  2              1000             1500      -1        4  session_T  run_0
        3                  3              1500             2000      -1        4  session_T  run_0
        4                  4              2000             2500      -1        4  session_T  run_0

    random_state : np.RandomState | int | None
        Random state.

    Attributes
    ----------
    info : pd.DataFrame
        Series with MultiIndex index which contains the subject, session, run
        and window indices information in an easily accessible structure for
        quick sampling of windows.
    """
    # XXX attributes n_recordings missing
    def __init__(self, metadata, random_state=None):
        self.metadata = metadata
        self._init_info()
        self.rng = check_random_state(random_state)

    def _init_info(self):
        keys = [k for k in ['subject', 'session', 'run']
                if k in self.metadata.columns]
        if not keys:
            raise ValueError(
                'metadata must contain at least one of the following columns: '
                'subject, session or run.')

        self.metadata = self.metadata.reset_index().rename(
            columns={'index': 'window_index'})
        self.info = self.metadata.reset_index().groupby(keys)[
            ['index', 'i_start_in_trial']].agg(['unique'])
        self.info.columns = self.info.columns.get_level_values(0)

    def sample_recording(self):
        """Return a random recording index.
        """
        # XXX docstring missing
        return self.rng.choice(self.n_recordings)

    def sample_classe(self):
        """Return a random recording index.
        """
        # XXX docstring missing
        return self.rng.choice(self.n_classes)

    def sample_window(self, rec_ind=None):
        """Return a specific window.
        """
        # XXX docstring missing
        if rec_ind is None:
            rec_ind = self.sample_recording()
        win_ind = self.rng.choice(self.info.iloc[rec_ind]['index'])
        return win_ind, rec_ind

    def __iter__(self):
        raise NotImplementedError

    @property
    def n_recordings(self):
        return self.info.shape[0]

    @property
    def n_classes(self):
        return self.metadata['target'].unique()


class SequenceSampler(RecordingSampler):
    """Sample sequences of consecutive windows.

    Parameters
    ----------
    metadata : pd.DataFrame
        See RecordingSampler.
    n_windows : int
        Number of consecutive windows in a sequence.
    n_windows_stride : int
        Number of windows between two consecutive sequences.
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
    def __init__(self, metadata, n_windows, n_windows_stride,
                 random_state=None):
        super().__init__(metadata, random_state=random_state)

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
        end_offset = 1 - self.n_windows if self.n_windows > 1 else None
        start_inds = self.info['index'].apply(
            lambda x: x[:end_offset:self.n_windows_stride]).values
        file_ids = [[i] * len(inds) for i, inds in enumerate(start_inds)]
        return np.concatenate(start_inds), np.concatenate(file_ids)

    def __len__(self):
        return len(self.start_inds)

    def __iter__(self):
        for start_ind in self.start_inds:
            yield tuple(range(start_ind, start_ind + self.n_windows))


class RandomSampler(RecordingSampler):
    """Sample sequences of consecutive windows.

    Parameters
    ----------
    metadata : pd.DataFrame
        See RecordingSampler.
    n_windows : int
        Number of consecutive windows in a sequence.
    seq_nbr : int
        Number of sequences in the sampler.
    random_state : np.random.RandomState | int | None
        Random state.
    """
    def __init__(self, metadata, n_windows, seq_nbr=10,
                 random_state=None):
        super().__init__(metadata, random_state=random_state)

        self.n_windows = n_windows
        self.n_windows_stride = 1
        self.seq_nbr = seq_nbr

        keys = [k for k in ['subject', 'session', 'run', 'target']
                if k in self.metadata.columns]
        if not keys:
            raise ValueError(
                'metadata must contain at least one of the following columns: '
                'subject, session or run.')
        self.info_class = self.metadata.reset_index().groupby(keys)[
            ['index', 'subject']].agg(['unique'])
        self.info_class.columns = self.info.columns.get_level_values(0)

    def _compute_seq_start_ind(self, rec_ind=None, class_ind=None):
        """Randomly compute sequence start indice.

        Choose a window associated with a random recording
        and a random class and place it randomly in a sequence.
        The function returns the beginning of this sequence.

        Returns
        -------
        start_ind : int
            The index of the first window of a possible sequence.

        rec_ind : int
            The random recording choosen for the indice of the f.

        class_ind : int
            The random class choosen.
        """
        if rec_ind is None:
            rec_ind = self.sample_recording()
        if class_ind is None:
            class_ind = self.sample_classe()

        rec_index = self.info.iloc[rec_ind]['index']
        len_rec_index = len(rec_index)

        win_ind = self.rng.choice(self.info_class.iloc[rec_ind*5 + class_ind]['index'])
        win_ind_in_rec = np.where(rec_index == win_ind)[0][0]

        win_pos = self.rng.choice(
                [i for i in range(self.n_windows)][
                    np.max((self.n_windows - len_rec_index + win_ind_in_rec, 0)):np.min(
                        (win_ind_in_rec, self.n_windows)
                    )
                ]
            )

        start_ind = win_ind - win_pos
        return start_ind, rec_ind, class_ind

    def __len__(self):
        return self.seq_nbr

    def __iter__(self):
        for _ in range(self.seq_nbr):
            start_ind, _, _ = self._compute_seq_start_ind()
            yield tuple(range(start_ind, start_ind + self.n_windows))
