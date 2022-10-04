"""
Sampler classes.
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Theo Gnassounou <>
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
        keys = [k for k in ['subject', 'session', 'run']
                if k in self.metadata.columns]
        if not keys:
            raise ValueError(
                'metadata must contain at least one of the following columns: '
                'subject, session or run.')

        if required_keys is not None:
            missing_keys = [
                k for k in required_keys if k not in self.metadata.columns]
            if len(missing_keys) > 0:
                raise ValueError(
                    f'Columns {missing_keys} were not found in metadata.')
            keys += required_keys

        metadata = metadata.reset_index().rename(
            columns={'index': 'window_index'})
        info = metadata.reset_index().groupby(keys)[
            ['index', 'i_start_in_trial']].agg(['unique'])
        info.columns = info.columns.get_level_values(0)

        return info

    def sample_recording(self):
        """Return a random recording index.
        """
        # XXX docstring missing
        return self.rng.choice(self.n_recordings)

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


class BalancedSequenceSampler(RecordingSampler):
    """Balanced sampling of sequences of consecutive windows with categorical
    targets.

    Balanced sampling of sequences inspired by the approach of [Perslev2021]_:
    1. Uniformly sample a recording out of the available ones.
    2. Uniformly sample one of the classes.
    3. Sample a window of the corresponding class in the selected recording.
    4. Extract a sequence of windows around the sampled window.

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
        self.info_class = self._init_info(metadata, required_keys=['target'])

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
        available_classes = self.info_class.loc[
            self.info.iloc[rec_ind].name].index
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

        rec_inds = self.info.iloc[rec_ind]['index']
        len_rec_inds = len(rec_inds)

        row = self.info.iloc[rec_ind].name
        if not isinstance(row, tuple):
            # Theres's only one category, e.g. "subject"
            row = tuple([row])
        available_indices = self.info_class.loc[
            row + tuple([class_ind]), 'index']
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
            start_ind, _, _ = self._sample_seq_start_ind()
            yield tuple(range(start_ind, start_ind + self.n_windows))
