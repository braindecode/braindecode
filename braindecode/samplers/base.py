"""
Sampler classes.
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)

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
        return self.rng.choice(self.n_recordings)

    def sample_window(self, rec_ind=None):
        """Return a specific window.
        """
        if rec_ind is None:
            rec_ind = self.sample_recording()
        win_ind = self.rng.choice(self.info.iloc[rec_ind]['index'])
        return win_ind, rec_ind

    def __iter__(self):
        raise NotImplementedError

    @property
    def n_recordings(self):
        return self.info.shape[0]
