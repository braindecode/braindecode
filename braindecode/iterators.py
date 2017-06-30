import numpy as np
from numpy.random import RandomState

def get_balanced_batches(n_trials, rng, shuffle, n_batches=None,
                         batch_size=None):
    """Create indices for batches balanced in size (batches will have maximum size difference of 1).
    Supply either batch size or number of batches. Resulting batches
    will not have the given batch size but rather the next largest batch size
    that allows to split the set into balanced batches (maximum size difference 1).

    Parameters
    ----------
    n_trials : int
        Size of set.
    rng :

    shuffle :
        Whether to shuffle indices before splitting set.
    n_batches :
         (Default value = None)
    batch_size :
         (Default value = None)

    Returns
    -------

    """
    assert batch_size is not None or n_batches is not None
    if n_batches is None:
        n_batches = int(np.round(n_trials / float(batch_size)))

    if n_batches > 0:
        min_batch_size = n_trials // n_batches
        n_batches_with_extra_trial = n_trials % n_batches
    else:
        n_batches = 1
        min_batch_size = n_trials
        n_batches_with_extra_trial = 0
    assert n_batches_with_extra_trial < n_batches
    all_inds = np.array(range(n_trials))
    if shuffle:
        rng.shuffle(all_inds)
    i_start_trial = 0
    i_stop_trial = 0
    batches = []
    for i_batch in range(n_batches):
        i_stop_trial += min_batch_size
        if i_batch < n_batches_with_extra_trial:
            i_stop_trial += 1
        batch_inds = all_inds[range(i_start_trial, i_stop_trial)]
        batches.append(batch_inds)
        i_start_trial = i_stop_trial
    assert i_start_trial == n_trials
    return batches


class BalancedBatchSizeIterator(object):
    """
    Create batches of balanced size.
    Parameters
    ----------
    batch_size: int
        Resulting batches will not necessarily have the given batch size
        but rather the next largest batch size that allows to split the set into
        balanced batches (maximum size difference 1).
    """
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.rng = RandomState(328774)

    def get_batches(self, dataset, shuffle):
        n_trials = dataset.X.shape[0]
        batches = get_balanced_batches(n_trials,
                                       batch_size=self.batch_size,
                                       rng=self.rng,
                                       shuffle=shuffle)
        for batch_inds in batches:
            yield (dataset.X[batch_inds], dataset.y[batch_inds])

    def reset_rng(self):
        self.rng = RandomState(328774)


class CropsFromTrialsIterator(object):
    def __init__(self, batch_size, input_time_length, n_preds_per_input,
                 check_preds_smaller_trial_len=True):
        # TODO: remove this check preds smaller trila len and check that
        # trials are smaller than input length
        """

        Parameters
        ----------
        batch_size: int
        input_time_length: int
            Input time length of the ConvNet, determines size of batches in
            3rd dimension.
        n_preds_per_input: int
            Number of predictions ConvNet makes per one input. Can be computed
            by making a forward pass with the given input time length, the
            output length in 3rd dimension is n_preds_per_input.
        """
        self.batch_size = batch_size
        self.input_time_length = input_time_length
        self.n_preds_per_input = n_preds_per_input
        self.check_preds_smaller_trial_len = check_preds_smaller_trial_len
        self.rng = RandomState((2017, 6, 28))

    def reset_rng(self):
        self.rng = RandomState((2017, 6, 28))

    def get_batches(self, dataset, shuffle):
        # start always at first predictable sample, so
        # start at end of receptive field
        n_receptive_field = self.input_time_length - self.n_preds_per_input + 1
        i_trial_starts = [n_receptive_field - 1] * len(dataset.X)
        i_trial_stops = [trial.shape[1] for trial in dataset.X]

        if self.check_preds_smaller_trial_len:
            check_trial_bounds(i_trial_starts, i_trial_stops,
                               self.n_preds_per_input)
        start_stop_blocks_per_trial = compute_start_stop_block_inds(
            i_trial_starts, i_trial_stops, self.input_time_length,
            self.n_preds_per_input, self.check_preds_smaller_trial_len)
        for i_trial, trial_blocks in enumerate(start_stop_blocks_per_trial):
            assert trial_blocks[0][0] == 0
            assert trial_blocks[-1][1] == i_trial_stops[i_trial]

        return self.yield_block_batches(dataset.X, dataset.y,
                                        start_stop_blocks_per_trial,
                                        shuffle=shuffle)

    def yield_block_batches(self, X, y, start_stop_blocks_per_trial, shuffle):
        # add trial nr to start stop blocks and flatten at same time
        i_trial_start_stop_block = [(i_trial, start, stop)
                                      for i_trial, block in
                                          enumerate(start_stop_blocks_per_trial)
                                      for (start, stop) in block]
        if shuffle:
            self.rng.shuffle(i_trial_start_stop_block)

        for i_block in range(0, len(i_trial_start_stop_block), self.batch_size):
            i_block_stop = min(i_block + self.batch_size,
                               len(i_trial_start_stop_block))
            start_stop_blocks = i_trial_start_stop_block[i_block:i_block_stop]
            batch = create_batch_from_i_trial_start_stop_blocks(
                X, y, start_stop_blocks, self.n_preds_per_input)
            yield batch


def check_trial_bounds(i_trial_starts, i_trial_stops, n_preds_per_input):
    for start, stop in zip(i_trial_starts, i_trial_stops):
        assert stop - start >= n_preds_per_input, (
            "Trial should be longer or equal than number of sample preds, "
            "Trial length: {:d}, sample preds {:d}...".
                format(stop - start, n_preds_per_input))


def compute_start_stop_block_inds(i_trial_starts, i_trial_stops,
                                 input_time_length, n_preds_per_input,
                                 check_preds_smaller_trial_len):
    # create start stop indices for all batches still 2d trial -> start stop
    start_stop_blocks_per_trial = []
    for i_trial in range(len(i_trial_starts)):
        trial_start = i_trial_starts[i_trial]
        trial_stop = i_trial_stops[i_trial]
        start_stop_blocks = get_start_stop_blocks_for_trial(
            trial_start, trial_stop, input_time_length,
            n_preds_per_input)

        if check_preds_smaller_trial_len:
            # check that block is correct, all predicted samples together
            # should be the trial samples
            all_predicted_samples = [
                range(stop - n_preds_per_input,
                      stop) for _,stop in start_stop_blocks]
            # this check takes about 50 ms in performance test
            # whereas loop itself takes only 5 ms.. deactivate it if not necessary
            assert np.array_equal(
                range(i_trial_starts[i_trial], i_trial_stops[i_trial]),
                np.unique(np.concatenate(all_predicted_samples)))

        start_stop_blocks_per_trial.append(start_stop_blocks)
    return start_stop_blocks_per_trial


def get_start_stop_blocks_for_trial(trial_start, trial_stop, input_time_length,
                                   n_preds_per_input):
    start_stop_blocks = []
    i_window_stop = trial_start  # now when we add sample preds in loop,
    # first sample of trial corresponds to first prediction
    while i_window_stop < trial_stop:
        i_window_stop += n_preds_per_input
        i_adjusted_stop = min(i_window_stop, trial_stop)
        i_window_start = i_adjusted_stop - input_time_length
        start_stop_blocks.append((i_window_start, i_adjusted_stop))

    return start_stop_blocks


def create_batch_from_i_trial_start_stop_blocks(X, y, i_trial_start_stop_block,
                                               n_preds_per_input):
    Xs = []
    ys = []
    for i_trial, start, stop in i_trial_start_stop_block:
        Xs.append(X[i_trial][:,start:stop])
        # one-hot-encode targets
        #block_y = np.zeros((n_classes, n_preds_per_input), dtype=np.float32)
        #block_y[y[i_trial]] = 1
        #block_y = np.ones((n_preds_per_input), dtype=np.int64) * y[i_trial]
        #ys.append(block_y)
        ys.append(y[i_trial])
    batch_X = np.array(Xs)
    batch_y = np.array(ys)
    return batch_X, batch_y
