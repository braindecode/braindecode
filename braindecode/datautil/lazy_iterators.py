from torch.utils.data import DataLoader
from numpy.random import RandomState
from functools import partial
import torch as th
import numpy as np

from braindecode.datautil.iterators import (
    _compute_start_stop_block_inds,
    get_balanced_batches,
)


def custom_collate(batch, rng_state=None):
    """ Puts each data field into a ndarray with outer dimension batch size.
    Taken and adapted from pytorch to return ndarrays instead of tensors:
    https://pytorch.org/docs/0.4.1/_modules/torch/utils/data/dataloader.html

    this function is needed, since tensors require more system RAM which we
    want to decrease using lazy loading
    """
    elem_type = type(batch[0])
    if elem_type.__module__ == "numpy":
        if rng_state is not None:
            th.random.set_rng_state(rng_state)
        return np.stack([b for b in batch], 0)

    elif isinstance(batch[0], tuple):
        transposed = zip(*batch)
        return [custom_collate(samples, rng_state) for samples in transposed]


class LazyCropsFromTrialsIterator(object):
    """ This is basically the same code as CropsFromTrialsIterator adapted to
    work with lazy datasets. It uses pytorch DataLoader to load recordings
    from hdd with multiple threads when the data is actually needed. Reduces
    overall RAM requirements.

    Parameters
    ----------
    input_time_length: int
        Input time length of the ConvNet, determines size of batches in
        3rd dimension.
    n_preds_per_input: int
        Number of predictions ConvNet makes per one input. Can be computed
        by making a forward pass with the given input time length, the
        output length in 3rd dimension is n_preds_per_input.
    batch_size: int
    seed: int
        Random seed for initialization of `numpy.RandomState` random generator
        that shuffles the batches.
    num_workers: int
        The number of workers to load crops in parallel
    collate_fn: func
        Merges a list of samples to form a mini-batch
    check_preds_smaller_trial_len: bool
        Checking validity of predictions and trial lengths. Disable to decrease
        runtime.
    """

    def __init__(
        self,
        input_time_length,
        n_preds_per_input,
        batch_size,
        seed=328774,
        num_workers=0,
        collate_fn=custom_collate,
        check_preds_smaller_trial_len=True,
        reset_rng_after_each_batch=False,
    ):
        self.batch_size = batch_size
        self.seed = seed
        self.rng = RandomState(self.seed)
        self.input_time_length = input_time_length
        self.n_preds_per_input = n_preds_per_input
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.check_preds_smaller_trial_len = check_preds_smaller_trial_len
        self.reset_rng_after_each_batch = reset_rng_after_each_batch

    def reset_rng(self):
        self.rng = RandomState(self.seed)

    def get_batches(self, dataset, shuffle):
        # in pytorch 1.0.0, internal random state is changed when using a
        # DataLoader, even if num_workers is 0. this did not happen in torch
        # 0.4.0 and breaks our equality tests of traditional and lazy loading
        # therefore, in the collate function of every batch, reset to the
        # random state before iterating through batches.
        if self.reset_rng_after_each_batch:
            random_state = th.random.get_rng_state()
            collate_fn = partial(self.collate_fn, rng_state=random_state)
        else:
            collate_fn = partial(self.collate_fn, rng_state=None)
        batch_indeces = self._get_batch_indeces(dataset=dataset, shuffle=shuffle)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=batch_indeces,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
        )
        return data_loader

    def _get_batch_indeces(self, dataset, shuffle):
        # start always at first predictable sample, so
        # start at end of receptive field
        n_receptive_field = self.input_time_length - self.n_preds_per_input + 1
        i_trial_starts = [n_receptive_field - 1] * len(dataset.X)
        i_trial_stops = [trial.shape[1] for trial in dataset.X]

        # Check whether input lengths ok
        input_lens = i_trial_stops
        for i_trial, input_len in enumerate(input_lens):
            assert input_len >= self.input_time_length, (
                "Input length {:d} of trial {:d} is smaller than the "
                "input time length {:d}".format(
                    input_len, i_trial, self.input_time_length
                )
            )

        start_stop_blocks_per_trial = _compute_start_stop_block_inds(
            i_trial_starts,
            i_trial_stops,
            self.input_time_length,
            self.n_preds_per_input,
            check_preds_smaller_trial_len=self.check_preds_smaller_trial_len,
        )
        for i_trial, trial_blocks in enumerate(start_stop_blocks_per_trial):
            assert trial_blocks[0][0] == 0
            assert trial_blocks[-1][1] == i_trial_stops[i_trial]

        i_trial_start_stop_block = np.array(
            [
                (i_trial, start, stop)
                for i_trial, block in enumerate(start_stop_blocks_per_trial)
                for start, stop in block
            ]
        )

        batches = get_balanced_batches(
            n_trials=len(i_trial_start_stop_block),
            rng=self.rng,
            shuffle=shuffle,
            batch_size=self.batch_size,
        )

        return [i_trial_start_stop_block[batch_ind] for batch_ind in batches]
