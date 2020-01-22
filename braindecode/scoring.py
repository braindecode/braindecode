# Authors: Maciej Sliwowski
#          Robin Tibor Schirrmeister
#          Alexandre Gramfort
#
# License: BSD-3

from contextlib import contextmanager

import numpy as np
import torch
from skorch.callbacks.scoring import EpochScoring
from skorch.utils import to_numpy
from skorch.dataset import unpack_data

from braindecode.monitors import compute_preds_per_trial_from_trial_n_samples
from .monitors import compute_preds_per_trial_from_crops


def trial_preds_from_supercrop_preds(
        preds, supercrop_inds):
    """
    Assigning supercrop predictions to trials  while removing duplicate
    predictions.

    Parameters
    ----------
    preds: list of ndarrays (atleast 2darrays)
        List of supercrop predictions, in each supercrop prediction
         time is in axis=1
    supercrop_inds: list of 3 lists
        Index/number of supercrop in trial, start position of supercrop in trial
        stop position of supercrop in trial

    Returns
    -------
    preds_per_trial: list of ndarrays
        Predictions in each trial, duplicates removed

    """
    # supercrop inds are list of
    # 3 lists (i_supercrop_in_trials, i_starts, i_stops)
    assert len(preds) == len(supercrop_inds[0])
    assert len(supercrop_inds[1]) == len(supercrop_inds[2])

    # Algorithm for assigning supercrop predictions to trials
    # while removing duplicate predictions:
    # Loop through supercrops:
    # In each iteration you have predictions (assumed: #classes x #timesteps,
    # or at least #timesteps must be in axis=1)
    # and you have i_supercrop_in_trial, i_start_in_trial, i_stop_in_trial
    # (i_trial removed from variable names for brevity)
    # You first check if the i_supercrop_in_trial is 1 larger
    # than in last iteration, then you are still in the same trial
    # Otherwise you are in a new trial
    # If you are in the same trial, you check for duplicate predictions
    # Only take predictions that are after (inclusive)
    # the stop of the last iteration (i.e., the index of final prediction
    # in the last iteration)
    # Then add the duplicate-removed predictions from this supercrop
    # to predictions for current trial
    preds_per_trial = []
    cur_trial_preds = []
    i_last_stop = None
    i_last_supercrop = -1
    # zip(*supercrop_inds) to make the three lists into
    # a single list of 3 tuples
    for supercrop_preds, (i_supercrop, i_start, i_stop) in zip(
            preds, zip(*supercrop_inds)):
        supercrop_preds = np.array(supercrop_preds)
        if i_supercrop != (i_last_supercrop + 1):
            assert i_supercrop == 0, (
                "supercrop numbers in new trial should start from 0")
            preds_per_trial.append(np.concatenate(cur_trial_preds, axis=1))
            cur_trial_preds = []
            i_last_stop = None

        if i_last_stop is not None:
            # Remove duplicates
            n_needed_preds = i_stop - i_last_stop
            supercrop_preds = supercrop_preds[:, -n_needed_preds:]
        cur_trial_preds.append(supercrop_preds)
        i_last_supercrop = i_supercrop
        i_last_stop = i_stop
    # add last trial preds
    preds_per_trial.append(np.concatenate(cur_trial_preds, axis=1))
    return preds_per_trial


@contextmanager
def _cache_net_forward_iter(net, use_caching, y_preds):
    """Caching context for ``skorch.NeuralNet`` instance.
    Returns a modified version of the net whose ``forward_iter``
    method will subsequently return cached predictions. Leaving the
    context will undo the overwrite of the ``forward_iter`` method.
    """
    if not use_caching:
        yield net
        return
    y_preds = iter(y_preds)

    # pylint: disable=unused-argument
    def cached_forward_iter(*args, device=net.device, **kwargs):
        for yp in y_preds:
            yield yp.to(device=device)

    net.forward_iter = cached_forward_iter
    try:
        yield net
    finally:
        # By setting net.forward_iter we define an attribute
        # `forward_iter` that precedes the bound method
        # `forward_iter`. By deleting the entry from the attribute
        # dict we undo this.
        del net.__dict__["forward_iter"]


class CroppedTrialEpochScoring(EpochScoring):
    """
    Class to compute scores for trials from a model that predicts (super)crops.
    """

    def __init__(
        self,
        scoring,
        lower_is_better=True,
        on_train=False,
        name=None,
        target_extractor=to_numpy,
        use_caching=True,
        input_time_length=None,
    ):
        self.input_time_length = input_time_length
        super().__init__(
            scoring=scoring,
            lower_is_better=lower_is_better,
            on_train=on_train,
            name=name,
            target_extractor=target_extractor,
            use_caching=use_caching,
        )

    def on_epoch_end(self, net, dataset_train, dataset_valid, **kwargs):
        assert self.use_caching == True
        if self.on_train:
            # Recompute Predictions for caching outside of training loop
            self.y_preds_ = list(
                net.forward_iter(dataset_train, training=False)
            )
            from copy import deepcopy # XXX: remove

            self.y_trues_ = deepcopy(self.y_preds_)

        X_test, _, y_pred = self.get_test_data(
            dataset_train, dataset_valid
        )
        if X_test is None:
            return

        if self.input_time_length is None:
            # Acquire loader to know input_time_length
            input_time_length = net.get_iterator(
                X_test, training=False
            ).input_time_length

        y_pred_np = [old_y_pred.cpu().numpy() for old_y_pred in y_pred]

        if self.on_train:
            dataset = dataset_train
        else:
            dataset = dataset_valid
        if self.input_time_length is None:
            # This assumes X_test is a dataset with X and y
            trial_X = X_test.X
            trial_y = X_test.y
            preds_per_crop = compute_preds_per_trial_from_crops(
                y_pred_np, input_time_length, trial_X
            )
        else:
            # HACK: just fix to get BCIC IV 2a example to run
            trial_lens_samples = dataset.dataset.get_trial_durations_samples()[:288]
            preds_per_crop = compute_preds_per_trial_from_trial_n_samples(
                y_pred_np, self.input_time_length, trial_lens_samples
            )
            # Let's get trial y
            y_per_super_crop = [dataset[i][1] for i in range(len(dataset))]

            # HACK: just for now to get BCIC IV 2a example to run
            n_super_crops = len(y_per_super_crop)
            n_trials = len(trial_lens_samples)
            n_super_crops_per_trial = int(n_super_crops / n_trials)
            trial_y = y_per_super_crop[::n_super_crops_per_trial]

        y_preds_per_trial = np.array(
            [np.mean(p, axis=1) for p in preds_per_crop]
        )

        # Move into format expected by skorch (list of torch tensors)
        y_preds_per_trial = [torch.tensor(y_preds_per_trial)]

        with _cache_net_forward_iter(
            net, self.use_caching, y_preds_per_trial
        ) as cached_net:
            current_score = self._scoring(cached_net, dataset, trial_y)

        self._record_score(net.history, current_score)


class PostEpochTrainScoring(EpochScoring):
    """
    Epoch Scoring class that recomputes predictions after the epoch
    on the training in validation mode.

    Note: For unknown reasons, this affects global random generator and
    therefore all results may change slightly if you add this scoring callback.

    Parameters
    ----------
    scoring : None, str, or callable (default=None)
      If None, use the ``score`` method of the model. If str, it
      should be a valid sklearn scorer (e.g. "f1", "accuracy"). If a
      callable, it should have the signature (model, X, y), and it
      should return a scalar. This works analogously to the
      ``scoring`` parameter in sklearn's ``GridSearchCV`` et al.
    lower_is_better : bool (default=True)
      Whether lower scores should be considered better or worse.
    name : str or None (default=None)
      If not an explicit string, tries to infer the name from the
      ``scoring`` argument.
    target_extractor : callable (default=to_numpy)
      This is called on y before it is passed to scoring.
    """

    def __init__(
        self,
        scoring,
        lower_is_better=True,
        name=None,
        target_extractor=to_numpy,
    ):
        super().__init__(
            scoring=scoring,
            lower_is_better=lower_is_better,
            on_train=True,
            name=name,
            target_extractor=target_extractor,
            use_caching=False,
        )

    def on_epoch_end(self, net, dataset_train, dataset_valid, **kwargs):
        if len(self.y_preds_) == 0:
            dataset = net.get_dataset(dataset_train)
            iterator = net.get_iterator(dataset, training=False)
            y_preds = []
            y_test = []
            for data in iterator:
                batch_X, batch_y = unpack_data(data)
                yp = net.evaluation_step(batch_X, training=False)
                yp = yp.to(device="cpu")
                y_test.append(self.target_extractor(batch_y))
                y_preds.append(yp)
            y_test = np.concatenate(y_test)

            # Adding the recomputed preds to all other
            # instances of PostEpochTrainScoring of this
            # Skorch-Net (NeuralNet, BraindecodeClassifier etc.)
            # (They will be reinitialized to empty lists by skorch
            # each epoch)
            cbs = net._default_callbacks + net.callbacks
            epoch_cbs = [
                cb for name, cb in cbs if isinstance(cb, PostEpochTrainScoring)
            ]
            for cb in epoch_cbs:
                cb.y_preds_ = y_preds
                cb.y_trues_ = y_test

        # y pred should be same as self.y_preds_
        with _cache_net_forward_iter(
            net, use_caching=True, y_preds=self.y_preds_
        ) as cached_net:
            current_score = self._scoring(
                cached_net, dataset_train, self.y_trues_
            )
        self._record_score(net.history, current_score)
