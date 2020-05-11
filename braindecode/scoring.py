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


def trial_preds_from_window_preds(
        preds, i_window_in_trials, i_stop_in_trials):
    """
    Assigning window predictions to trials  while removing duplicate
    predictions.

    Parameters
    ----------
    preds: list of ndarrays (atleast 2darrays)
        List of window predictions, in each window prediction
         time is in axis=1
    i_window_in_trials: list
        Index/number of window in trial
    i_stop_in_trials: list
        stop position of window in trial

    Returns
    -------
    preds_per_trial: list of ndarrays
        Predictions in each trial, duplicates removed

    """
    assert len(preds) == len(i_window_in_trials)
    assert len(i_window_in_trials) == len(i_stop_in_trials)

    # Algorithm for assigning window predictions to trials
    # while removing duplicate predictions:
    # Loop through windows:
    # In each iteration you have predictions (assumed: #classes x #timesteps,
    # or at least #timesteps must be in axis=1)
    # and you have i_window_in_trial, i_stop_in_trial
    # (i_trial removed from variable names for brevity)
    # You first check if the i_window_in_trial is 1 larger
    # than in last iteration, then you are still in the same trial
    # Otherwise you are in a new trial
    # If you are in the same trial, you check for duplicate predictions
    # Only take predictions that are after (inclusive)
    # the stop of the last iteration (i.e., the index of final prediction
    # in the last iteration)
    # Then add the duplicate-removed predictions from this window
    # to predictions for current trial
    preds_per_trial = []
    cur_trial_preds = []
    i_last_stop = None
    i_last_window = -1
    for window_preds, i_window, i_stop in zip(
            preds, i_window_in_trials, i_stop_in_trials):
        window_preds = np.array(window_preds)
        if i_window != (i_last_window + 1):
            assert i_window == 0, (
                "window numbers in new trial should start from 0")
            preds_per_trial.append(np.concatenate(cur_trial_preds, axis=1))
            cur_trial_preds = []
            i_last_stop = None

        if i_last_stop is not None:
            # Remove duplicates
            n_needed_preds = i_stop - i_last_stop
            window_preds = window_preds[:, -n_needed_preds:]
        cur_trial_preds.append(window_preds)
        i_last_window = i_window
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
    ):
        super().__init__(
            scoring=scoring,
            lower_is_better=lower_is_better,
            on_train=on_train,
            name=name,
            target_extractor=target_extractor,
            use_caching=use_caching,
        )
        if not self.on_train:
            self.window_inds_ = []


    def _initialize_cache(self):
        super()._initialize_cache()
        self.crops_to_trials_computed = False
        self.y_trues_ = []
        self.y_preds_ = []
        if not self.on_train:
            self.window_inds_ = []

    def on_epoch_end(self, net, dataset_train, dataset_valid, **kwargs):
        assert self.use_caching == True
        if not self.crops_to_trials_computed:
            if self.on_train:
                pred_results = net.predict_with_window_inds_and_ys(
                    dataset_train)

            else:
                pred_results = {}
                pred_results['i_window_in_trials'] = np.concatenate(
                    [i[0].cpu().numpy() for i in self.window_inds_]
                )
                pred_results['i_window_stops'] = np.concatenate(
                    [i[2].cpu().numpy() for i in self.window_inds_]
                )
                pred_results['preds'] = np.concatenate(
                    [y_pred.cpu().numpy() for y_pred in self.y_preds_])
                pred_results['window_ys'] = np.concatenate(
                    [y.cpu().numpy() for y in self.y_trues_])


            # A new trial starts
            # when the index of the window in trials
            # does not increment by 1
            # Add dummy infinity at start
            window_0_per_trial_mask = np.diff(
                pred_results['i_window_in_trials'], prepend=[np.inf]) != 1
            trial_ys = pred_results['window_ys'][window_0_per_trial_mask]
            trial_preds = trial_preds_from_window_preds(
                pred_results['preds'],
                pred_results['i_window_in_trials'],
                pred_results['i_window_stops'])
            # trial preds is a list
            # each item is an 2darray classes x time
            y_preds_per_trial = np.array(
                [np.mean(p, axis=1) for p in trial_preds]
            )
            # Move into format expected by skorch (list of torch tensors)
            y_preds_per_trial = [torch.tensor(y_preds_per_trial)]

            # Store the computed trial labels/preds for all Cropped Callbacks
            # that are also on same set
            cbs = net._default_callbacks + net.callbacks
            epoch_cbs = [
                cb for name, cb in cbs if
                isinstance(cb, CroppedTrialEpochScoring) and (
                    cb.on_train == self.on_train)
            ]
            for cb in epoch_cbs:
                cb.y_preds_ = y_preds_per_trial
                cb.y_trues_ = trial_ys
                cb.crops_to_trials_computed = True

        dataset = dataset_train if self.on_train else dataset_valid

        with _cache_net_forward_iter(
            net, self.use_caching, self.y_preds_
        ) as cached_net:
            current_score = self._scoring(cached_net, dataset, self.y_trues_)
        self._record_score(net.history, current_score)

        return


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
