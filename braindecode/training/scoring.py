# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Robin Tibor Schirrmeister <robintibor@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Lukas Gemein <l.gemein@gmail.com>
#          Mohammed Fattouh <mo.fattouh@gmail.com>
#
# License: BSD-3

from contextlib import contextmanager
import warnings

import numpy as np
import torch
from mne.utils.check import check_version
from skorch.callbacks.scoring import EpochScoring
from skorch.utils import to_numpy
from skorch.dataset import unpack_data
from torch.utils.data import DataLoader


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
    assert len(preds) == len(i_window_in_trials) == len(i_stop_in_trials), (
        f'{len(preds)}, {len(i_window_in_trials)}, {len(i_stop_in_trials)}')

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
    # XXX needs a docstring !!!

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
        assert self.use_caching
        if not self.crops_to_trials_computed:
            if self.on_train:
                # Prevent that rng state of torch is changed by
                # creation+usage of iterator
                rng_state = torch.random.get_rng_state()
                pred_results = net.predict_with_window_inds_and_ys(
                    dataset_train)
                torch.random.set_rng_state(rng_state)
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

            # Average across the timesteps of each trial so we have per-trial
            # predictions already, these will be just passed through the forward
            # method of the classifier/regressor to the skorch scoring function.
            # trial_preds is a list, each item is a 2d array classes x time
            y_preds_per_trial = np.array(
                [np.mean(p, axis=1) for p in trial_preds]
            )
            # Move into format expected by skorch (list of torch tensors)
            y_preds_per_trial = [torch.tensor(y_preds_per_trial)]

            # Store the computed trial preds for all Cropped Callbacks
            # that are also on same set
            cbs = net.callbacks_
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


class CroppedTimeSeriesEpochScoring(CroppedTrialEpochScoring):
    """
    Class to compute scores for trials from a model that predicts (super)crops with
    time series target.
    """
    def on_epoch_end(self, net, dataset_train, dataset_valid, **kwargs):
        assert self.use_caching
        if not self.crops_to_trials_computed:
            if self.on_train:
                # Prevent that rng state of torch is changed by
                # creation+usage of iterator
                rng_state = torch.random.get_rng_state()
                pred_results = net.predict_with_window_inds_and_ys(
                    dataset_train)
                torch.random.set_rng_state(rng_state)
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

            num_preds = pred_results['preds'][-1].shape[-1]
            # slice the targets to fit preds shape
            pred_results['window_ys'] = [
                targets[:, -num_preds:] for targets in pred_results['window_ys']
            ]

            trial_preds = trial_preds_from_window_preds(
                pred_results['preds'],
                pred_results['i_window_in_trials'],
                pred_results['i_window_stops'])

            trial_ys = trial_preds_from_window_preds(
                pred_results['window_ys'],
                pred_results['i_window_in_trials'],
                pred_results['i_window_stops'])

            # the output is a list of predictions/targets per trial where each item is a
            # timeseries of predictions/targets of shape (n_classes x timesteps)

            # mask NaNs form targets
            preds = np.hstack(trial_preds)  # n_classes x timesteps in all trials
            targets = np.hstack(trial_ys)
            # create valid targets mask
            mask = ~np.isnan(targets)
            # select valid targets that have a matching predictions
            masked_targets = targets[mask]
            # For classification there is only one row in targets and n_classes rows in preds
            if mask.shape[0] != preds.shape[0]:
                masked_preds = preds[:, mask[0, :]]
            else:
                masked_preds = preds[mask]

            # Store the computed trial preds for all Cropped Callbacks
            # that are also on same set
            cbs = net.callbacks_
            epoch_cbs = [
                cb for name, cb in cbs if
                isinstance(cb, CroppedTimeSeriesEpochScoring) and (
                    cb.on_train == self.on_train)
            ]
            masked_preds = [torch.tensor(masked_preds.T)]
            for cb in epoch_cbs:
                cb.y_preds_ = masked_preds
                cb.y_trues_ = masked_targets.T
                cb.crops_to_trials_computed = True

        dataset = dataset_train if self.on_train else dataset_valid

        with _cache_net_forward_iter(
            net, self.use_caching, self.y_preds_
        ) as cached_net:
            current_score = self._scoring(cached_net, dataset, self.y_trues_)
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
            # Prevent that rng state of torch is changed by
            # creation+usage of iterator
            # Unfortunatenly calling __iter__() of a pytorch
            # DataLoader will change the random state
            # Note line below setting rng state back
            rng_state = torch.random.get_rng_state()
            iterator = net.get_iterator(dataset, training=False)
            y_preds = []
            y_test = []
            for batch in iterator:
                batch_X, batch_y = unpack_data(batch)
                # TODO: remove after skorch 0.10 release
                if not check_version('skorch', min_version='0.10.1'):
                    yp = net.evaluation_step(batch_X, training=False)
                # X, y unpacking has been pushed downstream in skorch 0.10
                else:
                    yp = net.evaluation_step(batch, training=False)
                yp = yp.to(device="cpu")
                y_test.append(self.target_extractor(batch_y))
                y_preds.append(yp)
            y_test = np.concatenate(y_test)
            torch.random.set_rng_state(rng_state)

            # Adding the recomputed preds to all other
            # instances of PostEpochTrainScoring of this
            # Skorch-Net (NeuralNet, BraindecodeClassifier etc.)
            # (They will be reinitialized to empty lists by skorch
            # each epoch)
            cbs = net.callbacks_
            epoch_cbs = [
                cb for name, cb in cbs if isinstance(cb, PostEpochTrainScoring)
            ]
            for cb in epoch_cbs:
                cb.y_preds_ = y_preds
                cb.y_trues_ = y_test
        # y pred should be same as self.y_preds_
        # Unclear if this also leads to any
        # random generator call?
        with _cache_net_forward_iter(
            net, use_caching=True, y_preds=self.y_preds_
        ) as cached_net:
            current_score = self._scoring(
                cached_net, dataset_train, self.y_trues_
            )
        self._record_score(net.history, current_score)


def predict_trials(module, dataset, return_targets=True):
    """Create trialwise predictions and optionally also return trialwise
    labels from cropped dataset given module.

    Parameters
    ----------
    module: torch.nn.Module
        A pytorch model implementing forward.
    dataset: braindecode.datasets.BaseConcatDataset
        A braindecode dataset to be predicted.
    return_targets: bool
        If True, additionally returns the trial targets.

    Returns
    -------
        trial_predictions: np.ndarray
            3-dimensional array (n_trials x n_classes x n_predictions), where
            the number of predictions depend on the chosen window size and the
            receptive field of the network.
        trial_labels: np.ndarray
            2-dimensional array (n_trials x n_targets) where the number of
            targets depends on the decoding paradigm and can be either a single
            value, multiple values, or a sequence.
    """
    # we have a cropped dataset if there exists at least one trial with more
    # than one compute window
    more_than_one_window = sum(dataset.get_metadata()['i_window_in_trial'] != 0) > 0
    if not more_than_one_window:
        warnings.warn('This function was designed to predict trials from '
                      'cropped datasets, which typically have multiple compute '
                      'windows per trial. The given dataset has exactly one '
                      'window per trial.')
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
    )
    all_preds, all_ys, all_inds = [], [], []
    with torch.no_grad():
        for X, y, ind in loader:
            preds = module(X)
            all_preds.extend(preds.cpu().numpy().astype(np.float32))
            all_ys.extend(y.cpu().numpy().astype(np.float32))
            all_inds.extend(ind)
    preds_per_trial = trial_preds_from_window_preds(
        preds=all_preds,
        i_window_in_trials=torch.cat(all_inds[0::3]),
        i_stop_in_trials=torch.cat(all_inds[2::3]),
    )
    preds_per_trial = np.array(preds_per_trial)
    if return_targets:
        if all_ys[0].shape == ():
            all_ys = np.array(all_ys)
            ys_per_trial = all_ys[
                np.diff(torch.cat(all_inds[0::3]), prepend=[np.inf]) != 1]
        else:
            ys_per_trial = trial_preds_from_window_preds(
                preds=all_ys,
                i_window_in_trials=torch.cat(all_inds[0::3]),
                i_stop_in_trials=torch.cat(all_inds[2::3]),
            )
            ys_per_trial = np.array(ys_per_trial)
        return preds_per_trial, ys_per_trial
    return preds_per_trial
