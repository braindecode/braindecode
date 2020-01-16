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

from .monitors import compute_trial_labels_from_crop_preds


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

    def on_epoch_end(self, net, dataset_train, dataset_valid, **kwargs):
        assert self.use_caching == True
        if self.on_train:
            # Recompute Predictions for caching outside of training loop
            self.y_preds_ = list(
                net.forward_iter(dataset_train, training=False)
            )
            self.y_trues_ = [dataset_train.y]

        X_test, _, y_pred = self.get_test_data(dataset_train, dataset_valid)
        if X_test is None:
            return

        # Acquire loader to know input_time_length
        input_time_length = net.get_iterator(
            X_test, training=False
        ).input_time_length

        # This assumes X_test is a dataset with X and y :(
        trial_X = X_test.X
        trial_y = X_test.y

        y_pred_np = [old_y_pred.cpu().numpy() for old_y_pred in y_pred]

        y_preds_per_trial = compute_trial_labels_from_crop_preds(
            y_pred_np, input_time_length, trial_X
        )

        # Move into format expected by skorch (list of torch tensors)
        y_preds_per_trial = [torch.tensor(np.array(y_preds_per_trial))]

        with _cache_net_forward_iter(
            net, self.use_caching, y_preds_per_trial
        ) as cached_net:
            current_score = self._scoring(cached_net, trial_X, trial_y)

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
