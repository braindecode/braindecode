# Authors: Maciej Sliwowski
#          Robin Tibor Schirrmeister
#          Alexandre Gramfort
#
# License: BSD-3

from contextlib import contextmanager

import numpy as np
import torch
from skorch.callbacks.scoring import EpochScoring

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


# class PostEpochScoring(EpochScoring):
#    def on_epoch_end(self, net, dataset_train, dataset_valid, **kwargs):
#        EpochScoring.on_epoch_end()


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
