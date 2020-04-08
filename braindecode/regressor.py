import numpy as np
import torch
from sklearn.metrics import get_scorer
from skorch.callbacks import EpochTimer, BatchScoring, PrintLog, EpochScoring
from skorch.classifier import NeuralNet
from skorch.regressor import NeuralNetRegressor
from skorch.utils import train_loss_score, valid_loss_score, noop

from .scoring import PostEpochTrainScoring, CroppedTrialEpochScoring


class EEGRegressor(NeuralNetRegressor):
    """Regressor that does not assume softmax activation.
    Calls loss function directly without applying log or anything.

    Parameters
    ----------
    cropped: bool (default=False)
        Defines whether torch model passed to this class is cropped or not.
        Currently used for callbacks definition.
    callbacks: None or list of strings or list of Callback instances (default=None)
        More callbacks, in addition to those returned by
        ``get_default_callbacks``. Each callback should inherit from
        :class:`skorch.callbacks.Callback`. If not ``None``, callbacks can be a
        list of strings specifying `sklearn` scoring functions (for scoring
        functions names see: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)
        or a list of callbacks where the callback names are inferred from the
        class name. Name conflicts are resolved by appending a count suffix
        starting with 1, e.g. ``EpochScoring_1``. Alternatively,
        a tuple ``(name, callback)`` can be passed, where ``name``
        should be unique. Callbacks may or may not be instantiated.
        The callback name can be used to set parameters on specific
        callbacks (e.g., for the callback with name ``'print_log'``, use
        ``net.set_params(callbacks__print_log__keys_ignored=['epoch',
        'train_loss'])``).
    """
    # TODO: Update docstring to use NeuralNetRegressor docstring with some
    #  imporvements

    def __init__(self, *args, cropped=False, callbacks=None, **kwargs):
        self.cropped = cropped
        callbacks = self._parse_callbacks(callbacks)

        super().__init__(*args, callbacks=callbacks, **kwargs)

    def _parse_callbacks(self, callbacks):
        callbacks_list = []
        if callbacks is not None:
            for callback in callbacks:
                if isinstance(callback, tuple):
                    callbacks_list.append(callback)
                else:
                    assert isinstance(callback, str)
                    lower_is_better = False
                    train_name = f'train_{callback}'
                    valid_name = f'valid_{callback}'
                    if self.cropped:
                        # In case of cropped decoding we are using braindecode
                        # specific scoring created for cropped decoding
                        train_scoring = CroppedTrialEpochScoring(
                            callback, lower_is_better, on_train=True, name=train_name
                        )
                        valid_scoring = CroppedTrialEpochScoring(
                            callback, lower_is_better, on_train=False, name=valid_name
                        )
                    else:
                        train_scoring = PostEpochTrainScoring(
                            callback, lower_is_better, name=train_name
                        )
                        valid_scoring = EpochScoring(
                            callback, lower_is_better, on_train=False, name=valid_name
                        )
                    callbacks_list.extend([
                        (train_name, train_scoring),
                        (valid_name, valid_scoring)
                    ])

        return callbacks_list

    # pylint: disable=arguments-differ
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        """Return the loss for this batch by calling NeuralNet get_loss.
        Parameters
        ----------
        y_pred : torch tensor
          Predicted target values
        y_true : torch tensor
          True target values.
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:
            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset
          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.
        training : bool (default=False)
          Whether train mode should be used or not.

        """
        return NeuralNet.get_loss(self, y_pred, y_true, *args, **kwargs)

    def get_iterator(self, dataset, training=False, drop_index=True):
        iterator = super().get_iterator(dataset, training=training)
        if drop_index:
            return ThrowAwayIndexLoader(self, iterator)
        else:
            return iterator

    def on_batch_end(self, net, X, y, training=False, **kwargs):
        # If training is false, assume that our loader has indices for this
        # batch
        if not training:
            cbs = self._default_callbacks + self.callbacks
            epoch_cbs = []
            for name, cb in cbs:
                if (cb.__class__.__name__ == 'CroppedTrialEpochScoring') and (
                    hasattr(cb, 'supercrop_inds_')) and (cb.on_train == False):
                    epoch_cbs.append(cb)
            # for trialwise decoding stuffs it might also be we don't have
            # cropped loader, so no indices there
            if len(epoch_cbs) > 0:
                assert hasattr(self, '_last_supercrop_inds')
                for cb in epoch_cbs:
                    cb.supercrop_inds_.append(self._last_supercrop_inds)
                del self._last_supercrop_inds

    def predict_with_supercrop_inds_and_ys(self, dataset):
        preds = []
        i_supercrop_in_trials = []
        i_supercrop_stops = []
        supercrop_ys = []
        for X, y, i in self.get_iterator(dataset, drop_index=False):
            i_supercrop_in_trials.append(i[0].cpu().numpy())
            i_supercrop_stops.append(i[2].cpu().numpy())
            preds.append(self.predict_proba(X))
            supercrop_ys.append(y.cpu().numpy())
        preds = np.concatenate(preds)
        i_supercrop_in_trials = np.concatenate(i_supercrop_in_trials)
        i_supercrop_stops = np.concatenate(i_supercrop_stops)
        supercrop_ys = np.concatenate(supercrop_ys)
        return dict(
            preds=preds, i_supercrop_in_trials=i_supercrop_in_trials,
            i_supercrop_stops=i_supercrop_stops, supercrop_ys=supercrop_ys)

    # Removes default EpochScoring callback computing 'accuracy' to work properly
    # with cropped decoding.
    @property
    def _default_callbacks(self):
        return [
            ("epoch_timer", EpochTimer()),
            (
                "train_loss",
                BatchScoring(
                    train_loss_score,
                    name="train_loss",
                    on_train=True,
                    target_extractor=noop,
                ),
            ),
            (
                "valid_loss",
                BatchScoring(
                    valid_loss_score, name="valid_loss", target_extractor=noop,
                ),
            ),
            ("print_log", PrintLog()),
        ]


class ThrowAwayIndexLoader(object):
    def __init__(self, net, loader):
        self.net = net
        self.loader = loader
        self.last_i = None

    def __iter__(self, ):
        normal_iter = self.loader.__iter__()
        for batch in normal_iter:
            if len(batch) == 3:
                x,y,i = batch
                # Store for scoring callbacks
                self.net._last_supercrop_inds = i
            else:
                x,y = batch

            # TODO: should be on dataset side
            if hasattr(x, 'type'):
                x = x.type(torch.float32)
                y = y.type(torch.float32)
            yield x,y
