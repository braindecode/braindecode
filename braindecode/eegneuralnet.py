# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#          Pierre Guetschel
#
# License: BSD (3-clause)


import abc
import logging
import inspect

import numpy as np
import torch
from torch.utils.data import Dataset as torchDataset
from sklearn.metrics import get_scorer
from skorch.callbacks import BatchScoring, EpochScoring, EpochTimer, PrintLog
from skorch.utils import noop, to_numpy, train_loss_score, valid_loss_score

from .training.scoring import (CroppedTimeSeriesEpochScoring,
                               CroppedTrialEpochScoring, PostEpochTrainScoring)
from .datasets.base import BaseConcatDataset, WindowsDataset

log = logging.getLogger(__name__)


class _EEGNeuralNet(metaclass=abc.ABCMeta):
    @property
    def log(self):
        return log.getChild(self.__class__.__name__)

    def _yield_callbacks(self):
        # Here we parse the callbacks supplied as strings,
        # e.g. 'accuracy', to the callbacks skorch expects
        for name, cb, named_by_user in super()._yield_callbacks():
            if name == 'str':
                train_cb, valid_cb = self._parse_str_callback(cb)
                yield train_cb
                if self.train_split is not None:
                    yield valid_cb
            else:
                yield name, cb, named_by_user

    def _parse_str_callback(self, cb_supplied_name):
        scoring = get_scorer(cb_supplied_name)
        scoring_name = scoring._score_func.__name__
        assert scoring_name.endswith(
            ('_score', '_error', '_deviance', '_loss'))
        if (scoring_name.endswith('_score') or
                cb_supplied_name.startswith('neg_')):
            lower_is_better = False
        else:
            lower_is_better = True
        train_name = f'train_{cb_supplied_name}'
        valid_name = f'valid_{cb_supplied_name}'
        if self.cropped:
            train_scoring = CroppedTrialEpochScoring(
                cb_supplied_name, lower_is_better, on_train=True, name=train_name
            )
            valid_scoring = CroppedTrialEpochScoring(
                cb_supplied_name, lower_is_better, on_train=False, name=valid_name
            )
        else:
            train_scoring = PostEpochTrainScoring(
                cb_supplied_name, lower_is_better, name=train_name
            )
            valid_scoring = EpochScoring(
                cb_supplied_name, lower_is_better, on_train=False, name=valid_name
            )
        named_by_user = True
        train_valid_callbacks = [
            (train_name, train_scoring, named_by_user),
            (valid_name, valid_scoring, named_by_user)
        ]
        return train_valid_callbacks

    def on_batch_end(self, net, *batch, training=False, **kwargs):
        # If training is false, assume that our loader has indices for this
        # batch
        if not training:
            epoch_cbs = []
            for name, cb in self.callbacks_:
                if isinstance(cb, (CroppedTrialEpochScoring, CroppedTimeSeriesEpochScoring)) and (
                        hasattr(cb, 'window_inds_')) and (not cb.on_train):
                    epoch_cbs.append(cb)
            # for trialwise decoding stuffs it might also be we don't have
            # cropped loader, so no indices there
            if len(epoch_cbs) > 0:
                assert self._last_window_inds_ is not None
                for cb in epoch_cbs:
                    cb.window_inds_.append(self._last_window_inds_)
                self._last_window_inds_ = None

    def predict_with_window_inds_and_ys(self, dataset):
        self.module.eval()
        preds = []
        i_window_in_trials = []
        i_window_stops = []
        window_ys = []
        for X, y, i in self.get_iterator(dataset, drop_index=False):
            i_window_in_trials.append(i[0].cpu().numpy())
            i_window_stops.append(i[2].cpu().numpy())
            with torch.no_grad():
                preds.append(to_numpy(self.module.forward(X.to(self.device))))
            window_ys.append(y.cpu().numpy())
        preds = np.concatenate(preds)
        i_window_in_trials = np.concatenate(i_window_in_trials)
        i_window_stops = np.concatenate(i_window_stops)
        window_ys = np.concatenate(window_ys)
        return dict(
            preds=preds, i_window_in_trials=i_window_in_trials,
            i_window_stops=i_window_stops, window_ys=window_ys)

    # Removes default EpochScoring callback computing 'accuracy' to work
    # properly with cropped decoding.
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

    @abc.abstractmethod
    def _get_n_outputs(self, y):
        pass

    def _set_signal_args(self, X, y):
        # get kwargs from signal:
        signal_kwargs = dict()
        if isinstance(X, np.ndarray):
            if y is None:
                raise ValueError(f"y must be specified if X is a numpy array.")
            self.log.info(f"Using numpy array to find signal-related parameters.")
            Xshape = X.shape
            signal_kwargs["n_times"] = Xshape[-1]
            signal_kwargs["n_chans"] = Xshape[-2]
            signal_kwargs['n_outputs'] = self._get_n_outputs(y)

        elif isinstance(X, torchDataset):
            self.log.info(f"Using Dataset {X!r} to find signal-related parameters.")
            X0 = X[0][0]
            Xshape = X0.shape
            signal_kwargs["n_times"] = Xshape[-1]
            signal_kwargs["n_chans"] = Xshape[-2]
            if (
                    isinstance(X, BaseConcatDataset) and
                    all(ds.targets_from == 'metadata' for ds in X.datasets)
            ):
                y_target = X.get_metadata().target
                signal_kwargs['n_outputs'] = self._get_n_outputs(y_target)
            elif (
                    isinstance(X, WindowsDataset) and
                    X.targets_from == "metadata"
            ):
                y_target = X.windows.metadata.target
                signal_kwargs['n_outputs'] = self._get_n_outputs(y_target)
        else:
            raise ValueError(
                f"X must be a numpy array or a Dataset, "
                f"got {type(X)!r}."
            )

        # kick out missing kwargs:
        module_kwargs = dict()
        all_module_kwargs = inspect.signature(self.module.__init__).parameters.keys()
        for k, v in signal_kwargs.items():
            if v is None:
                continue
            if k in all_module_kwargs:
                module_kwargs[k] = v
            else:
                self.log.warning(
                    f"Module {self.module!r} "
                    f"is missing parameter {k!r}."
                )

        # save kwargs to self:
        self.log.info(
            f"Passing additional parameters {module_kwargs!r} "
            f"to module {self.module!r}.")
        module_kwargs = {f"module__{k}": v for k, v in module_kwargs.items()}
        vars(self).update(module_kwargs)

    def check_data(self, X, y):
        super().check_data(X, y)
        self._set_signal_args(X, y)
