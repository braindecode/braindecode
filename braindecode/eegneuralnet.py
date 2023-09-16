# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#          Pierre Guetschel <pierre.guetschel@gmail.com>
#
# License: BSD (3-clause)


import abc
import logging
import inspect

import mne
import numpy as np
import torch
from skorch import NeuralNet
from sklearn.metrics import get_scorer
from skorch.callbacks import BatchScoring, EpochScoring, EpochTimer, PrintLog
from skorch.utils import noop, to_numpy, train_loss_score, valid_loss_score, is_dataset

from .training.scoring import (CroppedTimeSeriesEpochScoring,
                               CroppedTrialEpochScoring, PostEpochTrainScoring)
from .models.util import models_dict
from .datasets.base import BaseConcatDataset, WindowsDataset

log = logging.getLogger(__name__)


def _get_model(model):
    ''' Returns the corresponding class in case the model passed is a string. '''
    if isinstance(model, str):
        if model in models_dict:
            model = models_dict[model]
        else:
            raise ValueError(f'Unknown model name {model!r}.')
    return model


class _EEGNeuralNet(NeuralNet, abc.ABC):
    signal_args_set_ = False

    @property
    def log(self):
        return log.getChild(self.__class__.__name__)

    def initialize_module(self):
        """Initializes the module.

        A Braindecode model name can also be passed as module argument.

        If the module is already initialized and no parameter was changed, it
        will be left as is.

        """
        kwargs = self.get_params_for('module')
        module = _get_model(self.module)
        module = self.initialized_instance(module, kwargs)
        # pylint: disable=attribute-defined-outside-init
        self.module_ = module
        return self

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
    def _get_n_outputs(self, y, classes):
        pass

    def _set_signal_args(self, X, y, classes):
        is_init = isinstance(self.module, torch.nn.Module)
        if is_init:
            self.log.info(
                "The module passed is already initialized which is not recommended. "
                "Instead, you can pass the module class and its parameters separately.\n"
                "For more details, see "
                "https://skorch.readthedocs.io/en/stable/user/neuralnet.html#module \n"
                "Skipping setting signal-related parameters from data."
            )
            return
        # get kwargs from signal:
        signal_kwargs = dict()
        if isinstance(X, mne.BaseEpochs) or isinstance(X, np.ndarray):
            if y is None:
                raise ValueError("y must be specified if X is a numpy array.")
            signal_kwargs['n_outputs'] = self._get_n_outputs(y, classes)
            if isinstance(X, mne.BaseEpochs):
                self.log.info("Using mne.Epochs to find signal-related parameters.")
                signal_kwargs["n_times"] = len(X.times)
                signal_kwargs["sfreq"] = X.info['sfreq']
                signal_kwargs["chs_info"] = X.info['chs']
            else:
                self.log.info("Using numpy array to find signal-related parameters.")
                signal_kwargs["n_times"] = X.shape[-1]
                signal_kwargs["n_chans"] = X.shape[-2]
        elif is_dataset(X):
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
                signal_kwargs['n_outputs'] = self._get_n_outputs(y_target, classes)
            elif (
                    isinstance(X, WindowsDataset) and
                    X.targets_from == "metadata"
            ):
                y_target = X.windows.metadata.target
                signal_kwargs['n_outputs'] = self._get_n_outputs(y_target, classes)
        else:
            self.log.warning(
                "Can only infer signal shape of numpy arrays or and Datasets, "
                f"got {type(X)!r}."
            )
            return

        # kick out missing kwargs:
        module_kwargs = dict()
        module = _get_model(self.module)
        all_module_kwargs = inspect.signature(module.__init__).parameters.keys()
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
        self.set_params(**module_kwargs)

    def get_dataset(self, X, y=None):
        """Get a dataset that contains the input data and is passed to
        the iterator.

        Override this if you want to initialize your dataset
        differently.

        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:

            * mne.Epochs
            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset

          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.

        y : target data, compatible with skorch.dataset.Dataset
          The same data types as for ``X`` are supported. If your X is
          a Dataset that contains the target, ``y`` may be set to
          None.

        Returns
        -------
        dataset
          The initialized dataset.

        """
        if isinstance(X, mne.BaseEpochs):
            X = X.get_data(units='uV')
        return super().get_dataset(X, y)

    def partial_fit(self, X, y=None, classes=None, **fit_params):
        """Fit the module.

        If the module is initialized, it is not re-initialized, which
        means that this method should be used if you want to continue
        training a model (warm start).
        If possible, signal-related parameters are inferred from the
        data and passed to the module at initialisation.
        Depending on the type of input passed, the following parameters
        are inferred:

          * mne.Epochs: ``n_times``, ``n_chans``, ``n_outputs``, ``chs_info``,
            ``sfreq``, ``input_window_seconds``
          * numpy array: ``n_times``, ``n_chans``, ``n_outputs``
          * WindowsDataset with ``targets_from='metadata'``
            (or BaseConcatDataset of such datasets): ``n_times``, ``n_chans``, ``n_outputs``
          * other Dataset: ``n_times``, ``n_chans``
          * other types: no parameters are inferred.

        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:

            * mne.Epochs
            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset

          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.

        y : target data, compatible with skorch.dataset.Dataset
          The same data types as for ``X`` are supported. If your X is
          a Dataset that contains the target, ``y`` may be set to
          None.

        classes : array, sahpe (n_classes,)
          Solely for sklearn compatibility, currently unused.

        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.

        """
        # this needs to be executed before the net is initialized:
        if not self.signal_args_set_:
            self._set_signal_args(X, y, classes)
            self.signal_args_set_ = True
        return super().partial_fit(X=X, y=y, classes=classes, **fit_params)

    def fit(self, X, y=None, **fit_params):
        """Initialize and fit the module.

        If the module was already initialized, by calling fit, the
        module will be re-initialized (unless ``warm_start`` is True).
        If possible, signal-related parameters are inferred from the
        data and passed to the module at initialisation.
        Depending on the type of input passed, the following parameters
        are inferred:

          * mne.Epochs: ``n_times``, ``n_chans``, ``n_outputs``, ``chs_info``,
            ``sfreq``, ``input_window_seconds``
          * numpy array: ``n_times``, ``n_chans``, ``n_outputs``
          * WindowsDataset with ``targets_from='metadata'``
            (or BaseConcatDataset of such datasets): ``n_times``, ``n_chans``, ``n_outputs``
          * other Dataset: ``n_times``, ``n_chans``
          * other types: no parameters are inferred.

        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:

            * mne.Epochs
            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset

          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.

        y : target data, compatible with skorch.dataset.Dataset
          The same data types as for ``X`` are supported. If your X is
          a Dataset that contains the target, ``y`` may be set to
          None.

        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.
        """
        # this needs to be executed before the net is initialized:
        if not self.signal_args_set_:
            self._set_signal_args(X, y, classes=None)
            self.signal_args_set_ = True
        return super().fit(X=X, y=y, **fit_params)
