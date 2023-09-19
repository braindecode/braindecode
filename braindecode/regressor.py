# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Robin Schirrmeister <robintibor@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#          Pierre Guetschel <pierre.guetschel@gmail.com>
#
# License: BSD (3-clause)

import warnings

import numpy as np
from skorch.regressor import NeuralNetRegressor

from .training.scoring import predict_trials
from .eegneuralnet import _EEGNeuralNet
from .util import ThrowAwayIndexLoader, update_estimator_docstring


class EEGRegressor(_EEGNeuralNet, NeuralNetRegressor):
    doc = """Regressor that calls loss function directly.

    Parameters
    ----------
    module: str or torch Module (class or instance)
        Either the name of one of the braindecode models (see
        :obj:`braindecode.models.util.models_dict`) or directly a PyTorch module.
        When passing directly a torch module, uninstantiated class should be prefered,
        although instantiated modules will also work.

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

    iterator_train__shuffle: bool (default=True)
        Defines whether train dataset will be shuffled. As skorch does not
        shuffle the train dataset by default this one overwrites this option.

    aggregate_predictions: bool (default=True)
        Whether to average cropped predictions to obtain window predictions. Used only in the
        cropped mode.

    """  # noqa: E501
    __doc__ = update_estimator_docstring(NeuralNetRegressor, doc)

    def __init__(self, module, *args, cropped=False, callbacks=None,
                 iterator_train__shuffle=True,
                 iterator_train__drop_last=True,
                 aggregate_predictions=True, **kwargs):
        self.cropped = cropped
        self.aggregate_predictions = aggregate_predictions
        self._last_window_inds_ = None
        super().__init__(module,
                         *args,
                         callbacks=callbacks,
                         iterator_train__shuffle=iterator_train__shuffle,
                         iterator_train__drop_last=iterator_train__drop_last,
                         **kwargs)

    def get_iterator(self, dataset, training=False, drop_index=True):
        iterator = super().get_iterator(dataset, training=training)
        if drop_index:
            return ThrowAwayIndexLoader(self, iterator, is_regression=True)
        else:
            return iterator

    def predict_proba(self, X):
        """Return the output of the module's forward method as a numpy
        array. In case of cropped decoding returns averaged values for
        each trial.

        If the module's forward method returns multiple outputs as a
        tuple, it is assumed that the first output contains the
        relevant information and the other values are ignored.
        If all values are relevant or module's output for each crop
        is needed, consider using :func:`~skorch.NeuralNet.forward`
        instead.

        Parameters
        ----------
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

        Warnings
        --------
        Regressors predict regression targets, so output of this method
        can't be interpreted as probabilities. We advise you to use
        `predict` method instead of `predict_proba`.

        Returns
        -------
        y_proba : numpy ndarray

        """
        y_pred = super().predict_proba(X)
        # Normally, we have to average the predictions across crops/timesteps
        # to get one prediction per window/trial
        # Predictions may be already averaged in CroppedTrialEpochScoring (y_pred.shape==2).
        # However, when predictions are computed outside of CroppedTrialEpochScoring
        # we have to average predictions, hence the check if len(y_pred.shape) == 3
        if self.cropped and self.aggregate_predictions and len(y_pred.shape) == 3:
            return y_pred.mean(-1)
        else:
            return y_pred

    def predict_trials(self, X, return_targets=True):
        """Create trialwise predictions and optionally also return trialwise
        labels from cropped dataset.

        Parameters
        ----------
        X: braindecode.datasets.BaseConcatDataset
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
        if not self.cropped:
            warnings.warn(
                "This method was designed to predict trials in cropped mode. "
                "Calling it when cropped is False will give the same result as "
                "'.predict'.", UserWarning)
            preds = self.predict(X)
            if return_targets:
                return preds, np.concatenate([X[i][1] for i in range(len(X))])
            return preds
        return predict_trials(
            module=self.module,
            dataset=X,
            return_targets=return_targets,
            batch_size=self.batch_size,
            num_workers=self.get_iterator(X, training=False).loader.num_workers,
        )

    def fit(self, X, y=None, **kwargs):
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
        if y is not None:
            if y.ndim == 1:
                y = np.array(y).reshape(-1, 1)
        super().fit(X=X, y=y, **kwargs)

    def _get_n_outputs(self, y, classes):
        if y is None:
            return None
        if y.ndim == 1:
            return 1
        else:
            return y.shape[-1]
