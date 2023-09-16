# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Robin Schirrmeister <robintibor@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#          Pierre Guetschel <pierre.guetschel@gmail.com>
#
# License: BSD (3-clause)

import warnings

import numpy as np
from skorch import NeuralNet
from skorch.classifier import NeuralNetClassifier

from .eegneuralnet import _EEGNeuralNet
from .training.scoring import predict_trials
from .util import ThrowAwayIndexLoader, update_estimator_docstring


class EEGClassifier(_EEGNeuralNet, NeuralNetClassifier):
    doc = """Classifier that does not assume softmax activation.
    Calls loss function directly without applying log or anything.

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
    __doc__ = update_estimator_docstring(NeuralNetClassifier, doc)

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
            return ThrowAwayIndexLoader(self, iterator, is_regression=False)
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
        if self.cropped and self.aggregate_predictions and len(
                y_pred.shape) == 3:
            return y_pred.mean(axis=-1)
        else:
            return y_pred

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

        Returns
        -------
        loss : float
            The loss value.
        """
        return NeuralNet.get_loss(self, y_pred, y_true, *args, **kwargs)

    def predict(self, X):
        """Return class labels for samples in X.

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

        Returns
        -------
        y_pred : numpy ndarray

        """
        return self.predict_proba(X).argmax(1)

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
                return preds, X.get_metadata()['target'].to_numpy()
            return preds
        return predict_trials(
            module=self.module,
            dataset=X,
            return_targets=return_targets,
            batch_size=self.batch_size,
            num_workers=self.get_iterator(X,
                                          training=False).loader.num_workers,
        )

    def _get_n_outputs(self, y, classes):
        classes_y = np.unique(y)
        if classes is not None:
            assert set(classes_y) <= set(classes)
        else:
            classes = classes_y
        return len(classes)
