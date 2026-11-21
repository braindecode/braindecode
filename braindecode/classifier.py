# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Robin Schirrmeister <robintibor@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#          Pierre Guetschel <pierre.guetschel@gmail.com>
#          Sarthak Tayal <sarthaktayal2@gmail.com>
#
# License: BSD (3-clause)

import warnings
from functools import partial

from skorch import NeuralNet
from skorch.callbacks import EpochScoring
from skorch.classifier import NeuralNetClassifier
from torch import Tensor
from torch.nn import CrossEntropyLoss

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
        When passing directly a torch module, uninstantiated class should be preferred,
        although instantiated modules will also work.

    cropped: bool (default=False)
        Defines whether torch model passed to this class is cropped or not.
        Currently used for callbacks definition. For input without window
        indices (such as numpy arrays), scoring callbacks treat each example
        as a separate trial and average its temporal predictions. To aggregate
        multiple windows of the same trial, pass a window dataset retaining
        its window indices. Cross-validation splits must keep windows from
        the same trial together.

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

    compile: bool (default=False)
        If True, compile the module and the criterion in place with
        :meth:`torch.nn.Module.compile`, passing the ``compile__*`` options
        (e.g. ``compile__dynamic=True``). Compilation is not guaranteed to be
        faster. The module keeps its Parameters, supplied and shared ones
        included, and its state-dict keys, so ``save_params`` checkpoints load
        into eager models and back.

        Checkpoints saved through skorch's ``torch.compile`` wrapper (with
        ``compile=True`` before braindecode 2.0) have keys that start with
        ``_orig_mod.``, and loading does not strip this prefix. Export
        ``_orig_mod.state_dict()`` from the wrapped module where it was saved,
        or remove only that verified prefix, then load the result strictly.

        A pickled, joblib-loaded or deep-copied estimator runs eagerly, and warm
        start and ``partial_fit`` do not compile again. ``initialize()``
        compiles again but also re-initializes the module, so call
        ``load_params`` after it to continue training. An estimator replaces or
        clears only a compiled call it installed itself. Call
        ``set_params(compile=False)`` on that estimator before reusing its
        module eagerly elsewhere.

    """  # noqa: E501
    __doc__ = update_estimator_docstring(NeuralNetClassifier, doc)

    def __init__(
        self,
        module,
        *args,
        criterion=CrossEntropyLoss,
        cropped=False,
        callbacks=None,
        iterator_train__shuffle=True,
        iterator_train__drop_last=True,
        aggregate_predictions=True,
        **kwargs,
    ):
        self.cropped = cropped
        self.aggregate_predictions = aggregate_predictions
        self._last_window_inds_ = None
        super().__init__(
            module,
            *args,
            criterion=criterion,
            callbacks=callbacks,
            iterator_train__shuffle=iterator_train__shuffle,
            iterator_train__drop_last=iterator_train__drop_last,
            **kwargs,
        )

    def get_iterator(self, dataset, training=False, drop_index=True):
        iterator = super().get_iterator(dataset, training=training)
        if drop_index:
            return ThrowAwayIndexLoader(self, iterator, is_regression=False)
        else:
            return iterator

    def _get_predict_nonlinearity(self):
        nonlinearity = super()._get_predict_nonlinearity()
        if self.predict_nonlinearity == "auto" and len(self._criteria) == 1:
            criterion = getattr(self, self._criteria[0] + "_")
            if isinstance(getattr(criterion, "_orig_mod", criterion), CrossEntropyLoss):
                # CrossEntropyLoss reads the classes on axis 1, also for dense outputs
                # (batch, classes, positions); skorch's softmax uses the last axis.
                return partial(Tensor.softmax, dim=1)
        return nonlinearity

    def predict_proba(self, X):
        """Return the output of the module's forward method as a numpy.

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
        if self.cropped and self.aggregate_predictions and len(y_pred.shape) == 3:
            return y_pred.mean(axis=-1)
        else:
            return y_pred

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        """Return the loss for this batch by calling NeuralNet get_loss.

        A floating ``y_true`` with the shape of ``y_pred`` is read as class
        probabilities; with any other shape, as class indices, cast to int64.

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
        y_true = _class_indices(y_pred, y_true)
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

    def predict_trials(self, X, return_targets=True, *, loader_kwargs=None):
        """Create trialwise predictions from a cropped dataset.

        Parameters
        ----------
        X : braindecode.datasets.BaseConcatDataset
            A braindecode dataset to be predicted.
        return_targets : bool
            If True, additionally returns the trial targets.
        loader_kwargs : dict | None
            Cropped mode only: DataLoader overrides, see
            :func:`braindecode.training.iter_trial_predictions`.

        Returns
        -------
        trial_predictions : np.ndarray | list of np.ndarray
            3-dimensional array (n_trials x n_classes x n_predictions), where
            the number of predictions depend on the chosen window size and the
            receptive field of the network. If trials have different lengths,
            a list with one (n_classes x n_predictions) array per trial.
        trial_targets : np.ndarray | list of np.ndarray
            Ground-truth targets from the dataset. Only returned when
            ``return_targets=True``. An ``np.ndarray`` with a leading trial
            dimension: ``(n_trials,)`` for scalar targets, with further
            dimensions for multi-value or equal-length sequence targets. If
            trial lengths differ, a list with one target array per trial.

        Notes
        -----
        In cropped mode, this calls :func:`braindecode.training.predict_trials`
        on ``module_`` with the estimator's ``batch_size`` and the
        ``num_workers`` of its evaluation loader. Other loader settings, such
        as those of the training loader, are not copied; pass them in
        ``loader_kwargs``.
        """
        if not self.cropped:
            if loader_kwargs is not None:
                raise ValueError("loader_kwargs requires cropped=True")
            warnings.warn(
                "This method was designed to predict trials in cropped mode. "
                "Calling it when cropped is False will give the same result as "
                "'.predict'.",
                UserWarning,
            )
            preds = self.predict(X)
            if return_targets:
                return preds, X.get_metadata()["target"].to_numpy()
            return preds
        self.check_is_fitted()
        return predict_trials(
            module=self.module_,
            dataset=X,
            return_targets=return_targets,
            batch_size=self.batch_size,
            num_workers=self.get_iterator(X, training=False).loader.num_workers,
            loader_kwargs=loader_kwargs,
        )

    @property
    def mode(self):
        return "classification"

    # Only add the 'accuracy' callback if we are not in cropped mode.
    @property
    def _default_callbacks(self):
        callbacks = list(super()._default_callbacks)
        if not self.cropped:
            callbacks.append(
                (
                    "valid_acc",
                    EpochScoring(
                        "accuracy",
                        name="valid_acc",
                        lower_is_better=False,
                    ),
                )
            )
        return callbacks


# Helper of EEGClassifier.get_loss, below the class.
def _class_indices(y_pred, y_true):
    """Cast a float target to int64 unless it has the shape of ``y_pred``.

    Window datasets slice targets out of the float32 signal, and labels read from
    a table can be floats: the loader keeps floats, and the shape tells class
    probabilities (the shape of the predictions) from class indices. A Mixup target
    ``(y_a, y_b, lam)`` casts ``y_a`` and ``y_b``, never the weights ``lam``.
    """
    if isinstance(y_true, (tuple, list)) and len(y_true) == 3:
        y_a, y_b, lam = y_true
        return _class_indices(y_pred, y_a), _class_indices(y_pred, y_b), lam
    if (
        isinstance(y_pred, Tensor)
        and isinstance(y_true, Tensor)
        and y_true.is_floating_point()
        and y_true.shape != y_pred.shape
    ):
        return y_true.long()
    return y_true
