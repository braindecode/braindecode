# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Robin Tibor Schirrmeister <robintibor@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Lukas Gemein <l.gemein@gmail.com>
#          Mohammed Fattouh <mo.fattouh@gmail.com>
#          Sarthak Tayal <sarthaktayal2@gmail.com>
#
# License: BSD-3

import warnings
from contextlib import contextmanager
from itertools import tee
from numbers import Integral
from typing import NamedTuple

import numpy as np
import torch
from skorch.callbacks.scoring import EpochScoring
from skorch.dataset import unpack_data
from skorch.utils import to_numpy
from torch.utils.data import DataLoader

from ._trial_aggregation import _iter_trial_pred_chunks, iter_trial_preds


def trial_preds_from_window_preds(preds, i_window_in_trials, i_stop_in_trials):
    """
    Assigning window predictions to trials  while removing duplicate
    predictions.

    Parameters
    ----------
    preds: list of ndarrays (at least 2darrays)
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
        f"{len(preds)}, {len(i_window_in_trials)}, {len(i_stop_in_trials)}"
    )
    # The overlap-removal algorithm lives in the streaming iterator (its single
    # owner); this public function keeps returning a materialized list.
    return list(iter_trial_preds(preds, i_window_in_trials, i_stop_in_trials))


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

    Window indices identify overlapping windows belonging to the same trial.
    Without indices (for example, numpy input), each example is a separate
    trial: average its temporal predictions, never combine different examples.
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

    def on_batch_end(self, net, batch, y_pred, training, **kwargs):
        # Skorch saves the predictions without moving them from GPU
        # https://github.com/skorch-dev/skorch/blob/fe71e3d55a4ae5f5f94ef7bdfc00fca3b3fd267f/skorch/callbacks/scoring.py#L385
        # This can cause memory issues in case of a large number of predictions
        # Therefore here we move them to CPU already
        super().on_batch_end(net, batch, y_pred, training, **kwargs)
        if self.use_caching and training == self.on_train:
            self.y_preds_[-1] = self.y_preds_[-1].cpu()

    def on_epoch_end(self, net, dataset_train, dataset_valid, **kwargs):
        assert self.use_caching
        if not self.crops_to_trials_computed:
            if self.on_train:
                # Prevent that rng state of torch is changed by
                # creation+usage of iterator
                rng_state = torch.random.get_rng_state()
                pred_results = net.predict_with_window_inds_and_ys(dataset_train)
                torch.random.set_rng_state(rng_state)
            else:
                pred_results = {}
                pred_results["i_window_in_trials"] = (
                    np.concatenate([i[0].cpu().numpy() for i in self.window_inds_])
                    if self.window_inds_
                    else None
                )
                pred_results["i_window_stops"] = (
                    np.concatenate([i[2].cpu().numpy() for i in self.window_inds_])
                    if self.window_inds_
                    else None
                )
                pred_results["preds"] = np.concatenate(
                    [y_pred.cpu().numpy() for y_pred in self.y_preds_]
                )
                pred_results["window_ys"] = np.concatenate(
                    [y.cpu().numpy() for y in self.y_trues_]
                )

            if pred_results["i_window_in_trials"] is None:
                trial_ys = pred_results["window_ys"]
                trial_preds = pred_results["preds"]
            else:
                # A new trial starts
                # when the index of the window in trials
                # does not increment by 1
                # Add dummy infinity at start
                window_0_per_trial_mask = (
                    np.diff(pred_results["i_window_in_trials"], prepend=[np.inf]) != 1
                )
                trial_ys = pred_results["window_ys"][window_0_per_trial_mask]
                trial_preds = trial_preds_from_window_preds(
                    pred_results["preds"],
                    pred_results["i_window_in_trials"],
                    pred_results["i_window_stops"],
                )

            # Average across the timesteps of each trial so we have per-trial
            # predictions already, these will be just passed through the forward
            # method of the classifier/regressor to the skorch scoring function.
            # trial_preds is a list, each item is a 2d array classes x time
            y_preds_per_trial = np.array([np.mean(p, axis=1) for p in trial_preds])
            # Move into format expected by skorch (list of torch tensors)
            y_preds_per_trial = [torch.tensor(y_preds_per_trial)]

            # Store the computed trial preds for all Cropped Callbacks
            # that are also on same set
            cbs = net.callbacks_
            epoch_cbs = [
                cb
                for name, cb in cbs
                if isinstance(cb, CroppedTrialEpochScoring)
                and (cb.on_train == self.on_train)
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
                pred_results = net.predict_with_window_inds_and_ys(dataset_train)
                torch.random.set_rng_state(rng_state)
            else:
                pred_results = {}
                pred_results["i_window_in_trials"] = np.concatenate(
                    [i[0].cpu().numpy() for i in self.window_inds_]
                )
                pred_results["i_window_stops"] = np.concatenate(
                    [i[2].cpu().numpy() for i in self.window_inds_]
                )
                pred_results["preds"] = np.concatenate(
                    [y_pred.cpu().numpy() for y_pred in self.y_preds_]
                )
                pred_results["window_ys"] = np.concatenate(
                    [y.cpu().numpy() for y in self.y_trues_]
                )

            num_preds = pred_results["preds"][-1].shape[-1]
            # slice the targets to fit preds shape
            pred_results["window_ys"] = [
                targets[:, -num_preds:] for targets in pred_results["window_ys"]
            ]

            trial_preds = trial_preds_from_window_preds(
                pred_results["preds"],
                pred_results["i_window_in_trials"],
                pred_results["i_window_stops"],
            )

            trial_ys = trial_preds_from_window_preds(
                pred_results["window_ys"],
                pred_results["i_window_in_trials"],
                pred_results["i_window_stops"],
            )

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
                cb
                for name, cb in cbs
                if isinstance(cb, CroppedTimeSeriesEpochScoring)
                and (cb.on_train == self.on_train)
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
            try:
                iterator = net.get_iterator(dataset, training=False)
                y_preds = []
                y_test = []
                for batch in iterator:
                    _, batch_y = unpack_data(batch)
                    yp = net.evaluation_step(batch, training=False)
                    yp = yp.to(device="cpu")
                    y_test.append(self.target_extractor(batch_y))
                    y_preds.append(yp)
                y_test = np.concatenate(y_test)
            finally:
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
            current_score = self._scoring(cached_net, dataset_train, self.y_trues_)
        self._record_score(net.history, current_score)


def predict_trials(
    module,
    dataset,
    return_targets=True,
    batch_size=1,
    num_workers=0,
    *,
    loader_kwargs=None,
):
    """Create trialwise predictions and optionally also return trialwise
    targets from a cropped dataset given a module.

    Parameters
    ----------
    module: torch.nn.Module
        A pytorch model implementing forward.
    dataset: braindecode.datasets.BaseConcatDataset
        A braindecode dataset to be predicted.
    return_targets: bool
        If True, additionally returns the trial targets.
    batch_size: int
        The batch size used to iterate the dataset.
    num_workers: int
        Number of workers used in DataLoader to iterate the dataset.
    loader_kwargs: dict | None
        Explicit chronological DataLoader overrides; see iter_trial_predictions
        for the whitelist and constraints. Defaults are unchanged when omitted.
        This function materializes every trial; equal-length stacking additionally
        allocates the full output.

    Returns
    -------
        trial_predictions: np.ndarray | list of np.ndarray
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
    Uses the streaming reconstruction underlying :func:`iter_trial_predictions`,
    but retains every returned trial. Equal-length stacking additionally allocates
    the full output. Unlike the iterators, this function leaves the model in eval
    mode. Single-window warnings occur at exhaustion. See the
    :ref:`cropped-inference` guide for streaming alternatives and loader settings.
    """
    # Preserve this API's historical policy: leave the module in eval mode.
    module.eval()
    trials = iter_trial_predictions(
        module,
        dataset,
        return_targets,
        batch_size,
        num_workers,
        loader_kwargs=loader_kwargs,
    )
    try:
        if return_targets:
            predictions, targets = zip(*trials, strict=True)
            return _stack_trials(list(predictions)), _stack_trials(list(targets))
        return _stack_trials(list(trials))
    finally:
        trials.close()


def _trial_prediction_loader(dataset, batch_size, num_workers, loader_kwargs):
    """Construct a chronological loader; never inherit unrelated net settings."""
    options = {"batch_size": batch_size, "num_workers": num_workers}
    if loader_kwargs is not None:
        allowed = {
            "batch_size",
            "num_workers",
            "pin_memory",
            "persistent_workers",
            "prefetch_factor",
            "worker_init_fn",
            "generator",
        }
        unsupported = set(loader_kwargs) - allowed
        if unsupported:
            raise ValueError(
                "Unsupported cropped prediction loader_kwargs: "
                + ", ".join(sorted(unsupported))
            )
        options.update(loader_kwargs)
    # Automatic batching is required for the window/metadata shape contract.
    if options["batch_size"] is None:
        raise ValueError("batch_size must be a positive integer, not None")
    return DataLoader(dataset=dataset, shuffle=False, **options)


def _iter_prediction_windows(module, loader, return_targets):
    """Yield rows of one predicted batch at a time; own the loader lifecycle."""
    device = next(module.parameters()).device
    batches = iter(loader)
    multiple_windows = False
    try:
        for batch in batches:
            if len(batch) > 3:
                raise ValueError(
                    "Cropped trial prediction does not support channel positions; "
                    "disable return_ch_pos (set_return_ch_pos(False)) for cropped "
                    "evaluation."
                )
            X, y, ind = batch
            # Keep no_grad local to forward: callers must not inherit it at yield.
            with torch.no_grad():
                predictions = module(X.to(device)).cpu().numpy().astype(np.float32)
            targets = y.cpu().numpy().astype(np.float32) if return_targets else None
            del batch, X, y
            for row, window, stop in zip(
                range(len(predictions)), ind[0], ind[2], strict=True
            ):
                multiple_windows |= bool(window != 0)
                yield (
                    predictions[row],
                    (targets[row] if return_targets else None),
                    int(window),
                    int(stop),
                )
        if not multiple_windows:
            warnings.warn(
                "This function was designed to predict trials from "
                "cropped datasets, which typically have multiple compute "
                "windows per trial. The given dataset has exactly one "
                "window per trial."
            )
    finally:
        # DataLoader has no public close; this also closes persistent workers.
        # Single-process iterators do not expose the shutdown hook.
        shutdown = getattr(batches, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()


def _reconstruct_trial_windows(rows, column):
    """Feed the shared overlap primitive three lazy, lockstep views."""
    values, windows, stops = tee(rows, 3)
    return iter_trial_preds(
        (row[column] for row in values),
        (row[2] for row in windows),
        (row[3] for row in stops),
    )


def _prepend_first(first, remaining):
    """Prepend one row, releasing it before advancing the remaining source."""
    yield first
    del first
    yield from remaining


def _iter_trial_targets(rows):
    """Reconstruct time targets or retain only the first scalar/vector target."""
    first = next(rows)
    is_time = first[1].ndim >= 2
    rows = _prepend_first(first, rows)
    del first
    if is_time:
        targets = _reconstruct_trial_windows(rows, 1)
        # Consuming tee views own their cursors; do not pin their source.
        del rows
        yield from targets
    else:
        target = None
        previous_window = -1
        for _, window_target, window, _ in rows:
            if target is None:
                target = window_target.copy()
            elif window != previous_window + 1:
                yield target
                target = window_target.copy()
            previous_window = window
        yield target


def iter_trial_predictions(
    module,
    dataset,
    return_targets=True,
    batch_size=1,
    num_workers=0,
    *,
    loader_kwargs=None,
):
    """Yield cropped predictions, optionally paired with targets, one trial at a time.

    Parameters
    ----------
    module : torch.nn.Module
        Model implementing forward. Inference uses eval mode and no gradients.
        Original child-module training modes are restored on exhaustion, error,
        or explicit close().
    dataset : torch.utils.data.Dataset
        Chronological cropped windows with targets and window indices.
    return_targets : bool
        Yield (prediction, target) if True, otherwise only predictions.
        Unused targets are not converted or accumulated.
    batch_size : int
        Batch size, default 1.
    num_workers : int
        DataLoader workers, default 0.
    loader_kwargs : dict | None
        Explicit overrides: batch_size, num_workers, pin_memory,
        persistent_workers, prefetch_factor, worker_init_fn, generator.
        PyTorch's loader constraints apply. Ordering, sampling, dropping,
        custom collation and all other options are rejected.

    Yields
    ------
    prediction : np.ndarray or tuple of np.ndarray
        Float32 prediction of shape (n_outputs, n_predictions), or a pair with
        its float32 scalar, vector, or reconstructed time target.

    Notes
    -----
    The stable import path is ``braindecode.training.iter_trial_predictions``.
    Working memory includes one active trial plus a window/lookahead, finite
    iterator buffers and loader batches/prefetch, not all completed trials.
    This is not a total process-RSS bound. One arbitrarily long trial still
    requires arbitrarily large memory. Consume and discard outputs for bounded
    use; collecting them defeats this benefit. The first window of the next
    trial (and its loader batch) is needed before the current trial is yielded.
    A suspended final yield may retain the last loop window. Invalid boundary
    indices are checked before the preceding accumulated trial is yielded.

    Scalar/vector targets use the first window's target in each trial. Time
    targets (window target ndim >= 2) use prediction overlap reconstruction.
    Construction is lazy; model mode changes begin on first advancement.
    Close explicitly on early exit, e.g. with ``contextlib.closing``. While
    suspended the model remains in eval mode; do not train or change its modes
    concurrently. Each child's original flag is restored even if cleanup fails.
    No-grad is local to the forward, not the caller between yields.
    Single-window warnings occur at exhaustion. See :ref:`cropped-inference`
    for an executable example and materialization tradeoffs.
    """
    loader = _trial_prediction_loader(dataset, batch_size, num_workers, loader_kwargs)
    modes = [(child, child.training) for child in module.modules()]
    rows = _iter_prediction_windows(module, loader, return_targets)
    predictions = targets = None
    try:
        module.eval()
        if return_targets:
            pred_rows, target_rows = tee(rows)
            predictions = _reconstruct_trial_windows(pred_rows, 0)
            targets = _iter_trial_targets(target_rows)
            # Both consumers own their cursor; locals would pin tee history.
            del pred_rows, target_rows
            yield from zip(predictions, targets, strict=True)
        else:
            predictions = _reconstruct_trial_windows(rows, 0)
            yield from predictions
    finally:
        try:
            try:
                try:
                    if predictions is not None:
                        predictions.close()
                finally:
                    if targets is not None:
                        targets.close()
            finally:
                rows.close()
        finally:
            for child, training in modes:
                child.training = training


class TrialPredictionChunk(NamedTuple):
    """One prediction segment from :func:`iter_trial_prediction_chunks`.

    Attributes
    ----------
    trial_index : int
        Zero-based trial index in dataset chronology.
    prediction_start : int
        Zero-based offset in reconstructed prediction positions within the
        trial, not an EEG sample index, source-window count or time in seconds.
    prediction : np.ndarray
        Owned float32 array, shape ``(n_outputs, n_predictions_in_chunk)``.
    target : np.ndarray | None
        First-window scalar/vector target repeated per chunk, reconstructed time
        target, or None when targets were not requested.

    Notes
    -----
    Import from ``braindecode.training``. Collecting records retains their
    arrays; the iterator's working-memory bound does not include caller storage.
    """

    trial_index: int
    prediction_start: int
    prediction: np.ndarray
    target: np.ndarray | None


def iter_trial_prediction_chunks(
    module,
    dataset,
    *,
    windows_per_chunk,
    return_targets=True,
    batch_size=1,
    num_workers=0,
    loader_kwargs=None,
):
    """Yield bounded cropped prediction chunks without materializing a trial.

    Parameters
    ----------
    module : torch.nn.Module
        Model implementing forward; eval/no_grad and cleanup match
        :func:`iter_trial_predictions`.
    dataset : torch.utils.data.Dataset
        Chronologically ordered cropped windows with targets and window indices.
    windows_per_chunk : int
        Positive maximum number of source windows per chunk. Overlap state is
        carried across chunks, never reset. Trial boundaries always split chunks.
    return_targets : bool
        Include float32 targets, otherwise each record's target is None.
        Scalar/vector targets use each trial's first window and repeat per chunk;
        time targets are rebuilt
        with the same overlap primitive as predictions.
    batch_size : int
        Loader batch size (independent of the chunk size), default 1.
    num_workers : int
        Loader workers, default 0.
    loader_kwargs : dict | None
        Same explicit chronological overrides as :func:`iter_trial_predictions`.

    Yields
    ------
    chunk : TrialPredictionChunk
        Zero-based trial index, zero-based prediction offset within that trial,
        owned float32 prediction array, and optional target. The offset counts
        reconstructed prediction positions, NOT EEG sample indices or seconds.
        Concatenating prediction chunks within each trial along axis 1 is
        bit-identical to iter_trial_predictions with the same loader settings.
        For time targets the same holds when concatenating the target chunks.

    Notes
    -----
    The stable import path is ``braindecode.training.iter_trial_prediction_chunks``.
    Working memory includes one active chunk of the configured source windows,
    one window/lookahead, finite iterator buffers and loader batches/prefetch,
    independent of trial duration for fixed window/output/loader sizes. A
    suspended chunk/final yield may retain the last loop window. This is not a
    byte limit or total process-RSS bound. Collecting chunks defeats the bound.
    Boundary validation precedes yielding a preceding accumulated trial.

    Construction is lazy; eval mode begins on first advancement. As with the
    per-trial API, close explicitly on early exit (e.g. ``contextlib.closing``),
    and do not train or change modes concurrently while suspended. Cleanup is
    attempted and each child's saved mode restored on exhaustion, close or error,
    including cleanup errors. No-grad surrounds only forward. Single-window
    warnings occur at exhaustion. See :ref:`cropped-inference` for a short
    early-close example and the complete loader whitelist.
    """
    if (
        isinstance(windows_per_chunk, bool)
        or not isinstance(windows_per_chunk, Integral)
        or windows_per_chunk <= 0
    ):
        raise ValueError("windows_per_chunk must be a positive integer")
    loader = _trial_prediction_loader(dataset, batch_size, num_workers, loader_kwargs)
    modes = [(child, child.training) for child in module.modules()]
    rows = _iter_prediction_windows(module, loader, return_targets)
    predictions = targets = None
    try:
        module.eval()
        if return_targets:
            pred_rows, target_rows = tee(rows)
            targets = _iter_chunk_targets(target_rows, windows_per_chunk)
        else:
            pred_rows = rows
        values, windows, stops = tee(pred_rows, 3)
        # Retaining an unadvanced tee cursor pins its entire history, even
        # when all three consuming views stay in lockstep.
        del pred_rows
        if return_targets:
            del target_rows
        predictions = _iter_trial_pred_chunks(
            (r[0] for r in values),
            (r[2] for r in windows),
            (r[3] for r in stops),
            windows_per_chunk,
        )
        if return_targets:
            for (trial, start, prediction), (target_trial, target_start, target) in zip(
                predictions, targets, strict=True
            ):
                assert trial == target_trial
                assert target_start is None or start == target_start
                yield TrialPredictionChunk(trial, start, prediction, target)
        else:
            for trial, start, prediction in predictions:
                yield TrialPredictionChunk(trial, start, prediction, None)
    finally:
        try:
            try:
                try:
                    if predictions is not None:
                        predictions.close()
                finally:
                    if targets is not None:
                        targets.close()
            finally:
                rows.close()
        finally:
            for child, training in modes:
                child.training = training


def _iter_chunk_targets(rows, limit):
    """Use the overlap owner for time targets, repeat first other trial target."""
    first = next(rows)
    is_time = first[1].ndim >= 2

    rows = _prepend_first(first, rows)
    del first
    values, windows, stops = tee(rows, 3)
    del rows
    if is_time:
        yield from _iter_trial_pred_chunks(
            (r[1] for r in values),
            (r[2] for r in windows),
            (r[3] for r in stops),
            limit,
        )
    else:
        # Tiny dummy arrays follow the exact same window partition, retaining
        # no prediction/time-target buffers in this branch.
        # Copy the first target lazily per trial, then align its emission with
        # the prediction partition. One lookahead must not overwrite the
        # previous trial's target before its final partial chunk is delivered.
        trial_targets = {}

        def dummy_values():
            trial = 0
            previous_window = -1
            for _, window_target, window, _ in values:
                if window != previous_window + 1:
                    trial += 1
                if trial not in trial_targets:
                    trial_targets[trial] = window_target.copy()
                previous_window = window
                yield np.empty((1, 0), dtype=np.float32)

        previous_trial = 0
        for trial, _, _dummy in _iter_trial_pred_chunks(
            dummy_values(), (r[2] for r in windows), (r[3] for r in stops), limit
        ):
            if trial != previous_trial:
                del trial_targets[previous_trial]
            previous_trial = trial
            yield trial, None, trial_targets[trial].copy()


def _stack_trials(per_trial):
    """Stack equal-length trials into one array; keep variable-length trials a list."""
    if len({trial.shape for trial in per_trial}) <= 1:
        return np.array(per_trial)
    return per_trial


def f1_event(pred_events, gt_events, iou_threshold: float = 0.5) -> float:
    """COCO-style greedy event F1: TP requires IoU > threshold AND class match.

    Inputs are lists of ``(start, end, class[, conf])`` tuples with class in
    ``1..n_classes-1`` (CLASS-0 CONTRACT: callers never pass class 0 -- decoded
    detections skip ``argmax == 0`` and the target builder never stores class 0).
    """
    matched_gt = set()
    tp = 0
    preds = sorted(
        pred_events, key=lambda ev: ev[3] if len(ev) > 3 else 1.0, reverse=True
    )
    for ev in preds:
        ps, pe, pc = ev[0], ev[1], int(ev[2])
        best_iou, best_j = 0.0, -1
        for j, (gs, ge, gc) in enumerate(gt_events):
            if j in matched_gt or int(gc) != pc:
                continue
            inter = max(0.0, min(pe, ge) - max(ps, gs))
            union = (pe - ps) + (ge - gs) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou > iou_threshold and best_j >= 0:
            matched_gt.add(best_j)
            tp += 1
    fp = len(pred_events) - tp
    fn = len(gt_events) - tp
    if tp == 0:
        # Nothing to detect and nothing predicted is trivially perfect (F1=1.0);
        # 0.0 only when one side has events the other does not.
        if not pred_events and not gt_events:
            return 1.0
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)
