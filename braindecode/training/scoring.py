# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Robin Tibor Schirrmeister <robintibor@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Lukas Gemein <l.gemein@gmail.com>
#          Mohammed Fattouh <mo.fattouh@gmail.com>
#          Sarthak Tayal <sarthaktayal2@gmail.com>
#
# License: BSD-3

import warnings
from contextlib import ExitStack, contextmanager
from itertools import tee
from operator import itemgetter
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
        DataLoader overrides, as in :func:`iter_trial_predictions`.

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
    Collects :func:`iter_trial_predictions` and, unlike it, leaves the module in
    eval mode.
    """
    module.eval()
    trials = iter_trial_predictions(
        module,
        dataset,
        return_targets,
        batch_size,
        num_workers,
        loader_kwargs=loader_kwargs,
    )
    if return_targets:
        predictions, targets = zip(*trials, strict=True)
        return _stack_trials(list(predictions)), _stack_trials(list(targets))
    return _stack_trials(list(trials))


def _trial_prediction_loader(dataset, batch_size, num_workers, loader_kwargs):
    """Construct a chronological loader; never inherit unrelated net settings."""
    options = {"batch_size": batch_size, "num_workers": num_workers}
    options.update(loader_kwargs or {})
    unsupported = set(options) - {
        "batch_size",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "worker_init_fn",
        "generator",
    }
    if unsupported:
        raise ValueError(
            "Unsupported cropped prediction loader_kwargs: "
            + ", ".join(sorted(unsupported))
        )
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


def _prepend_first(first, remaining):
    """Prepend one row, releasing it before advancing the remaining source."""
    yield first
    del first
    yield from remaining


def iter_trial_predictions(
    module,
    dataset,
    return_targets=True,
    batch_size=1,
    num_workers=0,
    *,
    loader_kwargs=None,
):
    """Yield cropped predictions, or ``(prediction, target)`` pairs, trial by trial.

    Parameters
    ----------
    module : torch.nn.Module
        Model implementing forward. It runs in eval mode without gradients; the
        mode of every submodule is restored on exhaustion, error or ``close()``.
    dataset : torch.utils.data.Dataset
        Chronological cropped windows with targets and window indices.
    return_targets : bool
        If True, yield ``(prediction, target)`` pairs.
    batch_size : int
        DataLoader batch size.
    num_workers : int
        DataLoader workers.
    loader_kwargs : dict | None
        DataLoader overrides. Only ``batch_size``, ``num_workers``,
        ``pin_memory``, ``persistent_workers``, ``prefetch_factor``,
        ``worker_init_fn`` and ``generator`` are accepted.

    Yields
    ------
    prediction : np.ndarray or tuple of np.ndarray
        Float32 prediction of one trial, ``(n_outputs, n_predictions)``, or a
        pair with its float32 target: the first window's target for scalar and
        vector targets, the overlap-reconstructed target for time targets.

    Notes
    -----
    Memory holds one trial, the loader batches and one window of lookahead, not
    the completed trials; use :func:`iter_trial_prediction_chunks` to bound a
    long trial. Close the iterator when stopping early. See
    :ref:`cropped-inference`.
    """
    return _iter_trial_outputs(
        module, dataset, None, return_targets, batch_size, num_workers, loader_kwargs
    )


class TrialPredictionChunk(NamedTuple):
    """One chunk yielded by :func:`iter_trial_prediction_chunks`.

    Attributes
    ----------
    trial_index : int
        Zero-based trial index.
    prediction_start : int
        Offset of the chunk in the trial's reconstructed predictions, not an
        EEG sample index.
    prediction : np.ndarray
        Float32 array of shape ``(n_outputs, n_predictions_in_chunk)``.
    target : np.ndarray | None
        The trial's scalar or vector target, this chunk of its time target, or
        None when targets are not requested.
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
    """Yield predictions in chunks of at most ``windows_per_chunk`` windows.

    Takes the arguments of :func:`iter_trial_predictions` and has the same
    lifecycle. A chunk never spans two trials and overlap removal carries
    across chunks, so concatenating a trial's chunks along axis 1 gives its
    :func:`iter_trial_predictions` output exactly. Memory depends on the chunk
    size, not on the trial length.

    Parameters
    ----------
    windows_per_chunk : int
        Maximum number of source windows per chunk.
    module, dataset, return_targets, batch_size, num_workers, loader_kwargs
        As in :func:`iter_trial_predictions`.

    Yields
    ------
    chunk : TrialPredictionChunk
    """
    if windows_per_chunk is None:
        raise ValueError("windows_per_chunk must be a positive integer")
    return _iter_trial_outputs(
        module,
        dataset,
        windows_per_chunk,
        return_targets,
        batch_size,
        num_workers,
        loader_kwargs,
    )


def _iter_trial_outputs(
    module,
    dataset,
    windows_per_chunk,
    return_targets,
    batch_size,
    num_workers,
    loader_kwargs,
):
    """Yield whole trials (``windows_per_chunk=None``) or chunk records."""
    loader = _trial_prediction_loader(dataset, batch_size, num_workers, loader_kwargs)
    modes = [(child, child.training) for child in module.modules()]
    # Callbacks run last-in first-out, each one even if an earlier one raises.
    with ExitStack() as cleanup:

        @cleanup.callback
        def restore_modes():
            for child, training in modes:
                child.training = training

        rows = _iter_prediction_windows(module, loader, return_targets)
        cleanup.callback(rows.close)
        module.eval()
        pred_rows = rows
        if return_targets:
            pred_rows, target_rows = tee(rows)
            targets = _iter_chunk_targets(target_rows, windows_per_chunk)
            cleanup.callback(targets.close)
            # An unadvanced tee cursor would keep every row read after it.
            del target_rows
        predictions = _iter_trial_pred_chunks(
            ((row[0], row[2], row[3]) for row in pred_rows), windows_per_chunk
        )
        cleanup.callback(predictions.close)
        del pred_rows
        for trial, start, prediction in predictions:
            target = next(targets) if return_targets else None
            if windows_per_chunk is not None:
                yield TrialPredictionChunk(trial, start, prediction, target)
            elif return_targets:
                yield prediction, target
            else:
                yield prediction
            # Release this output before computing the next one.
            del prediction, target


def _iter_chunk_targets(rows, limit):
    """Yield the target of each prediction chunk.

    Time targets (``ndim >= 2``) are reconstructed like predictions. Scalar and
    vector targets repeat the trial's first-window target in each chunk.
    """
    first = next(rows)
    is_time = first[1].ndim >= 2
    rows = _prepend_first(first, rows)
    del first
    if is_time:
        chunks = _iter_trial_pred_chunks(((r[1], r[2], r[3]) for r in rows), limit)
        yield from map(itemgetter(2), chunks)
        return
    # Same partition as the predictions: a chunk ends after `limit` windows or
    # when the next trial starts.
    first, previous_window, count = None, -1, 0
    for _, target, window, _ in rows:
        if first is None or window != previous_window + 1:
            if count:
                yield first.copy()
            first, count = target, 0
        previous_window = window
        count += 1
        if count == limit:
            yield first.copy()
            count = 0
    if count:
        yield first.copy()


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
