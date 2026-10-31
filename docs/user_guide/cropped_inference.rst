.. _cropped-inference:

#################################################
 Cropped inference: trials, chunks and lifecycle
#################################################

:func:`~braindecode.training.predict_trials` returns every trial at once.
:func:`~braindecode.training.iter_trial_predictions` yields one trial at a time and
:func:`~braindecode.training.iter_trial_prediction_chunks` yields
:class:`~braindecode.training.TrialPredictionChunk` records of at most
``windows_per_chunk`` windows. All three use the same chronological overlap
reconstruction. For a fitted classifier or regressor, pass its ``module_`` to the
iterators.

********
 Memory
********

``predict_trials`` keeps all outputs. ``iter_trial_predictions`` keeps one trial, so its
memory still grows with the trial length. ``iter_trial_prediction_chunks`` keeps one
chunk, so its memory does not depend on the trial length. Both iterators also hold the
loader batches and one window of lookahead. The bound only holds if each output is
discarded after use: ``list(iterator)`` materializes everything again.

***********************
 Targets and positions
***********************

Predictions and targets are float32; a trial prediction has shape ``(n_outputs,
n_predictions)``. A scalar or vector target is the trial's **first-window target**,
repeated in every chunk. A time target (at least two dimensions per window) is
reconstructed like the predictions. With ``return_targets=False`` the trial iterator
yields arrays and chunk targets are ``None``.

``prediction_start`` is the zero-based offset of a chunk in the trial's reconstructed
predictions, **not** an EEG sample index. Concatenating a trial's chunks along axis 1
reproduces the trial iterator's output exactly, for predictions and time targets.

***************************
 Lifecycle and early close
***************************

The iterators are lazy. On first advancement they record each submodule's training flag
and switch the model to eval mode; exhaustion, an error or ``close()`` shuts down loader
workers and restores the flags. ``torch.no_grad`` covers only the forward. Do not train
the model while an iterator is suspended. A ``break`` alone does not close a generator,
so wrap early exits in ``contextlib.closing``:

.. doctest::

    >>> import torch
    >>> from contextlib import closing
    >>> from braindecode.training import iter_trial_prediction_chunks
    >>> model = torch.nn.Conv1d(1, 1, kernel_size=1, bias=False)
    >>> _ = model.train()
    >>> windows = [(torch.ones(1, 4), 7, (i, 4 * i, 4 * (i + 1))) for i in range(2)]
    >>> with closing(
    ...     iter_trial_prediction_chunks(
    ...         model,
    ...         windows,
    ...         windows_per_chunk=1,
    ...         return_targets=False,
    ...     )
    ... ) as chunks:
    ...     chunk = next(chunks)
    ...     print(
    ...         chunk.trial_index,
    ...         chunk.prediction_start,
    ...         chunk.prediction.shape,
    ...         chunk.target,
    ...         model.training,
    ...     )
    ...
    0 0 (1, 4) None False
    >>> model.training
    True

***********************************
 Chronological DataLoader controls
***********************************

``loader_kwargs`` accepts only ``batch_size``, ``num_workers``, ``pin_memory``,
``persistent_workers``, ``prefetch_factor``, ``worker_init_fn`` and ``generator``, and
overrides the matching function arguments. Shuffling, samplers, ``drop_last``, custom
collation, ``batch_size=None`` and any other key are rejected, because reconstruction
relies on chronological batches. Batches with channel positions are not supported:
disable ``return_ch_pos``. Loader settings are not copied from an estimator's training
loader.
