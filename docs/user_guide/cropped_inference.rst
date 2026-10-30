.. _cropped-inference:

#################################################
 Cropped inference: trials, chunks and lifecycle
#################################################

Import :func:`~braindecode.training.iter_trial_predictions`,
:func:`~braindecode.training.iter_trial_prediction_chunks` and
:class:`~braindecode.training.TrialPredictionChunk` from ``braindecode.training``. They
use the same overlap reconstruction as :func:`~braindecode.training.predict_trials` and
preserve chronological order. The classifier and regressor also expose
``iter_trial_predictions`` in cropped mode; use the standalone chunk function with their
fitted ``module_`` for long trials.

***************************************
 Choosing an output and a memory bound
***************************************

- ``predict_trials`` materializes all outputs. Equal-length trials are stacked into a
  new array, temporarily retaining both the per-trial arrays and the full stacked
  output. Unequal-length trials remain a list. This API continues to leave the model in
  eval mode.
- ``iter_trial_predictions`` reconstructs one complete trial at a time. Its working set
  includes one active trial, a window/lookahead, finite iterator buffers, and DataLoader
  batches/prefetch. Completed trials are not accumulated internally. An arbitrarily long
  trial can still require arbitrarily large memory.
- ``iter_trial_prediction_chunks(..., windows_per_chunk=32)`` bounds the number of
  source windows contributing to each emitted chunk. Trial boundaries split chunks;
  overlap state carries across chunk flushes. Its working set includes one active chunk
  plus a window/lookahead, finite iterator buffers and loader batches/prefetch,
  independent of the total trial duration for fixed window, output and loader sizes.
  This is not a byte limit or a bound on dataset/model storage, worker copies, decoder
  intermediates or total process RSS.

Consume and discard results to retain the streaming bound. ``list(iterator)``, storing
every chunk, or concatenating all outputs materializes the result again. A suspended
final/chunk yield can retain the last loop window. Per-trial reconstruction reads the
next trial's first window (and its containing batch) before emitting the preceding
trial; malformed boundary indices are validated before that preceding trial is yielded.
These APIs are not zero-lookahead or causal EEG-clock schedulers.

***********************
 Targets and positions
***********************

Predictions and requested targets are converted to float32. A whole-trial prediction has
shape ``(n_outputs, n_predictions)``. Scalar/vector targets use the **first window's
target** in each trial; the chunk API repeats that target in every chunk. Time targets
(each window target has at least two dimensions) are reconstructed with the same
overlap-removal rule as predictions. With ``return_targets=False``, the trial iterator
yields arrays only, and chunk records have ``target=None``; unused targets are not
converted or accumulated.

``TrialPredictionChunk.trial_index`` is zero-based. ``prediction_start`` is a zero-based
offset in the reconstructed prediction sequence, **not** an EEG sample index, timestamp
or number of source windows. Use the model's receptive-field and dataset timing metadata
for EEG-clock mapping. Concatenating predictions along axis 1 within each trial
reproduces the whole-trial iterator with identical loader settings; time targets
concatenate in the same way. Scalar/vector targets should not be concatenated as time
data.

***************************
 Lifecycle and early close
***************************

The iterators are lazy: construction does not run inference or change model modes. On
first advancement they save each child's training flag and put the model in eval mode.
The model stays in eval mode while the generator is suspended. Exhaustion, errors or
explicit ``close()`` attempt owned loader-worker cleanup and restore the original flags,
including mixed child modes. Do not train, change modes or share the same module with
concurrent inference while an iterator is active. ``torch.no_grad`` surrounds only the
model forward, not the caller's code between yields. Single-window-per-trial warnings
are emitted at exhaustion, rather than before the first forward.

Use ``contextlib.closing`` when stopping early; a loop's ``break`` alone does not close
a retained generator. For an existing fitted model and chronological cropped ``windows``
dataset:

.. code-block:: python

    from contextlib import closing
    from braindecode.training import iter_trial_prediction_chunks

    with closing(
        iter_trial_prediction_chunks(
            model,
            windows,
            windows_per_chunk=32,
            batch_size=16,
            return_targets=False,
        )
    ) as chunks:
        first = next(chunks)
        print(first.trial_index, first.prediction_start, first.prediction.shape)
        # Process first.prediction here; leaving the context also handles early exit.
    # Original model/child training flags are restored here.

A small executable lifecycle example (two non-overlapping windows, one trial):

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
``persistent_workers``, ``prefetch_factor``, ``worker_init_fn`` and ``generator``. These
explicit values override the corresponding function arguments; PyTorch's constraints
still apply (for example, persistent workers require workers). Automatic batching is
required, so ``batch_size=None`` is rejected. Shuffling, samplers, batch samplers,
dropping, custom collation and every other override are rejected to preserve
reconstruction chronology. Cropped channel-position batches are unsupported: disable
``return_ch_pos`` for this path. Loader settings are not automatically copied from an
estimator's training loader.
