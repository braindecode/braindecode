# Real-window distributed evaluation recipe (CPU/Gloo)

Run `python -m benchmarks.acceleration.bench_distributed_windows --output /work/results`
inside a compute Job. This is an opt-in benchmark helper, not distributed support
in EEGClassifier or AccelerateMixin. Only CPU, trusted local processes, preloaded
MNE WindowsDataset, scalar classification targets, fixed window size and
batch-independent eval-mode models are qualified here. No training/DDP wrapper,
GPU/HPU, lazy disk, multi-node network or stochastic transforms are qualified.

## Identity and aggregation

Wrap the actual WindowsDataset with IdentifiedWindows to retain the **global
dataset row index** next to signal, target and all three crop indices. Trial-local
window indices repeat and cannot identify a record. Use an unpadded shuffled
rank-strided permutation (`seed=37`) and `drop_last=False`. Every real window is
forwarded exactly once, including partial batches; ranks may have no windows.
Do not put collectives or DDP forward hooks in the unequal local forward loop.

Gather predictions and `(target, window, start, stop)` jointly with the baseline
`gather_ordered_predictions`, `known_padding=0`, and the complete `expected_ids`.
Its exact duplicate and missing-record checks remain unchanged. Only then feed
chronologically ordered windows to the real `iter_trial_preds`, including its
bounded `windows_per_chunk` mode. Metrics are computed globally on reconstructed
trials, not averaged across ranks. Identity order is dataset chronology; arbitrary
external IDs require an explicit ID-to-chronology map before reconstruction.

The recipe gathers the full window-prediction tensor to every rank. Bounded
aggregation chunks do **not** bound this gather's memory; this is a correctness
recipe, not a memory-scalable all-trial streaming service. Replicated dataset
construction also consumes per-rank RAM. Production implementations can shard
complete trials or use an explicitly ordered coordinator stream, but need fresh
qualification before making either claim.

## Floating outputs and estimator boundary

Metadata, identities, targets, labels, balanced accuracy and elementwise fixture
outputs are exact. Float32 convolution output comparisons allow `rtol=1e-5,
atol=1e-6`: kernels may choose a different reduction path for partial/batch shapes.
This is a narrow output tolerance, not permission to alter labels or metrics.
NaN/Inf predictions and near-tied labels require a separate policy; this fixture
uses finite predictions and requires exact global labels/metrics.

The chronological oracle uses actual public `iter_trial_predictions` and an
initialized `EEGClassifier(..., cropped=True).predict_trials`. The distributed
helper intentionally bypasses estimator index-discarding iteration. It does not
make estimator prediction itself distributed, and cannot be supplied as a
shuffled sampler to the chronological trial API. Model weights are deterministically
recreated on each rank; production checkpoints must be identical on all ranks.

## Timing interpretation

Four reserved CPU cores, one thread per rank; 1/2/4 ranks use 1/2/4 compute threads
respectively. This is resource scaling, not constant-resource optimization.
53 uneven trials, 32 channels, 512-sample windows, overlapping crops, float32 dense
Conv1d/Tanh/Conv1d, batch size 32. Five alternating fresh-process pairs per size;
warmup stops at three evaluations within 5% or ten iterations. `evaluation_s`
includes forward, real loader, gather and trial aggregation; `including_startup_s`
includes interpreter/rank spawn, imports, per-rank dataset/model construction,
warmup, evaluation and process teardown. Thus it is the actual one-shot recipe
wall time, explicitly including warmup (not a startup-subtracted estimate).
Efficiency is paired single-rank time / multi-rank time / ranks. No distributed
speedup is asserted unless the prespecified report gate passes.
