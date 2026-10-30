# Performance benchmarks

These scripts measure performance, not decoding accuracy. They do not download
EEG data. Run from the repository root in an environment with Braindecode's
normal dependencies installed.

## Fixed-shape model inference

```sh
python -m benchmarks.bench_model_inference --model EEGNet --device cpu
python -m benchmarks.bench_model_inference --model EEGConformer --device cuda
python -m benchmarks.bench_model_inference --model EEGConformer --device cuda --compile
python -m benchmarks.bench_model_inference --model EEGConformer --device cuda --precision bfloat16
```

Models: EEGNet, ShallowFBCSPNet and EEGConformer. The default input is 16 windows
of 22 channels × 1,000 samples at 250 Hz; four output classes. Use `--help` for
shape, thread, warmup and timing options. Unsupported configurations fail rather
than silently falling back. CPU float16 and MPS bfloat16 are intentionally not
covered. Compile and autocast availability still depend on the installed PyTorch
and hardware; an exposed option is not a claim that every combination is faster
or supported.

JSON output includes configuration and software/backend metadata, first candidate
execution time, warm steady-state median and interquartile range, windows/s,
numerical error, and CUDA peak allocated memory (null on other devices).
PyTorch's `torch.utils.benchmark.Timer` supplies repeated block measurements;
CUDA and MPS are explicitly synchronized after each forward. This is synchronized
batch latency/throughput, not overlapped multi-request serving throughput. The IQR
is the dispersion of block-average timings, **not** per-request p95 latency.

The first timed call happens **after** an eager FP32 reference forward. It is not
full application startup, a cold-device measurement, or necessarily a cold
compiler-cache measurement. Model construction, imports, input creation,
`torch.compile` wrapper construction and device transfer are excluded. Compilation
triggered by the first forward is included; cache reuse is possible. Use a fresh
process and record compiler-cache conditions when comparing startup costs.

Every candidate is checked against eager FP32 outputs for the same random weights
and input. AMP uses looser, explicitly reported tolerances. This is a smoke check,
not a guarantee about all inputs, gradient fidelity, or real task accuracy. Model
forward methods may constrain weights; the reference runs before timing, matching
steady-state evaluation rather than untouched initialization.

### Interpreting comparisons

- Keep model, batch shape, threads, precision, versions, and backend settings fixed.
  Run on otherwise idle hardware, repeat full invocations, and record the source
  commit and hardware model alongside the JSON report.
- Do not average across different devices or advertise a universal speedup.
- Compare precision changes and compilation independently before combining them.
- CUDA memory is the peak total live tensor allocation during the timed phase,
  including model/input/reference tensors; it excludes compile-time peak memory,
  reserved allocator memory, and other processes.
- A shorter warmed-up batch time may not repay compilation in short EEG fits.
- These results exclude data loading, preprocessing, augmentation, host-to-device
  transfer, training/backward/optimizer work, validation and skorch callbacks.
  Profile and benchmark those separately before changing training defaults.
- No wall-clock performance thresholds run in ordinary CI. The smoke tests verify
  that the benchmark executes and reports meaningful measurements, not speed.

## Augmentation loading

```sh
python -m benchmarks.bench_augmentation_growth
```

The existing augmentation benchmark measures expansion scaling and DataLoader
worker scaling, separately from model inference. Worker-count improvements in this
synthetic benchmark do not establish improvements for disk-backed EEG windows.

## Shared acceleration experiment contract (version 1)

`benchmarks.acceleration._common` is an opt-in utility for new experiments.
**The model-inference CLI above keeps its original interface and JSON format.**
It is not silently migrated to this envelope. Review this schema before merging
consumers; infrastructure alone does not qualify any acceleration workstream.

### Timing interface and boundaries

```python
from pathlib import Path
import torch
from benchmarks.acceleration._common import measure_step, write_result

# run() contains exactly the work being compared, with caller-owned policies.
measurement = measure_step(run, device=torch.device("cpu"), warmup=5, min_run_time=1.0)
# Check separately, outside measure_step; no output conversion/check is timed.
# Keep outputs explicitly in caller-owned state if needed, or run another check.
# envelope = {... documented fields below ..., **measurement}
# write_result(envelope, Path("external-research/case.json"))
```

`measure_step(run: Callable[[], object], *, device: torch.device, warmup: int = 5,
min_run_time: float = 1.0) -> dict` returns two sections, `timing` and `memory`.
It does not retain/serialize outputs, select precision, disable gradients, seed
RNGs, choose workers or change the caller's thread count. Use caller-owned
contexts for autograd/autocast, and record them. The timer may invoke the callable
many times during calibration as well as measured blocks; use repeatable work.
Output allocation/destruction and per-call accelerator synchronization are part
of steady-state work. The first output's destruction is outside its clock.

The first call is synchronized wall time; explicit warmup follows it. The
first call and explicit warmup keep the caller's cyclic-GC policy. Native
Timer/timeit steady-state blocks disable cyclic GC (ordinary reference-count
cleanup still runs); do not interpret those blocks as a GC-inclusive service
latency measurement. Native
`torch.utils.benchmark.Timer.blocked_autorange` provides warmed median/IQR and
raw total block durations. `calls_per_block` divides each raw block into a
per-call block average; median/IQR describe these averages, **not** request-tail
latency. Calibration is additional warm work, not included in `warmup` counts.
CPU uses `perf_counter` explicitly (including NumPy/MNE CPU functions), without
implicitly synchronizing CUDA. CUDA synchronizes the selected device and MPS
synchronizes MPS after each call. This is serialized latency, not overlapped
throughput. Work on another device/stream must be synchronized by the caller.

`memory.peak_cuda_allocated_bytes` is the total allocated CUDA peak, including
pre-existing live tensors, across first call, warmup, calibration and blocks.
The utility resets that selected device's peak counter before the first call;
reserved/driver/non-PyTorch allocations and other processes are excluded. It is
null for CPU/MPS, **not zero**. This scope differs from the legacy CLI's peak.

Unavailable CUDA/MPS, invalid CUDA indices, and unqualified device types such as
HPU raise `NotImplementedError` before running the callable: callers serialize
these as `status="unsupported"`, never as passed CPU fallback. Other callable
errors propagate; callers report `failed`. Mocks only check dispatch; they do not
qualify native CUDA/HPU. A real measurement is not a performance claim by itself.

There is no automatic clean-cache label. Record all prior reference, warmup,
initialization and cache conditions in `config.cache_policy` and `notes`. First
call does not mean cold startup. Imports, construction and transfers are excluded
unless they occur inside `run`.

### Strict JSON envelope

`write_result(result: dict, path: pathlib.Path) -> None` validates before creating
any file, then writes UTF-8 strict JSON to a unique sibling temporary and replaces
`path` atomically. Parent directories must already exist. Invalid data leaves an
existing completed file unchanged. On I/O failure only this writer's temporary
is cleaned; no shared glob cleanup. This is atomic visibility, not a promise of
power-loss durability of directory metadata. JSON-native types only: nonfinite
floats, non-string keys, tuples, tensors and NumPy scalars must be handled by the
caller **outside timing**. Missing/extra fixed fields are rejected.

Required fields (all times in seconds):

- `schema_version`: integer `1`; `case_id`: nonempty stable identifier;
  `status`: `ok`, `unsupported`, or `failed`; `implementation`: `reference`,
  `candidate`, or an exact backend name (nonempty string, not normalized).
- `config`: exactly `inputs`, `seed`, `threads`, `workers`, `cache_policy`,
  `autograd`, `precision`, `parameters`. Threads are positive integers; workers
  nonnegative; seed is a nonnegative integer or null (no RNG/not controlled;
  explain which). Policy strings must be explicit; use `not_applicable` when
  appropriate. `parameters` is a JSON object for case-specific whitelisted
  configuration such as filter settings or compiler mode, never an environment
  dump. `inputs` lists `{name, shape, dtype, strides, stride_unit}` for every
  relevant array/tensor. Shapes are nonnegative integer lists (scalar `[]`),
  strides signed integer lists of matching rank; `stride_unit` is `elements`
  (PyTorch) or `bytes` (NumPy). Record actual inputs, not guessed contiguous
  strides. For MNE, describe underlying data arrays and put sampling frequency,
  channel picks and relevant preprocessing options in `parameters`. `inputs=[]`
  means genuinely no array input, not unknown. For failed/unsupported cases,
  `inputs=null` means unavailable before execution; explain in notes.
- `environment`: exactly `python`, `platform`, `dependencies`, `device`,
  `device_name`. Supply Python/platform strings, explicit dependency-name to
  exact-version mapping (include torch/braindecode and numpy/mne when relevant;
  null version means absent/unavailable, explain), selected device identifier,
  and actual hardware identity. `device_name=null` is allowed only for non-ok
  cases when discovery was unavailable. Never invent hardware for unsupported
  accelerators; record the requested identifier. No whole `os.environ`, secrets,
  arbitrary device dumps or automatic discovery are collected by this utility.
- `source`: exactly `commit` (full lowercase 40- or 64-hex commit), `dirty`
  (boolean), `dirty_source_hashes` (relative source path to lowercase SHA256).
  Clean trees have `{}`; dirty trees require hashes covering every relevant
  modified/untracked source. For a deleted path use SHA256 of empty bytes and
  identify the deletion in notes. Include external experiment scripts when
  relevant; separately identify other repositories in notes. The writer checks
  the shape of provenance, not its truth/completeness. In particular, relative
  path spelling is a caller requirement, not enforced path validation: the
  current writer also accepts absolute/parent-relative keys. The caller owns
  collection and must check path shape before claiming portable provenance.
- `correctness`: exactly `status` (`passed`, `failed`, `not_checked`), `method`
  (explicit nonempty comparison/tolerance description, or why not checked),
  `metrics` (JSON object for errors/tolerances). `ok` requires `passed`;
  correctness `failed` requires envelope `failed`. A skipped check is not success.
- `timing`: exactly `first_call_s`, `median_s`, `iqr_s`, `raw_block_s`,
  `measurement_blocks`, `calls_per_block`, `warmup`, `min_run_time_s`. Positive
  finite durations except nonnegative IQR, positive block/call counts, and
  nonnegative integer warmup. Raw total block list length must match block count.
  For failed/unsupported cases use null when not measured; completed timings may
  remain diagnostic after a failed correctness check, never evidence of speedup.
- `memory`: exactly `peak_cuda_allocated_bytes`, nonnegative integer for successful
  CUDA timing; otherwise null when not measured/not applicable.
- `notes`: string list; non-ok cases require an explanation. Include missing
  metadata, prior work, scope exclusions and uncertainty rather than fake values.

No `speedup` key is accepted anywhere in a v1 invocation record, including its
extension objects. Comparisons belong in separate external evidence, not a
single invocation; failed correctness never supplies a valid speedup. The writer
validates `median_s` and `iqr_s` against `raw_block_s / calls_per_block`, using
native Measurement's float64 linear-interpolation quantiles: median = Q(0.5),
IQR = Q(0.75) - Q(0.25). For sorted samples, Q(q) interpolates at index
`(n - 1) * q`; a single block has zero IQR. Each supplied summary must match via
`math.isclose(rel_tol=1e-9, abs_tol=1e-15)` (absolute tolerance in seconds), allowing
floating-point roundoff, not statistical uncertainty. Inconsistency is rejected
before creating or replacing a result; supplied values are not silently rewritten.
Use the actual `measure_step` output. The writer does not calculate winners or
attest to caller-provided metadata.

For **every retained performance claim**, run at least **5 independent paired
reference/candidate invocations**, holding inputs/policies/software/hardware
fixed. Retain raw blocks and the median/spread across pairs, with correctness
checks and source provenance for both. Explain pairing/order/cache policy; do not
select a winner from one invocation. Store profiler traces separately from timing
runs and keep all generated JSON/traces/reports outside git in the external
research directory. No wall-clock speed threshold belongs in ordinary CI.

## Compiled training and exported execution qualification

```sh
python -m benchmarks.acceleration.bench_execution --model EEGNet --mode compiled \
    --output external-research/eegnet-compiled.json
python -m benchmarks.acceleration.bench_execution --model EEGNet --mode exported \
    --dynamic-batch --output external-research/eegnet-exported.json
```

`bench_execution` writes one version-1 record per invocation for one of three
small fixtures (EEGNet, a reduced ShallowFBCSPNet, and a classifier around one
private LUNA rotary-attention block, not the full LUNA model). It runs FP32 on
CPU only; any other device is `unsupported`. Modes:

- `eager`: one training step of the configured model (seeded forward,
  cross-entropy against fixed targets, backward, SGD update). This is the
  oracle; its own check is self-repeatability, not an independent reference.
- `compiled`: the same step for the same configured model with the module
  wrapped by `torch.compile` (inductor). The forward and AOT backward are
  compiled; `zero_grad`, the loss and `optimizer.step` run eagerly. Timed
  eager and compiled cases do equal work (`timed_configuration` matches).
  Correctness is a set of structured checks in `correctness.metrics.checks`:
  dropout-off parity (`dropout_override: 0.0`), a partial final batch on the
  same pair with recompiles recorded, dropout-on parity under
  `torch._inductor.config.fallback_random=True`, and finiteness of the timed
  model's first step. **Limitation:** the timed model uses inductor's default
  RNG, whose dropout masks differ from eager's; those masks are not compared.
  Compile setup time, graph breaks, graph count and FX-graph cache hits/misses
  are recorded outside headline timing. `torch._dynamo.reset()` runs before
  every compile; the on-disk inductor cache is not cleared, so set
  `TORCHINDUCTOR_CACHE_DIR` per run when cold compiles matter.
- `exported`: eval-mode forward of a `torch.export` artifact after a real save
  and reload, compared with the eager forward (`rtol=1e-4`, `atol=1e-5`).
  `--dynamic-batch` exports one artifact with a dynamic batch dimension and
  executes it at the example batch, batch 1, a partial batch and a larger
  batch. Export, save, load and unlift costs are in `export_setup_s`; timed
  calls reuse one unlifted module under `torch.no_grad`.

Status: `ok` means the mode's implemented qualification checks passed;
`failed` means a compile/execution error, compiler timeout
(`--compile-timeout-s`; `compile_timed_out: true`) or detected numeric mismatch,
with correctness marked `failed` too. Eager/compiled modes additionally fail
if parameters, buffers or gradients are nonfinite after timing. Exported mode
records that same state check as `post_timing_finite` but does **not** gate
status/correctness on it: `ok`/`passed` can coexist with
`post_timing_finite=false`. These post-timing checks do not universally
revalidate timed outputs. Inspect the diagnostic explicitly before using an
exported record for qualification. Completed samples from failed runs remain
diagnostic, not qualified performance evidence. `unsupported` means
the request cannot be qualified (non-CPU device, invalid model configuration,
export not possible, or no bounded-compile support on this platform). The unit
tests treat any unsupported result for these fixtures as a failure unless it
is listed as known.

`effective_samples` and `optimizer_updates` describe one call; timing repeats
are not multiplied in. `memory.peak_cuda_allocated_bytes` stays null on CPU;
`cpu_peak_rss` is a process-level `ru_maxrss` high-water mark that includes
imports and earlier work, not a per-step allocation peak. Only the bare module
is qualified: skorch estimator compile behaviour is **not** covered. Paired
timing claims follow the five-pair rule above.

## Fold-safe repeated fits (cross-validation reuse)

```sh
mkdir -p /tmp/repeated-fits  # any private directory outside the checkout
python -m benchmarks.acceleration.bench_repeated_fits --mode cold --n-jobs 1 \
    --work-dir /tmp/repeated-fits --output /tmp/repeated-fits/cold.json
python -m benchmarks.acceleration.bench_repeated_fits --mode reuse --n-jobs 2 \
    --work-dir /tmp/repeated-fits --output /tmp/repeated-fits/reuse-2.json
```

`bench_repeated_fits` is a recipe, not a library feature: it shows how to reuse
prepared data across the independent fits of a cross-validation study without
leaking held-out information or state between folds. It uses only existing
APIs (`preprocess`, `create_fixed_length_windows`, `BaseConcatDataset.save`,
`load_concat_dataset`, `sklearn.base.clone`, joblib/loky) and changes no
Braindecode default or cache. Run the recipe in a disposable compute process:
its fitting helpers set Torch thread counts and fold seeds without restoring
the caller's prior thread/RNG state. They are not transparent in-process library
utilities. Exploratory summaries may configure their expected run count; retained
qualification claims still require five independent paired runs.

- The source is synthetic (lateralized 10 Hz rhythm in noise, 8 subjects,
  22 channels). `prepare_source` filters, scales units and windows it once,
  saves it, and writes a manifest with a `source_key` and a checksum of the
  saved files. Its `source_revision` records every generator input (generator
  version, seed, subject and trial counts). `load_concat_dataset` writes a header pickle next to each FIF
  file on its first load, so preparation does one lazy load before the
  checksum and reuse never writes. `load_source` refuses a source whose files
  changed and makes the loaded signal arrays read-only. Use a source in place:
  the header pickles hold absolute paths, so a moved copy is refused.
- `split_schedule` builds subject-disjoint folds fixed by the seed.
- The only fitted preprocessing artifact, a per-channel standardization, is fit
  on training windows only. `artifact_key(source_revision, preprocessing,
  split_ids, versions)` hashes canonical JSON of everything that defines it
  (channel order, units, sampling rate, filter and window parameters, target
  mapping, training IDs, library versions). It is a local identity for this
  benchmark's explicit, immutable source, **not** a transparent cache.
- Every fold fits `clone()` of an unfitted `EEGClassifier` template
  (`warm_start=False`), reseeds torch from `SeedSequence([seed, fold])`, and
  uses one torch thread.
- `BaseConcatDataset.split` returns views that share child datasets, so setting
  `transform` on one view changes every view. `fold_datasets` gives each fold
  shallow copies of the children instead: the transform stays on the copies,
  while the read-only signals are shared rather than duplicated. Only
  `transform` is fold-local: the MNE raw object, window metadata, labels (`y`)
  and descriptions stay shared by reference. Never call raw-modifying
  preprocessing (`pick`, `crop`, `resample`, ...) or mutate `y`/`metadata` on a
  fold view; that would reach every fold.
- `run_repeated_fits(source_dir=..., n_folds=..., n_jobs=..., seed=...)` spreads
  folds round-robin over `n_jobs` explicit loky workers; each worker loads the
  persisted source once. `n_jobs` must be a positive integer (`-1` is
  rejected), and the CLI refuses more workers than physical cores in its CPU
  affinity (SMT siblings count once). Each record stores the affinity and
  the physical-core count.
  Results do not depend on `n_jobs`.

Modes: `cold` is the serial no-reuse baseline. It prepares and saves a fresh
source for every fold, then loads it and fits. `reuse` prepares once, or skips
preparation with `--source-dir`, then runs `run_repeated_fits`. The JSON
records study makespan, preparation time, per-fold load/fit time, effective
samples and optimizer updates (from skorch batch history), per-fold parameter
and prediction digests for parity, and an aggregate RSS. The aggregate RSS is
the sum of per-process `ru_maxrss` high-water marks (parent plus workers): an
upper bound, not a concurrent sample. Makespan excludes interpreter start,
imports and removal of temporary sources, in both modes.

This is a one-shot study timing, so the record is this recipe's own JSON rather
than the v1 envelope above, but it keeps the envelope's correctness rule: each
record has `status` (`ok` or `failed`) and a within-run `correctness` check
(every fold fitted once, subject-disjoint, artifact fit on training IDs), and
the CLI exits non-zero when it fails. `summarize_runs` gates a study: it
reports paired makespan ratios only when every configuration has the expected
number of repeats (five for any comparison), every run exited 0 with status
`ok` and all folds, and per-fold parameter, prediction, artifact and count
digests are identical across modes, worker counts and repeats (parity).
Otherwise `ratios` is null and the failures are listed.

The cold baseline saves and reloads a source per fold, so a reuse/cold ratio
is a saving against that persisted cold baseline on the filesystem used (state
it, e.g. NFS scratch), not a general preparation saving. Compare worker
counts only with whole physical cores per worker, and state the resources of
each configuration.

`test/unit_tests/benchmarks/test_repeated_fit_contract.py` checks the contract:
- the key changes with each scientific input, and sources differing only in
  subject or trial counts get different keys;
- perturbing only held-out data leaves the training artifact and the fitted
  parameters unchanged, while perturbing training data changes them;
- estimators, modules and optimizers are distinct, with no `warm_start`;
- the split schedule is fixed;
- fold-local transforms do not leak between folds;
- `n_jobs=1` and `n_jobs=2` give the same results;
- runs carry a correctness status, and the study summary withholds ratios
  after a parity mismatch, a missing or failed run, a non-zero exit, missing
  folds or a wrong number of repeats.

## Exact online window replay (scheduling only)

```sh
python -m benchmarks.acceleration.bench_stream_replay --output /tmp/replay-1.json
python -m benchmarks.acceleration.bench_stream_replay \
    --summarize /tmp/replay-*.json --expected-runs 5
```

`bench_stream_replay` checks that windows built incrementally from streamed
chunks are exactly the offline windows, and times an unchanged eval model on
them. It is a benchmark-local scheduler, **not** a public streaming runtime, and
adds no dependency.

- `iter_completed_windows(chunks, *, window_size, stride, drop_last_window)`
  takes already-preprocessed samples of exactly one trial as chronological
  `(channels, time)` arrays and yields `(window, start, stop)` (exclusive
  `stop`) as soon as the chunk completing a window arrives. The end of
  `chunks` is the explicit flush: with `drop_last_window=False` the
  tail-aligned last window appears only then, and never duplicates a regular
  window that already ends at the trial end. A trial shorter than the window
  raises at flush, as the offline windower does. Each call starts empty.
- Windows are new arrays owned by the caller; inputs are not modified or
  retained. Between chunks at most `window_size` samples per channel are held
  (in a `2 * window_size` buffer), whatever the trial duration. Invalid size,
  stride, policy, dimensionality, channel count or dtype raises explicitly.
- The reference is `create_fixed_length_windows` read through
  `EEGWindowsDataset`; the tests also cross-check the event windower's index
  computation. They cover random chunk sizes including single samples and
  empty chunks, overlap, stride larger than the window, short trials, both
  tail policies, reset between calls, availability (`stop` never exceeds the
  samples received) and buffer bounds.

The CLI replays synthetic trials through `ShallowFBCSPNet` (or `EEGNet`) in
eval mode under `torch.no_grad`, one window per forward, without changing the
model. Streamed outputs must be bit-identical to the offline windows evaluated
one at a time; the difference from batched offline evaluation is only
reported. The JSON is this benchmark's own record, not the v1 envelope,
because it reports per-window latency quantiles. It keeps two kinds of delay
apart:

- `intrinsic_delay`: signal time inherent in windowing, in samples and
  seconds: window span, output period, chunk wait (samples between a window's
  stop and the end of the chunk that completes it) and whether the tail window
  needed the flush. It does not depend on hardware.
- `compute_latency`: wall time of an as-fast-as-possible replay, not paced in
  real time: per-window latency from handing in the completing chunk to the
  output (p50/p95/p99), the forward alone, first output per trial and the
  first forward of the process (before an evidence-based warmup).

The model-facing replay contract is float32. Feed identical float32 samples to
both paths: the scheduler preserves input dtype, while the offline dataset
casts signals to float32, so float64 chunks alone do not establish model parity.

Exact equivalence holds only when online and offline see the same samples. It
is **not** claimed for standardization reset per chunk, non-causal (e.g.
zero-phase) filters or any other preprocessing whose online form differs;
those need stateful preprocessing and their own quality budget. `summarize_runs`
pools runs only when there are exactly the expected number (five for any
claim), all `ok`, from one clean commit, one host and one configuration.

### One long trial: bounded cropped output (opt-in)

`acceleration/bench_long_trial_chunks.py --output <new-directory>` runs five
alternating reference/chunk pairs (zero-based labels 0 through 4) in fresh
processes for synthetic 1 h and 4 h
trials at 250 Hz. The reference is `iter_trial_predictions` (already bounded
across trials); the candidate is `iter_trial_prediction_chunks`, 32 windows per
chunk. Both use batch 16, 8 output channels, 1024 samples/window, stride 512,
workers 0 and one Torch thread. Set OMP/MKL/OPENBLAS thread counts to 1 as well.
It records raw warmups/samples, steady status, exact per-channel prediction and
time-target digests, cold verified latency and whole-process Linux peak RSS.
Analytic checks are outside timed samples. RSS includes imports and correctness
checks, not an incremental allocation measure. Execute in an isolated dedicated
CPU Job; no real EEG, model throughput, accelerator or training claim follows.

```python
from contextlib import closing
from braindecode.training import iter_trial_prediction_chunks

with closing(
    iter_trial_prediction_chunks(model, windows, windows_per_chunk=32)
) as chunks:
    for chunk in chunks:
        consume(
            chunk.trial_index, chunk.prediction_start, chunk.prediction, chunk.target
        )
```

`prediction_start` is an offset in the reconstructed prediction sequence, not an
EEG sample index. Concatenate predictions (and time targets) along axis 1 within
each trial to recover the old output exactly. Scalar/vector targets repeat per
chunk. Retaining every chunk defeats bounded memory. Explicitly close on early
exit to restore model modes and release DataLoader workers.

## Qualification boundary for historical acceleration recipes

The retained figures are historical, independently audited evidence on their
recorded stack, not a certificate for a new host or source revision. The Mixup,
direct-module execution, repeated-fit, replay, trial-inference and long-trial
CLIs now explicitly emit `qualification.status: historical_only` and
`new_claim_eligible: false` (inside `config.parameters` for v1 envelopes).
`ok`/`passed` means only the implemented diagnostic checks passed. It must not be
read as a new qualified result or a complete eager/compiled/exported matrix.
The CLIs offer no switch to promote it to qualified. New claims require retained
launch/resource/topology records, independently verified clean source hashes,
versions/environment and explicit scheduling/cache/measurement policy receipts,
finite raw samples, median/stability gates and complete five-pair results. These
receipts remain external to these historical recipes; no schema asserts their
truth. Earlier audited records retain their original scope without remeasurement.

Replay now saves `compute_latency.raw_latency_s` and `raw_model_s` in window
order. Its summary rejects missing/nonfinite/nonpositive samples, inconsistent
quantiles, non-steady warmup, and unverified/oversubscribed physical-core/thread
counts before aggregating **per-run quantiles** (not pooled window samples).
Old records without these fields are diagnostic only and fail the new summary.
Repeated-fit checks bind every fitted fold to its scheduled train/test IDs,
validate finite training artifacts and positive standard deviations, and require
artifact `n_windows == n_train_windows` and positive held-out counts. Summaries
require identical nonnegative integer repeat-label sets across configurations;
window counts participate in parity. They still do not attest an artifact's
provenance merely because its key has the right shape.

Direct-module compilation counts actual forward samples and optimizer steps in
an extra untimed invocation of the actual timed callable, rather than inferring
equal work solely from constructor metadata. Perturbed compiled outputs are a
regression fixture. Only `CompileTimeout` (including a wrapped sentinel) means a
compiler deadline; unrelated I/O `TimeoutError` remains an execution error.
Timeouts must be finite and positive at entry. `timing_compile_diagnostics`
records graph counts before/after all headline timing and resulting diagnostics;
a timing recompile is not hidden as setup-only compilation. This remains
mode-specific diagnostic evidence, not an automatically qualified matrix.

Trial-inference's `pre_call_process_high_water_rss_bytes` is **not current RSS**:
it is the process lifetime high-water before the call. Its peak includes all
prior allocations. The long-trial CLI's RSS also includes analytic correctness
allocations and **cannot reproduce the consumption-only memory gate**. Do not
promote its peak or the old external consumption-only receipt as a memory
headline reproducible by this CLI. It remains a latency/correctness diagnostic;
no new consumption-only or incremental-allocation claim is supported here.
Mixup, trial and long-trial RSS measurements explicitly require Linux, where
`ru_maxrss` is KiB; help works elsewhere but measurement rejects unsupported
units. Topology fallback counts logical CPUs only and is not proof of physical
cores on platforms without sysfs; no new scaling qualification follows.

The filterbank instrumentation covers only `order_by_frequency_band=False`
(the default) with `drop_original_signals=True` in the benchmark's real call.
It does not certify the frequency-band-ordering branch.

Mixup is a historical rejected-candidate replay: it needs both exact Git objects
`0dd2980f45682129d8458ac3a83f567573b8a2fa` (scalar baseline) and
`e77d2587623b52de96d4ac25df9b1aaa7e63affe` (rejected vectorization).
Use the retained private full-history clone, or explicitly fetch those objects
from an authorized historical remote before running. Shallow clones and source
archives need that provisioning; the CLI fails with the exact missing object
and instructions. Never substitute current production for e77d2587 or infer
that the rejected path is enabled today. Unit oracles remain history-independent.

Expensive repeated-fit scientific studies are marked `slow`; deterministic
record/topology gates and the default tiny CLI smoke remain separate. Use
`pytest test/unit_tests/benchmarks -m 'not slow'` for ordinary contracts and
`pytest test/unit_tests/benchmarks --run-slow` for the explicit complete
qualification-test partition. The shared test/conftest.py slow-tier selection
also supports `--run-slow -m slow`; do not register a second marker. Fit tests restore caller Torch thread/RNG and NumPy RNG state.
