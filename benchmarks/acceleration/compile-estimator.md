# Opt-in compilation through EEGClassifier

The estimator already inherits `compile=True` and `compile__*` from skorch.
Use those options, not a separately wrapped model:

```python
net = EEGClassifier(
    "EEGNet",
    module__n_chans=8,
    module__n_outputs=2,
    module__n_times=512,
    compile=True,
    compile__backend="inductor",
    compile__dynamic=False,
    iterator_train__drop_last=False,
)
```

Braindecode compiles modules/criteria in place with native `Module.compile` so
`module_.state_dict()` and `save_params` keep canonical unprefixed keys. This
also keeps optimizer references aligned with the trained module parameters.
Supplied module and criterion instances retain their exact Parameter objects,
including deliberately shared Parameters and reference-based optimizers; no
model copy or extra parameter/buffer storage is introduced. Compilation is
in place. The estimator records the exact module and callable it installed in
private transient state; it never writes attributes onto a compiled callable.
Reinitializing this estimator with compile=False clears only a slot still
identical to its recorded callable. This also handles TORCHDYNAMO_DISABLE=1,
where native compilation returns a plain bound method and execution stays eager. An independently user-compiled callable (including one replacing an
estimator-installed callable later) is deliberately neither replaced nor
cleared. A different estimator does not own that callable and does not clear it;
turn compilation off through its owning estimator before independent eager
reuse. Factory/class initialization and warm-start forwards retain their path.
Changing compile__backend reconfigures an estimator-owned callable; the existing
compile__disable=True option explicitly restores eager dispatch for that owner.
No new public parameter,
mandatory dependency or compilation/precision default is introduced.
Existing checkpoints made with the skorch OptimizedModule wrapper may carry
`_orig_mod.` keys; this change does not silently rewrite those legacy artifacts.
For an explicitly identified wrapper checkpoint, unwrap it before strict loading
(the module itself remains the authority for its keys).

The supported checkpoint workflow is save_params/load_params, including trained
optimizer state and history. Whole-estimator pickle/joblib is a different
contract: native Module serialization drops the active compiled callable, so
loading a full object does not itself re-enable compilation. Estimator-side
ownership references are excluded from __getstate__, so clone/pickle/deepcopy do
not serialize compiled closures or stale ownership. A loaded/copied net
re-establishes ownership on its next initialize; warm-start/partial_fit alone
does not promise reactivation of compilation.

## Scope of numerical qualification

The maintained test covers CPU FP32 (`rtol=2e-4, atol=2e-5`) with deterministic
tiny modules; the four real model qualification cells are also FP32. The existing
estimator loader casts signals to FP32, so FP64 input is not an FP64 execution
qualification and is not claimed. These predeclared tolerances allow compiler
floating-point reduction/reassociation rounding, not reduced precision.
Discrete predictions, batch order/counts, callbacks/history schema and saved
keys must agree exactly. Tests also cover clone/refit and partial/new batch sizes.
No FP64/BF16/FP16, minimum-version run or accelerator parity follows from these tests.
Native Module.compile exists at the project's torch>=2.6 floor; the exercised
stack is torch2.14.1/skorch1.4.0, not an empirical minimum-version certification.

Dropout is disabled only for numerical qualification, not performance timing.
Compiled dropout can advance a different random stream; identical seed does
NOT promise identical dropout-on training trajectory. Preserve the user's
scientific policy: compile is explicit opt-in and never a transparent default.

## Reproducible measurement

`bench_compile_estimator.py --suite --out <directory>` launches five fresh
process pairs per model, alternating eager/compiled ordering. Worker count0,
4 Torch/BLAS threads, FP32, 66 examples of 8x512, batch16 + final2, same SGD and
original dropout. EEGConformer is explicitly small (2 layers,20 filters,4 heads);
other constructors retain defaults. Both model and criterion are compiled via
the estimator. An epoch includes estimator callbacks, history and optimizer;
there is no validation split in this bounded training cell. Numerical
qualification is separate and must pass before suite execution.

Every compiled run uses a distinct Inductor cache. Use a fresh cache root for
each repeat; deterministic directory names are not a cold-cache guarantee when
reusing a previous root. The steady flag checks warmup only, not the subsequent
five-sample measured-window CV. First-epoch and initialize
costs are recorded separately from steady training (warmup until last3 CV<=10%,
max12 epochs, followed by5 measured epochs). Import/process startup is excluded.
Break-even is `1 + ceil(max(0, compiled_first-eager_first) / steady_saving)`;
`first` includes estimator initialization. N=1/10/100 projections are simple
linear extrapolations, not measured multi-hour training results. Report raw
samples, pair ratios, median/IQR, losing cases and unstable warmups. Recommend a
cell only when the predeclared median C/E<=0.95, >=4/5 pairs faster, all steady.
Hardware, versions and actual results live in the Wave2 compile-estimator report;
never generalize CPU fixtures into an A100 throughput claim.

### Current-stack dynamic-shape restriction

The measured recipe explicitly sets `compile__dynamic=False`, specializing
separate graphs for the full and partial final batch. Torch2.14.1 Inductor's
auto-dynamic path failed EEGNet's partial-tail backward with a symbolic-stride
sorting error (`1 < s69`). No fallback hides it; the exact failed log is retained.
This is not a default change or an assertion that arbitrary dynamic shapes work.
