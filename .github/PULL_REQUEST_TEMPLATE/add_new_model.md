<!--
Thank you for contributing a new model! Follow CONTRIBUTING.md ("Adding a model
to Braindecode") and test/unit_tests/models/test_integration.py. Tick completed
items; keep inapplicable items with "N/A — reason". Explain exceptions rather
than marking unrun checks as passing. Suggested title: [models] Add <ModelName>.
-->

## Model information

- **Model name**:
- **Paper**: <!-- link to the publication; models must be published -->
- **Motivation**: <!-- what this model adds over the models already in braindecode.models -->

## Implementation fidelity

- **Reference implementation**: <!-- source URL and commit/tag; license and license URL, or explain unavailable -->
- **Deviations**: <!-- architectural/numerical changes, renamed layers, reused building blocks; or none -->
- **Checkpoint/parity evidence**: <!-- if available: checkpoint source, weight mapping, comparison setup/tolerances/results; otherwise limitations. Distinguish implementation agreement from benchmark reproduction. -->

## Checklist

### Implementation (`braindecode/models/<name>.py`)

- [ ] Class inherits from `EEGModuleMixin` **before** `nn.Module` (or `nn.Sequential`); `license="<SPDX id>"`, attribution and any `NOTICE.txt` entry match the source license
- [ ] Signal parameters (`n_outputs`, `n_chans`, `chs_info`, `n_times`, `input_window_seconds`, `sfreq`) forwarded to `super().__init__(...)`
- [ ] Input/output shapes documented and tested: normally `(batch_size, n_chans, n_times)` → `(batch_size, n_outputs)`; explain temporal outputs or other task-specific shapes
- [ ] `self.final_layer` is among the last two child modules (`test_model_integration_has_final_layer`); classification head has no final softmax/log-softmax (`EEGClassifier` handles it)
- [ ] Activation parameters use `nn.Module` class defaults (e.g. `activation=nn.ELU`) and dropout parameters use `drop_prob` names, as applicable; explain exceptions against the integration tests
- [ ] Model docstring describes the architecture, parameters and paper (numpydoc, `[1]_` reference); deviations recorded above or linked to the docstring
- [ ] Runtime dependency changes declared and justified, or none

### Registration and documentation

- [ ] Exported in `braindecode/models/__init__.py` (import + `__all__`); exports are discovered automatically by `_init_models_dict()`
- [ ] Test case added to `models_mandatory_parameters` in `braindecode/models/util.py`, with suitable signal parameters; task-specific test exceptions explained if needed
- [ ] Row added to `braindecode/models/summary.csv`, including `#Parameters` and `get_#Parameters` (`test_completeness_summary_table` checks row presence)
- [ ] API entry in `docs/api.rst`
- [ ] Architecture figure at `docs/_static/model/<name>_arch.png`, if applicable (or N/A with reason)
- [ ] Entry in `docs/whats_new.rst` (required by the changelog CI check)

### Validation and compatibility

- [ ] Relevant integration and model-specific regression coverage added/updated in the shared model test suites; commands and results recorded below
- [ ] Style checks recorded below, e.g. `pre-commit run --files <changed files>`
- [ ] Shared components changed: <!-- none, or list affected models/APIs and regression coverage -->
- [ ] Where relevant, checkpoint/config save-load round trips tested, including after head changes; evidence or N/A below
- [ ] Where supported or changed, `reset_head` sentinel semantics (e.g. `0`/`None`) and train/eval behavior declared and tested, including mixed submodule modes where relevant; evidence or N/A below

## Validation evidence

<!-- Exact commands, environment/device, results (including failures/skips), and relevant CI links.
Separate focused checks from the full suite (`pytest test/`); say what was not run and why.
Include compatibility evidence or N/A reasons for the conditional items above. -->

## Benchmark reproduction

- [ ] Benchmark on at least one public dataset reported, or limitations explained below
- [ ] Table/figure compares to the paper's values (per subject or dataset), or explains why comparison is unavailable
- [ ] Dataset/version, splits, preprocessing, evaluation protocol, optimizer, training budget and deviations from the paper documented
- [ ] Any gap from the paper's results or incomplete reproduction explained

<!-- Results and protocol, or limitations (e.g. unavailable data/code/weights or compute).
Implementation/parity evidence above is not by itself reproduction of paper benchmarks. -->
