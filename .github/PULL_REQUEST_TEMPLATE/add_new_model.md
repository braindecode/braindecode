---
name: New model
description: Add a published PyTorch model to braindecode.models
title: "[models] Add <ModelName>"
---

<!--
Thank you for contributing a new model! This checklist mirrors the guide in
CONTRIBUTING.md ("Adding a model to Braindecode") and the conventions enforced
by test/unit_tests/models/test_integration.py. Fill in the model information,
tick the boxes as you go, and keep the sections that do not apply with a short
note explaining why.
-->

## Model information

- **Model name**:
- **Paper**: <!-- link to the publication; models must be published -->
- **Reference implementation**: <!-- link to the authors' code, and its license -->
- **Motivation**: <!-- one or two sentences: what this model adds over the models already in braindecode.models -->

## Checklist

### Implementation (`braindecode/models/<name>.py`)

- [ ] New class inheriting from `EEGModuleMixin` **before** `nn.Module` (or `nn.Sequential`), with `license="<SPDX id>"` matching the reference implementation
- [ ] Mandatory parameters (`n_outputs`, `n_chans`, `chs_info`, `n_times`, `input_window_seconds`, `sfreq`) forwarded to `super().__init__(...)`
- [ ] Forward pass consumes `(batch_size, n_chans, n_times)` and returns `(batch_size, n_outputs)`, transposing internally if the original model expects channels last
- [ ] The classification head is assigned to `self.final_layer` and is among the last two layers (`test_model_integration_full_last_layer`)
- [ ] No softmax or log-softmax after the final layer (`EEGClassifier` applies it)
- [ ] Activation functions exposed as `__init__` parameters with class defaults, e.g. `activation: type[nn.Module] = nn.ELU` (`test_model_has_activation_parameter`, `test_activation_default_parameters_are_nn_module_classes`)
- [ ] Dropout probabilities exposed as `__init__` parameters (`test_model_has_drop_prob_parameter`)
- [ ] Module docstring documents the model, its parameters and the paper (numpydoc, `[1]_` reference)
- [ ] Any deviation from the reference implementation (renamed layers, shared building blocks, different attention scaling, ...) is listed in the docstring or in this PR
- [ ] No new runtime dependency

### Registration and documentation

- [ ] Exported in `braindecode/models/__init__.py` (import + `__all__`)
- [ ] Registered in `braindecode/models/util.py`
- [ ] Row added to `braindecode/models/summary.csv`, with `#Parameters` and `get_#Parameters` filled in (`test_completeness_summary_table`)
- [ ] API entry in `docs/api.rst`
- [ ] Architecture figure at `docs/_static/model/<name>_arch.png`
- [ ] Entry in `docs/whats_new.rst` (required by the changelog CI check)

### Validation

- [ ] Integration tests pass for the new model (models registered in `util.py` are picked up automatically by `test/unit_tests/models/test_integration.py`)
- [ ] `pytest test/` and `pre-commit run --all-files` pass
- [ ] Benchmark reproducing the paper's results on at least one public dataset:
    - [ ] Table or figure comparing to the paper values (per subject or per dataset)
    - [ ] Training protocol described: dataset, splits, optimizer, training budget, and any deviation from the paper
    - [ ] If the paper's numbers could not be reproduced, an explanation of the gap

## Benchmarks

<!-- Paste the results table or figure and the training protocol here. -->
