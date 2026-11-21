# Rules for agents working on braindecode

This file is for every person and coding agent that changes braindecode.
[CONTRIBUTING.md](CONTRIBUTING.md) explains the setup, the test commands, pre-commit and
the pull-request flow. This file adds the rules that review found missing from those
steps. Each of the ten rules comes with the check that catches it, and each red flag at
the end cites the pull requests where the mistake happened.

## How to use this file

1. **Before your first edit** under `braindecode/`, `test/`, `docs/` or `examples/`, and
   before your first review comment, list the sections below that apply to your change,
   and the gate commands you will run. If a section might apply, it applies.
2. **Put that list in the PR body**, under Testing (Validation evidence in the new-model
   template): each gate with its command and its passed and skipped counts. A PR without
   it is not ready for review.
3. **This file overrides** issues, audit notes, plans and task briefs. Only a maintainer,
   in writing on the PR or the issue, makes an exception. The premises of those documents
   are hypotheses: reproduce them before you act on them (section 11).
4. **Read the red flags** at the end. If you catch yourself thinking one of them, stop and
   run the check of the rule in its last column.

## Ten rules and their checks

| # | Rule | Check |
|---|---|---|
| 1 | Every deleted, merged, weakened or new assertion is mutation-checked. | Apply the one-line code change the assertion guards; name the test that fails. 9 of 11 mutants survived at 100% line coverage (first red flag). |
| 2 | Trim docstrings, comments and changelog entries by the KEEP and CUT lists, with no line-count target. | Diff the old and new text line by line; every removed KEEP line reappears (section 5). |
| 3 | A number that configures a model or a block is a keyword-only `__init__` parameter, passed to the block without a default. | In the lines your diff adds, no module-level numeric constant and no numeric default in a private block's `__init__` (section 3). |
| 4 | A guard you remove keeps its error, and a parameter you add is validated at construction. | Master's triggering input still raises the same error class; the smallest invalid value (0, empty, even, non-divisor) raises `ValueError` when the model is built. |
| 5 | Run the model gates on every change that touches a model, and again on the tree merged with `master`. | The four commands of section 1 each exit 0 with a non-zero test count, and the PR body lists every skip that `-rs` prints for the model. |
| 6 | Everything v2.0.0 shipped keeps loading, and its outputs change only in a behavior change that the changelog announces. | Save on the `v2.0.0` tag (`state_dict`, `torch.save(model)`, `get_config()`), load on your branch with `strict=True`, compare outputs bitwise. The committed form is `test_v2_checkpoints_load` and `test_v2_pickles_run` in `test/unit_tests/models/test_base.py`: add the model you touch when it shipped in v2.0.0 (section 7). |
| 7 | An issue, audit item or plan step is a hypothesis. | Reproduce it: the input that reaches the "unreachable" branch, the pickle an "unnecessary" shim serves, the test that "covers" the one you delete. |
| 8 | Every behavioral sentence you publish is backed by a probe you ran. | The PR body maps each claim in the body, changelog, docstring, code comment, model card or tutorial to its command or test. |
| 9 | Merge only the head that was reviewed, on a green run. | `gh pr view <n> --json headRefOid` equals the reviewed SHA; `git merge-tree --write-tree origin/master <head>` exits 0; `gh pr checks <n>` is green; a red run is read before anyone re-runs it. |
| 10 | Released weights move together with the code that reads them. | A maintainer uploads to a Hub branch; verify each file by hash, pin that commit in the test, merge, then a maintainer moves `main`. The pinned test sees every tensor group (section 8). |

## 1. The model contract

CI enforces the tests below. Run these four commands on every change that touches a
model (`<Model>` is the class name):

```
pytest -rs test/unit_tests/models/ --select-models=<Model> -k "<Model> or completeness or summary_csv or categorization"
pytest -rs test/unit_tests/models/test_model_contract.py --select-models=<Model> -k <Model>
pytest -rs test/unit_tests/models/test_pretrained_compat.py --select-models=<Model> -k "<Model> or covers_every or no_strategy"
pytest -rs <the model's own test file, e.g. test/unit_tests/models/test_<model>.py>
```

- `--select-models=<Model>` does for your model what CI does on a pull request: CI passes
  it the models your diff reaches. Without it, `test_model_compiled` skips every model
  outside the seven of `_COMPILE_ON_PR` in `test_integration.py`, so a `torch.compile`
  failure shows up only in CI. `-rs` prints the reason of each skip: list the skips in the
  PR body.
- The first line already runs the model's contract tests. The second line is the quick
  loop while you work on the contract. The third line adds the two coverage gates,
  `test_compat_covers_every_pretrained_model` and
  `test_no_strategy_lists_every_model_without_the_argument`. They take no model parameter,
  so `-k <Model>` alone never selects them.
- Commit, then run `python scripts/ci_plan.py --base origin/master --head HEAD --pretty`.
  It prints the test files (`shard_files`) and the models (`select_models`) that CI runs
  for your diff. Run the listed files that the four lines do not cover, with the same
  `--select-models`. If the plan says `"full_suite": true`, CI runs the whole suite for
  your PR, and its `reason` says why.
- A selection of 0 tests (pytest exit code 5) counts as a failure: check the class name
  and the path.
- `-k` also matches marker names. A bare substring of a marker, such as `filter` (which
  matches `filterwarnings`), selects hundreds of heavy tests: 807 instead of 225 over
  `test/unit_tests/`.
- Run the gates again on your branch merged with the current `master` before merging.
  Another PR can change a contract test that every model runs.

What the gates enforce:

| Check | Failing test |
|---|---|
| `class X(EEGModuleMixin, nn.Module)`, mixin first; the six signal parameters (`n_outputs`, `n_chans`, `chs_info`, `n_times`, `input_window_seconds`, `sfreq`) default to `None` and are passed to `super().__init__`; no `*args` | `test_base.py`, `test_config.py` |
| The output layer is named `final_layer`, among the last two children | `test_integration.py` |
| An `activation` parameter whose default is a class, and a `drop_prob` parameter, unless the model is on a skip list | `test_integration.py` |
| `(batch, n_chans, n_times) -> (batch, n_outputs)`; foundation models return `{"features", "cls_token"}` with `return_features=True`; `reset_head` works | `test_return_features.py`, `test_integration.py` |
| One import line in the `if TYPE_CHECKING:` block of `braindecode/models/__init__.py`, which also gives `__all__`; a `models_mandatory_parameters` entry; a `summary.csv` row; a docstring badge for each category | `test_completeness__models_test_cases`, `test_all_models_in_summary_csv`, `test_model_has_categorization_badges` |
| Constructor parameters are JSON-serializable or a `type`, such as an activation class; config and Hugging Face round trips reproduce the outputs | `test_config.py`, `test_huggingface.py` |
| `torch.export`, TorchScript and `torch.compile` pass, unless the model is on a skip list | `test_integration.py` |
| A model that changes `embed_dim` alone still builds and runs | `test_integration.py::test_if_models_with_embedding_parameter` |
| Runtime, serialization, training, device, host-sync, accelerator-gap, reset-head, `ch_mask` and fewer-channel contracts; every `mapping` value is a distinct `state_dict` key | `test_model_contract.py` |
| A model whose source mentions released weights has an entry in `COMPAT` or `EXCLUDED` | `test_pretrained_compat.py::test_compat_covers_every_pretrained_model` |

Never add a skip or an exemption to make a test pass while the model could meet the
contract: fix the design (#1748 fixed GeoNIRS's widths and removed its skip of
`test_if_models_with_embedding_parameter`). The lists for models that cannot meet a
contract (`EXCLUDED` in `test_pretrained_compat.py` for models without EEG geometry, the
skip lists in `test_integration.py`, the private copies in `test_private_blocks.py`) take a
new entry only with a one-line reason, and the PR body says why the model cannot meet the
contract.

## 2. The quality bar

The exemplars are LaBraM's helper blocks (`braindecode/models/labram.py`) and EEGMiner as it
was before #1101.

- **Comment every operation** in `forward` and in the blocks, with its purpose and the
  shape it produces (`# x: (batch, electrodes, filters)`). Write one comment per operation.
  Do not comment each token of an expression. Check the shapes: run one forward with hooks
  and compare each `# x: (...)` comment.
- **Give every private block a docstring** with its steps and its input and output shapes.
- **Compute sizes with arithmetic in `__init__`** (`n_patches = -(-n_times // patch_size)`).
  Never run the network on a dummy tensor to find a size. A dummy forward you remove was
  also validating the configuration: replace that validation (section 4).
- **Delete the six signal parameters** after `super().__init__` and read them through
  `self.n_chans`, `self.n_times` and `self.n_outputs`. No test checks this.
- **Put the public model class first.** Private blocks, helpers and tables go below it,
  each table with a comment saying what it is and where it comes from.
- **Reuse** torch, MNE and `braindecode.modules` (`FeedForwardBlock`, `MultiHeadAttention`,
  `PatchTokenizer`, `MLP`, `DropPath` and so on) instead of re-implementing them. In code
  you add, do not import a private name from another model's file: move the helper to
  `braindecode.modules`.
- **No speculative code**: no option without a caller, no guess from data values, no shim
  for state that was never released, no hardware workaround in `forward`.
- **No `forward` argument that only raises**, such as a `ch_mask` accepted just to refuse
  it. The training loop already refuses extras that `forward` does not take.
- **One concern per PR.** A refactor changes no output (section 7). A behavior change is
  its own PR with its own changelog entry. Delete a test, a guard, a shim or a docstring
  line only after the check that covers it: rule 1, 4, 7 or 2.

Check the model file you touched, and read every hit. The pattern finds a size taken from
a dummy forward (`get_output_shape`), a `load_state_dict` override or load hook, a
download, and a private import from another model's file:

```
git grep -n -E 'get_output_shape\(|def (load_state_dict|_load_from_state_dict)\b|register_load_state_dict|hf_hub_download|pooch|from braindecode\.models\.[a-z0-9_]+ import _' -- braindecode/models/<file>.py | grep -v 'models\.util import'
```

## 3. Numbers are parameters

A number that configures a model or one of its blocks is a keyword-only parameter of the
public model. That covers sizes, kernels, head widths, numbers of slots or positions,
norm epsilons, numerical floors, initialization ranges and standard deviations, chunk
sizes and frequency ranges.

- **Defaults.** The default is the current value, so outputs and checkpoints do not
  change. Document it in the Parameters section with its default.
- **Blocks.** Pass the value to every private block that uses it, as a constructor argument
  without a default. A block default that no call overrides is a hidden constant.
- **Derived values** are computed in `__init__` with the formula in a comment: a "same"
  padding `k // 2`, a downsampling `pool_size ** n_stages`, a shortest window.
- **One-off facts** are written where they are used, with a comment: the sampling rate of a
  release passed to `warn_if_sfreq_differs`, meters to centimeters, a constant of an
  initializer.
- **Fixed layer stacks** get one tuple parameter per property (`encoder_widths=(64, 128,
  256, 512)`, `encoder_kernel_sizes=(49, 9, 9, 9)`). Sizes are recomputed from the tuples,
  and tuples of one stack with different lengths raise `ValueError`.
- **What stays:** structural numbers (x, y, z = 3; HbO and HbR = 2; q, k, v), mathematical
  idioms (`** -0.5`), validations, and data tables such as channel names and coordinates.

Check the lines your diff adds: no new module-level assignment of a number or a tuple of
numbers outside a table, and no new numeric default in the `__init__` of a private block.
Older models still hold such numbers, and a PR that touches one does not convert them.
Convert a whole file in a PR of its own.

## 4. Validate at construction

A configuration error raises a `ValueError` in `__init__`, before any `forward` runs. The
message names the parameter, its value and the bound, and each error has a test.

- **Removed guards.** A `raise`, a `check_*` call or a dummy forward that you delete may be
  the only validation of an input. Run master's triggering input on your branch: it must
  raise the same error class at construction, or the changelog says what changed.
- **Added parameters.** Build once with the smallest invalid value of each parameter you
  add (0, an empty tuple, an even kernel, a non-divisor), and run `forward` if it builds.
  A divisibility check also rejects non-positive values: `n_heads=0` must not become a
  `ZeroDivisionError`.
- **Same PR.** A guard that your change needs goes in your PR.
- **Outside `braindecode/models`.** A public function or class validates its arguments the
  same way (the first head of #1735 looped forever on `max_event_duration <= 0`).

## 5. Docstrings, comments and the changelog

The class docstring is the model's documentation page. Shorten it without losing
precision. There is no line-count target.

- **Order:** a header line (`"<Name> from <Author> et al (<year>) [key]_."`, or the
  `:footcite:` form if the file uses it), the badges, `.. versionadded::`, a short paragraph
  on what the model is and where the implementation comes from, then the rubrics
  `Architecture Overview` (with a steps block giving each shape), `Macro Components` (each
  `Model.attr` with *Operations* and *Role*), `Temporal, Spatial, and Spectral Encoding`
  (a modality axis such as chromophores gets a fourth bullet here), `Additional Mechanisms`
  and `Pretrained Weights` when weights are released (a `.. rubric::` section; #1724 and
  #1716 demoted it to an admonition and a bullet). Then Parameters (model-specific only,
  each with its default) and References.
- **KEEP** (never cut):
  - equations that define an operation;
  - the steps block with its shapes;
  - one-clause design reasons;
  - input requirements (channel order, units, preprocessing, sampling rate);
  - verified claims with their evidence link;
  - the reference implementation's parameter names;
  - equivalence notes with the reference;
  - what the model does not do.
- **CUT:**
  - quoted passages;
  - porting diaries;
  - replication tables and timings without a link;
  - text about code the class does not contain;
  - instructions for converting checkpoints;
  - prose that repeats the Parameters section.
- **Check:** diff the old and new docstrings line by line (`ast.get_docstring` at the commit
  before the first trim and at your head; `git log -p --follow -- <file>` lists every
  commit that changed the file). Each removed KEEP line must reappear, moved to its rubric
  place if needed. Never delete a section wholesale: sort its lines into KEEP and CUT first.
- **Changelog** (`docs/whats_new.rst`):
  - the entry your PR adds or edits has at most 3 lines, with no hardware timings; shorten
    other entries only in a PR of their own;
  - a behavior change says who is affected (for example "models trained with 2.0
    predict differently"), and a change visible to v2.0.0 users always gets an entry;
  - a change to code not yet released (a model new in the development version) takes the
    `no changelog` label, which a maintainer adds, instead of an entry; the PR that adds a
    model has its entry;
  - when you shorten an entry, apply the same line-by-line check.

## 6. Tests

- **One compact behavioral test per behavior.**
- **No line-count target for a test file.** A test trim is judged by the mutants that the
  remaining tests still catch (rule 1).
- **Not allowed:**
  - copies of the authors' models;
  - tests of random draw order or of initialization statistics;
  - tests of private or torch internals;
  - spawned processes where an in-process call covers the same code;
  - expected values computed with the code under test;
  - loops that can run zero times.
- **Mutation check** (rule 1): every new test, and every assertion you delete, merge or
  weaken, is checked against the one-line change it guards.
  - Before calling a test a duplicate, show the other test that fails on the same change.
  - A claim that a test catches a change is checked with the change that restores the old
    line.
  - A test that "a non-default value changes the output" must first pass on the unchanged
    code.
- **Never delete, skip or weaken a test to save time or memory.** Merge duplicated setups,
  or make the inputs smaller.
- **Network tests.** CI skips every `network` test under `test/unit_tests/`: only
  `test/acceptance_tests` runs with `--run-network`. A behavior that only a `network` test
  guards, such as a released-weights pin, is untested in CI. Add an offline test, or say so
  in the PR body.
- **Memory.** A new test builds small models: no sweep that rebuilds every registered model,
  and no fixture larger than the smallest input that reaches every branch (#1702 cut its
  v2.0.0 fixtures from 622 KB to 230 KB). Do not run the whole suite locally: on a pull
  request CI runs the test files your diff reaches, and the full suite runs on every push
  to `master` and every night. Give each pytest run its own `--basetemp` (create its parent
  first) when other pytest runs share the machine.
- **Selection changes.** A change to `test/conftest.py` changes what every CI run selects:
  compare `pytest --collect-only -q | tail -1` before and after, and give both counts in
  the PR body.

## 7. Change safety

- **Bitwise parity.** A refactor that claims no behavior change compares `master` and the
  branch in float64 with the same seed and `OMP_NUM_THREADS=1`, for the default
  configuration and for each option and input shape the diff touches (#1705's "bit for bit"
  held only for some input shapes):
  - outputs and `return_features`;
  - the input gradient and every parameter gradient;
  - the `state_dict` keys and shapes;
  - `from_config(get_config())`.

  Require `torch.equal`; use a tolerance only when the PR body explains it (for example a
  removed `.contiguous()`).
- **v2.0.0 compatibility.** Keep what the `v2.0.0` tag shipped (`git show v2.0.0:<path>`):
  - state dicts, whole-model pickles and public arguments with their defaults;
  - a moved class keeps an import at its old path, because v2.0.0 pickles name it;
  - adding a parameter with a default is compatible; removing or renaming a v2.0.0
    parameter needs a deprecation.

  Code that is new since v2.0.0 owes no compatibility, but its released weights must keep
  loading.

  The committed check is `test_v2_checkpoints_load` and `test_v2_pickles_run` in
  `test/unit_tests/models/test_base.py`; the comment above `_V2_DATA` says how
  `data/v2_pickles.pth` and `data/v2_state_dicts.pth` were made.
- **Deprecations** use `FutureWarning` (a `DeprecationWarning` is invisible from a user's
  module), name the release that removes the old API, and stay silent under TorchScript and
  `torch.compile`.

## 8. Released weights

- **No conversion code in models.** That means no key tables for third-party checkpoints,
  no `load_state_dict` overrides, no inserted heads, and no downloads at import or
  construction. The one exception is `self.mapping = {old: new}` for a small rename of
  braindecode's own keys, so that v2.0.0 state dicts keep loading; each value is an
  existing, distinct `state_dict` key (the serialization contract checks it).
  - Released weights are re-hosted on the braindecode Hugging Face organization, already
    converted to braindecode's parameter names.
  - The model loads them with `from_pretrained` and nothing else.
- **License.** Read the source license before converting, and say in the PR body which case
  applies: a permissive license, a restrictive one (non-commercial or no-derivatives terms,
  as NICE's CC-BY-NC-ND-4.0), or no license file (GeoNIRS). The maintainer who uploads
  decides on re-hosting.
- **Upload.** Only a maintainer uploads to the braindecode Hugging Face organization. A
  contributor prepares the release folder and the PR: `convert.py`, its parity output and
  the model card go in the PR, and a maintainer uploads the folder.
- **A re-hosted release** holds `model.safetensors`, `config.json`, a model card written for
  the release (never the stub README that `save_pretrained` writes), the `convert.py` that
  produced the files, and the license when the source has one. It never holds
  `pytorch_model.bin`.
  - The card has the base model, the license, what changed from the original release
    (renamed keys, dropped tensors, "values unchanged"), the verification and the citation.
- **Parity.** Every converted tensor is bitwise equal to its source. The outputs match the
  reference implementation, with a check that a 1e-3 change to one weight is visible.
  - Read untrusted checkpoints with `safetensors`, or with `torch.load(..., weights_only=True)`
    after a static scan of its opcodes with `pickletools`; never with a general unpickler.
- **Pinned tests.** The pinned-weights test reads every released tensor group: a 1e-3 change
  to any group fails it. `from_pretrained` does not warn when half or fewer of the
  parameters outside `final_layer` stay at their random initialization. Compare
  `set(model.state_dict())` with the keys of the release file; the PR body lists every key
  that is in only one of the two sets.
- **Order.** A key rename moves the Hub files and the code together:
  1. a maintainer uploads the new files to a branch of the Hub repository;
  2. download them back and compare each file's sha256 with the release;
  3. pin that commit in the test;
  4. merge the PR;
  5. a maintainer moves the Hub repository's `main` to the new files.

  Code that loads `main` without a revision reads the new layout as soon as `main` moves,
  and the two layouts load silently into each other's code (old and new MtReNIRS files and
  code loaded into each other with 6 of 104 tensors left random). When v2.0.0 code loads
  these weights, moving `main` changes what its users download: the PR body says so, and
  the maintainer decides (section 7).
- **Coverage gate.** A model class whose source, code or docstring, contains
  `from_pretrained(`, `huggingface.co/`, `Hugging Face Hub` or `hf_hub_download` needs an
  entry in `COMPAT` or `EXCLUDED` of `test_pretrained_compat.py`.

## 9. Accelerators and devices

- **No vendor-specific code in braindecode.** No branch, import, comment or docstring text
  for one accelerator. Fix an accelerator gap outside braindecode, in that accelerator's
  PyTorch extension or in PyTorch.
- **Device-agnostic formulations are fine** when they are bitwise equal to the plain
  version and every backend implements them. An example is SELU computed through the
  `scale == 1` ELU kernel. The accelerator-gap contract in `test_model_contract.py`
  checks this.
- **Torch floor.** CI installs the newest torch, and no job runs the floor (`torch>=2.6` in
  `pyproject.toml`). When you touch export or TorchScript code, run
  `pytest -rs test/unit_tests/models/test_integration.py --select-models=<Model> -k "<Model> and (exported or torch_script)"`
  with `torch==2.6.*`. The first head of #1706 undid #1690's export fix on torch 2.6 while
  CI stayed green.
- **Meta device.** `test_registered_model_follows_device` moves each registered model to
  `meta` and runs `forward` there: a plain tensor attribute that `.to()` does not move
  fails it. A build under `with torch.device("meta"):`, followed by `to_empty(device="cpu")`
  and `load_state_dict(..., strict=True)`, has no test, and some models do not match their
  CPU-built twin that way today. Treat it as guidance: when your change adds a tensor to
  `__init__`, try that build and compare the outputs. A plain tensor created in `__init__`
  stays on `meta` (#1697, FBCNet's IIR design). Reading tensor values there, such as
  `.tolist()`, raises (#1702, LUNA).

## 10. Pull requests and merging

- **Titles.** PRs are squash-merged, so the PR title becomes the commit message. It starts
  with `FIX`, `REF`, `TEST`, `DOC`, `MAINT`, `API`, `ENH`, `PERF` or `CI`; a new model's PR
  is titled `[models] Add <ModelName>`.
- **PR body.** What changed and why, the evidence (bitwise results, mutation checks), each
  gate command with its counts and the skips that `-rs` prints, and the old-to-new key
  table when weights move.
- **Branches.** Do not force-push a branch under review; merge `master` into it with
  `git merge origin/master`. Reviewers compare each new head with the commit they reviewed
  (rule 9), and a rewrite deletes that commit.
- **Merging:**
  - merge only the head that was reviewed (compare SHAs);
  - with `git merge-tree --write-tree origin/master <head>` exiting 0;
  - on green CI.
  - A red run is read before anything else. A log that ends with "The runner has received
    a shutdown signal" or "lost communication with the server" holds no test result:
    re-run the failed jobs (`gh run rerun <run-id> --failed`) and merge only on a green
    run.
  - PRs that add an entry at the same line of `docs/whats_new.rst` conflict once the first
    one merges. When `git merge-tree` reports a conflict only there, merge `origin/master`
    into the branch and keep both entries.

## 11. Claims and premises

- **What you publish.** Every behavioral sentence in a PR body, a changelog entry, a
  docstring, a code or test comment, a model card or a tutorial comes from a probe you ran.
  "Bitwise equal" needs the parity script, "this test catches X" needs the mutant, and
  "unchanged for released weights" needs those weights loaded. A claim you could not check
  is written as such, or left out.
- **What you receive.** An audit item, an issue, a review comment or a plan step is a
  hypothesis. Before you delete an "unreachable" check, an "unnecessary" shim or a
  "duplicate" test, reproduce the premise on the current code. The same goes for a
  "plain revert" and for a parameter you plan to drop as "unused".

## 12. Conduct

- **Follow-ups.** Name each follow-up by its issue number. If your change needs it, it
  goes in your PR.
- **Scope.** Do not delete or rewrite files outside your change. Do not kill processes you
  did not start.
- **Reviews.** Review the diff of the pull request (`gh pr diff <n>`, or
  `git diff origin/master...<head>`). The files in your working tree can belong to another
  branch.
- **Update this file.** When CI, a reviewer or the maintainer finds something an agent
  missed, the PR that fixes it adds a red flag below or a check above.

## Red flags

Each row gives a belief that cost a braindecode pull request a fix round on 20-21 Nov 2026,
what happened, and the rule or section whose check catches it.

| Thought | What happened | Check |
|---|---|---|
| "Coverage is still 100%, so the trim lost nothing." | 9 of 11 mutants survived the fNIRS test trims at 100% line coverage (#1711, #1715, #1718). | Rule 1 |
| "The plan sets the test file at about 600 lines." | The fNIRS test trims followed such targets, and 9 of 11 mutants survived them (#1711, #1715, #1718). | Section 6 |
| "This test duplicates another one." | #1722 deleted MtReNIRS's `_CASES` entry in `test_ch_mask.py` as a duplicate. It was the only test of per-sample coordinates with a channel mask. | Rule 1 |
| "That raise is never reached." | TinyMyo's raise was the only line of `tinymyo.py` that its tests missed, so CI stayed green when #1717 deleted it, and 14 dims with 3 heads silently attended 12. PBT's "unreachable" check is reached by a v2.0.0 pickle built with `n_times < d_input` (#1702). | Rules 4, 7 |
| "The check is redundant; forward fails later anyway." | The first head of #1710 deleted Pratibha2's `n_times >= patch_size` check. `n_times=100` with `patch_size=200` then built, and `forward` raised an `unfold` `RuntimeError` that names neither parameter. | Rule 4 |
| "It is bitwise equal on CPU, so the helper can go." | #1721 replaced `_SELU` with `nn.SELU`, and `master` went red on the accelerator-gap contract, which the review and the merge checks had not run. #1726 restored `_SELU`. | Rule 5 |
| "`-k <Model>` over the files I touched runs the gates." | The checks before #1721 ran `-k` over four test files without `test_model_contract.py`. `-k <Model>` never selects the coverage gate, which failed CI on #1693, #1695 and #1739. | Section 1 |
| "Named constants are the clean version of magic numbers." | The maintainer wanted parameters. Five follow-up PRs turned the constants into parameters (#1741 to #1744, #1748). | Rule 3 |
| "The docstring is too long; aim for about 85 lines." | The trims, several aimed at line targets such as FNIRST's "198 -> about 85", lost 34 major KEEP items (equations, input requirements, evidence, reference names, equivalence notes) in 12 of the 17 docstrings of the models added since v2.0.0. Four PRs restored them (#1722, #1749, #1754, #1755). | Rule 2 |
| "The 'Differences from the reference' section is a porting diary." | It held the one-line parity claims and their evidence. #1749 restored them. | Section 5 |
| "CI is green, so it works on every torch we support." | CI installs the newest torch. On torch 2.6, the first head of #1702 failed TorchScript of CBraMod and CSBrain, and the first head of #1706 undid #1690's export fix. CI passed both, and reviews that ran torch 2.6 caught them before merge. | Section 9 |
| "It is just the runner; merge." | #1669 merged after two runner shutdowns, with no test result. #1691's red run was a test failure from a stale base. | Rule 9 |
| "The review fixes are staged, so the merge has them." | #1720 merged without them. They came as #1721. | Rule 9 |
| "The pinned weights test guards the release." | GeoNIRS's pin read only HbO tokens (#1695). MtReNIRS's used one window, where the rotary encoding is the identity (#1693). CI skips every `network` test. | Section 8 |
| "`from_pretrained` warns if keys are missing." | It warns only when more than half of the parameters outside `final_layer` stay unloaded. Old and new MtReNIRS files and code loaded into each other with 6 of 104 tensors left random and no warning (#1748). | Section 8 |
| "The spec says it is unnecessary." | AttnSleep's "unnecessary" `__setstate__` serves v2.0.0 pickles (#1699). An EEGMiner change from the spec altered v2.0.0 features and was reverted (#1697). | Rule 7 |
| "The precedent is not to keep pickle aliases." | #1720 deleted `_TransformerEncoderBlock`, and v2.0.0 STEEGFormer pickles stopped loading. #1721 added the alias. | Rule 6 |
| "Dropping the dummy forward changes nothing." | Short windows built zero-size layers in 4 models, and seeded weights changed in MSVTNet and EEGConformer (#1694). Labram lost two decoder-mode checks (#1691). | Rule 4 |
| "The body says the outputs are unchanged." | #1705 claimed "bit for bit", which held only for the 21-channel parity input. A changelog line for USleep said "released weights still load", and USleep has none (#1730). | Rule 8 |
| "My new test catches the ignored-kernel mutant." | The mutant that restored the old line passed all 67 BBL tests (#1743). | Rule 1 |
| "A non-default value changes the output, so the parameter is wired." | In the small MEGXL every token gets the same code, so the test could not see `min_rms` (#1743). | Section 6 |
| "I'll list the guard as a follow-up." | #1743 was opened with CORTEG accepting `n_temporal_positions=2`, which broadcast one temporal embedding over all 8 patches. The implementer had listed the guard as a follow-up, and the review put it in the PR. | Section 4 |
| "A divisibility check is enough." | Applied literally, a review fix turned `n_heads=0` from torch's `ValueError` into a `ZeroDivisionError` (#1744). | Section 4 |
| "That is a robustness layer; leave it out." | #1700 left `get_split_datasets` unguarded "rather than add a robustness layer", and it then failed before `initialize()` for every estimator. The same PR dropped #1644's `cheb_order < 1` check for the same reason. | Rules 4, 6 |
| "A skip makes the contract test pass for my model." | #1695 skipped `test_if_models_with_embedding_parameter` for GeoNIRS, whose `projection_dims` had to start at `embed_dim`. #1748 fixed the widths and removed the skip. | Section 1 |
| "The shorter changelog entry says the same thing." | It stated the fix as the cause (#1703). | Rule 2 |
