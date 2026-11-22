#!/usr/bin/env python3
"""2D Deterministic CI Diff Planner for Braindecode.

Computes minimal sound test file sets, ``--select-models`` filtering, dynamic
shard counts (0, 1, or 4), and job gates (``acceptance``, ``docs``) from a git
diff using:

1. Barrel ``__init__.py`` symbol-level AST diffing (PEP 562 ``_LAZY``,
   ``TYPE_CHECKING``, explicit re-exports, and ``__getattr__`` tables) plus
   subpackage-object importer tracking.
2. Surgical AST diffing of ``braindecode/models/util.py``
   (``models_mandatory_parameters`` entries vs non-core helpers vs core
   model-config infrastructure).
3. Non-Python package/test data asset dependency rules (``summary.csv`` row
   diffing, ``model.html.jinja``, ``test/unit_tests/models/data/*``, and docs
   contract files).
4. Deleted/renamed file safety (never passing deleted test paths to ``pytest``
   while checking reverse importers and registry completeness).
5. Duration-balanced LPT sharding across 0, 1, or 4 shards with AST-based
   ``@pytest.mark.slow`` detection and ``BRAINDECODE_CI_SELECTION=full``
   kill-switch support.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import os
import re
import subprocess  # nosec B404
from collections import defaultdict
from pathlib import Path
from typing import Any

# Global triggers: any change here forces a 4-shard full-suite run with no model filter.
GLOBAL_FALLBACK_EXACT = frozenset(
    {
        "pyproject.toml",
        "uv.lock",
        "setup.cfg",
        "MANIFEST.in",
        ".github/workflows/tests.yml",
        "scripts/ci_plan.py",
        "test/conftest.py",
        "test/dataset.py",
        "test/unit_tests/models/_fixtures.py",
        "braindecode/models/base.py",
        "braindecode/models/config.py",
        "braindecode/classifier.py",
        "braindecode/regressor.py",
        "braindecode/eegneuralnet.py",
        "braindecode/util.py",
        "braindecode/version.py",
    }
)

GLOBAL_FALLBACK_PREFIXES = (".github/scripts/",)

# Core symbols in braindecode/models/util.py whose modification impacts all models.
UTIL_CORE_SYMBOLS = frozenset(
    {
        "__getattr__",
        "_CONFIG_MODULE_TYPES",
        "resolve_config_type",
        "_has_type_annotation",
        "SigArgName",
        "_EEG_PARAMS",
        "_JSON_SAFE",
        "_BATCH_NORM_MODULES",
        "_disable_batch_norm_training_if_batch_size_one",
        "_is_jsonable",
        "track_model_init_kwargs",
        "build_model_config",
        "resolve_type_kwargs",
        "_init_models_dict",
        "_get_model_class",
        "_fixture_montage_chs",
        "_draw_chs_info",
        "get_summary_table",
    }
)

# Universal model-registry test files that dynamically exercise registered models.
CROSS_MODEL_TEST_FILES = frozenset(
    {
        "test/unit_tests/models/test_integration.py",
        "test/unit_tests/models/test_input_contract.py",
        "test/unit_tests/models/test_model_contract.py",
        "test/unit_tests/models/test_pretrained_compat.py",
        "test/unit_tests/models/test_config.py",
        "test/unit_tests/models/test_model_init.py",
        "test/unit_tests/models/test_huggingface.py",
        "test/unit_tests/models/test_return_features.py",
        "test/unit_tests/models/test_ssl_contract.py",
        "test/unit_tests/models/test_models.py",
        "test/unit_tests/models/test_model_categorization.py",
        "test/unit_tests/models/test_model_modality.py",
        "test/unit_tests/training/test_ch_mask_training.py",
        "test/unit_tests/test_docs_model_links.py",
    }
)

# Additional test files parametrized over subsets of models where editing test
# function logic must disable --select-models so all parametrized cases run.
PARAMETRIZED_MODEL_TEST_FILES = CROSS_MODEL_TEST_FILES | frozenset(
    {
        "test/unit_tests/models/test_foundation_models.py",
        "test/unit_tests/models/test_ch_mask.py",
        "test/unit_tests/models/test_on_non_divisible.py",
    }
)

DOCS_CONTRACT_TARGETS = frozenset(
    {
        "docs/_static/landing.js",
        "docs/models/models_table.rst",
        "docs/index.rst",
        "docs/api.rst",
        "docs/models/channel_strategies.rst",
        "docs/sphinxext/html_redirects.py",
    }
)

# Measured per-file CPU durations (seconds) on Ubuntu 2-vCPU GitHub runners;
# newly added or unlisted test files default to 0.3 s.
DEFAULT_FAST_CPU_SECONDS: dict[str, float] = {
    "test/unit_tests/models/test_integration.py": 559.69,
    "test/unit_tests/models/test_pretrained_compat.py": 232.12,
    "test/unit_tests/models/test_model_contract.py": 216.70,
    "test/unit_tests/models/test_models.py": 110.97,
    "test/unit_tests/models/test_foundation_models.py": 42.99,
    "test/unit_tests/models/test_return_features.py": 36.28,
    "test/unit_tests/models/test_huggingface.py": 29.79,
    "test/unit_tests/models/test_model_init.py": 27.78,
    "test/unit_tests/preprocessing/test_preprocess.py": 26.33,
    "test/unit_tests/datasets/test_zarr_to_memmap_concurrency.py": 24.78,
    "test/unit_tests/test_lazy_imports.py": 18.77,
    "test/unit_tests/samplers/test_samplers.py": 16.68,
    "test/unit_tests/models/test_codebrain_compile.py": 16.24,
    "test/unit_tests/models/test_config.py": 16.00,
    "test/unit_tests/training/test_ch_mask_training.py": 15.59,
    "test/unit_tests/preprocessing/test_mne_preprocessor.py": 15.10,
    "test/unit_tests/models/test_ch_mask.py": 14.01,
    "test/unit_tests/datautil/test_serialization.py": 13.22,
    "test/unit_tests/datasets/test_hub_integration.py": 9.68,
    "test/unit_tests/preprocessing/test_windowers.py": 8.69,
    "test/unit_tests/preprocessing/test_eegprep_preprocessor.py": 7.87,
    "test/unit_tests/models/test_guetschel2026.py": 7.61,
    "test/unit_tests/datasets/test_coalesced_reads.py": 7.32,
    "test/unit_tests/models/test_modules.py": 7.31,
    "test/unit_tests/models/test_eegconformer_checkpointing.py": 6.51,
    "test/unit_tests/modules/test_channels.py": 6.01,
    "test/unit_tests/modules/test_stat_acceleration_contract.py": 5.73,
    "test/unit_tests/models/test_neurorvq_ddp.py": 5.46,
    "test/acceptance_tests/test_variable_length_trials_decoding.py": 5.46,
    "test/unit_tests/models/test_base.py": 4.65,
    "test/unit_tests/augmentation/test_transforms.py": 4.11,
    "test/unit_tests/test_compile_estimator.py": 3.18,
    "test/unit_tests/models/test_brainomni.py": 2.69,
    "test/unit_tests/visualization/test_attribution.py": 2.61,
    "test/unit_tests/datautil/test_hub_formats.py": 2.54,
    "test/unit_tests/test_dependencies.py": 2.26,
    "test/unit_tests/models/test_util.py": 2.02,
    "test/unit_tests/test_eegneuralnet.py": 1.68,
    "test/unit_tests/test_conftest_selection.py": 1.61,
    "test/unit_tests/datasets/test_moabb.py": 1.61,
    "test/unit_tests/models/test_on_non_divisible.py": 1.57,
    "test/unit_tests/training/test_scoring.py": 1.35,
    "test/unit_tests/datasets/test_bids.py": 1.24,
    "test/unit_tests/models/test_config_numpydantic_fallback.py": 1.19,
    "test/unit_tests/datasets/test_dataset.py": 1.07,
    "test/unit_tests/models/test_meggpt.py": 1.01,
    "test/unit_tests/models/test_signal_jepa.py": 0.99,
    "test/unit_tests/training/test_trial_chunks.py": 0.95,
    "test/unit_tests/models/test_eegdino.py": 0.68,
    "test/unit_tests/datasets/test_bids_validation.py": 0.65,
    "test/unit_tests/training/test_streaming_lifecycle.py": 0.56,
    "test/unit_tests/datasets/test_mne.py": 0.54,
    "test/unit_tests/models/test_attention_exact.py": 0.51,
    "test/unit_tests/datasets/test_tuh.py": 0.50,
    "test/unit_tests/training/test_trial_aggregation.py": 0.46,
    "test/unit_tests/models/test_emg_pose_baselines.py": 0.41,
    "test/unit_tests/test_bench_model_inference.py": 0.39,
    "test/unit_tests/augmentation/test_base.py": 0.35,
}

MAX_SURGICAL_MODELS = 25
SINGLE_SHARD_MAX_WALL_S = 150.0


def _git_show(repo_git_dir: Path, rev: str, rel_path: str) -> str | None:
    res = subprocess.run(  # nosec B603, B607
        ["git", "-C", str(repo_git_dir), "show", f"{rev}:{rel_path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return res.stdout if res.returncode == 0 else None


def _path_to_module(rel_path: str) -> str | None:
    if not rel_path.endswith(".py"):
        return None
    parts = list(Path(rel_path).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def _resolve_relative_module(pkg_module: str, level: int, mod: str | None) -> str:
    pkg_parts = pkg_module.split(".")
    base_parts = pkg_parts[: len(pkg_parts) - (level - 1)] if level > 1 else pkg_parts
    if mod:
        return ".".join([*base_parts, mod])
    return ".".join(base_parts)


def classify_test_file_slow_tier(source: str) -> str:
    """Classify a test module as ``'none'``, ``'mixed_slow'``, or ``'slow_only'``.

    Files whose module-level ``pytestmark`` includes ``pytest.mark.network``
    follow ``--run-network`` alone (see ``test/conftest.py``) and return
    ``'none'``.
    """
    tree = ast.parse(source)

    def _marks_in_node(node: ast.AST) -> set[str]:
        found: set[str] = set()
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Attribute)
                and sub.attr in {"slow", "network"}
                and isinstance(sub.value, ast.Attribute)
                and sub.value.attr == "mark"
            ):
                found.add(sub.attr)
        return found

    module_marks: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            if any(
                isinstance(t, ast.Name) and t.id == "pytestmark" for t in stmt.targets
            ):
                module_marks |= _marks_in_node(stmt.value)

    if "network" in module_marks:
        return "none"
    if "slow" in module_marks:
        return "slow_only"

    all_marks = _marks_in_node(tree)
    if "slow" in all_marks:
        return "mixed_slow"
    return "none"


def parse_barrel_exports(
    source: str, barrel_module: str
) -> tuple[dict[str, str], set[str], str]:
    """Parse ``(symbol -> defining_module, __all__ set, runtime_signature)``."""
    tree = ast.parse(source)
    sym_to_mod: dict[str, str] = {}
    dunder_all: set[str] = set()
    runtime_stmts: list[str] = []

    def _collect_import_from(node: ast.ImportFrom) -> None:
        target_base = (
            _resolve_relative_module(barrel_module, node.level, node.module)
            if node.level > 0
            else (node.module or "")
        )
        for alias in node.names:
            exported = alias.asname or alias.name
            if node.module is None and node.level > 0:
                sym_to_mod[exported] = f"{target_base}.{alias.name}"
            else:
                sym_to_mod[exported] = target_base

    for idx, stmt in enumerate(tree.body):
        if (
            idx == 0
            and isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            continue
        if isinstance(stmt, ast.ImportFrom):
            _collect_import_from(stmt)
            continue
        if isinstance(stmt, ast.If):
            cond = ast.unparse(stmt.test).strip()
            if cond == "TYPE_CHECKING":
                for sub in stmt.body:
                    if isinstance(sub, ast.ImportFrom):
                        _collect_import_from(sub)
                for sub in stmt.orelse:
                    runtime_stmts.append(ast.unparse(sub))
                continue
        if isinstance(stmt, ast.Assign):
            targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
            if targets == ["__all__"] and isinstance(stmt.value, (ast.List, ast.Tuple)):
                for elt in stmt.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        dunder_all.add(elt.value)
                continue
            if targets == ["_SUBMODULES"]:
                for sub_node in ast.walk(stmt.value):
                    if isinstance(sub_node, ast.Constant) and isinstance(
                        sub_node.value, str
                    ):
                        sym_to_mod[sub_node.value] = f"{barrel_module}.{sub_node.value}"
                continue
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if stmt.target.id == "_LAZY" and stmt.value is None:
                continue
        runtime_stmts.append(ast.unparse(stmt))

    if barrel_module == "braindecode.datautil":
        sym_to_mod.setdefault("create_from_X_y", "braindecode.datasets.xy")
        sym_to_mod.setdefault("create_from_mne_raw", "braindecode.datasets.mne")
        sym_to_mod.setdefault("create_from_mne_epochs", "braindecode.datasets.mne")
        for n in (
            "exponential_moving_demean",
            "exponential_moving_standardize",
            "filterbank",
            "preprocess",
            "Preprocessor",
        ):
            sym_to_mod.setdefault(n, "braindecode.preprocessing.preprocess")
        for n in ("create_windows_from_events", "create_fixed_length_windows"):
            sym_to_mod.setdefault(n, "braindecode.preprocessing.windowers")

    return sym_to_mod, dunder_all, "\n".join(runtime_stmts)


def diff_models_util(
    base_src: str | None, head_src: str | None
) -> tuple[bool, set[str], set[str]]:
    """Compare BASE vs HEAD of ``braindecode/models/util.py``.

    Returns
    -------
    tuple[bool, set[str], set[str]]
        ``(global_fallback, changed_registry_models, changed_noncore_helpers)``
    """
    if base_src is None or head_src is None:
        return True, set(), set()

    def _split_util(
        src: str,
    ) -> tuple[dict[str, str] | None, dict[str, str] | None, str | None]:
        tree = ast.parse(src)
        reg_map: dict[str, str] = {}
        defs: dict[str, str] = {}
        other: list[str] = []
        for idx, stmt in enumerate(tree.body):
            if (
                idx == 0
                and isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            ):
                continue
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                other.append(ast.unparse(stmt))
                continue
            is_mmp = False
            val_node = None
            if (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and stmt.target.id == "models_mandatory_parameters"
            ):
                is_mmp = True
                val_node = stmt.value
            elif (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == "models_mandatory_parameters"
            ):
                is_mmp = True
                val_node = stmt.value
            if is_mmp and isinstance(val_node, ast.List):
                for elt in val_node.elts:
                    if (
                        isinstance(elt, ast.Tuple)
                        and len(elt.elts) >= 1
                        and isinstance(elt.elts[0], ast.Constant)
                        and isinstance(elt.elts[0].value, str)
                    ):
                        reg_map[elt.elts[0].value] = ast.unparse(elt)
                    else:
                        return None, None, None
                continue
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defs[stmt.name] = ast.unparse(stmt)
            elif (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
            ):
                defs[stmt.targets[0].id] = ast.unparse(stmt)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                defs[stmt.target.id] = ast.unparse(stmt)
            else:
                other.append(ast.unparse(stmt))
        return reg_map, defs, "\n".join(other)

    b_reg, b_defs, b_other = _split_util(base_src)
    h_reg, h_defs, h_other = _split_util(head_src)
    if (
        b_reg is None
        or h_reg is None
        or b_defs is None
        or h_defs is None
        or b_other != h_other
    ):
        return True, set(), set()

    changed_defs = {
        k for k in (set(b_defs) | set(h_defs)) if b_defs.get(k) != h_defs.get(k)
    }
    if changed_defs & UTIL_CORE_SYMBOLS:
        return True, set(), set()

    changed_models = {
        m for m in (set(b_reg) | set(h_reg)) if b_reg.get(m) != h_reg.get(m)
    }
    if changed_defs:
        for m, code_str in h_reg.items():
            if any(re.search(rf"\b{re.escape(fn)}\b", code_str) for fn in changed_defs):
                changed_models.add(m)

    return False, changed_models, changed_defs


def diff_summary_csv(
    base_csv: str | None, head_csv: str | None
) -> tuple[bool, set[str]]:
    """Return ``(schema_changed, changed_model_names)`` for ``summary.csv``."""
    if base_csv is None or head_csv is None:
        return True, set()
    b_reader = csv.DictReader(io.StringIO(base_csv.lstrip("\ufeff")))
    h_reader = csv.DictReader(io.StringIO(head_csv.lstrip("\ufeff")))
    if b_reader.fieldnames != h_reader.fieldnames or not h_reader.fieldnames:
        return True, set()
    b_rows = {
        r["Model"].strip(): {k: (v or "").strip() for k, v in r.items()}
        for r in b_reader
    }
    h_rows = {
        r["Model"].strip(): {k: (v or "").strip() for k, v in r.items()}
        for r in h_reader
    }
    changed = {m for m in (set(b_rows) | set(h_rows)) if b_rows.get(m) != h_rows.get(m)}
    return False, changed


def test_functions_changed(
    base_src: str | None, head_src: str | None, known_models: set[str]
) -> bool:
    """Return True if any function/fixture logic changed (ignoring model skip-lists)."""
    if base_src is None or head_src is None:
        return True

    class _StripModelListItems(ast.NodeTransformer):
        def _clean_elts(self, elts: list[ast.expr]) -> list[ast.expr]:
            out = []
            for e in elts:
                if isinstance(e, ast.Name) and e.id in known_models:
                    continue
                if (
                    isinstance(e, ast.Constant)
                    and isinstance(e.value, str)
                    and e.value in known_models
                ):
                    continue
                out.append(self.visit(e))
            return out

        def visit_List(self, node: ast.List) -> ast.AST:
            node.elts = self._clean_elts(node.elts)
            return node

        def visit_Set(self, node: ast.Set) -> ast.AST:
            node.elts = self._clean_elts(node.elts)
            return node

        def visit_Tuple(self, node: ast.Tuple) -> ast.AST:
            node.elts = self._clean_elts(node.elts)
            return node

    def _funcs(src: str) -> dict[str, str]:
        tree = _StripModelListItems().visit(ast.parse(src))
        out = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = node.body
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    node.body = body[1:] or [ast.Pass()]
                out[node.name] = ast.unparse(node)
        return out

    bf, hf = _funcs(base_src), _funcs(head_src)
    return bf != hf


def diff_test_file_models(
    base_src: str | None, head_src: str | None, known_models: set[str]
) -> set[str]:
    """Return known model names whose references or registry entries changed."""
    if base_src is None or head_src is None:
        return set()

    def _extract(src: str) -> tuple[dict[str, int], dict[str, list[str]]]:
        tree = ast.parse(src)
        counts: dict[str, int] = defaultdict(int)
        entries: dict[str, list[str]] = defaultdict(list)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in known_models:
                counts[node.id] += 1
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in known_models
            ):
                counts[node.value] += 1
            elif isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values):
                    key_name = None
                    if isinstance(k, ast.Name) and k.id in known_models:
                        key_name = k.id
                    elif (
                        isinstance(k, ast.Constant)
                        and isinstance(k.value, str)
                        and k.value in known_models
                    ):
                        key_name = k.value
                    if key_name is not None and v is not None:
                        entries[key_name].append(ast.unparse(v))
            elif isinstance(node, ast.Tuple) and len(node.elts) >= 2:
                first = node.elts[0]
                key_name = None
                if isinstance(first, ast.Name) and first.id in known_models:
                    key_name = first.id
                elif (
                    isinstance(first, ast.Constant)
                    and isinstance(first.value, str)
                    and first.value in known_models
                ):
                    key_name = first.value
                if key_name is not None:
                    entries[key_name].append(ast.unparse(node))
        return dict(counts), {k: sorted(v) for k, v in entries.items()}

    b_counts, b_entries = _extract(base_src)
    h_counts, h_entries = _extract(head_src)
    changed = {
        m
        for m in (set(b_counts) | set(h_counts))
        if b_counts.get(m, 0) != h_counts.get(m, 0)
    }
    changed |= {
        m
        for m in (set(b_entries) | set(h_entries))
        if b_entries.get(m) != h_entries.get(m)
    }
    return changed


def _discover_test_files(test_dir: Path, checkout_root: Path) -> list[str]:
    if not test_dir.is_dir():
        return []
    found = {
        p.relative_to(checkout_root).as_posix()
        for pat in ("test_*.py", "*_test.py")
        for p in test_dir.rglob(pat)
        if p.is_file()
    }
    return sorted(found)


class CodebaseGraph:
    """2D module + barrel-symbol dependency graph of a checkout tree."""

    def __init__(self, root: Path):
        self.root = root
        self.module_to_rel: dict[str, str] = {}
        self.rel_to_module: dict[str, str] = {}
        self.barrel_exports: dict[str, dict[str, str]] = {}
        self.rev_imports: dict[str, set[str]] = defaultdict(set)
        self.rev_symbol_imports: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.barrel_package_importers: dict[str, set[str]] = defaultdict(set)
        self.model_mod_to_classes: dict[str, set[str]] = defaultdict(set)
        self.known_model_names: set[str] = set()
        self.slow_only_files: set[str] = set()
        self.mixed_slow_files: set[str] = set()

        self._index()

    def _index(self) -> None:
        bd_files = (
            sorted((self.root / "braindecode").rglob("*.py"))
            if (self.root / "braindecode").is_dir()
            else []
        )
        test_files = (
            sorted((self.root / "test").rglob("*.py"))
            if (self.root / "test").is_dir()
            else []
        )
        py_files = bd_files + test_files

        for p in py_files:
            rel = p.relative_to(self.root).as_posix()
            mod = _path_to_module(rel)
            if mod:
                self.module_to_rel[mod] = rel
                self.rel_to_module[rel] = mod

        for p in sorted(
            (self.root / "braindecode").rglob("__init__.py")
            if (self.root / "braindecode").is_dir()
            else []
        ):
            rel = p.relative_to(self.root).as_posix()
            mod = _path_to_module(rel)
            if mod:
                sym_map, _, _ = parse_barrel_exports(p.read_text(encoding="utf-8"), mod)
                self.barrel_exports[mod] = sym_map

        models_lazy = self.barrel_exports.get("braindecode.models", {})
        non_model_exports = {
            "EEGModuleMixin",
            "TFMTokenizerOutput",
            "_init_models_dict",
            "build_model_config",
            "extract_channel_locations_from_chs_info",
            "models_mandatory_parameters",
            "positions_from_chs_info",
        }
        for sym, target_mod in models_lazy.items():
            if sym not in non_model_exports and target_mod != "braindecode.models.util":
                self.model_mod_to_classes[target_mod].add(sym)
                self.known_model_names.add(sym)

        for p in py_files:
            rel = p.relative_to(self.root).as_posix()
            cur_mod = self.rel_to_module[rel]
            is_pkg_init = p.name == "__init__.py"
            cur_pkg = (
                cur_mod
                if is_pkg_init
                else (cur_mod.rsplit(".", 1)[0] if "." in cur_mod else "")
            )
            src = p.read_text(encoding="utf-8")

            if rel.startswith("test/") and (
                p.name.startswith("test_") or p.name.endswith("_test.py")
            ):
                tier = classify_test_file_slow_tier(src)
                if tier == "slow_only":
                    self.slow_only_files.add(rel)
                elif tier == "mixed_slow":
                    self.mixed_slow_files.add(rel)

            # Do not add static edges out of barrel __init__.py re-exports:
            # barrel re-exports are already resolved symbol-by-symbol in barrel_exports.
            if is_pkg_init and cur_mod in self.barrel_exports:
                continue

            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith(("braindecode", "test")):
                            self._add_mod_dep(rel, alias.name)
                            if (
                                rel.startswith("test/")
                                and alias.name != "braindecode"
                                and alias.name in self.barrel_exports
                            ):
                                self.barrel_package_importers[alias.name].add(rel)
                elif isinstance(node, ast.ImportFrom):
                    if node.level > 0:
                        base_mod = _resolve_relative_module(
                            cur_pkg, node.level, node.module
                        )
                    else:
                        base_mod = node.module or ""
                    if not base_mod.startswith(("braindecode", "test")):
                        continue
                    barrel_map = self.barrel_exports.get(base_mod)
                    for alias in node.names:
                        sym = alias.name
                        self.rev_symbol_imports[(base_mod, sym)].add(rel)
                        submod_candidate = f"{base_mod}.{sym}"
                        resolved_target = (
                            barrel_map[sym]
                            if (barrel_map and sym in barrel_map)
                            else None
                        )
                        if resolved_target:
                            self._add_mod_dep(rel, resolved_target)
                            self.rev_symbol_imports[(resolved_target, sym)].add(rel)
                            if (
                                rel.startswith("test/")
                                and resolved_target != "braindecode"
                                and resolved_target in self.barrel_exports
                            ):
                                self.barrel_package_importers[resolved_target].add(rel)
                        elif submod_candidate in self.module_to_rel:
                            self._add_mod_dep(rel, submod_candidate)
                            if (
                                rel.startswith("test/")
                                and submod_candidate != "braindecode"
                                and submod_candidate in self.barrel_exports
                            ):
                                self.barrel_package_importers[submod_candidate].add(rel)
                        else:
                            self._add_mod_dep(rel, base_mod)

    def _add_mod_dep(self, importer_rel: str, target_mod: str) -> None:
        parts = target_mod.split(".")
        for i in range(len(parts), 0, -1):
            cand = ".".join(parts[:i])
            if cand in self.module_to_rel:
                if self.module_to_rel[cand] != importer_rel:
                    self.rev_imports[cand].add(importer_rel)
                break


def compute_plan(
    checkout_root: Path,
    git_dir: Path,
    base_rev: str | None,
    head_rev: str | None,
    durations_map: dict[str, dict[str, float]] | None = None,
    force_full: bool = False,
) -> dict[str, Any]:
    """Compute the 2D CI test plan for ``base_rev...head_rev``."""
    graph = CodebaseGraph(checkout_root)
    durations = durations_map or {}

    all_test_files = _discover_test_files(checkout_root / "test", checkout_root)
    slow_tier_set = graph.slow_only_files | graph.mixed_slow_files
    all_suite_files = [f for f in all_test_files if f not in graph.slow_only_files]
    all_slow_files = sorted(f for f in all_test_files if f in slow_tier_set)

    env_kill_switch = (
        os.environ.get("BRAINDECODE_CI_SELECTION", "").strip().lower() == "full"
    )
    if env_kill_switch or force_full or not base_rev or not head_rev:
        reason = (
            "BRAINDECODE_CI_SELECTION=full kill-switch active."
            if env_kill_switch
            else "Full suite requested (--full / push / schedule)."
        )
        return _pack_plan(
            full_suite=True,
            suite_files=all_suite_files,
            slow_files=all_slow_files,
            select_models=set(),
            run_acceptance=True,
            run_docs=True,
            reason=reason,
            durations_map=durations,
        )

    diff_proc = subprocess.run(  # nosec B603, B607
        [
            "git",
            "-C",
            str(git_dir),
            "diff",
            "--name-status",
            "--find-renames",
            f"{base_rev}...{head_rev}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if diff_proc.returncode != 0:
        diff_proc = subprocess.run(  # nosec B603, B607
            [
                "git",
                "-C",
                str(git_dir),
                "diff",
                "--name-status",
                "--find-renames",
                base_rev,
                head_rev,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    if diff_proc.returncode != 0:
        return _pack_plan(
            full_suite=True,
            suite_files=all_suite_files,
            slow_files=all_slow_files,
            select_models=set(),
            run_acceptance=True,
            run_docs=True,
            reason="Fallback to full suite (git diff failed).",
            durations_map=durations,
        )

    changes: list[tuple[str, str, str | None]] = []
    for line in diff_proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        if status.startswith(("R", "C")) and len(parts) >= 3:
            changes.append((status[0], parts[1], parts[2]))
        else:
            changes.append((status[0], parts[1], None))

    if not changes:
        return _pack_plan(
            full_suite=False,
            suite_files=[],
            slow_files=[],
            select_models=set(),
            run_acceptance=False,
            run_docs=False,
            reason="Empty diff.",
            durations_map=durations,
        )

    all_touched_paths: set[str] = set()
    deleted_paths: set[str] = set()
    active_changed_paths: list[tuple[str, str]] = []
    for st, p1, p2 in changes:
        all_touched_paths.add(p1)
        if st == "D":
            deleted_paths.add(p1)
        elif st in ("R", "C") and p2 is not None:
            all_touched_paths.add(p2)
            if st == "R":
                deleted_paths.add(p1)
            active_changed_paths.append(("A", p2))
        else:
            active_changed_paths.append((st, p1))

    run_docs = any(
        p.startswith(("docs/", "examples/", "braindecode/"))
        or p in ("pyproject.toml", "uv.lock", ".github/workflows/docs.yml")
        for p in all_touched_paths
    )

    for p in sorted(all_touched_paths):
        if p in GLOBAL_FALLBACK_EXACT or p.startswith(GLOBAL_FALLBACK_PREFIXES):
            return _pack_plan(
                full_suite=True,
                suite_files=all_suite_files,
                slow_files=all_slow_files,
                select_models=set(),
                run_acceptance=True,
                run_docs=True,
                reason=f"Global trigger modified: {p}",
                durations_map=durations,
            )

    dirty_modules: set[str] = set()
    impacted_tests: set[str] = set()
    impacted_models: set[str] = set()
    disable_model_filter = False
    reasons: list[str] = []

    # 1. Handle deleted paths (including old paths of renames)
    for dp in sorted(deleted_paths):
        if dp.startswith("braindecode/") and dp.endswith(".py"):
            old_mod = _path_to_module(dp)
            if old_mod:
                dirty_modules.add(old_mod)
            impacted_tests.update(
                {
                    "test/unit_tests/test_lazy_imports.py",
                    "test/unit_tests/test_dependencies.py",
                }
            )
            if dp.startswith("braindecode/models/"):
                disable_model_filter = True
                impacted_tests.update(CROSS_MODEL_TEST_FILES)
                reasons.append(f"deleted model module {dp}")

    # 2. Handle active changed paths
    for st, path in active_changed_paths:
        if path.startswith("test/") and path.endswith("/conftest.py"):
            parent = Path(path).parent
            impacted_tests.update(
                _discover_test_files(checkout_root / parent, checkout_root)
            )
            continue

        if path.startswith("test/") and path.endswith(".py"):
            fname = Path(path).name
            if fname.startswith("test_") or fname.endswith("_test.py"):
                impacted_tests.add(path)
                if path in PARAMETRIZED_MODEL_TEST_FILES:
                    base_s = _git_show(git_dir, base_rev, path)
                    head_s = _git_show(git_dir, head_rev, path)
                    if st == "A" or test_functions_changed(
                        base_s, head_s, graph.known_model_names
                    ):
                        disable_model_filter = True
                        reasons.append(f"test logic edited in {fname}")
                    else:
                        impacted_models.update(
                            diff_test_file_models(
                                base_s, head_s, graph.known_model_names
                            )
                        )
            else:
                mod = _path_to_module(path)
                if mod:
                    dirty_modules.add(mod)
            continue

        if path == "braindecode/models/summary.csv":
            b_csv = _git_show(git_dir, base_rev, path)
            h_csv = _git_show(git_dir, head_rev, path)
            schema_chg, csv_models = diff_summary_csv(b_csv, h_csv)
            if schema_chg:
                disable_model_filter = True
            else:
                impacted_models.update(csv_models)
            impacted_tests.update(
                {
                    "test/unit_tests/models/test_input_contract.py",
                    "test/unit_tests/models/test_model_categorization.py",
                    "test/unit_tests/models/test_model_modality.py",
                    "test/unit_tests/models/test_integration.py",
                    "test/unit_tests/models/test_pretrained_compat.py",
                    "test/unit_tests/models/test_model_contract.py",
                    "test/unit_tests/test_lazy_imports.py",
                }
            )
            continue

        if path == "braindecode/models/model.html.jinja":
            impacted_tests.update(
                {
                    "test/unit_tests/models/test_huggingface.py",
                    "test/unit_tests/models/test_base.py",
                }
            )
            impacted_models.update({"EEGNet", "BIOT"})
            continue

        if path.startswith("test/unit_tests/models/data/"):
            impacted_tests.update(
                {
                    "test/unit_tests/models/test_base.py",
                    "test/unit_tests/models/test_foundation_models.py",
                }
            )
            continue

        if path in (
            "scripts/monthly_release.py",
            ".github/workflows/release.yml",
            ".github/workflows/monthly-release.yml",
            "docs/_static/versions.json",
        ):
            impacted_tests.add("test/unit_tests/test_version.py")
            continue

        if path == "scripts/check_codebrain_weights.py":
            impacted_tests.add("test/unit_tests/models/test_codebrain_compile.py")
            impacted_models.add("CodeBrain")
            continue

        if path == "benchmarks/bench_distributed_windows.py":
            impacted_tests.add("test/integration_tests/test_distributed_windows.py")
            continue

        if path in DOCS_CONTRACT_TARGETS:
            impacted_tests.add("test/unit_tests/test_docs_model_links.py")
            continue

        if (
            path.startswith("braindecode/")
            and path.endswith("/__init__.py")
            or path == "braindecode/__init__.py"
        ):
            barrel_mod = _path_to_module(path) or "braindecode"
            b_src = _git_show(git_dir, base_rev, path)
            h_src = _git_show(git_dir, head_rev, path)
            impacted_tests.update(
                {
                    "test/unit_tests/test_lazy_imports.py",
                    "test/unit_tests/test_dependencies.py",
                }
            )
            impacted_tests.update(graph.barrel_package_importers.get(barrel_mod, ()))
            if barrel_mod == "braindecode.preprocessing":
                impacted_tests.update(
                    {
                        "test/unit_tests/preprocessing/test_preprocess.py",
                        "test/unit_tests/preprocessing/test_mne_preprocessor.py",
                    }
                )
            if b_src is None or h_src is None:
                dirty_modules.add(barrel_mod)
                if barrel_mod in ("braindecode", "braindecode.models"):
                    return _pack_plan(
                        full_suite=True,
                        suite_files=all_suite_files,
                        slow_files=all_slow_files,
                        select_models=set(),
                        run_acceptance=True,
                        run_docs=True,
                        reason=f"Barrel {path} added/removed.",
                        durations_map=durations,
                    )
                continue
            b_map, b_all, b_rt = parse_barrel_exports(b_src, barrel_mod)
            h_map, h_all, h_rt = parse_barrel_exports(h_src, barrel_mod)
            if b_rt != h_rt:
                if barrel_mod in ("braindecode", "braindecode.models"):
                    return _pack_plan(
                        full_suite=True,
                        suite_files=all_suite_files,
                        slow_files=all_slow_files,
                        select_models=set(),
                        run_acceptance=True,
                        run_docs=True,
                        reason=f"Runtime logic in {path} modified.",
                        durations_map=durations,
                    )
                dirty_modules.add(barrel_mod)
            else:
                changed_syms = {
                    s for s in (set(b_map) | set(h_map)) if b_map.get(s) != h_map.get(s)
                } | (b_all ^ h_all)
                for s in changed_syms:
                    for m_target in (b_map.get(s), h_map.get(s)):
                        if m_target:
                            dirty_modules.add(m_target)
                    for imp_rel in graph.rev_symbol_imports.get((barrel_mod, s), ()):
                        if imp_rel.startswith("test/"):
                            impacted_tests.add(imp_rel)
                        elif imp_rel in graph.rel_to_module:
                            dirty_modules.add(graph.rel_to_module[imp_rel])
            continue

        if path == "braindecode/models/util.py":
            b_src = _git_show(git_dir, base_rev, path)
            h_src = _git_show(git_dir, head_rev, path)
            is_global, util_models, util_helpers = diff_models_util(b_src, h_src)
            if is_global:
                return _pack_plan(
                    full_suite=True,
                    suite_files=all_suite_files,
                    slow_files=all_slow_files,
                    select_models=set(),
                    run_acceptance=True,
                    run_docs=True,
                    reason="Core helper modified in braindecode/models/util.py.",
                    durations_map=durations,
                )
            impacted_tests.add("test/unit_tests/models/test_util.py")
            impacted_tests.add("test/unit_tests/models/test_model_rules.py")
            impacted_tests.add("test/unit_tests/test_no_error_catching.py")
            impacted_tests.add("test/unit_tests/test_dependencies.py")
            if util_models:
                impacted_models.update(util_models)
                impacted_tests.update(CROSS_MODEL_TEST_FILES)
            for helper_name in util_helpers:
                for imp_rel in graph.rev_symbol_imports.get(
                    ("braindecode.models.util", helper_name), ()
                ):
                    if imp_rel.startswith("test/"):
                        impacted_tests.add(imp_rel)
                    elif imp_rel in graph.rel_to_module:
                        dirty_modules.add(graph.rel_to_module[imp_rel])
            continue

        if path.startswith("braindecode/") and path.endswith(".py"):
            mod = _path_to_module(path)
            if mod:
                dirty_modules.add(mod)
            if path.startswith(
                (
                    "braindecode/models/",
                    "braindecode/modules/",
                    "braindecode/functional/",
                )
            ):
                impacted_tests.add("test/unit_tests/test_no_error_catching.py")
            impacted_tests.add("test/unit_tests/test_dependencies.py")
            continue

    # 3. Transitive reverse-dependency closure over dirty_modules
    visited_mods: set[str] = set()
    queue = list(dirty_modules)
    while queue:
        m = queue.pop()
        if m in visited_mods:
            continue
        visited_mods.add(m)
        if m in graph.model_mod_to_classes:
            impacted_models.update(graph.model_mod_to_classes[m])
            impacted_tests.update(CROSS_MODEL_TEST_FILES)
        for pkg_name, pkg_importers in graph.barrel_package_importers.items():
            if m == pkg_name or m.startswith(f"{pkg_name}."):
                impacted_tests.update(pkg_importers)
        for imp_rel in graph.rev_imports.get(m, ()):
            if imp_rel == "braindecode/models/util.py" and m.startswith(
                "braindecode.models."
            ):
                continue
            if imp_rel.startswith("test/"):
                fname = Path(imp_rel).name
                if fname.startswith("test_") or fname.endswith("_test.py"):
                    impacted_tests.add(imp_rel)
                elif imp_rel in graph.rel_to_module:
                    queue.append(graph.rel_to_module[imp_rel])
            elif imp_rel in graph.rel_to_module:
                queue.append(graph.rel_to_module[imp_rel])

    impacted_tests = {
        f
        for f in impacted_tests
        if f not in deleted_paths and (checkout_root / f).is_file()
    }

    run_acceptance = any(f.startswith("test/acceptance_tests/") for f in impacted_tests)

    slow_selected = sorted(f for f in impacted_tests if f in slow_tier_set)
    suite_selected = sorted(
        f
        for f in impacted_tests
        if f not in graph.slow_only_files
        and f != "test/integration_tests/test_pretrained_hub_models.py"
        and not (
            f.startswith("test/acceptance_tests/")
            and f
            in {
                "test/acceptance_tests/test_cropped_decoding.py",
                "test/acceptance_tests/test_eeg_classifier.py",
                "test/acceptance_tests/test_trialwise_decoding.py",
            }
        )
    )

    if len(impacted_models) > MAX_SURGICAL_MODELS:
        return _pack_plan(
            full_suite=True,
            suite_files=all_suite_files,
            slow_files=all_slow_files,
            select_models=set(),
            run_acceptance=True,
            run_docs=run_docs,
            reason=(
                f"{len(impacted_models)} models impacted (> {MAX_SURGICAL_MODELS}); "
                "promoting to 4-shard full suite."
            ),
            durations_map=durations,
        )

    active_select_models = set() if disable_model_filter else impacted_models
    summary_reason = (
        "; ".join(reasons)
        if reasons
        else (
            f"Surgical model run ({','.join(sorted(active_select_models))})"
            if active_select_models
            else (
                f"Surgical file run ({len(suite_selected)} default + "
                f"{len(slow_selected)} slow files)"
            )
        )
    )
    if not suite_selected and not slow_selected:
        summary_reason = "No Python/test targets affected by diff."

    return _pack_plan(
        full_suite=False,
        suite_files=suite_selected,
        slow_files=slow_selected,
        select_models=active_select_models,
        run_acceptance=run_acceptance,
        run_docs=run_docs,
        reason=summary_reason,
        durations_map=durations,
    )


def _pack_plan(
    *,
    full_suite: bool,
    suite_files: list[str],
    slow_files: list[str],
    select_models: set[str],
    run_acceptance: bool,
    run_docs: bool,
    reason: str,
    durations_map: dict[str, dict[str, float]],
) -> dict[str, Any]:
    if not suite_files and not slow_files:
        return {
            "full_suite": False,
            "run_tests": False,
            "shard_count": 0,
            "shards": [],
            "shard_files": {},
            "slow_files_by_shard": {},
            "select_models": "",
            "run_acceptance": run_acceptance,
            "run_docs": run_docs,
            "est_max_wall_s": 0.0,
            "reason": reason,
        }

    model_scale = (
        min(1.0, (len(select_models | {"EEGNet"}) + 1.0) / 103.0)
        if select_models
        else 1.0
    )

    def _est_cpu(f: str) -> float:
        base = durations_map.get(f, {}).get(
            "fast_cpu", DEFAULT_FAST_CPU_SECONDS.get(f, 0.3)
        )
        if f == "test/unit_tests/models/test_integration.py":
            base = max(10.0, base - 53.5)
        elif f == "test/unit_tests/models/test_config.py":
            base = max(1.0, base - 14.6)
        if select_models and f in PARAMETRIZED_MODEL_TEST_FILES:
            return max(3.0, base * model_scale)
        return base

    slow_serial_s = 22.5 * len(slow_files) if slow_files else 0.0
    total_xdist_cpu = sum(_est_cpu(f) for f in suite_files)
    est_single_shard_wall = total_xdist_cpu / 2.0 + slow_serial_s + 8.0

    shard_count = (
        4 if (full_suite or est_single_shard_wall > SINGLE_SHARD_MAX_WALL_S) else 1
    )
    shards = list(range(shard_count))
    shard_files: dict[str, list[str]] = {str(s): [] for s in shards}
    slow_by_shard: dict[str, list[str]] = {str(s): [] for s in shards}

    if shard_count == 1:
        shard_files["0"] = list(suite_files)
        slow_by_shard["0"] = list(slow_files)
        est_max_wall = est_single_shard_wall
    else:
        # LPT balancing across 4 shards; place slow_files on Shard 3 and keep
        # mixed default+slow files on the same shard so shard file-unions are
        # strictly pairwise disjoint across shards 0..3.
        loads = [0.0] * shard_count
        slow_set = set(slow_files)
        slow_shard_idx = shard_count - 1
        if slow_files:
            slow_by_shard[str(slow_shard_idx)] = list(slow_files)
            loads[slow_shard_idx] += slow_serial_s

        pinned_mixed = [f for f in suite_files if f in slow_set]
        for f in pinned_mixed:
            shard_files[str(slow_shard_idx)].append(f)
            loads[slow_shard_idx] += _est_cpu(f) / 2.0

        remaining = [f for f in suite_files if f not in slow_set]
        for f in sorted(remaining, key=lambda x: (-_est_cpu(x), x)):
            best = min(range(shard_count), key=lambda s: loads[s])
            shard_files[str(best)].append(f)
            loads[best] += _est_cpu(f) / 2.0
        est_max_wall = max(loads) + 10.0

    return {
        "full_suite": full_suite,
        "run_tests": True,
        "shard_count": shard_count,
        "shards": shards,
        "shard_files": shard_files,
        "slow_files_by_shard": slow_by_shard,
        "select_models": ",".join(sorted(select_models)),
        "run_acceptance": run_acceptance,
        "run_docs": run_docs,
        "est_max_wall_s": round(est_max_wall, 1),
        "reason": reason,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--git-dir", type=Path, default=None)
    parser.add_argument("--base", type=str, default=None)
    parser.add_argument("--head", type=str, default="HEAD")
    parser.add_argument("--durations", type=Path, default=None)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--github-output", type=Path, default=None)
    args = parser.parse_args(argv)

    root = args.root.resolve()
    git_dir = (args.git_dir or root).resolve()
    durations = (
        json.loads(args.durations.read_text(encoding="utf-8"))
        if args.durations and args.durations.is_file()
        else None
    )
    plan = compute_plan(
        checkout_root=root,
        git_dir=git_dir,
        base_rev=args.base,
        head_rev=args.head,
        durations_map=durations,
        force_full=args.full,
    )

    if args.github_output:
        with args.github_output.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(f"full_suite={str(plan['full_suite']).lower()}\n")
            fh.write(f"run_tests={str(plan['run_tests']).lower()}\n")
            fh.write(f"shard_count={plan['shard_count']}\n")
            fh.write(f"shards={json.dumps(plan['shards'])}\n")
            fh.write(f"select_models={plan['select_models']}\n")
            fh.write(f"run_acceptance={str(plan['run_acceptance']).lower()}\n")
            fh.write(f"run_docs={str(plan['run_docs']).lower()}\n")
            fh.write(f"plan_json={json.dumps(plan, separators=(',', ':'))}\n")

    if args.pretty:
        print(json.dumps(plan, indent=2))
    else:
        print(json.dumps(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
