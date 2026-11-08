"""Text and HTML summaries of a model and of the checkpoint it was loaded from.

:meth:`EEGModuleMixin.from_pretrained` records which parameters came from the
checkpoint and reads the checkpoint's model card; the helpers below turn that
into a footer for ``print(model)`` and into the notebook view
(``model._repr_html_``): the signal geometry, the Hub repository and revision,
the datasets of the card, and the module tree with the parameters of each module
marked as loaded or new.
"""

# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

from __future__ import annotations

import html
import re

from torch.nn.parameter import UninitializedParameter

# A Hub card lists datasets by id; these patterns give the page to link to.
_DATASET_URLS = (
    (re.compile(r"^ds\d{6}$"), "https://openneuro.org/datasets/{}"),
    (re.compile(r"^(nm|on)\d{6}$"), "https://nemar.org/dataset/{}"),
    (re.compile(r"^[\w.-]+/[\w.-]+$"), "https://huggingface.co/datasets/{}"),
)

_CSS = """
.bd-model{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:13px;
 line-height:1.45;color:inherit;border:1px solid rgba(128,128,128,.35);border-radius:8px;
 padding:10px 14px;max-width:880px;margin:4px 0}
.bd-model a{color:#2f6fbf;text-decoration:none}.bd-model a:hover{text-decoration:underline}
.bd-model .bd-title{font-size:15px;font-weight:600}
.bd-model .bd-pill{display:inline-block;border-radius:10px;padding:0 8px;margin-left:6px;
 font-size:11px;font-weight:600;vertical-align:1px}
.bd-model .bd-loaded{background:rgba(46,160,67,.18);color:#1a7f37}
.bd-model .bd-new{background:rgba(212,130,0,.2);color:#9a6700}
.bd-model .bd-mixed{background:rgba(47,111,191,.18);color:#2f6fbf}
.bd-model .bd-grid{display:grid;grid-template-columns:max-content 1fr;gap:2px 14px;margin:8px 0}
.bd-model .bd-grid>div:nth-child(odd){opacity:.7}
.bd-model .bd-chip{display:inline-block;border:1px solid rgba(128,128,128,.4);border-radius:12px;
 padding:0 8px;margin:1px 4px 1px 0;font-size:12px}
.bd-model .bd-bar{display:flex;height:8px;border-radius:4px;overflow:hidden;margin:6px 0 2px;
 background:rgba(128,128,128,.15)}
.bd-model .bd-bar span{display:block;height:100%}
.bd-model .bd-bar .bd-l{background:#2ea043}.bd-model .bd-bar .bd-n{background:#d48200}
.bd-model .bd-note{font-size:12px;opacity:.75}
.bd-model details{margin:0}.bd-model summary{cursor:pointer}
.bd-model .bd-tree{margin-top:8px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
 font-size:12px}
.bd-model .bd-tree details>div,.bd-model .bd-tree details>details{margin-left:16px}
.bd-model .bd-row{display:flex;gap:8px;align-items:baseline;padding:1px 0}
.bd-model .bd-row .bd-type{opacity:.6}.bd-model .bd-row .bd-n{margin-left:auto;opacity:.75}
"""


def format_count(n: int) -> str:
    """Write a parameter count with a unit: ``3.2M``, ``850.1K``."""
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= size:
            return f"{n / size:.1f}{unit}"
    return str(n)


def _numel(param) -> int:
    """Size of a parameter; 0 for a lazy one that is not built yet."""
    return 0 if isinstance(param, UninitializedParameter) else param.numel()


def _percent(part: int, total: int) -> str:
    """``part / total`` in percent, never shown as 100 when a part is missing."""
    value = 100 * part / max(total, 1)
    if part < total:
        value = min(value, 99.9)
    return f"{value:.1f}%"


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def _dataset_chip(name: str) -> str:
    for pattern, url in _DATASET_URLS:
        if pattern.match(name):
            href = _esc(url.format(name))
            return (
                f'<a class="bd-chip" href="{href}" target="_blank" rel="noopener">'
                f"{_esc(name)}</a>"
            )
    return f'<span class="bd-chip">{_esc(name)}</span>'


def _param_origin(model):
    """Return ``{parameter name: (numel, loaded)}``, or None when unknown.

    ``loaded`` is True for a parameter whose values came from the checkpoint
    and False for one that was initialised, e.g. a head rebuilt for a new
    ``n_outputs``.
    """
    provenance = model.__dict__.get("_pretrained_provenance")
    if not provenance or provenance.get("loaded") is None:
        return None
    named = list(model.named_parameters(remove_duplicate=False))
    # A tied parameter is saved once, under one of its names.
    loaded_ids = {id(p) for name, p in named if name in provenance["loaded"]}
    return {
        name: (_numel(param), id(param) in loaded_ids)
        for name, param in model.named_parameters()
    }


def _status(n_loaded: int, n_total: int) -> tuple[str, str]:
    if n_loaded == n_total:
        return "bd-loaded", "pretrained"
    if n_loaded == 0:
        return "bd-new", "new"
    return "bd-mixed", f"{_percent(n_loaded, n_total)} pretrained"


def _module_counts(origin, prefix: str) -> tuple[int, int]:
    """Parameters (total, loaded) under the module named ``prefix``."""
    total = loaded = 0
    for name, (numel, is_loaded) in origin.items():
        if not prefix or name == prefix or name.startswith(prefix + "."):
            total += numel
            loaded += numel if is_loaded else 0
    return total, loaded


def _tree_html(module, name, prefix, origin, depth, max_depth) -> str:
    """Nested ``<details>`` of the module tree, with the parameter counts."""
    if origin is None:
        total = sum(_numel(p) for p in module.parameters())
        pill = ""
    else:
        total, loaded = _module_counts(origin, prefix)
        pill = ""
        if total:
            css, text = _status(loaded, total)
            pill = f'<span class="bd-pill {css}">{_esc(text)}</span>'
    row = (
        f'<span>{_esc(name)}</span><span class="bd-type">{_esc(type(module).__name__)}'
        f'</span>{pill}<span class="bd-n">{_esc(format_count(total))}</span>'
    )
    children = list(module.named_children())
    if not children or depth >= max_depth:
        return f'<div class="bd-row">{row}</div>'
    inner = "".join(
        _tree_html(
            child,
            child_name,
            f"{prefix}.{child_name}" if prefix else child_name,
            origin,
            depth + 1,
            max_depth,
        )
        for child_name, child in children
    )
    is_open = " open" if depth == 0 else ""
    return (
        f'<details{is_open}><summary class="bd-row" style="display:list-item">'
        f'<div class="bd-row" style="display:inline-flex;width:calc(100% - 20px)">'
        f"{row}</div></summary>{inner}</details>"
    )


def _geometry(model) -> list[tuple[str, str]]:
    rows = []
    for attr, label in (
        ("n_chans", "Channels"),
        ("n_times", "Samples"),
        ("sfreq", "Sampling rate"),
        ("input_window_seconds", "Window"),
        ("n_outputs", "Outputs"),
    ):
        try:
            value = getattr(model, attr)
        except (ValueError, AttributeError):
            continue
        if value is None:
            continue
        if attr == "sfreq":
            value = f"{value:g} Hz"
        elif attr == "input_window_seconds":
            value = f"{value:g} s"
        rows.append((label, str(value)))
    return rows


def _commit_link(provenance) -> str:
    repo = provenance.get("repo_id")
    commit = provenance.get("commit") or provenance.get("revision")
    if provenance.get("local"):
        return _esc(repo)
    link = f'<a href="https://huggingface.co/{_esc(repo)}" target="_blank">{_esc(repo)}</a>'
    if commit:
        tree = f"https://huggingface.co/{_esc(repo)}/tree/{_esc(commit)}"
        link += f' @ <a href="{tree}" target="_blank">{_esc(str(commit)[:7])}</a>'
    return link


def pretrained_summary_lines(model) -> list[str]:
    """Lines added below ``print(model)`` for a model from :meth:`from_pretrained`."""
    provenance = model.__dict__.get("_pretrained_provenance")
    if not provenance:
        return []
    source = str(provenance.get("repo_id"))
    commit = provenance.get("commit") or provenance.get("revision")
    if commit and not provenance.get("local"):
        source += f" @ {str(commit)[:7]}"
    lines = [f"Pretrained weights: {source}"]
    card = model.card_data
    datasets = getattr(card, "datasets", None) if card is not None else None
    lines.append(
        "Trained on (model card): "
        + (", ".join(datasets) if datasets else "not listed")
    )
    origin = _param_origin(model)
    if origin:
        total = sum(n for n, _ in origin.values())
        loaded = sum(n for n, is_loaded in origin.values() if is_loaded)
        line = (
            f"Loaded from the checkpoint: {loaded:,} of {total:,} parameters"
            f" ({_percent(loaded, total)})"
        )
        new = _new_modules(origin)
        if new:
            line += "; new: " + ", ".join(new)
        lines.append(line)
    return lines


def _new_modules(origin, limit: int = 6) -> list[str]:
    """Shortest module prefixes that cover the new parameters."""
    new = sorted(name for name, (_, loaded) in origin.items() if not loaded)
    loaded = [name for name, (_, is_loaded) in origin.items() if is_loaded]
    prefixes: list[str] = []
    for name in new:
        parts = name.split(".")
        # Climb while the parent module holds no loaded parameter.
        best = name
        for i in range(len(parts) - 1, 0, -1):
            parent = ".".join(parts[:i])
            if any(k == parent or k.startswith(parent + ".") for k in loaded):
                break
            best = parent
        if not any(best == p or best.startswith(p + ".") for p in prefixes):
            prefixes.append(best)
    if len(prefixes) > limit:
        return prefixes[:limit] + [f"... ({len(prefixes) - limit} more)"]
    return prefixes


def model_html(model, max_depth: int = 4) -> str:
    """HTML view of a braindecode model for notebooks and the docs gallery."""
    n_params = sum(_numel(p) for p in model.parameters())
    n_trainable = sum(_numel(p) for p in model.parameters() if p.requires_grad)
    n_bytes = sum(
        _numel(p) and _numel(p) * p.element_size() for p in model.parameters()
    )
    provenance = model.__dict__.get("_pretrained_provenance")
    origin = _param_origin(model)
    name = type(model).__name__
    docs = getattr(model, "_hub_mixin_info", None)
    docs_url = getattr(docs, "docs_url", None) if docs is not None else None

    title = f'<span class="bd-title">{_esc(name)}</span>'
    if docs_url:
        title = f'<a href="{_esc(docs_url)}" target="_blank">{title}</a>'
    if provenance:
        title += '<span class="bd-pill bd-loaded">pretrained</span>'

    grid = [
        (
            "Parameters",
            f"{n_params:,} ({format_count(n_params)}, {n_bytes / 2**20:.1f} MiB)",
        )
    ]
    if n_trainable != n_params:
        grid.append(("Trainable", f"{n_trainable:,}"))
    grid += _geometry(model)

    parts = [f"<style>{_CSS}</style>", '<div class="bd-model">', f"<div>{title}</div>"]
    card = model.card_data
    if provenance:
        grid.append(("Weights", _commit_link(provenance)))
        license_ = getattr(card, "license", None) if card is not None else None
        if license_:
            grid.append(("License", _esc(license_)))
        base = getattr(card, "base_model", None) if card is not None else None
        if base:
            bases = [base] if isinstance(base, str) else list(base)
            grid.append(
                (
                    "Fine-tuned from",
                    ", ".join(
                        f'<a href="https://huggingface.co/{_esc(b)}" target="_blank">'
                        f"{_esc(b)}</a>"
                        for b in bases
                    ),
                )
            )
        datasets = getattr(card, "datasets", None) if card is not None else None
        if datasets:
            chips = "".join(_dataset_chip(d) for d in datasets)
        else:
            chips = '<span class="bd-note">not listed on the model card</span>'
        grid.append(("Trained on", chips))
    cells = "".join(
        f"<div>{_esc(label)}</div><div>{value if label in _RAW else _esc(value)}</div>"
        for label, value in grid
    )
    parts.append(f'<div class="bd-grid">{cells}</div>')

    if origin:
        total = sum(n for n, _ in origin.values()) or 1
        loaded = sum(n for n, is_loaded in origin.values() if is_loaded)
        frac = 100 * loaded / total
        parts.append(
            f'<div class="bd-bar" title="{_percent(loaded, total)} loaded from the checkpoint">'
            f'<span class="bd-l" style="width:{frac:.2f}%"></span>'
            f'<span class="bd-n" style="width:{100 - frac:.2f}%"></span></div>'
        )
        note = f"{_percent(loaded, total)} of the parameters come from the checkpoint"
        new = _new_modules(origin)
        if new:
            note += "; initialised here (train these): " + ", ".join(
                f"<code>{_esc(n)}</code>" for n in new
            )
        parts.append(f'<div class="bd-note">{note}</div>')

    tree = _tree_html(model, name, "", origin, 0, max_depth)
    parts.append(f'<div class="bd-tree">{tree}</div>')
    parts.append("</div>")
    return "".join(parts)


# Grid labels whose value is already HTML.
_RAW = {"Weights", "Fine-tuned from", "Trained on"}
