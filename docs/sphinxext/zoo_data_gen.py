"""Generate static ``generated/zoo_browser.html`` from ``braindecode/models/summary.csv``.

Pre-rendering the model zoo browser at Sphinx build time eliminates ~3,800 px of
client-side layout shift, makes all 100+ model links crawlable without JavaScript,
and drives dual-axis Modality x Architecture filtering via native CSS ``:has()``.
"""

from __future__ import annotations

import ast
import csv
import os
import os.path as op
from html import escape
from urllib.parse import quote

from sphinx.util import logging

logger = logging.getLogger(__name__)

FEATURED_MODELS: tuple[str, ...] = (
    "EEGPT",
    "Labram",
    "REVE",
    "CBraMod",
    "SAMBA",
    "EEGNeX",
    "EEGConformer",
    "ShallowFBCSPNet",
    "SleepFM",
    "SeizureTransformer",
    "GeoNIRS",
    "MEGXL",
)

MODALITY_ORDER: tuple[tuple[str, str], ...] = (
    ("all", "All"),
    ("eeg", "EEG"),
    ("ieeg", "iEEG"),
    ("fnirs", "fNIRS"),
    ("meg", "MEG"),
    ("semg", "sEMG"),
)

ARCHITECTURE_ORDER: tuple[tuple[str, str], ...] = (
    ("all", "All"),
    ("foundation", "Foundation"),
    ("convolution", "Convolution"),
    ("attention", "Attention"),
    ("channel", "Channel"),
    ("recurrent", "Recurrent"),
    ("filterbank", "Filterbank"),
    ("interpretable", "Interpretable"),
    ("graph", "Graph"),
)

CATEGORY_RULES: tuple[tuple[str, str, str], ...] = (
    ("Foundation Model", "foundation", "Foundation"),
    ("Filterbank", "filterbank", "Filterbank"),
    ("FilterBank", "filterbank", "Filterbank"),
    ("Graph Neural Network", "graph", "Graph"),
    ("GNN", "graph", "Graph"),
    ("Attention", "attention", "Attention"),
    ("Transformer", "attention", "Attention"),
    ("Interpretab", "interpretable", "Interpretable"),
    ("Channel", "channel", "Channel"),
    ("Recurrent", "recurrent", "Recurrent"),
    ("Convolution", "convolution", "Convolution"),
)


def _extract_categories(cat_field: str) -> tuple[str, str, list[str]]:
    """Return primary ``(id, label)`` plus deduplicated list of matched category IDs."""
    if not cat_field:
        return ("other", "Other", ["other"])
    lower = cat_field.lower()
    matched: list[tuple[str, str]] = []
    seen: set[str] = set()
    for needle, cid, label in CATEGORY_RULES:
        if needle.lower() in lower and cid not in seen:
            seen.add(cid)
            matched.append((cid, label))
    if not matched:
        return ("other", "Other", ["other"])
    primary_id, primary_label = matched[0]
    return (primary_id, primary_label, [cid for cid, _ in matched])


def _normalize_modalities(mod_field: str) -> tuple[list[str], str]:
    """Return ``(normalized_tokens, display_label)`` from CSV ``Modality`` cell."""
    raw_parts = [p.strip() for p in (mod_field or "").split(",") if p.strip()]
    if not raw_parts:
        return (["eeg"], "EEG")
    tokens: list[str] = []
    labels: list[str] = []
    canonical = {
        "eeg": "EEG",
        "ieeg": "iEEG",
        "ecog": "iEEG",
        "seeg": "iEEG",
        "fnirs": "fNIRS",
        "meg": "MEG",
        "semg": "sEMG",
        "emg": "sEMG",
    }
    for part in raw_parts:
        key = part.lower()
        norm_key = (
            "ieeg" if key in {"ecog", "seeg"} else ("semg" if key == "emg" else key)
        )
        if norm_key not in tokens:
            tokens.append(norm_key)
        disp = canonical.get(key, part)
        if disp not in labels:
            labels.append(disp)
    display = " · ".join(labels)
    return (tokens, display)


def _format_params(raw: str) -> str:
    """Format an integer parameter count as compact ``25.5 M`` / ``46.1 K``."""
    try:
        n = int((raw or "").strip().replace(",", ""))
    except (TypeError, ValueError):
        return (raw or "").strip()
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} M".replace(".0 ", " ")
    if n >= 1_000:
        return f"{n / 1_000:.1f} K".replace(".0 ", " ")
    return str(n)


def _format_sfreq(raw: str) -> str:
    """Format sampling frequency cleanly with ``Hz`` unit."""
    val = (raw or "").strip()
    if not val:
        return ""
    try:
        num = float(val)
        clean = str(int(num)) if num.is_integer() else f"{num:g}"
        return f"{clean} Hz"
    except ValueError:
        return f"{val} Hz"


def _clean_type(type_field: str) -> str:
    parts = [p.strip() for p in (type_field or "").split(",") if p.strip()]
    if not parts:
        return "Prediction"
    return " & ".join(parts)


def _build_description(
    mod_tokens: list[str],
    mod_display: str,
    cat_field: str,
    application: str,
    type_field: str,
) -> str:
    """Synthesize a domain-specific description without vague 'General' copy."""
    type_label = _clean_type(type_field)
    app = (application or "").strip()

    if app and app.lower() != "general":
        app_parts = [p.strip() for p in app.split(",") if p.strip()]
        app_clean = " & ".join(
            "Clinical Normal/Abnormal" if p == "Normal Abnormal" else p
            for p in app_parts
        )
        if type_label.lower() in app_clean.lower():
            return app_clean
        return f"{app_clean} · {type_label}"

    mod_prefix = (
        f"Cross-modal {' + '.join(mod_display.split(' · '))}"
        if len(mod_tokens) > 1
        else mod_display
    )
    cat_lower = (cat_field or "").lower()
    if "foundation model" in cat_lower:
        if "tokeniz" in type_label.lower():
            arch = f"{mod_prefix} Neural Tokenizer"
        else:
            arch = f"{mod_prefix} Foundation Model"
    elif "attention" in cat_lower or "transformer" in cat_lower:
        if "convolution" in cat_lower and "recurrent" in cat_lower:
            arch = f"{mod_prefix} Conv-Recurrent Transformer"
        elif "convolution" in cat_lower:
            arch = f"{mod_prefix} Conv-Transformer"
        else:
            arch = f"{mod_prefix} Transformer"
    elif "filterbank" in cat_lower:
        arch = f"{mod_prefix} Filterbank Network"
    elif "graph" in cat_lower:
        arch = f"{mod_prefix} Graph Network"
    elif "recurrent" in cat_lower and "convolution" in cat_lower:
        arch = f"{mod_prefix} Temporal Conv-Recurrent Net"
    elif "recurrent" in cat_lower:
        arch = f"{mod_prefix} Recurrent Network"
    elif "interpretab" in cat_lower:
        arch = f"{mod_prefix} Interpretable Network"
    elif "convolution" in cat_lower:
        arch = f"{mod_prefix} Convolutional Network"
    else:
        arch = f"{mod_prefix} Neural Architecture"

    return f"{arch} · {type_label}"


MODALITY_PREVIEWS: dict[str, tuple[str, str]] = {
    "all": (
        "Unified EEGModuleMixin · 103 models",
        'from braindecode import EEGClassifier\nclf = EEGClassifier("EEGNeX", max_epochs=20)\nclf.fit(windows, y=None)',
    ),
    "eeg": (
        "Scalp EEG & Sleep PSG · 77 models",
        'from braindecode.models import EEGNeX\nmodel = EEGNeX(n_chans=22, n_outputs=4, n_times=1000)',
    ),
    "ieeg": (
        "Intracranial iEEG / sEEG / ECoG · 9 models",
        'from braindecode.models import DuIN\nmodel = DuIN.from_pretrained("braindecode/duin-pretrained", n_outputs=4)',
    ),
    "fnirs": (
        "Optical fNIRS (HbO / HbR) · 8 models",
        'from braindecode.models import GeoNIRS\nmodel = GeoNIRS.from_pretrained("braindecode/geonirs-base-pretrained")',
    ),
    "meg": (
        "Magnetoencephalography (SQUID / OPM) · 7 models",
        'from braindecode.models import MEGXL\nmodel = MEGXL.from_pretrained("braindecode/megxl-pretrained", n_outputs=2)',
    ),
    "semg": (
        "Surface Electromyography (sEMG) · 7 models",
        'from braindecode.models import VEMG2Pose\nmodel = VEMG2Pose.from_pretrained("braindecode/VEMG2Pose-emg2pose")',
    ),
}

CATEGORY_PREVIEWS: dict[str, tuple[str, str]] = {
    "all": (
        "All Registered Backbones · 103 models",
        'from braindecode.models import Labram\nmodel = Labram.from_pretrained("braindecode/labram-pretrained", n_outputs=4)',
    ),
    "foundation": (
        "Pretrained Foundation Models · 48 models",
        'from braindecode.models import EEGPT\nmodel = EEGPT.from_pretrained("braindecode/eegpt-pretrained", n_outputs=4)',
    ),
    "convolution": (
        "Spatial-Temporal Convolutions · 69 models",
        'from braindecode.models import ShallowFBCSPNet\nmodel = ShallowFBCSPNet(n_chans=22, n_outputs=4, n_times=1000)',
    ),
    "attention": (
        "Attention & Patch Transformers · 50 models",
        'from braindecode.models import EEGConformer\nmodel = EEGConformer(n_chans=22, n_outputs=4, n_times=1000)',
    ),
    "channel": (
        "Channel & 3D Montage-Aware · 23 models",
        'windows.set_return_ch_pos(True)  # (x, y, z) coordinates + ch_mask\nmodel = REVE(chs_info=raw.info["chs"], n_outputs=4)\nlogits = model(X, pos=ch_pos, ch_mask=ch_mask)',
    ),
    "recurrent": (
        "Recurrent & State-Space · 13 models",
        'from braindecode.models import SAMBA\nmodel = SAMBA(n_chans=19, n_outputs=2, n_times=256)',
    ),
    "filterbank": (
        "Multi-Band Filterbank Networks · 4 models",
        'from braindecode.models import FBCNet\nmodel = FBCNet(n_chans=22, n_outputs=4, n_times=1000, sfreq=250)',
    ),
    "interpretable": (
        "Interpretable Spectral & Spatial Nets · 3 models",
        'from braindecode.models import SincShallowNet\nmodel = SincShallowNet(n_chans=22, n_outputs=4, n_times=1000, sfreq=250)',
    ),
    "graph": (
        "Electrode Graph Neural Networks · 2 models",
        'from braindecode.models import DGCNN\nmodel = DGCNN(n_chans=62, n_outputs=3, n_times=200)',
    ),
}

HUB_REPO_HINTS: dict[str, str] = {
    "EEGPT": "braindecode/eegpt-pretrained",
    "Labram": "braindecode/labram-pretrained",
    "REVE": "brain-bzh/reve-base",
    "CBraMod": "braindecode/cbramod-pretrained",
    "SAMBA": "braindecode/samba-2s-pretrained",
    "DuIN": "braindecode/duin-pretrained",
    "MEGXL": "braindecode/megxl-pretrained",
    "GeoNIRS": "braindecode/geonirs-base-pretrained",
    "BrainBERT": "braindecode/brainbert-pretrained",
    "BIOT": "braindecode/biot-pretrained-six-datasets-18chs",
    "BENDR": "braindecode/braindecode-bendr",
    "LUNA": "PulpBio/LUNA",
    "PopulationTransformer": "braindecode/popt-pretrained",
    "Brant": "braindecode/brant-pretrained",
    "BrainOmni": "braindecode/brainomni-base-pretrained",
    "CodeBrain": "braindecode/codebrain-pretrained",
    "VEMG2Pose": "braindecode/VEMG2Pose-emg2pose",
    "TinyMyo": "braindecode/tinymyo-pretrained",
}


def _load_init_signatures(models_dir: str) -> dict[str, str]:
    """Extract concise AST constructor signatures for every model class."""
    sigs: dict[str, str] = {}
    if not op.isdir(models_dir):
        return sigs
    for fname in sorted(os.listdir(models_dir)):
        if not fname.endswith(".py"):
            continue
        fpath = op.join(models_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                tree = ast.parse(fh.read())
        except OSError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    args = [a.arg for a in item.args.args if a.arg != "self"]
                    if not args:
                        args = [a.arg for a in item.args.kwonlyargs]
                    keep = [a for a in args if not a.startswith("_")][:4]
                    sigs[node.name] = ", ".join(keep) if keep else "n_chans, n_outputs, n_times"
    return sigs


def _build_model_api_snippet(m: dict, sig_str: str) -> str:
    """Build a verified 2-line Python API snippet for hovering a model card."""
    name = m["name"]
    call_args = sig_str or "n_chans, n_outputs, n_times"
    lines = [f"from braindecode.models import {name}"]
    if name in HUB_REPO_HINTS:
        lines.append(f'model = {name}.from_pretrained("{HUB_REPO_HINTS[name]}")')
    elif "foundation" in m["cats"]:
        lines.append(f"model = {name}({call_args})  # + .from_pretrained(...)")
    else:
        lines.append(f"model = {name}({call_args})")
    return "\n".join(lines)


def _badge_tone(cid: str) -> str:
    """Semantic non-purple badge modifier class for ``.zoo-card-cat``."""
    return {
        "foundation": "warm",
        "convolution": "",
        "attention": "",
        "filterbank": "green",
        "graph": "green",
        "channel": "slate",
        "recurrent": "slate",
        "interpretable": "slate",
        "other": "slate",
    }.get(cid, "slate")


def parse_summary_csv(csv_path: str) -> list[dict]:
    """Read ``summary.csv`` and return structured model records."""
    by_name: dict[str, dict] = {}
    ordered_names: list[str] = []
    sigs = _load_init_signatures(op.dirname(op.abspath(csv_path)))

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            name = (row.get("Model") or "").strip()
            if not name:
                continue
            cat_field = (row.get("Categorization") or "").strip()
            mod_field = (row.get("Modality") or "").strip()
            app_field = (row.get("Application") or "").strip()
            type_field = (row.get("Type") or "").strip()

            cid, clabel, all_cats = _extract_categories(cat_field)
            mod_tokens, mod_display = _normalize_modalities(mod_field)
            params = _format_params(row.get("#Parameters") or "")
            sfreq = _format_sfreq(row.get("Sampling Frequency (Hz)") or "")
            desc = _build_description(
                mod_tokens, mod_display, cat_field, app_field, type_field
            )

            record = {
                "name": name,
                "cat": cid,
                "cat_label": clabel,
                "cats": all_cats,
                "mods": mod_tokens,
                "mod_display": mod_display,
                "params": params,
                "sfreq": sfreq,
                "desc": desc,
                "featured": name in FEATURED_MODELS,
                "sig": sigs.get(name, "n_chans, n_outputs, n_times"),
            }
            by_name[name] = record
            ordered_names.append(name)

    featured_list = [by_name[n] for n in FEATURED_MODELS if n in by_name]
    extra_list = [by_name[n] for n in ordered_names if n not in FEATURED_MODELS]
    return featured_list + extra_list


def render_zoo_html(models: list[dict]) -> str:
    """Render static, accessible HTML for ``generated/zoo_browser.html``."""
    total = len(models)
    featured_count = sum(1 for m in models if m["featured"])

    mod_counts = {"all": total}
    for mid, _ in MODALITY_ORDER[1:]:
        mod_counts[mid] = sum(1 for m in models if mid in m["mods"])

    cat_counts = {"all": total}
    for cid, _ in ARCHITECTURE_ORDER[1:]:
        cat_counts[cid] = sum(1 for m in models if cid in m["cats"])

    mod_pills: list[str] = []
    for mid, mlabel in MODALITY_ORDER:
        checked = " checked" if mid == "all" else ""
        p_head, p_code = MODALITY_PREVIEWS.get(mid, (mlabel, ""))
        mod_pills.append(
            f'<label class="zoo-mod-pill" data-api-head="{escape(p_head)}" '
            f'data-api-code="{escape(p_code)}">'
            f'<input type="radio" name="bd-zoo-mod" value="{escape(mid)}" class="sr-only"{checked}/>'
            f"<span>{escape(mlabel)}</span>"
            f'<span class="chip-count">{mod_counts.get(mid, 0)}</span>'
            f"</label>"
        )

    arch_chips: list[str] = []
    for cid, clabel in ARCHITECTURE_ORDER:
        checked = " checked" if cid == "all" else ""
        c_head, c_code = CATEGORY_PREVIEWS.get(cid, (clabel, ""))
        arch_chips.append(
            f'<label class="zoo-chip" data-api-head="{escape(c_head)}" '
            f'data-api-code="{escape(c_code)}">'
            f'<input type="radio" name="bd-zoo-cat" value="{escape(cid)}" class="sr-only"{checked}/>'
            f"<span>{escape(clabel)}</span>"
            f'<span class="chip-count">{cat_counts.get(cid, 0)}</span>'
            f"</label>"
        )

    cards: list[str] = []
    for m in models:
        name = m["name"]
        cid = m["cat"]
        clabel = m["cat_label"]
        tone = _badge_tone(cid)
        cat_cls = f"zoo-card-cat {tone}".strip()
        mods_attr = " ".join(m["mods"])
        cats_attr = " ".join(m["cats"])
        fold_attr = 'data-featured="1"' if m["featured"] else 'data-fold="extra"'
        search_hay = escape(
            f"{name} {m['mod_display']} {clabel} {' '.join(m['cats'])} {m['desc']}".lower()
        )
        href = f"generated/braindecode.models.{quote(name)}.html"
        api_head = f"braindecode.models.{name}"
        api_code = _build_model_api_snippet(m, m.get("sig", ""))

        meta_parts: list[str] = []
        if m["params"]:
            meta_parts.append(f"<span><b>{escape(m['params'])}</b> params</span>")
        if m["sfreq"]:
            meta_parts.append(f"<span>{escape(m['sfreq'])}</span>")
        meta_html = "".join(meta_parts)

        cards.append(
            f'<a class="zoo-card" href="{href}" data-mod="{escape(mods_attr)}" '
            f'data-cat="{escape(cid)}" data-cats="{escape(cats_attr)}" '
            f'{fold_attr} data-search="{search_hay}" '
            f'data-api-head="{escape(api_head)}" data-api-code="{escape(api_code)}">'
            f'<span class="zoo-card-top">'
            f'<span class="{escape(cat_cls)}">{escape(clabel)}</span>'
            f'<span class="zoo-mod-tag">{escape(m["mod_display"])}</span>'
            f"</span>"
            f'<span class="zoo-card-name">{escape(name)}</span>'
            f'<span class="zoo-card-desc">{escape(m["desc"])}</span>'
            f'<span class="zoo-card-meta">{meta_html}</span>'
            f"</a>"
        )

    shown_initial = featured_count if featured_count else total
    expand_html = (
        '  <div class="zoo-expand-wrap">\n'
        '    <label class="zoo-expand-btn">\n'
        '      <input type="checkbox" id="bd-zoo-more" class="sr-only"/>\n'
        "      <span>Show all architectures ↓</span>\n"
        "    </label>\n"
        "  </div>\n"
        if total > shown_initial
        else ""
    )

    return (
        '<div class="bd-zoo">\n'
        '  <div class="zoo-toolbar">\n'
        '    <div class="zoo-toolbar-top">\n'
        '      <fieldset class="zoo-mods" aria-label="Filter models by signal modality">\n'
        f"        {''.join(mod_pills)}\n"
        "      </fieldset>\n"
        '      <search class="zoo-search bd-groove" aria-label="Search model zoo">\n'
        '        <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
        'stroke-linejoin="round"><circle cx="11" cy="11" r="7"/>'
        '<path d="m20 20-3.5-3.5"/></svg>\n'
        '        <input type="search" id="bd-zoo-q" '
        'placeholder="Search models, tasks, or modalities (/)" '
        'aria-label="Search models by name, modality, architecture, or task" '
        'autocomplete="off"/>\n'
        "      </search>\n"
        "    </div>\n"
        '    <fieldset class="zoo-chips" aria-label="Filter models by architecture backbone">\n'
        f"      {''.join(arch_chips)}\n"
        "    </fieldset>\n"
        "  </div>\n"
        f'  <div class="zoo-grid">{"".join(cards)}</div>\n'
        '  <p class="zoo-empty" role="status">No architectures match the active modality, backbone, or search query.</p>\n'
        f"{expand_html}"
        '  <div class="zoo-foot">\n'
        f'    <output class="zoo-foot-count" for="bd-zoo-q" aria-live="polite">'
        f"Showing {shown_initial} featured of {total} architectures</output>\n"
        '    <a href="models/models_table.html">Open full sortable comparison table '
        '<span aria-hidden="true">→</span></a>\n'
        "  </div>\n"
        '  <div class="zoo-api-pop" id="bd-zoo-pop" hidden aria-hidden="true">\n'
        '    <div class="zoo-api-pop-head bd-mono">\n'
        '      <span class="zoo-api-pop-kicker">API PREVIEW</span>\n'
        '      <span class="zoo-api-pop-title"></span>\n'
        "    </div>\n"
        '    <pre class="zoo-api-pop-code bd-mono"></pre>\n'
        "  </div>\n"
        "</div>\n"
    )


def generate_zoo_data(app, *_args) -> None:
    """Sphinx ``builder-inited`` hook writing ``generated/zoo_browser.html``."""
    here = op.dirname(op.abspath(__file__))
    csv_path = op.normpath(
        op.join(here, "..", "..", "braindecode", "models", "summary.csv")
    )
    out_dir = op.join(app.srcdir, "generated")
    out_path = op.join(out_dir, "zoo_browser.html")
    os.makedirs(out_dir, exist_ok=True)

    if not op.exists(csv_path):
        logger.warning("zoo_data_gen: %s not found, writing empty fallback", csv_path)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(render_zoo_html([]))
        return

    if op.exists(out_path):
        out_mtime = op.getmtime(out_path)
        if out_mtime >= op.getmtime(csv_path) and out_mtime >= op.getmtime(__file__):
            return

    models = parse_summary_csv(csv_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(render_zoo_html(models))


def setup(app):
    app.connect("builder-inited", generate_zoo_data)
    return {"version": "2.0", "parallel_read_safe": True}
