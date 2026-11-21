"""Programmatic SEO metadata, Markdown discovery, and JSON-LD for Sphinx HTML pages."""

from __future__ import annotations

import csv
import html
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

from sphinx.util import logging

logger = logging.getLogger(__name__)

_MODEL_PAGE_PREFIX = "generated/braindecode.models."
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _load_agent_page_specs() -> set[str]:
    """Load exported Markdown page names from ``docs/build_agent_docs.py``."""
    agent_docs_path = Path(__file__).resolve().parents[1] / "build_agent_docs.py"
    if not agent_docs_path.is_file():
        return set()
    spec = importlib.util.spec_from_file_location("build_agent_docs", agent_docs_path)
    if spec is None or spec.loader is None:
        return set()
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pages = getattr(module, "PAGE_SPECS", None) or getattr(
        module, "CRITICAL_PAGES", None
    )
    return set(pages or ())


def format_params(raw: str) -> str:
    """Format a raw integer parameter count into a compact human string."""
    cleaned = (raw or "").strip().replace(",", "")
    try:
        count = int(cleaned)
    except ValueError:
        return (raw or "").strip()
    if count >= 1_000_000:
        val = f"{count / 1_000_000:.1f}".rstrip("0").rstrip(".")
        return f"{val}M"
    if count >= 1_000:
        val = f"{count / 1_000:.1f}".rstrip("0").rstrip(".")
        return f"{val}K"
    return f"{count:,}"


def _clean_modality(raw: str) -> str:
    parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
    return ", ".join(parts) or "EEG"


def _join_natural(parts: list[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


def _clean_categorization_parts(raw: str) -> list[str]:
    parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
    if not parts:
        return ["deep learning"]
    cleaned: list[str] = []
    for part in parts:
        if part.upper() == "SPD":
            cleaned.append("SPD")
        elif part == "Attention/Transformer":
            cleaned.append("attention/transformer")
        else:
            cleaned.append(part.lower())
    return cleaned


def _clean_application(raw: str) -> str:
    stripped = re.sub(r"\s*\(semg\)\s*", "", (raw or "").strip(), flags=re.IGNORECASE)
    parts = [p.strip() for p in stripped.split(",") if p.strip()]
    normalized: list[str] = []
    for part in parts:
        words = [
            w if (w.isupper() and len(w) >= 2) else w.lower() for w in part.split()
        ]
        normalized.append(" ".join(words))
    return _join_natural(normalized) or "biosignal"


def build_model_description(row: dict[str, str]) -> str:
    """Build a factual 150-190 char summary for a ``summary.csv`` model row."""
    model_name = (row.get("Model") or "").strip()
    cat_parts = _clean_categorization_parts(row.get("Categorization") or "")
    application = _clean_application(row.get("Application") or "")
    modality_parts = [
        p.strip() for p in (row.get("Modality") or "").split(",") if p.strip()
    ]
    modality_prose = _join_natural(modality_parts) or "EEG"
    params = format_params(row.get("#Parameters") or "")
    sfreq = (row.get("Sampling Frequency (Hz)") or "").strip()

    if params and sfreq:
        spec_clause = f" ({params} parameters at {sfreq} Hz)"
    elif params:
        spec_clause = f" ({params} parameters)"
    elif sfreq:
        spec_clause = f" (at {sfreq} Hz)"
    else:
        spec_clause = ""

    suffixes = (
        " with standardized (batch, n_chans, n_times) tensor inputs.",
        " with standardized (batch, n_chans, n_times) inputs.",
        " with (batch, n_chans, n_times) tensor inputs.",
        " on (batch, n_chans, n_times) signals.",
        " in Python.",
        ".",
    )

    # Try full categorization list first, then compact to primary 2 or 1 tags.
    candidate_cat_lists = [cat_parts]
    if len(cat_parts) > 2:
        candidate_cat_lists.append(cat_parts[:2])
    if len(cat_parts) > 1:
        candidate_cat_lists.append(cat_parts[:1])

    for cats in candidate_cat_lists:
        cat_text = ", ".join(cats)
        base = (
            f"{model_name} is an open-source PyTorch {cat_text} architecture in "
            f"Braindecode for {application} {modality_prose} decoding{spec_clause}"
        )
        for suffix in suffixes:
            candidate = base + suffix
            if 150 <= len(candidate) <= 190:
                return candidate

    # Fallback if a minimal row is shorter than 150 chars or an unusually long row > 190.
    fallback = (
        f"{model_name} is an open-source PyTorch {cat_parts[0]} architecture in "
        f"Braindecode for {application} {modality_prose} decoding{spec_clause}"
    )
    if len(fallback) + 1 < 150:
        padded = (
            fallback + " with standardized (batch, n_chans, n_times) tensor inputs and "
            "MNE-Python integration."
        )
        if len(padded) <= 190:
            return padded
        clipped = padded[:189].rsplit(" ", 1)[0].rstrip(",;:") + "."
        return clipped
    if len(fallback) + 1 > 190:
        return fallback[:189].rsplit(" ", 1)[0].rstrip(",;:") + "."
    return fallback + "."


def load_summary_csv(csv_path: Path) -> dict[str, dict[str, str]]:
    """Parse ``braindecode/models/summary.csv`` keyed by model class name."""
    if not csv_path.is_file():
        logger.warning("model_seo: summary CSV not found at %s", csv_path)
        return {}
    models: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        for row in reader:
            name = (row.get("Model") or "").strip()
            if name:
                models[name] = {
                    k: (v or "").strip() for k, v in row.items() if k is not None
                }
    return models


def _meta_tag_pattern(attr: str, key: str) -> re.Pattern[str]:
    return re.compile(
        rf"<meta\b[^>]*\b{attr}\s*=\s*([\"']){re.escape(key)}\1[^>]*>\s*",
        flags=re.IGNORECASE,
    )


def _extract_meta_content(metatags: str, attr: str, key: str) -> str | None:
    match = _meta_tag_pattern(attr, key).search(metatags or "")
    if not match:
        return None
    content_match = re.search(
        r"\bcontent\s*=\s*([\"'])(.*?)\1",
        match.group(0),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not content_match:
        return None
    return html.unescape(content_match.group(2)).strip()


def upsert_meta_tag(metatags: str, attr: str, key: str, content: str) -> str:
    """Replace any existing ``<meta {attr}="{key}" ...>`` or append a single tag."""
    cleaned = _meta_tag_pattern(attr, key).sub("", metatags or "")
    escaped = html.escape(content, quote=True)
    tag = f'<meta {attr}="{key}" content="{escaped}" />\n'
    if cleaned and not cleaned.endswith("\n"):
        cleaned += "\n"
    return cleaned + tag


def upsert_markdown_alternate(metatags: str, href: str) -> str:
    """Ensure a single ``<link rel="alternate" type="text/markdown" ...>`` tag."""
    pattern = re.compile(
        r"<link\b[^>]*\btype\s*=\s*([\"'])text/markdown\1[^>]*>\s*",
        flags=re.IGNORECASE,
    )
    cleaned = pattern.sub("", metatags or "")
    escaped = html.escape(href, quote=True)
    tag = f'<link rel="alternate" type="text/markdown" href="{escaped}" />\n'
    if cleaned and not cleaned.endswith("\n"):
        cleaned += "\n"
    return cleaned + tag


def _serialize_jsonld(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False).replace("</", r"<\/")
    return f'<script type="application/ld+json">{raw}</script>\n'


def _strip_html(text: str) -> str:
    plain = _HTML_TAG_RE.sub("", text or "")
    # Remove trailing Sphinx permalink pilcrows if present.
    plain = plain.replace("¶", "").replace("#", " ").strip()
    return re.sub(r"\s+", " ", html.unescape(plain)).strip()


def _build_model_jsonld(
    *,
    model_name: str,
    row: dict[str, str],
    description: str,
    canonical_base: str,
    pagename: str,
    release: str,
) -> dict[str, Any]:
    modality = _clean_modality(row.get("Modality") or "")
    page_url = f"{canonical_base}{pagename}.html"
    cat_tags = [
        p.strip() for p in (row.get("Categorization") or "").split(",") if p.strip()
    ]
    app_tags = [
        p.strip() for p in (row.get("Application") or "").split(",") if p.strip()
    ]
    keywords = list(
        dict.fromkeys(
            [
                model_name,
                f"braindecode.models.{model_name}",
                "PyTorch",
                modality,
                *app_tags,
                *cat_tags,
                "Braindecode",
            ]
        )
    )
    target_product: dict[str, Any] = {
        "@type": "SoftwareApplication",
        "name": "Braindecode",
        "url": canonical_base,
    }
    if release:
        target_product["softwareVersion"] = release

    return {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "BreadcrumbList",
                "itemListElement": [
                    {
                        "@type": "ListItem",
                        "position": 1,
                        "name": "Braindecode",
                        "item": canonical_base,
                    },
                    {
                        "@type": "ListItem",
                        "position": 2,
                        "name": "Models",
                        "item": f"{canonical_base}models/models_table.html",
                    },
                    {
                        "@type": "ListItem",
                        "position": 3,
                        "name": "API Reference",
                        "item": f"{canonical_base}api.html",
                    },
                    {
                        "@type": "ListItem",
                        "position": 4,
                        "name": model_name,
                        "item": page_url,
                    },
                ],
            },
            {
                "@type": "SoftwareSourceCode",
                "@id": f"{page_url}#code",
                "name": f"braindecode.models.{model_name}",
                "alternateName": f"{model_name} — PyTorch {modality} Model",
                "description": description,
                "url": page_url,
                "programmingLanguage": "Python",
                "runtimePlatform": "PyTorch",
                "codeRepository": "https://github.com/braindecode/braindecode",
                "license": "https://spdx.org/licenses/BSD-3-Clause.html",
                "targetProduct": target_product,
                "keywords": keywords,
            },
        ],
    }


def _is_gallery_tutorial(pagename: str) -> bool:
    return (
        pagename.startswith("auto_examples/")
        and pagename != "auto_examples/index"
        and not pagename.endswith("/index")
        and "sg_execution_times" not in pagename
    )


def _build_tutorial_jsonld(
    *,
    pagename: str,
    title: str,
    description: str,
    canonical_base: str,
) -> dict[str, Any]:
    page_url = f"{canonical_base}{pagename}.html"
    clean_title = _strip_html(title) or pagename.rsplit("/", 1)[-1]
    clean_desc = (
        description
        or f"{clean_title} — executable PyTorch biosignal decoding tutorial in Braindecode."
    )

    breadcrumbs: list[dict[str, Any]] = [
        {
            "@type": "ListItem",
            "position": 1,
            "name": "Braindecode",
            "item": canonical_base,
        },
        {
            "@type": "ListItem",
            "position": 2,
            "name": "Examples",
            "item": f"{canonical_base}auto_examples/index.html",
        },
    ]
    rel_parts = pagename.removeprefix("auto_examples/").split("/")
    if len(rel_parts) >= 2:
        section_slug = rel_parts[0]
        section_label = section_slug.replace("_", " ").title()
        breadcrumbs.append(
            {
                "@type": "ListItem",
                "position": 3,
                "name": section_label,
                "item": f"{canonical_base}auto_examples/{section_slug}/index.html",
            }
        )
    breadcrumbs.append(
        {
            "@type": "ListItem",
            "position": len(breadcrumbs) + 1,
            "name": clean_title,
            "item": page_url,
        }
    )

    return {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "BreadcrumbList",
                "itemListElement": breadcrumbs,
            },
            {
                "@type": "TechArticle",
                "@id": f"{page_url}#article",
                "headline": clean_title,
                "name": clean_title,
                "description": clean_desc,
                "url": page_url,
                "inLanguage": "en",
                "author": {
                    "@type": "Organization",
                    "name": "Braindecode Developers",
                    "url": canonical_base,
                },
                "publisher": {
                    "@type": "Organization",
                    "name": "Braindecode Developers",
                    "url": canonical_base,
                },
                "isPartOf": {
                    "@type": "WebSite",
                    "name": "Braindecode",
                    "url": canonical_base,
                },
                "about": {
                    "@type": "SoftwareSourceCode",
                    "name": "Braindecode",
                    "programmingLanguage": "Python",
                    "runtimePlatform": "PyTorch",
                    "codeRepository": "https://github.com/braindecode/braindecode",
                },
            },
        ],
    }


def load_model_seo_data(app) -> None:
    """Sphinx ``builder-inited`` hook loading ``summary.csv`` and ``PAGE_SPECS``."""
    configured_csv = getattr(app.config, "model_seo_summary_csv", None)
    if configured_csv:
        csv_path = Path(configured_csv)
    else:
        srcdir_candidate = (
            Path(app.srcdir).resolve().parent / "braindecode" / "models" / "summary.csv"
        )
        if srcdir_candidate.is_file():
            csv_path = srcdir_candidate
        else:
            csv_path = (
                Path(__file__).resolve().parents[2]
                / "braindecode"
                / "models"
                / "summary.csv"
            )
    app.env.model_seo_models = load_summary_csv(csv_path)
    configured_pages = getattr(app.config, "model_seo_agent_pages", None)
    if configured_pages is not None:
        app.env.model_seo_agent_pages = set(configured_pages)
    else:
        app.env.model_seo_agent_pages = _load_agent_page_specs()


def inject_model_seo(app, pagename: str, templatename: str, context: dict, doctree):
    """Sphinx ``html-page-context`` hook running after ``sphinxext.opengraph``."""
    if getattr(app.builder, "format", "") != "html" or app.builder.name == "epub":
        return

    metatags = context.get("metatags") or ""
    is_stable_build = bool(context.get("is_stable_build", False))
    robots_content = (
        "index, follow, max-image-preview:large, max-snippet:-1"
        if is_stable_build
        else "noindex, follow"
    )
    metatags = upsert_meta_tag(metatags, "name", "robots", robots_content)

    deploy_url = (
        context.get("docs_url")
        or getattr(app.config, "ogp_site_url", "")
        or "https://braindecode.org/dev/"
    ).rstrip("/") + "/"
    canonical_base = (
        context.get("canonical_base_url")
        or getattr(app.config, "html_baseurl", "")
        or "https://braindecode.org/stable/"
    ).rstrip("/") + "/"

    agent_pages: set[str] = getattr(app.env, "model_seo_agent_pages", set())
    if pagename in agent_pages:
        metatags = upsert_markdown_alternate(metatags, f"{deploy_url}{pagename}.md")

    models: dict[str, dict[str, str]] = getattr(app.env, "model_seo_models", {})
    if pagename.startswith(_MODEL_PAGE_PREFIX):
        model_name = pagename.removeprefix(_MODEL_PAGE_PREFIX)
        row = models.get(model_name)
        if row is not None:
            modality = _clean_modality(row.get("Modality") or "")
            seo_title = f"{model_name} — PyTorch {modality} Model"
            context["title"] = seo_title
            metatags = upsert_meta_tag(metatags, "property", "og:title", seo_title)

            description = build_model_description(row)
            metatags = upsert_meta_tag(metatags, "name", "description", description)
            metatags = upsert_meta_tag(
                metatags, "property", "og:description", description
            )

            release = str(
                getattr(app.config, "release", "") or context.get("release") or ""
            )
            jsonld = _build_model_jsonld(
                model_name=model_name,
                row=row,
                description=description,
                canonical_base=canonical_base,
                pagename=pagename,
                release=release,
            )
            metatags += _serialize_jsonld(jsonld)
    elif pagename != "index" and _is_gallery_tutorial(pagename):
        existing_desc = _extract_meta_content(
            metatags, "name", "description"
        ) or _extract_meta_content(metatags, "property", "og:description")
        if existing_desc and not _extract_meta_content(
            metatags, "property", "og:description"
        ):
            metatags = upsert_meta_tag(
                metatags, "property", "og:description", existing_desc
            )
        jsonld = _build_tutorial_jsonld(
            pagename=pagename,
            title=str(context.get("title") or ""),
            description=existing_desc or "",
            canonical_base=canonical_base,
        )
        metatags += _serialize_jsonld(jsonld)
    else:
        existing_desc = _extract_meta_content(metatags, "name", "description")
        if existing_desc and not _extract_meta_content(
            metatags, "property", "og:description"
        ):
            metatags = upsert_meta_tag(
                metatags, "property", "og:description", existing_desc
            )

    context["metatags"] = metatags


def setup(app):
    app.add_config_value("model_seo_summary_csv", None, "html")
    app.add_config_value("model_seo_agent_pages", None, "html")
    app.connect("builder-inited", load_model_seo_data)
    app.connect("html-page-context", inject_model_seo, priority=600)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
