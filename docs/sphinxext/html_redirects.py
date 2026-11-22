"""Sphinx extension generating static HTML redirect stubs for moved pages."""

from __future__ import annotations

import html
import json
from pathlib import Path, PurePosixPath
from urllib.parse import urldefrag, urljoin

_REDIRECT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Redirecting...</title>
  <meta name="robots" content="noindex, follow" />
{canonical_line}  <meta http-equiv="refresh" content="0; url={escaped_target}" />
  <script>window.location.replace({js_target});</script>
</head>
<body>
  <p>Redirecting to <a href="{escaped_target}">{escaped_target}</a>.</p>
</body>
</html>
"""


def _validate_source_path(src_rel: str, outdir: Path) -> Path:
    if not isinstance(src_rel, str) or not src_rel or "\\" in src_rel:
        raise ValueError(f"Invalid redirect source path: {src_rel!r}")
    posix_path = PurePosixPath(src_rel)
    if posix_path.is_absolute() or ".." in posix_path.parts or src_rel.endswith("/"):
        raise ValueError(f"Invalid redirect source path: {src_rel!r}")
    resolved_out = outdir.resolve()
    dest = (resolved_out / posix_path).resolve()
    if dest == resolved_out or not dest.is_relative_to(resolved_out):
        raise ValueError(f"Redirect source path escapes outdir: {src_rel!r}")
    return dest


def _render_redirect_html(src_rel: str, target: str, html_baseurl: str = "") -> str:
    if not isinstance(target, str) or not target.strip():
        raise ValueError(f"Invalid redirect target for {src_rel!r}: {target!r}")
    target_no_frag, frag = urldefrag(target)
    canonical_line = ""
    if html_baseurl:
        base = html_baseurl.rstrip("/") + "/"
        canonical_url = urljoin(urljoin(base, src_rel), target_no_frag)
        canonical_line = (
            f'  <link rel="canonical" '
            f'href="{html.escape(canonical_url, quote=True)}" />\n'
        )
    escaped_target = html.escape(target, quote=True)
    if frag:
        js_target = (
            f"{json.dumps(target_no_frag)} + "
            f"(window.location.hash || {json.dumps('#' + frag)})"
        )
    else:
        js_target = json.dumps(target)
    js_target = js_target.replace("</", "<\\/")
    return _REDIRECT_TEMPLATE.format(
        canonical_line=canonical_line,
        escaped_target=escaped_target,
        js_target=js_target,
    )


def write_html_redirects(app, exception) -> None:
    """Write static HTML redirect stubs after a successful HTML build."""
    if (
        exception is not None
        or getattr(app.builder, "format", "") != "html"
        or app.builder.name == "epub"
    ):
        return

    redirects = getattr(app.config, "html_redirects", None) or {}
    if not redirects:
        return

    outdir = Path(app.outdir)
    found_docs = getattr(getattr(app, "env", None), "found_docs", None) or ()
    html_baseurl = getattr(app.config, "html_baseurl", "") or ""

    for src_rel, target in sorted(redirects.items()):
        dest = _validate_source_path(src_rel, outdir)
        docname = src_rel.removesuffix(".html")
        if docname in found_docs:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            _render_redirect_html(src_rel, target, html_baseurl=html_baseurl),
            encoding="utf-8",
        )


def setup(app) -> dict[str, object]:
    """Register ``html_redirects`` config value and ``build-finished`` hook."""
    app.add_config_value("html_redirects", {}, "html")
    app.connect("build-finished", write_html_redirects)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
