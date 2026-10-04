"""Keep Markdown assets and citations linked to authoritative HTML output."""

from urllib.parse import urlsplit

from docutils import nodes
from sphinx.util.osutil import relative_uri
from sphinx_markdown_builder.translator import MarkdownTranslator


class AgentMarkdownTranslator(MarkdownTranslator):
    """Adapt known Sphinx nodes, retaining warnings for other unknown nodes."""

    def visit_image(self, node):
        uri = node["uri"]
        if not urlsplit(uri).scheme and not uri.startswith("//"):
            # HTML's image collector assigns collision-safe filenames. The
            # Markdown builder does not copy images or rewrite their URIs.
            if uri in self.builder.env.images:
                filename = self.builder.env.images[uri][1]
                uri = relative_uri(
                    self.builder.get_target_uri(self.builder.current_doc_name),
                    "_images/" + filename,
                )
        self.add(f"![{node.get('alt', 'image')}]({uri})")
        raise nodes.SkipNode

    def visit_raw(self, node):
        # Raw HTML contains executable scripts and CSS, not Markdown prose.
        # The page's canonical HTML link preserves access to those widgets.
        raise nodes.SkipNode

    def visit_citation(self, node):
        for anchor in node.get("ids", []):
            self._add_anchor(anchor)

    def depart_citation(self, node):
        self.add("\n\n")

    def visit_label(self, node):
        self.add(f"[{node.astext()}] ")
        raise nodes.SkipNode

    def visit_toctree(self, node):
        # Gallery thumbnail grids contain otherwise-unrendered toctrees.
        for title, docname in node.get("entries", []):
            if not title and docname in self.builder.env.titles:
                title = self.builder.env.titles[docname].astext()
            target = docname
            if not urlsplit(docname).scheme:
                target = relative_uri(
                    self.builder.get_target_uri(self.builder.current_doc_name),
                    self.builder.get_target_uri(docname),
                )
            self.add(f"\n- [{title or docname}]({target})\n")
        raise nodes.SkipNode


def setup(app):
    app.set_translator("markdown", AgentMarkdownTranslator, override=True)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
