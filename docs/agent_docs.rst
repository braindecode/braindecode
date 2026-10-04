:orphan:

#######################################
 Machine-readable documentation builds
#######################################

The documentation workflow renders selected canonical Sphinx pages with
``sphinx-markdown-builder`` after the HTML build, then runs
``docs/build_agent_docs.py``. The resulting ``llms.txt`` and Markdown siblings live
inside the same version directory as their HTML pages. Nothing is written to the
generated hosting repository by this script.

The curated index covers installation, model selection, API conventions, two model
references, tutorials and citation. It is not a complete Markdown mirror. Each exported
page identifies the package version, documentation URL and exact source commit. Internal
document links deliberately point to HTML so that pages outside this selection remain
accessible. A small translator adapter preserves HTML image paths, bibliography anchors
and gallery navigation, and omits raw HTML scripts/styles. Other unsupported nodes
retain the converter's warnings rather than being silently accepted. Interactive tables
and custom Sphinx content may not survive Markdown conversion; follow the canonical HTML
link.

**********************
 Build and validation
**********************

After a successful normal HTML build, run from the repository root:

::

    sphinx-build -b markdown -d docs/_build/doctrees -D plot_gallery=0 docs docs/_build/markdown
    python docs/build_agent_docs.py \
        --markdown-root docs/_build/markdown \
        --html-root docs/_build/html \
        --site-url https://braindecode.org/dev/ \
        --version "$(python -c 'import braindecode; print(braindecode.__version__)')" \
        --revision "$(git rev-parse HEAD)"

Use a clean output directory for each source revision. The Markdown pass disables
example execution; the preceding normal HTML build still has its usual dataset and model
requirements. Publication fails if any curated HTML or Markdown page is missing or its
Markdown is empty. Offline tests require pytest, Sphinx, sphinx-markdown-builder,
sphinx-gallery, sphinx-design, sphinxcontrib-bibtex and matplotlib. The integration
fixture builds HTML then Markdown with shared doctrees and checks that its tiny,
data-free example executes only once:

::

    python -m pytest -q docs/tests/test_agent_docs.py

******************
 Deployment scope
******************

The existing master-push workflow deploys only ``dev/``. A development checkout can
contain a release-looking package version; the URL scope and source commit are therefore
essential provenance. This change does not create a root index, retarget ``stable/`` or
rebuild any historical version. A future release build must use its own source checkout,
matching HTML and explicit version URL; do not copy development Markdown into stable
documentation.
