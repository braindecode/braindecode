# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = Braindecode
SOURCEDIR     = .
BUILDDIR      = _build/
ALLSPHINXOPTS   = -d _build/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

apidoc:
	sphinx-apidoc -M -o source ../braindecode/

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


removeipynbcheckpoints: Makefile
	rm -rf notebooks/.ipynb_checkpoints/ notebooks/visualization/.ipynb_checkpoints/

removesource: Makefile
	rm -rf source/

rmanddoc: removesource removeipynbcheckpoints apidoc html
	echo "Done"

clean:
	-rm -rf _build auto_examples generated

html-pattern:
	$(SPHINXBUILD) -D plot_gallery=1 -D sphinx_gallery_conf.filename_pattern=$(PATTERN) -b html $(ALLSPHINXOPTS) _build/html
	@echo
	@echo "Build finished. The HTML pages are in _build/html"

html-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -b html $(ALLSPHINXOPTS) _build/html
	@echo
	@echo "Build finished. The HTML pages are in _build/html."
