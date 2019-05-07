# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = /home/schirrmr/anaconda3/envs/default/bin/python3.6 -msphinx
SPHINXPROJ    = Braindecode
SOURCEDIR     = .
BUILDDIR      = ../../braindecode-docs/

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
