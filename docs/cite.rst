:orphan:

.. _cite:

#########################
 How to cite Braindecode
#########################

************************
 Citing the Braindecode
************************

If you use Braindecode in scientific work, please cite the software using the global
Zenodo DOI shown in the badge below:

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.17699192.svg
    :target: https://doi.org/10.5281/zenodo.17699192
    :alt: DOI

You can use the following BibTeX entry:

.. code-block:: bibtex

    @software{braindecode,
      author = {Aristimunha, Bruno and
                Guetschel, Pierre and
                Wimpff, Martin and
                Gemein, Lukas and
                Rommel, Cedric and
                Banville, Hubert and
                Sliwowski, Maciej and
                Wilson, Daniel and
                Brandt, Simon and
                Gnassounou, Théo and
                Paillard, Joseph and
                {Junqueira Lopes}, Bruna and
                Sedlar, Sara and
                Moreau, Thomas and
                Chevallier, Sylvain and
                Gramfort, Alexandre and
                Schirrmeister, Robin Tibor},
      title = {Braindecode: toolbox for decoding raw electrophysiological brain data
               with deep learning models},
      url = {https://github.com/braindecode/braindecode},
      doi = {10.5281/zenodo.17699192},
      publisher = {Zenodo},
      license = {BSD-3-Clause},
    }

Or the following APA reference:

    Aristimunha, B., Guetschel, P., Wimpff, M., Gemein, L., Rommel, C., Banville, H.,
    Sliwowski, M., Wilson, D., Brandt, S., Gnassounou, T., Paillard, J., Junqueira
    Lopes, B., Sedlar, S., Moreau, T., Chevallier, S., Gramfort, A., & Schirrmeister, R.
    T. *Braindecode: toolbox for decoding raw electrophysiological brain data with deep
    learning models* [Software]. Zenodo. https://doi.org/10.5281/zenodo.17699192

Additionally, we highly encourage you to cite the article that originally introduced the
Braindecode library and has served as a foundational reference for many works on deep
learning with EEG recordings. Please use the following reference:

.. code-block:: bibtex

    @article {HBM:HBM23730,
    author = {Schirrmeister, Robin Tibor and Springenberg, Jost Tobias and Fiederer,
      Lukas Dominique Josef and Glasstetter, Martin and Eggensperger, Katharina and Tangermann, Michael and
      Hutter, Frank and Burgard, Wolfram and Ball, Tonio},
    title = {Deep learning with convolutional neural networks for EEG decoding and visualization},
    journal = {Human Brain Mapping},
    issn = {1097-0193},
    url = {http://dx.doi.org/10.1002/hbm.23730},
    doi = {10.1002/hbm.23730},
    month = {aug},
    year = {2017},
    keywords = {electroencephalography, EEG analysis, machine learning, end-to-end learning, brain–machine interface,
      brain–computer interface, model interpretability, brain mapping},
    }

****************
 Citing the MNE
****************

Braindecode is built on top of the `MNE-Python <https://mne.tools>`_ software. you can
use the DOIs provided by `Zenodo <https://doi.org/10.5281/zenodo.592483>`_.
Additionally, we also ask that when citing the MNE-Python package, you cite the
canonical journal article reference below:

.. code-block:: bibtex

    @article{10.3389/fnins.2013.00267,
    author={Gramfort, Alexandre and Luessi, Martin and Larson, Eric and Engemann,
            Denis and Strohmeier, Daniel and Brodbeck, Christian and Goj, Roman and Jas,
            Mainak and Brooks, Teon and Parkkonen, Lauri and Hämäläinen, Matti},
    title={{MEG and EEG data analysis with MNE-Python}},
    journal={Frontiers in Neuroscience},
    volume={7},
    pages={267},
    year={2013},
    url={https://www.frontiersin.org/article/10.3389/fnins.2013.00267},
    doi={10.3389/fnins.2013.00267},
    issn={1662-453X},
    }

*************************
 Citing other algorithms
*************************

Depending on your research topic, it may also be appropriate to cite related method
papers, some of which are listed in the documentation strings of the relevant functions
or methods. We recommend always check the documentation of the function.
