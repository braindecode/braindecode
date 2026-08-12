.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.17699192.svg
    :target: https://doi.org/10.5281/zenodo.17699192
    :alt: DOI

.. image:: https://github.com/braindecode/braindecode/workflows/docs/badge.svg
    :target: https://github.com/braindecode/braindecode/actions/workflows/docs.yml
    :alt: Docs Build Status

.. image:: https://github.com/braindecode/braindecode/workflows/tests/badge.svg
    :target: https://github.com/braindecode/braindecode/actions/workflows/tests.yml
    :alt: Test Build Status

.. image:: https://codecov.io/gh/braindecode/braindecode/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/braindecode/braindecode
    :alt: Code Coverage

.. image:: https://img.shields.io/pypi/v/braindecode?color=blue&style=flat-square
    :target: https://pypi.org/project/braindecode/
    :alt: PyPI

.. image:: https://img.shields.io/pypi/v/braindecode?label=version&color=orange&style=flat-square
    :target: https://pypi.org/project/braindecode/
    :alt: Version

.. image:: https://img.shields.io/pypi/pyversions/braindecode?style=flat-square
    :target: https://pypi.org/project/braindecode/
    :alt: Python versions

.. image:: https://pepy.tech/badge/braindecode
    :target: https://pepy.tech/project/braindecode
    :alt: Downloads

.. |Braindecode| image:: https://user-images.githubusercontent.com/42702466/177958779-b00628aa-9155-4c51-96d1-d8c345aff575.svg

.. _braindecode: braindecode.org/

#############
 Braindecode
#############

Braindecode is an open-source Python toolbox for decoding raw electrophysiological brain
data with deep learning models. It includes dataset fetchers, data preprocessing and
visualization tools, as well as implementations of several deep learning architectures
and data augmentations for analysis of EEG, ECoG and MEG.

For neuroscientists who want to work with deep learning and deep learning researchers
who want to work with neurophysiological data.

##########################
 Installation Braindecode
##########################

1. Install pytorch from http://pytorch.org/ (you don't need to install torchvision).
2. If you want to download EEG datasets from `MOABB
   <https://github.com/NeuroTechX/moabb>`_, install it:

.. code-block:: bash

    pip install moabb

3. Install latest release of braindecode via pip:

.. code-block:: bash

    pip install braindecode

If you want to install the latest development version of braindecode, please refer to
`contributing page
<https://github.com/braindecode/braindecode/blob/master/CONTRIBUTING.md>`__

###############
 Documentation
###############

Documentation is online under https://braindecode.org, both in the stable and dev
versions.

#############################
 Contributing to Braindecode
#############################

Guidelines for contributing to the library can be found on the braindecode github:

https://github.com/braindecode/braindecode/blob/master/CONTRIBUTING.md

########
 Citing
########

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

as well as the `MNE-Python <https://mne.tools>`_ software that is used by braindecode:

.. code-block:: bibtex

    @article{10.3389/fnins.2013.00267,
    author={Gramfort, Alexandre and Luessi, Martin and Larson, Eric and Engemann, Denis and Strohmeier, Daniel and Brodbeck, Christian and Goj, Roman and Jas, Mainak and Brooks, Teon and Parkkonen, Lauri and Hämäläinen, Matti},
    title={{MEG and EEG data analysis with MNE-Python}},
    journal={Frontiers in Neuroscience},
    volume={7},
    pages={267},
    year={2013},
    url={https://www.frontiersin.org/article/10.3389/fnins.2013.00267},
    doi={10.3389/fnins.2013.00267},
    issn={1662-453X},
    }

***********
 Licensing
***********

This project is primarily licensed under the BSD-3-Clause License.

Additional Components
=====================

Some components within this repository are licensed under the Creative Commons
Attribution-NonCommercial 4.0 International License.

Please refer to the ``LICENSE`` and ``NOTICE`` files for more detailed information.
