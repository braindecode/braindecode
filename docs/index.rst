Welcome to Braindecode
======================

A deep learning toolbox to decode raw time-domain EEG.

For EEG researchers that want to work with deep learning and
deep learning researchers that want to work with EEG data.
For now focussed on convolutional networks.

Installation
============

1. Install pytorch from http://pytorch.org/ (you don't need to install torchvision).

2. Install `MOABB <https://github.com/NeuroTechX/moabb>`_ via pip (needed if you want to use MOABB datasets utilities):

.. code-block:: bash

  pip install moabb

3. Install braindecode via pip:

.. code-block:: bash

  pip install braindecode

alternatively, if you use conda, you could create a dedicated environment with the following:

.. code-block:: bash

  curl -O https://raw.githubusercontent.com/braindecode/braindecode/master/environment.yml
  conda env create -f environment.yml
  conda activate braindecode

alternatively, install the latest version of braindecode via pip:

.. code-block:: bash

  pip install -U https://api.github.com/repos/braindecode/braindecode/zipball/master


Get Started
===========

Learn how to use braindecode for ...

.. toctree::
   :maxdepth: 1

    Basic trialwise decoding <auto_examples/plot_bcic_iv_2a_moabb_trial.rst>
    More data-efficient "cropped decoding" <auto_examples/plot_bcic_iv_2a_moabb_cropped.rst>
    Your own datasets through MNE <auto_examples/plot_mne_dataset_example.rst>
    Your own datasets through Numpy <auto_examples/plot_custom_dataset_example.rst>

Examples
========
.. toctree::
   :maxdepth: 1

   auto_examples/index

Public API
==========
.. toctree::
   :maxdepth: 1

   api

What's new
==========
.. toctree::
   :maxdepth: 1

   whats_new

Troubleshooting
===============

Please report any issues on github: https://github.com/braindecode/braindecode/issues



Citing
======

If you use this code in a scientific publication, please cite us as:

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _GitHub: https://github.com/braindecode/braindecode
