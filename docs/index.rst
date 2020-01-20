Welcome to Braindecode
======================

A deep learning toolbox to decode raw time-domain EEG.

For EEG researchers that want to work with deep learning and
deep learning researchers that want to work with EEG data.
For now focussed on convolutional networks.


Installation
============

1. Install pytorch from http://pytorch.org/ (you don't need to install torchvision).

2. Install numpy (necessary for resamply installation to work), e.g.:

.. code-block:: bash

  pip install numpy

3. Install braindecode via pip:

.. code-block:: bash

  pip install braindecode


Install the latest version of braindecode via pip:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  pip install -U https://api.github.com/repos/braindecode/braindecode/zipball/master


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


.. Tutorials
.. =========
.. .. toctree::
..    :maxdepth: 1

..    notebooks/Trialwise_Decoding.ipynb
..    notebooks/Cropped_Decoding.ipynb
..    notebooks/Trialwise_Manual_Training_Loop.ipynb
..    notebooks/Cropped_Manual_Training_Loop.ipynb
..    notebooks/visualization/Perturbation.ipynb


Troubleshooting
===============

Please report any issues on github: https://github.com/braindecode/braindecode/issues


.. API
.. ===

.. .. autosummary::
..    :toctree: source

..    braindecode.datasets
..    braindecode.experiments
..    braindecode.mne_ext
..    braindecode.models
..    braindecode.torch_ext
..    braindecode.visualization

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

as well as the `MNE-Python <https://mne.tools>`_` software that is used by braindecode:

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
