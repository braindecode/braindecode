Welcome to Braindecode
======================

**Under construction**

A deep learning toolbox to decode raw time-domain EEG.


Installation
============

1. Install pytorch from http://pytorch.org/ (you don't need to install torchvision).

2. Install numpy (necessary for resamply installation to work), e.g.:

.. code-block:: bash

  pip install numpy

3. Install newest version of python-mne:

.. code-block:: bash

  git clone git://github.com/mne-tools/mne-python.git
  cd mne-python
  python setup.py install


4. Install braindecode via pip:

.. code-block:: bash

  pip install braindecode



Tutorials
=========
.. toctree::
   :maxdepth: 1

   notebooks/TrialWise_Decoding.ipynb
   notebooks/Cropped_Decoding.ipynb
   notebooks/Experiment_Class.ipynb
   notebooks/visualization/Perturbation.ipynb


API
===

.. autosummary::
   :toctree: source

   braindecode.datautil
   braindecode.experiments
   braindecode.mne_ext
   braindecode.models
   braindecode.torch_ext

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _GitHub: https://github.com/robintibor/braindecode
