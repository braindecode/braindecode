Welcome to Braindecode
======================

A deep learning toolbox to decode raw time-domain EEG.

For EEG researchers that want to want to work with deep learning and
deep learning researchers that want to work with EEG data.
For now focussed on convolutional networks.


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


Troubleshooting
===============
Please report any issues on github: https://github.com/robintibor/braindecode


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
