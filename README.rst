Note: The old braindecode repository has been moved to
https://github.com/robintibor/braindevel.

Braindecode
===========

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



Documentation
=============

Documentation is online under https://robintibor.github.io/braindecode/
