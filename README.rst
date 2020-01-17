Braindecode
===========

.. image:: https://badges.gitter.im/braindecodechat/community.svg
   :alt: Join the chat at https://gitter.im/braindecodechat/community
   :target: https://gitter.im/braindecodechat/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

.. image:: https://travis-ci.org/braindecode/braindecode.svg?branch=master
   :target: https://travis-ci.org/braindecode/braindecode

.. image:: https://circleci.com/gh/braindecode/braindecode.svg?style=svg
   :target: https://circleci.com/gh/braindecode/braindecode
   :alt: Doc build on CircleCI

.. image:: https://codecov.io/gh/braindecode/braindecode/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/braindecode/braindecode
   :alt: Code Coverage

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

3. Install latest release of braindecode via pip:

.. code-block:: bash

  pip install braindecode

alternatively, if you use conda, you could create a dedicated environment with the following:

.. code-block:: bash

	curl -O https://raw.githubusercontent.com/braindecode/braindecode/master/environment.yml
	conda env create -f environment.yml
	conda activate braindecode

Install the latest version of braindecode via pip:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  pip install -U https://api.github.com/repos/braindecode/braindecode/zipball/master


Documentation
=============

Documentation is online under https://braindecode.github.io/


Dataset
=======
The high-gamma dataset used in our publication (see below), including trained models, is available under:
https://web.gin.g-node.org/robintibor/high-gamma-dataset/

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
