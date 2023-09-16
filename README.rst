.. image:: https://badges.gitter.im/braindecodechat/community.svg
   :alt: Join the chat at https://gitter.im/braindecodechat/community
   :target: https://gitter.im/braindecodechat/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

.. image:: https://github.com/braindecode/braindecode/workflows/docs/badge.svg
   :target: https://github.com/braindecode/braindecode/actions

.. image:: https://circleci.com/gh/braindecode/braindecode.svg?style=svg
   :target: https://circleci.com/gh/braindecode/braindecode
   :alt: Doc build on CircleCI

.. image:: https://codecov.io/gh/braindecode/braindecode/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/braindecode/braindecode
   :alt: Code Coverage

.. |Braindecode| image:: https://user-images.githubusercontent.com/42702466/177958779-b00628aa-9155-4c51-96d1-d8c345aff575.svg
.. _braindecode: braindecode.org/

Braindecode
===========

Braindecode is an open-source Python toolbox for decoding raw electrophysiological brain
data with deep learning models. It includes dataset fetchers, data preprocessing and
visualization tools, as well as implementations of several deep learning
architectures and data augmentations for analysis of EEG, ECoG and MEG.

For neuroscientists who want to work with deep learning and
deep learning researchers who want to work with neurophysiological data.


Installation Braindecode
========================

1. Install pytorch from http://pytorch.org/ (you don't need to install torchvision).

2. If you want to download EEG datasets from `MOABB <https://github.com/NeuroTechX/moabb>`_, install it:

.. code-block:: bash

  pip install moabb

3. Install latest release of braindecode via pip:

.. code-block:: bash

  pip install braindecode

If you want to install the latest development version of braindecode, please refer to `contributing page <https://github.com/braindecode/braindecode/blob/master/CONTRIBUTING.md>`__


Documentation
=============

Documentation is online under https://braindecode.org, both in the stable and dev versions.


Contributing to Braindecode
===========================

Guidelines for contributing to the library can be found on the braindecode github:

https://github.com/braindecode/braindecode/blob/master/CONTRIBUTING.md

Braindecode chat
================

https://gitter.im/braindecodechat/community

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




Licensing
^^^^^^^^^

Braindecode is **BSD-licenced** (BSD-3-Clause):

    This software is OSI Certified Open Source Software.
    OSI Certified is a certification mark of the Open Source Initiative.

    Copyright (c) 2011-2022, authors of Braindecode.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the names of braindecode authors nor the names of any
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    **This software is provided by the copyright holders and contributors
    "as is" and any express or implied warranties, including, but not
    limited to, the implied warranties of merchantability and fitness for
    a particular purpose are disclaimed. In no event shall the copyright
    owner or contributors be liable for any direct, indirect, incidental,
    special, exemplary, or consequential damages (including, but not
    limited to, procurement of substitute goods or services; loss of use,
    data, or profits; or business interruption) however caused and on any
    theory of liability, whether in contract, strict liability, or tort
    (including negligence or otherwise) arising in any way out of the use
    of this software, even if advised of the possibility of such
    damage.**

