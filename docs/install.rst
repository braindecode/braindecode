.. include:: links.inc


Install via :code:`pip` or :code:`conda`
========================================

Braindecode requires two libraries already installed before being installed, ``PyTorch`` and ``moabb`` if you want to use MOABB datasets utilities.

Preparing the environment for braindecode installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We suggest installing the Pytorch into its own ``conda`` or ``pip`` environment.
Installing the PyTorch library depends on the architecture and operational system you are running, as well as driver versions and optimizations for deep learning.

Given this complexity, we recommend that you go directly to the ``pytorch``
page and follow the instructions in http://pytorch.org/. You don't need to install torchvision.

To install moabb, run in your terminal:

.. code-block:: console

   $ pip install moabb

Installing the braindecode with pip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
After preparing the environment for braindecode, you can install via :code:`pip`:

.. code-block:: console

   $ pip install braindecode

alternatively, you can install the latest version of braindecode via pip:


.. code-block:: console

  pip install -U https://api.github.com/repos/braindecode/braindecode/zipball/master


Installing Braindecode with conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also create a conda environment for installing the library.
This might help depending on how you are managing other libraries.

.. code-block:: bash

  curl -O https://raw.githubusercontent.com/braindecode/braindecode/master/environment.yml
  conda env create -f environment.yml
  conda activate braindecode

This will create a new ``conda`` environment called ``braindecode`` (you can adjust
this by passing a different name via ``--name``).


