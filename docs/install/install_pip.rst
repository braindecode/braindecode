.. _install_pip:

Installing from PyPI
~~~~~~~~~~~~~~~~~~~~

Braincode can be installed via pip from `PyPI <braindecode-pypi_>`_.

.. note::
    We recommend the most updated version of pip to install from PyPI.

Below are the installation commands for the most common use cases.

.. code-block:: console

   pip install braindecode

Braindecode can also be installed along `MOABB <moabb_>`_ to download open datasets:

.. code-block:: bash

   pip install moabb

To use the potential of the deep learning modules PyTorch with GPU, we recommend the following sequence before installing the braindecode:

#. Install the latest NVIDIA driver.
#. Check `PyTorch's <PyTorch_>`_ official guide, for the recommended CUDA versions. For the Pip package, the user must download the CUDA manually, install it on the system, and ensure CUDA_PATH is appropriately set and working!
#. Continue to follow the guide and install PyTorch.

See the :doc:`Frequently Asked Questions (FAQ) </help>` section if you have a problem.

.. include:: /links.inc
