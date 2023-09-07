.. _install_pip:

Installing from PyPI
~~~~~~~~~~~~~~~~~~~~

Braincode can be installed via pip from `PyPI <https://pypi.org/project/braindecode>`__.

.. note::
    We recommend the most updated version of pip to install from PyPI.

Below are the installation commands for the most common use cases.

.. code-block:: console

   pip install braindecode

Braindecode can also be installed with sets of optional dependencies, for example to import datasets from MOABB:

.. code-block:: bash

   pip install braindecode[moabb]

There is also optional dependencies for unit test and building the documentation.

To use the potential of the deep learning modules PyTorch with GPU, we recommend the following sequence before installing the braindecode:

#. Install the latest NVIDIA driver.
#. Check `PyTorch Official Guide <https://pytorch.org/get-started/locally/>`__, for the recommended CUDA versions. For the Pip package, the user must download the CUDA manually, install it on the system, and ensure CUDA_PATH is appropriately set and working!
#. Continue to follow the guide and install PyTorch.

See `Troubleshooting <braindecode.faq>`__ section if you have a problem.
