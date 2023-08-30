.. _install_pip:

Installing from PyPI
~~~~~~~~~~~~~~~~~~~~

MOABB can be installed via pip from `PyPI <https://pypi.org/project/moabb>`__.

.. note::
    We recommend the most updated version of pip to install from PyPI.

Below are the installation commands for the most common use cases.

.. code-block:: console

   pip install bash

MOABB can also be installed with sets of optional dependencies to enable the deep learning modules, the tracker of the carbon emission module or the documentation environment.

For example, to install all the optional dependencies.


.. code-block:: bash

   pip install moabb[deepleaning, carbonemission, docs]

If you install the deep learning submodule using the pip, the installation commands above usually install a CPU variant of PyTorch or Tensorflow.

To use the potential of the deep learning modules (TensorFlow or PyTorch) with GPU, we recommend the following sequence before installing the moabb with dependencies:

#. Install the latest NVIDIA driver.
#. Check `PyTorch Official Guide <https://pytorch.org/get-started/locally/>`__ or `TensorFlow Official Guide <https://www.tensorflow.org/install/gpu>`__, for the recommended CUDA versions. For the Pip package, the user must download the CUDA manually, install it on the system, and ensure CUDA_PATH is appropriately set and working!
#. Continue to follow the guide and install PyTorch or TensorFlow.
#. Install moabb using one of the ways with the dependency parameter `[deeplearning].

See `Troubleshooting <moabb.Troubleshooting.com>`__ section if you have a problem.
