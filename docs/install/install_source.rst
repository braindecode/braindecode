.. _install_source:

Installing from sources
~~~~~~~~~~~~~~~~~~~~~~~

If you want to test features under development or contribute to the library, or if you want to test the new tools that have been tested in braindecode and not released yet, this is the right tutorial for you!

.. note::

   If you are only trying to install Braindecode, we recommend using the pip installation `Installation <https://braindecode.org/braindecode/install/install_pip.html#install-pip>`__ for details on that.

.. _system-level:

Clone the repository from GitHub
--------------------------------

The first thing you should do is clone the Braindecode repository to your computer and enter inside the repository.

.. code-block:: bash

   git clone https://github.com/braindecode/braindecode && cd braindecode

You should now be in the root directory of the Braindecode repository.

Installing Braindecode from the source
--------------------------------------

If you want to only install Braindecode from source once and not do any development
work, then the recommended way to build and install is to use ``pip``::

For the latest development version, directly from GitHub:

.. code-block:: bash

  pip install -U https://api.github.com/repos/braindecode/braindecode/zipball/master#egg=braindecode

If you have a local clone of the Braindecode git repository:

.. code-block:: bash

   pip install -e .

This will install Braindecode in editable mode, i.e., changes to the source code could be used
directly in python.

You could also install optional dependency, like to import datasets from MOABB.

.. code-block:: bash

   pip install -e .[moabb]

There is also optional dependencies for unit testing and building documentation, you could install
them if you want to contribute to Braindecode.

.. code-block:: bash

   pip install -e .[moabb,tests,docs]


Testing if your installation is working
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To verify that Braindecode is installed and running correctly, run the following command:

.. code-block:: console

   python -m "import braindecode; braindecode.__version__"

For more information, please see the `contributors' guidelines <https://github.com/braindecode/braindecode/blob/master/CONTRIBUTING.md>`__.
