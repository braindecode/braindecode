.. _install_source:

Installing from sources
~~~~~~~~~~~~~~~~~~~~

If you want to test features under development or contribute to the library, or if you want to test the new tools that have been tested in moabb and not released yet, this is the right tutorial for you!

.. note::

   If you are only trying to install Braindecode, we recommend using the pip installation `Installation <https://braindecode.org/braindecode/install/install_pip.html#install-pip>`__ for details on that.

.. _system-level:

Clone the repository from GitHub
--------------------------------------------------

The first thing you should do is clone the Braindecode repository to your computer and enter inside the repository.

.. code-block:: bash

   git clone https://github.com/braindecode/braindecode && cd braindecode

You should now be in the root directory of the Braindecode repository.

Installing Braindecode from the source
--------------------------------------------------------------------------------------------------------------------------------

If you want to only install Moabb from source once and not do any development
work, then the recommended way to build and install is to use ``pip``::

For the latest development version, directly from GitHub:

.. code-block:: bash

  pip install -U https://api.github.com/repos/braindecode/braindecode/zipball/master#egg=braindecode[moabb]

If you have a local clone of the MOABB git repository:

.. code-block:: bash

   pip install .

You can also install MOABB in editable mode (i.e. changes to the source code).

Testing if your installation is working
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To verify that Braindecode is installed and running correctly, run the following command:

.. code-block:: console

   python -m "import braindecode; braindecode.__version__"

For more information, please see the contributors' guidelines.
