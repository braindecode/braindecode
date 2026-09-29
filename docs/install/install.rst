.. _installation:

##############
 Installation
##############

Braindecode requires Python >= 3.11, PyTorch >= 2.4, and TorchAudio >= 2.4. macOS
requires Apple Silicon; Intel Macs are no longer supported because `PyTorch discontinued
macOS x86_64 binaries after 2.2
<https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690>`_.

The package is distributed via Python package index (`PyPI <braindecode-pypi_>`_), and
you can access the source code via `Github <braindecode-github_>`_ repository.

There are different ways to install Braindecode, depending on your needs and:

.. grid:: 2

    .. grid-item-card::
        :text-align: center

        .. rst-class:: font-weight-bold mb-0

            Install via ``pip``

        .. rst-class:: card-subtitle text-muted mt-0

            For Beginners

        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        .. image:: /_static/braindecode_install.png
           :alt: Braindecode Installer with pip

        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


        .. button-ref:: install_pip
            :ref-type: ref
            :color: primary
            :shadow:
            :class: font-weight-bold

            Installing from PyPI


    .. grid-item-card::
        :text-align: center

        .. rst-class:: font-weight-bold mb-0

           Building from source code

        .. rst-class:: card-subtitle text-muted mt-0

            For Advanced Users

        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        .. image:: https://mne.tools/stable/_images/mne_installer_console.png
           :alt: Terminal Window

        **Already familiar with Python?**
        Follow our setup instructions for building from Github and start to contribute!
        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        .. button-ref:: install_source
            :ref-type: ref
            :color: primary
            :shadow:
            :class: font-weight-bold

            From Source Code

.. toctree::
    :hidden:

    install_pip
    install_source

.. include:: /links.inc
