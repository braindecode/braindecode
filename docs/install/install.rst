.. _installation:

##############
 Installation
##############

Braindecode requires **Python >= 3.11**, **PyTorch >= 2.6**, and **TorchAudio >= 2.6**
on Linux (``x86_64``), Windows (``AMD64``), or macOS with Apple Silicon (``arm64``;
Intel Macs are unsupported because `PyTorch ended macOS x86_64 builds after 2.2
<https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690>`_).

Packages are published on `PyPI <braindecode-pypi_>`_ and developed openly on
`GitHub <braindecode-github_>`_.

***************
 Quick install
***************

Install the core library from PyPI, or include optional extras for Hugging Face Hub
pretrained weights (``hub``) and benchmark dataset fetchers (``moabb``):

.. code-block:: bash

    pip install braindecode
    pip install "braindecode[hub]"

*********************
 Installation guides
*********************

Choose the setup guide matching your workflow:

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: Install from PyPI

        Recommended for research, clinical pipelines, and benchmarking. Covers core
        installation, optional extras (``hub``, ``moabb``, ``eegprep``, ``viz``), and
        CUDA GPU setup.

        .. code-block:: bash

            pip install "braindecode[hub,moabb]"

        +++

        .. button-ref:: install_pip
            :ref-type: ref
            :color: primary
            :expand:

            PyPI installation guide →

    .. grid-item-card:: Build from source

        Recommended for contributors and users testing unreleased changes from ``master``.
        Covers editable ``pip install -e .`` setups, test dependencies, and verification.

        .. code-block:: bash

            git clone https://github.com/braindecode/braindecode
            pip install -e ".[tests,docs]"

        +++

        .. button-ref:: install_source
            :ref-type: ref
            :color: primary
            :expand:

            Source & contributor guide →

To run training workloads on Intel Gaudi accelerators, see :ref:`install_gaudi`.

.. toctree::
    :hidden:

    install_pip
    install_source
    gaudi

.. include:: /links.inc
