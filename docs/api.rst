:orphan:

.. _api_reference:

=============
API Reference
=============

This is the reference for classes (``CamelCase`` names) and functions
(``underscore_case`` names) of Braindecode.

.. contents::
   :local:
   :depth: 2


:py:mod:`braindecode`:

.. automodule:: braindecode
   :no-members:
   :no-inherited-members:

Models
======

:py:mod:`braindecode.models`:

.. currentmodule:: braindecode.models

.. autosummary::
   :toctree: generated/

    ShallowFBCSPNet
    Deep4Net
    EEGNetv1
    EEGNetv4
    HybridNet
    EEGResNet

Datasets
==========

:py:mod:`braindecode.datasets`:

.. currentmodule:: braindecode.datasets

.. autosummary::
   :toctree: generated/

    WindowsDataset
    MOABBDataset


Data Utils
==========

:py:mod:`braindecode.datautil`:

.. currentmodule:: braindecode.datautil

.. autosummary::
   :toctree: generated/

    SignalAndTarget
    CropsDataLoader
    BaseWindower
    FixedLengthWindower
    EventWindower
    FilterRaw
    ZscoreRaw
    FilterWindow
    ZscoreWindow

Utils
=====

:py:mod:`braindecode.util`:

.. currentmodule:: braindecode.util

.. autosummary::
   :toctree: generated/

    set_random_seeds
