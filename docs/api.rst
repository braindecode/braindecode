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

Classifier
==========

:py:mod:`braindecode.classifier`:

.. currentmodule:: braindecode.classifier

.. autosummary::
   :toctree: generated/

    EEGClassifier

Regressor
==========

:py:mod:`braindecode.regressor`:

.. currentmodule:: braindecode.regressor

.. autosummary::
   :toctree: generated/

    EEGRegressor

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

Training
==========

:py:mod:`braindecode.training`:

.. currentmodule:: braindecode.training

.. autosummary::
   :toctree: generated/

    CroppedLoss
    CroppedTrialEpochScoring
    PostEpochTrainScoring
    trial_preds_from_window_preds

Datasets
==========

:py:mod:`braindecode.datasets`:

.. currentmodule:: braindecode.datasets

.. autosummary::
   :toctree: generated/

    BaseDataset
    BaseConcatDataset
    WindowsDataset
    MOABBDataset
    create_from_X_y
    create_from_mne_raw
    create_from_mne_epochs


Data Utils
==========

:py:mod:`braindecode.datautil`:

.. currentmodule:: braindecode.datautil

.. autosummary::
   :toctree: generated/

    create_fixed_length_windows
    create_windows_from_events
    exponential_moving_demean
    exponential_moving_standardize
    zscore
    scale

Utils
=====

:py:mod:`braindecode.util`:

.. currentmodule:: braindecode.util

.. autosummary::
   :toctree: generated/

    set_random_seeds
