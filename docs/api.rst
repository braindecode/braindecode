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
=========

:py:mod:`braindecode.regressor`:

.. currentmodule:: braindecode.regressor

.. autosummary::
   :toctree: generated/

    EEGRegressor

Models
======

:py:mod:`braindecode.models.base`:

.. currentmodule:: braindecode.models

.. autosummary::
   :toctree: generated/

    EEGModuleMixin

:py:mod:`braindecode.models`:

.. currentmodule:: braindecode.models

.. autosummary::
   :toctree: generated/

    ATCNet
    AttentionBaseNet
    BDTCN
    BIOT
    ContraWR
    CTNet
    Deep4Net
    DeepSleepNet
    EEGConformer
    EEGInceptionERP
    EEGInceptionMI
    EEGITNet
    EEGNetv1
    EEGNetv4
    EEGNeX
    EEGMiner
    EEGResNet
    EEGSimpleConv
    EEGTCNet
    Labram
    MSVTNet
    SCCNet
    SignalJEPA
    SignalJEPA_Contextual
    SignalJEPA_PostLocal
    SignalJEPA_PreLocal
    SincShallowNet
    ShallowFBCSPNet
    SleepStagerBlanco2020
    SleepStagerChambon2018
    SleepStagerEldele2021
    SPARCNet
    SyncNet
    TSceptionV1
    TIDNet
    USleep

.. currentmodule:: braindecode.models.modules

:py:mod:`braindecode.models`:

.. autosummary::
   :toctree: generated/

    FilterBankLayer
    GeneralizedGaussianFilter

Training
========

:py:mod:`braindecode.training`:

.. currentmodule:: braindecode.training

.. autosummary::
   :toctree: generated/

    CroppedLoss
    TimeSeriesLoss
    CroppedTrialEpochScoring
    CroppedTimeSeriesEpochScoring
    PostEpochTrainScoring
    mixup_criterion
    trial_preds_from_window_preds
    predict_trials

Datasets
========

:py:mod:`braindecode.datasets`:

.. currentmodule:: braindecode.datasets

.. autosummary::
   :toctree: generated/

    BaseDataset
    BaseConcatDataset
    BIDSDataset
    BIDSEpochsDataset
    WindowsDataset
    MOABBDataset
    HGD
    BNCI2014001
    TUH
    TUHAbnormal
    NMT
    SleepPhysionet
    BCICompetitionIVDataset4
    create_from_X_y
    create_from_mne_raw
    create_from_mne_epochs

Preprocessing
=============

:py:mod:`braindecode.preprocessing`:

.. currentmodule:: braindecode.preprocessing

.. autosummary::
   :toctree: generated/

    create_windows_from_events
    create_fixed_length_windows
    create_windows_from_target_channels
    exponential_moving_demean
    exponential_moving_standardize
    filterbank
    preprocess
    Preprocessor
    Resample
    DropChannels
    SetEEGReference
    Filter
    Pick
    Crop

Data Utils
==========

:py:mod:`braindecode.datautil`:

.. currentmodule:: braindecode.datautil

.. autosummary::
   :toctree: generated/

    save_concat_dataset
    load_concat_dataset

Samplers
========

:py:mod:`braindecode.samplers`:

.. currentmodule:: braindecode.samplers

.. autosummary::
   :toctree: generated/

   RecordingSampler
   DistributedRecordingSampler
   SequenceSampler
   RelativePositioningSampler
   DistributedRelativePositioningSampler
   BalancedSequenceSampler

.. _augmentation_api:

Augmentation
============

:py:mod:`braindecode.augmentation`:

.. currentmodule:: braindecode.augmentation

.. autosummary::
   :toctree: generated/

    Transform
    IdentityTransform
    Compose
    AugmentedDataLoader
    TimeReverse
    SignFlip
    FTSurrogate
    ChannelsShuffle
    ChannelsDropout
    GaussianNoise
    ChannelsSymmetry
    SmoothTimeMask
    BandstopFilter
    FrequencyShift
    SensorsRotation
    SensorsZRotation
    SensorsYRotation
    SensorsXRotation
    Mixup
    SegmentationReconstruction
    MaskEncoding

    functional.identity
    functional.time_reverse
    functional.sign_flip
    functional.ft_surrogate
    functional.channels_dropout
    functional.channels_shuffle
    functional.channels_permute
    functional.gaussian_noise
    functional.smooth_time_mask
    functional.bandstop_filter
    functional.frequency_shift
    functional.sensors_rotation
    functional.mixup
    functional.segmentation_reconstruction
    functional.mask_encoding


Utils
=====

:py:mod:`braindecode.util`:

.. currentmodule:: braindecode.util

.. autosummary::
   :toctree: generated/

    set_random_seeds

Visualization
=============

:py:mod:`braindecode.visualization`:

.. currentmodule:: braindecode.visualization

.. autosummary::
   :toctree: generated/

    compute_amplitude_gradients
    plot_confusion_matrix
