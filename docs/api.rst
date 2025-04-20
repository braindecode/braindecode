:orphan:

.. _api_reference:

=============
API Reference
=============

This is the reference for classes (``CamelCase`` names) and functions
(``underscore_case`` names) of Braindecode.


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
   :recursive:
   
    EEGModuleMixin

:py:mod:`braindecode.models`:

.. currentmodule:: braindecode.models

.. autosummary::
   :toctree: generated/
   :recursive:
   
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
    EEGMiner
    EEGNetv1
    EEGNetv4
    EEGNeX
    EEGResNet
    EEGSimpleConv
    EEGTCNet
    FBCNet
    FBLightConvNet
    FBMSNet
    IFNet
    Labram
    MSVTNet
    SCCNet
    ShallowFBCSPNet
    SignalJEPA
    SignalJEPA_Contextual
    SignalJEPA_PostLocal
    SignalJEPA_PreLocal
    SincShallowNet
    SleepStagerBlanco2020
    SleepStagerChambon2018
    SleepStagerEldele2021
    SPARCNet
    SyncNet
    TIDNet
    TSceptionV1
    USleep


Modules
=======

:py:mod:`braindecode.modules`:

This module contains the building blocks for Braindecode models. It
contains activation functions, convolutional layers, attention mechanisms,
filter banks, and other utilities.

.. currentmodule:: braindecode.modules

Activation
----------
These modules wrap specialized activation functions—e.g., safe logarithms for numerical stability.

.. autosummary::
    :toctree: generated/activation
    :recursive:
   
    LogActivation
    SafeLog

Attention
---------

These modules implement various attention mechanisms, including
multi-head attention and squeeze-and-excitation layers.

.. autosummary::
    :toctree: generated/attention
    :recursive:
   
    CAT
    CBAM
    ECA
    FCA
    GCT
    SRM
    CATLite
    EncNet
    GatherExcite
    GSoP
    MultiHeadAttention
    SqueezeAndExcitation

Blocks
------
These modules are specialized building blocks for neural networks,
including multi-layer perceptrons (MLPs) and inception blocks.

.. autosummary::
    :toctree: generated/blocks
    :recursive:
   
    MLP
    FeedForwardBlock
    InceptionBlock

Convolution
-----------
These modules implement constraints convolutional layers, including
depthwise convolutions and causal convolutions. They also include
convolutional layers with constraints and pooling layers.

.. autosummary::
    :toctree: generated/convolution
    :recursive:
   
    AvgPool2dWithConv
    CausalConv1d
    CombinedConv
    Conv2dWithConstraint
    DepthwiseConv2d

Filter
------
These modules implement Filter Bank as Layer and generalizer Gaussian
layer. 

.. autosummary::
    :toctree: generated/filter
    :recursive:
   
    FilterBankLayer
    GeneralizedGaussianFilter

Layers
------
These modules implement various types of layers, including dropout
layers, normalization layers, and time-distributed layers. They also
include layers for handling different input shapes and dimensions.

.. autosummary::
    :toctree: generated/layers
    :recursive:
   
    Chomp1d
    DropPath
    Ensure4d
    TimeDistributed

Linear
------
These modules implement linear layers with various constraints and
initializations. They include linear layers with max-norm constraints
and linear layers with specific initializations.

.. autosummary::
    :toctree: generated/linear
    :recursive:
   
    LinearWithConstraint
    MaxNormLinear

Stats
-----
These modules implement statistical layers, including layers for
calculating the mean, standard deviation, and variance of input
data. They also include layers for calculating the log power and log
variance of input data. Mostly used on FilterBank models.

.. autosummary::
    :toctree: generated/stats
    :recursive:
       
    StatLayer
    LogPowerLayer
    LogVarLayer
    MaxLayer
    MeanLayer
    StdLayer
    VarLayer

Utilities
---------
These modules implement various utility functions and classes for
change to cropped model.

.. autosummary::
    :toctree: generated/util
    :recursive:
   
    get_output_shape
    to_dense_prediction_model

Wrappers
--------
These modules implement wrappers for various types of models,
including wrappers for models with multiple outputs and wrappers for
models with intermediate outputs. 

.. autosummary::
    :toctree: generated/wrapper
    :recursive:
   
    Expression
    IntermediateOutputWrapper


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
.. currentmodule:: braindecode.datasets
:py:mod:`braindecode.datasets`:

Pytorch Datasets structure for common EEG datasets, and function to create the dataset from several
different data formats. The options available are: `Numpy Arrays`, `MNE Raw` and `MNE Epochs`. 


Base classes
------------

.. autosummary::
   :toctree: generated/

    BaseConcatDataset
    BaseDataset
    WindowsDataset
    BIDSDataset
    BIDSEpochsDataset

   
Common Datasets
----------------

.. autosummary::
   :toctree: generated/
   :recursive:

    BCICompetitionIVDataset4
    BNCI2014001
    HGD
    MOABBDataset
    NMT
    SleepPhysionet
    SleepPhysionetChallenge2018
    TUH
    TUHAbnormal


Dataset Builders Functions
--------------------------
Functions to create datasets from different data formats


.. currentmodule:: braindecode.datasets

.. autosummary::
   :toctree: generated/

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
Samplers that can used to sample EEG data for training and testing
and to create batches of data, used on Self-Supervised Learning
and other tasks.

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

The augmentation module follow the pytorch transforms API. It contains
transformations that can be applied to EEG data. The transformations
can be used to augment the data during training, which can help improve
the performance of the model. The transformations can be applied to
the data in a variety of ways, including time-domain transformations,
frequency-domain transformations, and spatial transformations. 

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


Functional API
--------------
The functional API contains the same transformations as the
transforms API, but they are implemented as functions. 

.. currentmodule:: braindecode.augmentation.functional

.. autosummary::
    identity
    time_reverse
    sign_flip
    ft_surrogate
    channels_dropout
    channels_shuffle
    channels_permute
    gaussian_noise
    smooth_time_mask
    bandstop_filter
    frequency_shift
    sensors_rotation
    mixup
    segmentation_reconstruction
    mask_encoding


Utils
=====
Util functions available in braindecode util module. 

:py:mod:`braindecode.util`:

.. currentmodule:: braindecode.util

.. autosummary::
   :toctree: generated/

    set_random_seeds

Visualization
=============
Visualization module contains functions for visualizing EEG data,
including plotting the confusion matrix and computing amplitude
gradients. The visualization module is useful for understanding the
performance of the model and for interpreting the results. 

:py:mod:`braindecode.visualization`:

.. currentmodule:: braindecode.visualization

.. autosummary::
   :toctree: generated/

    compute_amplitude_gradients
    plot_confusion_matrix
