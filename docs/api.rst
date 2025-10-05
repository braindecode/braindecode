:orphan:

.. _api_reference:

=======================
Braindece API Reference
=======================

.. automodule:: braindecode
   :no-members:
   :no-inherited-members:

Models
======

Model zoo availables in braindecode. The models are implemented as
``PyTorch`` :py:class:`torch.nn.Modules` and can be used for various EEG decoding ways tasks.

All the models have the convention of having the signal related parameters
named the same way, following the braindecode's standards:

+ :fa:`shapes`\  ``n_outputs``: Number of labels or outputs of the model.
+ :fa:`wave-square`\  ``n_chans``: Number of EEG channels.
+ :fa:`clock`\  ``n_times``: Number of time points of the input window.
+ :fa:`wifi`\  ``sfreq``: Sampling frequency of the EEG recordings.
+ (:fa:`clock`\ / :fa:`wifi`\)  ``input_window_seconds``: Length of the input window in seconds.
+ :fa:`info-circle`\  ``chs_info``: Information about each individual EEG channel. Refer to :class:`mne.Info["chs"]`.
+ :fa:`shapes`\  ``n_outputs``: Number of labels or outputs of the model.
+ :fa:`wave-square`\  ``n_chans``: Number of EEG channels.
+ :fa:`clock`\  ``n_times``: Number of time points of the input window.
+ :fa:`wifi`\  ``sfreq``: Sampling frequency of the EEG recordings.
+ (:fa:`clock`\ / :fa:`wifi`\)  ``input_window_seconds``: Length of the input window in seconds.
+ :fa:`info-circle`\  ``chs_info``: Information about each individual EEG channel. Refer to :class:`mne.Info["chs"]`.

All the models assume that the input data is a 3D tensor of shape
``(batch_size, n_chans, n_times)``, and some models also accept a 4D tensor of shape
``(batch_size, n_chans, n_times, n_epochs)``, in case of cropped model.

All the models are implemented as subclasses of :py:class:`EEGModuleMixin`, which is a
base class for all EEG models in Braindecode. The :class:`EEGModuleMixin` class
provides a common interface for all EEG models and derivate variables names if necessary.

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
    AttnSleep
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
    EEGNet
    EEGNeX
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

    SPARCNet
    SyncNet
    TIDNet
    TSception
    USleep


Modules
=======

:py:mod:`braindecode.modules`:

This module contains the building blocks for Braindecode models. It
contains activation functions, convolutional layers, attention mechanisms,
filter banks, and other utilities.

.. currentmodule:: braindecode.modules

Activation
''''''''''
These modules wrap specialized activation functions—e.g., safe logarithms for numerical stability.

:py:mod:`braindecode.modules.activation`:

.. autosummary::
    :toctree: generated/activation
    :recursive:

    LogActivation
    SafeLog

Attention
'''''''''

These modules implement various attention mechanisms, including
multi'head attention and squeeze and excitation layers.

:py:mod:`braindecode.modules.attention`:

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
''''''
These modules are specialized building blocks for neural networks,
including multi'layer perceptrons (MLPs) and inception blocks.

:py:mod:`braindecode.modules.blocks`:

.. autosummary::
    :toctree: generated/blocks
    :recursive:

    MLP
    FeedForwardBlock
    InceptionBlock

Convolution
'''''''''''
These modules implement constraints convolutional layers, including
depthwise convolutions and causal convolutions. They also include
convolutional layers with constraints and pooling layers.

:py:mod:`braindecode.modules.convolution`:

.. autosummary::
    :toctree: generated/convolution
    :recursive:

    AvgPool2dWithConv
    CausalConv1d
    CombinedConv
    Conv2dWithConstraint
    DepthwiseConv2d

Filter
''''''
These modules implement Filter Bank as Layer and generalizer Gaussian
layer.

:py:mod:`braindecode.modules.filter`:

.. autosummary::
    :toctree: generated/filter
    :recursive:

    FilterBankLayer
    GeneralizedGaussianFilter

Layers
''''''
These modules implement various types of layers, including dropout
layers, normalization layers, and time'distributed layers. They also
include layers for handling different input shapes and dimensions.

:py:mod:`braindecode.modules.layers`:

.. autosummary::
    :toctree: generated/layers
    :recursive:

    Chomp1d
    DropPath
    Ensure4d
    TimeDistributed

Linear
''''''
These modules implement linear layers with various constraints and
initializations. They include linear layers with max'norm constraints
and linear layers with specific initializations.

:py:mod:`braindecode.modules.linear`:

.. autosummary::
    :toctree: generated/linear
    :recursive:

    LinearWithConstraint
    MaxNormLinear

Stats
'''''
These modules implement statistical layers, including layers for
calculating the mean, standard deviation, and variance of input
data. They also include layers for calculating the log power and log
variance of input data. Mostly used on FilterBank models.

:py:mod:`braindecode.modules.stats`:

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
'''''''''
These modules implement various utility functions and classes for
change to cropped model.

:py:mod:`braindecode.modules.util`:

.. autosummary::
    :toctree: generated/util
    :recursive:

    aggregate_probas


Wrappers
''''''''
These modules implement wrappers for various types of models,
including wrappers for models with multiple outputs and wrappers for
models with intermediate outputs.

:py:mod:`braindecode.modules.wrapper`:

.. autosummary::
    :toctree: generated/wrapper
    :recursive:

    Expression
    IntermediateOutputWrapper


Functional
===========
:py:mod:`braindecode.functional`:

.. currentmodule:: braindecode.functional

The functional module contains various functions that can be used
like functional API.

.. autosummary::
    :toctree: generated
    :recursive:

     drop_path
     glorot_weight_zero_bias
     hilbert_freq
     identity
     plv_time
     rescale_parameter
     safe_log
     square


Datasets
========
:py:mod:`braindecode.datasets`:

.. currentmodule:: braindecode.datasets

Pytorch Datasets structure for common EEG datasets, and function to create the dataset from several
different data formats. The options available are: `Numpy Arrays`, `MNE Raw` and `MNE Epochs`.


Base classes
''''''''''''

.. autosummary::
   :toctree: generated/

    BaseConcatDataset
    BaseDataset
    WindowsDataset
    BIDSDataset
    BIDSEpochsDataset


Common Datasets
''''''''''''''''

.. autosummary::
   :toctree: generated/

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
''''''''''''''''''''''''''
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
and to create batches of data, used on Self'Supervised Learning
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
the data in a variety of ways, including time'domain transformations,
frequency'domain transformations, and spatial transformations.

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


The functional augmentation API contains the same transformations as the
transforms API, but they are implemented as functions.

:py:mod:`braindecode.augmentation.functional`:

.. currentmodule:: braindecode.augmentation.functional

.. autosummary::
   :toctree: generated/

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


Classifier
==========

Skorch wrapper for braindecode models. The skorch wrapper
allows to use braindecode models with scikit'learn
API.

:py:mod:`braindecode.classifier`:

.. currentmodule:: braindecode.classifier

.. autosummary::
   :toctree: generated/

    EEGClassifier

Regressor
=========

Skorch wrapper for braindecode models focus on regression tasks.
The skorch wrapper allows to use braindecode models with scikit'learn
API.

:py:mod:`braindecode.regressor`:

.. currentmodule:: braindecode.regressor

.. autosummary::
   :toctree: generated/

    EEGRegressor


Training
========

Training module contains functions and classes for training
and evaluating EEG models. It is inside the Classifier and
Regressor skorch classes, and it is used to train the models
and evaluate their performance.

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

Utils
=====
Functions available in braindecode util module.

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
