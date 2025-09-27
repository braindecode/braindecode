:Orphan:

.. _whats_new:

What's new
==========
.. currentmodule:: braindecode

.. NOTE: we are now using links to highlight new functions and classes.
    Please follow the examples below like :class:`braindecode.EEGClassifier`, so the
    whats_new page will have a link to the function/class documentation.

.. NOTE: there are 3 separate sections for changes, based on type:
    - "Enhancements" for new features
    - "Bugs" for bug fixes
    - "API changes" for backward-incompatible changes

.. _current:

Current 1.3 (dev)
----------------------

Enhancements
~~~~~~~~~~~~

API changes
~~~~~~~~~~~

Bugs
~~~~
- Fixing the issues with :class:`braindecode.models.ATCNet` (:gh:`782` by `Hamdi Altaheri`_)


Current 1.2 (stable)
----------------------

Enhancements
~~~~~~~~~~~~
- Improving the docstring for :class:`braindecode.models.EEGNetv4`  (:gh:`768` by `Bruno Aristimunha`_)
- Improving the docstring for :class:`braindecode.models.EEGConformer`  (:gh:`769` by `Bruno Aristimunha`_)
- Experimental BIDS Iterable Dataset support (:gh:`766` by `Bruno Aristimunha`_ and `Pierre Guetschel`_)
- Improving the docstring for :class:`braindecode.models.ATCNet`  (:gh:`771` by `Bruno Aristimunha`_)
- Improving the docstring for :class:`braindecode.models.AttentionBaseNet`  (:gh:`772` by `Bruno Aristimunha`_)
- Improving the docstring for :class:`braindecode.models.EEGNeX`  (:gh:`773` by `Bruno Aristimunha`_)
- Massive refactor of the models webpage (:gh:`774` by `Bruno Aristimunha`_)
- Massive refactor of the models webpage (:gh:`775` by `Bruno Aristimunha`_), creating the models categorization, table and many details.


API changes
~~~~~~~~~~~
- Using the name from the original name and deprecation models that we create for no reason, models :gh:`775` by `Bruno Aristimunha`_
- Deprecated the version name in :class:`braindecode.models.EEGNetv4` in favour of :class:`braindecode.models.EEGNetv`.
- Deprecated the version name in :class:`braindecode.models.SleepStagerEldele2021` in favour of :class:`braindecode.models.AttnSleep`.
- Deprecated the version name in :class:`braindecode.models.TSceptionV1` in favour of :class:`braindecode.models.TSception`.

Bugs
~~~~


Version 1.1.1
----------------------

Enhancements
~~~~~~~~~~~~
- Massive refactor of the model webpage

API changes
~~~~~~~~~~~


Bugs
~~~~
- Fixing several models to be be compatibility with small windows size(:gh:`756` by `Bruno Aristimunha`_)
- Fix initialization of :class:`braindecode.models.Deep4Net` (:gh:`752` by `Bruno Aristimunha`_)
- Fix initialization of :class:`braindecode.models.ATCNet` (:gh:`754` by `Bruno Aristimunha`_)
- Fix Labram model for small input (:gh:`757` by `Bruno Aristimunha`_)
- Fix the CTNet parameters to make the embedding configuration easy (:gh:`760` by `Bruno Aristimunha`_)

API changes
~~~~~~~~~~~


Version 1.0
----------------------

Enhancements
~~~~~~~~~~~~
- Make sure all the models start at train model (:gh:`745` by `Bruno Aristimunha`_)
- Enable more models to be pytorch compatible (:gh:`732` by `Bruno Aristimunha`_ and `Lucas Heck`_)
- Making the braindecode.models compatibility with torch compile, torch export and torch jit (:gh:`729` by `Bruno Aristimunha` and `Pierre Guetschel`_)
- Reorder the modules, functional and re-organize the codebase (:gh:`728` by `Bruno Aristimunha`_)
- Adding :class:`braindecode.models.IFNet` (:gh:`725` by `Bruno Aristimunha`_ )
- Adding :class:`braindecode.models.FBMSNet` (:gh:`724` by `Bruno Aristimunha`_ )
- Adding :class:`braindecode.models.FBCNet` (:gh:`722` by `Bruno Aristimunha`_ )
- Adding :class:`braindecode.models.FBLightConvNet` (:gh:`723` by `Bruno Aristimunha`_ )
- Added dropdown menus for selecting `paradigm`, `type`, and `hyperparameters` on the model summary page. (:gh:`718` by `Ganasekhar Kalla`)
- Adding model page (:gh:`715` by `Lucas Heck`_)
- Inference of shape for string module when using :class:`skorch.helper.SliceDataset` (:gh:`716` by `Bruno Aristimunha`_)
- Fix error when using n_njobs > 1 on Windows (:gh:`700` by `Arnaud Delorme`_)
- Adding :class:`braindecode.models.AttentionBaseNet`  (:gh:`572` by `Bruno Aristimunha`_ and `Martin Wimpff`_)
- Adding :class:`braindecode.datasets.NMT` dataset (:gh:`443` by `Mohammad Javad D`_  and `Bruno Aristimunha`_)
- Adding an integration test for all the models (:gh: 570 by `Bruno Aristimunha`_)
- Adding :class:`braindecode.models.BIOT` (:gh:`573` by `Bruno Aristimunha`_)
- Adding :class:`braindecode.models.Labram` (:gh:`578` by `Bruno Aristimunha`_)
- Applying black to the codebase (:gh:`579` by `Bruno Aristimunha`_)
- Adding :class:`braindecode.models.EEGSimpleConv` (:gh:`581` by `Yassine El Ouahidi`_ and `Bruno Aristimunha`_)
- Increasing the coverage of the tests (:gh:`592` by `Bruno Aristimunha`_)
- Adding cache and pre-processing option to :class:`braindecode.datasets.MOABBDataset` (:gh:`582` by `Bruno Aristimunha`_)
- Add type hints to datasets (:gh:`590` by `Pierre Guetschel`_)
- Add channel names and montage to  :class:`braindecode.datasets.TUH` (:gh:`593` by `Pierre Guetschel`_)
- Add ``offset`` arg to :func:`braindecode.preprocessing.preprocess` (:gh:`599` by `Pierre Guetschel`_)
- Add type hints to preprocessing (:gh:`600` by `Pierre Guetschel`_)
- Add ``mypy`` type checks to pre-commit and CI (:gh:`606` by `Pierre Guetschel`_)
- Code clarity changes in `windowers.py` (:gh:`615` by `John Muradeli`_)
- Adding SegmentationReconstruction augmentation :class:`braindecode.augmentation.SegmentationReconstruction` (:gh:`608` by `Gustavo Rodrigues`_)
- Add two models :class:`braindecode.models.ContraWR` and :class:`braindecode.models.SPARCNet` (:gh:`611` by `Bruno Aristimunha`_)
- Add Sleep Physionet 18 dataset (:gh:`621` by `Hubert Banville`_ and `Bruno Aristimunha`_)
- Optimize the CI by executing only the last commit (:gh:`612` by `Bruno Aristimunha`_)
- Add experimental `lazy_metadata` parameter to :func:`braindecode.preprocessing.create_fixed_length_windows` (:gh:`597` by `Pierre Guetschel`_)
- Increasing moabb version to 1.1.0 (:gh:`632` by `Bruno Aristimunha`_)
- Add MaskEncoding augmentation :class:`braindecode.augmentation.MaskEncoding` (:gh:`631` by `Gustavo Rodrigues`_)
- Adding :class:`braindecode.models.EEGNex`  (:gh:`635` by `Bruno Aristimunha`_ )
- Adding :class:`braindecode.models.TSception`  (:gh:`641` by `Bruno Aristimunha`_ )
- Adding :class:`braindecode.models.EEGTCNet`  (:gh:`640` by `Bruno Aristimunha`_ )
- Ensure consistency in the last layer using tests (:gh:`642` by `Bruno Aristimunha`_ )
- Ensuring consistency on the expose of the activation function (:gh:`637` by `Bruno Aristimunha`_ )
- Adding :class:`braindecode.models.SyncNet` (:gh:`643` by `Bruno Aristimunha`_ )
- Adding :class:`braindecode.models.MSVTNet` (:gh:`659` by `Bruno Aristimunha`_ )
- Creating the `FilterBanklayer` module for new models (:gh:`656` by `Bruno Aristimunha`_ )
- Including PytorchAudio as dependency and remove copied code (:gh:`661` by `Bruno Aristimunha`_)
- Adding :class:`braindecode.models.EEGMiner` (:gh:`667` by `Bruno Aristimunha`_ )
- Adding :class:`braindecode.models.CTNet`  (:gh:`666` by `Bruno Aristimunha`_ )
- Fix warnings not being suppressed when creating a TUHAbnormal dataset in parallel (:gh:`670` by `Aphel`_)
- Exposing :class:`braindecode.models.EEGITNet` hyper-parameters (:gh:`672` by `Bruno Aristimunha`_)
- Adding :class:`braindecode.models.SincShallowNet` (:gh:`678` by `Bruno Aristimunha`_ )
- Adding :class:`braindecode.models.SCCNet` (:gh:`679` by `Bruno Aristimunha`_ )
- Fix error when using NMT dataset with n_jobs > 1 (:gh:`690` by `Aphel`_)
- Adding support for distributed samplers (:gh:`695` by `Young Truong`_)
- Adding :class:`braindecode.datasets.BIDSDataset` and :class:`braindecode.datasets.BIDSEpochsDataset` plus tutorial (:gh:`701` :gh:`702` :gh:`704` by `Pierre Guetschel`_ )
- Add :class:`braindecode.models.SignalJEPA` plus downstream architectures (:gh:`703` by `Pierre Guetschel`_ )
- Various improvements on documentation:

  * :gh:`706` by `Pierre Guetschel`_

Bugs
~~~~
- Making the :class:`braindecode.models.SCCNet` more compatible with paper instead of source code (:gh:`742` by `Bruno Aristimunha`_)
- Making the :class:`braindecode.models.EEGNeX` and :class:`braindecode.models.CTNet` more compatible with paper instead of source code (:gh:`740` by `Bruno Aristimunha`_)
- Exposing extra variable to avoid problem with the parallel process (:gh:`736` by `Pierre Guetschel`_)
- Fixing the IFNet (:gh:`739` by `Bruno Aristimunha`_)
- Fix dead links in the documentation (:gh:`727` by `Lucas Heck`_)
- Fix annotations bug for moabb datasets with non-zero interval (:gh: 561 by `Daniel Wilson`_)
- Fix deprecated test and changing the what's new checker (:gh: 569 by `Bruno Aristimunha`_)
- Fix issue with coverage CI and adding a condition on the test for avoid HTML errors (:gh: 591 by `Bruno Aristimunha`_)
- Constraint the version of mne (:gh: 594 by `Bruno Aristimunha`_)
- Fix type errors (:gh:`606` by `Pierre Guetschel`_)
- Warn when applying preprocessing steps on a :class:`braindecode.datasets.base.EEGWindowsDataset` (:gh:`607` by `Pierre Guetschel`_)
- Fix matplotlib colormaps deprecation (:gh:`608` by `Bruno Aristimunha`_)
- Ensure mypy to work for every commit (:gh:`619` by `Bruno Aristimunha`_)
- Deprecate moabb version 1.0.0 because of incorrect epoching (:gh:`627` by `Pierre Guetschel`_)
- Fixing tutorial benchmark lazy eager loagin (:gh:`` by `Bruno Aristimunha`_ and `Aphel`_)
- Improve doc build's time with better caching (:gh:`693` by `Thomas Moreau`_)
- Fixing the MOABBDataset to work with the cache (:gh:`694` by `Bruno Aristimunha`_)

API changes
~~~~~~~~~~~
- Removing deprecated variable ``use_log_softmax`` from :class:`braindecode.models.base` (:gh:`729` by `Bruno Aristimunha`_)
- Removing function ``squeze_final_output`` with Expression because of not compatibility with torch script (:gh:`729` by `Bruno Aristimunha`_)
- Expose the ``use_mne_epochs parameter`` of :func:`braindecode.preprocessing.create_windows_from_events` (:gh:`607` by `Pierre Guetschel`_)
- Parameter ``use_log_softmax`` is default as `False` for all the models in (:gh:`624` by `Bruno Aristimunha`_)
- Normalizing the parameters for dropout as part of normalization of model parameters  (:gh:`624` by `Bruno Aristimunha`_)
- Removing ``use_log_softmax`` and old parameters (:gh:`671` by `Bruno Aristimunha`_)
- Moving :class:`braindecode.models.TCN` and :class:`braindecode.models.HybridNet` to module and creating :class:`braindecode.models.BDTCN` (:gh:`673` by `Bruno Aristimunha`_)
- Removing dead parameters from :class:`braindecode.models.EEGNetv4` (:gh:`676` by `Bruno Aristimunha`_)
- Including Linear Layer at the end :class:`braindecode.models.EEGNetv4` (:gh:`680` by `Bruno Aristimunha`_)
- Expose ``copy_data`` parameter in :func:`braindecode.preprocessing.preprocess` (:gh:`736` by `Pierre Guetschel`_)

.. _changes_0_8_0:

Version 0.8 (11-2022)
----------------------

Enhancements
~~~~~~~~~~~~
- Adding :class:`braindecode.models.EEGInceptionMI` network for motor imagery (:gh:`428` by `Cedric Rommel`_)
- Adding :class:`braindecode.models.ATCNet` network for motor imagery (:gh:`429` by `Cedric Rommel`_)
- Adding to :class:`braindecode.datasets.tuh.TUH` compatibility with version 3.0 of TUH dataset (:gh:`431` by `Mohammad Javad D`_, `Bruno Aristimunha`_, `Robin Tibor Schirrmeister`_, `Lukas Gemein`_, `Denis A. Engemann`_ and `Oskar Størmer`_)
- Adding :class:`braindecode.models.DeepSleepNet` network for sleep staging (:gh:`417` by `Théo Gnassounou`_)
- Adding :class:`braindecode.models.EEGConformer` network (:gh:`454` by `Yonghao Song`_ and `Bruno Aristimunha`_)
- Adding einops in the requirements (:gh:`466` by `Bruno Aristimunha`_)
- Have moabb as an extra dependency (:gh:`467` by `Marco Zamboni`_)
- Replacing the replacing Pytorch layers to Rearrange from einops #468  (:gh:`468` by `Bruno Aristimunha`_)
- Refactoring the documentation and creating a sub-structure for the examples (:gh:`470` by `Denis A. Engemann`_ and `Bruno Aristimunha`_)
- Solving issues with slow conda and splitting the doc and test .yml to speed the CI. (:gh:`479` by `Bruno Aristimunha`_)
- Improving the GitHub Actions CI and solving the skorch compatibility in the examples (:gh:`472` by `Bruno Aristimunha`_)
- Changing the documentation order (:gh:`489` by `Bruno Aristimunha`_)
- Improve the documentation for the Temple University Hospital (TUH) EEG Corpus with discrete targets (:gh:`485` by `Pierre Guetschel`_ and `Bruno Aristimunha`_)
- Improving documentation for MOABB dataset, Trialwise Decoding & Cropped Decoding (:gh:`490` by `Daniel Wilson`_)
- Improving the documentation for the sleep stage examples (:gh:`487` by `Bruno Aristimunha`_)
- Improving the tutorial Hyperparameter tuning with scikit-learn (:gh:`473` by `Bruno Aristimunha`_)
- Add :class:`braindecode.models.base.EEGModuleMixin` base class for all braindecode models (:gh:`488` by `Pierre Guetschel`_)
- Normalize all models common parameters and leaving the old ones as deprecated (:gh:`488` by `Pierre Guetschel`_)
- Improving the tutorial with a Data Augmentation Search (:gh:`495` by `Sylvain Chevallier`_)
- Improving documentation for "Split Dataset" and "Process a big data EEG resource" examples (:gh:`494` by `Bruna Lopes`_)
- Improving documentation for the Convolutional neural network regression model on fake data (:gh:`491` by `Sara Sedlar`_)
- Enforcing the eval mode in the function predict trial. (:gh:`497` by `Bruno Aristimunha`_)
- Adding extra requirements for pip install, update doc, removing conda env file (:gh:`505` by `Sylvain Chevallier`_)
- Add models user-friendly representation with torchinfo tables to :class:`braindecode.models.base.EEGModuleMixin` (:gh:`488` by `Maciej Śliwowski`_)
- Merged temporal and spatial convolutions for Deep4 and ShallowFBCSP (by `Daniel Wilson`_ and `Sara Sedlar`_)
- Enabling data augmentation of single inputs (with no batch dimension). (:gh:`503` by `Cedric Rommel`_)
- Adding `randomize` parameter to :class:`braindecode.samplers.SequenceSampler` (:gh:`504` by `Théo Gnassounou`_.)
- Creating new preprocessor objects based on mne's raw/Epochs methods :class:`braindecode.preprocessing.Resample`, :class:`braindecode.preprocessing.DropChannels`, :class:`braindecode.preprocessing.SetEEGReference`, :class:`braindecode.preprocessing.Filter`, :class:`braindecode.preprocessing.Pick`, :class:`braindecode.preprocessing.Crop` (:gh:`500` by `Bruna Lopes`_ and `Bruno Aristimunha`_)
- Moving :class:`braindecode.models.util.get_output_shape` and :func:`braindecode.models.util.to_dense_prediction_model` to :class:`braindecode.models.base.EEGModuleMixin` (:gh:`514` by `Maciej Śliwowski`_)
- Automatically populate signal-related parameters in :class:`braindecode.EEGClassifier` and :class:`braindecode.EEGRegressor` (:gh:`517` by `Pierre Guetschel`_)
- Adding a pure PyTorch tutorial (:gh:`523` by `Remi Delbouys`_  and `Bruno Aristimunha`_)
- Add ``models_dict`` to :mod:`braindecode.models.util` (:gh:`524` by `Pierre Guetschel`_)
- Keep using `mne.Raw` after windowing to speed up windowing, do not create `mne.Epochs` (:gh:`515` by `Robin Tibor Schirrmeister`_)
- Changing :class:`braindecode.models.Deep4Net` `final_conv_length` default value to 'auto' (:gh:`535` by `Maciej Śliwowski`_)
- Add support for :class:`mne.Epochs` in :class:`braindecode.EEGClassifier` and :class:`braindecode.EEGRegressor` (:gh:`529` by `Pierre Guetschel`_)
- Allow passing only the name of a braindecode model to :class:`braindecode.EEGClassifier` and :class:`braindecode.EEGRegressor` (:gh:`528` by `Pierre Guetschel`_)
- Standardizing models' last layer names (:gh:`520` by `Bruna Lopes`_ and `Pierre Guetschel`_)
- Add basic training example with MNE epochs (:gh:`539` by `Pierre Guetschel`_)
- Log validation accuracy in :class:`braindecode.EEGClassifier` (:gh:`541` by `Pierre Guetschel`_)
- Better type hints in :mod:`braindecode.augmentation.base` (:gh:`551` by `Valentin Iovene`_)
- Support for MOABB 1.0.0 and switch to pyproject.toml (:gh:`553` by `Sylvain Chevallier`_)
- Adding pre-commit hooks (:gh:`556` by `Bruno Aristimunha`_)

Bugs
~~~~
- Fixing conda env in the CI (:gh:`461` by `Bruno Aristimunha`_)
- Fixing E231 missing whitespace after ',' untraceable error in old flake8 (:gh:`460` by `Bruno Aristimunha`_)
- Removing deprecation warning due to torch transposition in :func:`braindecode.augmentation.functional._frequency_shift` (:gh:`446` by `Matthieu Terris`_)
- Fix padding's device in :class:`braindecode.models.EEGResNet` (:gh:`451` by `Pierre Guetschel`_)
- Fix skorch version issue (:gh:`465` by `Marco Zamboni`_)
- Fix wrong `kernel_size` dtype when running torchinfo in :class:`braindecode.models.USleep` (:gh:`538` by `Maciej Śliwowski`_)
- Fix bug when using GPU and channel shuffle transform (:gh:`546` by `Robin Tibor Schirrmeister`_)


API changes
~~~~~~~~~~~
- Renaming the :class:`braindecode.models.EEGInception` network as :class:`braindecode.models.EEGInceptionERP` (:gh:`428` by `Cedric Rommel`_)
- Removing support for Python 3.7 (:gh:`397` by `Bruno Aristimunha`_)
- Removing the LogSoftmax layer from the models and adding deprecated warnings and temporary flags (:gh:`513` by `Sara Sedlar`_)

.. _changes_0_7_0:

Version 0.7 (10-2022)
----------------------

Enhancements
~~~~~~~~~~~~
- Adding EEG-Inception Network :class:`braindecode.models.EEGInception` (:gh:`390` by `Bruno Aristimunha`_ and `Cedric Rommel`_)
- Adding EEG-ITNet Network :class:`braindecode.models.EEGITNet` (:gh:`400` by `Ghaith Bouallegue`_)
- Allowing target_names as list for BaseDataset (:gh:`371` by `Mohammad Javad D`_ and `Robin Tibor Schirrmeister`_)
- Adding tutorial with GridSearchCV for data augmentation on the BCIC IV 2a with module `braindecode.augmentation` (:gh:`389` by `Bruno Aristimunha`_ and `Cedric Rommel`_)
- Adding tutorial with GridSearchCV to exemplify how to tune hyperparameters, for instance with the learning rate (:gh:`349` by `Lukas Gemein`_ and by `Bruno Aristimunha`_)
- Adding tutorial with a Unified Validation scheme (:gh:`378` by `Bruno Aristimunha`_ and `Martin Wimpff`_)
- Adding `verbose` parameter to :func:`braindecode.preprocessing.create_windows_from_events`, :func:`braindecode.preprocessing.create_windows_from_target_channels`, and :func:`braindecode.preprocessing.create_fixed_length_windows` (:gh:`391` by `Lukas Gemein`_)
- Enable augmentation on GPU within :class:`AugmentedDataloader` via a new `device` parameter (:gh:`406` by `Martin Wimpff`_, `Bruno Aristimunha`_ and `Cedric Rommel`_)
- Adding `randomize` parameter to :class:`braindecode.samplers.SequenceSampler` (:gh:`504` by `Théo Gnassounou`_.)

Bugs
~~~~
- Fixing parameter `subject_ids` to `recoding_ids` in TUHAbnormal example (:gh:`402` by `Bruno Aristimunha`_ and `Lukas Gemein`_)
- Bug fix :func:`braindecode.augmentation.functional.ft_surrogate` and add option to sample independently per-channel (:gh:`409` by `Martin Wimpff`_ and `Cedric Rommel`_)


API changes
~~~~~~~~~~~
- Renaming the method `get_params` to `get_augmentation_params` in augmentation classes. This makes the Transform module compatible with scikit-learn cloning mechanism (:gh:`388` by `Bruno Aristimunha`_ and `Alex Gramfort`_)
- Delaying the deprecation of the preprocessing scale function :func:`braindecode.preprocessing.scale` and updates tutorials where the function were used. (:gh:`413` by `Bruno Aristimunha`_)
- Removing deprecated functions and classes :func:`braindecode.preprocessing.zscore`, :class:`braindecode.datautil.MNEPreproc` and :class:`braindecode.datautil.NumpyPreproc`  (:gh:`415` by `Bruno Aristimunha`_)
- Setting `iterator_train__drop_last=True` by default for :class:`braindecode.EEGClassifier` and :class:`braindecode.EEGRegressor` (:gh:`411` by `Robin Tibor Schirrmeister`_)

.. _changes_0_6_0:

Version 0.6 (2021-12-06)
------------------------

Enhancements
~~~~~~~~~~~~
- Adding :class:`braindecode.samplers.SequenceSampler` along with support for returning sequences of windows in :class:`braindecode.datasets.BaseConcatDataset` and an updated sleep staging example to show how to train on sequences of windows (:gh:`263` by `Hubert Banville`_)
- Adding Thinker Invariance Network :class:`braindecode.models.TIDNet` (:gh:`170` by `Ann-Kathrin Kiessner`_, `Daniel Wilson`_, `Henrik Bonsmann`_, `Vytautas Jankauskas`_)
- Adding a confusion matrix plot generator :func:`braindecode.visualization.plot_confusion_matrix` (:gh:`274` by `Ann-Kathrin Kiessner`_, `Daniel Wilson`_, `Henrik Bonsmann`_, `Vytautas Jankauskas`_)
- Adding data :ref:`augmentation_api` module (:gh:`254` by `Cedric Rommel`_, `Alex Gramfort`_ and `Thomas Moreau`_)
- Adding Mixup augmentation :class:`braindecode.augmentation.Mixup` (:gh:`254` by `Simon Brandt`_)
- Adding saving of preprocessing and windowing choices in :func:`braindecode.preprocessing.preprocess`, :func:`braindecode.preprocessing.create_windows_from_events` and :func:`braindecode.preprocessing.create_fixed_length_windows` to datasets to facilitate reproducibility (:gh:`287` by `Lukas Gemein`_)
- Adding :func:`braindecode.models.util.aggregate_probas` to perform self-ensembling of predictions with sequence-to-sequence models (:gh:`294` by `Hubert Banville`_)
- Adding :func:`braindecode.training.scoring.predict_trials` to generate trialwise predictions after cropped training (:gh:`312` by `Lukas Gemein`_)
- Preprocessing and windowing choices are now saved on the level of individual datasets (:gh:`288` by `Lukas Gemein`_)
- Serialization now happens entirely on dataset level creating subsets for individual datasets that contain 'fif' and 'json' files (:gh:`288` `Lukas Gemein`_)
- Instantiation of TUH :class:`braindecode.datasets.tuh.TUH` and TUHAbnormal :class:`braindecode.datasets.tuh.TUHAbnormal`, as well as loading :func:`braindecode.datautil.serialization.load_concat_dataset` of stored datasets now support multiple workers (:gh:`288` by `Lukas Gemein`_)
- Adding balanced sampling of sequences of windows with :class:`braindecode.samplers.BalancedSequenceSampler`  as proposed in U-Sleep paper (:gh:`295` by `Théo Gnassounou`_ and `Hubert Banville`_)
- :func:`braindecode.preprocessing.preprocess` can now work in parallel and serialize datasets to enable lazy-loading (i.e. `preload=False`) (:gh:`277` by `Hubert Banville`_)
- Adding :class:`braindecode.models.TimeDistributed` to apply a module on a sequence (:gh:`318` by `Hubert Banville`_)
- Adding time series targets decoding together with :class:`braindecode.datasets.BCICompetitionIVDataset4` and fingers flexion decoding from ECoG examples (:gh:`261` by `Maciej Śliwowski`_ and `Mohammed Fattouh`_)
- Make EEGClassifier and EEGRegressor cloneable for scikit-learn (:gh:`347` by `Lukas Gemein`_, `Robin Tibor Schirrmeister`_, `Maciej Śliwowski`_ and `Alex Gramfort`_)
- Allow to raise a warning when a few trials are shorter than the windows length, instead of raising an error and stopping all computation. (:gh:`353` by `Cedric Rommel`_)
- Setting `torch.backends.cudnn.benchmark` in :func:`braindecode.util.set_random_seeds`, adding warning and more info to the docstring to improve reproducibility (:gh:`333` by `Maciej Śliwowski`_)
- Adding option to pass arguments through :class:`braindecode.datasets.MOABBDataset` (:gh:`365` by `Pierre Guetschel`_)
- Adding a possibility to use a dict to split a BaseConcatDataset in :meth:`braindecode.datasets.BaseConcatDataset.split` (:gh:`367` by `Alex Gramfort`_)
- Adding ``crop`` parameter to :class:`braindecode.datasets.SleepPhysionet` dataset to speed up examples (:gh:`367` by `Alex Gramfort`_)

Bugs
~~~~
- Correctly computing recording length in :func:`braindecode.preprocessing.windowers.create_fixed_length_windows` in case recording was cropped (:gh:`304` by `Lukas Gemein`_)
- Fixing :class:`braindecode.datasets.SleepPhysionet` to allow serialization and avoid mismatch in channel names attributes (:gh:`327` by `Hubert Banville`_)
- Propagating `target_transform` to all datasets when using :meth:`braindecode.datasets.BaseConcatDataset.subset` (:gh:`261` by `Maciej Śliwowski`_)

API changes
~~~~~~~~~~~
- Removing the default sampling frequency sfreq value in :func:`braindecode.datasets.create_windows_from_events` (:gh:`256` by `Ann-Kathrin Kiessner`_, `Daniel Wilson`_, `Henrik Bonsmann`_, `Vytautas Jankauskas`_)
- Made windowing arguments optional in :func:`braindecode.preprocessing.windowers.create_fixed_length_windows` & :func:`braindecode.preprocessing.windowers.create_windows_from_events` (:gh:`269` by `Ann-Kathrin Kiessner`_, `Daniel Wilson`_, `Henrik Bonsmann`_, `Vytautas Jankauskas`_)
- Deprecating preprocessing functions :func:`braindecode.preprocessing.zscore` and :func:`braindecode.preprocessing.scale` in favour of sklearn's implementation (:gh:`292` by `Hubert Banville`_)
- :func:`braindecode.preprocessing.preprocess` now returns a :class:`braindecode.dataset.BaseConcatDataset` object (:gh:`277` by `Hubert Banville`_)

.. _changes_0_5_1:

Version 0.5.1 (2021-07-14)
--------------------------

Enhancements
~~~~~~~~~~~~
- Adding `n_jobs` parameter to windowers :func:`braindecode.datautil.create_windows_from_events` and :func:`braindecode.datautil.create_fixed_length_windows` to allow for parallelization of the windowing process (:gh:`199` by `Hubert Banville`_)
- Adding support for on-the-fly transforms (:gh:`198` by `Hubert Banville`_)
- Unifying preprocessors under the :class:`braindecode.datautil.Preprocessor` class (:gh:`197` by `Hubert Banville`_)
- Adding self-supervised learning example on the Sleep Physionet dataset along with new sampler module `braindecode.samplers` (:gh:`178` by `Hubert Banville`_)
- Adding sleep staging example on the Sleep Physionet dataset (:gh:`161` by `Hubert Banville`_)
- Adding new parameters to windowers :func:`braindecode.datautil.create_windows_from_events` and :func:`braindecode.datautil.create_fixed_length_windows` for finer control over epoching (:gh:`152` by `Hubert Banville`_)
- Adding Temporal Convolutional Network :class:`braindecode.models.TCN` (:gh:`138` by `Lukas Gemein`_)
- Adding option to use BaseConcatDataset as input to BaseConcatDataset (:gh:`142` by `Lukas Gemein`_)
- Adding a simplified API for splitting of BaseConcatDataset: parameters `property` and `split_ids` in :meth:`braindecode.datasets.BaseConcatDataset.split` are replaced by `by` (:gh:`147` by `Lukas Gemein`_)
- Adding a preprocessor that realizes a filterbank: :func:`braindecode.datautil.filterbank` (:gh:`158` by `Lukas Gemein`_)
- Removing code duplicate in BaseDataset and WindowsDataset (:gh:`159` by `Lukas Gemein`_)
- Only load data if needed during preprocessing (e.g., allow timecrop without loading) (:gh:`164` by `Robin Tibor Schirrmeister`_)
- Adding option to sort filtered channels by frequency band for the filterbank in :func:`braindecode.datautil.filterbank` (:gh:`185` by `Lukas Gemein`_)
- Adding the USleep model :class:`braindecode.models.USleep` (:gh:`282` by `Théo Gnassounou`_ and `Omar Chehab`_)
- Adding :class:`braindecode.models.SleepStagerEldele2021` and :class:`braindecode.models.SleepStagerBlanco2020` models for sleep staging  (:gh:`341` by `Divyesh Narayanan`_)

Bugs
~~~~
- Amplitude gradients are correctly computed for layers with multiple filters
  (before, they were accidentally summed over all previous filters in the layer) (:gh:`167` by `Robin Tibor Schirrmeister`_)
- :func:`braindecode.models.get_output_shape` and :func:`braindecode.visualization.compute_amplitude_gradients` assume 3d, not 4d inputs (:gh:`166` by `Robin Tibor Schirrmeister`_)
- Fixing windower functions when the continuous data has been cropped (:gh:`152` by `Hubert Banville`_)
- Fixing incorrect usage of recording ids in TUHAbnormal (:gh:`146` by `Lukas Gemein`_)
- Adding check for correct input dimensions (4d) in TCN (:gh:`169` by `Lukas Gemein`_)
- Fixing :func:`braindecode.datautil.create_windows_from_events` when `window_size` is not given but there is a :code:`trial_stop_offset_samples` (:gh:`148` by `Lukas Gemein`_)
- Fixing :meth:`braindecode.classifier.EEGClassifier.predict_proba` and :meth:`braindecode.regressor.EEGRegressor.predict` behavior in the cropped mode (:gh:`171` by `Maciej Śliwowski`_)
- Freeze torch random generator for scoring functions for reproducibility (:gh:`155` by `Robin Tibor Schirrmeister`_)
- Make EEGResNet work for :code:`final_pool_length='auto'` (:gh:`223` by `Robin Tibor Schirrmeister`_ and `Maciej Śliwowski`_)

API changes
~~~~~~~~~~~
- Preprocessor classes :class:`braindecode.datautil.MNEPreproc` and :class:`braindecode.datautil.NumpyPreproc` are deprecated in favor of :class:`braindecode.datautil.Preprocessor` (:gh:`197` by `Hubert Banville`_)
- Parameter `stop_offset_samples` of :func:`braindecode.datautil.create_fixed_length_windows` must now be set to `None` instead of 0 to indicate the end of the recording (:gh:`152` by `Hubert Banville`_)

Authors
~~~~~~~

.. _Arnaud Delorme: https://github.com/arnodelorme
.. _Hubert Banville: https://github.com/hubertjb
.. _Robin Tibor Schirrmeister: https://github.com/robintibor
.. _Lukas Gemein: https://github.com/gemeinl
.. _Maciej Śliwowski: https://github.com/sliwy
.. _Ann-Kathrin Kiessner: https://github.com/Ann-KathrinKiessner
.. _Daniel Wilson: https://github.com/dcwil
.. _Henrik Bonsmann: https://github.com/HenrikBons
.. _Vytautas Jankauskas: https://github.com/vytjan
.. _Théo Gnassounou: https://github.com/Tgnassou
.. _Omar Chehab: https://github.com/l-omar-chehab
.. _Divyesh Narayanan: https://github.com/Div12345
.. _Alex Gramfort: http://alexandre.gramfort.net
.. _Cedric Rommel: https://cedricrommel.github.io
.. _Simon Brandt: https://github.com/sbbrandt
.. _Thomas Moreau: https://tommoral.github.io
.. _Mohammed Fattouh: https://github.com/MFattouh
.. _Pierre Guetschel: https://github.com/PierreGtch
.. _Mohammad Javad D: https://github.com/MohammadJavadD
.. _Bruno Aristimunha: https://github.com/bruAristimunha
.. _Martin Wimpff: https://github.com/martinwimpff
.. _Ghaith Bouallegue: https://github.com/GhBlg
.. _Denis A. Engemann: https://github.com/dengemann
.. _Oskar Størmer: https://github.com/ostormer
.. _Matthieu Terris: https://github.com/matthieutrs
.. _Yonghao Song: https://github.com/eeyhsong
.. _Marco Zamboni: https://github.com/ZamboniMarco99
.. _Sara Sedlar: https://github.com/Sara04
.. _Bruna Lopes: https://github.com/brunaafl
.. _Sylvain Chevallier: https://github.com/sylvchev
.. _Remi Delbouys: https://github.com/remidbs
.. _Valentin Iovene: https://github.com/tgy
.. _Yassine El Ouahidi: https://github.com/elouayas
.. _John Muradeli: https://github.com/OverLordGoldDragon/
.. _Gustavo Rodrigues: https://github.com/gustavohenriquesr
.. _Aphel: https://github.com/itsaphel
.. _Young Truong: https://github.com/dungscout96
.. _Lucas Heck: https://github.com/lucas-heck
.. _Ganasekhar Kalla: https://github.com/Ganasekhar-gif
.. _Hamdi Altaheri: https://github.com/Altaheri
