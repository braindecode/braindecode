:orphan:

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
Current 0.8 (dev0)
--------------------

Enhancements
~~~~~~~~~~~~

Bugs
~~~~

API changes
~~~~~~~~~~~

.. _changes_0_8_0:

Current 0.8 ()
----------------------

Enhancements
~~~~~~~~~~~~
- Adding :class:`braindecode.models.EEGInceptionMI` network for motor imagery (:gh:`428` by `Cedric Rommel`_)
- Adding :class:`braindecode.models.ATCNet` network for motor imagery (:gh:`429` by `Cedric Rommel`_)
- Adding to :class:`braindecode.datasets.tuh.TUH` compatibility with version 3.0 of TUH dataset (:gh:`431` by `Mohammad Javad D`_, `Bruno Aristimunha`_, `Robin Tibor Schirrmeister`_, `Lukas Gemein`_, `Denis A. Engemann`_ and `Oskar Størmer`_)
- Adding :class:`braindecode.models.DeepSleepNet` network for sleep staging (:gh:`417` by `Théo Gnassounou`_)

Bugs
~~~~


API changes
~~~~~~~~~~~
- Renaming the :class:`braindecode.models.EEGInception` network as :class:`braindecode.models.EEGInceptionERP` (:gh:`428` by `Cedric Rommel`_)
.. _changes_0_7_0:

Current 0.7 (10-2022)
----------------------

Enhancements
~~~~~~~~~~~~
- Adding EEG-Inception Network :class:`braindecode.models.EEGInception` (:gh:`390` by `Bruno Aristimunha`_ and `Cedric Rommel`_)
- Adding EEG-ITNet Network :class:`braindecode.models.EEGITNet` (:gh:`400` by `Ghaith Bouallegue`_)
- Allowing target_names as list for BaseDataset (:gh:`371` by `Mohammad Javad D`_ and `Robin Tibor Schirrmeister`_)
- Adding tutorial with GridSearchCV for data augmentation on the BCIC IV 2a with module `braindecode.augmentation` (:gh:`389` by `Bruno Aristimunha`_ and `Cedric Rommel`_)
- Adding tutorial with GridSearchCV to exemplify how to tune hyperparameters, for instance with the learning rate (:gh:`349` by `Lukas Gemein`_ and by `Bruno Aristimunha`_)
- Adding tutorial with a Unified Validation sheme (:gh:`378` by `Bruno Aristimunha`_ and `Martin Wimpff`_)
- Adding `verbose` parameter to :func:`braindecode.preprocessing.create_windows_from_events`, :func:`braindecode.preprocessing.create_windows_from_target_channels`, and :func:`braindecode.preprocessing.create_fixed_length_windows` (:gh:`391` by `Lukas Gemein`_)
- Enable augmentation on GPU within :class:`AugmentedDataloader` via a new `device` parameter (:gh:`406` by `Martin Wimpff`_, `Bruno Aristimunha`_ and `Cedric Rommel`_)

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
- Adding Thinker Invariance Network :class:`braindecode.models.TIDNet` (:gh:`170` by `Ann-Kathrin Kiessner`_, `Dan Wilson`_, `Henrik Bonsmann`_, `Vytautas Jankauskas`_)
- Adding a confusion matrix plot generator :func:`braindecode.visualization.plot_confusion_matrix` (:gh:`274` by `Ann-Kathrin Kiessner`_, `Dan Wilson`_, `Henrik Bonsmann`_, `Vytautas Jankauskas`_)
- Adding data :ref:`augmentation_api` module (:gh:`254` by `Cedric Rommel`_, `Alex Gramfort`_ and `Thomas Moreau`_)
- Adding Mixup augmentation :class:`braindecode.augmentation.Mixup` (:gh:`254` by `Simon Brandt`_)
- Adding saving of preprocessing and windowing choices in :func:`braindecode.preprocessing.preprocess`, :func:`braindecode.preprocessing.create_windows_from_events` and :func:`braindecode.preprocessing.create_fixed_length_windows` to datasets to facilitate reproducibility (:gh:`287` by `Lukas Gemein`_)
- Adding :func:`braindecode.models.util.aggregate_probas` to perform self-ensembling of predictions with sequence-to-sequence models (:gh:`294` by `Hubert Banville`_)
- Adding :func:`braindecode.training.scoring.predict_trials` to generate trialwise predictions after cropped training (:gh:`312` by `Lukas Gemein`_)
- Preprocessing and windowing choices are now saved on the level of individual datasets (:gh:`288` by `Lukas Gemein`_)
- Serialization now happens entirely on dataset level creating subsets for individual datasets that contain 'fif' and 'json' files (:gh:`288` `Lukas Gemein`_)
- Instantiation of TUH :class:`braindecode.datasets.tuh.TUH` and TUHAbnormal :class:`braindecode.datasets.tuh.TUHAbnormal`, as well as loading :func:`braindecode.datautil.serialization.load_concat_dataset` of stored datasets now support multiple workers (:gh:`288` by `Lukas Gemein`_)
- Adding balanced sampling of sequences of windows with :class:`braindecode.samplers.BalancedSequenceSampler`  as proposed in U-Sleep paper (:gh:`295` by `Theo Gnassounou`_ and `Hubert Banville`_)
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
- Removing the default sampling frequency sfreq value in :func:`braindecode.datasets.create_windows_from_events` (:gh:`256` by `Ann-Kathrin Kiessner`_, `Dan Wilson`_, `Henrik Bonsmann`_, `Vytautas Jankauskas`_)
- Made windowing arguments optional in :func:`braindecode.preprocessing.windowers.create_fixed_length_windows` & :func:`braindecode.preprocessing.windowers.create_windows_from_events` (:gh:`269` by `Ann-Kathrin Kiessner`_, `Dan Wilson`_, `Henrik Bonsmann`_, `Vytautas Jankauskas`_)
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
- Adding the USleep model :class:`braindecode.models.USleep` (:gh:`282` by `Theo Gnassounou`_ and `Omar Chehab`_)
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

.. _Hubert Banville: https://github.com/hubertjb
.. _Robin Tibor Schirrmeister: https://github.com/robintibor
.. _Lukas Gemein: https://github.com/gemeinl
.. _Maciej Śliwowski: https://github.com/sliwy
.. _Ann-Kathrin Kiessner: https://github.com/Ann-KathrinKiessner
.. _Dan Wilson: https://github.com/dcwil
.. _Henrik Bonsmann: https://github.com/HenrikBons
.. _Vytautas Jankauskas: https://github.com/vytjan
.. _Theo Gnassounou: https://github.com/Tgnassou
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
