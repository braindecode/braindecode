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

Current (0.5.2.dev0)
--------------------

Enhancements
~~~~~~~~~~~~
- Adding :class:`braindecode.samplers.SequenceSampler` along with support for returning sequences of windows in :class:`braindecode.datasets.BaseConcatDataset` and an updated sleep staging example to show how to train on sequences of windows (:gh:`263` by `Hubert Banville`_)
- Adding Thinker Invariance Network :class:`braindecode.models.TIDNet` (:gh:`170` by `Ann-Kathrin Kiessner`_, `Dan Wilson`_, `Henrik Bonsmann`_, `Vytautas Jankauskas`_)
- Adding a confusion matrix plot generator :func:`braindecode.visualization.plot_confusion_matrix` (:gh:`274` by `Ann-Kathrin Kiessner`_, `Dan Wilson`_, `Henrik Bonsmann`_, `Vytautas Jankauskas`_)
- Adding data :ref:`augmentation_api` module (:gh:`254` by `Cedric Rommel`_, `Alex Gramfort`_ and `Thomas Moreau`_)
- Adding Mixup augmentation :class:`braindecode.augmentation.Mixup` (:gh:`254` by `Simon Brandt`_)
- Adding saving of preprocessing and windowing choices in :func:`braindecode.preprocessing.preprocess`, :func:`braindecode.preprocessing.create_windows_from_events` and :func:`braindecode.preprocessing.create_fixed_length_windows` to datasets to facilitate reproducibility (:gh:`287` by `Lukas Gemein`_)
- Adding :func:`braindecode.models.util.aggregate_probas` to perform self-ensembling of predictions with sequence-to-sequence models (:gh:`294` by `Hubert Banville`_)
- Preprocessing and windowing choices are now saved on the level of individual datasets (:gh:`288` by `Lukas Gemein`_)
- Serialization now happens entirely on dataset level creating subsets for individual datasets that contain 'fif' and 'json' files (:gh:`288` `Lukas Gemein`_)
- Instantiation of TUH :class:`braindecode.datasets.tuh.TUH` and TUHAbnormal :class:`braindecode.datasets.tuh.TUHAbnormal`, as well as loading :func:`braindecode.datautil.serialization.load_concat_dataset` of stored datasets now support multiple workers (:gh:`288` by `Lukas Gemein`_)

Bugs
~~~~
- Correctly computing recording length in :func:`braindecode.preprocessing.windowers.create_fixed_length_windows` in case recording was cropped (:gh:`304` by `Lukas Gemein`_)

API changes
~~~~~~~~~~~
- Removing the default sampling frequency sfreq value in :func:`braindecode.datasets.create_windows_from_events` (:gh:`256` by `Ann-Kathrin Kiessner`_, `Dan Wilson`_, `Henrik Bonsmann`_, `Vytautas Jankauskas`_)
- Made windowing arguments optional in :func:`braindecode.preprocessing.windowers.create_fixed_length_windows` & :func:`braindecode.preprocessing.windowers.create_windows_from_events` (:gh:`269` by `Ann-Kathrin Kiessner`_, `Dan Wilson`_, `Henrik Bonsmann`_, `Vytautas Jankauskas`_)
- Deprecating preprocessing functions :func:`braindecode.preprocessing.zscore` and :func:`braindecode.preprocessing.scale` in favour of sklearn's implementation (:gh:`292` by `Hubert Banville`_)

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
.. _Alex Gramfort: http://alexandre.gramfort.net
.. _Cedric Rommel: https://cedricrommel.github.io
.. _Simon Brandt: https://github.com/sbbrandt
.. _Thomas Moreau: https://tommoral.github.io
