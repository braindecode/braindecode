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

Current (0.5.1.dev0)
-------------------

Enhancements
~~~~~~~~~~~~
- Adding `n_jobs` parameter to windowers :func:`braindecode.datautil.windowers.create_windows_from_events` and :func:`braindecode.datautil.windowers.create_fixed_length_windows` to allow for parallelization of the windowing process (:gh:`199` by `Hubert Banville`_)
- Adding support for on-the-fly transforms (:gh:`198` by `Hubert Banville`_)
- Unifying preprocessors under the :class:`braindecode.datautil.preprocess.Preprocessor` class (:gh:`197` by `Hubert Banville`_)
- Adding self-supervised learning example on the Sleep Physionet dataset along with new sampler module `braindecode.samplers` (:gh:`178` by `Hubert Banville`_)
- Adding sleep staging example on the Sleep Physionet dataset (:gh:`161` by `Hubert Banville`_)
- Adding new parameters to windowers :func:`braindecode.datautil.windowers.create_windows_from_events` and :func:`braindecode.datautil.windowers.create_fixed_length_windows` for finer control over epoching (:gh:`152` by `Hubert Banville`_)
- Adding Temporal Convolutional Network (:gh:`138` by `Lukas Gemein`_)
- Adding tests for TUHAbnormal dataset (:gh:`139` by `Lukas Gemein`_)
- Adding option to use BaseConcatDataset as input to BaseConcatDataset (:gh:`142` by `Lukas Gemein`_)
- Adding a simplified API for splitting of BaseConcatDataset: parameters `property` and `split_ids` in :func:`braindecode.datasets.base.BaseConcatDataset.split` are replaced by `by` (:gh:`147` by `Lukas Gemein`_)
- Adding a preprocessor that realizes a filterbank: :func:`braindecode.datautil.preprocess.filterbank` (:gh:`158` by `Lukas Gemein`_)
- Removing code duplicate in BaseDataset and WindowsDataset (:gh:`159` by `Lukas Gemein`_)
- Only load data if needed during preprocessing (e.g., allow timecrop without loading) (:gh:`164` by `Robin Tibor Schirrmeister`_)
- Adding option to sort filtered channels by frequency band for the filterbank in :func:`braindecode.datautil.preprocess.filterbank` (:gh:`185` by `Lukas Gemein`_)

Bugs
~~~~
- Amplitude gradients are correctly computed for layers with multiple filters
  (before, they were accidentally summed over all previous filters in the layer) (:gh:`167` by `Robin Tibor Schirrmeister`_)
- get_output_shape and compute_amplitude_gradients assume 3d, not 4d inputs (:gh:`166` by `Robin Tibor Schirrmeister`_)
- Fixing windower functions when the continuous data has been cropped (:gh:`152` by `Hubert Banville`_)
- Fixing incorrect usage of recording ids in TUHAbnormal (:gh:`146` by `Lukas Gemein`_)
- Adding check for correct input dimensions (4d) in TCN (:gh:`169` by `Lukas Gemein`_)
- Fixing :func:`braindecode.datautil.windowers.create_windows_from_event` when `window_size` is not given but there is a `trial_stop_offset_samples` (:gh:`148` by `Lukas Gemein`_)
- Fixing :meth:`braindecode.classifier.EEGClassifier.predict_proba` and :meth:`braindecode.regressor.EEGRegressor.predict` behavior in the cropped mode (:gh:`171` by `Maciej Śliwowski`_)
- Freeze torch random generator for scoring functions for reproducibility (:gh:`155` by `Robin Tibor Schirrmeister`_

API changes
~~~~~~~~~~~
- Preprocessor classes :class:`braindecode.datautil.preprocess.MNEPreproc` and :class:`braindecode.datautil.preprocess.NumpyPreproc` are deprecated in favor of :class:`braindecode.datautil.preprocess.Preprocessor` (:gh:`197` by `Hubert Banville`_)
- Parameter `stop_offset_samples` of :func:`braindecode.datautil.windowers.create_fixed_length_windows` must now be set to `None` instead of 0 to indicate the end of the recording (:gh:`152` by `Hubert Banville`_)

Authors
~~~~~~~

.. _Hubert Banville: https://github.com/hubertjb
.. _Robin Tibor Schirrmeister: https://github.com/robintibor
.. _Lukas Gemein: https://github.com/gemeinl
.. _Maciej Śliwowski: https://github.com/sliwy
