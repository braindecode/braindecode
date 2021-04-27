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
- Adding support for on-the-fly transforms (:gh:`198` by `Hubert Banville`_)

Bugs
~~~~
- amplitude gradients are correctly computed for layers with multiple filters
  (before, they were accidentally summed over all previous filters in the layer) (:gh:`167` by `Robin Tibor Schirrmeister`_)
- get_output_shape and compute_amplitude_gradients assume 3d, not 4d inputs (:gh:`166` by `Robin Tibor Schirrmeister`_)

API changes
~~~~~~~~~~~
- 

Authors
~~~~~~~

.. _Hubert Banville: https://github.com/hubertjb
.. _Robin Tibor Schirrmeister: https://github.com/robintibor
