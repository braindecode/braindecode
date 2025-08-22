:html_theme.sidebar_secondary.remove: true

.. _models:

Braindecode Models
~~~~~~~~~~~~~~~~~~
Braindecode is an open-source, PyTorch-based deep learning architecture toolbox for learning
and evaluating representations from EEG, MEG, and ECoG.

All the models in this library tackle the following problem:
given time-series signals :math:`X \in \mathbb{R}^{C \times T}` and labels
:math:`y \in \mathcal{Y}`, :class:`braindecode` implements neural networks
:math:`f` that **decode** brain activity.

Equation
--------

.. math::

   f_{\theta} : X \mapsto \hat{y}

where :math:`C` is the number of channels/electrodes and :math:`T` is the temporal window
length over the interval of interest.

The definition of :math:`y` is broad; it may be anchored in a cognitive stimulus (BCI),
brain age, an image, visual/audio/text inputs, or any target that can be quantized
and modeled as a decoding task.

We aim to translate recorded brain activity into its originating stimulus, behavior,
or mental state :cite:`king2014characterizing,king:2020`, i.e., :math:`f(X) \to y`.
The model :math:`f` learns a representation that is useful for the encoded stimulus
in the subject’s brain over time series—also known as *reverse inference*.


.. toctree::
    :hidden:

    models_categorization
    models_table
    models_visualization

.. include:: /links.inc
