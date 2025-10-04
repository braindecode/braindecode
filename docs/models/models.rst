:html_theme.sidebar_secondary.remove: true

.. currentmodule:: braindecode.models

.. _models:

The brain decode problem
~~~~~~~~~~~~~~~~~~~~~~~~

All the models in this library tackle the following problem:
given time-series signals :math:`X \in \mathbb{R}^{C \times T}` and labels
:math:`y \in \mathcal{Y}`, :class:`braindecode` implements neural networks
:math:`f` that **decode** brain activity, i.e., it applies a series of transformations
layers (e.g. :class:`~torch.nn.Conv2d`, :class:`~torch.nn.Linear`, :class:`~torch.nn.ELU`) to the data
to allow us to filter and extract features that are relevant to what we are modeling, in other words:

.. math::

   f_{\theta} : X \to y,

where :math:`C` (``n_chans``) is the number of channels/electrodes and :math:`T` (``n_times``) is the temporal window
length/epoch size over the interval of interest.

The definition of :math:`y` is broad; it may be anchored in a cognitive stimulus (e.g., BCI, ERP, SSVEP, cVEP),
mental state (sleep stage), brain age, visual/audio/text/action inputs, or any target
that can be quantized and modeled as a decoding task, see references :cite:label:`sleep2023,aristimunhaetal23,chevallier2024largest,levy2025brain,benchetrit2024brain,d2024decoding,engemann2022reusable,xu2024alljoined`.

We aim to translate recorded brain activity into its originating stimulus, behavior,
or mental state, :cite:t:`king2014characterizing,king2020`, again, :math:`f(X) \to y`.

The neural networks model :math:`f` learns a representation that is useful for the encoded stimulus
in the subject's brain over time series—also known as *reverse inference*.

In supervised decoding, we usually learn the network parameters :math:`\theta` by minimizing
the regularized the average loss over the training set :math:`\mathcal{D}_{\text{tr}}=\{(x_i,y_i)\}_{i=1}^{N_{\text{tr}}}`.

.. math::

   \begin{aligned}
   \theta^{*}
     &= \arg\min_{\theta}\, \hat{\mathcal{R}}(\theta) \\
     &= \arg\min_{\theta}\, \frac{1}{N_{\text{tr}}}\sum_{i=1}^{N_{\text{tr}}}
        \ell\!\left(f_{\theta}(x_i),\, y_i\right) \;+\; \lambda\,\Omega(\theta)\,,
   \end{aligned}

where :math:`\ell` is the task loss (e.g., cross-entropy :class:`~torch.nn.CrossEntropyLoss`), :math:`\Omega` is an optional regularizer, and :math:`\lambda \ge 0` its weight (e.g. ``weight_decay`` parameter in :class:`~torch.optim.Adam` is the example of regularization).

Equivalently, the goal is to minimize the expected risk :math:`\mathcal{R}(\theta)=\mathbb{E}_{(x,y)\sim P_{\text{tr}}}
[\ell(f_{\theta}(x),y)]`, for which the empirical average above is a finite-sample
approximation.

With this, in this model's sub-pages, we provide:

    - 1) Our definition of the brain decoding problem (here);
    - 2) :doc:`The categorization of the neural networks based on what is inside them <models_categorization>`;
    - 3) :doc:`A table overview to understand what is inside the models  <models_table>`;
    - 4) :doc:`A visualization of the common important information from the models <models_visualization>`.

.. button-ref:: models_categorization
   :ref-type: doc
   :color: primary
   :expand:

   Next: Models categorization →


.. rubric:: References
.. bibliography::


.. toctree::
    :hidden:

    models_categorization
    models_table
    models_visualization

.. include:: /links.inc
