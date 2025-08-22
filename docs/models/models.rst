:html_theme.sidebar_secondary.remove: true

.. _models:

The brain decode problem
~~~~~~~~~~~~~~~~~~~~~~~~

All the models in this library tackle the following problem:
given time-series signals :math:`X \in \mathbb{R}^{C \times T}` and labels
:math:`y \in \mathcal{Y}`, :class:`braindecode` implements neural networks
:math:`f` that **decode** brain activity, i.e., it applies a series of transformations
layers (e.g. :class:`torch.nn.Conv2d`, :class:`torch.nn.Linear`, :class:`torch.nn.ELU`) to the data
to allow us to filter and extract features that are relevant to what we are modeling:

.. math::

   f_{\theta} : X \mapsto y

where :math:`C` is the number of channels/electrodes/``n_chans`` and :math:`T` is the temporal window
length/``n_times`` over the interval of interest.

The definition of :math:`y` is broad; it may be anchored in a cognitive stimulus (e.g., BCI),
mental state (sleep stage), brain age, an image, visual/audio/text inputs, or any target
that can be quantized and modeled as a decoding task.

We aim to translate recorded brain activity into its originating stimulus, behavior,
or mental state :cite:p:`king2014characterizing,king2020`, i.e., :math:`f(X) \to y`.

The model :math:`f` learns a representation that is useful for the encoded stimulus
in the subject's brain over time series—also known as *reverse inference*.

In supervised decoding, we learn the network parameters :math:`\theta` by minimizing
the regularized the average loss over the training set :math:`\mathcal{D}_{\text{tr}}=\{(x_i,y_i)\}_{i=1}^{N_{\text{tr}}}`.

.. math::

   \theta^{\*} \;=\; \arg\min_{\theta}\; \hat{\mathcal{R}}(\theta)
   \;=\; \arg\min_{\theta}\;
   \frac{1}{N_{\text{tr}}}\sum_{i=1}^{N_{\text{tr}}}
   \ell\!\big(f_{\theta}(x_i),\, y_i\big),

where :math:`\ell` is the task loss (e.g., cross-entropy :class:`torch.nn.CrossEntropyLoss`).

Equivalently, the goal is to minimize the expected risk :math:`\mathcal{R}(\theta)=\mathbb{E}_{(x,y)\sim P_{\text{tr}}}
[\ell(f_{\theta}(x),y)]`, for which the empirical average above is a finite-sample
approximation.

With this, in this model's sub-pages, we provide:

    - 1) Our definition of the brain decoding problem;
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
