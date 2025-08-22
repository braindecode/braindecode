:html_theme.sidebar_secondary.remove:

Models Categorization
~~~~~~~~~~~~~~~~~~~~~

Some text from the thesis here....



.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: |braille| Convolution layers
      :shadow: sm

      :bdg-success:`Convolution`

      .. figure:: ../_static/model_cat/convolution.png
        :width: 90%
        :align: center
        :alt: Diagram of a convolutional layer
        :class: no-scaled-link

      Applies convolutional layers to extract spatial, temporal, and spectral features from brain signals.

   .. grid-item-card:: |repeat| Recurrent Layers
      :shadow: sm

      :bdg-secondary:`Recurrent`

      .. figure:: ../_static/model_cat/rnn.png
        :class: no-scaled-link
        :width: 90%
        :align: center
        :alt: Diagram of a convolutional layer

      Processes sequential data using recurrent units (e.g., LSTM, GRU, TCN) to model temporal dependencies.

   .. grid-item-card:: |magnifying-glass-chart| Small Attention
      :shadow: sm

      :bdg-info:`Small Attention`

      .. figure:: ../_static/model_cat/attention.png
        :class: no-scaled-link
        :width: 90%
        :align: center
        :alt: Diagram of a convolutional layer

      Uses attention mechanisms for feature focusing. Can be trained effectively without self-supervised pre-training.

   .. grid-item-card:: |layer-group| Filterbank models
      :shadow: sm

      :bdg-primary:`Filterbank`

      .. figure:: ../_static/model_cat/filterbank.png
        :class: no-scaled-link
        :width: 90%
        :align: center
        :alt: Diagram of a recurrent models

      Employs multiple non-learneable filterbanks to decompose signals and capture frequency-specific information.

   .. grid-item-card:: |eye| Interpretability-by-design
      :shadow: sm

      :bdg-warning:`Interpretability`

      .. figure:: ../_static/model_cat/interpre.png
        :class: no-scaled-link
        :width: 90%
        :align: center
        :alt: Diagram of a interpretable

      Architectures with inherently interpretable layers, allowing for direct neuroscientific validation of learned features.

   .. grid-item-card:: |circle-nodes| Symmetric Positive-Definite
      :shadow: sm

      :bdg-dark:`SPD` :bdg-danger-line:`To be released soon!`

      .. figure:: ../_static/model_cat/spd.png
        :class: no-scaled-link
        :width: 90%
        :align: center
        :alt: Diagram of a SPD Learn
        :figclass: unavailable

      Learns representations by leveraging Symmetric Positive-Definite (SPD) matrices, which typically encode functional connectivity (covariance).

   .. grid-item-card:: |lightbulb| Large Language Model
      :shadow: sm

      :bdg-danger:`Large Language Model`

      .. figure:: ../_static/model_cat/llm.png
        :class: no-scaled-link
        :width: 90%
        :align: center
        :alt: Diagram of a LLM models

      Large-scale transformer layers that require self-supervised pre-training to effectively model complex data patterns.

   .. grid-item-card:: |share-nodes| Graph Neural Network
      :shadow: sm

      :bdg-light:`Graph Neural Network`

      .. figure:: ../_static/model_cat/gnn.png
        :class: no-scaled-link
        :width: 90%
        :align: center
        :alt: Diagram for Graph neural Network models
        :figclass: unavailable

      Models the relationships between channels as a graph to explicitly learn from functional connectivity patterns.

   .. grid-item-card:: |clone| Channel-domain
      :shadow: sm

      :bdg-dark-line:`Channel`

      .. figure:: ../_static/model_cat/channel.png
        :class: no-scaled-link
        :width: 90%
        :align: center
        :alt: Diagram for Channel-domain models

      Employ the channel-montage feature for more robust, domain-aware learning in neuroscience.

We are continually expanding this collection and welcome contributions! If you have implemented a model relevant to EEG, EcoG, or MEG analysis, consider adding it to Braindecode.


Submit a new model
~~~~~~~~~~~~~~~~~~


Want to contribute a new model to Braindecode? Great! You can propose a new model by opening an `issue <braindecode-issues_>`_ (please include a link to the relevant publication or description) or, even better, directly submit your implementation via a `pull request <braindecode-pulls_>`_. We appreciate your contributions to expanding the library!

.. button-ref:: models_table
   :ref-type: doc
   :color: primary
   :expand:

   Next: Models Table →

.. include:: /links.inc
