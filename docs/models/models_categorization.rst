:html_theme.sidebar_secondary.remove: true

Models Categorization
~~~~~~~~~~~~~~~~~~~~~

Given the brain-decoding framework from the previous page, we define our neural networks,
denoted :math:`f`, as a composition of sequential transformations:

.. math::

   f_{\mathrm{method}}
   \;=\;
   f_{\mathrm{convolution}} \circ \cdots \circ f_{\mathrm{MLP}}\,

where each :math:`f_\ell` is a specific transformation (:math:`\ell` layer) in the network.
Learning is the mapping :math:`f_{\mathrm{method}} : \mathcal{X} \to \mathcal{Y}` on the
training data, with parameters :math:`\theta \in \Theta`.


.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: |braille| Convolution Layers
      :shadow: sm

      :bdg-success:`Convolution`

      .. figure:: ../_static/model_cat/convolution.png
         :width: 90%
         :align: center
         :alt: Diagram of a convolutional layer
         :class: no-scaled-link

      Applies temporal and/or spatial convolutions to extract local features from brain signals.

   .. grid-item-card:: |repeat| Recurrent Layers
      :shadow: sm

      :bdg-secondary:`Recurrent`

      .. figure:: ../_static/model_cat/rnn.png
         :class: no-scaled-link
         :width: 90%
         :align: center
         :alt: Diagram of recurrent/TCN models

      Models temporal dependencies via recurrent units or TCNs with dilations.

   .. grid-item-card:: |magnifying-glass-chart| Small Attention
      :shadow: sm

      :bdg-info:`Small Attention`

      .. figure:: ../_static/model_cat/attention.png
         :class: no-scaled-link
         :width: 90%
         :align: center
         :alt: Diagram of attention modules

      Uses attention mechanisms for feature focusing. Can be trained effectively without self-supervised pre-training.

   .. grid-item-card:: |layer-group| Filterbank Models
      :shadow: sm

      :bdg-primary:`Filterbank`

      .. figure:: ../_static/model_cat/filterbank.png
         :class: no-scaled-link
         :width: 90%
         :align: center
         :alt: Diagram of filterbank models

      Decomposes signals into multiple bands (learned or fixed) to capture frequency-specific information.

   .. grid-item-card:: |eye| Interpretability-by-Design
      :shadow: sm

      :bdg-warning:`Interpretability`

      .. figure:: ../_static/model_cat/interpre.png
         :class: no-scaled-link
         :width: 90%
         :align: center
         :alt: Diagram of interpretable architectures

     Architectures with inherently interpretable layers allow direct neuroscientific validation of learned features.

   .. grid-item-card:: |circle-nodes| Symmetric Positive-Definite
      :shadow: sm

      :bdg-dark:`SPD` :bdg-danger-line:`To be released soon!`

      .. figure:: ../_static/model_cat/spd.png
         :class: no-scaled-link
         :width: 90%
         :align: center
         :alt: Diagram of SPD learning
         :figclass: unavailable

      Learns on covariance/connectivity as SPD matrices using BiMap/ReEig/LogEig layers.

   .. grid-item-card:: |lightbulb| Large Transformer Models
      :shadow: sm

      :bdg-danger:`Large Language Model`

      .. figure:: ../_static/model_cat/llm.png
         :class: no-scaled-link
         :width: 90%
         :align: center
         :alt: Diagram of transformer models

     Large-scale transformer layers require self-supervised pre-training to work effectively.

   .. grid-item-card:: |share-nodes| Graph Neural Network
      :shadow: sm

      :bdg-light:`Graph Neural Network`

      .. figure:: ../_static/model_cat/gnn.png
         :class: no-scaled-link
         :width: 90%
         :align: center
         :alt: Diagram of GNN models
         :figclass: unavailable

      Treats channels/regions as nodes with learned/static edges to model connectivity.

   .. grid-item-card:: |clone| Channel-Domain
      :shadow: sm

      :bdg-dark-line:`Channel`

      .. figure:: ../_static/model_cat/channel.png
         :class: no-scaled-link
         :width: 90%
         :align: center
         :alt: Diagram of channel-domain methods

      Usage montage information with spatial filtering / channel / hemisphere / brain region selection strategies.


- Across most architectures, the earliest stages are convolutional (:bdg-success:`Convolution`), reflecting the brain time series's noisy, locally structured nature.
  These layers apply temporal and/or spatial convolutions—often depthwise-separable as in EEGNet, per-channel or across channel groups to extract robust local features.
  :class:`braindecode.models.EEGNetv4`, :class:`braindecode.models.ShallowFBCSPNet`, :class:`braindecode.models.EEGNeX`, and :class:`braindecode.models.EEGInceptionERP`
- In the **recurrent** family (:bdg-secondary:`Recurrent`), many modern EEG models actually rely on *temporal convolutional networks* (TCNs) with dilations to grow the receptive field, rather than explicit recurrence (:cite:label:`bai2018tcn`),  :class:`braindecode.models.BDTCN`,
- In contrast, several methods employ **small attention** modules (:bdg-info:`Small Attention`) to capture longer-range dependencies efficiently, e.g., :class:`braindecode.models.EEGConformer`, :class:`braindecode.models.CTNet`, :class:`braindecode.models.ATCNet`, :class:`braindecode.models.AttentionBaseNet` (:cite:label:`song2022eeg,zhao2024ctnet,altaheri2022atcnet`).
- **Filterbank-style models** (:bdg-primary:`Filterbank`) explicitly decompose signals into multiple bands before (or while) learning, echoing the classic FBCSP pipeline; examples include :class:`braindecode.models.FBCNet` and :class:`braindecode.models.FBMSNet` (:cite:label:`mane2021fbcnet,liu2022fbmsnet`).
- **Interpretability-by-design** (:bdg-warning:`Interpretability`) architectures expose physiologically meaningful primitives (e.g., band-pass/sinc filters, variance or connectivity features), enabling direct neuroscientific inspection; see :class:`braindecode.models.SincShallowNet` and :class:`braindecode.models.EEGMiner` (:cite:label:`borra2020interpretable,ludwig2024eegminer`).
- **SPD / Riemannian** (:bdg-dark:`SPD`) methods operate on covariance (or connectivity) matrices as points on the SPD manifold, combining layers such as BiMap, ReEig, and LogEig; deep SPD networks and Riemannian classifiers motivate this family (:cite:label:`huang2017riemannian`). *(Coming soon in a dedicate repository.)*
- **Large-model / Transformer** (:bdg-danger:`Large Language Model`) approaches pretrain attention-based encoders on diverse biosignals and fine-tune for EEG tasks; e.g., :class:`braindecode.models.BIOT` (:cite:label:`yang2023biot`). These typically need a heavily self-supervised pre-training before decoding.
- **Graph neural networks** (:bdg-light:`Graph Neural Network`) treat channels/regions as nodes with learned (static or dynamic) edges to model functional connectivity explicitly; representative EEG-GNN, more common in the epileptic decoding (:cite:label:`klepl2024graph`).
- **Channel-domain robustness** (:bdg-dark-line:`Channel`) techniques target variability in electrode layouts by learning montage-agnostic or channel-selective layers (e.g., dynamic spatial filtering, differentiable channel re-ordering); these strategies improve cross-setup generalization :class:`braindecode.models.SignalJEPA` (:cite:label:`guetschel2024sjepa,chen2024eegprogress`).


We are continually expanding this collection and welcome contributions! If you have implemented a
model relevant to EEG, ECoG, or MEG analysis, consider adding it to Braindecode.

Submit a new model
~~~~~~~~~~~~~~~~~~

Want to contribute a new model to Braindecode? Great! You can propose a new model by opening an
`issue <braindecode-issues_>`_ (please include a link to the relevant publication or description) or,
even better, directly submit your implementation via a `pull request <braindecode-pulls_>`_.
We appreciate your contributions to expanding the library!

.. button-ref:: models_table
   :ref-type: doc
   :color: primary
   :expand:

   Next: Models Table

.. include:: /links.inc
