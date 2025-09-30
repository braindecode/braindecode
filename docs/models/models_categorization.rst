:html_theme.sidebar_secondary.remove: true

.. currentmodule:: braindecode.models

Models Categorization
~~~~~~~~~~~~~~~~~~~~~

Given the brain-decoding framework from the previous page, we define our neural networks,
denoted :math:`f`, as a composition of sequential transformations:

.. math::

 f_{\mathrm{method}}
 \;=\;
 f_{\mathrm{convolution}} \circ f_{\ell} \circ \cdots  \circ f_{\mathrm{linear}}\,

where each :math:`f_\ell` is a specific :math:`\ell` layer in the neural network,
focusing mostly of time in learning the mapping :math:`f_{\mathrm{method}} : \mathcal{X} \to \mathcal{Y}` on the
training data, with parameters :math:`\theta \in \Theta`.
How these *core* :math:`\ell` sequence transformations are structured and combined defines the overall focus and strength of the models.

Here, we categorize the main families of brain decoding models based on their core components and design philosophies.
The categories are not mutually exclusive, but an indication of what governs that neural network model; many models blend elements from multiple families to leverage their combined strengths.
Beginning directly, the categories are nine: :bdg-success:`Convolution`, :bdg-secondary:`Recurrent`, :bdg-info:`Small Attention`, :bdg-primary:`Filterbank`, :bdg-warning:`Interpretability`, :bdg-danger:`Large Brain Model`, :bdg-light:`Graph Neural Network`, :bdg-dark:`Symmetric Positive-Definite` and :bdg-dark-line:`Channel`.

At the moment, not all the categories are implemented, validated, and tested, but there are some that are noteworthy for introducing or popularizing concepts or layer designs that can take decoding further.

The convolutional layer appears as the core primitive across most architectures.
This is because **convolutions are filtering** operations, such as band-pass `filters <https://mne.tools/stable/auto_tutorials/preprocessing/25_background_filtering.html>`__, useful and needed to extract local features from brain signals.
More details about each categories can be found in the respective sections below.

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
   :width: 90%
   :align: center
   :alt: Diagram of recurrent/TCN models
   :class: no-scaled-link

  Models temporal dependencies via recurrent units or TCNs with dilations.

 .. grid-item-card:: |magnifying-glass-chart| Small Attention
  :shadow: sm

  :bdg-info:`Small Attention`

  .. figure:: ../_static/model_cat/attention.png
   :width: 90%
   :align: center
   :alt: Diagram of attention modules
   :class: no-scaled-link

  Uses attention mechanisms for feature focusing. Can be trained effectively without self-supervised pre-training.

 .. grid-item-card:: |layer-group| Filterbank Models
  :shadow: sm

  :bdg-primary:`Filterbank`

  .. figure:: ../_static/model_cat/filterbank.png
   :width: 90%
   :align: center
   :alt: Diagram of filterbank models
   :class: no-scaled-link

  Decomposes signals into multiple bands (learned or fixed) to capture frequency-specific information.

 .. grid-item-card:: |eye| Interpretability-by-Design
  :shadow: sm

  :bdg-warning:`Interpretability`

  .. figure:: ../_static/model_cat/interpre.png
   :width: 90%
   :align: center
   :alt: Diagram of interpretable architectures
   :class: no-scaled-link

  Architectures with inherently interpretable layers allow direct neuroscientific validation of learned features.

 .. grid-item-card:: |circle-nodes| Symmetric Positive-Definite
  :shadow: sm

  :bdg-dark:`SPD` :bdg-danger-line:`To be released soon!`

  .. figure:: ../_static/model_cat/spd.png
   :width: 90%
   :align: center
   :alt: Diagram of SPD learning
   :figclass: unavailable
   :class: no-scaled-link

  Learns on covariance/connectivity as SPD matrices using BiMap/ReEig/LogEig layers.

 .. grid-item-card:: |lightbulb| Large Brain Models
  :shadow: sm

  :bdg-danger:`Large Brain Model`

  .. figure:: ../_static/model_cat/lbm.png
   :width: 90%
   :align: center
   :alt: Diagram of transformer models
   :class: no-scaled-link

  Large-scale brain model layers require self-supervised pre-training to work effectively.

 .. grid-item-card:: |share-nodes| Graph Neural Network
  :shadow: sm

  :bdg-light:`Graph Neural Network`

  .. figure:: ../_static/model_cat/gnn.png
   :width: 90%
   :align: center
   :alt: Diagram of GNN models
   :figclass: unavailable
   :class: no-scaled-link

  Treats channels/regions as nodes with learned/static edges to model connectivity.

 .. grid-item-card:: |clone| Channel-Domain
  :shadow: sm

  :bdg-dark-line:`Channel`

  .. figure:: ../_static/model_cat/channel.png
   :width: 90%
   :align: center
   :alt: Diagram of channel-domain methods
   :class: no-scaled-link

  Usage montage information with spatial filtering / channel / hemisphere / brain region selection strategies.


- Across most architectures, the earliest stages are convolutional (:bdg-success:`Convolution`), reflecting the brain time series's noisy, locally structured nature.
  These layers apply temporal and/or spatial convolutions—often depthwise-separable as in EEGNet, per-channel or across channel groups to extract robust local features.
  :class:`EEGNet`, :class:`ShallowFBCSPNet`, :class:`EEGNeX`, and :class:`EEGInceptionERP`
- In the **recurrent** family (:bdg-secondary:`Recurrent`), many modern EEG models actually rely on *temporal convolutional networks* (TCNs) with dilations to grow the receptive field, rather than explicit recurrence (:cite:label:`bai2018tcn`), :class:`BDTCN`,
- In contrast, several methods employ **small attention** modules (:bdg-info:`Small Attention`) to capture longer-range dependencies efficiently, e.g., :class:`EEGConformer`, :class:`CTNet`, :class:`ATCNet`, :class:`AttentionBaseNet` (:cite:label:`song2022eeg,zhao2024ctnet,altaheri2022atcnet`).
- **Filterbank-style models** (:bdg-primary:`Filterbank`) explicitly decompose signals into multiple bands before (or while) learning, echoing the classic FBCSP pipeline; examples include :class:`FBCNet` and :class:`FBMSNet` (:cite:label:`mane2021fbcnet,liu2022fbmsnet`).
- **Interpretability-by-design** (:bdg-warning:`Interpretability`) architectures expose physiologically meaningful primitives (e.g., band-pass/sinc filters, variance or connectivity features), enabling direct neuroscientific inspection; see :class:`SincShallowNet` and :class:`EEGMiner` (:cite:label:`borra2020interpretable,ludwig2024eegminer`).
- **SPD / Riemannian** (:bdg-dark:`SPD`) methods operate on covariance (or connectivity) matrices as points on the SPD manifold, combining layers such as BiMap, ReEig, and LogEig; deep SPD networks and Riemannian classifiers motivate this family (:cite:label:`huang2017riemannian`). *(Coming soon in a dedicate repository.)*
- **Large-model / Transformer** (:bdg-danger:`Large Brain Model`) approaches pretrain attention-based encoders on diverse biosignals and fine-tune for EEG tasks; e.g., :class:`BIOT` (:cite:label:`yang2023biot`), :class:`Labram` (:cite:label:`jiang2024large`). These typically need a heavily self-supervised pre-training before decoding.
- **Graph neural networks** (:bdg-light:`Graph Neural Network`) treat channels/regions as nodes with learned (static or dynamic) edges to model functional connectivity explicitly; representative EEG-GNN, more common in the epileptic decoding (:cite:label:`klepl2024graph`).
- **Channel-domain robustness** (:bdg-dark-line:`Channel`) techniques target variability in electrode layouts by learning montage-agnostic or channel-selective layers (e.g., dynamic spatial filtering, differentiable channel re-ordering); these strategies improve cross-setup generalization :class:`SignalJEPA` (:cite:label:`guetschel2024sjepa,chen2024eegprogress`).


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

.. toctree::
   :caption: Categorization
   :maxdepth: 2
   :hidden:

   categorization/convolution
   categorization/recurrent
   categorization/attention
   categorization/filterbank
   categorization/interpretable
   categorization/spd
   categorization/lbm
   categorization/gnn
   categorization/channel

.. include:: /links.inc
