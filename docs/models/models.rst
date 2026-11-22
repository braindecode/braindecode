:html_theme.sidebar_secondary.remove: true

.. currentmodule:: braindecode.models

.. _models:

##########################
 The brain decode problem
##########################

All models in Braindecode address a shared supervised or self-supervised problem: given
multichannel time-series recordings :math:`X \in \mathbb{R}^{C \times T}` and targets
:math:`y \in \mathcal{Y}`, :mod:`braindecode.models` implements neural networks
:math:`f` that **decode** brain activity by applying a sequence of transformation layers
(such as :class:`~torch.nn.Conv2d`, :class:`~torch.nn.Linear`, and
:class:`~torch.nn.ELU`) to filter signals and extract task-relevant representations:

.. math::

    f_{\theta} : X \to y,

where :math:`C` (``n_chans``) is the number of recording channels or electrodes and
:math:`T` (``n_times``) is the temporal window length (epoch size in samples) over the
interval of interest.

The target :math:`y` can represent a sensory or cognitive paradigm (motor imagery BCI,
ERP, SSVEP, cVEP), a physiological or clinical state (sleep stage, pathology, brain
age), continuous visual, auditory, speech, or motor variables, or any quantity modeled
as a decoding task
:cite:label:`sleep2023,aristimunhaetal23,chevallier2024largest,levy2025brain,benchetrit2024brain,d2024decoding,engemann2022reusable,xu2024alljoined`.

Our goal is to map recorded brain activity back to its originating stimulus, behavior,
or clinical state :cite:t:`king2014characterizing,king2020`, written compactly as
:math:`f(X) \to y`. The neural network :math:`f` learns representations of time-series
activity that predict the underlying variable of interest, a setting often called
*reverse inference*.

In supervised decoding, we estimate the network parameters :math:`\theta` by minimizing
the regularized average loss over the training set
:math:`\mathcal{D}_{\text{tr}}=\{(x_i,y_i)\}_{i=1}^{N_{\text{tr}}}`:

.. math::

    \begin{aligned}
    \theta^{*}
      &= \arg\min_{\theta}\, \hat{\mathcal{R}}(\theta) \\
      &= \arg\min_{\theta}\, \frac{1}{N_{\text{tr}}}\sum_{i=1}^{N_{\text{tr}}}
         \ell\!\left(f_{\theta}(x_i),\, y_i\right) \;+\; \lambda\,\Omega(\theta)\,,
    \end{aligned}

where :math:`\ell` is the task loss (for example, cross-entropy
:class:`~torch.nn.CrossEntropyLoss`), :math:`\Omega` is an optional regularizer, and
:math:`\lambda \ge 0` controls regularization strength (for example, the
``weight_decay`` parameter in :class:`~torch.optim.Adam`).

Equivalently, training minimizes a finite-sample approximation to the expected risk
:math:`\mathcal{R}(\theta)=\mathbb{E}_{(x,y)\sim P_{\text{tr}}}
[\ell(f_{\theta}(x),y)]`.

Across the pages in this section, you will find:

1. The problem formulation above;
2. :doc:`Architecture categorization by core building blocks <models_categorization>`;
3. :doc:`Interactive models summary table <models_table>`;
4. :doc:`Channel and montage adaptation strategies for pretrained models
   <channel_strategies>`;
5. :doc:`Comparative visualizations across model families <models_visualization>`.

.. button-ref:: models_table
    :ref-type: doc
    :color: primary
    :expand:

    Browse the models table →

.. rubric:: References

.. bibliography::

.. toctree::
    :hidden:

    models_categorization
    models_table
    channel_strategies
    models_visualization

.. include:: /links.inc
