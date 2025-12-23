..
    The page title must be in rST for it to show in next/prev page buttons.
    Therefore we add a special style rule to only this page that hides h1 tags

.. raw:: html

    <style type="text/css">h1 {display:none;}</style>

.. include:: links.inc

.. meta::
    :description: Braindecode is an open-source Python toolbox for deep learning on EEG, ECoG, and MEG with models, preprocessing, visualization, and dataset tooling.
    :keywords: EEG deep learning, brain-computer interface, MOABB, EEGDASH, EEG visualization, EEGPrep, MNE, foundation models

######################
 Braindecode Homepage
######################

.. container:: braindecode-hero

    .. image:: _static/braindecode.svg
        :alt: Braindecode
        :class: logo mainlogo only-light
        :align: center

    .. image:: _static/braindecode.svg
        :alt: Braindecode
        :class: logo mainlogo only-dark
        :align: center

    .. rst-class:: braindecode-hero-title

        Deep learning for EEG, ECoG, and MEG.

    .. rst-class:: h4 text-center font-weight-light my-3

        Braindecode is an open-source Python toolbox for decoding brain signals with
        ready-to-use models, preprocessing, and visualization.

    .. rst-class:: h5 text-center font-weight-light my-3

        Build reliable pipelines faster with benchmarks powered by MOABB_ and EEGDASH.

    .. container:: bd-hero-actions

        .. button-ref:: install/install
            :ref-type: doc
            :color: primary
            :shadow:
            :class: bd-hero-button

            Get started

        .. button-ref:: auto_examples/index
            :ref-type: doc
            :color: secondary
            :outline:
            :class: bd-hero-button

            Browse examples

.. grid:: 1 2 2 3
    :gutter: 3
    :padding: 1 0 2 0
    :class-container: bd-feature-grid

    .. grid-item-card:: |braille| Models
        :text-align: left
        :shadow: md

        .. rst-class:: bd-feature-kicker

            15+ EEG-ready networks from compact convolutions to foundation-style models.

        .. rst-class:: bd-feature-tags

            MNE-compatible / transfer learning / self-supervision

    .. grid-item-card:: |layer-group| Preprocessing
        :text-align: left
        :shadow: md

        .. rst-class:: bd-feature-kicker

            MNE pipelines and EEGPrep defaults to clean, resample, and epoch data.

        .. rst-class:: bd-feature-tags

            Filtering and referencing / reusable recipes / dataset defaults

    .. grid-item-card:: |eye| Visualization
        :text-align: left
        :shadow: md

        .. rst-class:: bd-feature-kicker

            Trial-wise diagnostics and interpretability for model behavior.

        .. rst-class:: bd-feature-tags

            Confusion matrices / attributions / signal inspection

    .. grid-item-card:: |rocket| Advanced training
        :text-align: left
        :shadow: md

        .. rst-class:: bd-feature-kicker

            Augmentation, self-supervision, and fine-tuning tailored to EEG.

        .. rst-class:: bd-feature-tags

            Contrastive pretraining / robust augmentation / fine-tuning

    .. grid-item-card:: |share-nodes| Datasets & ecosystem
        :text-align: left
        :shadow: md

        .. rst-class:: bd-feature-kicker

            Benchmark-ready datasets and community tooling in one place.

        .. rst-class:: bd-feature-tags

            MOABB_ loaders / EEGDASH discovery / PyTorch, MNE, scikit-learn

.. grid:: 1 2 2 2
    :gutter: 3
    :class-container: bd-cta-grid

    .. grid-item-card::
        :class-body: bd-cta-card
        :shadow: md

        .. rst-class:: bd-cta-title

            Quick start

        .. code-block:: bash

            pip install braindecode moabb
            python -c "import braindecode; print(braindecode.__version__)"

        .. rst-class:: bd-cta-copy

            Explore tutorials in :doc:`auto_examples/index` or try your first model from :doc:`models/models`.

    .. grid-item-card::
        :class-body: bd-cta-card
        :shadow: md

        .. rst-class:: bd-cta-title

            Stay in the loop

        .. rst-class:: bd-cta-list

        - Compare architectures in the :doc:`models/models` gallery.
        - Browse preprocessing and visualization in :doc:`auto_examples/index`.
        - Read :doc:`whats_new` for the latest features and releases.

.. frontpage gallery is added by a conditional in _templates/layout.html

.. toctree::
    :hidden:

    Install <install/install>
    Models <models/models>
    Cite <cite>
    Tutorial and Examples <auto_examples/index>
    API <api>
    Get help <help>
    What's new <whats_new>
