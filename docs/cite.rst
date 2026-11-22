.. _cite:

#########################
 How to cite Braindecode
#########################

Please cite Braindecode whenever you use the library in any context, including journal
articles, conference papers, preprints, clinical studies, benchmark suites, master's or
PhD dissertations, competition entries, and grant proposals.

Use the **Zenodo software DOI** (`10.5281/zenodo.8214376
<https://doi.org/10.5281/zenodo.8214376>`_) first as your primary citation. That concept
DOI covers every Braindecode release, resolves automatically to the latest version, and
credits the active maintainer and contributor team. When space permits, also cite the
foundational *Human Brain Mapping* (2017) paper and the specific neural architecture
papers used in your experiments.

.. raw:: html

    <div class="bd-cite-hub" role="region" aria-label="One-click citation export">
      <div class="bd-cite-hub__header">
        <div class="bd-cite-hub__meta">
          <span class="bd-cite-hub__eyebrow">PRIMARY SOFTWARE CITATION · ZENODO CONCEPT DOI</span>
          <a class="bd-cite-hub__doi" href="https://doi.org/10.5281/zenodo.8214376" target="_blank" rel="noopener">
            https://doi.org/10.5281/zenodo.8214376
          </a>
        </div>
        <a class="bd-cite-hub__badge" href="https://doi.org/10.5281/zenodo.8214376" target="_blank" rel="noopener">
          <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.8214376.svg" alt="Zenodo DOI 10.5281/zenodo.8214376"/>
        </a>
      </div>
      <div class="bd-cite-hub__toolbar">
        <button type="button" class="bd-cite-hub-btn bd-cite-hub-btn--primary" data-bd-cite-copy="zenodo-bibtex">
          Copy Zenodo BibTeX (Primary)
        </button>
        <button type="button" class="bd-cite-hub-btn" data-bd-cite-copy="both-bibtex">
          Copy Both BibTeX (Zenodo + HBM 2017)
        </button>
        <button type="button" class="bd-cite-hub-btn" data-bd-cite-copy="zenodo-apa">
          Copy APA
        </button>
        <button type="button" class="bd-cite-hub-btn" data-bd-cite-copy="zenodo-ris">
          Copy RIS (Zotero / EndNote / Mendeley)
        </button>
        <a class="bd-cite-hub-btn bd-cite-hub-btn--download" href="_static/braindecode.bib" download="braindecode.bib">
          Download braindecode.bib ↓
        </a>
      </div>
    </div>

****************************************************
 1. Cite the Braindecode software (Zenodo, primary)
****************************************************

Citing the Zenodo software archive directly tracks open-source software impact and
credits all contributors who build and maintain Braindecode across releases:

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.8214376.svg
    :target: https://doi.org/10.5281/zenodo.8214376
    :alt: Zenodo DOI 10.5281/zenodo.8214376

**BibTeX / BibLaTeX:**

.. code-block:: bibtex

    @software{braindecode,
      author = {Aristimunha, Bruno and
                Guetschel, Pierre and
                Wimpff, Martin and
                Gemein, Lukas and
                Rommel, Cedric and
                Banville, Hubert and
                Sliwowski, Maciej and
                Wilson, Daniel and
                Brandt, Simon and
                Gnassounou, Th{\'e}o and
                Paillard, Joseph and
                Srivastava, Aman and
                {Junqueira Lopes}, Bruna and
                Sedlar, Sara and
                Tayal, Sarthak and
                Truong, Young and
                Gadonneix, Julien and
                Mounir, Adam and
                Zhang, Zhibo and
                Soloviev, Anton and
                Li, Qing and
                Rathi, Raghav and
                Shah, Vandit and
                Arthur and
                Jain, Mahir and
                Kowshik, Bhargav and
                Li, Yiheng and
                Siddique, Fashad Ahmed and
                Patel, Aayushya and
                Moreau, Thomas and
                Chevallier, Sylvain and
                Gramfort, Alexandre and
                Schirrmeister, Robin Tibor},
      title = {Braindecode: toolbox for decoding raw electrophysiological brain data
               with deep learning models},
      year = {2026},
      publisher = {Zenodo},
      howpublished = {Zenodo},
      url = {https://github.com/braindecode/braindecode},
      doi = {10.5281/zenodo.8214376},
      license = {BSD-3-Clause}
    }

**APA:**

    Aristimunha, B., Guetschel, P., Wimpff, M., Gemein, L., Rommel, C., Banville, H.,
    Sliwowski, M., Wilson, D., Brandt, S., Gnassounou, T., Paillard, J., Srivastava, A.,
    Junqueira Lopes, B., Sedlar, S., Tayal, S., Truong, Y., Gadonneix, J., Mounir, A.,
    Zhang, Z., Soloviev, A., Li, Q., Rathi, R., Shah, V., Arthur, Jain, M., Kowshik, B.,
    Li, Y., Siddique, F. A., Patel, A., Moreau, T., Chevallier, S., Gramfort, A., &
    Schirrmeister, R. T. (2026). *Braindecode: toolbox for decoding raw
    electrophysiological brain data with deep learning models* [Computer software].
    Zenodo. https://doi.org/10.5281/zenodo.8214376

************************************************************
 2. Cite the foundational paper (Human Brain Mapping, 2017)
************************************************************

Alongside the Zenodo software DOI, we encourage citing the journal article that
introduced Braindecode, ShallowFBCSPNet, Deep4Net, and cropped convolutional decoding
for raw EEG:

.. raw:: html

    <div class="bd-cite-inline-actions">
      <button type="button" class="bd-cite-hub-btn" data-bd-cite-copy="hbm-bibtex">Copy HBM 2017 BibTeX</button>
      <button type="button" class="bd-cite-hub-btn" data-bd-cite-copy="hbm-apa">Copy HBM 2017 APA</button>
    </div>

.. code-block:: bibtex

    @article{HBM:HBM23730,
      author = {Schirrmeister, Robin Tibor and Springenberg, Jost Tobias and
                Fiederer, Lukas Dominique Josef and Glasstetter, Martin and
                Eggensperger, Katharina and Tangermann, Michael and
                Hutter, Frank and Burgard, Wolfram and Ball, Tonio},
      title = {Deep learning with convolutional neural networks for {EEG} decoding and visualization},
      journal = {Human Brain Mapping},
      issn = {1097-0193},
      url = {https://doi.org/10.1002/hbm.23730},
      doi = {10.1002/hbm.23730},
      month = {aug},
      year = {2017},
      keywords = {electroencephalography, EEG analysis, machine learning, end-to-end learning, brain-machine interface, brain-computer interface, model interpretability, brain mapping}
    }

*******************************************************
 3. Cite individual model architectures and MNE-Python
*******************************************************

**Model and dataset papers.** When you train, benchmark, or fine-tune specific neural
architectures or foundation models from :mod:`braindecode.models`, please also cite the
original papers introducing those models and datasets. Every model class lists its
primary publication in its docstring and in the :doc:`Models summary table
<models/models_table>`.

**MNE-Python.** Braindecode builds on `MNE-Python <https://mne.tools>`_ for
electrophysiological data structures, montages, and signal processing. You can cite the
`MNE-Python Zenodo archive <https://doi.org/10.5281/zenodo.592483>`_ and the canonical
MNE-Python journal article:

.. code-block:: bibtex

    @article{10.3389/fnins.2013.00267,
      author = {Gramfort, Alexandre and Luessi, Martin and Larson, Eric and Engemann,
                Denis and Strohmeier, Daniel and Brodbeck, Christian and Goj, Roman and
                Jas, Mainak and Brooks, Teon and Parkkonen, Lauri and H{\"a}m{\"a}l{\"a}inen, Matti},
      title = {{MEG and EEG data analysis with MNE-Python}},
      journal = {Frontiers in Neuroscience},
      volume = {7},
      pages = {267},
      year = {2013},
      url = {https://www.frontiersin.org/article/10.3389/fnins.2013.00267},
      doi = {10.3389/fnins.2013.00267},
      issn = {1662-453X}
    }
