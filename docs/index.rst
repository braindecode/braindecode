Welcome to Braindecode
======================

A deep learning toolbox to decode raw time-domain EEG.

For EEG researchers that want to work with deep learning and
deep learning researchers that want to work with EEG data.
For now focussed on convolutional networks.

Installation
============

1. Install pytorch from http://pytorch.org/ (you don't need to install torchvision).

2. Install `MOABB <https://github.com/NeuroTechX/moabb>`_ via pip (needed if you want to use MOABB datasets utilities):

.. code-block:: bash

  pip install moabb

3. Install braindecode via pip:

.. code-block:: bash

  pip install braindecode

alternatively, if you use conda, you could create a dedicated environment with the following:

.. code-block:: bash

  curl -O https://raw.githubusercontent.com/braindecode/braindecode/master/environment.yml
  conda env create -f environment.yml
  conda activate braindecode

alternatively, install the latest version of braindecode via pip:

.. code-block:: bash

  pip install -U https://api.github.com/repos/braindecode/braindecode/zipball/master


Get Started
===========
Here an example to give you a quick idea how Braindecode is used to load and preprocess an EEG dataset
and train a deep network on it.
This example may not give particularly good performance,
note the examples below for better-performing training pipelines.

.. code-block:: python

    import torch
    from skorch.callbacks import LRScheduler
    from skorch.helper import predefined_split
    from braindecode.datasets.moabb import MOABBDataset
    from braindecode.preprocessing.preprocess import preprocess, Preprocessor
    from braindecode.preprocessing.windowers import create_windows_from_events
    from braindecode.models import ShallowFBCSPNet
    from braindecode import EEGClassifier

    # Load the data
    dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[1])

    # Preprocess the data
    preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
        Preprocessor(lambda x: x * 1e6) # Convert from V to uV
    ]
    preprocess(dataset, preprocessors)

    # Cut trials windows from the data
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        preload=True,
    )

    # Create the model
    model = ShallowFBCSPNet(
        in_chans=train_set[0][0].shape[0],
        n_classes=4,
        input_window_samples=train_set[0][0].shape[1],
        final_conv_length='auto',
    )

    # Create the skorch classifier object for training.
    # These learning rates and weight decay work well for the shallow network:
    clf = EEGClassifier(
        model,
        criterion=torch.nn.NLLLoss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),  # using valid_set for validation
        optimizer__lr=0.0625 * 0.01,
        optimizer__weight_decay=0,
        batch_size=64,
        callbacks=[
            "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
        ],
    )

    # Model training for a specified number of epochs. `y` is None as it is already supplied
    # in the dataset.
    clf.fit(train_set, y=None, epochs=4)



Learn how to use braindecode for ...

.. toctree::
   :maxdepth: 1

    Basic trialwise decoding <auto_examples/plot_bcic_iv_2a_moabb_trial.rst>
    More data-efficient "cropped decoding" <auto_examples/plot_bcic_iv_2a_moabb_cropped.rst>
    Your own datasets through MNE <auto_examples/plot_mne_dataset_example.rst>
    Your own datasets through Numpy <auto_examples/plot_custom_dataset_example.rst>

Examples
========
.. toctree::
   :maxdepth: 1

   auto_examples/index

Public API
==========
.. toctree::
   :maxdepth: 1

   api

What's new
==========
.. toctree::
   :maxdepth: 1

   whats_new

Troubleshooting
===============

Please report any issues on github: https://github.com/braindecode/braindecode/issues



Citing
======

If you use this code in a scientific publication, please cite us as:

.. code-block:: bibtex

  @article {HBM:HBM23730,
  author = {Schirrmeister, Robin Tibor and Springenberg, Jost Tobias and Fiederer,
    Lukas Dominique Josef and Glasstetter, Martin and Eggensperger, Katharina and Tangermann, Michael and
    Hutter, Frank and Burgard, Wolfram and Ball, Tonio},
  title = {Deep learning with convolutional neural networks for EEG decoding and visualization},
  journal = {Human Brain Mapping},
  issn = {1097-0193},
  url = {http://dx.doi.org/10.1002/hbm.23730},
  doi = {10.1002/hbm.23730},
  month = {aug},
  year = {2017},
  keywords = {electroencephalography, EEG analysis, machine learning, end-to-end learning, brain–machine interface,
    brain–computer interface, model interpretability, brain mapping},
  }

as well as the `MNE-Python <https://mne.tools>`_ software that is used by braindecode:

.. code-block:: bibtex

  @article{10.3389/fnins.2013.00267,
  author={Gramfort, Alexandre and Luessi, Martin and Larson, Eric and Engemann, Denis and Strohmeier, Daniel and Brodbeck, Christian and Goj, Roman and Jas, Mainak and Brooks, Teon and Parkkonen, Lauri and Hämäläinen, Matti},
  title={{MEG and EEG data analysis with MNE-Python}},
  journal={Frontiers in Neuroscience},
  volume={7},
  pages={267},
  year={2013},
  url={https://www.frontiersin.org/article/10.3389/fnins.2013.00267},
  doi={10.3389/fnins.2013.00267},
  issn={1662-453X},
  }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _GitHub: https://github.com/braindecode/braindecode
