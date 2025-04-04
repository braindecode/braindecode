.. _data_summary:
.. raw:: html

   <style>
    .tag {
      display: inline-block;
      background-color: #e0f7fa;
      color: #00796b;
      padding: 4px 8px;
      margin: 2px;
      border-radius: 4px;
      font-size: 0.9em;
    }
   </style>
   <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
   <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/v/bm/dt-1.13.4/datatables.min.css"/>

.. automodule:: braindecode.models

.. currentmodule:: braindecode.models

Models Summary
~~~~~~~~~~~~~~~

This page offers a summary of many implemented models. Please note that this list may not be exhaustive. For the definitive and most current list, including detailed class documentation, please consult the :doc:`API documentation <api>`.

We are continually expanding this collection and welcome contributions! If you have implemented a model relevant to EEG, EcoG, or MEG analysis, consider adding it to Braindecode. See the "Submit a New Model" section below for details.

.. figure:: _static/model/models_analysis.png
   :alt: Braindecode Models
   :align: center

   Visualization comparing the models based on their total number of parameters (left plot) and the primary experimental they were designed for (right plot).

Columns definitions:
    - **Model**: The name of the model.
    - **Paradigm**: The paradigm(s) the model is typically used for (e.g., Motor Imagery, P300, Sleep Staging). 'General' indicates applicability across multiple paradigms or no specific paradigm focus.
    - **Type**: The model's primary function (e.g., Classification, Regression, Embedding).
    - **Freq (Hz)**: The data sampling rate (in Hertz) the model is designed for. Note that this might be adaptable depending on the specific dataset and application.
    - **Hyperparameters**: The mandatory hyperparameters required for instantiating the model class. These may include `n_chans` (number of channels), `n_outputs` (number of output classes or regression targets), `n_times` (number of time points in the input window), or `sfreq` (sampling frequency). Also, `n_times` can be derived implicitly by providing both `sfreq` and `input_window_seconds`.
    - **#Parameters**: The approximate total number of trainable parameters in the model, calculated using a consistent configuration (see note below).

.. raw:: html
   :file: _build/models_summary_table.html

The parameter counts shown in the table were calculated using consistent hyperparameters for models within the same paradigm, based largely on Braindecode's default implementation values. These counts provide a relative comparison but may differ from those reported in the original publications due to variations in specific architectural details, input dimensions used in the paper, or calculation methods.


Submit a new model
~~~~~~~~~~~~~~~~~~~~

Want to contribute a new model to Braindecode? Great! You can propose a new model by opening an `issue <https://github.com/braindecode/braindecode/issues>`__ (please include a link to the relevant publication or description) or, even better, directly submit your implementation via a `pull request <https://github.com/braindecode/braindecode/pulls>`__. We appreciate your contributions to expanding the library!

.. raw:: html

   <script type="text/javascript" src="https://cdn.datatables.net/v/bm/dt-1.13.4/datatables.min.js"></script>
   <script type="text/javascript">
    $(document).ready(function() {
     var table = $('.sortable').DataTable({
       "paging": false,
       "searching": true,
       "info": false,
       language: {
         "search": "Filter models:"
       }
     });
    });
   </script>
