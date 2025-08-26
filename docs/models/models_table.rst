:html_theme.sidebar_secondary.remove: true

.. currentmodule:: braindecode.models

.. raw:: html

  <style>
  /* Base tag */
  .tag,
  a.tag {
    display: inline-block;
    padding: 4px 10px;
    margin: 2px 2px;
    border-radius: 9999px;
    font-size: 0.85em;
    font-weight: 600;
    line-height: 1;
    text-decoration: none;
    border: 1px solid transparent;
    background: #eef2f7;   /* neutral default for non-categorization tags */
    color: #334155;
  }

  /* Palette mapped to your figure’s badges */
  .tag-conv         { background:#22c55e; color:#ffffff; }  /* Convolution (green) */
  .tag-recurrent    { background:#8b5cf6; color:#ffffff; }  /* Recurrent (violet) */
  .tag-small-attn   { background:#3b82f6; color:#ffffff; }  /* Small Attention (blue) */
  .tag-filterbank   { background:#06b6d4; color:#073042; }  /* Filterbank (teal) */
  .tag-interp       { background:#f59e0b; color:#1f2937; }  /* Interpretability (amber) */
  .tag-spd          { background:#111827; color:#ffffff; }  /* SPD / Riemannian (near-black) */
  .tag-llm          { background:#e11d48; color:#ffffff; }  /* Large Language Model (rose/red) */
  .tag-gnn          { background:#475569; color:#ffffff; }  /* Graph Neural Network (slate) */
  .tag-channel      { background:#64748b; color:#ffffff; }  /* Channel (grey-blue) */

  /* Optional: subtle shadow & hover polish */
  .tag { box-shadow: 0 0 0 0 rgba(0,0,0,0); transition: box-shadow .15s ease-in-out, transform .15s; }
  .tag:hover { box-shadow: 0 2px 8px rgba(0,0,0,.08); transform: translateY(-1px); }
  </style>
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/v/bm/dt-1.13.4/datatables.min.css"/>

Models Summary
~~~~~~~~~~~~~~

This page offers a summary of all :class:`braindecode` implemented models. For more information on each model, please consult the API.

Columns definitions:

 - **Model**: The name of the model.
 - **Paradigm**: The paradigm(s) the model is typically used for (e.g., Motor Imagery, P300, Sleep Staging). 'General' indicates applicability across multiple paradigms or no specific paradigm focus.
 - **Type**: The model's primary function (e.g., Classification, Regression, Embedding).
 - **Freq (Hz)**: The data sampling rate (in Hertz) the model is designed for. Note that this might be adaptable depending on the specific dataset and application.
 - **Categorization**: models categorization based on the main building blocks used in the architecture. See Models Categorization page for more details.
 - **Hyperparameters**: The mandatory hyperparameters required for instantiating the model class. These may include:
    - :fa:`wave-square`\  **n_chans**, number of channels/electrodes/sensors,
    - :fa:`shapes`\  **n_outputs**, number of output classes or regression targets,
    - :fa:`clock`\  **n_times**, number of time points in the input window,
    - :fa:`wifi`\  **freq (Hz)**, sampling frequency,
    - :fa:`info-circle`\  **chs_info**, information about each individual EEG channel. Refer to :class:`mne.Info` (see its `"chs"` field for details)

 Also, `n_times` can be derived implicitly by providing both `sfreq` and `input_window_seconds`.

 - **#Parameters**: The approximate total number of trainable parameters in the model, calculated using a consistent configuration (see note below).

.. raw:: html

  <!-- Dropdown filter container -->
  <div id="custom-filters" style="margin-bottom: 15px;"></div>

.. raw:: html
  :file: ../generated/models_summary_table.html

The parameter counts shown in the table were calculated using consistent hyperparameters for models within the same paradigm, based largely on Braindecode's default implementation values. These counts provide a relative comparison but may differ from those reported in the original publications due to variations in specific architectural details, input dimensions used in the paper, or calculation methods.

We are continually expanding this collection and welcome contributions! If you have implemented a model relevant to EEG, EcoG, or MEG analysis, consider adding it to Braindecode.

.. button-ref:: models_visualization
   :ref-type: doc
   :color: primary
   :expand:

    Next: Models Parameter Visualization

.. raw:: html

  <script type="text/javascript" src="https://cdn.datatables.net/v/bm/dt-1.13.4/datatables.min.js"></script>
  <script type="text/javascript">
  $(document).ready(function () {
    /* === 1) Colorize categorization badges by their text =================== */
    var map = {
      "Convolution": "tag-conv",
      "Recurrent": "tag-recurrent",
      "Small Attention": "tag-small-attn",
      "Filterbank": "tag-filterbank",
      "Interpretability": "tag-interp",
      "SPD": "tag-spd",
      "Riemannian": "tag-spd",
      "Large Language Model": "tag-llm",
      "Graph Neural Network": "tag-gnn",
      "Channel": "tag-channel"
    };
    $('.tag').each(function () {
      var t = $(this).text().trim();
      // Exact match
      if (map[t]) { $(this).addClass(map[t]); return; }
      // Handle composite labels like "SPD / Riemannian"
      if (t.indexOf('SPD') > -1 || t.indexOf('Riemannian') > -1) $(this).addClass('tag-spd');
    });

    /* === 2) DataTable init ================================================ */
    var table = $('.sortable').DataTable({
      paging: false,
      searching: true,
      info: false,
      language: { search: "Filter models:" }
    });

    var filterColumns = {
      1: "Select Paradigm",
      2: "Choose Model Type",
      3: "Select Model Category",
      4: "Select Required Hyperparameter"
    };

    var $filterContainer = $('#custom-filters');
    var dropdowns = {};

    $.each(filterColumns, function (colIdx, label) {
      var select = $('<select><option value="" disabled selected>' + label + '</option></select>')
        .css({ 'margin-right': '10px', 'padding': '5px', 'margin-bottom': '10px' });

      var uniqueTags = new Set();

      table.column(colIdx).data().each(function (d) {
        if (d) {
          var tags = [];
          var $temp = $('<div>' + d + '</div>');
          var $foundTags = $temp.find('.tag');

          if ($foundTags.length > 0) {
            $foundTags.each(function () {
              var tag = $(this).text().trim();
              if (tag) tags.push(tag);
            });
          } else {
            var plainText = $temp.text();
            tags = plainText.split(/,\s*|\s+/).map(function (tag) { return tag.trim(); });
          }

          tags.forEach(function (tag) { if (tag) uniqueTags.add(tag); });
        }
      });

      Array.from(uniqueTags).sort().forEach(function (tag) {
        select.append('<option value="' + tag + '">' + tag + '</option>');
      });

      select.on('change', function () {
        var val = $(this).val();
        table.column(colIdx).search(val ? val : '', true, false).draw();
      });

      dropdowns[colIdx] = select;
      $filterContainer.append(select);
    });

    var clearBtn = $('<button>Clear Filters</button>')
      .css({ 'margin-left': '10px', 'padding': '5px 10px', 'cursor': 'pointer' })
      .on('click', function () {
        $.each(dropdowns, function (colIdx, select) {
          select.prop('selectedIndex', 0);
          table.column(colIdx).search('').draw();
        });
      });

    $filterContainer.append(clearBtn);
  });
  </script>

.. include:: /links.inc
