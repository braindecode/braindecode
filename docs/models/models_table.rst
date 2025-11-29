:html_theme.sidebar_secondary.remove: true

.. currentmodule:: braindecode.models

.. raw:: html

  <style>
  /* ==== Badges ========================================================== */
  .tag, a.tag {
    display:inline-block; padding:4px 10px; margin:2px;
    border-radius:9999px; font-size:.85em; font-weight:600; line-height:1;
    text-decoration:none; border:1px solid transparent;
    background:#eef2f7; color:#334155;
  }
  /* Palette (force override against theme/link styles) */
  .tag.tag-conv      { background:#22c55e!important; color:#fff!important; border-color:#16a34a!important; }
  .tag.tag-recurrent { background:#8b5cf6!important; color:#fff!important; border-color:#7c3aed!important; }
  .tag.tag-smallattn { background:#3b82f6!important; color:#fff!important; border-color:#2563eb!important; }
  .tag.tag-filterbank{ background:#06b6d4!important; color:#073042!important; border-color:#0891b2!important; }
  .tag.tag-interp    { background:#f59e0b!important; color:#1f2937!important; border-color:#d97706!important; }
  .tag.tag-spd       { background:#111827!important; color:#fff!important;  border-color:#0b1220!important; }
  .tag.tag-lbm       { background:#e11d48!important; color:#fff!important;  border-color:#be123c!important; }
  .tag.tag-gnn       { background:#475569!important; color:#fff!important;  border-color:#334155!important; }
  .tag.tag-channel   { background:#64748b!important; color:#fff!important;  border-color:#475569!important; }

  /* Optional polish */
  .tag { box-shadow:0 0 0 0 rgba(0,0,0,0); transition:box-shadow .15s, transform .15s; }
  .tag:hover { box-shadow:0 2px 8px rgba(0,0,0,.08); transform:translateY(-1px); }
  </style>

  <!-- jQuery + DataTables core CSS -->
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <link rel="stylesheet" href="https://cdn.datatables.net/v/bm/dt-1.13.4/datatables.min.css"/>

Models Summary
~~~~~~~~~~~~~~

This page offers a summary of all :class:`braindecode` implemented models. For more information on each model, please consult the API.

Columns definitions:

 - **Model**: The name of the model.
 - **Application**: The application(s) the model is typically used for (e.g., Motor Imagery, P300, Sleep Staging). 'General' indicates applicability across multiple applications or no specific application focus.
 - **Type**: The model's primary function (e.g., Classification, Regression, Embedding).
 - **Sampling Frequency**: The data sampling rate (in Hertz) the model is designed for. Note that this might be adaptable depending on the specific dataset and application.
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
  :file: ../generated/models_summary_table.html

The parameter counts shown in the table were calculated using consistent hyperparameters for models within the same paradigm, based largely on Braindecode's default implementation values. These counts provide a relative comparison but may differ from those reported in the original publications due to variations in specific architectural details, input dimensions used in the paper, or calculation methods.

We are continually expanding this collection and welcome contributions! If you have implemented a model relevant to EEG, EcoG, or MEG analysis, consider adding it to Braindecode.

.. button-ref:: models_visualization
   :ref-type: doc
   :color: primary
   :expand:

    Next: Models Parameter Visualization
.. raw:: html

  <style>
  /* === Tag palette (unchanged, with stronger specificity) ============== */
  .tag, a.tag {
    display:inline-block; padding:4px 10px; margin:2px; border-radius:9999px;
    font-size:.85em; font-weight:600; line-height:1; text-decoration:none;
    border:1px solid transparent; background:#eef2f7; color:#334155;
  }
  .tag.tag-conv      { background:#22c55e!important; color:#fff!important;  border-color:#16a34a!important; }
  .tag.tag-recurrent { background:#8b5cf6!important; color:#fff!important;  border-color:#7c3aed!important; }
  .tag.tag-smallattn { background:#3b82f6!important; color:#fff!important;  border-color:#2563eb!important; }
  .tag.tag-filterbank{ background:#06b6d4!important; color:#073042!important; border-color:#0891b2!important; }
  .tag.tag-interp    { background:#f59e0b!important; color:#1f2937!important; border-color:#d97706!important; }
  .tag.tag-spd       { background:#111827!important; color:#fff!important;  border-color:#0b1220!important; }
  .tag.tag-lbm       { background:#e11d48!important; color:#fff!important;  border-color:#be123c!important; }
  .tag.tag-gnn       { background:#475569!important; color:#fff!important;  border-color:#334155!important; }
  .tag.tag-channel   { background:#64748b!important; color:#fff!important;  border-color:#475569!important; }
  .tag { box-shadow:0 0 0 0 rgba(0,0,0,0); transition:box-shadow .15s, transform .15s; }
  .tag:hover { box-shadow:0 2px 8px rgba(0,0,0,.08); transform:translateY(-1px); }
  </style>

  <!-- jQuery + DataTables core -->
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <link rel="stylesheet" href="https://cdn.datatables.net/v/bm/dt-1.13.4/datatables.min.css"/>
  <script src="https://cdn.datatables.net/v/bm/dt-1.13.4/datatables.min.js"></script>

  <!-- Buttons + SearchPanes (+ Select required by SearchPanes) -->
  <link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.dataTables.min.css">
  <script src="https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js"></script>
  <link rel="stylesheet" href="https://cdn.datatables.net/select/1.7.0/css/select.dataTables.min.css">
  <link rel="stylesheet" href="https://cdn.datatables.net/searchpanes/2.3.1/css/searchPanes.dataTables.min.css">
  <script src="https://cdn.datatables.net/select/1.7.0/js/dataTables.select.min.js"></script>
  <script src="https://cdn.datatables.net/searchpanes/2.3.1/js/dataTables.searchPanes.min.js"></script>

  <script>
  $(function () {
    /* --- Palette assign -------------------------------------------------- */
    function applyTagPalette(ctx) {
      var map = {
        "Convolution":"tag-conv","Recurrent":"tag-recurrent","Small Attention":"tag-smallattn",
        "Filterbank":"tag-filterbank","Interpretability":"tag-interp",
        "SPD":"tag-spd","Riemannian":"tag-spd",
        "Large Brain Model":"tag-lbm","Graph Neural Network":"tag-gnn","Channel":"tag-channel"
      };
      $(ctx).find('.tag').each(function () {
        var t = $(this).text().trim();
        $(this).removeClass('tag-conv tag-recurrent tag-smallattn tag-filterbank tag-interp tag-spd tag-lbm tag-gnn tag-channel');
        if (map[t]) $(this).addClass(map[t]);
        else if (t.includes('SPD') || t.includes('Riemannian')) $(this).addClass('tag-spd');
      });
    }

    /* --- Helpers for panes: split multi-badge cells ---------------------- */
    function tagsArrayFromHtml(html) {
      var div = document.createElement('div'); div.innerHTML = html;
      var tags = Array.from(div.querySelectorAll('.tag')).map(el => el.textContent.trim()).filter(Boolean);
      if (!tags.length) {
        var txt = $(div).text();
        tags = txt.split(/,\s*|\s+/).map(s => s.trim()).filter(Boolean);
      }
      return tags;
    }

    /* --- DataTable: panes hidden until button click ---------------------- */
    var FILTER_COLS = [1,2,3,4]; // Paradigm, Type, Categorization, Hyperparameters
    var table = $('.sortable').DataTable({
      dom: 'Blfrtip',                           // B = Buttons (no 'P' here)
      paging: false,
      searching: true,
      info: false,
      language: {
        search: "Filter models:",
        // Make the button compact and show active count when open
        searchPanes: { collapse: { 0: 'Filters', _: 'Filters (%d)' } }  // button text
      },
      buttons: [{
        extend: 'searchPanes',                  // open panes on demand via button
        text: 'Filters',
        config: {
          cascadePanes: true,
          viewTotal: true,
          layout: 'columns-4',
          initCollapsed: false                  // start panes collapsed for a tidy UI
        }
      }],
      columnDefs: [
        { searchPanes: { show: true }, targets: FILTER_COLS },
        {
          targets: FILTER_COLS,
          render: { _: d => d, sp: d => tagsArrayFromHtml(d) }, // feed arrays to panes
          searchPanes: { orthogonal: 'sp'}
        }
      ],
      drawCallback: function (s) { applyTagPalette(s.nTableWrapper || document); }
    });

    // Initial color pass
    applyTagPalette(document);

    /* --- Bonus UX: click a column header to open panes focused there ----- */
    $('.sortable thead th').each(function (i) {
      if (!FILTER_COLS.includes(i)) return;
      $(this).css('cursor','pointer').attr('title','Click to filter this column');
      $(this).on('click', function () {
        // Open the SearchPanes via the button (panes init on first click)
        table.button('.buttons-searchPanes').trigger();

        // After panes are created, expand only the matching pane
        setTimeout(function () {
          var idx = FILTER_COLS.indexOf(i);
          var $container = $(table.searchPanes.container());           // get panes container
          var $pane = $container.find('.dtsp-pane').eq(idx);
          // If collapsed, click its title to expand
          var $title = $pane.find('.dtsp-title');
          if ($title.length) $title.trigger('click');
        }, 0);
      });
    });
  });
  </script>
