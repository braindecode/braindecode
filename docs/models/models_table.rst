:html_theme.sidebar_secondary.remove: true

.. currentmodule:: braindecode.models

################
 Models Summary
################

This page offers a summary of all :class:`braindecode` implemented models. For more
information on each model, please consult the API.

Columns definitions:

    - **Model**: The name of the model.
    - **Application**: The application(s) the model is typically used for (e.g., Motor
      Imagery, P300, Sleep Staging). 'General' indicates applicability across multiple
      applications or no specific application focus.
    - **Modality**: The recording modality (bio-signal type) the model is designed and
      validated for, e.g. EEG, MEG, iEEG, ECoG, sEMG, or fNIRS. Most Braindecode models
      target EEG; a few support more than one modality (e.g. EEG, MEG).
    - **Type**: The model's output interface. ``Prediction`` indicates a supervised head
      that can be used for classification or regression, depending on the wrapper, loss,
      and target. ``Embedding`` indicates a model that exposes an embedding
      representation through its documented public interface.
    - **Sampling Frequency**: The data sampling rate (in Hertz) the model is designed
      for. Note that this might be adaptable depending on the specific dataset and
      application.
    - **Categorization**: models categorization based on the main building blocks used
      in the architecture. See Models Categorization page for more details.
    - **Hyperparameters**: The mandatory hyperparameters required for instantiating the model class. These may include:
          - :fa:`wave-square` **n_chans**, number of channels/electrodes/sensors,
          - :fa:`shapes` **n_outputs**, number of output classes or regression targets,
          - :fa:`clock` **n_times**, number of time points in the input window,
          - :fa:`wifi` **freq (Hz)**, sampling frequency,
          - :fa:`info-circle` **chs_info**, information about each individual EEG
            channel. Refer to :class:`mne.Info` (see its `"chs"` field for details)

    Also, `n_times` can be derived implicitly by providing both `sfreq` and
    `input_window_seconds`.

    - **#Parameters**: The approximate total number of trainable parameters in the
      model, calculated using a consistent configuration (see note below).

.. raw:: html
    :file: ../generated/models_summary_table.html

The parameter counts shown in the table were calculated using consistent hyperparameters
for models within the same paradigm, based largely on Braindecode's default
implementation values. These counts provide a relative comparison but may differ from
those reported in the original publications due to variations in specific architectural
details, input dimensions used in the paper, or calculation methods.

We are continually expanding this collection and welcome contributions! If you have
implemented a model relevant to EEG, ECoG, MEG, fNIRS, or sEMG analysis, consider adding
it to Braindecode.

.. button-ref:: models_visualization
    :ref-type: doc
    :color: primary
    :expand:

     Next: Models Parameter Visualization

.. raw:: html

    <style>
    /* === Tabular numerals on numeric columns & SearchPanes counts ========= */
    .sortable td:nth-child(6),
    .sortable td:nth-child(7),
    .sortable th:nth-child(6),
    .sortable th:nth-child(7),
    .dtsp-pill {
      font-variant-numeric: tabular-nums;
    }

    /* === Accessible semantic tag palette (light & dark) =================== */
    .tag,
    a.tag,
    html[data-theme="light"] .tag,
    html[data-theme="light"] a.tag {
      display: inline-block;
      padding: 4px 10px;
      margin: 2px;
      border-radius: 9999px;
      font-size: 0.85em;
      font-weight: 600;
      line-height: 1;
      text-decoration: none;
      border: 1px solid #d6dbe1;
      background: #f1f3f5;
      color: #2a323d;
      box-shadow: 0 0 0 0 rgba(0, 0, 0, 0);
      transition: box-shadow 0.15s ease;
    }
    html[data-theme="dark"] .tag,
    html[data-theme="dark"] a.tag {
      background: #252b36;
      color: #dfd9cc;
      border-color: #3a4150;
    }

    /* Foundation Model — warm copper */
    .tag.tag-lbm,
    html[data-theme="light"] .tag.tag-lbm {
      background: #f9efe6 !important;
      color: #8f511e !important;
      border-color: #ebcfb5 !important;
    }
    html[data-theme="dark"] .tag.tag-lbm {
      background: #3a2a1a !important;
      color: #f5b982 !important;
      border-color: #5c4025 !important;
    }

    /* Convolution & Attention — steel cerulean */
    .tag.tag-conv,
    .tag.tag-smallattn,
    html[data-theme="light"] .tag.tag-conv,
    html[data-theme="light"] .tag.tag-smallattn {
      background: #eef3f8 !important;
      color: #234b70 !important;
      border-color: #cfdeec !important;
    }
    html[data-theme="dark"] .tag.tag-conv,
    html[data-theme="dark"] .tag.tag-smallattn {
      background: #1d2c3a !important;
      color: #96cbef !important;
      border-color: #2e475e !important;
    }

    /* Filterbank & Graph Neural Network — clinical sage */
    .tag.tag-filterbank,
    .tag.tag-gnn,
    html[data-theme="light"] .tag.tag-filterbank,
    html[data-theme="light"] .tag.tag-gnn {
      background: #e7f2ea !important;
      color: #2f633d !important;
      border-color: #c3decb !important;
    }
    html[data-theme="dark"] .tag.tag-filterbank,
    html[data-theme="dark"] .tag.tag-gnn {
      background: #1b3122 !important;
      color: #9adba8 !important;
      border-color: #2c5038 !important;
    }

    /* Recurrent, Channel, Interpretability, SPD — crisp slate */
    .tag.tag-recurrent,
    .tag.tag-channel,
    .tag.tag-interp,
    .tag.tag-spd,
    html[data-theme="light"] .tag.tag-recurrent,
    html[data-theme="light"] .tag.tag-channel,
    html[data-theme="light"] .tag.tag-interp,
    html[data-theme="light"] .tag.tag-spd {
      background: #f1f3f5 !important;
      color: #2a323d !important;
      border-color: #d6dbe1 !important;
    }
    html[data-theme="dark"] .tag.tag-recurrent,
    html[data-theme="dark"] .tag.tag-channel,
    html[data-theme="dark"] .tag.tag-interp,
    html[data-theme="dark"] .tag.tag-spd {
      background: #252b36 !important;
      color: #dfd9cc !important;
      border-color: #3a4150 !important;
    }

    .tag:hover {
      box-shadow: 0 2px 8px rgba(20, 24, 31, 0.08);
    }
    @media (prefers-reduced-motion: no-preference) {
      .tag {
        transition: box-shadow 0.15s ease, transform 0.15s ease;
      }
      .tag:hover {
        transform: translateY(-1px);
      }
    }
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
          "Convolution":"tag-conv","Recurrent":"tag-recurrent","Attention/Transformer":"tag-smallattn",
          "Filterbank":"tag-filterbank","Interpretability":"tag-interp",
          "SPD":"tag-spd","Riemannian":"tag-spd",
          "Foundation Model":"tag-lbm","Graph Neural Network":"tag-gnn","Channel":"tag-channel"
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
      var FILTER_COLS = [1,2,3,4]; // Application, Modality, Type, Categorization
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

      function applyInitialModelFilter() {
        var model = new URLSearchParams(window.location.search).get('model');
        if (!model) return;
        table.search(model).draw();
        $(table.table().container()).find('div.dataTables_filter input').val(model);
      }

      applyInitialModelFilter();

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
