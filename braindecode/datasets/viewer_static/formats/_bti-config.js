/* ============================================================
   formats/_bti-config.js — placeholder for the BTi `config` file
   parser. Currently a deliberately empty stub: the BTi reader in
   formats/bti.js opens recordings purely from the PDF (data file)
   tail header, which carries n_channels / n_samples / sampling
   rate / dtype. The `config` file's value-add is:

     - Per-channel `units_per_bit` + `gain` for absolute Tesla-scale
       calibration of integer-format PDFs (data_format ∈ {1, 2}).
     - Channel labels from the B_ch_labels user block (A1..A248 +
       E1..E64 + TRIGGER + RESPONSE for a typical Magnes WH3600).
     - The weight tables (B_E_table_used + B_weights_used) that
       MNE-Python uses to reconstruct the BTi → Neuromag coordinate
       transform.

   The real `config` is a multi-megabyte binary with dozens of
   variable-length user blocks. Parsing it correctly requires
   following every UB_B_* case in mne/io/bti/bti.py:_read_config
   (lines 200-618 of /tmp/mne_bti.py at the time of authorship);
   that's deferred until we hit a recording where the PDF tail
   header doesn't suffice.

   When this module gains a parser it should expose:
     BtiConfig.parse(arrayBuffer) → { channels: [{name, gain, ...}], ... }
     BtiConfig.parseLabelsOnly(arrayBuffer) → string[]   // fast path

   The reader in formats/bti.js falls back to indexed labels Ch1..ChN
   in the meantime — the channel-name table is a quality-of-life
   improvement, not a correctness blocker.
   ============================================================ */
(function () {
  'use strict';

  const api = {
    /**
     * Placeholder. Throws so an accidental early caller fails loudly
     * rather than silently returning a stub. When the parser ships,
     * replace this with the actual implementation.
     */
    parse() {
      throw new Error('BtiConfig.parse: not yet implemented — see formats/_bti-config.js');
    },
    parseLabelsOnly() {
      throw new Error('BtiConfig.parseLabelsOnly: not yet implemented — see formats/_bti-config.js');
    },
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.BtiConfig = api;
})();
