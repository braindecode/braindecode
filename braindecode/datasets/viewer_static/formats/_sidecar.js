/* ============================================================
   formats/_sidecar.js — small consistency-check helpers shared by
   every format reader. Each binary reader is the source of truth
   for n_channels, fs, and channel order; BIDS sidecars are
   redundant validations. When they disagree we trust the binary
   and surface a console.warn so the user can spot a malformed
   sidecar without it bricking the load.
   ============================================================ */
(function () {
  'use strict';

  const api = {};

  // Compare the binary header's channel count + per-row labels against
  // `_channels.tsv`. Length mismatch warns once; per-row name divergence
  // (case-insensitive — real-world headers are sloppy) warns per row.
  // Used by EDF + BrainVision; EEGLAB derives labels from channels.tsv
  // so no cross-check makes sense there.
  api.crossCheckChannelOrder = function (binLabels, bidsChannels, formatLabel) {
    if (!bidsChannels) return;
    if (bidsChannels.length !== binLabels.length) {
      console.warn(`channels.tsv has ${bidsChannels.length} rows but ${formatLabel} has ${binLabels.length} non-annotation signals.`);
      return;
    }
    for (let c = 0; c < binLabels.length; c++) {
      if (binLabels[c].toLowerCase() !== bidsChannels[c].name.toLowerCase()) {
        console.warn(`channels.tsv[${c}]="${bidsChannels[c].name}" ≠ ${formatLabel} label "${binLabels[c]}"; using ${formatLabel} order.`);
      }
    }
  };

  // The "_eeg.json declared fs disagrees with the binary header" case.
  // We always trust the binary; this just surfaces the discrepancy so
  // a user staring at confused-looking traces can find the cause.
  api.warnFsMismatch = function (sidecarFs, headerFs, formatLabel) {
    if (sidecarFs == null) return;
    if (Math.abs(sidecarFs - headerFs) <= 1e-3) return;
    console.warn(`${formatLabel} fs (${headerFs} Hz) disagrees with sidecar (${sidecarFs} Hz); trusting ${formatLabel} header.`);
  };

  // Common open()-time validation: probe the binary's byte length and
  // assert it's a whole multiple of `recordBytes`. Returns the byte
  // length on success; throws with format-specific context on failure.
  api.probeAndValidate = async function (url, recordBytes, formatLabel) {
    if (recordBytes <= 0) throw new Error(`${formatLabel}: zero-byte record (n_channels or bps is 0)`);
    if (typeof HttpRange === 'undefined') {
      // Surface a wiring bug at the call site rather than the
      // confusing global-not-defined that you'd otherwise see.
      throw new Error('SidecarChecks.probeAndValidate: HttpRange must be loaded first');
    }
    const totalBytes = await HttpRange.probeLength(url);
    if (totalBytes % recordBytes !== 0) {
      throw new Error(
        `${formatLabel} size ${totalBytes} is not a multiple of ${recordBytes}B; ` +
        `channel count or sample stride from the header may be wrong.`
      );
    }
    return totalBytes;
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.SidecarChecks = api;
})();
