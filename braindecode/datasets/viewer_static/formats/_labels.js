/* ============================================================
   formats/_labels.js — channel-label fallback helpers shared by
   format readers. Centralises the "meta has labels or generate
   Ch1..ChN" pattern that every reader implemented before B2.
   ============================================================ */
(function () {
  'use strict';

  // Generate indexed labels Ch1..ChN. Used when no metadata-derived
  // names are available. Kept a plain for-loop so a hot-path caller
  // (very large channel counts) avoids the Array.from constructor
  // dispatch.
  function indexed(n) {
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = 'Ch' + (i + 1);
    return out;
  }

  // Pull labels from a BIDS sidecar meta object iff it has exactly
  // the expected channel count. Otherwise fall back to indexed
  // labels. The length check is intentionally strict (===) because
  // a mismatched sidecar usually means the .tsv is wrong, not that
  // we should silently truncate or pad.
  function fromMetaOr(meta, nCh) {
    if (meta && Array.isArray(meta.channels) && meta.channels.length === nCh) {
      return meta.channels.map((c) => c.name);
    }
    return indexed(nCh);
  }

  const api = { indexed, fromMetaOr };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.ChannelLabels = api;
})();
