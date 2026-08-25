/* ============================================================
   formats/_decode.js — typed-array decode helpers shared by LE
   format readers. WARNING: this module covers ONLY little-endian
   (or already-typed-array) sources. CTF + FIFF use big-endian
   on-disk + a different scaling formula `(raw - offsets[c]) * cals[c]`
   and intentionally keep their own DataView loops.
   ============================================================ */
(function () {
  'use strict';

  // De-interleave a flat row-major LE typed array into per-channel
  // typed arrays.
  //
  //   out[c][s] = source[s * nCh + c] * (scales ? scales[c] : 1)
  //
  // `source` may be any TypedArray (Float32, Int16, …); `out` must
  // already be allocated (typically via ChannelBuffers.alloc) so the
  // caller controls per-channel storage type. The two-branch split
  // keeps the no-scale path branch-free.
  function deinterleaveInto(out, source, nCh, nWin, scales) {
    if (scales) {
      for (let s = 0; s < nWin; s++) {
        const base = s * nCh;
        for (let c = 0; c < nCh; c++) out[c][s] = source[base + c] * scales[c];
      }
    } else {
      for (let s = 0; s < nWin; s++) {
        const base = s * nCh;
        for (let c = 0; c < nCh; c++) out[c][s] = source[base + c];
      }
    }
    return out;
  }

  const api = { deinterleaveInto };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.ChannelDecode = api;
})();
