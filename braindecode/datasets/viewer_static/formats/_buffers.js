/* ============================================================
   formats/_buffers.js — per-pan channel buffer allocation shared
   by every format reader. The two helpers exist so the renderer
   can subscript per-channel typed arrays without copying, and
   we get one Float32Array allocation per pan instead of one per
   channel.
   ============================================================ */
(function () {
  'use strict';

  function empty(nChannels) {
    return Array.from({ length: nChannels }, () => new Float32Array(0));
  }

  // Returns N typed-array views over a single backing buffer.
  // Mutating any view writes through to `backing`, so the renderer
  // must treat each view as owned for the duration of the pan.
  function alloc(nChannels, nWin) {
    const backing = new Float32Array(nChannels * nWin);
    const out = new Array(nChannels);
    for (let c = 0; c < nChannels; c++) {
      out[c] = backing.subarray(c * nWin, (c + 1) * nWin);
    }
    return out;
  }

  // Bounds-clamp a requested window against [0, nSamples). Returns
  // null when the request is degenerate (past EOF or zero-width), so
  // callers can early-return their own empty/null sentinel. Otherwise
  // returns the clamped {start, end, nWin} where 0 <= start < end <= nSamples.
  // Centralises the off-by-one near EOF + negative-start pattern that
  // every format reader implemented (subtly differently) before B1.
  function clampWindow(startSample, nWin, nSamples) {
    const start = Math.max(0, startSample | 0);
    const n = Math.max(0, nWin | 0);
    if (start >= nSamples || n === 0) return null;
    const end = Math.min(start + n, nSamples);
    return { start, end, nWin: end - start };
  }

  const api = { empty, alloc, clampWindow };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.ChannelBuffers = api;
})();
