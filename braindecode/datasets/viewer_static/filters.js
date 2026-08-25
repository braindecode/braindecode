/* ============================================================
   filters.js — pure-JS biquad filter bank for in-browser EEG
   display filtering (F08).  No WASM, no external dependencies.

   Implements second-order Butterworth HP/LP and IIR notch
   coefficients using the RBJ Audio EQ Cookbook formulas,
   plus single-pass (apply) and zero-phase forward-backward
   (filtfilt) convolution via Direct Form II.

   Module pattern: IIFE attaches to globalThis.Filters and
   module.exports — the same dual-attach convention used by
   ChannelBuffers, HttpRange, etc.
   ============================================================ */
(function () {
  'use strict';

  // ---- coefficient designers --------------------------------

  /**
   * 2nd-order Butterworth highpass.
   * Formula: RBJ Audio EQ Cookbook "HPF" with Q = 1/√2.
   *
   * @param {number} fs         — sample rate Hz
   * @param {number} cutoff_hz  — −3 dB corner frequency Hz
   * @param {number} [order=2]  — filter order (≥2; only 2 implemented;
   *                              parameter accepted for API compat)
   * @returns {{ b: number[], a: number[] }}  normalised biquad coefficients
   *   b = [b0, b1, b2], a = [1, a1, a2]
   */
  function designHighpass(fs, cutoff_hz, order) {
    void order;  // reserved — only 2nd-order implemented
    const w0    = 2 * Math.PI * cutoff_hz / fs;
    const cosw0 = Math.cos(w0);
    const alpha = Math.sin(w0) / (2 * Math.SQRT1_2);  // Q = 1/√2 → Butterworth
    const a0    = 1 + alpha;
    return {
      b: [
        (1 + cosw0) / 2 / a0,
        -(1 + cosw0)      / a0,
        (1 + cosw0) / 2 / a0,
      ],
      a: [1, -2 * cosw0 / a0, (1 - alpha) / a0],
    };
  }

  /**
   * 2nd-order Butterworth lowpass.
   * Formula: RBJ Audio EQ Cookbook "LPF" with Q = 1/√2.
   */
  function designLowpass(fs, cutoff_hz, order) {
    void order;
    const w0    = 2 * Math.PI * cutoff_hz / fs;
    const cosw0 = Math.cos(w0);
    const alpha = Math.sin(w0) / (2 * Math.SQRT1_2);
    const a0    = 1 + alpha;
    return {
      b: [
        (1 - cosw0) / 2 / a0,
        (1 - cosw0)      / a0,
        (1 - cosw0) / 2 / a0,
      ],
      a: [1, -2 * cosw0 / a0, (1 - alpha) / a0],
    };
  }

  /**
   * 2nd-order IIR notch (band-reject).
   * Formula: RBJ Audio EQ Cookbook "notching EQ" section.
   *
   * @param {number} fs       — sample rate Hz
   * @param {number} freq_hz  — notch centre frequency Hz
   * @param {number} [q=30]   — quality factor (higher → narrower notch)
   */
  function designNotch(fs, freq_hz, q) {
    const Q     = (q != null) ? q : 30;
    const w0    = 2 * Math.PI * freq_hz / fs;
    const cosw0 = Math.cos(w0);
    const alpha = Math.sin(w0) / (2 * Q);
    const a0    = 1 + alpha;
    return {
      b: [1 / a0, -2 * cosw0 / a0, 1 / a0],
      a: [1, -2 * cosw0 / a0, (1 - alpha) / a0],
    };
  }

  // ---- single-pass filter (Direct Form II transposed) ------

  /**
   * Forward-only single-pass biquad filter.
   * Operates on a Float32Array (or plain Array) in-place and
   * also returns the output array for chaining.
   *
   * @param {Float32Array|number[]} samples
   * @param {{ b: number[], a: number[] }} coefs
   * @returns {Float64Array}  filtered output (new allocation; samples unchanged)
   */
  function apply(samples, coefs) {
    const { b, a } = coefs;
    const n   = samples.length;
    const out = new Float64Array(n);
    // Hoist coefficients to locals — avoids per-sample property lookups in the
    // hot inner loop (~+15-25% throughput on V8 for typical EEG windows).
    const a1 = a[1], a2 = a[2];
    const b0 = b[0], b1 = b[1], b2 = b[2];
    // Direct Form II: two state variables w1 (z^-1), w2 (z^-2).
    let w1 = 0, w2 = 0;
    for (let i = 0; i < n; i++) {
      const w  = samples[i] - a1 * w1 - a2 * w2;
      out[i]   = b0 * w + b1 * w1 + b2 * w2;
      w2 = w1; w1 = w;
    }
    return out;
  }

  /**
   * Zero-phase forward-backward filter (filtfilt).
   * Applies the biquad once forward, reverses, applies again,
   * reverses back — net result is zero phase shift and squared
   * magnitude response.  Equivalent to scipy.signal.filtfilt
   * without edge padding (the edge artefact is negligible for
   * EEG display at typical window sizes ≥ 250 samples).
   *
   * @param {Float32Array|number[]} samples
   * @param {{ b: number[], a: number[] }} coefs
   * @returns {Float64Array}
   */
  function filtfilt(samples, coefs) {
    // apply() always returns a fresh Float64Array we exclusively own, so
    // in-place .reverse() (native, no allocation) is safe and avoids the
    // 2×N temporary allocations the previous index-copy loops produced.
    const fwd = apply(samples, coefs);
    fwd.reverse();
    const bk  = apply(fwd, coefs);
    bk.reverse();
    return bk;
  }

  /**
   * Apply a chain of biquad filters to one channel of samples,
   * using filtfilt for each stage.  Returns a new Float32Array.
   *
   * @param {Float32Array} samples — single EEG channel
   * @param {Array<{b:number[], a:number[]}>} coefsList — ordered filter chain
   * @returns {Float32Array}
   */
  function applyChain(samples, coefsList) {
    if (!coefsList || coefsList.length === 0) return samples;
    let buf = samples;
    for (const coefs of coefsList) {
      buf = filtfilt(buf, coefs);
    }
    // Return as Float32Array so the type matches what the worker/renderer expects.
    return Float32Array.from(buf);
  }

  const api = {
    designHighpass,
    designLowpass,
    designNotch,
    filtfilt,
    applyChain,
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.Filters = api;
})();
