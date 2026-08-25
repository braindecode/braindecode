/* ============================================================
   formats/snirf.js — read SNIRF (Shared Near Infrared Spectroscopy
   File Format) for the eegdash-viewer. SNIRF is a pure HDF5 file
   (no MAT v7.3 wrapper) defined by the Society for fNIRS at
   https://github.com/fNIRS/snirf — we read the canonical layout:

     /formatVersion              STRING dataset (e.g. "1.0")
     /nirs                       GROUP (or /nirs1, /nirs2, … for
                                 multi-recording files; we read the
                                 first one only in v1)
     /nirs/data1                 GROUP
     /nirs/data1/dataTimeSeries  float dataset shape [nSamples, nCh]
     /nirs/data1/time            float dataset shape [nSamples]
     /nirs/data1/measurementList1..N  GROUPs per channel
     /nirs/probe/wavelengths     float dataset (optional)
     /nirs/stim1..N              GROUPs with onset/duration/value
                                 (optional — surfaced as annotation_events)

   We use jsfive (already vendored at formats/_jsfive.js for MAT v7.3)
   to walk the HDF5. Unlike MAT v7.3, SNIRF has NO 512-byte MAT stub —
   the HDF5 magic is at offset 0.

   What we DON'T handle (deliberately):
     - Multi-recording files (/nirs1, /nirs2, ...) — read the first only
     - Variable-rate time arrays — assert roughly uniform sampling
     - Compressed datasets that jsfive doesn't transparently handle
     - Aux channels (/nirs/aux1..N) — out of scope for v1 trace viewer
   ============================================================ */
(function () {
  'use strict';

  const api = {};

  // jsfive resolves differently in Node (CJS via npm) and browser
  // (vendored IIFE attaches globalThis.hdf5). Same helper as _mat73.js.
  function getJsfive() {
    if (typeof globalThis !== 'undefined' && globalThis.hdf5) return globalThis.hdf5;
    if (typeof require !== 'undefined') {
      try { return require('jsfive'); } catch (_) { /* fall through */ }
    }
    throw new Error(
      'jsfive not available: include formats/_jsfive.js before ' +
      'formats/snirf.js in the browser, or `npm install jsfive` ' +
      'for the Node tests.'
    );
  }

  // SNIRF starts with the HDF5 magic at byte 0 (unlike MAT v7.3 which
  // has the 512-byte MAT stub first).
  function isHdf5AtZero(buf) {
    const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
    if (u8.length < 8) return false;
    return u8[0] === 0x89 && u8[1] === 0x48 && u8[2] === 0x44 &&
           u8[3] === 0x46 && u8[4] === 0x0d && u8[5] === 0x0a &&
           u8[6] === 0x1a && u8[7] === 0x0a;
  }

  // Decode an HDF5 STRING dataset value into a regular JS string.
  // jsfive returns either a 1-element array of strings, a plain string,
  // or a typed byte array depending on the variant — handle each.
  function readStringDataset(ds) {
    if (!ds) return null;
    const v = ds.value;
    if (v == null) return null;
    if (typeof v === 'string') return v;
    if (Array.isArray(v) && v.length === 1 && typeof v[0] === 'string') return v[0];
    if (Array.isArray(v) && v.length === 1 && v[0] && typeof v[0] === 'object' && typeof v[0].toString === 'function') {
      return String(v[0]);
    }
    if (v.length && typeof v[0] === 'number') {
      // Byte array — drop trailing NUL.
      let s = '';
      for (let i = 0; i < v.length; i++) {
        if (v[i] === 0) break;
        s += String.fromCharCode(v[i]);
      }
      return s;
    }
    return null;
  }

  // Pick the first /nirs* group at the root. Per the spec the typical
  // names are /nirs (single recording) or /nirs1, /nirs2 (multi).
  function pickNirsGroup(root) {
    if (root.keys.includes('nirs')) return root.get('nirs');
    for (const k of root.keys) {
      if (/^nirs\d+$/.test(k)) return root.get(k);
    }
    throw new Error('SNIRF: no /nirs (or /nirs1, /nirs2, …) group found');
  }

  // Pick the first /data* group inside /nirs. Same multi-recording
  // shape — usually /nirs/data1.
  function pickDataGroup(nirsGroup) {
    if (nirsGroup.keys.includes('data1')) return nirsGroup.get('data1');
    for (const k of nirsGroup.keys) {
      if (/^data\d+$/.test(k)) return nirsGroup.get(k);
    }
    if (nirsGroup.keys.includes('data')) return nirsGroup.get('data');
    throw new Error('SNIRF: no /nirs/data1 group found');
  }

  // Extract every /nirs/stim* group as { onset, duration, label } events.
  // Per spec, each stim group has a `data` 2-D dataset (Nx3 columns =
  // onset, duration, value) and a `name` string. We surface `name` as
  // the event label so all entries from one stim group share a label.
  function extractStimEvents(nirsGroup) {
    const events = [];
    for (const k of nirsGroup.keys) {
      if (!/^stim\d+$/.test(k)) continue;
      const stim = nirsGroup.get(k);
      if (!stim || !stim.keys) continue;
      let label = k;
      if (stim.keys.includes('name')) {
        const got = readStringDataset(stim.get('name'));
        if (got) label = got;
      }
      if (!stim.keys.includes('data')) continue;
      const ds = stim.get('data');
      const v = ds.value;
      const shape = ds.shape;  // expected [N, 3]
      if (!v || !shape || shape.length !== 2 || shape[1] < 2) continue;
      const n = shape[0];
      const cols = shape[1];
      // jsfive returns row-major; row i columns 0..cols-1 live at
      // v[i*cols..i*cols+cols-1] when v is a flat typed array. For
      // chunked datasets jsfive may return a nested array — normalise.
      const flat = (v.length === n * cols) ? v : v.flat();
      for (let i = 0; i < n; i++) {
        events.push({
          onset: Number(flat[i * cols + 0]),
          duration: Number(flat[i * cols + 1]),
          label,
        });
      }
    }
    events.sort((a, b) => a.onset - b.onset);
    return events;
  }

  function readScalar(group, name) {
    if (!group.keys || !group.keys.includes(name)) return null;
    const ds = group.get(name);
    if (!ds) return null;
    const v = ds.value;
    if (v == null) return null;
    if (typeof v === 'number') return v;
    if (typeof v.length === 'number' && v.length > 0) return Number(v[0]);
    return null;
  }

  function buildChannelLabels(nirs, data, nChannels) {
    const labels = new Array(nChannels);
    // Read probe wavelengths once for the suffix.
    let wavelengths = null;
    if (nirs.keys.includes('probe')) {
      const probe = nirs.get('probe');
      if (probe.keys && probe.keys.includes('wavelengths')) {
        const ds = probe.get('wavelengths');
        wavelengths = ds.value;
      }
    }
    for (let i = 0; i < nChannels; i++) {
      const key = `measurementList${i + 1}`;
      if (!data.keys.includes(key)) { labels[i] = `Ch${i + 1}`; continue; }
      const ml = data.get(key);
      const src = readScalar(ml, 'sourceIndex');
      const det = readScalar(ml, 'detectorIndex');
      const wlIdx = readScalar(ml, 'wavelengthIndex');
      let wlSuffix = '';
      if (wavelengths && wlIdx != null && wavelengths[wlIdx - 1] != null) {
        wlSuffix = '-' + Math.round(Number(wavelengths[wlIdx - 1])) + 'nm';
      }
      if (src != null && det != null) {
        labels[i] = `S${src}D${det}${wlSuffix}`;
      } else {
        labels[i] = `Ch${i + 1}`;
      }
    }
    return labels;
  }

  function normaliseToFloat32(value, nSamples, nChannels) {
    const expected = nSamples * nChannels;
    if (value && typeof value.length === 'number' && value.length === expected) {
      if (value instanceof Float32Array) return value;
      return Float32Array.from(value);
    }
    // Nested-array shape — flatten.
    const out = new Float32Array(expected);
    let i = 0;
    for (let s = 0; s < nSamples; s++) {
      const row = value[s];
      for (let c = 0; c < nChannels; c++) {
        out[i++] = Number(row[c]);
      }
    }
    return out;
  }

  /**
   * Open a SNIRF file for windowed reading.
   *
   * @param {{ eeg_url: string, [k: string]: any }} meta
   * @returns {Promise<object>} reader matching the cross-format contract:
   *   { n_channels, sampling_frequency, duration_s, n_samples,
   *     channel_labels, bytes_per_sample, recording_start_iso,
   *     annotation_events, readWindow(start, n) }
   */
  api.open = async function (meta) {
    const url = meta && (meta.eeg_url || meta.url);
    if (!url) throw new Error('snirf.open: meta.eeg_url is required');
    const HttpRange = globalThis.HttpRange;
    if (!HttpRange) throw new Error('snirf.open: globalThis.HttpRange missing');

    const buf = await HttpRange.fetchBuffer(url);
    if (!isHdf5AtZero(buf)) {
      throw new Error('SNIRF: file is not a valid HDF5 (magic mismatch at byte 0)');
    }
    const jsfive = getJsfive();
    const ab = buf instanceof ArrayBuffer
      ? buf
      : buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    // jsfive crashes with assertion errors on some valid HDF5 features
    // it doesn't fully support (e.g. fractal heap with rare extension
    // types). Wrap construction + initial reads to surface a precise
    // reader-side message instead of "thing is not a function" from the
    // jsfive internals. Observed on ds007463 nirs file.
    let file;
    try {
      file = new jsfive.File(ab);
    } catch (e) {
      throw new Error(
        `SNIRF: jsfive failed to read this HDF5 file — likely a feature ` +
        `our HDF5 library doesn't yet support (compression filter, ` +
        `fractal heap extension, …). Original: ${e.message}`,
      );
    }

    let nirs, data;
    try {
      nirs = pickNirsGroup(file);
      data = pickDataGroup(nirs);
    } catch (e) {
      // Same wrap for jsfive errors that fire during group traversal
      // rather than file construction (observed: assert failures inside
      // FractalHeap when walking root groups).
      if (/is not a function|cannot read/i.test(e.message)) {
        throw new Error(
          `SNIRF: jsfive crashed while walking the HDF5 group tree — ` +
          `the file uses an HDF5 feature our library doesn't fully ` +
          `support. Original: ${e.message}`,
        );
      }
      throw e;
    }

    if (!data.keys.includes('dataTimeSeries')) {
      throw new Error('SNIRF: /nirs/data1/dataTimeSeries missing');
    }
    if (!data.keys.includes('time')) {
      throw new Error('SNIRF: /nirs/data1/time missing');
    }
    const dts = data.get('dataTimeSeries');
    const timeDs = data.get('time');
    const shape = dts.shape;  // [nSamples, nChannels]
    if (!shape || shape.length !== 2) {
      throw new Error(
        'SNIRF: dataTimeSeries must be 2-D, got [' +
        (shape ? shape.join(',') : '?') + ']'
      );
    }
    const nSamples = shape[0];
    const nChannels = shape[1];
    if (nSamples <= 0 || nChannels <= 0) {
      throw new Error('SNIRF: empty dataTimeSeries shape [' + shape.join(',') + ']');
    }

    // Derive sampling frequency from the time array. Use the first two
    // samples; if the array isn't uniformly sampled, warn but trust the
    // mean spacing (the viewer assumes uniform fs).
    const t = timeDs.value;
    if (!t || t.length < 2) {
      throw new Error('SNIRF: /nirs/data1/time has fewer than 2 samples');
    }
    const dt = Number(t[1]) - Number(t[0]);
    if (!(dt > 0)) throw new Error('SNIRF: non-positive time delta ' + dt);
    const fs = 1 / dt;
    const dtMean = (Number(t[t.length - 1]) - Number(t[0])) / (t.length - 1);
    if (Math.abs(dtMean - dt) / dt > 0.05) {
      console.warn(
        `SNIRF: time array is non-uniform ` +
        `(dt[0]=${dt.toExponential(3)}, dt_mean=${dtMean.toExponential(3)}); ` +
        `v1 assumes uniform fs.`
      );
    }

    // Channel labels: build "S<src>D<det>-<wavelength_nm>" from
    // /nirs/data1/measurementList<i>/{sourceIndex,detectorIndex,wavelengthIndex}
    // and /nirs/probe/wavelengths. Falls back to "Ch1..ChN" if any of
    // those datasets are missing.
    const channelLabels = buildChannelLabels(nirs, data, nChannels);

    // Convert dataTimeSeries to a flat Float32Array up front so
    // readWindow can index it directly without re-promoting per call.
    // jsfive returns nested arrays for some chunked datasets — normalise.
    const flat = normaliseToFloat32(dts.value, nSamples, nChannels);
    if (flat.length !== nSamples * nChannels) {
      throw new Error(
        `SNIRF: dataTimeSeries length ${flat.length} != ` +
        `nSamples(${nSamples}) * nChannels(${nChannels})`
      );
    }

    // Optional /nirs/stim* groups become annotation_events.
    const annotation_events = extractStimEvents(nirs);

    return {
      n_channels: nChannels,
      sampling_frequency: fs,
      duration_s: nSamples / fs,
      n_samples: nSamples,
      channel_labels: channelLabels,
      // SNIRF dataTimeSeries is typically float64; we display Float32
      // but quote the source width so the UI can show the on-disk dtype.
      bytes_per_sample: 8,
      recording_start_iso: null,
      annotation_events,
      readWindow: async (startSample, nWin) => {
        const win = globalThis.ChannelBuffers.clampWindow(startSample, nWin, nSamples);
        if (!win) return globalThis.ChannelBuffers.empty(nChannels);
        const { start, end } = win;
        const out = globalThis.ChannelBuffers.alloc(nChannels, end - start);
        // dataTimeSeries is row-major [nSamples, nChannels]: sample s
        // channel c lives at flat[s * nChannels + c].
        for (let s = start; s < end; s++) {
          const base = s * nChannels;
          for (let c = 0; c < nChannels; c++) {
            out[c][s - start] = flat[base + c];
          }
        }
        return out;
      },
    };
  };

  // Re-exposed for tests.
  api._isHdf5AtZero = isHdf5AtZero;
  api._extractStimEvents = extractStimEvents;

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.SnirfReader = api;
})();
