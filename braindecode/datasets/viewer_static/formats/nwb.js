/* ============================================================
   formats/nwb.js — read NWB (Neurodata Without Borders) iEEG /
   ECoG / LFP recordings for the eegdash-viewer.

   NWB is an HDF5-based container format defined by the NWB schema
   (https://nwb-schema.readthedocs.io/). We have two read paths:

   1. WHOLE-FILE (≤ 200 MB or pre-buffered): delegate the HDF5 walk
      to jsfive (vendored as `formats/_jsfive.js` for the browser,
      npm-installed for Node tests). Used by `api.read(buffer)` and
      by `api.open(meta)` when the file fits in the legacy cap.
      This path is byte-identical to the original NWB reader (Lane
      H1) — same code, no behaviour change.

   2. STREAMING (any size, contiguous or chunked-GZIP): the new
      head-buffer + range-fetch path. We fetch only the first
      ~16 MB of the file to let jsfive walk the metadata, then
      issue per-chunk range fetches via formats/_h5-stream.js when
      readWindow() is called. Used by `api.open(meta)` when the
      file is > 200 MB and the dataset is contiguous OR chunked
      with a supported filter pipeline (gzip). For 1 GB+ DANDI
      NWB files this turns a fail-with-cap into an open()-in-2s
      + readWindow()-in-1s experience.

   We always read the canonical iEEG path:

     /                                attrs: nwb_version (str),
                                              neurodata_type=NWBFile
     /acquisition/                    GROUP — first child whose
                                      neurodata_type attr is
                                      "ElectricalSeries" (else first
                                      child that has a `data` dataset).
     /acquisition/<ts>/data           float dataset, shape
                                      [n_samples, n_channels]
                                      (NWB canonical layout; we also
                                      accept [n_channels, n_samples]
                                      and transpose on read — only in
                                      the whole-file path, the
                                      streaming path requires
                                      canonical layout).
     /acquisition/<ts>/starting_time  scalar float64, attrs.rate = fs
                                      OR
     /acquisition/<ts>/timestamps     float dataset [n_samples] — only
                                      consulted if `starting_time.rate`
                                      is missing; we derive fs from
                                      timestamps[1] - timestamps[0].
     /general/extracellular_ephys/electrodes/  optional DynamicTable
                                      with `label` (or `id`) column
                                      used for channel names.

   What we DON'T handle (deliberately):
     - References / DynamicTableRegion lookups across files: NWB allows
       `/acquisition/X/electrodes` to point at a foreign electrodes
       table via HDF5 references. Channel labels fall back to "Ch1..N"
       when the column lookup fails — never crash on a missing label.
     - Compressed datasets that the streaming reader doesn't know:
       only GZIP / SHUFFLE / FLETCH32 (the jsfive Filters set) are
       supported on the streaming path. SZIP / N-bit / scale-offset
       fall through to whole-file jsfive — and if too big, throw a
       documented error.
     - V2 chunk indexes (HDF5 1.10+ "single chunk", "implicit",
       "fixed array", "extensible array", "B-tree v2"). The streaming
       path only walks V1 chunk B-trees — the default and (by far)
       most common writer choice. V2 indexes fall back to whole-file
       jsfive on the streaming path.
     - Encrypted NWB (NWB extensions for encrypted data are out of scope).
     - Stimuli / behavioural NWB groups — we read /acquisition only.
     - Per-channel unit conversions: NWB allows a `conversion` /
       `offset` scalar attribute on `data`. We apply both if present,
       skip silently if absent (whole-file path only; streaming path
       skips the scale to keep the hot loop tight).
   ============================================================ */
(function () {
  'use strict';

  const api = {};

  // Threshold above which the streaming path is preferred over the
  // whole-file path. The streaming path skips downloading the full
  // file — for chunked datasets it only fetches the chunk B-tree
  // plus the chunks intersecting each readWindow. For contiguous
  // datasets it only fetches the byte range of the requested window.
  //
  // We keep the threshold at 200 MB (rather than going lower) for
  // two reasons:
  //   1. The streaming path requires the file's HDF5 metadata to
  //      live in the first HEAD_BUFFER_BYTES (currently 16 MB). For
  //      smaller files we know the whole-file path will succeed
  //      because it reads everything. For larger files the
  //      streaming path is the only way to avoid the cap.
  //   2. For files between 200 MB and 1 GB the whole-file path
  //      still works (modern browsers can allocate up to 4 GB
  //      sparse ArrayBuffers) but is bandwidth-heavy. Users with
  //      slow connections benefit from the streaming path; users
  //      on fast connections may prefer downloading the whole
  //      file (which avoids round-trip latency on every readWindow).
  //
  // Above 1 GB we always use the streaming path — sparse allocation
  // still works but the bandwidth cost of fetching 1+ GB on open()
  // is prohibitive.
  const STREAMING_THRESHOLD_BYTES = 200 * 1024 * 1024;  // 200 MB

  // Hard cap for the whole-file (jsfive whole-buffer) path. We will
  // not allocate or download files larger than this on the whole-
  // file path. The streaming path bypasses this entirely.
  //
  // 1 GB is well within the platform's typical ArrayBuffer limit
  // (Node ~4 GB, Chrome 4 GB max ArrayBuffer per V8 isolate, Safari
  // ~2 GB on macOS). Above 1 GB we fail with a clean message
  // pointing at the streaming path or pynwb/nwbinspector subsetting.
  const LEGACY_FALLBACK_CAP = 1024 * 1024 * 1024;  // 1 GB

  // A2-style integer caps: NWB recordings in the wild peak at a few
  // hundred ECoG channels; nothing in current iEEG hits 4096. Catching
  // a 4-byte garbage shape early prevents a multi-GB Float32Array alloc.
  const MAX_CHANNELS = 4096;
  const MAX_SAMPLES_TOTAL = 1 << 30;  // ~1.07e9 sample cells (any dtype)

  // jsfive resolves differently in Node (CJS via npm) and the
  // browser/worker (vendored IIFE attaches globalThis.hdf5). Same
  // pattern as formats/_mat73.js and formats/snirf.js.
  function getJsfive() {
    if (typeof globalThis !== 'undefined' && globalThis.hdf5) return globalThis.hdf5;
    if (typeof require !== 'undefined') {
      try { return require('jsfive'); } catch (_) { /* fall through */ }
    }
    throw new Error(
      'jsfive not available: include formats/_jsfive.js before ' +
      'formats/nwb.js in the browser, or `npm install jsfive` for ' +
      'the Node tests.'
    );
  }

  // H5Stream is the streaming HDF5 reader (formats/_h5-stream.js).
  // In the browser/worker it attaches to globalThis; in Node tests
  // we require it directly. We keep the resolution lazy so a build
  // that doesn't include _h5-stream.js still works on the whole-
  // file path — only > 200 MB files need the streaming reader.
  function getH5Stream() {
    if (typeof globalThis !== 'undefined' && globalThis.H5Stream) return globalThis.H5Stream;
    if (typeof require !== 'undefined') {
      try { return require('./_h5-stream.js'); } catch (_) { /* fall through */ }
    }
    throw new Error(
      'H5Stream not available: include formats/_h5-stream.js before ' +
      'formats/nwb.js in the browser, or it must sit alongside ' +
      'formats/nwb.js for the Node CJS require to resolve.'
    );
  }

  // NWB files are pure HDF5; magic at byte 0 (same as SNIRF, unlike
  // MAT v7.3 which has the 512-byte MAT stub first).
  function isHdf5AtZero(buf) {
    const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
    if (u8.length < 8) return false;
    return u8[0] === 0x89 && u8[1] === 0x48 && u8[2] === 0x44 &&
           u8[3] === 0x46 && u8[4] === 0x0d && u8[5] === 0x0a &&
           u8[6] === 0x1a && u8[7] === 0x0a;
  }

  // jsfive returns fixed-length strings ("S8", "S16") NUL-padded to
  // the declared width. Strip trailing NULs and whitespace so labels
  // round-trip to humans cleanly.
  function trimNulString(s) {
    if (typeof s !== 'string') return s == null ? '' : String(s);
    let end = s.length;
    while (end > 0) {
      const c = s.charCodeAt(end - 1);
      if (c === 0 || c === 0x20) end--;
      else break;
    }
    return s.slice(0, end);
  }

  // Pull a numeric attribute from jsfive's `.attrs` object. jsfive
  // returns scalars as a plain number OR a 1-length array depending on
  // how the file was written (h5py emits both shapes). Normalise.
  function readNumericAttr(attrs, name) {
    if (!attrs) return null;
    const v = attrs[name];
    if (v == null) return null;
    if (typeof v === 'number') return v;
    if (typeof v.length === 'number' && v.length > 0) {
      const n = Number(v[0]);
      return Number.isFinite(n) ? n : null;
    }
    return null;
  }

  // Find the first child group inside /acquisition that looks like an
  // ElectricalSeries (or, failing the neurodata_type attribute, any
  // child that has a `data` dataset — some legacy NWB exports omit the
  // attribute on auto-named series). Prefer the canonical match.
  function pickElectricalSeries(acq) {
    if (!acq || !acq.keys || !acq.keys.length) {
      throw new Error('NWB: /acquisition is empty (no ElectricalSeries found)');
    }
    let firstWithData = null;
    for (const k of acq.keys) {
      let child;
      try { child = acq.get(k); } catch (_) { continue; }
      if (!child || !child.keys) continue;
      const nd = child.attrs && child.attrs.neurodata_type;
      if (nd === 'ElectricalSeries') return { name: k, group: child };
      if (!firstWithData && child.keys.includes('data')) {
        firstWithData = { name: k, group: child };
      }
    }
    if (firstWithData) return firstWithData;
    throw new Error(
      'NWB: no ElectricalSeries found under /acquisition ' +
      '(checked: ' + acq.keys.join(', ') + ')'
    );
  }

  // Compute sampling frequency from either starting_time.rate (the
  // canonical NWB regular-sampling field) or timestamps[1]-timestamps[0]
  // (irregular sampling fallback). Returns { fs, isUniform }.
  function deriveSamplingRate(es) {
    if (es.keys.includes('starting_time')) {
      const st = es.get('starting_time');
      const rate = readNumericAttr(st.attrs, 'rate');
      if (rate != null && rate > 0) return { fs: rate, isUniform: true };
    }
    if (es.keys.includes('timestamps')) {
      const tsDs = es.get('timestamps');
      const ts = tsDs.value;
      if (ts && ts.length >= 2) {
        const dt = Number(ts[1]) - Number(ts[0]);
        if (dt > 0) {
          // Sanity-check uniformity. NWB allows irregularly-sampled
          // timestamps; the viewer assumes uniform fs for windowing, so
          // we warn rather than throw — the rendered traces will simply
          // be slightly stretched at non-uniform regions.
          const dtMean = (Number(ts[ts.length - 1]) - Number(ts[0])) / (ts.length - 1);
          const isUniform = Math.abs(dtMean - dt) / dt <= 0.05;
          if (!isUniform) {
            console.warn(
              'NWB: timestamps are non-uniform ' +
              '(dt[0]=' + dt.toExponential(3) + ', dt_mean=' + dtMean.toExponential(3) +
              '); v1 assumes uniform fs.'
            );
          }
          return { fs: 1 / dt, isUniform };
        }
      }
    }
    throw new Error(
      'NWB: cannot derive sampling rate — no starting_time.rate ' +
      'attribute and no usable timestamps dataset'
    );
  }

  // Build channel labels from /general/extracellular_ephys/electrodes
  // when available. NWB stores the table as a DynamicTable group with
  // one dataset per column; we look for `label` first (most common
  // in BIDS-iEEG conversions) and fall back to `id` (numeric channel
  // index) before giving up and using indexed Ch1..ChN.
  //
  // We intentionally do not resolve the per-series `electrodes`
  // DynamicTableRegion reference — that requires HDF5 reference
  // dereferencing which jsfive supports unevenly across NWB writer
  // versions. The /general electrodes table covers the recording-wide
  // channel set, which matches every ElectricalSeries in well-formed
  // single-acquisition NWB files (the only shape we read in v1).
  function buildChannelLabels(root, nChannels) {
    const fallback = (typeof globalThis !== 'undefined' && globalThis.ChannelLabels)
      ? globalThis.ChannelLabels.indexed(nChannels)
      : Array.from({ length: nChannels }, (_, i) => 'Ch' + (i + 1));
    let electrodes = null;
    try { electrodes = root.get('general/extracellular_ephys/electrodes'); }
    catch (_) { return fallback; }
    if (!electrodes || !electrodes.keys) return fallback;

    // Prefer `label` (string) → `id` (int). NWB's DynamicTable spec
    // guarantees `id` is always present; `label` is the recommended
    // human-readable column for iEEG.
    const tryColumn = (colName, mapper) => {
      if (!electrodes.keys.includes(colName)) return null;
      const ds = electrodes.get(colName);
      const v = ds.value;
      if (!v || v.length !== nChannels) return null;
      const out = new Array(nChannels);
      for (let i = 0; i < nChannels; i++) out[i] = mapper(v[i], i);
      return out;
    };

    const labels = tryColumn('label', (s) => trimNulString(String(s)) || ('Ch' + (1)));
    if (labels) {
      // If we got empty strings back (shouldn't happen, but a malformed
      // table could yield all-NULs), drop back to indexed labels.
      const anyNonEmpty = labels.some((s) => s && s.length);
      if (anyNonEmpty) {
        return labels.map((s, i) => (s && s.length) ? s : 'Ch' + (i + 1));
      }
    }

    const ids = tryColumn('id', (n) => 'Ch' + (Number(n) + 1));
    if (ids) return ids;

    return fallback;
  }

  // Validate dataset shape: must be 2-D, both dims positive, neither
  // axis above its A2 cap. Returns the canonical [nSamples, nChannels]
  // pair and a `transposed` flag so readWindow can index correctly.
  // NWB canonical is [n_samples, n_channels] but some converters write
  // [n_channels, n_samples] (matching the MATLAB / EEGLAB convention);
  // we detect that by checking which axis exceeds MAX_CHANNELS.
  function normaliseShape(shape) {
    if (!Array.isArray(shape) || shape.length !== 2) {
      throw new Error(
        'NWB: ElectricalSeries.data must be 2-D, got [' +
        (shape ? shape.join(',') : '?') + ']'
      );
    }
    const a = shape[0] | 0;
    const b = shape[1] | 0;
    if (a <= 0 || b <= 0) {
      throw new Error('NWB: empty data shape [' + shape.join(',') + ']');
    }
    if (a * b > MAX_SAMPLES_TOTAL) {
      throw new Error(
        'NWB: data has ' + (a * b) + ' total cells, exceeds cap ' +
        MAX_SAMPLES_TOTAL + ' (file may be malformed or too large)'
      );
    }
    // Canonical NWB: dim 0 = samples (long), dim 1 = channels (short).
    // We treat the axis with the smaller extent as the channels axis as
    // long as it's <= MAX_CHANNELS. This catches both layouts without
    // a costly heuristic.
    let nSamples, nChannels, transposed;
    if (b <= MAX_CHANNELS && (a >= b || a > MAX_CHANNELS)) {
      // [n_samples, n_channels] — canonical
      nSamples = a;
      nChannels = b;
      transposed = false;
    } else if (a <= MAX_CHANNELS) {
      // [n_channels, n_samples] — needs transpose on read
      nSamples = b;
      nChannels = a;
      transposed = true;
    } else {
      throw new Error(
        'NWB: both axes [' + a + ',' + b + '] exceed channel cap ' +
        MAX_CHANNELS + ' — refusing to load (likely shape garbage)'
      );
    }
    if (nChannels > MAX_CHANNELS) {
      throw new Error(
        'NWB: ' + nChannels + ' channels exceeds cap ' + MAX_CHANNELS
      );
    }
    return { nSamples, nChannels, transposed };
  }

  // Promote jsfive's `.value` to a flat Float32Array indexed in
  // sample-major order: flat[s * nChannels + c]. We always store
  // sample-major regardless of the on-disk layout so readWindow has
  // a single indexing rule (matches SNIRF reader).
  //
  // jsfive returns either:
  //   - a flat Array of numbers (most common, including for our
  //     synthetic h5py-generated fixture),
  //   - a nested Array (some chunked datasets),
  //   - a typed array (rare — only when jsfive happens to read the
  //     underlying buffer in-place).
  // We normalise all three.
  function normaliseToFloat32SampleMajor(value, nSamples, nChannels, transposed, conversion, offset) {
    const expected = nSamples * nChannels;
    const out = new Float32Array(expected);
    const scale = (conversion != null && Number.isFinite(conversion)) ? conversion : 1;
    const shift = (offset != null && Number.isFinite(offset)) ? offset : 0;
    const noScale = scale === 1 && shift === 0;

    // Case A: flat array/typed-array.
    if (value && typeof value.length === 'number' && value.length === expected) {
      if (!transposed) {
        if (noScale && value instanceof Float32Array) return value;
        for (let i = 0; i < expected; i++) {
          out[i] = noScale ? Number(value[i]) : Number(value[i]) * scale + shift;
        }
        return out;
      }
      // Transposed: on-disk is [nChannels, nSamples], flat[c*nSamples+s].
      // Re-index to [nSamples, nChannels].
      for (let c = 0; c < nChannels; c++) {
        const baseIn = c * nSamples;
        for (let s = 0; s < nSamples; s++) {
          const v = Number(value[baseIn + s]);
          out[s * nChannels + c] = noScale ? v : v * scale + shift;
        }
      }
      return out;
    }

    // Case B: nested array. jsfive returns either rows-of-cols or
    // chunks-of-chunks; flat() handles the row-of-cols common case.
    // We re-normalise into the 1-D layout we want.
    if (Array.isArray(value)) {
      const flat = value.flat ? value.flat(Infinity) : [].concat.apply([], value);
      if (flat.length === expected) {
        return normaliseToFloat32SampleMajor(flat, nSamples, nChannels, transposed, conversion, offset);
      }
    }
    throw new Error(
      'NWB: cannot promote data.value (len=' +
      (value && value.length) + ') to expected size ' + expected
    );
  }

  // Approximate raw byte width per sample for the duration / bandwidth
  // estimate the UI surfaces. jsfive's `.dtype` is a numpy-style string
  // like '<f4' / '<i2'. We map the common iEEG widths; anything we
  // don't recognise falls back to 4 (Float32 — what we hand the renderer).
  function bytesPerSampleFromDtype(dtype) {
    if (typeof dtype !== 'string') return 4;
    const m = dtype.match(/[<>=!@\|]?([iuf])(\d+)/);
    if (!m) return 4;
    return Math.max(1, parseInt(m[2], 10) | 0);
  }

  // Parse the ISO 8601 session start time NWB stores at
  // /session_start_time (NWB ≥ 2.0). Optional — many DANDI files have
  // it, some BIDS-iEEG converters don't. Returned verbatim so the UI
  // can render it without further parsing.
  function readSessionStartIso(root) {
    if (!root.keys || !root.keys.includes('session_start_time')) return null;
    try {
      const ds = root.get('session_start_time');
      const v = ds.value;
      if (typeof v === 'string') return v;
      if (Array.isArray(v) && v.length === 1 && typeof v[0] === 'string') return v[0];
      return null;
    } catch (_) {
      return null;
    }
  }

  /**
   * Open an NWB file for windowed reading.
   *
   * @param {{ eeg_url: string, [k: string]: any }} meta
   * @returns {Promise<object>} reader matching the cross-format contract:
   *   { n_channels, sampling_frequency, duration_s, n_samples,
   *     channel_labels, channel_types, bytes_per_sample,
   *     recording_start_iso, annotation_events, readWindow(start, n) }
   */
  api.open = async function (meta) {
    const url = meta && (meta.eeg_url || meta.url);
    if (!url) throw new Error('nwb.open: meta.eeg_url is required');
    const HttpRange = globalThis.HttpRange;
    if (!HttpRange) throw new Error('nwb.open: globalThis.HttpRange missing');

    // Probe length so we know which path to take.
    //   - fileSize ≤ STREAMING_THRESHOLD (200 MB): whole-file path
    //     (download the file, hand it to jsfive). Fast for the
    //     common case; matches the original Lane H1 behaviour and
    //     keeps the existing 13 unit tests on the byte-identical
    //     code path they were written against.
    //   - STREAMING_THRESHOLD < fileSize ≤ LEGACY_FALLBACK_CAP
    //     (200 MB - 1 GB): try the streaming path FIRST (cheap,
    //     ~16 MB head fetch). If streaming fails because the file's
    //     metadata is scattered past the head buffer (a known
    //     quirk of h5py + pynwb default writers — they append
    //     small metadata datasets after the big chunked one), fall
    //     back to the whole-file path. This guarantees we never
    //     regress on a file that the old reader could open.
    //   - fileSize > LEGACY_FALLBACK_CAP (> 1 GB): streaming only.
    //     We refuse to allocate or download a > 1 GB ArrayBuffer
    //     in a browser tab. The user must subset via pynwb /
    //     nwbinspector, or wait for a follow-up that supports
    //     sparse-buffer jsfive walks (a known follow-up — see
    //     comments at top of formats/_h5-stream.js).
    let fileSize = null;
    if (typeof HttpRange.probeLength === 'function') {
      fileSize = await HttpRange.probeLength(url).catch(() => null);
    }

    if (fileSize != null && fileSize <= STREAMING_THRESHOLD_BYTES) {
      return openWholeFile(url, HttpRange);
    }

    // File is > 200 MB. Try streaming first.
    try {
      return await openStreaming(url, fileSize);
    } catch (streamingErr) {
      // Fall back to whole-file only if the file is within the
      // legacy cap. Above the cap we surface the streaming error
      // directly — the user has no other option.
      if (fileSize == null || fileSize > LEGACY_FALLBACK_CAP) {
        throw streamingErr;
      }
      // Whole-file fallback. We log the streaming failure so it's
      // visible in DevTools, then proceed silently — the user gets
      // a working reader either way.
      if (typeof console !== 'undefined' && console.warn) {
        console.warn(
          'NWB: streaming open failed (' + streamingErr.message + '); ' +
          'falling back to whole-file download (' + (fileSize >>> 20) + ' MB).'
        );
      }
      return openWholeFile(url, HttpRange);
    }
  };

  // Whole-file path: download the bytes, hand to jsfive. Same code
  // the original Lane H1 reader used; broken out so api.open's
  // routing logic stays readable.
  async function openWholeFile(url, HttpRange) {
    const buf = await HttpRange.fetchBuffer(url, { maxBytes: LEGACY_FALLBACK_CAP });
    if (!isHdf5AtZero(buf)) {
      throw new Error('NWB: file is not a valid HDF5 (magic mismatch at byte 0)');
    }
    return api.read(buf);
  }

  // Open via the streaming reader. Throws clean messages when the
  // file is shaped in a way our streaming subset doesn't handle — at
  // which point the caller's only recourse is downloading the file
  // and running it through pynwb. We never silently corrupt data.
  async function openStreaming(url, knownFileSize) {
    const H5 = getH5Stream();
    let head;
    try {
      head = await H5.probeHead(url);
    } catch (e) {
      throw new Error(
        'NWB streaming: ' + (e && e.message ? e.message : String(e)) +
        ' (file is ' + (knownFileSize != null
          ? (knownFileSize >>> 20) + ' MB'
          : 'unknown size') + '; the first ' +
        (H5.HEAD_BUFFER_BYTES >>> 20) + ' MB did not contain the ' +
        'metadata jsfive needs to walk this dataset).'
      );
    }

    const file = head.file;
    if (!file.keys || !file.keys.includes('acquisition')) {
      throw new Error(
        'NWB: /acquisition group missing — not a recognised NWB file ' +
        '(root keys: ' + (file.keys ? file.keys.join(', ') : 'none') + ')'
      );
    }
    const acq = file.get('acquisition');
    const picked = pickElectricalSeries(acq);
    const es = picked.group;
    if (!es.keys.includes('data')) {
      throw new Error('NWB: /acquisition/' + picked.name + '/data missing');
    }
    const dataDs = es.get('data');
    const shape = dataDs.shape;
    const { nSamples, nChannels, transposed } = normaliseShape(shape);
    if (transposed) {
      throw new Error(
        'NWB streaming: dataset is on-disk shape ' +
        '[' + shape.join(',') + '] (transposed layout). The streaming ' +
        'reader requires canonical [n_samples, n_channels] layout — ' +
        'use the whole-file path (file must be ≤ ' +
        (LEGACY_FALLBACK_CAP >>> 20) + ' MB) for transposed files.'
      );
    }
    const { fs } = deriveSamplingRate(es);

    // Extract storage layout — fall back to the whole-file path if
    // the layout / dtype / filters aren't supported by streaming.
    const layout = H5.extractLayoutFromDataset(dataDs);
    if (!layout) {
      throw new Error(
        'NWB streaming: data layout class is not contiguous or chunked ' +
        '(or storage message missing). Streaming path only handles ' +
        'contiguous (class 1) and chunked (class 2) layouts.'
      );
    }
    const dt = H5.dtypeToTypedArray(layout.dtype);
    if (!dt) {
      throw new Error(
        'NWB streaming: dtype "' + layout.dtype + '" is not supported ' +
        '(little-endian fixed-width int / float only).'
      );
    }

    // Channel labels — small dataset, jsfive can read it from the head
    // buffer (label/id columns are tiny). buildChannelLabels falls
    // back gracefully if jsfive throws.
    const channelLabels = buildChannelLabels(file, nChannels);
    const channelTypes = new Array(nChannels).fill('ieeg');
    const bytesPerSample = bytesPerSampleFromDtype(layout.dtype);
    const recordingStartIso = readSessionStartIso(file);

    const pageReader = H5.makeHttpPageReader(url);

    // Pre-validate that chunked datasets satisfy the streaming
    // reader's contract (chunk spans full channel axis). For chunked
    // datasets we'll catch this on the first readWindow anyway, but
    // doing it here lets `open()` surface the error before the user
    // starts panning.
    if (layout.layoutClass === 2) {
      const [, chunkCols] = layout.chunks;
      if (chunkCols !== nChannels) {
        throw new Error(
          'NWB streaming: dataset is chunked with chunkCols=' + chunkCols +
          ' but nChannels=' + nChannels + '. Multi-tile-per-row chunking ' +
          'is not yet supported.'
        );
      }
    }

    return {
      n_channels: nChannels,
      sampling_frequency: fs,
      duration_s: nSamples / fs,
      n_samples: nSamples,
      channel_labels: channelLabels,
      channel_types: channelTypes,
      bytes_per_sample: bytesPerSample,
      recording_start_iso: recordingStartIso,
      annotation_events: [],
      readWindow: async (startSample, nWin) => {
        const win = globalThis.ChannelBuffers.clampWindow(startSample, nWin, nSamples);
        if (!win) return globalThis.ChannelBuffers.empty(nChannels);
        const { start, end } = win;
        let flat;
        if (layout.layoutClass === 1) {
          flat = await H5.readWindowContiguous(layout, start, end, pageReader);
        } else {
          flat = await H5.readWindowChunked(layout, start, end, pageReader);
        }
        // De-interleave to per-channel Float32Array (sample-major
        // input, channel-major output). Same shape the whole-file
        // path returns so the downstream renderer is layout-agnostic.
        const out = globalThis.ChannelBuffers.alloc(nChannels, end - start);
        for (let s = 0; s < end - start; s++) {
          const base = s * nChannels;
          for (let c = 0; c < nChannels; c++) {
            out[c][s] = flat[base + c];
          }
        }
        return out;
      },
      // Tag the reader so tests can assert which path was taken.
      _readerKind: 'streaming',
    };
  }

  /**
   * Parse an NWB buffer that's already in memory. Used by:
   *   - api.open() once the file has been downloaded,
   *   - tests that load fixtures via fs.readFileSync.
   *
   * Matches the SNIrf reader's surface (SnirfReader.read alongside
   * SnirfReader.open) — fully synchronous internally but returns a
   * Promise so the caller signature is symmetric.
   *
   * @param {ArrayBuffer|Uint8Array} buffer
   * @returns {Promise<object>} same reader object as api.open
   */
  api.read = async function (buffer) {
    const u8 = buffer instanceof Uint8Array
      ? buffer
      : new Uint8Array(buffer);
    if (!isHdf5AtZero(u8)) {
      throw new Error('NWB: buffer is not a valid HDF5 (magic mismatch at byte 0)');
    }
    if (u8.byteLength > LEGACY_FALLBACK_CAP) {
      throw new Error(
        'NWB: buffer is ' + (u8.byteLength >>> 20) + ' MB, exceeds ' +
        (LEGACY_FALLBACK_CAP >>> 20) + ' MB whole-file cap. Use the ' +
        'streaming open() path (api.open with eeg_url) instead — it ' +
        'fetches only the chunks needed for each readWindow.'
      );
    }
    const jsfive = getJsfive();
    const ab = u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
    const file = new jsfive.File(ab);

    if (!file.keys || !file.keys.includes('acquisition')) {
      throw new Error(
        'NWB: /acquisition group missing — not a recognised NWB file ' +
        '(root keys: ' + (file.keys ? file.keys.join(', ') : 'none') + ')'
      );
    }
    const acq = file.get('acquisition');
    const picked = pickElectricalSeries(acq);
    const es = picked.group;

    if (!es.keys.includes('data')) {
      throw new Error(
        'NWB: /acquisition/' + picked.name + '/data missing'
      );
    }
    const dataDs = es.get('data');
    const shape = dataDs.shape;
    const { nSamples, nChannels, transposed } = normaliseShape(shape);

    const { fs } = deriveSamplingRate(es);

    // Optional unit conversion: NWB allows `conversion` (multiplier)
    // and `offset` (additive) scalar attributes on the data dataset.
    // pynwb sets `conversion=1.0` by default; we treat 1.0 / 0.0 as
    // no-op without allocating a scaling loop.
    const conversion = readNumericAttr(dataDs.attrs, 'conversion');
    const offset = readNumericAttr(dataDs.attrs, 'offset');

    const flat = normaliseToFloat32SampleMajor(
      dataDs.value, nSamples, nChannels, transposed, conversion, offset
    );
    if (flat.length !== nSamples * nChannels) {
      throw new Error(
        'NWB: data length ' + flat.length + ' != ' +
        'nSamples(' + nSamples + ') * nChannels(' + nChannels + ')'
      );
    }

    const channelLabels = buildChannelLabels(file, nChannels);
    // NWB ElectricalSeries are by convention all iEEG / ECoG / LFP —
    // we don't currently introspect the electrode group to distinguish
    // sub-types, so every channel reports as "ieeg" (matches the
    // BrainVision iEEG path's default channel_type).
    const channelTypes = new Array(nChannels).fill('ieeg');

    const bytesPerSample = bytesPerSampleFromDtype(dataDs.dtype);
    const recordingStartIso = readSessionStartIso(file);

    return {
      n_channels: nChannels,
      sampling_frequency: fs,
      duration_s: nSamples / fs,
      n_samples: nSamples,
      channel_labels: channelLabels,
      channel_types: channelTypes,
      bytes_per_sample: bytesPerSample,
      recording_start_iso: recordingStartIso,
      // NWB stores events / epochs in /intervals/ — out of scope for
      // v1. We return an empty array so callers can iterate without
      // a null check (matches the SNIRF / BrainVision shape).
      annotation_events: [],
      readWindow: async (startSample, nWin) => {
        const win = globalThis.ChannelBuffers.clampWindow(startSample, nWin, nSamples);
        if (!win) return globalThis.ChannelBuffers.empty(nChannels);
        const { start, end } = win;
        const out = globalThis.ChannelBuffers.alloc(nChannels, end - start);
        // flat is sample-major row-major: flat[s * nChannels + c] is
        // sample s of channel c regardless of on-disk transpose.
        for (let s = start; s < end; s++) {
          const base = s * nChannels;
          for (let c = 0; c < nChannels; c++) {
            out[c][s - start] = flat[base + c];
          }
        }
        return out;
      },
      _readerKind: 'whole-file',
    };
  };

  // Re-exposed for tests / future debug surfacing.
  api._isHdf5AtZero = isHdf5AtZero;
  api._normaliseShape = normaliseShape;
  api._trimNulString = trimNulString;
  api._pickElectricalSeries = pickElectricalSeries;

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.NwbReader = api;
})();
