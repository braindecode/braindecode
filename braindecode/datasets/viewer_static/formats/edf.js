/* ============================================================
   formats/edf.js — read EDF / EDF+ / BDF over HTTP Range.

   Header (256 bytes fixed) + n_signals × 256 bytes of signal
   metadata, followed by n_records data records. Each record holds
   every signal's samples back-to-back in declaration order. Sample
   bytes per signal in record i:  signal[i].samples_per_record × bps,
   bps = 2 (EDF / EDF+) or 3 (BDF, 24-bit signed little-endian).

   Within a single record, signal layout is signal-major (NOT
   sample-interleaved — that's the EEGLAB layout). Reading a window
   = pulling the records that overlap the window with one Range fetch
   and slicing each signal's stripe out of every record.

   What we trust:
     - EDF header for binary layout (n_signals, samples_per_record,
       record_duration, digital/physical scaling).
     - BIDS `_channels.tsv` for display order, names, types — when
       it disagrees with the EDF labels we surface a warning, never
       silently reorder, because reorders would mis-align the binary.

   What we skip in v1:
     - Discontinuous EDF+ (reserved field "EDF+D"). Most BIDS data is
       continuous; we warn and treat as continuous.
     - Per-signal sample rates differing from the EEG channels'.
     - The "EDF Annotations" channel — surfaced as events when we
       extend Phase 1's events.tsv pipeline; for now it's just hidden
       from the display channel set.
   ============================================================ */
(function () {
  'use strict';

  const api = {};
  const HEADER_FIXED = 256;
  // EDF lays out signal-header fields field-major: all labels first,
  // then all transducers, etc. Iterating the outer loop over fields
  // (rather than signals) matches the file order so we walk bytes
  // linearly.
  const SIGNAL_HEADER_FIELDS = [
    ['label',              16],
    ['transducer',         80],
    ['physical_dimension',  8],
    ['physical_min',        8],
    ['physical_max',        8],
    ['digital_min',         8],
    ['digital_max',         8],
    ['prefiltering',       80],
    ['samples_per_record',  8],
    ['reserved',           32],
  ];
  // EDF+ "EDF Annotations" and BDF+ "BDF Annotations" are spec-defined
  // labels for the same TAL events channel. Each is matched only for
  // its own format so EDF-only behaviour stays bit-identical.
  const ANNOTATION_LABEL = /^EDF Annotations\b/i;
  const BDF_ANNOTATION_LABEL = /^BDF Annotations\b/i;

  function ascii(view, offset, length) {
    let s = '';
    for (let i = 0; i < length; i++) {
      const c = view[offset + i];
      if (c === 0) break;
      s += String.fromCharCode(c);
    }
    return s.trim();
  }

  function parseAsciiInt(s, label) {
    const n = parseInt(s, 10);
    if (!Number.isFinite(n)) throw new Error(`EDF header: ${label} not an integer (${JSON.stringify(s)})`);
    return n;
  }

  function parseAsciiFloat(s, label) {
    const n = parseFloat(s);
    if (!Number.isFinite(n)) throw new Error(`EDF header: ${label} not a float (${JSON.stringify(s)})`);
    return n;
  }

  // Most-frequent samples_per_record across `idx`s into `signals`.
  // Tie-break: largest spr (favours real EEG over low-rate auxiliary
  // markers when counts are equal). Exported for unit tests.
  function pickModalSamplesPerRecord(idx, signals) {
    const counts = new Map();
    for (const i of idx) {
      const spr = signals[i].samples_per_record;
      counts.set(spr, (counts.get(spr) || 0) + 1);
    }
    let best = -1, bestCount = -1;
    for (const [spr, n] of counts) {
      if (n > bestCount || (n === bestCount && spr > best)) { best = spr; bestCount = n; }
    }
    return best;
  }

  /**
   * Parse the 256-byte fixed header + per-signal 256-byte metadata blocks
   * of an EDF / EDF+ / BDF file.
   *
   * Returned object (loosely typed as `object` to keep tsc happy while
   * the per-signal metadata is built up across several passes) includes
   * at least:
   *   - version, reserved: string
   *   - isBDF, isEdfPlus, isContinuous: boolean
   *   - header_bytes, n_records, n_signals: number
   *   - record_duration: number  (seconds)
   *   - signals: Array of per-signal records with label, transducer,
   *     physical_dimension, physical_min/max, digital_min/max,
   *     prefiltering, samples_per_record, plus derived scale, offset,
   *     and is_annotation flags.
   *
   * @param {ArrayBuffer} arrayBuf - The first 256 + 256*n_signals bytes.
   *   Must be at least 256 bytes (just the fixed header is OK for n_signals=0).
   * @returns {object}
   */
  api.parseHeader = function (arrayBuf) {
    const v = new Uint8Array(arrayBuf);
    if (v.length < HEADER_FIXED) {
      throw new Error(`EDF header underflow: ${v.length} < ${HEADER_FIXED} bytes`);
    }

    // First byte 0xFF identifies BDF (BioSemi 24-bit). Otherwise it's
    // EDF or EDF+, distinguished by the "reserved" field at byte 192.
    const isBDF = v[0] === 0xFF;
    const reserved = ascii(v, 192, 44);
    const isEdfPlus = reserved.startsWith('EDF+');
    const isContinuous = reserved !== 'EDF+D';

    const headerBytes = parseAsciiInt(ascii(v, 184, 8), 'header_bytes');
    const nRecords    = parseAsciiInt(ascii(v, 236, 8), 'n_records');
    const recordDur   = parseAsciiFloat(ascii(v, 244, 8), 'record_duration');
    const nSignals    = parseAsciiInt(ascii(v, 252, 4), 'n_signals');

    // The EDF/BDF spec mandates header_bytes = 256 × (n_signals + 1).
    // Real-world files sometimes ship with a mis-declared header_bytes
    // (n_signals and the data layout are correct; only the header_bytes
    // field is wrong). Example from the EEGDash catalog:
    //   ds003343: declares header_bytes=4608, n_signals=20.
    //     Spec formula gives 5376. Bytes 4608-5376 are still ASCII
    //     prefiltering metadata; binary data begins at 5376. So the
    //     spec formula is right and the file's header_bytes is wrong.
    //
    // Strategy: when header_bytes disagrees with the spec formula,
    // trust the spec formula (it's derived from n_signals which is
    // also in the binary header and easier to corrupt-check via
    // signal-record parsing). Warn instead of throwing.
    const expectedHeaderBytes = HEADER_FIXED * (nSignals + 1);
    let effectiveHeaderBytes = headerBytes;
    if (headerBytes !== expectedHeaderBytes) {
      console.warn(
        `EDF/BDF: header_bytes field declares ${headerBytes}B but spec formula ` +
        `256·(${nSignals}+1)=${expectedHeaderBytes}B. Using the spec formula ` +
        `(observed mis-declared header_bytes on ds003343 in the wild). ` +
        `If render is corrupt, the file may have a different layout — please report.`,
      );
      effectiveHeaderBytes = expectedHeaderBytes;
    }
    if (v.length < effectiveHeaderBytes) {
      throw new Error(`EDF header buffer ${v.length}B < declared ${effectiveHeaderBytes}B`);
    }

    /** @type {Array<any>} */
    const signals = Array.from({ length: nSignals }, () => ({}));
    let off = HEADER_FIXED;
    for (const [field, size] of /** @type {Array<[string, number]>} */ (SIGNAL_HEADER_FIELDS)) {
      for (let i = 0; i < nSignals; i++) {
        signals[i][field] = ascii(v, off, size);
        off += size;
      }
    }

    for (let i = 0; i < nSignals; i++) {
      const s = signals[i];
      s.physical_min = parseAsciiFloat(s.physical_min, `signal[${i}].physical_min`);
      s.physical_max = parseAsciiFloat(s.physical_max, `signal[${i}].physical_max`);
      s.digital_min  = parseAsciiInt  (s.digital_min,  `signal[${i}].digital_min`);
      s.digital_max  = parseAsciiInt  (s.digital_max,  `signal[${i}].digital_max`);
      s.samples_per_record = parseAsciiInt(s.samples_per_record, `signal[${i}].samples_per_record`);
      // EDF spec requires digital_max > digital_min. A reversed range
      // would silently invert polarity, so reject up front.
      const dRange = s.digital_max - s.digital_min;
      if (dRange <= 0) throw new Error(`signal[${i}] has non-positive digital range (${s.digital_min}..${s.digital_max})`);
      s.scale  = (s.physical_max - s.physical_min) / dRange;
      s.offset = s.physical_min - s.digital_min * s.scale;
      s.is_annotation = ANNOTATION_LABEL.test(s.label) ||
                        (isBDF && BDF_ANNOTATION_LABEL.test(s.label));
    }

    return {
      version: ascii(v, 0, 8),
      isBDF, isEdfPlus, isContinuous, reserved,
      header_bytes: effectiveHeaderBytes,
      n_records: nRecords,
      record_duration: recordDur,
      n_signals: nSignals,
      header_bytes_declared: headerBytes,
      header_bytes_used: effectiveHeaderBytes,
      signals,
    };
  };

  // TAL = Time-stamped Annotation List.
  // Format: +<onset>[\x15<duration>]\x14<text>\x14[<text2>\x14...]\x00
  // \x14 = 0x14 (field separator), \x15 = 0x15 (onset-duration sep), \x00 = record end.
  // The first TAL in each data record is the timestamp anchor (empty text) — skip it.
  api.parseTAL = function (bytes) {
    const events = [];
    const dec = new TextDecoder('utf-8');
    let i = 0;
    const n = bytes.length;

    while (i < n) {
      // Find the next \x00 record boundary.
      let end = i;
      while (end < n && bytes[end] !== 0x00) end++;
      if (end === i) { i++; continue; }   // empty record

      const record = dec.decode(bytes.subarray(i, end));
      i = end + 1;

      // Each record is a set of TALs separated by... actually each
      // record already is one TAL. But the spec allows multiple TALs
      // concatenated before the \x00 — split on the '+' sign at position 0
      // of each sub-record (EDF+ uses '+' prefix for onset).
      // Strategy: split on \x14 first to get all fields, then reconstruct.
      // A TAL is: onset_str [\x15 duration_str] \x14 text \x14 [text2 \x14 ...] \x00
      // After splitting on \x00, each piece is one TAL.
      // We already have one record (between two \x00s). Now parse it.

      // The onset field starts with '+' or '-'. Find the first \x14.
      const sep14 = record.indexOf('\x14');
      if (sep14 < 0) continue;          // malformed, skip

      const onsetPart = record.slice(0, sep14);
      const restPart = record.slice(sep14 + 1);

      // Split onset and duration by \x15.
      const sep15 = onsetPart.indexOf('\x15');
      let onsetStr, durationStr;
      if (sep15 >= 0) {
        onsetStr    = onsetPart.slice(0, sep15);
        durationStr = onsetPart.slice(sep15 + 1);
      } else {
        onsetStr    = onsetPart;
        durationStr = '';
      }

      const onset = parseFloat(onsetStr);
      if (!isFinite(onset)) continue;     // skip anchor records (no '+' prefix? unlikely but safe)

      // The rest after the first \x14 is annotation text(s) terminated by \x14.
      // Multiple annotations are separated by \x14; the record ends with \x14.
      // The very first TAL in a data record has empty text — skip those.
      const textParts = restPart.split('\x14').filter(t => t.length > 0);
      if (!textParts.length) continue;    // timestamp anchor — skip

      const duration = durationStr ? parseFloat(durationStr) : 0;
      for (const text of textParts) {
        events.push({
          onset,
          duration: isFinite(duration) ? duration : 0,
          label: text,
        });
      }
    }
    return events;
  };

  api.open = async function (meta) {
    const url = meta.eeg_url;

    // Run the length probe in parallel with the first 256-byte fetch.
    // The first fetch tells us how many more bytes to ask for; the
    // probe is independent so there's no point blocking on it.
    const [firstBuf, totalBytes] = await Promise.all([
      HttpRange.rangeFetch(url, 0, HEADER_FIXED - 1, HEADER_FIXED),
      HttpRange.probeLength(url),
    ]);
    const declaredHeaderBytes = parseAsciiInt(
      ascii(new Uint8Array(firstBuf), 184, 8), 'header_bytes (first probe)');
    // ds003343 in the wild: declared header_bytes=4608, n_signals=20,
    // but data actually starts at the spec-formula offset 5376 (header
    // bytes 4608-5375 contain ASCII Prefiltering metadata). To handle
    // this, we ALSO read n_signals here and fetch enough bytes to
    // satisfy MAX(declared, spec-formula).
    const declaredNSignals = parseAsciiInt(
      ascii(new Uint8Array(firstBuf), 252, 4), 'n_signals (first probe)');
    const specHeaderBytes = HEADER_FIXED * (declaredNSignals + 1);
    const headerFetchBytes = Math.max(declaredHeaderBytes, specHeaderBytes);

    let headerBuf;
    if (headerFetchBytes === HEADER_FIXED) {
      headerBuf = firstBuf;
    } else {
      // Re-use what we already fetched: pull only the remaining
      // signal-header bytes and concatenate, instead of re-downloading
      // the fixed 256 we just got.
      const restBuf = await HttpRange.rangeFetch(
        url, HEADER_FIXED, headerFetchBytes - 1, headerFetchBytes - HEADER_FIXED);
      const merged = new Uint8Array(headerFetchBytes);
      merged.set(new Uint8Array(firstBuf), 0);
      merged.set(new Uint8Array(restBuf), HEADER_FIXED);
      headerBuf = merged.buffer;
    }
    const hdr = api.parseHeader(headerBuf);

    if (!hdr.isContinuous) {
      console.warn('EDF+D (discontinuous) treated as continuous in v1; sample timing may be off across record boundaries.');
    }

    const dataBytes = totalBytes - hdr.header_bytes;
    const bytesPerSample = hdr.isBDF ? 3 : 2;
    const samplesPerRecord = hdr.signals.reduce((sum, s) => sum + s.samples_per_record, 0);
    const recordSize = samplesPerRecord * bytesPerSample;

    if (recordSize === 0) throw new Error('EDF: zero-byte data record (signals report samples_per_record=0)');
    if (dataBytes < 0) {
      throw new Error(`EDF data section is negative (${dataBytes}B) — header_bytes > totalBytes`);
    }
    // Trailing-bytes tolerance — observed on ds003343 (BDF data section
    // 11909232B vs record size 30000B, 29232B trailing). MNE-Python rejects
    // here, but truncating to floor(dataBytes / recordSize) records lets
    // the user view the available 396 of 397 records with a warning instead
    // of a hard reject. Symmetry with the EEGLAB-truncated-fdt fix.
    const recRem = dataBytes % recordSize;
    if (recRem !== 0) {
      console.warn(
        `EDF/BDF data section ${dataBytes}B is not a multiple of record size ` +
        `${recordSize}B (${recRem}B trailing). File is likely truncated mid-record; ` +
        `displaying first ${Math.floor(dataBytes / recordSize)} complete records ` +
        `(observed on ds003343 in the wild). If render looks corrupt, the file may ` +
        `have a different layout — please report.`
      );
    }
    const actualRecords = Math.floor(dataBytes / recordSize);
    if (actualRecords === 0) {
      throw new Error(
        `EDF data section ${dataBytes}B < record size ${recordSize}B — no complete records.`
      );
    }
    if (hdr.n_records !== -1 && hdr.n_records !== actualRecords) {
      console.warn(`EDF declared ${hdr.n_records} records but file has ${actualRecords}; trusting file.`);
    }

    // v1: require uniform sample rate across display signals so the
    // returned window is a clean (n_channels × n_samples) shape.
    // Annotation channels are excluded from this check.
    let displayIdx = hdr.signals
      .map((_, i) => i)
      .filter(i => !hdr.signals[i].is_annotation);
    if (!displayIdx.length) throw new Error('EDF has no non-annotation signals');

    // BDF-only: BIDS-converted BioSemi files often carry a marker /
    // Status / TRIG channel at a much lower rate than the EEG block
    // (and those don't always carry the "BDF Annotations" label, so
    // the label-based filter above misses them). Drop signals whose
    // samples_per_record disagrees with the modal rate of the rest;
    // for uniform-rate files this is a no-op. Scoped to BDF so EDF/
    // EDF+ behaviour stays bit-identical for OpenNeuro datasets.
    // Drop auxiliary signals at a non-modal sample rate so the displayed
    // window is rectangular (channels × samples). Originally scoped to
    // BDF only out of caution, but the same pattern is real on plain
    // EDF too — markers/triggers/status channels at lower sample rates
    // alongside the EEG block. Extended to all EDF variants as part of
    // the "be more generous" pass (2026-05-22, audit category fix).
    // The warn names every dropped channel so users can audit.
    if (displayIdx.length > 1) {
      const modalSpr = pickModalSamplesPerRecord(displayIdx, hdr.signals);
      const filtered = displayIdx.filter(i => hdr.signals[i].samples_per_record === modalSpr);
      if (filtered.length !== displayIdx.length) {
        const dropped = displayIdx.filter(i => !filtered.includes(i));
        const labels = dropped.map(i => `${hdr.signals[i].label}(${hdr.signals[i].samples_per_record})`).join(', ');
        console.warn(
          `${hdr.isBDF ? 'BDF' : 'EDF'}: dropping ${dropped.length} auxiliary ` +
          `channel(s) at non-modal rate: ${labels}; keeping ${filtered.length} ` +
          `at samples_per_record=${modalSpr}.`,
        );
        displayIdx = filtered;
      }
    }

    const sprPerDisplay = displayIdx.map(i => hdr.signals[i].samples_per_record);
    const sprFirst = sprPerDisplay[0];
    if (sprPerDisplay.some(s => s !== sprFirst)) {
      throw new Error(`v1 requires uniform samples_per_record across display signals; got ${[...new Set(sprPerDisplay)].join(', ')}`);
    }
    const fs = sprFirst / hdr.record_duration;
    const nSamples = actualRecords * sprFirst;

    SidecarChecks.warnFsMismatch(meta.eeg_json.sampling_frequency, fs, 'EDF');

    // Pre-compute everything readWindow needs keyed by display-channel
    // index `c`, so the hot path doesn't pay an indirection per channel
    // per record. Using typed arrays keeps lookup tight.
    const nDisplay = displayIdx.length;
    const cumByOrigIdx = new Int32Array(hdr.n_signals);
    let cum = 0;
    for (let i = 0; i < hdr.n_signals; i++) {
      cumByOrigIdx[i] = cum * bytesPerSample;
      cum += hdr.signals[i].samples_per_record;
    }
    const sigOffsetInRec = new Int32Array(nDisplay);
    const scales = new Float64Array(nDisplay);
    const offsets = new Float64Array(nDisplay);
    const channelLabels = new Array(nDisplay);
    for (let c = 0; c < nDisplay; c++) {
      const oi = displayIdx[c];
      sigOffsetInRec[c] = cumByOrigIdx[oi];
      scales[c]  = hdr.signals[oi].scale;
      offsets[c] = hdr.signals[oi].offset;
      channelLabels[c] = hdr.signals[oi].label;
    }

    SidecarChecks.crossCheckChannelOrder(channelLabels, meta.channels, 'EDF');

    // Parse recording start date+time from the EDF fixed header.
    // Offset 168: "startdate of recording" (8 bytes, format "dd.mm.yy")
    // Offset 176: "starttime of recording" (8 bytes, format "hh.mm.ss")
    // EDF spec year heuristic: yy < 85 → 20yy, else 19yy.
    let recording_start_iso = null;
    try {
      const v = new Uint8Array(headerBuf);
      const dateStr = ascii(v, 168, 8);  // "dd.mm.yy"
      const timeStr = ascii(v, 176, 8);  // "hh.mm.ss"
      const dateParts = dateStr.split('.');
      const timeParts = timeStr.split('.');
      if (dateParts.length === 3 && timeParts.length === 3) {
        const dd = dateParts[0].padStart(2, '0');
        const mm = dateParts[1].padStart(2, '0');
        const yy = parseInt(dateParts[2], 10);
        const yyyy = yy < 85 ? 2000 + yy : 1900 + yy;
        const hh = timeParts[0].padStart(2, '0');
        const mn = timeParts[1].padStart(2, '0');
        const ss = timeParts[2].padStart(2, '0');
        // Validate all parts are numbers
        if (!isNaN(yy) && !isNaN(parseInt(dd, 10)) && !isNaN(parseInt(mm, 10)) &&
            !isNaN(parseInt(hh, 10)) && !isNaN(parseInt(mn, 10)) && !isNaN(parseInt(ss, 10))) {
          recording_start_iso = `${yyyy}-${mm}-${dd}T${hh}:${mn}:${ss}`;
        }
      }
    } catch (_) {
      // Leave recording_start_iso as null if parsing fails
    }

    // Layout state for readWindow lives in this closure rather than on
    // the public handle so callers can't accidentally rely on internals.
    const layout = {
      url,
      isBDF: hdr.isBDF,
      header_bytes: hdr.header_bytes,
      record_size_bytes: recordSize,
      samples_per_record: sprFirst,
      n_records: actualRecords,
      n_samples: nSamples,
      n_channels: nDisplay,
      sigOffsetInRec, scales, offsets,
    };
    const readerFn = hdr.isBDF ? readWindowBDF : readWindowEDF;

    // Parse TAL records from every annotation channel and surface them
    // as structured events. Annotation channels store raw bytes (not
    // calibrated int16s) so we read them differently: pull all data
    // records for each annotation channel and decode as ASCII TAL.
    let annotation_events = [];
    const annotIdx = hdr.signals
      .map((s, i) => s.is_annotation ? i : -1)
      .filter(i => i >= 0);

    if (annotIdx.length) {
      // Pre-compute byte offset of each annotation channel within each record.
      // cumByOrigIdx was computed above for display channels; reuse the logic.
      const annotOffsets = annotIdx.map(i => {
        let cum = 0;
        for (let j = 0; j < i; j++) cum += hdr.signals[j].samples_per_record;
        return cum * bytesPerSample;
      });
      const annotByteLens = annotIdx.map(i =>
        hdr.signals[i].samples_per_record * bytesPerSample);

      // Read all data records in one range fetch (files with annotations
      // tend to be small; the entire data section is usually < 1 MB).
      const dataStart = hdr.header_bytes;
      const dataEnd   = dataStart + actualRecords * recordSize - 1;
      try {
        const dataBuf = await HttpRange.rangeFetch(url, dataStart, dataEnd, actualRecords * recordSize);
        const u8 = new Uint8Array(dataBuf);

        for (let ai = 0; ai < annotIdx.length; ai++) {
          const chanOff  = annotOffsets[ai];
          const chanLen  = annotByteLens[ai];
          // Gather all bytes for this annotation channel across all records.
          const allBytes = new Uint8Array(actualRecords * chanLen);
          for (let r = 0; r < actualRecords; r++) {
            const srcOff = r * recordSize + chanOff;
            allBytes.set(u8.subarray(srcOff, srcOff + chanLen), r * chanLen);
          }
          const parsed = api.parseTAL(allBytes);
          annotation_events = annotation_events.concat(parsed);
        }
        // Sort by onset time for consistent display.
        annotation_events.sort((a, b) => a.onset - b.onset);
      } catch (e) {
        console.warn('EDF+: could not read annotation channel bytes; events skipped.', e.message);
      }
    }

    return {
      n_channels: nDisplay,
      n_samples: nSamples,
      sampling_frequency: fs,
      duration_s: nSamples / fs,
      bytes_per_sample: bytesPerSample,
      url,
      channel_labels: channelLabels,
      bids_channels: meta.channels || null,
      recording_start_iso,
      annotation_events,
      readWindow: (start, n, opts) => readerFn(layout, start, n, opts),
      readWindowStreaming: (start, n, opts) => streamWindowEDF(layout, hdr.isBDF, start, n, opts),
    };
  };

  // Hot path. Pull every record overlapping the requested window in
  // one Range, then slice each display channel out of each record.
  // Common scaffolding lives here; EDF (Int16) and BDF (Int24)
  // diverge only in the inner per-sample decode, hoisted into
  // dedicated `readWindowEDF` / `readWindowBDF` so the inner loop
  // is branch-free.
  async function fetchWindowBuffer(layout, startSample, nWinReq, opts) {
    const win = ChannelBuffers.clampWindow(startSample, nWinReq, layout.n_samples);
    if (!win) return null;
    const { start, end, nWin } = win;
    const spr = layout.samples_per_record;
    const firstRec = Math.floor(start / spr);
    const lastRec = Math.ceil(end / spr);
    const nRecs = lastRec - firstRec;
    const byteStart = layout.header_bytes + firstRec * layout.record_size_bytes;
    const byteEnd = byteStart + nRecs * layout.record_size_bytes - 1;
    const buf = await HttpRange.rangeFetch(layout.url, byteStart, byteEnd, nRecs * layout.record_size_bytes, opts);
    return { buf, nWin, nRecs, startOffsetInFirstRec: start - firstRec * spr };
  }

  async function readWindowEDF(layout, startSample, nWinReq, opts) {
    const win = await fetchWindowBuffer(layout, startSample, nWinReq, opts);
    if (win == null) return ChannelBuffers.empty(layout.n_channels);
    const { buf, nWin, nRecs, startOffsetInFirstRec } = win;
    // EDF samples are int16 little-endian. Records start at byte
    // offsets that are always even (header_bytes is 256·(n_sig+1) and
    // record_size is samples_per_record_total × 2), so we can view the
    // whole buffer as Int16Array indexed in 16-bit units.
    const i16 = new Int16Array(buf);
    const halfRec = layout.record_size_bytes >> 1;        // record size in int16 units
    const out = ChannelBuffers.alloc(layout.n_channels, nWin);
    const spr = layout.samples_per_record;

    for (let c = 0; c < layout.n_channels; c++) {
      const sigOffI16 = layout.sigOffsetInRec[c] >> 1;    // bytes → int16 index
      const scale = layout.scales[c];
      const offset = layout.offsets[c];
      const ch = out[c];
      let outIdx = 0;
      for (let r = 0; r < nRecs && outIdx < nWin; r++) {
        const sigBase = r * halfRec + sigOffI16;
        const recStart = (r === 0) ? startOffsetInFirstRec : 0;
        const recEnd = Math.min(spr, recStart + (nWin - outIdx));
        for (let s = recStart; s < recEnd; s++) {
          ch[outIdx++] = i16[sigBase + s] * scale + offset;
        }
      }
    }
    return out;
  }

  async function readWindowBDF(layout, startSample, nWinReq, opts) {
    const win = await fetchWindowBuffer(layout, startSample, nWinReq, opts);
    if (win == null) return ChannelBuffers.empty(layout.n_channels);
    const { buf, nWin, nRecs, startOffsetInFirstRec } = win;
    // BDF samples are 24-bit signed little-endian. Pack the 3 bytes
    // into the high 24 bits of a 32-bit int and arithmetic-shift
    // right 8 to sign-extend in one shot.
    const u8 = new Uint8Array(buf);
    const recSize = layout.record_size_bytes;
    const out = ChannelBuffers.alloc(layout.n_channels, nWin);
    const spr = layout.samples_per_record;

    for (let c = 0; c < layout.n_channels; c++) {
      const sigOff = layout.sigOffsetInRec[c];
      const scale = layout.scales[c];
      const offset = layout.offsets[c];
      const ch = out[c];
      let outIdx = 0;
      for (let r = 0; r < nRecs && outIdx < nWin; r++) {
        const sigBase = r * recSize + sigOff;
        const recStart = (r === 0) ? startOffsetInFirstRec : 0;
        const recEnd = Math.min(spr, recStart + (nWin - outIdx));
        for (let s = recStart; s < recEnd; s++) {
          const o = sigBase + s * 3;
          const raw = ((u8[o] | (u8[o + 1] << 8) | (u8[o + 2] << 16)) << 8) >> 8;
          ch[outIdx++] = raw * scale + offset;
        }
      }
    }
    return out;
  }

  // Streaming decode: yields per-batch { firstSampleIdx, lastSampleIdx, channels }
  // as EDF data-record bytes arrive. Emits every STREAM_BATCH_RECORDS records
  // (or fewer at boundaries). Uses decodeChunkBoundary to handle chunk splits.
  // Non-streaming-safe when opts.filtered — caller should collapse to single chunk.
  const STREAM_BATCH_RECORDS = 8;

  async function* streamWindowEDF(layout, isBDF, startSample, nWinReq, opts) {
    const win = ChannelBuffers.clampWindow(startSample, nWinReq, layout.n_samples);
    if (!win) return;
    const { start, end, nWin } = win;
    const spr = layout.samples_per_record;
    const nCh = layout.n_channels;
    const firstRec = Math.floor(start / spr);
    const lastRec = Math.ceil(end / spr);
    const startOffsetInFirstRec = start - firstRec * spr;
    const byteStart = layout.header_bytes + firstRec * layout.record_size_bytes;
    const byteEnd = byteStart + (lastRec - firstRec) * layout.record_size_bytes - 1;

    // Accumulators for the streaming decode
    let leftover = new Uint8Array(0);
    let recIdx = 0;                  // record index within the fetched range
    const totalRecs = lastRec - firstRec;
    const bytesPerSample = isBDF ? 3 : 2;

    // Output buffer — grows batch by batch, reused across yields
    let outSamples = 0;              // samples written so far across all batches
    let batchBuf = null;             // Float32Array[] for current batch
    let batchRecStart = 0;           // record index of current batch start
    let batchSampleStart = 0;        // global first sample index of current batch

    function flushBatch(isLast) {
      if (!batchBuf || batchBuf[0].length === 0) return null;
      const firstSampleIdx = start + batchSampleStart;
      const lastSampleIdx = firstSampleIdx + batchBuf[0].length - 1;
      return { firstSampleIdx, lastSampleIdx, channels: batchBuf };
    }

    // Decode N complete records from a flat Uint8Array, starting at recIdx.
    // Returns samples decoded across all channels.
    function decodeRecords(u8, nRecs, startingRecIdx) {
      const halfRec = layout.record_size_bytes >> 1;
      // How many output samples does this batch contribute?
      let totalSamples = 0;
      for (let r = 0; r < nRecs; r++) {
        const globalRec = startingRecIdx + r;
        const recStart = (globalRec === 0) ? startOffsetInFirstRec : 0;
        const recEndLimit = (globalRec === totalRecs - 1)
          ? Math.min(spr, recStart + (nWin - outSamples))
          : spr;
        totalSamples += recEndLimit - recStart;
      }
      if (totalSamples <= 0) return null;

      const out = ChannelBuffers.alloc(nCh, totalSamples);
      let outIdx = 0;

      if (isBDF) {
        const recSize = layout.record_size_bytes;
        for (let r = 0; r < nRecs; r++) {
          const globalRec = startingRecIdx + r;
          const recStart = (globalRec === 0) ? startOffsetInFirstRec : 0;
          const recEndLimit = (globalRec === totalRecs - 1)
            ? Math.min(spr, recStart + (nWin - outSamples - outIdx))
            : spr;
          for (let c = 0; c < nCh; c++) {
            const sigOff = layout.sigOffsetInRec[c];
            const scale = layout.scales[c];
            const offset = layout.offsets[c];
            const ch = out[c];
            const sigBase = r * recSize + sigOff;
            for (let s = recStart; s < recEndLimit; s++) {
              const o = sigBase + s * 3;
              const raw = ((u8[o] | (u8[o + 1] << 8) | (u8[o + 2] << 16)) << 8) >> 8;
              ch[outIdx + (s - recStart)] = raw * scale + offset;
            }
          }
          outIdx += recEndLimit - recStart;
        }
      } else {
        // EDF: int16 little-endian
        const i16 = new Int16Array(u8.buffer, u8.byteOffset, u8.byteLength >> 1);
        for (let r = 0; r < nRecs; r++) {
          const globalRec = startingRecIdx + r;
          const recStart = (globalRec === 0) ? startOffsetInFirstRec : 0;
          const recEndLimit = (globalRec === totalRecs - 1)
            ? Math.min(spr, recStart + (nWin - outSamples - outIdx))
            : spr;
          for (let c = 0; c < nCh; c++) {
            const sigOffI16 = layout.sigOffsetInRec[c] >> 1;
            const scale = layout.scales[c];
            const offset = layout.offsets[c];
            const ch = out[c];
            const sigBase = r * halfRec + sigOffI16;
            for (let s = recStart; s < recEndLimit; s++) {
              ch[outIdx + (s - recStart)] = i16[sigBase + s] * scale + offset;
            }
          }
          outIdx += recEndLimit - recStart;
        }
      }
      return out;
    }

    const streamOpts = opts;
    for await (const { bytes } of HttpRange.rangeFetchStreaming(
      layout.url, byteStart, byteEnd, streamOpts
    )) {
      const boundary = StreamingUtils.decodeChunkBoundary(
        leftover, bytes, layout.record_size_bytes
      );
      leftover = boundary.leftover;
      const completeBytes = boundary.completeRecordBytes;
      const nNewRecs = Math.floor(completeBytes.length / layout.record_size_bytes);
      if (nNewRecs === 0) continue;

      // Accumulate records into batch; emit when batch is full
      let rOff = 0;
      while (rOff < nNewRecs && outSamples < nWin) {
        const recsThisBatch = Math.min(STREAM_BATCH_RECORDS, nNewRecs - rOff);
        const batchU8 = completeBytes.subarray(
          rOff * layout.record_size_bytes,
          (rOff + recsThisBatch) * layout.record_size_bytes
        );
        const decoded = decodeRecords(batchU8, recsThisBatch, recIdx + rOff);
        if (decoded && decoded[0].length > 0) {
          const firstSampleIdx = start + outSamples;
          const lastSampleIdx = firstSampleIdx + decoded[0].length - 1;
          outSamples += decoded[0].length;
          yield { firstSampleIdx, lastSampleIdx, channels: decoded };
        }
        rOff += recsThisBatch;
      }
      recIdx += nNewRecs;
    }
  }

  api._pickModalSamplesPerRecord = pickModalSamplesPerRecord;

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.EDFReader = api;
})();
