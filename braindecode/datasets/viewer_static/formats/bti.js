/* ============================================================
   formats/bti.js — minimal BTi / 4D Neuroimaging (Magnes WH3600)
   MEG reader for eegdash-viewer.

   BTi recordings are *directory bundles* without a common
   extension — the directory itself has NO suffix, and the files
   inside carry literal names with no extension either:

     config         binary system metadata (channel maps, calibration
                    tables, weight tables, …). Multi-megabyte in real
                    recordings; the bulk of it isn't needed to open the
                    file because the PDF (next) carries its own header.
                    See formats/_bti-config.js for the (deferred)
                    config-block parser.
     c,rfDC         the "PDF" (Patient Data File) — raw continuous
                    samples. Naming convention encodes the acquisition
                    filter:
                       c,rfDC      raw, no high-pass
                       c,rfhp1.0Hz raw, 1.0 Hz HPF
                       c,rfhp0.1Hz raw, 0.1 Hz HPF
                    All carry identical binary structure. The reader
                    auto-discovers which name is present.
     hs_file        head-shape digitisation points (optional, ignored
                    by this reader — viewer doesn't render head shape).

   Binary format (verified against MNE-Python's mne/io/bti/bti.py
   `_read_bti_header_pdf` + mne/io/bti/read.py at the time of
   authorship). ALL multi-byte values are BIG-ENDIAN on disk —
   distinct from KIT (LE) and EEGLAB (LE), shared with CTF and FIFF.

   PDF file layout:
     offset 0..N-1       interleaved per-sample, per-channel signal
                         values. dtype is one of:
                           data_format=1 → int16 BE  (i2)
                           data_format=2 → int32 BE  (i4)
                           data_format=3 → float32 BE (f4)
                           data_format=4 → float64 BE (f8)
     offset N            PDF header (variable length — minimum ~568 B
                         for one epoch and a handful of channels; real
                         files are larger because of channel records,
                         event lists and process metadata).
     offset fileLen-8    int64 BE = `header_position` = N. Read first
                         to discover where the header starts.

   This reader does NOT yet:
     - Apply per-channel calibration from the `config` user blocks
       (B_E_table_used + per-channel `gain` × `units_per_bit`). The
       viewer's auto-scale handles the magnitude; absolute Tesla
       units are deferred.
     - Parse the `hs_file` head-shape points (the viewer has no
       head-shape view).
     - Handle multi-epoch PDFs (real continuous recordings have
       total_epochs=1; epoched data is out of scope for the initial
       reader). The bounds check below rejects multi-epoch files
       with a clean error.
     - Parse channel labels from the config's `B_ch_labels` user block.
       Falls back to indexed labels Ch1..ChN with a TODO. Real BTi
       recordings name channels A1..A248 (MEG) + E1..E64 (EEG); a
       follow-up should parse them.

   References (vendored, BSD-3-clause):
   - mne/io/bti/bti.py        _read_bti_header_pdf, _read_epoch,
                              _read_channel
   - mne/io/bti/read.py       read primitives (read_int16, read_int32,
                              read_int64, read_float, read_double)
   - mne/io/bti/constants.py  BTI.FILE_* (FILE_MASK = 2147483647,
                              FILE_CURPOS = 8 = alignment quantum)

   ============================================================
   Portions derived from MNE-Python — Copyright the MNE-Python
   contributors, BSD-3-clause license. See:
   https://github.com/mne-tools/mne-python/blob/main/LICENSE.txt
   ============================================================ */
(function () {
  'use strict';

  const api = {};

  // ---- constants (BTI.*) -------------------------------------------
  // Mirror the constants in mne/io/bti/constants.py. Kept here so the
  // reader is self-contained — the only cross-format dep is HttpRange.
  const FILE_MASK = 2147483647;        // BTI.FILE_MASK — 32-bit truncate mask
  const FILE_CURPOS = 8;               // BTI.FILE_CURPOS — alignment quantum
  const FILE_END_PTR_BYTES = 8;        // int64 BE at end of file

  // PDF header is at most ~1 MB in practice (large channel rosters +
  // event tables). We cap our tail-fetch at 1 MiB to bound memory
  // before we know the true header size. If a real-world recording's
  // header exceeds this, the reader throws — much better than silently
  // truncating into garbage.
  const HEADER_TAIL_FETCH_CAP = 1024 * 1024;

  // Bytes-per-sample table, indexed by `data_format` (1..4) per MNE's
  // `DTYPES = {1: ">i2", 2: ">i4", 3: ">f4", 4: ">f8"}` (bti.py:41-42).
  // We carry both the byte width and an opaque dtype tag so readWindow
  // can dispatch to the right decode loop.
  const DTYPE_INFO = {
    1: { bytes: 2, kind: 'i2' },
    2: { bytes: 4, kind: 'i4' },
    3: { bytes: 4, kind: 'f4' },
    4: { bytes: 8, kind: 'f8' },
  };

  // Common PDF filenames in priority order. The literal naming encodes
  // the acquisition-time filter; we don't care which is present, only
  // that we find one. `c,rfDC` is by far the most common in real BIDS
  // BTi recordings (the original archive at Aston University, the NIH
  // MEG core data, and BIDS-BTi conversions in OpenNeuro all default
  // to it). MNE-Python documents the same priority list in
  // mne.io.read_raw_bti's `pdf_fname` discovery.
  const PDF_CANDIDATES = ['c,rfDC', 'c,rfhp1.0Hz', 'c,rfhp0.1Hz', 'c,rfhp10Hz', 'c,rfhp100Hz'];

  // ---- public API --------------------------------------------------

  /**
   * Parse a BTi PDF ArrayBuffer into a header object.
   * Synchronous entry point exposed for unit + property tests so the
   * tail-header logic can be exercised without a network roundtrip.
   * Production `api.open` uses Range fetches to pull only the tail.
   *
   * @param {ArrayBuffer} buf - the PDF (e.g. `c,rfDC`) as one buffer.
   * @returns {{
   *   n_channels: number,
   *   sampling_frequency: number,
   *   n_samples: number,
   *   data_format: number,
   *   sample_size: number,
   *   header_position: number,
   *   total_epochs: number,
   *   version: number,
   *   file_type: string
   * }}
   * @throws {Error} on any parse failure (truncated file, bad pointer,
   *   unsupported data_format, multi-epoch).
   */
  api.read = function (buf) {
    if (!buf || buf.byteLength < FILE_END_PTR_BYTES) {
      throw new Error(
        `bti.read: buffer too small (${buf ? buf.byteLength : 0} bytes) — ` +
          `need at least ${FILE_END_PTR_BYTES}B for the trailing header pointer`,
      );
    }
    const view = new DataView(buf);
    const fileLen = buf.byteLength;
    const headerPosition = resolveHeaderPosition(view, fileLen);
    const hdrBytes = fileLen - FILE_END_PTR_BYTES - headerPosition;
    if (hdrBytes < 0 || headerPosition >= fileLen) {
      throw new Error(
        `bti.read: header_position ${headerPosition} is past end-of-file (${fileLen})`,
      );
    }
    return parsePdfHeader(view, headerPosition, hdrBytes, fileLen);
  };

  /**
   * Open a BTi recording for windowed reading.
   *
   * `meta.eeg_url` should point at the PDF file inside the bundle
   * (e.g. `…/bti-tiny/c,rfDC`). If it points at the bundle directory
   * or at the `config` file, we probe the standard PDF filenames in
   * the same directory and pick whichever exists.
   *
   * @param {object} meta - { eeg_url: string, … }.
   * @returns {Promise<object>} reader with the cross-format contract:
   *   n_channels, sampling_frequency, n_samples, duration_s,
   *   channel_labels, channel_types, bytes_per_sample,
   *   readWindow(start, n).
   */
  api.open = async function (meta) {
    const HttpRange = globalThis.HttpRange;
    if (!HttpRange) throw new Error('bti.open: globalThis.HttpRange missing');

    const inputUrl = meta && (meta.eeg_url || meta.url);
    if (!inputUrl) throw new Error('bti.open: meta.eeg_url is required');

    // ── Step 1: resolve the PDF URL ────────────────────────────────
    // The caller can pass any of:
    //   - <bundle>/c,rfDC        — direct PDF URL
    //   - <bundle>/c,rfhp1.0Hz   — direct PDF URL (alternate filter)
    //   - <bundle>/config        — sibling file in the bundle
    //   - <bundle>/               — the bundle directory itself
    // We normalise to the PDF URL by probing common names.
    const pdfUrl = await resolvePdfUrl(inputUrl, HttpRange);

    // ── Step 2: range-fetch the trailing header ────────────────────
    // Step 2a: get total length (probeLengthNoHead avoids the HEAD-
    // caches-as-200 trap that bit fiff.js / eeglab.js on Cloudflare).
    const totalBytes = await HttpRange.probeLengthNoHead(pdfUrl);
    if (totalBytes < FILE_END_PTR_BYTES) {
      throw new Error(
        `bti.open: PDF file too small (${totalBytes}B) — need at least ` +
          `${FILE_END_PTR_BYTES}B for the trailing pointer`,
      );
    }

    // Step 2b: the last 8 bytes give us header_position. Fetch those
    // first; everything else follows.
    const ptrBuf = await HttpRange.rangeFetch(
      pdfUrl,
      totalBytes - FILE_END_PTR_BYTES,
      totalBytes - 1,
      FILE_END_PTR_BYTES,
    );
    const ptrView = new DataView(ptrBuf);
    const rawPtr = readUint64BE(ptrView, 0);
    // MNE's mask: take low 31 bits only when the value is large.
    // The exact logic in bti.py:770-775 is:
    //   if (start + FILE_CURPOS - check_value) <= FILE_MASK:
    //       header_position = check_value
    // where check_value = header_position & FILE_MASK. In practice
    // (real BTi files) the unmasked value already lives below FILE_MASK,
    // so we apply the mask unconditionally and trust the bounds check.
    let headerPosition = Number(rawPtr & BigInt(FILE_MASK));
    // 8-byte alignment (bti.py:778)
    if (headerPosition % FILE_CURPOS !== 0) {
      headerPosition += FILE_CURPOS - (headerPosition % FILE_CURPOS);
    }
    if (headerPosition < 0 || headerPosition >= totalBytes - FILE_END_PTR_BYTES) {
      throw new Error(
        `bti.open: header_position ${headerPosition} out of bounds (file ${totalBytes}B)`,
      );
    }
    const headerLen = totalBytes - FILE_END_PTR_BYTES - headerPosition;
    if (headerLen > HEADER_TAIL_FETCH_CAP) {
      throw new Error(
        `bti.open: PDF header is ${headerLen}B — exceeds ${HEADER_TAIL_FETCH_CAP}B safety cap. ` +
          `If this is a legitimate BTi recording the cap can be raised.`,
      );
    }

    // Step 2c: fetch the header. We need to parse the leading 96 bytes
    // (fixed fields) + total_epochs*56 + total_chans*104 — but we
    // don't know n_chans / n_epochs until we've started reading. The
    // simplest correct approach: fetch the whole header in one go,
    // which is well-bounded by the cap above.
    const hdrBuf = await HttpRange.rangeFetch(
      pdfUrl,
      headerPosition,
      totalBytes - FILE_END_PTR_BYTES - 1,
      headerLen,
    );
    const hdrView = new DataView(hdrBuf);
    const header = parsePdfHeaderBlock(hdrView, headerLen);

    // ── Step 3: sanity-check data span against file length ─────────
    // n_samples (sum of epoch pts_in_epoch) × n_channels × sample_size
    // MUST equal `header_position` (= the size of the data section).
    const dataBytes = header.n_samples * header.n_channels * header.sample_size;
    if (dataBytes !== headerPosition) {
      throw new Error(
        `bti.open: data section size mismatch — header says ` +
          `${header.n_samples} samples × ${header.n_channels} channels × ` +
          `${header.sample_size}B = ${dataBytes}B, but data section is ` +
          `${headerPosition}B`,
      );
    }

    const channel_labels = globalThis.ChannelLabels
      ? globalThis.ChannelLabels.indexed(header.n_channels)
      : indexedLabelsFallback(header.n_channels);

    // ── Step 4: build readWindow ───────────────────────────────────
    // Interleaved BE decode loop. We use a DataView (not a TypedArray
    // wrap) because TypedArrays adopt host endianness and the platforms
    // we ship to are LE — a direct Int16Array view of BE bytes would
    // byte-swap silently.
    const nch = header.n_channels;
    const sampleSize = header.sample_size;
    const dtypeKind = DTYPE_INFO[header.data_format].kind;
    const totalSamples = header.n_samples;

    async function readWindow(startSample, nWin, opts) {
      const ChannelBuffers = globalThis.ChannelBuffers;
      if (!ChannelBuffers) throw new Error('bti.readWindow: globalThis.ChannelBuffers missing');
      const win = ChannelBuffers.clampWindow(startSample, nWin, totalSamples);
      if (!win) return ChannelBuffers.empty(nch);
      const { start, end } = win;
      const nOut = end - start;

      // Byte range: data section starts at offset 0; per-sample stride
      // is nch * sampleSize.
      const byteStart = start * nch * sampleSize;
      const byteEnd = byteStart + nOut * nch * sampleSize - 1;
      const buf = await HttpRange.rangeFetch(
        pdfUrl,
        byteStart,
        byteEnd,
        byteEnd - byteStart + 1,
        opts,
      );
      const dv = new DataView(buf);
      const out = ChannelBuffers.alloc(nch, nOut);

      // Per-dtype decode loop. Two-branch split keeps the inner loop
      // tight (no per-sample switch).
      if (dtypeKind === 'f4') {
        for (let t = 0; t < nOut; t++) {
          const base = t * nch * sampleSize;
          for (let c = 0; c < nch; c++) {
            out[c][t] = dv.getFloat32(base + c * sampleSize, false);
          }
        }
      } else if (dtypeKind === 'f8') {
        for (let t = 0; t < nOut; t++) {
          const base = t * nch * sampleSize;
          for (let c = 0; c < nch; c++) {
            out[c][t] = dv.getFloat64(base + c * sampleSize, false);
          }
        }
      } else if (dtypeKind === 'i2') {
        for (let t = 0; t < nOut; t++) {
          const base = t * nch * sampleSize;
          for (let c = 0; c < nch; c++) {
            out[c][t] = dv.getInt16(base + c * sampleSize, false);
          }
        }
      } else {
        // i4 — covered by the gate above. dtypeKind is one of i2/i4/f4/f8.
        for (let t = 0; t < nOut; t++) {
          const base = t * nch * sampleSize;
          for (let c = 0; c < nch; c++) {
            out[c][t] = dv.getInt32(base + c * sampleSize, false);
          }
        }
      }
      return out;
    }

    return {
      n_channels: nch,
      sampling_frequency: header.sampling_frequency,
      n_samples: totalSamples,
      duration_s: totalSamples / header.sampling_frequency,
      channel_labels,
      channel_types: new Array(nch).fill('mag'),  // BTi is MEG; refined later
      bytes_per_sample: sampleSize,
      recording_start_iso: null,    // TODO: parse from header timestamp
      annotation_events: [],
      bad_channels: [],
      // Expose a small subset of the parsed header for tests + debug.
      _bti: {
        version: header.version,
        file_type: header.file_type,
        data_format: header.data_format,
        total_epochs: header.total_epochs,
        header_position: headerPosition,
        pdf_url: pdfUrl,
      },
      readWindow,
    };
  };

  // ---- internal helpers --------------------------------------------

  // Try common PDF filenames and return the first that exists. We treat
  // any successful HEAD/Range probe as "exists"; a 404 is the signal to
  // try the next candidate. Local blobs short-circuit to a direct
  // probeLength check (the local registry has no concept of 404).
  async function resolvePdfUrl(inputUrl, HttpRange) {
    // If the input already names one of the known PDF files, use it
    // verbatim — avoids probing the directory listing when the caller
    // already knew the right name (which is what bids-recording.js
    // produces for ext=bti).
    const tail = inputUrl.replace(/\/$/, '').split('/').pop();
    if (PDF_CANDIDATES.includes(tail)) {
      return inputUrl;
    }

    // Otherwise treat `inputUrl` as a sibling file in the bundle (or
    // the bundle dir itself) and try each candidate name.
    const bundleDir = inputUrl.endsWith('/')
      ? inputUrl
      : inputUrl.slice(0, inputUrl.lastIndexOf('/') + 1);
    for (const name of PDF_CANDIDATES) {
      const url = bundleDir + name;
      try {
        // probeLengthNoHead returns the file size when reachable; on
        // 404 / network error it throws (which we swallow to try the
        // next candidate).
        const n = await HttpRange.probeLengthNoHead(url);
        if (n && n > FILE_END_PTR_BYTES) return url;
      } catch (_e) {
        // try next
      }
    }
    throw new Error(
      `bti.open: could not find a PDF file in ${bundleDir} — tried ${PDF_CANDIDATES.join(', ')}`,
    );
  }

  // Read a uint64 big-endian from a DataView at `off`. DataView lacks
  // a native BigInt path until ES2020+; we still want BigInt arithmetic
  // because the BTI pointer can legally span the full 64-bit range.
  function readUint64BE(view, off) {
    const hi = view.getUint32(off, false);
    const lo = view.getUint32(off + 4, false);
    return (BigInt(hi) << 32n) | BigInt(lo);
  }

  // Resolve header_position from the trailing 8-byte pointer (sync
  // path used by api.read). Mirrors the steps in api.open but on a
  // single in-memory buffer.
  function resolveHeaderPosition(view, fileLen) {
    const raw = readUint64BE(view, fileLen - FILE_END_PTR_BYTES);
    let p = Number(raw & BigInt(FILE_MASK));
    if (p % FILE_CURPOS !== 0) p += FILE_CURPOS - (p % FILE_CURPOS);
    return p;
  }

  // Parse the PDF header block. `view` is over [headerPosition, fileLen-8).
  // Field layout cross-checked against /tmp/mne_bti.py:766-848 on the
  // date of authorship.
  function parsePdfHeader(fullView, headerPosition, headerLen, fileLen) {
    // Create a sub-view starting at headerPosition. Use buffer + offset
    // so we don't allocate; ArrayBuffer.prototype.slice would copy.
    const buf = fullView.buffer.slice(
      fullView.byteOffset + headerPosition,
      fullView.byteOffset + headerPosition + headerLen,
    );
    return parsePdfHeaderBlock(new DataView(buf), headerLen);
  }

  function parsePdfHeaderBlock(hv, headerLen) {
    // Fixed prefix — see file-level layout doc.
    if (headerLen < 96) {
      throw new Error(
        `bti: header is ${headerLen}B, need at least 96B for the fixed prefix`,
      );
    }
    const version = hv.getInt16(0, false);
    const file_type = readAscii(hv, 2, 5);
    // +7 pad 1 → cursor at 8
    const data_format = hv.getInt16(8, false);
    if (!DTYPE_INFO[data_format]) {
      throw new Error(
        `bti: unsupported data_format ${data_format} — only 1 (int16), 2 (int32), ` +
          `3 (float32), 4 (float64) are documented (all big-endian)`,
      );
    }
    // acq_mode at +10 — not validated.
    const total_epochs = hv.getInt32(12, false);
    if (!Number.isInteger(total_epochs) || total_epochs < 1) {
      throw new Error(`bti: implausible total_epochs ${total_epochs}`);
    }
    if (total_epochs !== 1) {
      // Multi-epoch / evoked PDFs would need us to honour epoch
      // boundaries when range-fetching — a continuous-only reader
      // would silently merge them. Reject up-front; supporting multi-
      // epoch is tracked as a future enhancement.
      throw new Error(
        `bti: total_epochs=${total_epochs} (multi-epoch / evoked) is not supported ` +
          `by this initial reader — only continuous (total_epochs=1) PDFs load. ` +
          `Epoched/evoked support is tracked as a future enhancement.`,
      );
    }
    // input_epochs at +16, total_events at +20, total_fixed_events at +24
    // — not validated.
    const sample_period = hv.getFloat32(28, false);
    if (!Number.isFinite(sample_period) || sample_period <= 0) {
      throw new Error(`bti: invalid sample_period ${sample_period}`);
    }
    const sampling_frequency = 1 / sample_period;
    // xaxis_label at +32 (16 bytes), total_processes at +48 — skipped.
    const total_chans = hv.getInt16(52, false);
    if (!Number.isInteger(total_chans) || total_chans <= 0 || total_chans > 4096) {
      throw new Error(`bti: implausible total_chans ${total_chans}`);
    }

    // Skip ahead to the epoch record. After the fixed prefix (cursor
    // at 92) MNE calls `_correct_offset` which aligns to 8. 92 mod 8
    // = 4, so the cursor advances by 4 → 96. The epoch record (56
    // bytes) starts at +96.
    if (headerLen < 96 + 56) {
      throw new Error(
        `bti: header is ${headerLen}B but needs at least 152B for the epoch record`,
      );
    }
    const pts_in_epoch = hv.getInt32(96, false);
    if (!Number.isInteger(pts_in_epoch) || pts_in_epoch < 0) {
      throw new Error(`bti: invalid pts_in_epoch ${pts_in_epoch}`);
    }
    // For total_epochs=1 the sum-of-epoch-points reduces to the single
    // epoch's pts_in_epoch. When we lift the multi-epoch restriction
    // we'll iterate 56-byte records here and sum.
    const n_samples = pts_in_epoch;

    const sample_size = DTYPE_INFO[data_format].bytes;

    return {
      version,
      file_type,
      data_format,
      sampling_frequency,
      n_samples,
      n_channels: total_chans,
      sample_size,
      total_epochs,
      header_position: null,    // filled by caller when meaningful
    };
  }

  // Read up to `n` ASCII bytes from `view` at `off`, stopping at the
  // first NUL (BTi pads C strings with NUL like every other binary
  // format in the western hemisphere).
  function readAscii(view, off, n) {
    let s = '';
    for (let i = 0; i < n; i++) {
      const b = view.getUint8(off + i);
      if (b === 0) break;
      s += String.fromCharCode(b);
    }
    return s;
  }

  // Bare-bones channel-label generator used only when ChannelLabels
  // isn't loaded (api.read in a test context that didn't bootstrap
  // the shared helpers). Production callers always go through
  // globalThis.ChannelLabels.indexed.
  function indexedLabelsFallback(n) {
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = 'Ch' + (i + 1);
    return out;
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.BtiReader = api;
})();
