/* ============================================================
   formats/itab.js — minimal ITAB (Chieti ARGOS) MEG reader for
   eegdash-viewer.

   ITAB recordings come as a `.raw` data file plus an associated
   `.raw.mhd` binary header sidecar:

     <basename>_meg.raw       binary header (~291 KB) followed by
                              binary samples — header layout is the
                              same fixed-binary structure documented
                              in `.raw.mhd` (FieldTrip uses ONE parser
                              for both file types).
     <basename>_meg.raw.mhd   sidecar copy of the binary header — same
                              structure, sometimes carrying post-
                              acquisition edits (sensor refits, marker
                              additions). For windowed sample readout
                              we only need the .raw; the .mhd is
                              fetched as a fallback when the .raw
                              fails to disclose its own header.

   The BIDS-MEG appendix at
     https://bids-specification.readthedocs.io/en/stable/appendices/meg-file-formats.html#itab
   describes the `.raw` header as "ASCII"; in practice only the
   first 22 bytes are ASCII (the header identifier "FORMAT: ATB-
   BIOMAGDATA"), and everything past that is fixed-binary little-
   endian integers + floats. FieldTrip's filetype detector
   matches on the "FORMAT: ATB-BIOMAGDATA" prefix to recognise an
   ITAB `.raw`; the same bytes parse cleanly through the binary
   reader. (Verified against /tmp/read_itab_mhd.m,
   /tmp/ft_filetype.m, and /tmp/ft_read_header.m, vendored at
   the time of authorship.)

   Binary layout — fields the reader consumes (all multi-byte
   values are LITTLE-ENDIAN):

     +0       char[10]  stname    Header identifier — first 10
                                  bytes of "FORMAT: ATB-BIOMAGDATA".
                                  Used as a magic-byte gate.
     +684     int32     nchan     Total number of channels.
     +720     int32     data_type 0..2 = BE (HP-PA legacy, rejected)
                                  3 = LE_SHORT  (int16 LE)
                                  4 = LE_LONG   (int32 LE)
                                  5 = LE_FLOAT  (float32 LE)
                                  6..7 RTE_A_* (HP-A900, rejected)
                                  8   ASCII data (rejected)
     +724     float32   smpfq     Sampling frequency in Hz.
     +748     int32     ntpdata   Total samples per channel.
     +85428   int32     start_data Byte offset of data within .raw.
     +85440   int32     isns      Sensor code (e.g. 153 = ARGOS-153) —
                                  kept around for surface metadata,
                                  not gated on.
     +85444   ch[640]   each 328 B — (type, number, label, flag,
                                  amvbit, calib, unit, ncoils, wgt,
                                  positions). Only the first nchan
                                  records are read; the trailing
                                  640-nchan records are skipped.

   Per-channel record layout (inside ch[]):
     +0    uint8    type    (1=ele/EEG, 2=mag/MEG, 4=ele ref,
                             8=mag ref, 16=aux, 32=param, 64=digit,
                             128=flag)
     +4    int32    number
     +8    char[16] label
     +24   uint8    flag    (0=working, 1=noisy, 2=very noisy, 3=broken)
     +28   float32  amvbit  (LSB → mV calibration)
     +32   float32  calib   (mV → unit calibration; per FieldTrip's
                             ft_read_data.m, samples are divided by
                             this value to produce engineering units)
     +36   char[6]  unit
     +44   int32    ncoils
     +48   ... wgt[10] + 10×position(r_s[3]+u_s[3]) (skipped)

   What this reader does NOT yet do (deliberately deferred — the
   viewer auto-scales per channel for display, so the initial port
   skips these and documents them as future enhancements):
     - data_type ∈ {0,1,2,6,7,8} branches (legacy HP-PA big-endian
       and ASCII formats — vanishingly rare in BIDS-MEG datasets).
     - Sensor position / marker / filter metadata blocks (offsets
       295,364 onwards).
     - Bad-channel detection from `ch[i].flag` field — the value is
       read but not surfaced. The viewer already supports a
       `bad_channels` list via the cross-format contract, so this is
       a one-line follow-up.
     - The "EVENT" segments table (`smpl[i]` records, offsets 860..
       82,780) which carries trigger annotations. ITAB events live
       there in real recordings.

   References (vendored, BSD-3-clause):
   - FieldTrip fileio/private/read_itab_mhd.m  — binary header parser
                                                 (source of truth for
                                                 field offsets)
   - FieldTrip fileio/ft_filetype.m            — filetype detection
                                                 (the "FORMAT: ATB-
                                                 BIOMAGDATA" magic)
   - FieldTrip fileio/ft_read_header.m         — itab_raw + itab_mhd
                                                 case (single parser
                                                 for both files)
   - FieldTrip fileio/ft_read_data.m           — itab_raw data branch
                                                 (data_type → endian/
                                                 width dispatch + the
                                                 `dat ./ calib` per-
                                                 channel scaling)
   ============================================================
   Portions derived from FieldTrip — Copyright the FieldTrip
   contributors, BSD-3-clause license. See:
   https://github.com/fieldtrip/fieldtrip/blob/master/LICENSE
   ============================================================ */
(function () {
  'use strict';

  const api = {};

  // ---- offset table (mirrors scripts/make-itab-fixture.mjs) -----------
  // Field offsets enumerated from /tmp/read_itab_mhd.m. ALL multi-byte
  // values are LITTLE-ENDIAN.
  const OFF_STNAME            = 0;     // 10 bytes — header identifier
  const OFF_NCHAN             = 684;   // int32
  const OFF_DATA_TYPE         = 720;   // int32
  const OFF_SMPFQ             = 724;   // float32
  const OFF_NTPDATA           = 748;   // int32
  const OFF_START_DATA        = 85428; // int32
  const OFF_ISNS              = 85440; // int32 (sensor code, e.g. 153)
  const OFF_CH_ARRAY          = 85444; // start of ch[640]
  const CH_RECORD_SIZE        = 328;   // bytes per channel record

  // Per-channel sub-offsets (relative to a ch[c] record's base).
  const CH_OFF_TYPE   = 0;     // uint8
  const CH_OFF_NUMBER = 4;     // int32
  const CH_OFF_LABEL  = 8;     // char[16]
  const CH_OFF_FLAG   = 24;    // uint8
  const CH_OFF_AMVBIT = 28;    // float32
  const CH_OFF_CALIB  = 32;    // float32
  const CH_OFF_UNIT   = 36;    // char[6]

  // Minimum byte span the header probe must cover to expose every scalar
  // the reader cares about + the per-channel array up to a configurable
  // nchan. For the initial probe we fetch enough to expose everything
  // through OFF_CH_ARRAY + 1 record; once we know nchan we may extend.
  const PROBE_INITIAL_BYTES = OFF_CH_ARRAY + CH_RECORD_SIZE;  // 85,772

  // Header identifier — the first 10 bytes of "FORMAT: ATB-BIOMAGDATA",
  // which is also what FieldTrip's ft_filetype.m matches on. We accept
  // either "FORMAT: AT" (the binary-header variant — the modern format)
  // or "[HeaderTyp" (the older `[HeaderType]` text-style variant —
  // explicitly rejected with a clean error so the viewer can fall back
  // to "format not yet supported"). Both are case-sensitive ASCII per
  // the FieldTrip convention.
  const SIG_BINARY  = 'FORMAT: AT';   // exactly 10 bytes
  const SIG_LEGACY  = '[HeaderTyp';   // 10 bytes of `[HeaderType]`

  // data_type codes per /tmp/read_itab_mhd.m's comments. We only support
  // the LE branches (3/4/5); BE legacy + ASCII variants are rejected
  // with a clean error.
  const DATA_TYPE_LE_SHORT = 3;
  const DATA_TYPE_LE_LONG  = 4;
  const DATA_TYPE_LE_FLOAT = 5;

  // Stable error message — referenced by tests + future viewer routing
  // logic. Kept as a string (not Error subclass) for consistency with
  // formats/kriss.js + formats/*.js.
  const ERR_NOT_ITAB =
    'itab: file does not appear to be a valid ITAB .raw recording ' +
    '(expected "FORMAT: ATB-BIOMAGDATA" header identifier in the ' +
    'first 22 bytes)';

  // ---- public API --------------------------------------------------

  /**
   * Parse an ITAB `.raw` (or `.mhd`) ArrayBuffer into a header object.
   * Synchronous entry point exposed for unit + property tests so the
   * parser can be exercised without network. Production `api.open`
   * uses HTTP Range requests instead of materialising the whole file.
   *
   * @param {ArrayBuffer | Uint8Array} buf - the .raw / .mhd file as one buffer.
   * @returns {{
   *   n_channels: number,
   *   sampling_frequency: number,
   *   n_samples: number,
   *   data_type: number,
   *   sample_width: number,
   *   start_data: number,
   *   isns: number,
   *   chs: Array<{ name: string, type: number, calib: number, unit: string }>
   * }}
   * @throws {Error} on any parse failure (truncated file, unsupported
   *   data_type, missing magic bytes).
   */
  api.read = function (buf) {
    if (!buf) {
      throw new Error(
        'itab.read: buffer is required (got ' +
        (buf === null ? 'null' : typeof buf) + ')',
      );
    }
    // Accept either ArrayBuffer or Uint8Array — same convention as
    // formats/kriss.js. Downstream we always work through a DataView,
    // so coerce the ArrayBuffer view here.
    const ab = buf instanceof Uint8Array
      ? buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)
      : buf;
    if (ab.byteLength < PROBE_INITIAL_BYTES) {
      throw new Error(
        `itab.read: buffer too small (${ab.byteLength}B) — need at ` +
        `least ${PROBE_INITIAL_BYTES}B to expose the header + first ` +
        `channel record`,
      );
    }
    return parseHeaderBuffer(ab);
  };

  /**
   * Heuristic "does this look like an ITAB .raw header?" check.
   *
   * Reads the first 10 bytes as ASCII and matches against the modern
   * binary signature "FORMAT: AT" (= first 10 chars of "FORMAT: ATB-
   * BIOMAGDATA", what FieldTrip's ft_filetype.m gates on) OR the
   * legacy text signature "[HeaderTyp" (= first 10 chars of
   * "[HeaderType]").
   *
   * Returns:
   *   'binary'  — modern binary header (this reader supports it)
   *   'legacy'  — older `[HeaderType]` text variant (not supported)
   *   null      — neither signature found
   *
   * @param {Uint8Array} u8
   * @returns {'binary'|'legacy'|null}
   */
  function detectVariant(u8) {
    if (!(u8 instanceof Uint8Array)) u8 = new Uint8Array(u8);
    if (u8.byteLength < 10) return null;
    const head = String.fromCharCode(
      u8[0], u8[1], u8[2], u8[3], u8[4],
      u8[5], u8[6], u8[7], u8[8], u8[9],
    );
    if (head === SIG_BINARY) return 'binary';
    if (head === SIG_LEGACY) return 'legacy';
    return null;
  }

  /**
   * Open an ITAB `.raw` recording for windowed reading. Uses HTTP Range
   * requests for everything — at no point do we materialise the full
   * file in memory (real ITAB MEG recordings carry 153 channels at
   * ~1 kHz, which adds up to tens of GB for long sessions).
   *
   * `meta.eeg_url` must point at the `.raw` file (this is the BIDS
   * convention for ITAB recordings; the `.raw.mhd` sidecar lives
   * alongside it).
   *
   * @param {object} meta - { eeg_url: string, … } as produced by
   *   bids-recording.js or a drag-and-drop bundle.
   * @returns {Promise<object>} reader with the cross-format contract:
   *   n_channels, sampling_frequency, n_samples, duration_s,
   *   channel_labels, channel_types, bytes_per_sample,
   *   readWindow(start, n), readWindowStreaming(start, n).
   */
  api.open = async function (meta) {
    const url = meta && (meta.eeg_url || meta.url);
    if (!url) throw new Error('itab.open: meta.eeg_url is required');

    const HttpRange = globalThis.HttpRange;
    if (!HttpRange) throw new Error('itab.open: globalThis.HttpRange missing');

    // Probe total length so the data-section bounds checks have ground
    // truth. probeLengthNoHead avoids the HEAD-caching anomaly some
    // CDNs exhibit (same reason fiff.js / eeglab.js / kit.js use it).
    const totalBytes = await HttpRange.probeLengthNoHead(url);
    if (totalBytes < PROBE_INITIAL_BYTES) {
      // Range-fetch what we have so detectVariant() can produce a
      // helpful error message ("looks like legacy ITAB" vs "not ITAB
      // at all") instead of a generic "file too small".
      const head = totalBytes >= 10
        ? new Uint8Array(await HttpRange.rangeFetch(url, 0, 9, 10))
        : new Uint8Array(0);
      const variant = detectVariant(head);
      if (variant === 'legacy') {
        throw new Error(legacyError());
      }
      throw new Error(
        `itab.open: file too small (${totalBytes}B) — need at least ` +
        `${PROBE_INITIAL_BYTES}B to expose the header. ` +
        (variant === null ? ERR_NOT_ITAB : '(file is ITAB-shaped but truncated)'),
      );
    }

    // Step 1: fetch enough header to cover every scalar field AND the
    // first per-channel record. We can't know nchan yet, so we fetch a
    // generous initial probe and (if needed) extend.
    const initialBuf = await HttpRange.rangeFetch(
      url, 0, PROBE_INITIAL_BYTES - 1, PROBE_INITIAL_BYTES,
    );
    const initialU8 = new Uint8Array(initialBuf);

    // Step 2: gate on the magic signature BEFORE any byte-arithmetic
    // touches the file. A non-ITAB file (e.g. EDF) would parse
    // arbitrary integers out of the offset table and produce
    // wildly-wrong nchan / sample-rate values; the magic check kills
    // that path early.
    const variant = detectVariant(initialU8);
    if (variant === 'legacy') throw new Error(legacyError());
    if (variant !== 'binary') throw new Error(ERR_NOT_ITAB);

    // Step 3: read nchan from the initial probe. If nchan > 1 we need
    // to re-fetch a longer header so the entire ch[0..nchan-1] array
    // is in memory.
    const v0 = new DataView(initialBuf);
    const nchan = v0.getInt32(OFF_NCHAN, true);
    if (!Number.isInteger(nchan) || nchan <= 0 || nchan > 640) {
      // Real ITAB ch[] array has exactly 640 slots (matching ref_ch[]).
      // 640 is the ceiling per /tmp/read_itab_mhd.m.
      throw new Error(
        `itab.open: implausible nchan ${nchan} — must be in 1..640 ` +
        `(ITAB ch[] array has 640 slots)`,
      );
    }

    let headerU8 = initialU8;
    const neededHeaderBytes = OFF_CH_ARRAY + nchan * CH_RECORD_SIZE;
    if (neededHeaderBytes > PROBE_INITIAL_BYTES) {
      // Fetch only the extra bytes we don't already have, then
      // concatenate so the byte arithmetic stays uniform.
      const extra = await HttpRange.rangeFetch(
        url, PROBE_INITIAL_BYTES, neededHeaderBytes - 1,
        neededHeaderBytes - PROBE_INITIAL_BYTES,
      );
      const merged = new Uint8Array(neededHeaderBytes);
      merged.set(initialU8, 0);
      merged.set(new Uint8Array(extra), PROBE_INITIAL_BYTES);
      headerU8 = merged;
    }
    const headerBuf = headerU8.buffer.slice(
      headerU8.byteOffset, headerU8.byteOffset + headerU8.byteLength,
    );

    const header = parseHeaderBuffer(headerBuf);

    // Step 4: bound-check start_data + data span against the actual
    // file size. A real ITAB file has start_data = 297,664 (full fixed
    // header); our synthetic fixture compresses this. Either is fine
    // as long as start_data clears the byte range we read from header.
    if (header.start_data < OFF_CH_ARRAY + nchan * CH_RECORD_SIZE) {
      // Pathological: start_data would overlap the per-channel array
      // we just read. Real files never do this; defensively reject.
      throw new Error(
        `itab.open: start_data=${header.start_data} overlaps the ` +
        `per-channel header array (ends at ` +
        `${OFF_CH_ARRAY + nchan * CH_RECORD_SIZE})`,
      );
    }
    const expectedDataBytes = header.n_samples * header.n_channels * header.sample_width;
    if (header.start_data + expectedDataBytes > totalBytes) {
      throw new Error(
        `itab.open: data span ${header.start_data}..` +
        `${header.start_data + expectedDataBytes - 1} exceeds file ` +
        `size ${totalBytes} (n_samples=${header.n_samples}, ` +
        `n_channels=${header.n_channels}, sample_width=${header.sample_width})`,
      );
    }

    // Step 5: best-effort fetch of the `.raw.mhd` sidecar. If it
    // exists we cross-check the scalar fields; if it differs we WARN
    // (sidecars sometimes carry post-acquisition edits — sensor refits,
    // updated markers — but the recording-time scalars must agree
    // with the .raw, otherwise the .raw is corrupt). Missing sidecar
    // is fine: BIDS keeps it as a sibling but a drag-dropped .raw can
    // arrive without one.
    try {
      const mhdUrl = url + '.mhd';
      const mhdLen = await HttpRange.probeLengthNoHead(mhdUrl);
      if (mhdLen >= PROBE_INITIAL_BYTES) {
        const mhdProbeLen = Math.min(mhdLen, OFF_NTPDATA + 4);
        const mhdBuf = await HttpRange.rangeFetch(
          mhdUrl, 0, mhdProbeLen - 1, mhdProbeLen,
        );
        const mhdView = new DataView(mhdBuf);
        const mhdNchan   = mhdView.getInt32(OFF_NCHAN,    true);
        const mhdSmpfq   = mhdView.getFloat32(OFF_SMPFQ,  true);
        const mhdNtp     = mhdView.getInt32(OFF_NTPDATA,  true);
        if (mhdNchan !== header.n_channels ||
            Math.abs(mhdSmpfq - header.sampling_frequency) > 1e-3 ||
            mhdNtp !== header.n_samples) {
          console.warn(
            `itab.open: .mhd sidecar disagrees with .raw header — ` +
            `.raw: nchan=${header.n_channels}, sfreq=${header.sampling_frequency}, ` +
            `n_samples=${header.n_samples}; ` +
            `.mhd: nchan=${mhdNchan}, sfreq=${mhdSmpfq}, n_samples=${mhdNtp}. ` +
            `Trusting .raw.`,
          );
        }
      }
    } catch (_) {
      // Sidecar fetch failed (404, network error, no .mhd alongside) —
      // continue with just the .raw header. ITAB MEG works fine without
      // the sidecar for windowed sample readout.
    }

    const channel_labels = header.chs.map(c => c.name);
    const channel_types  = header.chs.map(c => itabTypeToString(c.type));
    // Per-channel scale = 1 / calib (matches FieldTrip's
    // ft_read_data.m: `dat = dat ./ tmp(:,ones(...))` where tmp is the
    // per-channel calib array). Calib of 0 falls back to 1 to avoid
    // division-by-zero (same fallback FieldTrip uses).
    const scales = new Float32Array(header.n_channels);
    for (let c = 0; c < header.n_channels; c++) {
      const calib = header.chs[c].calib;
      scales[c] = (Number.isFinite(calib) && calib !== 0) ? (1 / calib) : 1;
    }

    async function readWindow(startSample, nWin, opts) {
      const win = globalThis.ChannelBuffers.clampWindow(
        startSample, nWin, header.n_samples,
      );
      if (!win) return globalThis.ChannelBuffers.empty(header.n_channels);
      const { start, end } = win;
      const nOut = end - start;

      // Sample[t,c] at byte start_data + (t * nchan + c) * sample_width.
      // Range arithmetic mirrors edf.js's record-major fetch pattern,
      // adapted for ITAB's sample-interleaved layout (cf. kit.js /
      // ctf.js, which do the same thing).
      const sw = header.sample_width;
      const byteStart = header.start_data + start * header.n_channels * sw;
      const byteEnd   = byteStart + nOut * header.n_channels * sw - 1;
      const buf = await HttpRange.rangeFetch(
        url, byteStart, byteEnd, byteEnd - byteStart + 1, opts,
      );

      // Wrap the buffer in the right typed-array view per data_type.
      // ITAB LE data + LE host (our supported targets) means this is a
      // zero-copy view — same bet edf.js + eeglab.js + kit.js make.
      let source;
      if (header.data_type === DATA_TYPE_LE_SHORT) {
        source = new Int16Array(buf);
      } else if (header.data_type === DATA_TYPE_LE_LONG) {
        source = new Int32Array(buf);
      } else {
        // DATA_TYPE_LE_FLOAT — already verified in parseHeaderBuffer.
        source = new Float32Array(buf);
      }
      const out = globalThis.ChannelBuffers.alloc(header.n_channels, nOut);
      globalThis.ChannelDecode.deinterleaveInto(
        out, source, header.n_channels, nOut, scales,
      );
      return out;
    }

    // Streaming variant — funnels through readWindow and yields the
    // whole window at once. Mirrors kit.js's implementation; a future
    // enhancement could honour partial chunks the way edf.js does.
    async function* readWindowStreaming(startSample, nWin, opts) {
      const data = await readWindow(startSample, nWin, opts);
      yield { offset: 0, data };
    }

    return {
      n_channels:          header.n_channels,
      sampling_frequency:  header.sampling_frequency,
      n_samples:           header.n_samples,
      duration_s:          header.n_samples / header.sampling_frequency,
      channel_labels,
      channel_types,
      bytes_per_sample:    header.sample_width,
      recording_start_iso: null,  // TODO: parse `time` + `date` fields
                                  // at offsets 656 / 668 in a follow-up.
      annotation_events:   [],    // TODO: surface smpl[] event table.
      bad_channels:        [],    // TODO: surface ch[i].flag > 0.
      // Surface a small subset of the parsed header for tests + debug
      // overlays. Not part of the canonical reader API but harmless to
      // expose alongside it.
      _itab: {
        data_type:  header.data_type,
        start_data: header.start_data,
        isns:       header.isns,
      },
      readWindow,
      readWindowStreaming,
    };
  };

  // ---- internal helpers --------------------------------------------

  // Parse a header buffer that's been verified to be at least
  // PROBE_INITIAL_BYTES long and to carry the binary magic signature.
  // Used by both api.read (synchronous) and api.open (after the range
  // fetches assemble the right span).
  function parseHeaderBuffer(ab) {
    const view = new DataView(ab);
    const u8 = new Uint8Array(ab);

    // Re-validate the magic — api.read() callers may have skipped the
    // detector entirely, and api.open() already gated but we want a
    // belt-and-braces guard so a future refactor can't slip a non-ITAB
    // buffer through.
    const variant = detectVariant(u8);
    if (variant === 'legacy') throw new Error(legacyError());
    if (variant !== 'binary') throw new Error(ERR_NOT_ITAB);

    const nchan = view.getInt32(OFF_NCHAN, true);
    if (!Number.isInteger(nchan) || nchan <= 0 || nchan > 640) {
      throw new Error(
        `itab: implausible nchan ${nchan} — must be in 1..640 ` +
        `(ch[] array has 640 slots)`,
      );
    }
    if (ab.byteLength < OFF_CH_ARRAY + nchan * CH_RECORD_SIZE) {
      throw new Error(
        `itab: header buffer too small for ${nchan} channels — ` +
        `need ${OFF_CH_ARRAY + nchan * CH_RECORD_SIZE}B, got ${ab.byteLength}B`,
      );
    }
    const data_type = view.getInt32(OFF_DATA_TYPE, true);
    if (data_type !== DATA_TYPE_LE_SHORT &&
        data_type !== DATA_TYPE_LE_LONG &&
        data_type !== DATA_TYPE_LE_FLOAT) {
      // Reject BE-legacy (0..2), RTE_A_* (6..7), ASCII (8). FieldTrip's
      // ft_read_data.m has the same three-way dispatch; everything else
      // hits "unsupported data_type in itab format".
      throw new Error(
        `itab: unsupported data_type=${data_type} — only LE_SHORT (3), ` +
        `LE_LONG (4), and LE_FLOAT (5) variants are decoded. The big-` +
        `endian HP-PA legacy variants (0..2, 6..7) and ASCII variant ` +
        `(8) are vanishingly rare in BIDS-MEG datasets and not yet ` +
        `supported by this reader.`,
      );
    }
    const sample_width = (data_type === DATA_TYPE_LE_SHORT) ? 2 : 4;

    const smpfq = view.getFloat32(OFF_SMPFQ, true);
    if (!Number.isFinite(smpfq) || smpfq <= 0) {
      throw new Error(`itab: invalid sampling frequency ${smpfq}`);
    }
    const n_samples = view.getInt32(OFF_NTPDATA, true);
    if (!Number.isInteger(n_samples) || n_samples < 0) {
      throw new Error(`itab: invalid n_samples ${n_samples}`);
    }
    const start_data = view.getInt32(OFF_START_DATA, true);
    if (!Number.isInteger(start_data) || start_data < 0) {
      throw new Error(`itab: invalid start_data ${start_data}`);
    }
    const isns = view.getInt32(OFF_ISNS, true);  // informational only

    const chs = new Array(nchan);
    for (let c = 0; c < nchan; c++) {
      const base = OFF_CH_ARRAY + c * CH_RECORD_SIZE;
      const type   = view.getUint8(base + CH_OFF_TYPE);
      // Don't gate on type — real recordings carry a mix of types
      // (MEG + EEG + reference + trigger). The viewer's channel_types
      // surface lets the renderer color-code them.
      const number = view.getInt32(base + CH_OFF_NUMBER, true);
      // label is char[16] — null-terminate at the first NUL.
      const label = readAsciiCString(u8, base + CH_OFF_LABEL, 16) ||
                    `Ch${number > 0 ? number : c + 1}`;
      const calib  = view.getFloat32(base + CH_OFF_CALIB, true);
      const unit   = readAsciiCString(u8, base + CH_OFF_UNIT, 6);
      chs[c] = { name: label, type, calib, unit };
    }

    return {
      n_channels: nchan,
      sampling_frequency: smpfq,
      n_samples,
      data_type,
      sample_width,
      start_data,
      isns,
      chs,
    };
  }

  // Read a NUL-terminated ASCII string from a Uint8Array at [off, off+len).
  // Trims trailing whitespace + control characters because ITAB fixed-
  // width string fields pad with NUL but real-world files sometimes
  // pad with spaces or never reset between rewrites.
  function readAsciiCString(u8, off, len) {
    let end = off;
    const max = Math.min(off + len, u8.length);
    while (end < max && u8[end] !== 0) end++;
    let s = '';
    for (let i = off; i < end; i++) s += String.fromCharCode(u8[i]);
    return s.replace(/[\s\x00-\x1f]+$/, '');
  }

  // Map an ITAB channel `type` byte to the cross-format string the
  // viewer's channel-type column uses. Categories follow
  // /tmp/read_itab_mhd.m's per-type comments.
  function itabTypeToString(type) {
    switch (type) {
      case 1:   return 'eeg';      // ele
      case 2:   return 'mag';      // mag (MEG)
      case 4:   return 'ref';      // ele ref
      case 8:   return 'ref_meg';  // mag ref
      case 16:  return 'misc';     // aux
      case 32:  return 'misc';     // param
      case 64:  return 'stim';     // digit (trigger)
      case 128: return 'stim';     // flag
      default:  return 'misc';
    }
  }

  // Build the legacy-format error message. Kept as a function so the
  // message stays consistent between api.read and api.open without
  // re-defining the string literal.
  function legacyError() {
    return (
      'itab: file uses the legacy "[HeaderType]" text-format variant ' +
      '— only the modern "FORMAT: ATB-BIOMAGDATA" binary variant is ' +
      'supported by this reader. If you have a legacy ITAB recording ' +
      'you would like supported, please open an issue on the ' +
      'eegdash-viewer repository with a small sample file.'
    );
  }

  // Expose internals so tests can pin every code path without grepping
  // for substrings. Stable for future Tier-2 refactors.
  api._detect = detectVariant;
  api._ERR_NOT_ITAB = ERR_NOT_ITAB;

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.ItabReader = api;
})();
