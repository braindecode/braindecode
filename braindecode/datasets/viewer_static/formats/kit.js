/* ============================================================
   formats/kit.js — minimal KIT/Yokogawa/Ricoh MEG reader for
   eegdash-viewer.

   Handles both `.con` and `.sqd` extensions (identical binary
   format — the BIDS-MEG appendix lists `.con` and `.sqd` as the
   two file-extension forms a KIT recording can take, with no
   on-disk distinction).

   Binary format (verified against MNE-Python's mne/io/kit/kit.py
   `get_kit_info` + mne/io/kit/constants.py — both vendored to
   /tmp/kit_kit.py + /tmp/kit_constants.py — at the time of
   authorship). ALL multi-byte values are LITTLE-ENDIAN on disk.

   File layout:
     0  : directory table — 16 bytes per entry × N entries.
          Entry 0 is DIR_INDEX_DIR; its `count` field carries N.
          Each entry is (uint32 offset, int32 size, int32 max_count,
          int32 count).
     dirs[1].offset  : SYSTEM block — version/revision/sysid +
                       per-system metadata + ADC parameters.
     dirs[8].offset  : ACQ_COND block — acq_type, sfreq, n_samples
                       (we only support acq_type=1 / CONTINUOUS;
                       evoked/epoched files are rejected with a
                       clean error in this initial reader).
     dirs[9].offset  : RAW_DATA — interleaved per-sample, per-channel
                       signed integers (sample_width = adc_allocated/8;
                       typically 2 = int16 LE, occasionally 4 = int32 LE).

   What this reader does NOT yet do (deliberately deferred — the
   viewer auto-scales per channel for display, so the initial port
   skips these and documents them as future enhancements):
     - Per-channel calibration from DIR_INDEX_CALIBRATION (dirs[5]).
     - Amplifier-gain table from DIR_INDEX_AMP_FILTER (dirs[7]) for
       full Tesla-scale unit conversion on MEG channels.
     - Channel name table from DIR_INDEX_CHANNELS (dirs[4]) — for
       now we generate Ch1..ChN via ChannelLabels.indexed(). The
       per-channel record layout is system-dependent and parsing
       it pulls in the FLL settings table; defer to a follow-up.
     - Epoched / Evoked acquisition modes (acq_type ∈ {2, 3}).

   The simple `ad_to_volt = adc_range / 2^adc_stored` factor is
   applied so Float32 samples come out in volts (or volt-scaled
   linear units for MEG — the viewer's auto-scale takes it from
   there). This matches what `np.fromfile(fid, dtype=sqd["dtype"])
   * conv_factor` produces in MNE-Python's `_read_segment_file`
   (mne/io/kit/kit.py lines 203-225), minus the per-channel
   calibration step we're deferring.

   References (vendored, BSD-3-clause):
   - mne/io/kit/kit.py        get_kit_info / _read_dir / _read_dirs
   - mne/io/kit/constants.py  KIT.* (CONTINUOUS, DIR_INDEX_*)

   ============================================================
   Portions derived from MNE-Python — Copyright the MNE-Python
   contributors, BSD-3-clause license. See:
   https://github.com/mne-tools/mne-python/blob/main/LICENSE.txt
   ============================================================ */
(function () {
  'use strict';

  const api = {};

  // Directory entry size — 4 × 4-byte fields (offset, size, max_count,
  // count). Locked by the on-disk format; do not change.
  const DIR_ENTRY_SIZE = 16;

  // KIT.DIR_INDEX_* — verified against /tmp/kit_constants.py lines
  // 243-259. Only the indices we actually dereference are named here;
  // the file may contain more entries we ignore.
  const DIR_INDEX_DIR       = 0;
  const DIR_INDEX_SYSTEM    = 1;
  const DIR_INDEX_ACQ_COND  = 8;
  const DIR_INDEX_RAW_DATA  = 9;

  // KIT.CONTINUOUS (= 1) — the only acq_type we accept. CONST mirrors
  // /tmp/kit_constants.py line 139.
  const ACQ_TYPE_CONTINUOUS = 1;

  // Per-channel byte offsets inside the SYSTEM block, derived from the
  // exact sequence of reads in get_kit_info (kit.py:535-575):
  //   +0   int32  version
  //   +4   int32  revision
  //   +8   int32  sysid
  //   +12  char[128]  system_name
  //   +140 char[128]  model_name
  //   +268 int32  nchan
  //   +272 char[256]  comment
  //   +528 int32  create_time
  //   +532 int32  last_modified
  //   +536 int32[3]  reserved
  //   +548 int32  dewar_style
  //   +552 int32[3]  spare
  //   +564 int32  fll_type
  //   +568 int32[3]  spare
  //   +580 int32  trigger_type
  //   +584 int32[3]  spare
  //   +596 int32  adboard_type
  //   +600 int32[29] reserved
  //   +716 (V<=2.3) int32 adc_range OR (>V2R3) float64 adc_range
  //        then immediately:
  //        int32  adc_polarity
  //        int32  adc_allocated
  //        int32  adc_stored
  // We only read the fields the viewer needs: nchan, adc_range,
  // adc_allocated, adc_stored — plus version/revision for the format
  // gate.
  const OFF_VERSION       = 0;
  const OFF_REVISION      = 4;
  const OFF_NCHAN         = 268;
  const OFF_ADC_RANGE     = 716;

  // ACQ_COND continuous block (kit.py:712-718):
  //   +0   int32   acq_type
  //   +4   float64 sfreq
  //   +12  int32   samples_count  (skipped)
  //   +16  int32   n_samples
  const OFF_ACQ_TYPE      = 0;
  const OFF_ACQ_SFREQ     = 4;
  const OFF_ACQ_NSAMPLES  = 16;

  // ---- public API --------------------------------------------------

  /**
   * Parse a KIT `.con` / `.sqd` ArrayBuffer into a header object.
   * Synchronous entry point exposed so unit tests and the worker can
   * exercise the parser without a network roundtrip. Production
   * `api.open` uses HTTP Range requests to fetch only the directory
   * + SYSTEM + ACQ_COND blocks (typically < 1 KiB total), so calling
   * `read()` on a full buffer is reserved for tests + drag-drop.
   *
   * @param {ArrayBuffer} buf - the .con / .sqd file as one buffer.
   * @returns {{
   *   n_channels: number,
   *   sampling_frequency: number,
   *   n_samples: number,
   *   adc_range: number,
   *   adc_allocated: number,
   *   adc_stored: number,
   *   sample_width: number,
   *   raw_data_offset: number,
   *   chs: Array<{ name: string }>
   * }}
   * @throws {Error} on any parse failure (truncated file, unsupported
   *   format version, non-continuous acquisition).
   */
  api.read = function (buf) {
    if (!buf || buf.byteLength < DIR_ENTRY_SIZE) {
      throw new Error(
        `kit.read: buffer too small (${buf ? buf.byteLength : 0} bytes) — ` +
        `need at least ${DIR_ENTRY_SIZE}B for the first dir entry`,
      );
    }
    const view = new DataView(buf);
    const dirs = readDirs(view);
    return parseHeaderFromDirs(view, dirs);
  };

  /**
   * Open a KIT `.con` / `.sqd` recording for windowed reading. Uses
   * HTTP Range requests for everything — at no point do we materialise
   * the full file in memory (KIT recordings routinely hit a few GB).
   *
   * @param {object} meta - { eeg_url: string, … } as produced by
   *   bids-recording.js. The URL must point at the `.con` / `.sqd`
   *   file itself (KIT is single-file, not a directory bundle).
   * @returns {Promise<object>} reader with the cross-format contract:
   *   n_channels, sampling_frequency, n_samples, duration_s,
   *   channel_labels, channel_types, bytes_per_sample,
   *   readWindow(start, n), readWindowStreaming(start, n).
   */
  api.open = async function (meta) {
    const url = meta && (meta.eeg_url || meta.url);
    if (!url) throw new Error('kit.open: meta.eeg_url is required');

    const HttpRange = globalThis.HttpRange;
    if (!HttpRange) throw new Error('kit.open: globalThis.HttpRange missing');

    // Probe total length up-front so readWindow's range fetches can
    // validate against it. probeLengthNoHead avoids the HEAD-caching
    // anomaly some CDNs exhibit — same reason fiff.js + eeglab.js
    // both moved off probeLength().
    const totalBytes = await HttpRange.probeLengthNoHead(url);
    if (totalBytes < DIR_ENTRY_SIZE) {
      throw new Error(
        `kit.open: file too small (${totalBytes}B) — need at least ` +
        `${DIR_ENTRY_SIZE}B for the first directory entry`,
      );
    }

    // Step 1: the first 16 bytes give us N (total dir entries).
    const firstEntryBuf = await HttpRange.rangeFetch(
      url, 0, DIR_ENTRY_SIZE - 1, DIR_ENTRY_SIZE,
    );
    const firstEntry = readDirEntry(new DataView(firstEntryBuf), 0);
    const nDirs = firstEntry.count;
    if (!Number.isInteger(nDirs) || nDirs <= 0 || nDirs > 4096) {
      // Bounds check: real KIT files have ~30-60 dir entries (per
      // /tmp/kit_constants.py DIR_INDEX_* peaks at 29). 4096 is a
      // generous ceiling that still refuses obviously corrupt files
      // (where reading e.g. an EEGLAB .set as KIT would parse 'A'…
      // bytes as a huge int32).
      throw new Error(
        `kit.open: implausible directory entry count ${nDirs} — file is ` +
        `probably not a KIT .con/.sqd recording`,
      );
    }
    // We need DIR_INDEX_RAW_DATA (= 9) at minimum.
    if (nDirs <= DIR_INDEX_RAW_DATA) {
      throw new Error(
        `kit.open: directory has only ${nDirs} entries — need at least ` +
        `${DIR_INDEX_RAW_DATA + 1} to reach RAW_DATA`,
      );
    }

    // Step 2: fetch the remaining (N-1) entries in one range.
    const dirTableTotal = nDirs * DIR_ENTRY_SIZE;
    let dirBytes;
    if (nDirs === 1) {
      dirBytes = firstEntryBuf;
    } else {
      const restBuf = await HttpRange.rangeFetch(
        url, DIR_ENTRY_SIZE, dirTableTotal - 1,
        dirTableTotal - DIR_ENTRY_SIZE,
      );
      // Concatenate the first 16 bytes with the rest so the dir parser
      // can walk a single contiguous buffer.
      const combined = new Uint8Array(dirTableTotal);
      combined.set(new Uint8Array(firstEntryBuf), 0);
      combined.set(new Uint8Array(restBuf), DIR_ENTRY_SIZE);
      dirBytes = combined.buffer;
    }
    const dirs = readDirs(new DataView(dirBytes));

    // Step 3: range-fetch the SYSTEM + ACQ_COND blocks. We fetch them
    // separately because their offsets aren't necessarily contiguous
    // (real KIT files interleave channel/calibration data between them).
    const sysDir = dirs[DIR_INDEX_SYSTEM];
    const acqDir = dirs[DIR_INDEX_ACQ_COND];
    const rawDir = dirs[DIR_INDEX_RAW_DATA];

    // The SYSTEM block we need to peek at goes up to offset 736 (adc_stored
    // int32 ends at 732+4=736). We fetch 740 bytes to leave a small margin.
    const SYSTEM_PROBE_BYTES = 740;
    const sysEnd = Math.min(sysDir.offset + SYSTEM_PROBE_BYTES - 1, totalBytes - 1);
    if (sysDir.offset < 0 || sysDir.offset >= totalBytes) {
      throw new Error(`kit.open: SYSTEM dir offset ${sysDir.offset} out of bounds`);
    }
    const sysBuf = await HttpRange.rangeFetch(
      url, sysDir.offset, sysEnd, sysEnd - sysDir.offset + 1,
    );

    // ACQ_COND is exactly 20 bytes for continuous data (acq_type +
    // sfreq + samples_count + n_samples). Fetch 20.
    const ACQ_PROBE_BYTES = 20;
    if (acqDir.offset < 0 || acqDir.offset + ACQ_PROBE_BYTES > totalBytes) {
      throw new Error(`kit.open: ACQ_COND dir offset ${acqDir.offset} out of bounds`);
    }
    const acqBuf = await HttpRange.rangeFetch(
      url, acqDir.offset, acqDir.offset + ACQ_PROBE_BYTES - 1, ACQ_PROBE_BYTES,
    );

    // Parse SYSTEM.
    const sysView = new DataView(sysBuf);
    const version  = sysView.getInt32(OFF_VERSION,  true);
    const revision = sysView.getInt32(OFF_REVISION, true);
    if (version < 2 || (version === 2 && revision < 3)) {
      throw new Error(
        `kit.open: unsupported KIT format version V${version}R${revision} — ` +
        `MNE-Python requires V2R3 or newer`,
      );
    }
    const nchan = sysView.getInt32(OFF_NCHAN, true);
    if (!Number.isInteger(nchan) || nchan <= 0 || nchan > 4096) {
      throw new Error(`kit.open: implausible nchan ${nchan} in SYSTEM block`);
    }
    // adc_range: float64 for V > 2 or V2R > 3 ; int32 for older.
    // Our format gate above already rejected anything below V2R3, so
    // V2R3 is the only int32 case left.
    let adc_range;
    if (version === 2 && revision <= 3) {
      adc_range = sysView.getInt32(OFF_ADC_RANGE, true);
    } else {
      adc_range = sysView.getFloat64(OFF_ADC_RANGE, true);
    }
    if (!Number.isFinite(adc_range) || adc_range <= 0) {
      throw new Error(`kit.open: invalid adc_range ${adc_range}`);
    }
    // The 3 int32s immediately follow adc_range. For V2R<=3 that means
    // they start at OFF_ADC_RANGE + 4 (int32 adc_range); otherwise at
    // OFF_ADC_RANGE + 8 (float64 adc_range).
    const offAfterRange = OFF_ADC_RANGE + (version === 2 && revision <= 3 ? 4 : 8);
    // adc_polarity at offAfterRange — value not consumed (matches the
    // `del adc_polarity` in kit.py:575).
    const adc_allocated = sysView.getInt32(offAfterRange + 4, true);
    const adc_stored    = sysView.getInt32(offAfterRange + 8, true);
    if (adc_allocated <= 0 || adc_allocated % 8 !== 0) {
      throw new Error(
        `kit.open: invalid adc_allocated ${adc_allocated} — must be a positive ` +
        `multiple of 8 (per MNE-Python's assert)`,
      );
    }
    if (adc_stored <= 0 || adc_stored > adc_allocated) {
      throw new Error(`kit.open: invalid adc_stored ${adc_stored} (adc_allocated=${adc_allocated})`);
    }

    // Parse ACQ_COND.
    const acqView = new DataView(acqBuf);
    const acq_type = acqView.getInt32(OFF_ACQ_TYPE, true);
    const sfreq    = acqView.getFloat64(OFF_ACQ_SFREQ, true);
    if (acq_type !== ACQ_TYPE_CONTINUOUS) {
      // Epoched (3) / Evoked (2) KIT files are uncommon in BIDS. Reject
      // with a clean error so the viewer can fall back to a "not yet
      // supported" message instead of returning a half-useful reader.
      // See kit.py:719-727 for the epoched/evoked field layout we'd
      // need to support; deferred to a follow-up.
      throw new Error(
        `kit.open: acq_type=${acq_type} (non-continuous KIT) is not supported ` +
        `by this initial reader — only CONTINUOUS (1) recordings load. ` +
        `Epoched/evoked support is tracked as a future enhancement.`,
      );
    }
    if (!Number.isFinite(sfreq) || sfreq <= 0) {
      throw new Error(`kit.open: invalid sfreq ${sfreq}`);
    }
    const n_samples = acqView.getInt32(OFF_ACQ_NSAMPLES, true);
    if (!Number.isInteger(n_samples) || n_samples < 0) {
      throw new Error(`kit.open: invalid n_samples ${n_samples}`);
    }

    // Per-sample byte width (2 for int16 LE, 4 for int32 LE).
    const sample_width = adc_allocated / 8;
    if (sample_width !== 2 && sample_width !== 4) {
      // 1 byte (int8) and 8 byte (int64) widths aren't documented in
      // the MNE-Python source. Reject up-front rather than silently
      // misdecode.
      throw new Error(
        `kit.open: unsupported sample width ${sample_width}B (adc_allocated=${adc_allocated}) — ` +
        `only int16 (2B) and int32 (4B) widths have been verified`,
      );
    }

    // RAW_DATA must fit within the file.
    const expectedRawBytes = n_samples * nchan * sample_width;
    if (rawDir.offset < 0 || rawDir.offset + expectedRawBytes > totalBytes) {
      throw new Error(
        `kit.open: RAW_DATA span ${rawDir.offset}..${rawDir.offset + expectedRawBytes - 1} ` +
        `exceeds file size ${totalBytes}`,
      );
    }

    // Conversion factor: raw int → volts (or volt-scaled linear unit).
    // ad_to_volt mirrors kit.py:813 — `adc_range / (2**adc_stored)`.
    // For the minimal initial reader we apply this uniformly; per-channel
    // amplifier gain + calibration are deferred (see file header).
    const ad_to_volt = adc_range / Math.pow(2, adc_stored);
    const scales = new Float32Array(nchan);
    for (let c = 0; c < nchan; c++) scales[c] = ad_to_volt;

    const channel_labels = globalThis.ChannelLabels.indexed(nchan);
    // Channel types — KIT files in BIDS are MEG datasets, so default
    // every channel to 'mag'. A future enhancement reading dirs[4]
    // (DIR_INDEX_CHANNELS) would split MEG / TRIGGER / EEG / EXG /
    // MISC per KIT.CHANNEL_* constants.
    const channel_types = new Array(nchan).fill('mag');

    async function readWindow(startSample, nWin, opts) {
      const win = globalThis.ChannelBuffers.clampWindow(startSample, nWin, n_samples);
      if (!win) return globalThis.ChannelBuffers.empty(nchan);
      const { start, end } = win;
      const nOut = end - start;

      // Byte range for [start, end). All multi-byte values are LE per
      // the format; the byte arithmetic mirrors kit.py:216 — `pointer =
      // start * nchan * n_bytes` from dirs[DIR_INDEX_RAW_DATA].offset.
      const byteStart = rawDir.offset + start * nchan * sample_width;
      const byteEnd   = byteStart + nOut * nchan * sample_width - 1;
      const buf = await HttpRange.rangeFetch(
        url, byteStart, byteEnd, byteEnd - byteStart + 1, opts,
      );

      // Decode interleaved samples into per-channel Float32 with the
      // simple ad_to_volt scale. Both int16 and int32 widths route
      // through ChannelDecode.deinterleaveInto — that helper takes an
      // already-typed source array, so we wrap the buffer in the right
      // typed-array view first. KIT is LE; typed arrays use host
      // endianness; the platforms we ship to are LE, so the wrap is
      // a zero-copy view. (This is the same "everywhere is LE" bet
      // edf.js + eeglab.js make for their LE binary payloads.)
      let source;
      if (sample_width === 2) {
        source = new Int16Array(buf);
      } else {
        source = new Int32Array(buf);
      }
      const out = globalThis.ChannelBuffers.alloc(nchan, nOut);
      globalThis.ChannelDecode.deinterleaveInto(out, source, nchan, nOut, scales);
      return out;
    }

    // Streaming variant — chunks the range fetch through
    // rangeFetchStreaming and emits per-chunk samples as soon as bytes
    // arrive. Mirrors the eeglab.js / edf.js pattern. For now we keep
    // this simple by funnelling through the non-streaming readWindow
    // and yielding the whole window at once; a future enhancement
    // could honour partial chunks the way edf.js does.
    async function* readWindowStreaming(startSample, nWin, opts) {
      const data = await readWindow(startSample, nWin, opts);
      // Match the contract used by other readers' streaming variants:
      // yield { offset: 0, data } with `offset` measured in samples.
      yield { offset: 0, data };
    }

    return {
      n_channels:          nchan,
      sampling_frequency:  sfreq,
      n_samples,
      duration_s:          n_samples / sfreq,
      channel_labels,
      channel_types,
      // null (not undefined) so downstream code that does `bids_channels.map`
      // or `bids_channels[0]` after a null-check guards correctly. Every
      // other reader uses this convention; KIT diverged and caused a
      // viewer pageerror on ds004738.
      bids_channels:       (meta && Array.isArray(meta.channels) && meta.channels.length) ? meta.channels : null,
      bytes_per_sample:    sample_width,
      recording_start_iso: null,
      annotation_events:   [],
      bad_channels:        [],
      // Surface a small subset of the parsed header for tests + debug
      // overlays. Not part of the canonical reader API but harmless to
      // expose alongside it.
      _kit: {
        version, revision, adc_range, adc_allocated, adc_stored,
        ad_to_volt, raw_data_offset: rawDir.offset,
      },
      readWindow,
      readWindowStreaming,
    };
  };

  // ---- internal helpers --------------------------------------------

  // Read one 16-byte directory entry at offset `idx * 16` of `view`.
  function readDirEntry(view, idx) {
    const base = idx * DIR_ENTRY_SIZE;
    return {
      offset:    view.getUint32(base + 0,  true),
      size:      view.getInt32 (base + 4,  true),
      max_count: view.getInt32 (base + 8,  true),
      count:     view.getInt32 (base + 12, true),
    };
  }

  // Walk the full directory table. Mirrors _read_dirs in kit.py:493-500:
  // read entry 0 first, then read (entry0.count - 1) more entries. The
  // assertion in MNE that `len(dirs) == dirs[0].count` is enforced
  // implicitly by the loop bounds.
  function readDirs(view) {
    const dirs = [readDirEntry(view, 0)];
    const n = dirs[0].count;
    if (!Number.isInteger(n) || n <= 0) {
      throw new Error(`kit: dirs[0].count = ${n} (must be > 0)`);
    }
    const need = n * DIR_ENTRY_SIZE;
    if (view.byteLength < need) {
      throw new Error(
        `kit: directory table needs ${need} bytes but buffer is ${view.byteLength}`,
      );
    }
    for (let i = 1; i < n; i++) {
      dirs.push(readDirEntry(view, i));
    }
    // Cross-check against MNE's assertion. If the count in entry 0
    // disagrees with the actual number of entries we just read, the
    // file is corrupt or wasn't a KIT file in the first place.
    if (dirs.length !== dirs[DIR_INDEX_DIR].count) {
      throw new Error(
        `kit: dir count mismatch — dirs[0].count=${dirs[DIR_INDEX_DIR].count} ` +
        `vs read=${dirs.length}`,
      );
    }
    return dirs;
  }

  // Pull the minimal SYSTEM + ACQ_COND values from a synchronous
  // buffer that contains the whole file. Used by api.read for tests
  // and (future) drag-drop. Production `api.open` does the same work
  // but against range-fetched sub-buffers without holding the whole
  // file in memory.
  function parseHeaderFromDirs(view, dirs) {
    if (dirs.length <= DIR_INDEX_RAW_DATA) {
      throw new Error(
        `kit.read: directory has ${dirs.length} entries — need at least ` +
        `${DIR_INDEX_RAW_DATA + 1} to reach RAW_DATA`,
      );
    }
    const sysDir = dirs[DIR_INDEX_SYSTEM];
    const acqDir = dirs[DIR_INDEX_ACQ_COND];
    const rawDir = dirs[DIR_INDEX_RAW_DATA];

    const version  = view.getInt32(sysDir.offset + OFF_VERSION,  true);
    const revision = view.getInt32(sysDir.offset + OFF_REVISION, true);
    if (version < 2 || (version === 2 && revision < 3)) {
      throw new Error(
        `kit.read: unsupported KIT format version V${version}R${revision} — ` +
        `MNE-Python requires V2R3 or newer`,
      );
    }
    const nchan = view.getInt32(sysDir.offset + OFF_NCHAN, true);
    let adc_range;
    if (version === 2 && revision <= 3) {
      adc_range = view.getInt32(sysDir.offset + OFF_ADC_RANGE, true);
    } else {
      adc_range = view.getFloat64(sysDir.offset + OFF_ADC_RANGE, true);
    }
    const offAfterRange = sysDir.offset + OFF_ADC_RANGE +
      (version === 2 && revision <= 3 ? 4 : 8);
    const adc_allocated = view.getInt32(offAfterRange + 4, true);
    const adc_stored    = view.getInt32(offAfterRange + 8, true);

    const acq_type = view.getInt32(acqDir.offset + OFF_ACQ_TYPE, true);
    if (acq_type !== ACQ_TYPE_CONTINUOUS) {
      throw new Error(
        `kit.read: acq_type=${acq_type} (non-continuous) is not supported`,
      );
    }
    const sfreq     = view.getFloat64(acqDir.offset + OFF_ACQ_SFREQ, true);
    const n_samples = view.getInt32  (acqDir.offset + OFF_ACQ_NSAMPLES, true);

    const sample_width = adc_allocated / 8;
    const chs = new Array(nchan);
    for (let c = 0; c < nchan; c++) chs[c] = { name: 'Ch' + (c + 1) };

    return {
      n_channels:        nchan,
      sampling_frequency: sfreq,
      n_samples,
      adc_range,
      adc_allocated,
      adc_stored,
      sample_width,
      raw_data_offset:   rawDir.offset,
      chs,
    };
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.KitReader = api;
})();
