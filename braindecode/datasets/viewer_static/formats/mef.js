/* ============================================================
   formats/mef.js — MEF3 (Multiscale Electrophysiology Format
   version 3) iEEG reader for eegdash-viewer.

   MEF3 is the Mayo Clinic epilepsy iEEG standard. Unlike CTF/KIT
   which interleave channels into a single binary blob, MEF3 stores
   each channel INDEPENDENTLY as its own directory tree:

     <session>.mefd/                    session directory
       <ch1>.timd/                      one time-series channel
         <ch1>-NNNNNN.tmet              metadata (UH + sec1 + sec2 + sec3)
         <ch1>-NNNNNN.tdat              RED-compressed sample blocks
         <ch1>-NNNNNN.tidx              per-block index (start_sample, offset)
       <ch2>.timd/
         ...

   NNNNNN is a 6-digit segment number (000000, 000001, ...).
   This reader only handles **continuous, single-segment, unencrypted**
   recordings — the most common in-the-wild case.

   TIER ACHIEVED: 3 (full sample decode). The reader fetches the
   .tidx file once per channel to build a sample→block map, then
   range-fetches only the RED blocks that overlap the requested
   window. Each block is decoded via formats/_mef-red.js (a literal
   port of meflib's RED_decode), Int32 samples → Float32 with no
   rescaling (the RED codec already restored the original integer
   sample values via scale_factor + detrend).

   Real-world EEGDash datasets aren't MEF3, so this reader is still
   primarily a structure-aware fallback for users who drag .mefd/
   bundles into the viewer — but it now produces real waveforms
   instead of a "format unsupported" placeholder.

   References:
   - Spec: msel-source/meflib (Apache 2.0)
     https://github.com/msel-source/meflib
   - pymef Python bindings (BSD-2-clause)
     https://github.com/MaxvandenBoom/pymef

   ============================================================ */
(function () {
  'use strict';

  const api = {};

  // Universal header is 1024 bytes. Reading the full .tmet file is
  // cheap (16384 bytes total — UH + sec1 1536 + sec2 10752 + sec3 3072)
  // and avoids juggling multiple range reads. .tdat / .tidx can be
  // arbitrarily large, so those we leave to range-fetch on demand.
  const UH_BYTES   = 1024;
  const TMET_BYTES = 16384;

  // Plausibility bound: a real .mefd/ in iEEG contexts has anywhere from
  // 4 to ~256 channels. We accept up to 2048 to leave headroom for
  // research recordings; beyond that something is wrong.
  const MAX_CHANNELS = 2048;

  // TSI entry size (meflib.h L499). One per RED block in the .tdat.
  const TSI_ENTRY_BYTES = 56;

  /**
   * Parse a `.tmet` ArrayBuffer into a per-segment metadata object.
   * Synchronous entry point — exposed so tests can exercise the parser
   * without HTTP. Production `api.open` calls this internally.
   *
   * @param {ArrayBuffer | Uint8Array} buf - one .tmet file
   * @returns {object} segment metadata (see _mef-segment.js parseTmet)
   * @throws {Error} on any parse failure — never returns null.
   */
  api.read = function (buf) {
    if (!globalThis.MefSegment) {
      throw new Error('mef.read: globalThis.MefSegment missing — load formats/_mef-segment.js first');
    }
    return globalThis.MefSegment.parseTmet(buf);
  };

  /**
   * Open a MEF3 `.mefd/` recording for windowed reading.
   *
   * `meta.eeg_url` must point at the bundle DIRECTORY (e.g.
   * `…/sub-01_ses-01_task-rest_ieeg.mefd/`). The trailing slash is
   * optional but the URL must resolve as a directory (the viewer's
   * controller routes .mefd/ extensions here).
   *
   * Alternatively, the caller can pass `meta.channel_urls` — a pre-
   * resolved array of `<channel>.timd/` directory URLs — to bypass
   * the listing step. This is how production callers (which can list
   * a remote directory via a manifest) wire it up.
   *
   * @param {object} meta
   * @param {string} meta.eeg_url - .mefd/ directory URL
   * @param {string[]} [meta.channel_urls] - pre-resolved .timd/ URLs
   * @param {string[]} [meta.segment_urls] - pre-resolved {.tmet, .tdat,
   *   .tidx} URL triples, one per channel. When supplied, no directory
   *   listing is needed.
   * @returns {Promise<object>} reader matching the cross-format contract
   */
  api.open = async function (meta) {
    const HttpRange = globalThis.HttpRange;
    if (!HttpRange) throw new Error('mef.open: globalThis.HttpRange missing');

    const MefSegment = globalThis.MefSegment;
    if (!MefSegment) throw new Error('mef.open: globalThis.MefSegment missing — load formats/_mef-segment.js first');

    const sessionUrl = meta && (meta.eeg_url || meta.url);
    if (!sessionUrl && !meta.segment_urls) {
      throw new Error('mef.open: meta.eeg_url is required (point at <session>.mefd/)');
    }

    // Resolve which segment .tmet/.tdat/.tidx URL triples to read. The
    // viewer + controller hand us either:
    //   (a) meta.segment_urls = [{ tmet, tdat, tidx, channel_dir }, ...]
    //       — pre-listed bundle, used by tests + future remote listing
    //   (b) meta.eeg_url = bundle dir + an HttpRange.listDir function
    //       — for file:// roots or CDNs that expose directory indexes
    // We attempt (a) first; if absent, we try (b).
    let segmentTriples = meta.segment_urls;
    if (!segmentTriples) {
      segmentTriples = await listSegmentsFromDirectory(sessionUrl, HttpRange);
    }
    if (!Array.isArray(segmentTriples) || segmentTriples.length === 0) {
      throw new Error('mef.open: could not resolve any .timd channel segments from the bundle');
    }
    if (segmentTriples.length > MAX_CHANNELS) {
      throw new Error(
        `mef.open: ${segmentTriples.length} channels exceeds the safety bound ` +
        `${MAX_CHANNELS} — refusing to load`,
      );
    }

    // Fetch all .tmet files in parallel — each is at most 16 KiB so the
    // total payload is bounded by MAX_CHANNELS * 16 KiB = 32 MiB worst
    // case. Real iEEG recordings have ~64-256 channels → 1-4 MiB.
    const tmetBuffers = await Promise.all(
      segmentTriples.map((t) => HttpRange.fetchBuffer(t.tmet)),
    );

    // Parse each .tmet. Reject if any channel is encrypted, or if
    // channels disagree on sample rate / length (we don't yet handle
    // ragged MEF3 sessions — pymef itself emits a warning in that case).
    const channelMeta = tmetBuffers.map((b, idx) => {
      try {
        return MefSegment.parseTmet(b);
      } catch (e) {
        throw new Error(
          `mef.open: channel ${idx} (${segmentTriples[idx].tmet}) failed to parse: ${e.message}`,
        );
      }
    });

    const sfreq0 = channelMeta[0].sampling_frequency;
    const nsamp0 = channelMeta[0].n_samples;
    for (let i = 1; i < channelMeta.length; i++) {
      if (channelMeta[i].sampling_frequency !== sfreq0) {
        throw new Error(
          `mef.open: channel ${i} sample rate ${channelMeta[i].sampling_frequency} ` +
          `differs from channel 0 ${sfreq0} — ragged MEF3 sessions are not supported`,
        );
      }
      if (channelMeta[i].n_samples !== nsamp0) {
        throw new Error(
          `mef.open: channel ${i} length ${channelMeta[i].n_samples} ` +
          `differs from channel 0 ${nsamp0} — ragged MEF3 sessions are not supported`,
        );
      }
    }

    const channel_labels = channelMeta.map((m, idx) => {
      // Prefer the channel_name carried in the universal header. Falls
      // back to indexed labels if the field is empty (which happens
      // when a write path failed to populate it — rare but seen in the
      // wild).
      const nm = m.universal_header.channel_name;
      return (nm && nm.length > 0) ? nm : ('Ch' + (idx + 1));
    });

    const n_channels = channelMeta.length;

    // Recording start ISO — μUTC stored in universal_header.start_time.
    // μUTC is microseconds since the Unix epoch. We convert to ms (the
    // input the JS Date constructor accepts). If the field is the
    // UUTC_NO_ENTRY sentinel (0x8000000000000000) it'll appear as a
    // very large negative number after Number() conversion — we treat
    // that as "no time" and emit null.
    let recording_start_iso = null;
    const startUUTC = channelMeta[0].universal_header.start_time;
    // UUTC_NO_ENTRY = 0x8000000000000000 = INT64_MIN ≈ -9.22e18 after
    // BigInt → Number conversion. Plausible recording times sit well
    // above zero (post-1970) and below ~2e15 μs (year 2033 or so).
    if (Number.isFinite(startUUTC) && startUUTC > 0 && startUUTC < 1e18) {
      const ms = startUUTC / 1000;
      const d = new Date(ms);
      if (!isNaN(d.getTime())) recording_start_iso = d.toISOString();
    }

    const MefRed = globalThis.MefRed;
    if (!MefRed) {
      throw new Error('mef.open: globalThis.MefRed missing — load formats/_mef-red.js first');
    }

    // Per-channel cache of the parsed .tidx (array of block descriptors).
    // Built lazily on the first readWindow call to avoid paying ~N_channels
    // extra HTTP round-trips when the caller only wants metadata.
    const ChannelBuffers = globalThis.ChannelBuffers;
    if (!ChannelBuffers) {
      throw new Error('mef.open: globalThis.ChannelBuffers missing — load formats/_buffers.js first');
    }
    /** @type {Array<Array<{file_offset: number, start_sample: number, number_of_samples: number, block_bytes: number}> | null>} */
    const tidxCache = new Array(n_channels).fill(null);

    /**
     * Fetch + parse one channel's .tidx, returning the array of block
     * descriptors. Cached for subsequent calls. We always fetch the full
     * .tidx because it's small (≈ 56 bytes × n_blocks); for a 1-hour
     * 1 kHz recording with 1-second blocks that's 56 × 3600 = 200 KiB.
     */
    async function loadTidx(chIdx) {
      if (tidxCache[chIdx]) return tidxCache[chIdx];
      const tidxUrl = segmentTriples[chIdx].tidx;
      const buf = await HttpRange.fetchBuffer(tidxUrl);
      const bytes = new Uint8Array(buf);
      // Validate the .tidx universal header — magic + LE byte order.
      const uh = MefSegment.parseUniversalHeader(bytes);
      if (uh.file_type !== 'tidx') {
        throw new Error(
          `mef.readWindow: ${tidxUrl} magic=${JSON.stringify(uh.file_type)} ` +
          `(expected "tidx")`,
        );
      }
      const declared = channelMeta[chIdx].n_blocks;
      // .tidx body is densely-packed 56-byte entries after the 1024-byte UH.
      const indexRegion = (buf.byteLength - UH_BYTES);
      const expectedBytes = declared * TSI_ENTRY_BYTES;
      if (indexRegion < expectedBytes) {
        throw new Error(
          `mef.readWindow: .tidx ${tidxUrl} body is ${indexRegion} bytes ` +
          `but .tmet declares ${declared} blocks × ${TSI_ENTRY_BYTES} = ${expectedBytes}`,
        );
      }
      const dv = new DataView(buf, UH_BYTES, expectedBytes);
      const entries = new Array(declared);
      let runningEnd = 0;   // track the prior block's [start..end) for sanity
      for (let b = 0; b < declared; b++) {
        const e = MefSegment.parseTidxEntry(dv, b * TSI_ENTRY_BYTES);
        // Sanity: start_sample must be non-decreasing across the segment.
        if (b > 0 && e.start_sample < runningEnd - entries[b - 1].number_of_samples) {
          throw new Error(
            `mef.readWindow: .tidx ${tidxUrl} block ${b} start_sample=${e.start_sample} ` +
            `precedes block ${b-1} start_sample=${entries[b-1].start_sample}`,
          );
        }
        entries[b] = e;
        runningEnd = e.start_sample + e.number_of_samples;
      }
      tidxCache[chIdx] = entries;
      return entries;
    }

    /**
     * Find the index of the first block whose sample range overlaps
     * `startSample`. The TSI table is monotonic in start_sample, so a
     * binary search is correct + fast. Returns 0 if startSample is
     * before block 0 (the reader will simply emit leading samples from
     * block 0).
     */
    function findStartBlock(blocks, startSample) {
      let lo = 0, hi = blocks.length;
      while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        const m = blocks[mid];
        const blockEnd = m.start_sample + m.number_of_samples;
        if (blockEnd <= startSample) lo = mid + 1;
        else hi = mid;
      }
      // Clamp into valid range — startSample past EOF returns
      // blocks.length, caller handles via clampWindow.
      return Math.min(lo, Math.max(blocks.length - 1, 0));
    }

    /**
     * Read [startSample, startSample+nWin) samples from every channel.
     * Returns one Float32Array per channel (length nWin clamped to the
     * recording's actual sample count — see ChannelBuffers.clampWindow).
     *
     * RED blocks decode to int32; we widen to float32 with no scaling
     * because the RED codec has already applied scale_factor + detrend.
     * Sentinel sample values (RED_NAN, ±INFINITY) are passed through as
     * the corresponding finite float values; the renderer is responsible
     * for filtering if it cares (most do not — sentinels appear only in
     * corrupted recordings).
     *
     * @param {number} startSample
     * @param {number} nWin
     * @param {object} [opts] - forwarded to HttpRange.rangeFetch
     * @returns {Promise<Float32Array[]>} one array per channel
     */
    async function readWindow(startSample, nWin, opts) {
      const win = ChannelBuffers.clampWindow(startSample, nWin, nsamp0);
      if (!win) return ChannelBuffers.empty(n_channels);
      const { start, end, nWin: actualWin } = win;

      const out = ChannelBuffers.alloc(n_channels, actualWin);

      // Load tidx in parallel for all channels we haven't seen yet.
      await Promise.all(
        segmentTriples.map((_, idx) => loadTidx(idx)),
      );

      // Per channel: find overlapping blocks, range-fetch them, decode,
      // copy the window slice. Each channel's I/O is independent so we
      // run them in parallel.
      await Promise.all(channelMeta.map(async (_meta, chIdx) => {
        const blocks = tidxCache[chIdx];
        const tdatUrl = segmentTriples[chIdx].tdat;
        const startBlockIdx = findStartBlock(blocks, start);
        const target = out[chIdx];
        let writeP = 0;
        // The .tdat file layout: 1024-byte UH + concatenated blocks.
        // Per meflib.c (write_mef_ts_data_and_indices, L921
        // `file_offset = UNIVERSAL_HEADER_BYTES;`) the TSI file_offset
        // is ABSOLUTE — measured from byte 0 of the .tdat file, so block
        // 0 sits at offset 1024 (just past the universal header).
        //
        // Older synthetic fixtures (pre-fix make-mef-fixture.mjs) wrote
        // RELATIVE offsets where block 0 sits at file_offset=0. We
        // decide per channel based on the first block: if block-0's
        // file_offset is 0 the whole channel is relative — every entry
        // gets rebased by UH_BYTES. Subsequent blocks in a relative
        // channel cannot be re-detected individually (their offsets fall
        // back into the valid absolute range).
        const isRelativeOffsets = blocks.length > 0 && blocks[0].file_offset === 0;
        for (let bi = startBlockIdx; bi < blocks.length && writeP < actualWin; bi++) {
          const b = blocks[bi];
          // If we're past the requested window, stop. Theoretically
          // findStartBlock guarantees b.start_sample <= start, but for
          // subsequent iterations b.start_sample >= end means we're done.
          if (b.start_sample >= end) break;
          const byteStart = isRelativeOffsets
            ? (UH_BYTES + b.file_offset)
            : b.file_offset;
          const byteEnd   = byteStart + b.block_bytes - 1;
          const blockBuf  = await HttpRange.rangeFetch(
            tdatUrl, byteStart, byteEnd, b.block_bytes, opts,
          );
          const blockBytes = new Uint8Array(blockBuf);
          // decodeBlock takes the buffer + an in-buffer offset. We've
          // fetched exactly one block so offset=0.
          const { samples } = MefRed.decodeBlock(blockBytes, 0);
          // Map the [b.start_sample, b.start_sample+N) range into the
          // requested [start, end) window:
          //   srcOffset = max(0, start - b.start_sample)
          //   dstOffset = max(0, b.start_sample - start)
          //   copyLen   = min(samples.length - srcOffset, actualWin - dstOffset,
          //                   end - max(start, b.start_sample))
          const srcOffset = Math.max(0, start - b.start_sample);
          const dstOffset = Math.max(0, b.start_sample - start);
          const blockSamplesRemaining = samples.length - srcOffset;
          const windowSamplesRemaining = actualWin - dstOffset;
          const copyLen = Math.min(blockSamplesRemaining, windowSamplesRemaining);
          if (copyLen <= 0) continue;
          // Int32 → Float32 copy.
          for (let i = 0; i < copyLen; i++) {
            target[dstOffset + i] = samples[srcOffset + i];
          }
          writeP = dstOffset + copyLen;
        }
      }));

      return out;
    }

    return {
      n_channels,
      sampling_frequency:  sfreq0,
      n_samples:           nsamp0,
      duration_s:          nsamp0 / sfreq0,
      channel_labels,
      channel_types:       new Array(n_channels).fill('ieeg'),
      bytes_per_sample:    4,         // RED decodes to si4 internally
      recording_start_iso,
      annotation_events:   [],        // .rdat record files not yet parsed
      bad_channels:        [],
      // Surface a small subset of the parsed header for tests + debug
      // overlays. Not part of the canonical reader API.
      _mef: {
        tier:              3,
        channels:          channelMeta.map((m, idx) => ({
          name:                m.universal_header.channel_name || ('Ch' + (idx + 1)),
          channel_name:        m.universal_header.channel_name || ('Ch' + (idx + 1)),
          segment_number:      m.universal_header.segment_number,
          n_blocks:            m.n_blocks,
          maximum_block_bytes: m.maximum_block_bytes,
          mef_version:         `${m.universal_header.mef_version_major}.${m.universal_header.mef_version_minor}`,
          // Resolved URLs — handy for the BIDS layout assertion + debug
          // overlays. tmet_url ends in `.segd/<ch>-NNNNNN.tmet` when the
          // recording uses the canonical layout, or `<ch>-NNNNNN.tmet`
          // directly under .timd/ when produced by legacy flat fixtures.
          tmet_url:            segmentTriples[idx].tmet,
          tdat_url:            segmentTriples[idx].tdat,
          tidx_url:            segmentTriples[idx].tidx,
        })),
      },
      readWindow,
    };
  };

  // ---- helpers -----------------------------------------------------

  /**
   * Discover the .timd/ channel sub-directories of a .mefd/ bundle and
   * resolve the .tmet/.tdat/.tidx URL triples for each one. Uses an
   * HttpRange.listDir hook if available; falls back to throwing because
   * directory listing isn't part of the generic Range interface.
   *
   * Production wires this through a controller-supplied manifest. Tests
   * install their own listDir on the local HttpRange shim.
   *
   * Two on-disk layouts are accepted (single-segment scope only):
   *
   *   Flat (synthesised fixtures, pre-1.4 pymef):
   *     <ch>.timd/<ch>-NNNNNN.{tmet,tdat,tidx}
   *
   *   Real BIDS / pymef 1.4+ (spec-canonical):
   *     <ch>.timd/<ch>-NNNNNN.segd/<ch>-NNNNNN.{tmet,tdat,tidx}
   *
   * Detection rule: list <ch>.timd/. If any entry ends in `.segd`, we
   * descend exactly one level into the first `.segd/` we find. Otherwise
   * we look for the triple inline (flat layout).
   *
   * @param {string} sessionUrl
   * @param {object} HttpRange
   * @returns {Promise<Array<{ tmet: string, tdat: string, tidx: string, channel_dir: string }>>}
   */
  async function listSegmentsFromDirectory(sessionUrl, HttpRange) {
    if (typeof HttpRange.listDir !== 'function') {
      throw new Error(
        'mef.open: HttpRange.listDir is not available — pass meta.segment_urls ' +
        'with pre-resolved {.tmet, .tdat, .tidx} triples',
      );
    }

    // Normalise trailing slash so URL string arithmetic works.
    const sessionDir = sessionUrl.endsWith('/') ? sessionUrl : (sessionUrl + '/');
    const sessionEntries = await HttpRange.listDir(sessionDir);
    if (!Array.isArray(sessionEntries)) {
      throw new Error(`mef.open: listDir(${sessionDir}) did not return an array`);
    }
    // Each .timd/ entry contains exactly one segment in the continuous-
    // single-segment case we support. Sort the channels by name so the
    // reader yields a stable channel order (the order on disk is
    // filesystem-dependent and unspecified by the MEF3 spec).
    const channelDirs = sessionEntries
      .filter((name) => /\.timd\/?$/.test(name))
      .map((name) => name.replace(/\/$/, ''))
      .sort();

    const triples = [];
    for (const chName of channelDirs) {
      const chDir = sessionDir + chName + '/';
      const chEntries = await HttpRange.listDir(chDir);
      if (!Array.isArray(chEntries)) {
        throw new Error(`mef.open: listDir(${chDir}) did not return an array`);
      }

      // Detect layout: BIDS-spec recordings nest the segment files inside
      // a `<ch>-NNNNNN.segd/` subdirectory. Sort alphabetically so segment
      // 000000 is preferred over later segments (we only read the first).
      const segdEntries = chEntries
        .filter((n) => /\.segd\/?$/.test(n))
        .map((n) => n.replace(/\/$/, ''))
        .sort();

      let lookupDir = chDir;
      let lookupEntries = chEntries;
      if (segdEntries.length > 0) {
        const segdName = segdEntries[0];
        lookupDir = chDir + segdName + '/';
        lookupEntries = await HttpRange.listDir(lookupDir);
        if (!Array.isArray(lookupEntries)) {
          throw new Error(`mef.open: listDir(${lookupDir}) did not return an array`);
        }
      }

      // Find one matching trio. We accept only the first segment found
      // (initial scope: single-segment recordings).
      const tmet = lookupEntries.find((n) => /\.tmet$/.test(n));
      const tdat = lookupEntries.find((n) => /\.tdat$/.test(n));
      const tidx = lookupEntries.find((n) => /\.tidx$/.test(n));
      if (!tmet || !tdat || !tidx) {
        throw new Error(
          `mef.open: channel ${chName} is missing one of {.tmet, .tdat, .tidx} in ${lookupDir} ` +
          `(found tmet=${tmet || '-'}, tdat=${tdat || '-'}, tidx=${tidx || '-'})`,
        );
      }
      triples.push({
        channel_dir: lookupDir,
        tmet: lookupDir + tmet,
        tdat: lookupDir + tdat,
        tidx: lookupDir + tidx,
      });
    }
    return triples;
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.MefReader = api;
})();
