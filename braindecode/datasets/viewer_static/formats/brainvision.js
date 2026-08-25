/* ============================================================
   formats/brainvision.js — read BrainVision Core Data Format
   (.vhdr / .eeg / .vmrk) over HTTP Range.

   The header is INI-style text in the `.vhdr` file. The data is
   a flat binary matrix in the `.eeg` file in one of two layouts:

     MULTIPLEXED  (default, sample-major)
       byte offset of sample s, channel c = (s·N + c) · bps
       — same layout as EEGLAB .fdt; readers can deinterleave
       a single contiguous range fetch.

     VECTORIZED  (channel-major, rare)
       byte offset of sample s, channel c = (c·NSAMPLES + s) · bps

   v1 supports only MULTIPLEXED; VECTORIZED would cost N range
   fetches per pan and we haven't seen it in practice.

   Per-channel scaling is a simple scalar `resolution_per_unit`
   (typically µV per integer), applied uniformly per channel.
   No digital min/max gymnastics like EDF.

   We read sidecars first if available (BIDS sources of truth),
   but fall back to the .vhdr's own SamplingInterval and
   [Channel Infos] when sidecars are missing — many real datasets
   (ds002336) inherit `_channels.tsv` only at the dataset root,
   or omit it entirely.
   ============================================================ */
(function () {
  'use strict';

  const api = {};

  // Maps the BrainVision spec's BinaryFormat tag to (a) bytes per
  // sample and (b) the typed-array view we use to read the buffer.
  // All supported formats are little-endian on the host architectures
  // we run; the float case is covered by the host-endianness check
  // in eeglab.js.
  const BIN_FORMATS = {
    INT_16:        { bps: 2, view: Int16Array  },
    UINT_16:       { bps: 2, view: Uint16Array },
    INT_32:        { bps: 4, view: Int32Array  },
    IEEE_FLOAT_32: { bps: 4, view: Float32Array },
  };

  // Permissive INI parser. BrainVision allows `;` line comments,
  // square-bracket section headers, and `key=value` pairs. Section
  // names are lower-cased so callers can index without remembering
  // the original capitalisation.
  /**
   * Parse a BrainVision .vhdr file's INI-flavoured text into a flat key-
   * value map keyed by section.
   * @param {string} text - The full UTF-8 content of the .vhdr file.
   * @returns {Object<string, Object<string, string>>} - Maps section name
   *   (e.g. "Common Infos") to its key/value pairs.
   */
  api.parseIni = function (text) {
    /** @type {Object<string, Object<string, string>>} */
    const sections = {};
    let cur = null;
    for (const raw of text.split(/\r?\n/)) {
      const line = raw.trim();
      if (!line || line.startsWith(';')) continue;
      const sec = /^\[(.+)\]$/.exec(line);
      if (sec) {
        cur = sec[1].trim().toLowerCase();
        sections[cur] = sections[cur] || {};
        continue;
      }
      if (cur == null) continue;
      const eq = line.indexOf('=');
      if (eq < 0) continue;
      sections[cur][line.substring(0, eq).trim()] = line.substring(eq + 1).trim();
    }
    return sections;
  };

  // BrainVision encodes commas inside channel names as `\1`. This is
  // the only escape the spec defines. Restore them after splitting.
  function splitCh(value) {
    return value.split(',').map(p => p.replace(/\\1/g, ',').trim());
  }

  function parseFiniteIntOrNull(v) {
    if (v == null) return null;
    const n = parseInt(v, 10);
    return Number.isFinite(n) ? n : null;
  }

  /**
   * Parse a BrainVision .vhdr file and return the recording metadata.
   *
   * The returned object (loosely typed as `object` because additional
   * derived fields are tacked on after parsing) includes at least:
   *   - n_channels, sampling_frequency, bytes_per_sample,
   *     data_points_declared: number
   *   - sampling_interval: number  (microseconds, raw value from header)
   *   - binary_format, data_orientation, data_format: string
   *   - data_file, marker_file: string  (relative filenames)
   *   - channels: Array of { name, ref, resolution, units, scale }
   *
   * @param {string} text - The full .vhdr text.
   * @returns {object}
   */
  api.parseHeader = function (text) {
    const sec = api.parseIni(text);
    const common = sec['common infos'];
    const binary = sec['binary infos'];
    const channels = sec['channel infos'];
    if (!common || !binary || !channels) {
      throw new Error('.vhdr missing required section: [Common Infos] / [Binary Infos] / [Channel Infos]');
    }

    if (common.DataFormat !== 'BINARY') {
      throw new Error(`v1 only supports DataFormat=BINARY (got "${common.DataFormat}")`);
    }
    if ((common.DataType || 'TIMEDOMAIN') !== 'TIMEDOMAIN') {
      throw new Error(`v1 only supports DataType=TIMEDOMAIN (got "${common.DataType}")`);
    }
    const orientation = (common.DataOrientation || 'MULTIPLEXED').toUpperCase();
    if (orientation !== 'MULTIPLEXED' && orientation !== 'VECTORIZED') {
      throw new Error(`unknown DataOrientation "${orientation}" (expected MULTIPLEXED or VECTORIZED)`);
    }

    const nChannels = parseInt(common.NumberOfChannels, 10);
    if (!Number.isFinite(nChannels) || nChannels <= 0) {
      throw new Error(`Invalid NumberOfChannels: ${common.NumberOfChannels}`);
    }
    const samplingIntervalUs = parseFloat(common.SamplingInterval);
    if (!Number.isFinite(samplingIntervalUs) || samplingIntervalUs <= 0) {
      throw new Error(`Invalid SamplingInterval: ${common.SamplingInterval}`);
    }
    const fs = 1e6 / samplingIntervalUs;

    const binaryFormat = binary.BinaryFormat;
    if (!BIN_FORMATS[binaryFormat]) {
      throw new Error(`Unsupported BinaryFormat "${binaryFormat}" (supported: ${Object.keys(BIN_FORMATS).join(', ')})`);
    }
    const bytesPerSample = BIN_FORMATS[binaryFormat].bps;

    const channelInfos = new Array(nChannels);
    for (let i = 0; i < nChannels; i++) {
      const v = channels[`Ch${i + 1}`];
      if (!v) throw new Error(`[Channel Infos] missing Ch${i + 1}`);
      const parts = splitCh(v);
      const scale = parseFloat(parts[2]);
      channelInfos[i] = {
        name:      parts[0] || `Ch${i + 1}`,
        reference: parts[1] || null,
        // BrainVision spec: empty resolution means "1 in unit".
        // mne also defaults to 1 when missing.
        scale:     Number.isFinite(scale) ? scale : 1,
        unit:      parts[3] || 'µV',
      };
    }

    return {
      data_file: common.DataFile,
      marker_file: common.MarkerFile || null,
      n_channels: nChannels,
      sampling_frequency: fs,
      sampling_interval_us: samplingIntervalUs,
      data_points_declared: parseFiniteIntOrNull(common.DataPoints),
      binary_format: binaryFormat,
      bytes_per_sample: bytesPerSample,
      orientation,
      channels: channelInfos,
    };
  };

  function warnIf(cond, msg) { if (cond) console.warn(msg); }

  api.open = async function (meta) {
    const vhdrUrl = meta.eeg_url;
    const vhdrText = await HttpRange.fetchText(vhdrUrl);
    // Reject macOS AppleDouble (._*) sidecar files early. These are
    // metadata files macOS Finder creates next to real ones; they have
    // the same name with a `._` prefix and the magic bytes 00 05 16 07.
    // Observed in the wild: ds007216 ships `._sub-...vhdr` alongside
    // the real `sub-...vhdr` in the BIDS layout. When the catalog walk
    // picks the AppleDouble URL by mistake, surface a precise message
    // instead of an opaque parse failure.
    if (vhdrText.charCodeAt(0) === 0x00 && vhdrText.charCodeAt(1) === 0x05 &&
        vhdrText.charCodeAt(2) === 0x16 && vhdrText.charCodeAt(3) === 0x07) {
      throw new Error(
        'BrainVision: file looks like a macOS AppleDouble metadata file ' +
        '(magic 00 05 16 07), not a real .vhdr header. Real .vhdr files ' +
        'usually start with "Brain Vision Data Exchange Header File". ' +
        'Try the sibling file without the "._" filename prefix.',
      );
    }
    const hdr = api.parseHeader(vhdrText);

    // For BIDS-pathed sources (OpenNeuro, localdrop), the .eeg lives
    // next to the .vhdr — `new URL(relative, base)` handles absolute,
    // relative, and bare-filename forms uniformly.
    //
    // For SHA-keyed sources (NEMAR), the sibling URL doesn't share a
    // path prefix with the .vhdr, so we look it up in the
    // pre-computed map (keyed by the bare filename the .vhdr's
    // DataFile field carries).
    //
    // Fallback for STALE DataFile: real BIDS datasets (e.g. ds002158)
    // sometimes have a .vhdr whose `DataFile=` still names the
    // original acquisition file (`s2_run1_08062017.eeg`) even though
    // the BIDS curator renamed the sibling on disk to the canonical
    // pattern (`sub-02_..._eeg.eeg`). When the DataFile-derived URL
    // 404s, retry with the same basename as the .vhdr but `.eeg`
    // extension — matches MNE-Python's behaviour in
    // mne/io/brainvision/brainvision.py::_check_paths_for_consistency.
    const primaryEegUrl = meta.sibling_urls?.[hdr.data_file] ??
                          new URL(hdr.data_file, vhdrUrl).href;
    let eegUrl = primaryEegUrl;
    let totalBytes;
    try {
      totalBytes = await HttpRange.probeLength(eegUrl);
    } catch (e) {
      // Detect the 404/missing-file class — but NOT every error type,
      // we want to surface genuine network failures.
      const msg = e && e.message ? e.message : String(e);
      if (!/404|HTTP 4\d\d|Cannot determine length/.test(msg)) throw e;
      const fallbackEegUrl = vhdrUrl.replace(/\.vhdr(\?|$)/i, '.eeg$1');
      if (fallbackEegUrl === primaryEegUrl) throw e;
      try {
        totalBytes = await HttpRange.probeLength(fallbackEegUrl);
        console.warn(
          `BrainVision: DataFile="${hdr.data_file}" 404'd at ${primaryEegUrl}; ` +
          `falling back to vhdr-basename sibling ${fallbackEegUrl}.`,
        );
        eegUrl = fallbackEegUrl;
      } catch {
        // Re-throw the ORIGINAL error with both paths annotated so the
        // user sees both attempts.
        throw new Error(
          `BrainVision: cannot find .eeg sibling. Tried (1) ${primaryEegUrl} ` +
          `from DataFile=${hdr.data_file}, (2) ${fallbackEegUrl} from vhdr basename. ` +
          `Original error: ${msg}`,
        );
      }
    }
    const recordBytes = hdr.n_channels * hdr.bytes_per_sample;
    if (recordBytes === 0) throw new Error('BrainVision: zero-byte sample (n_channels or bps is 0)');
    // Tolerate trailing partial sample — symmetric with the EEGLAB .fdt
    // truncation fix and the EDF/BDF trailing-record fix. Observed on
    // ds003816 (6893875 B vs 128 ch × 4 bps = 512 B per sample, 51 B
    // trailing). The file is likely truncated mid-sample; we floor to
    // complete samples and warn, since the user gets MORE value from
    // viewing the available 13464 samples than from a hard reject.
    const sampleRem = totalBytes % recordBytes;
    if (sampleRem !== 0) {
      console.warn(
        `BrainVision .eeg: file size ${totalBytes}B is not a multiple of ` +
        `n_channels·bps=${recordBytes}B (${sampleRem}B trailing). File ` +
        `likely truncated mid-sample; displaying first ` +
        `${Math.floor(totalBytes / recordBytes)} complete samples ` +
        `(observed on ds003816). If header misreports n_channels or ` +
        `binary_format the render will be wrong — please report.`,
      );
    }
    const nSamples = Math.floor(totalBytes / recordBytes);
    if (nSamples === 0) {
      throw new Error(
        `BrainVision .eeg: file size ${totalBytes}B < record size ${recordBytes}B — no complete sample.`,
      );
    }

    warnIf(hdr.data_points_declared != null && hdr.data_points_declared !== nSamples,
      `.vhdr DataPoints=${hdr.data_points_declared} ≠ derived ${nSamples}; trusting file.`);
    SidecarChecks.crossCheckChannelOrder(
      hdr.channels.map(c => c.name), meta.channels, 'BrainVision');
    SidecarChecks.warnFsMismatch(meta.eeg_json.sampling_frequency, hdr.sampling_frequency, 'BrainVision');

    const scales = new Float64Array(hdr.n_channels);
    const channelLabels = new Array(hdr.n_channels);
    for (let c = 0; c < hdr.n_channels; c++) {
      scales[c] = hdr.channels[c].scale;
      channelLabels[c] = hdr.channels[c].name;
    }

    const layout = {
      url: eegUrl,
      n_channels: hdr.n_channels,
      n_samples: nSamples,
      bytes_per_sample: hdr.bytes_per_sample,
      view_ctor: BIN_FORMATS[hdr.binary_format].view,
      scales,
      orientation: hdr.orientation,
    };

    const isVectorized = hdr.orientation === 'VECTORIZED';
    return {
      n_channels: hdr.n_channels,
      n_samples: nSamples,
      sampling_frequency: hdr.sampling_frequency,
      duration_s: nSamples / hdr.sampling_frequency,
      bytes_per_sample: hdr.bytes_per_sample,
      binary_format: hdr.binary_format,
      data_orientation: hdr.orientation,
      url: eegUrl,
      vhdr_url: vhdrUrl,
      channel_labels: channelLabels,
      bids_channels: meta.channels || null,
      readWindow: isVectorized
        ? (start, n, opts) => readVectorizedWindow(layout, start, n, opts)
        : (start, n, opts) => readMultiplexedWindow(layout, start, n, opts),
      // VECTORIZED has no fast streaming path (each channel needs its own
      // disjoint range); the non-streaming readWindow does N parallel
      // range fetches which is acceptable for typical EEG channel counts.
      readWindowStreaming: isVectorized
        ? undefined
        : (start, n, opts) => streamMultiplexedWindow(layout, start, n, opts),
    };
  };

  // VECTORIZED layout: byte offset of sample s, channel c = (c·N + s)·bps,
  // where N = layout.n_samples (the total sample count). Channel c's
  // samples live in a contiguous [c·N, (c+1)·N) byte block.
  //
  // For a window [start, start+nWin) we issue N parallel range fetches
  // — one per channel, each covering only that channel's nWin samples.
  // For typical EEG (64–128 channels, 10s window @ 500 Hz = 5k samples)
  // each per-channel fetch is ~10–20 KB; total bandwidth is the same as
  // MULTIPLEXED. HTTP/2 multiplexing keeps the 64–128 concurrent fetches
  // light. Observed in the wild: ds003944, ds004000, ds004621, ds007655,
  // ds003947 (all five formerly rejected with "v1 only supports
  // MULTIPLEXED").
  async function readVectorizedWindow(layout, startSample, nWinReq, opts) {
    const win = ChannelBuffers.clampWindow(startSample, nWinReq, layout.n_samples);
    if (!win) return ChannelBuffers.empty(layout.n_channels);
    const { start, nWin } = win;
    const nCh = layout.n_channels;
    const bps = layout.bytes_per_sample;
    const N = layout.n_samples;
    const scales = layout.scales;
    const out = ChannelBuffers.alloc(nCh, nWin);

    // One range fetch per channel, in parallel.
    const fetches = new Array(nCh);
    for (let c = 0; c < nCh; c++) {
      const byteStart = (c * N + start) * bps;
      const byteEnd = byteStart + nWin * bps - 1;
      fetches[c] = HttpRange.rangeFetch(layout.url, byteStart, byteEnd, nWin * bps, opts);
    }
    const bufs = await Promise.all(fetches);
    for (let c = 0; c < nCh; c++) {
      const view = new layout.view_ctor(bufs[c]);
      const dst = out[c];
      const s = scales[c];
      for (let i = 0; i < nWin; i++) dst[i] = view[i] * s;
    }
    return out;
  }

  // Hot path. Single contiguous range fetch, single typed-array view
  // over the buffer chosen at open() time, then a linear walk that
  // deinterleaves + applies per-channel scale in one pass.
  async function readMultiplexedWindow(layout, startSample, nWinReq, opts) {
    const win = ChannelBuffers.clampWindow(startSample, nWinReq, layout.n_samples);
    if (!win) return ChannelBuffers.empty(layout.n_channels);
    const { start, nWin } = win;
    const nCh = layout.n_channels;
    const byteStart = start * nCh * layout.bytes_per_sample;
    const expectedBytes = nWin * nCh * layout.bytes_per_sample;
    const buf = await HttpRange.rangeFetch(layout.url, byteStart, byteStart + expectedBytes - 1, expectedBytes, opts);
    const interleaved = new layout.view_ctor(buf);

    const out = ChannelBuffers.alloc(nCh, nWin);
    ChannelDecode.deinterleaveInto(out, interleaved, nCh, nWin, layout.scales);
    return out;
  }

  // Streaming decode for BrainVision MULTIPLEXED format (same interleaved
  // layout as EEGLAB .fdt but with per-channel scale). Yields chunks of
  // complete frames as bytes arrive.
  const STREAM_BATCH_FRAMES_BV = 512;

  async function* streamMultiplexedWindow(layout, startSample, nWinReq, opts) {
    const win = ChannelBuffers.clampWindow(startSample, nWinReq, layout.n_samples);
    if (!win) return;
    const { start, nWin } = win;
    const nCh = layout.n_channels;
    const bps = layout.bytes_per_sample;
    const frameSize = nCh * bps;

    const byteStart = start * frameSize;
    const expectedBytes = nWin * frameSize;

    let leftover = new Uint8Array(0);
    let outSamples = 0;
    const scales = layout.scales;

    for await (const { bytes } of HttpRange.rangeFetchStreaming(
      layout.url, byteStart, byteStart + expectedBytes - 1, opts
    )) {
      const boundary = StreamingUtils.decodeChunkBoundary(leftover, bytes, frameSize);
      leftover = boundary.leftover;
      const completeBytes = boundary.completeRecordBytes;
      const nFrames = Math.floor(completeBytes.length / frameSize);
      if (nFrames === 0) continue;

      let fOff = 0;
      while (fOff < nFrames && outSamples < nWin) {
        const batchFrames = Math.min(STREAM_BATCH_FRAMES_BV, nFrames - fOff, nWin - outSamples);
        const batchU8 = completeBytes.subarray(fOff * frameSize, (fOff + batchFrames) * frameSize);
        // Create typed-array view of correct type for this format
        let interleaved;
        if (bps === 2) {
          interleaved = new layout.view_ctor(batchU8.buffer, batchU8.byteOffset, batchFrames * nCh);
        } else if (bps === 4) {
          interleaved = new layout.view_ctor(batchU8.buffer, batchU8.byteOffset, batchFrames * nCh);
        } else {
          interleaved = new layout.view_ctor(batchU8.buffer, batchU8.byteOffset, batchFrames * nCh);
        }
        const out = ChannelBuffers.alloc(nCh, batchFrames);
        ChannelDecode.deinterleaveInto(out, interleaved, nCh, batchFrames, scales);
        const firstSampleIdx = start + outSamples;
        const lastSampleIdx = firstSampleIdx + batchFrames - 1;
        outSamples += batchFrames;
        yield { firstSampleIdx, lastSampleIdx, channels: out };
        fOff += batchFrames;
      }
    }
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.BrainVisionReader = api;
})();
