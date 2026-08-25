/* ============================================================
   formats/ctf.js — minimal CTF MEG reader for eegdash-viewer.

   CTF recordings are *directory bundles*: the user-facing URL is
   `<entities>_meg.ds/`, a directory containing:
     <entities>_meg.res4    big-endian binary header (channels, srate,
                            gains) — parsed by formats/_ctf-res4.js
     <entities>_meg.meg4    big-endian int16 interleaved samples;
                            8-byte "MEG4xCP\0" magic + body
     <entities>_meg.acq     text acquisition metadata (ignored)
     <entities>_meg.hc      text head coordinates (ignored)
     <entities>_meg.hist    text history log (ignored)
     MarkerFile.mrk         text events → annotation_events
     BadChannels            text — one bad channel per line
     ClassFile.cls          text trial classifications (ignored)

   This reader fetches the .res4 + .meg4 over HTTP Range and serves
   windows directly from a cached `.meg4` body (small datasets) or
   range-fetches every readWindow call (large datasets). The cutoff
   is FULL_LOAD_MAX_BYTES below.

   References:
   - MNE-Python  mne/io/ctf/info.py         (CTF info-block assembly)
   - MNE-Python  mne/io/ctf/res4.py         (binary layout source of truth)
   - MNE-Python  mne/io/ctf/eeg.py          (.meg4 read path)
   - MNE-Python  mne/io/ctf/markers.py      (MarkerFile.mrk parsing)
   ============================================================ */
(function () {
  'use strict';

  const api = {};

  // CTF .meg4 samples are int32 BE; 4 bytes per sample. Verified against
  // MNE-Python's mne/io/ctf/ctf.py (`np.fromfile(fid, ">i4", ...)`) on
  // 2026-05-21. The previous BYTES_PER_SAMPLE=2 came from misreading the
  // .res4 header (which carries a 16-bit channel count) as the .meg4
  // sample width — ds002908's 1.1 GiB .meg4 only divides cleanly under
  // int32.
  const BYTES_PER_SAMPLE = 4;
  // 8-byte ASCII magic at the head of .meg4 — "MEG41CP\0" or "MEG42CP\0".
  const MEG4_HEADER_BYTES = 8;
  // Above this size we don't pre-fetch the full .meg4 — each readWindow
  // issues its own HTTP range. 64 MiB ≈ 100 channels × 30 minutes @ 1 kHz,
  // which still fits in browser memory but bumping the cutoff would have
  // us greedily holding multi-GB MEG sessions.
  const FULL_LOAD_MAX_BYTES = 64 * 1024 * 1024;

  /**
   * Parse a CTF `.res4` ArrayBuffer into a header object.
   * Synchronous entry point exposed for unit + property tests so the
   * parser can be exercised without network. Production `api.open`
   * calls this internally after HttpRange.fetchBuffer'ing the .res4.
   *
   * @param {ArrayBuffer} buf - the .res4 file as one buffer.
   * @returns {{
   *   no_samples: number, no_channels: number, sample_rate: number,
   *   epoch_time: number, no_trials: number,
   *   channels: Array<{ name: string, sensor_type: number, cal: number,
   *     io_offset: number, proper_gain: number, q_gain: number, io_gain: number }>
   * }}
   * @throws {Error} on any parse failure — never returns null.
   */
  api.read = function (buf) {
    // Delegates to the per-format helper. _ctf-res4.js is loaded into
    // globalThis.CTFRes4 by its own IIFE (in worker.js + index.html).
    if (!globalThis.CTFRes4) {
      throw new Error('ctf.read: globalThis.CTFRes4 missing — load formats/_ctf-res4.js first');
    }
    return globalThis.CTFRes4.parse(buf);
  };

  /**
   * Open a CTF MEG `.ds/` recording for windowed reading.
   *
   * `meta.eeg_url` must point at the `.meg4` file INSIDE the bundle
   * (e.g. `…/foo_meg.ds/foo_meg.meg4`); this is what bids-recording.js
   * builds when ext=ds. The reader derives the sibling URLs by string
   * arithmetic: replace `.meg4` with `.res4` / strip filename and
   * append `MarkerFile.mrk` / `BadChannels`.
   *
   * @param {object} meta - { eeg_url: string, … } as produced by
   *   bids-recording.js or a drag-and-drop bundle.
   * @returns {Promise<object>} reader with the cross-format contract:
   *   n_channels, sampling_frequency, n_samples, duration_s,
   *   channel_labels, bytes_per_sample, recording_start_iso,
   *   annotation_events, readWindow(start, n).
   */
  api.open = async function (meta) {
    const meg4Url = meta && (meta.eeg_url || meta.url);
    if (!meg4Url) throw new Error('ctf.open: meta.eeg_url is required (point at <entities>_meg.meg4)');

    const HttpRange = globalThis.HttpRange;
    if (!HttpRange) throw new Error('ctf.open: globalThis.HttpRange missing');

    // Derive sibling URLs. CTF guarantees the bundle's binary children
    // share the bundle basename: .meg4 → .res4 within the same .ds/.
    const res4Url = meg4Url.replace(/\.meg4$/, '.res4');
    if (res4Url === meg4Url) {
      throw new Error(`ctf.open: meta.eeg_url must end with .meg4 (got ${meg4Url})`);
    }
    const bundleDir = meg4Url.slice(0, meg4Url.lastIndexOf('/') + 1);
    const markerUrl = `${bundleDir}MarkerFile.mrk`;
    const badUrl    = `${bundleDir}BadChannels`;

    // Header first — everything else depends on it.
    const res4Buf = await HttpRange.fetchBuffer(res4Url);
    const header = api.read(res4Buf);

    // .meg4 body length tells us the true sample count when the recording
    // is continuous (no_trials=1) and gives us the per-trial chunk size
    // otherwise. We always trust the body length over no_samples *
    // no_trials because some converters write the header before knowing
    // the final sample count.
    const meg4Length = await HttpRange.probeLength(meg4Url);
    const bodyBytes = meg4Length - MEG4_HEADER_BYTES;
    if (bodyBytes < 0) {
      throw new Error(`ctf.open: .meg4 is ${meg4Length} bytes, smaller than the 8-byte magic header`);
    }
    if (bodyBytes % (header.no_channels * BYTES_PER_SAMPLE) !== 0) {
      throw new Error(
        `ctf.open: .meg4 body ${bodyBytes} bytes is not a multiple of ` +
        `${header.no_channels} channels x ${BYTES_PER_SAMPLE} bytes — header/body mismatch`
      );
    }
    const n_samples = bodyBytes / (header.no_channels * BYTES_PER_SAMPLE);

    // Markers + bad channels are optional. Failures are warnings, never
    // hard errors — viewer should still load the recording.
    let annotation_events = [];
    let bad_channels = [];
    try {
      const mrkText = await HttpRange.fetchTextOrNull(markerUrl);
      if (mrkText && globalThis.CTFMarker) {
        annotation_events = globalThis.CTFMarker.parseMarkerFile(mrkText);
      }
    } catch (e) {
      console.warn(`ctf.open: MarkerFile.mrk fetch failed (${e.message}); events skipped`);
    }
    try {
      const badText = await HttpRange.fetchTextOrNull(badUrl);
      if (badText && globalThis.CTFMarker) {
        bad_channels = globalThis.CTFMarker.parseBadChannels(badText);
      }
    } catch (e) {
      console.warn(`ctf.open: BadChannels fetch failed (${e.message}); bad-list skipped`);
    }

    // Pre-fetch the whole .meg4 if it fits in the in-memory budget;
    // otherwise readWindow issues a Range fetch per call. The cutoff
    // matches what eeglab.js does for inline-data .set files.
    let cachedBody = null;
    if (meg4Length <= FULL_LOAD_MAX_BYTES) {
      cachedBody = await HttpRange.fetchBuffer(meg4Url);
      // Sanity-check the magic; this is the first byte the user sees
      // out of the .meg4, so getting it wrong here means something is
      // very wrong with the bundle.
      const mag = new Uint8Array(cachedBody, 0, 8);
      const magStr = String.fromCharCode(...mag).replace(/\0.*$/, '');
      if (!/^MEG4[12]CP$/.test(magStr)) {
        throw new Error(`ctf.open: .meg4 bad magic ${JSON.stringify(magStr)} — expected MEG41CP or MEG42CP`);
      }
    }

    const channel_labels = header.channels.map(c => c.name);
    const cals    = header.channels.map(c => c.cal);
    const offsets = header.channels.map(c => c.io_offset);
    const nch     = header.no_channels;

    async function readWindow(startSample, nWin) {
      // Empty channels, length 0 — matches what edf.js returns at EOF.
      const win = globalThis.ChannelBuffers.clampWindow(startSample, nWin, n_samples);
      if (!win) return globalThis.ChannelBuffers.empty(nch);
      const { start, end } = win;
      const nOut = end - start;

      // CTF samples are interleaved: sample[t] of channel[c] sits at
      // body byte (t * nch + c) * BYTES_PER_SAMPLE. Read the required
      // byte range (one shot via Range or one slice of the cached body),
      // then de-interleave into channel-major Float32 with calibration.
      const byteStart = MEG4_HEADER_BYTES + start * nch * BYTES_PER_SAMPLE;
      const byteEnd   = MEG4_HEADER_BYTES + end   * nch * BYTES_PER_SAMPLE - 1;
      let buf;
      if (cachedBody) {
        // cachedBody includes the 8-byte magic; slice by the absolute
        // byte offsets so the arithmetic matches the range-fetch branch.
        buf = cachedBody.slice(byteStart, byteEnd + 1);
      } else {
        buf = await HttpRange.rangeFetch(meg4Url, byteStart, byteEnd, byteEnd - byteStart + 1);
      }
      const dv = new DataView(buf);

      const out = new Array(nch);
      for (let c = 0; c < nch; c++) out[c] = new Float32Array(nOut);

      for (let t = 0; t < nOut; t++) {
        const base = t * nch * BYTES_PER_SAMPLE;
        for (let c = 0; c < nch; c++) {
          const raw = dv.getInt32(base + c * BYTES_PER_SAMPLE, false);
          out[c][t] = (raw - offsets[c]) * cals[c];
        }
      }
      return out;
    }

    return {
      n_channels:          nch,
      sampling_frequency:  header.sample_rate,
      duration_s:          n_samples / header.sample_rate,
      channel_labels,
      bytes_per_sample:    BYTES_PER_SAMPLE,
      n_samples,
      recording_start_iso: null,  // TODO: parse from .acq dataTime field
      annotation_events,
      bad_channels,
      readWindow,
    };
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.CTFReader = api;
})();
