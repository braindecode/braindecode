/* ============================================================
   formats/eeglab.js — read EEGLAB recordings in either layout:

   1. Split `.set` + `.fdt`: the .set is just a MATLAB header
      (channel info, srate, etc.) and the data lives in a sibling
      .fdt file as flat little-endian float32 in MATLAB column-
      major order — `data[chan, sample] @ byte (sample*nCh+chan)*4`.
      Range-friendly: we fetch only the needed window via HTTP
      Range and de-interleave once.

   2. Inline-data `.set` (most modern EEGLAB / MNE-Python exports):
      the .set is a MAT v5 file containing the data as a typed
      array INSIDE itself. We download the whole .set, parse it
      via _matv5.js, extract `EEG.data` (or top-level `data`),
      and serve windows from the in-memory column-major array.

   We skip BIDS sidecar `.set` parsing for the split-layout case
   (channel count + sample rate come from `_channels.tsv` and
   `_eeg.json`); the inline case has no choice but to parse the
   MAT structure since that's where the data is. Standalone inline
   .set files without a BIDS sidecar are supported too — nbchan and
   srate come from the EEG struct and channel labels default to
   Ch1..ChN when _channels.tsv is absent.

   Epoched (3-D) data `[n_channels, n_pnts, n_trials]` is treated
   as continuous: we flatten the trial axis so the viewer's flat
   time-axis pans across concatenated trials.
   ============================================================ */
(function () {
  'use strict';

  const api = {};
  const BYTES_PER_SAMPLE = 4;
  const HOST_LITTLE_ENDIAN =
    new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

  // Security: when a v7.3 .set CHAR pointer names a sibling .fdt file
  // we treat the value as a BASENAME ONLY. Reject anything that could
  // escape the .set's directory (path separator, leading dot, scheme).
  // Threat model: hostile .set embeds e.g. "../../../etc/passwd" or
  // "//evil.com/x" as the /EEG/data CHAR; the reader would otherwise
  // concatenate dir + namedFdt and fetch the resulting URL.
  function _validateCrossFdtName(namedFdt) {
    if (typeof namedFdt !== 'string' || namedFdt.length === 0) {
      throw new Error(`eeglab: refusing cross-basename .fdt with empty or non-string name`);
    }
    if (namedFdt.includes('/') || namedFdt.includes('\\') ||
        namedFdt.startsWith('.') || /^[a-z]+:/i.test(namedFdt)) {
      throw new Error(`eeglab: refusing cross-basename .fdt with path separator or scheme: ${namedFdt}`);
    }
    return namedFdt;
  }
  api._validateCrossFdtName = _validateCrossFdtName;

  // Security: reject pathological nbchan/pnts/trials advertised by a
  // hostile .set before they reach the allocator. Threat model: a .set
  // with pnts=1e9, nbchan=10000, trials=1 makes nbchan*pnts*trials*4
  // ≥ 4e13 bytes, triggering OOM the moment downstream code allocates.
  // Caps chosen well above any plausible real-world EEG recording.
  const _MAX_SAMPLES = 1 << 30;   // ~1.07B samples per channel
  const _MAX_CH = 4096;
  const _MAX_TRIALS = 4096;
  function _validateScalars(nbchan, pnts, trials) {
    if (!Number.isInteger(nbchan) || nbchan <= 0 || nbchan > _MAX_CH) {
      throw new Error(`eeglab: rejecting nbchan=${nbchan} (must be 1..${_MAX_CH})`);
    }
    if (!Number.isInteger(pnts) || pnts <= 0 || pnts > _MAX_SAMPLES) {
      throw new Error(`eeglab: rejecting pnts=${pnts} (must be 1..${_MAX_SAMPLES})`);
    }
    if (!Number.isInteger(trials) || trials <= 0 || trials > _MAX_TRIALS) {
      throw new Error(`eeglab: rejecting trials=${trials} (must be 1..${_MAX_TRIALS})`);
    }
    // Overflow-safe product check: each cap below 2^31 so the products
    // can't overflow Number precision before we compare.
    if (nbchan * pnts > _MAX_SAMPLES || nbchan * pnts * trials > _MAX_SAMPLES) {
      throw new Error(`eeglab: nbchan*pnts*trials=${nbchan * pnts * trials} exceeds cap ${_MAX_SAMPLES}`);
    }
  }
  api._validateScalars = _validateScalars;
  api._SCALAR_CAPS = { MAX_SAMPLES: _MAX_SAMPLES, MAX_CH: _MAX_CH, MAX_TRIALS: _MAX_TRIALS };

  api.fdtUrlFor = function (eegUrl) {
    const { dir, prefix, ext } = BIDSRecording.parseEegUrl(eegUrl);
    if (ext !== 'set') {
      throw new Error(`EEGLAB reader expects *_eeg.set, got *_eeg.${ext}`);
    }
    return `${dir}${prefix}_eeg.fdt`;
  };

  // Detect whether a file/sidecar duration mismatch is "this is just
  // an epoched .fdt the sidecar doesn't know about" or a real problem.
  // Returning kind keeps the open() control flow flat.
  function classifyDurationMismatch(fileDur, declaredDur) {
    if (declaredDur == null || declaredDur <= 0) return { kind: 'no-declared' };
    if (Math.abs(fileDur - declaredDur) <= 0.01) return { kind: 'ok' };
    const ratio = fileDur / declaredDur;
    const intRatio = Math.round(ratio);
    if (intRatio > 1 && Math.abs(ratio - intRatio) < 0.01) {
      return { kind: 'epoched', trials: intRatio };
    }
    return { kind: 'mismatch' };
  }

  /**
   * Open an EEGLAB .set file (with optional external .fdt) for windowed reading.
   *
   * The returned descriptor (loosely typed as `object` because the .set
   * format has several optional fields and extra metadata is attached
   * conditionally) exposes at least:
   *   - n_channels, sampling_frequency, n_samples, bytes_per_sample,
   *     duration_s: number
   *   - url, channel_labels, bids_channels: pass-through metadata
   *   - readWindow(start, n, opts?): Promise<Float32Array[]>
   *   - readWindowStreaming(start, n, opts?) when supported by the layout
   *
   * @param {object} meta - The recording descriptor from bids-recording.js.
   * @returns {Promise<object>}
   */
  api.open = async function (meta) {
    if (!HOST_LITTLE_ENDIAN) {
      throw new Error('EEGLAB .fdt reader requires a little-endian host.');
    }
    // BIDS-strict gate has been relaxed: when _channels.tsv is absent
    // we can still serve an inline-data .set (MAT v5 parser fills in
    // nbchan / srate / labels from the EEG struct itself). The split
    // .set+.fdt layout, on the other hand, still needs _channels.tsv
    // because the .fdt is a raw float32 blob with no header — we have
    // no way to know nChannels without it. So: only require channels
    // up front; defer the BIDS-sidecar requirement to the .fdt branch.
    const eegJson = meta.eeg_json || {};
    const sidecarFs = eegJson.sampling_frequency;
    const sidecarFsValid = isFinite(sidecarFs) && sidecarFs > 0;
    const hasChannels = !!(meta.channels && meta.channels.length);
    const nChannelsFromSidecar = hasChannels ? meta.channels.length : null;

    // Resolve the .fdt sibling URL. For BIDS-pathed sources
    // (OpenNeuro) we derive it by string-replace on .set; for SHA-
    // keyed sources (NEMAR) we look it up in the pre-resolved map.
    // A null result is a strong signal that this is an inline-data
    // .set — we'll fall through to the MAT parser below.
    const fdtUrl = resolveFdtUrl(meta);

    // Probe the .fdt; on 404 (or NEMAR with no .fdt entry) switch to
    // the inline-data .set path. Other errors propagate.
    let totalBytes = null;
    if (fdtUrl) {
      try {
        totalBytes = await HttpRange.probeLength(fdtUrl);
      } catch (e) {
        if (!/HTTP 404/.test(e.message)) throw e;
      }
    }
    if (totalBytes == null) {
      // Inline-data path: nbchan / srate / data live inside the .set;
      // we can produce a working reader without the BIDS sidecar.
      // Pass nulls for the sidecar values when missing; openInlineSet
      // will use the .set's own metadata and warn only if they conflict.
      return openInlineSet(meta, nChannelsFromSidecar, sidecarFsValid ? sidecarFs : null);
    }

    // Split .set + .fdt path: the .fdt is a flat float32 blob with no
    // header. We need nChannels + SamplingFrequency to interpret it.
    //
    // Source priority — the .set is the authority, not the BIDS sidecar:
    //   - sidecar _channels.tsv may list ALL acquired channels (including
    //     bad/dropped ones, MEG system channels, status, etc.) while the
    //     .set stores only the channels that were actually written to
    //     the .fdt (after preprocessing / ICA / channel selection).
    //   - Real example (ds003645): _channels.tsv has 404 entries (full
    //     MEG sensor array + EEG + triggers); .set says nbchan=75 (the
    //     subset that was preprocessed). .fdt size 162030000 = 75 × 4 ×
    //     540100 → matches the .set, not the sidecar.
    //
    // Strategy:
    //   1. Always try to parse the .set to get its authoritative nbchan
    //      + srate (the .fdt-data writer wrote these in lockstep with
    //      the .fdt's actual layout).
    //   2. If .set is unparseable, fall back to the BIDS sidecar values.
    //   3. Warn if sidecar and .set disagree (BIDS data-curation hint).
    let nChannels = null;
    let fs = null;
    let setParseFailed = false;
    let setParseError = null;
    if (nChannels == null || fs == null) {
      try {
        const setBuf = await HttpRange.fetchBuffer(meta.eeg_url);
        const matVer = MatV5.detectMatVersion(setBuf);
        let vars;
        if (matVer === 'v7.3' && typeof globalThis.Mat73 !== 'undefined') {
          vars = await Mat73.parse(setBuf);
        } else {
          vars = await MatV5.parse(setBuf);
        }
        const eegStruct = vars.get('EEG');
        const fieldFrom = (name) => {
          if (vars.has(name)) return vars.get(name);
          if (eegStruct && eegStruct.class === 'struct' && eegStruct.data.has(name)) {
            return eegStruct.data.get(name);
          }
          return null;
        };
        const scalarFrom = (name) => {
          const v = fieldFrom(name);
          if (!v || !v.data || !v.data.length) return null;
          return Number(v.data[0]);
        };
        const nbchanFromSet = scalarFrom('nbchan');
        const srateFromSet  = scalarFrom('srate');
        if (nbchanFromSet) nChannels = nbchanFromSet;
        if (srateFromSet && srateFromSet > 0) fs = srateFromSet;
        // Warn loudly when sidecar and .set disagree — almost always a
        // sign of post-acquisition channel selection / preprocessing
        // that wasn't reflected in the BIDS curation.
        if (nChannelsFromSidecar != null && nChannels != null &&
            nChannels !== nChannelsFromSidecar) {
          console.warn(
            `EEGLAB .set+.fdt: sidecar _channels.tsv lists ` +
            `${nChannelsFromSidecar} channels but the .set declares ` +
            `EEG.nbchan=${nChannels}. Trusting the .set (it matches ` +
            `the .fdt's actual layout; the sidecar likely lists all ` +
            `acquired channels including dropped/system ones).`,
          );
        }
        if (sidecarFsValid && fs != null && Math.abs(fs - sidecarFs) > 0.5) {
          console.warn(
            `EEGLAB .set+.fdt: sidecar SamplingFrequency=${sidecarFs} but ` +
            `EEG.srate=${fs} in the .set. Trusting the .set.`,
          );
        }
      } catch (e) {
        setParseFailed = true;
        setParseError = e;
      }
    }
    // Sidecar fallback: only if .set parse failed AND sidecar has values.
    if (nChannels == null && nChannelsFromSidecar != null) {
      nChannels = nChannelsFromSidecar;
    }
    if (fs == null && sidecarFsValid) {
      fs = sidecarFs;
    }
    if (!nChannels && setParseFailed) {
      throw new Error(
        `EEGLAB .set+.fdt: need either parseable .set with EEG.nbchan + EEG.srate ` +
        `OR _channels.tsv + _eeg.json BIDS sidecars. ` +
        `Set parse error: ${setParseError ? setParseError.message : 'unknown'}`,
      );
    }
    if (!nChannels) {
      throw new Error(
        'EEGLAB .fdt reader needs nChannels (either from _channels.tsv ' +
        'sidecar or EEG.nbchan in the .set file).',
      );
    }
    if (!fs) {
      throw new Error(
        'EEGLAB .fdt reader needs SamplingFrequency (either from ' +
        '_eeg.json sidecar or EEG.srate in the .set file).',
      );
    }

    // Truncation tolerance — observed on ds003570 (97MB short) and ds003751
    // (only 1.7% present) on OpenNeuro. The .set is authoritative about
    // nChannels (it was written in lockstep with the .fdt's record layout)
    // so a leftover-bytes remainder means the .fdt is truncated, not that
    // nChannels is wrong. Accept floor(file_bytes / record_size) samples
    // with a loud warning — partial data is more useful than a hard reject,
    // especially since these files are broken at the publisher's end and
    // unlikely to be re-uploaded.
    const recordSize = nChannels * BYTES_PER_SAMPLE;
    const sampleRem = totalBytes % recordSize;
    if (sampleRem !== 0) {
      console.warn(
        `EEGLAB .fdt: file size ${totalBytes}B is not a multiple of ` +
        `${nChannels}×4=${recordSize}B (${sampleRem}B trailing); truncating ` +
        `to ${Math.floor(totalBytes / recordSize)} samples. File is likely ` +
        `truncated at the publisher — observed on ds003570 + ds003751.`
      );
    }
    const nSamples = Math.floor(totalBytes / recordSize);
    if (nSamples === 0) {
      throw new Error(
        `EEGLAB .fdt: file size ${totalBytes}B < record size ` +
        `${recordSize}B — no full sample available.`
      );
    }
    const fileDur = nSamples / fs;
    const mismatch = classifyDurationMismatch(fileDur, meta.eeg_json.recording_duration);
    let trialsHint = null;
    if (mismatch.kind === 'epoched') {
      trialsHint = mismatch.trials;
      console.warn(
        `.fdt appears epoched: ${trialsHint} trials. v1 treats it as continuous; ` +
        `epoch boundaries will not be marked.`
      );
    } else if (mismatch.kind === 'mismatch') {
      console.warn(
        `.fdt duration (${fileDur.toFixed(3)}s) disagrees with sidecar ` +
        `(${meta.eeg_json.recording_duration}s); trusting file.`
      );
    }

    return {
      n_channels: nChannels,
      n_samples: nSamples,
      sampling_frequency: fs,
      duration_s: fileDur,
      // .fdt is always Float32 little-endian per the EEGLAB spec.
      // Exposed so callers (e.g. the adaptive default-window picker
      // in viewer.js) can compute per-pan byte cost uniformly across
      // formats without special-casing EEGLAB.
      bytes_per_sample: 4,
      trials_hint: trialsHint,
      url: fdtUrl,
      // Channel labels: use the BIDS sidecar IFF its row count matches the
      // nChannels we just derived (either from EEG.nbchan or from the sidecar
      // itself when .set was unparseable). On length mismatch — sidecar has
      // 404 entries but .set says 75 (ds003645) — fall back to Ch1..ChN since
      // the sidecar names belong to channels that aren't in the .fdt. On
      // missing sidecar entirely, same fallback. This mirrors the inline-data
      // path and unblocks .set+.fdt loading when _channels.tsv is absent.
      channel_labels: ChannelLabels.fromMetaOr(meta, nChannels),
      bids_channels: (meta.channels && meta.channels.length === nChannels) ? meta.channels : null,
      // Bounds-clamp here so callers can pan past the end without
      // worrying about negative ranges or off-by-one near EOF.
      readWindow: async (startSample, nSamplesWindow, opts) => {
        const win = ChannelBuffers.clampWindow(startSample, nSamplesWindow, nSamples);
        if (!win) return ChannelBuffers.empty(nChannels);
        return readInterleavedWindow(fdtUrl, nChannels, win.start, win.nWin, opts);
      },
      readWindowStreaming: (startSample, nSamplesWindow, opts) =>
        streamInterleavedWindow(fdtUrl, nChannels, nSamples, startSample, nSamplesWindow, opts),
    };
  };

  // BIDS-pathed sources string-derive .fdt; SHA-keyed sources look
  // it up in the pre-resolved sibling map. Returns null when neither
  // produces a candidate URL (used as the signal to try inline-data
  // parsing instead).
  function resolveFdtUrl(meta) {
    if (meta.sibling_urls) {
      return meta.sibling_urls[`${meta.prefix}_eeg.fdt`] || null;
    }
    return api.fdtUrlFor(meta.eeg_url);
  }

  // Inline-data .set: the EEG signal lives inside the MAT file
  // itself (no sibling .fdt). We download the whole file once,
  // parse it with the minimal MAT v5 reader, and serve windows
  // from the in-memory column-major typed array.
  //
  // Memory cost: scales with file size (typical .set is 1-200 MB).
  // For huge multi-hour recordings the upfront download will be
  // perceptible; range-streaming inside a MAT structure is non-
  // trivial (variable-length elements, optional zlib compression)
  // and out of scope for v1.
  // Range-based inline-set open. Range-fetches the head (first 16 MB)
  // to scan top-level MAT v5 elements, then serves `readWindow` by
  // range-fetching just the column slice of the `data` matrix. This
  // unblocks files > 200 MB that previously hit the legacy cap.
  //
  // Falls back to the whole-file parse path for:
  //   - MAT v7.3 (HDF5 needs the whole file for jsfive)
  //   - Compressed (miCOMPRESSED) elements (zlib needs the whole stream)
  //   - Non-float32 data classes (we don't range-stream int16/double yet)
  //   - Struct-wrapped EEG (scanElements only sees top-level matrices;
  //     EEG struct hides `data` inside, so we fall through to MatV5.parse)
  // Each fallback retains its own 200 MB ceiling so we don't OOM the
  // page on a non-streamable huge file.
  const INLINE_METADATA_BUDGET_BYTES = 16 * 1024 * 1024;  // 16 MB head probe
  // Legacy whole-file fallback cap. 1 GB is well within modern desktop
  // browser ArrayBuffer limits (Chrome supports up to 4 GB on 64-bit
  // systems). The cap exists to avoid catastrophic OOMs on low-end
  // devices, not to protect against any specific browser limit.
  //
  // Affects 6 of 11 inline-data audit failures (286–903 MB struct-
  // wrapped .set files): ds004019, ds004040, ds004151, ds005178,
  // ds006648, ds006866. Files >1 GB still fail with a clear message.
  const INLINE_LEGACY_FALLBACK_CAP   = 1024 * 1024 * 1024;  // 1 GB

  // Shared fallback shape used by 6 different branches inside
  // openInlineSet (v7.3, scan-failed, compressed, no-data, no-srate,
  // non-float32). Each branch needs to either throw a branch-specific
  // error when the file exceeds INLINE_LEGACY_FALLBACK_CAP, or pull
  // the whole file and hand it to openInlineSetLegacy. The error
  // strings differ per branch (and some embed extra context like
  // `miType` or `probeBytes`) so the caller passes a builder lambda
  // that receives (mbStr, capMb) and returns the exact message.
  async function fallbackToLegacyOrThrow(
    setUrl, totalBytes, meta, nChannelsFromSidecar, fsFromSidecar, matVersion, buildErr,
  ) {
    if (totalBytes > INLINE_LEGACY_FALLBACK_CAP) {
      const mb  = (totalBytes / 1024 / 1024).toFixed(0);
      const cap = INLINE_LEGACY_FALLBACK_CAP / 1024 / 1024;
      throw new Error(buildErr(mb, cap));
    }
    const buf = await HttpRange.rangeFetch(setUrl, 0, totalBytes - 1, totalBytes);
    return await openInlineSetLegacy(setUrl, meta, buf, nChannelsFromSidecar, fsFromSidecar, matVersion);
  }

  async function openInlineSet(meta, nChannelsFromSidecar, fsFromSidecar) {
    const setUrl = meta.eeg_url;
    // Use a 1-byte Range GET to learn total size — HEAD requests
    // against cdn.eegdash.org poison the Range cache (see
    // tests/evidence/streaming-large/README.md for the discovery).
    const totalBytes = await HttpRange.probeLengthNoHead(setUrl);

    // Range-fetch the head probe (capped at 16 MB or totalBytes).
    const probeBytes = Math.min(totalBytes, INLINE_METADATA_BUDGET_BYTES);
    const probeBuf   = await HttpRange.rangeFetch(setUrl, 0, probeBytes - 1, probeBytes);

    // Detect MAT version. v7.3 (HDF5) is NOT range-streamable in v1 —
    // jsfive needs the whole file. Fall back to the legacy whole-file
    // path, with the 200 MB cap kept as a safety net.
    const matVersion = MatV5.detectMatVersion(probeBuf);
    if (matVersion === 'v7.3') {
      return await fallbackToLegacyOrThrow(
        setUrl, totalBytes, meta, nChannelsFromSidecar, fsFromSidecar, matVersion,
        (mb, cap) =>
          `EEGLAB inline v7.3 .set is ${mb} MB ` +
          `(exceeds ${cap} MB v7.3 cap). ` +
          `Streaming v7.3 is not supported in v1.`,
      );
    }

    // v5 path: scan the probe buffer for top-level elements.
    // If scan fails (e.g. struct-wrapped EEG whose payload exceeds
    // the 16 MB probe), fall back to whole-file parse with the
    // 200 MB cap as a safety net.
    let elements;
    try {
      elements = MatV5.scanElements(probeBuf);
    } catch (e) {
      return await fallbackToLegacyOrThrow(
        setUrl, totalBytes, meta, nChannelsFromSidecar, fsFromSidecar, 'v5',
        (mb, cap) =>
          `EEGLAB inline .set scan failed and file is ` +
          `${mb} MB ` +
          `(exceeds ${cap} MB ` +
          `legacy cap). Original error: ${e.message}`,
      );
    }

    // If any compressed element is present, fall back to whole-file parse.
    const hasCompressed = elements.some(el => el.miType === 15);
    if (hasCompressed) {
      return await fallbackToLegacyOrThrow(
        setUrl, totalBytes, meta, nChannelsFromSidecar, fsFromSidecar, 'v5',
        (mb, cap) =>
          `EEGLAB inline .set is compressed and ${mb} MB ` +
          `(exceeds ${cap} MB legacy cap).`,
      );
    }

    // Find the top-level `data` matrix. EEGLAB writes either a single
    // struct named "EEG" wrapping data/srate/nbchan/etc., or top-level
    // variables with those names. scanElements only walks top-level —
    // struct-wrapped layouts fall back to the legacy whole-file parse.
    const dataElem = elements.find(el => el.name === 'data' && el.dataSubOffset != null);
    if (!dataElem) {
      // Probably EEG-wrapped struct, or the head probe didn't reach
      // far enough. Either way, fall back to whole-file parse.
      return await fallbackToLegacyOrThrow(
        setUrl, totalBytes, meta, nChannelsFromSidecar, fsFromSidecar, 'v5',
        (mb, _cap) =>
          `EEGLAB inline .set: top-level 'data' not found in first ${probeBytes}B ` +
          `(file is ${mb} MB, exceeds legacy cap). ` +
          `Re-export as top-level (non-struct-wrapped) inline .set or split .set+.fdt.`,
      );
    }

    // Pull the small metadata fields from the scanned elements. EEGLAB
    // writes scalars compactly: when a value (srate=250, nbchan=74)
    // fits in a smaller integer type, the MAT writer encodes the
    // realdata sub-element in that type — independent of the matrix's
    // mxClass. We've seen ds002718 store srate=250 as a single uint8
    // byte, dims-array-style. Handle every integer width that EEGLAB
    // emits in the wild, plus the two float types.
    function readScalar(name) {
      const el = elements.find(x => x.name === name && x.dataSubOffset != null);
      if (!el) return null;
      const localOff = el.dataSubOffset;
      if (localOff < 0 || localOff + el.dataSubBytes > probeBuf.byteLength) return null;
      const dv = new DataView(probeBuf, localOff, el.dataSubBytes);
      switch (el.dataSubMiType) {
        case 1: return dv.getInt8(0);            // miINT8
        case 2: return dv.getUint8(0);           // miUINT8
        case 3: return dv.getInt16(0, true);     // miINT16
        case 4: return dv.getUint16(0, true);    // miUINT16
        case 5: return dv.getInt32(0, true);     // miINT32
        case 6: return dv.getUint32(0, true);    // miUINT32
        case 7: return dv.getFloat32(0, true);   // miSINGLE
        case 9: return dv.getFloat64(0, true);   // miDOUBLE
        default: return null;
      }
    }

    const srate  = readScalar('srate');
    const nbchan = readScalar('nbchan') ?? dataElem.dims[0];
    const pnts   = readScalar('pnts')   ?? dataElem.dims[1];
    const trials = readScalar('trials') ?? (dataElem.dims[2] || 1);
    if (!srate || !isFinite(srate) || srate <= 0) {
      // EEG.srate isn't a top-level matrix — file is struct-wrapped or
      // uses miCOMPRESSED scalars. Fall back to whole-file parse so
      // extractEegInline can descend into the EEG struct.
      return await fallbackToLegacyOrThrow(
        setUrl, totalBytes, meta, nChannelsFromSidecar, fsFromSidecar, 'v5',
        (mb, cap) =>
          `EEGLAB inline .set: srate not at top level and file is ${mb} MB ` +
          `(exceeds ${cap} MB legacy cap).`,
      );
    }
    if (nChannelsFromSidecar != null && nbchan !== nChannelsFromSidecar) {
      console.warn(
        `EEGLAB inline .set: nbchan=${nbchan} disagrees with _channels.tsv (${nChannelsFromSidecar}); ` +
        `trusting the .set.`
      );
    }
    if (fsFromSidecar != null && Math.abs(srate - fsFromSidecar) > 0.5) {
      console.warn(
        `EEGLAB inline .set: srate=${srate} Hz disagrees with _eeg.json (${fsFromSidecar} Hz); ` +
        `trusting the .set.`
      );
    }

    // Security A2: reject hostile scalar values before allocator math.
    _validateScalars(nbchan, pnts, trials);
    const nSamples = pnts * trials;
    const expectedDataBytes = nbchan * nSamples * 4;
    if (dataElem.dataSubMiType !== 7) {
      // Non-float32 data — fall back to whole-file parse.
      return await fallbackToLegacyOrThrow(
        setUrl, totalBytes, meta, nChannelsFromSidecar, fsFromSidecar, 'v5',
        (mb, cap) =>
          `EEGLAB inline .set: data is non-float32 (miType=${dataElem.dataSubMiType}) ` +
          `and file is ${mb} MB — exceeds ${cap} MB legacy cap.`,
      );
    }
    if (dataElem.dataSubBytes !== expectedDataBytes) {
      // Keep the canonical "data length != nbchan × pnts × trials" wording
      // so callers (and tests) can match on a stable string.
      throw new Error(
        `EEGLAB inline .set: data length ${dataElem.dataSubBytes / 4} != ` +
        `nbchan(${nbchan}) × pnts(${pnts}) × trials(${trials}) (= ${nbchan * pnts * trials})`,
      );
    }

    const duration_s = nSamples / srate;
    const trialsHint = trials > 1 ? trials : null;
    if (trialsHint) {
      console.warn(`EEGLAB inline .set is epoched (${trialsHint} trials); v1 flattens to continuous.`);
    }
    const channelLabels = ChannelLabels.fromMetaOr(meta, nbchan);

    const dataAbsOffset = dataElem.dataSubOffset;

    return {
      n_channels:         nbchan,
      n_samples:          nSamples,
      sampling_frequency: srate,
      duration_s,
      bytes_per_sample:   4,
      trials_hint:        trialsHint,
      url:                setUrl,
      channel_labels:     channelLabels,
      bids_channels:      meta.channels || null,
      streaming:          true,
      async readWindow(startSample, nSamplesWindow, opts) {
        const win = ChannelBuffers.clampWindow(startSample, nSamplesWindow, nSamples);
        if (!win) return ChannelBuffers.empty(nbchan);
        const { start, end, nWin } = win;
        const byteStart = dataAbsOffset + start * nbchan * 4;
        const byteEnd   = dataAbsOffset + end   * nbchan * 4 - 1;
        const buf = await HttpRange.rangeFetch(setUrl, byteStart, byteEnd, nWin * nbchan * 4, opts);
        const flat = new Float32Array(buf);
        // Column-major slice: data[chan, sample] @ sample*nchan + chan.
        const out = ChannelBuffers.alloc(nbchan, nWin);
        ChannelDecode.deinterleaveInto(out, flat, nbchan, nWin, null);
        return out;
      },
    };
  }

  // HEAD-avoidant length probe lives in formats/_http_range.js as
  // HttpRange.probeLengthNoHead (promoted in B4 — fiff.js shipped the
  // same workaround for the same CDN cache-poisoning bug).

  // Legacy whole-file parse path. Used as the fallback for v7.3,
  // compressed, struct-wrapped, or non-float32 inline .set files.
  // Identical to the pre-refactor behaviour but factored out so the
  // streaming path can reuse it.
  async function openInlineSetLegacy(setUrl, meta, buf, nChannelsFromSidecar, fsFromSidecar, matVersion) {
    let vars;
    if (matVersion === 'v7.3' && typeof globalThis.Mat73 !== 'undefined') {
      try {
        vars = await Mat73.parse(buf);
      } catch (e) {
        // Cross-basename .fdt fallback: when /EEG/data is a CHAR
        // pointer to a sibling whose basename differs from the .set,
        // Mat73.parse throws a precise message containing the filename
        // in double-quoted form. Parse the filename out, derive the
        // sibling URL, and serve windows from the named .fdt. See
        // tests/evidence/v73-real-data/README.md for the rationale.
        const fdtMatch = /CHAR sidecar filename \("([^"]+)"\)/.exec(e.message || '');
        if (fdtMatch) {
          const namedFdt = _validateCrossFdtName(fdtMatch[1]);
          const dir = setUrl.slice(0, setUrl.lastIndexOf('/') + 1);
          const fdtUrl = dir + namedFdt;
          console.warn(
            `EEGLAB v7.3: /EEG/data points at sibling "${namedFdt}" ` +
            `(different basename from the .set); following the named .fdt.`
          );
          // Without the .set's inline numeric data we can't recover
          // nbchan/srate from the file itself. The BIDS sidecar
          // (passed in `meta`) is the only source.
          if (!nChannelsFromSidecar || !fsFromSidecar) {
            throw new Error(
              `EEGLAB v7.3 cross-basename: need _channels.tsv and ` +
              `SamplingFrequency in _eeg.json to interpret the named ` +
              `.fdt sibling "${namedFdt}"`
            );
          }
          const totalBytesFdt = await HttpRange.probeLength(fdtUrl);
          if (totalBytesFdt % (nChannelsFromSidecar * BYTES_PER_SAMPLE) !== 0) {
            throw new Error(
              `.fdt size ${totalBytesFdt} is not a multiple of ` +
              `${nChannelsFromSidecar}×${BYTES_PER_SAMPLE} — sidecar ` +
              `channel count may be wrong`
            );
          }
          const nSamplesFdt = totalBytesFdt / (nChannelsFromSidecar * BYTES_PER_SAMPLE);
          const durationFdt = nSamplesFdt / fsFromSidecar;
          const labels = ChannelLabels.fromMetaOr(meta, nChannelsFromSidecar);
          return {
            n_channels: nChannelsFromSidecar,
            n_samples: nSamplesFdt,
            sampling_frequency: fsFromSidecar,
            duration_s: durationFdt,
            bytes_per_sample: BYTES_PER_SAMPLE,
            url: fdtUrl,
            channel_labels: labels,
            bids_channels: meta.channels || null,
            readWindow: async (startSample, nSamplesWindow, opts) => {
              const win = ChannelBuffers.clampWindow(startSample, nSamplesWindow, nSamplesFdt);
              if (!win) return ChannelBuffers.empty(nChannelsFromSidecar);
              return readInterleavedWindow(
                fdtUrl,
                nChannelsFromSidecar,
                win.start,
                win.nWin,
                opts,
              );
            },
          };
        }
        throw new Error(`EEGLAB inline .set (v7.3) parse failed at ${setUrl}: ${e.message}`);
      }
    } else {
      try {
        vars = await MatV5.parse(buf);
      } catch (e) {
        throw new Error(`EEGLAB inline .set parse failed at ${setUrl}: ${e.message}`);
      }
    }
    let eeg;
    try {
      eeg = MatV5.extractEegInline(vars);
    } catch (e) {
      // V5 CHAR-sidecar fallback: EEG.data is a string referencing a
      // sibling .fdt (e.g. ds003078). Mirror the v7.3 path: parse the
      // filename out, derive the sibling URL, and serve windows from
      // the named .fdt. The fdtFilename is set on the error by
      // _matv5.js when class==='char'.
      if (e && e.code === 'EEGLAB_DATA_IS_CHAR') {
        const namedFdt = _validateCrossFdtName(e.fdtFilename);
        const dir = setUrl.slice(0, setUrl.lastIndexOf('/') + 1);
        const fdtUrl = dir + namedFdt;
        console.warn(
          `EEGLAB v5: /EEG/data is a CHAR sidecar pointer "${namedFdt}"; ` +
          `following the named .fdt at ${fdtUrl}.`
        );
        // For the v5 CHAR-sidecar case we DO have access to the parsed
        // numeric scalars (nbchan, srate, pnts, etc.) sitting in `vars`
        // — they're separate top-level fields from EEG.data. Try to
        // pull them directly from `vars`; fall back to the sidecar if
        // they're absent.
        const eegStruct = vars.get('EEG');
        const fieldFrom = (name) => {
          if (vars.has(name)) return vars.get(name);
          if (eegStruct && eegStruct.class === 'struct' && eegStruct.data.has(name)) {
            return eegStruct.data.get(name);
          }
          return null;
        };
        const scalarFrom = (name) => {
          const v = fieldFrom(name);
          if (!v || !v.data || !v.data.length) return null;
          return Number(v.data[0]);
        };
        const nchanFromVars = scalarFrom('nbchan');
        const fsFromVars = scalarFrom('srate');
        const nchan = nchanFromVars ?? nChannelsFromSidecar;
        const fs = fsFromVars ?? fsFromSidecar;
        if (!nchan || !fs) {
          throw new Error(
            `EEGLAB v5 cross-basename: need nbchan and srate (either in ` +
            `.set or in _channels.tsv + _eeg.json) to interpret named ` +
            `.fdt sibling "${namedFdt}"`
          );
        }
        // Fallback for STALE filename: real BIDS datasets (e.g.
        // ds003078) ship .set files whose EEG.data CHAR still names
        // the original acquisition file (`S_1_cond1_run1.fdt`) even
        // though the BIDS curator renamed the sibling on disk to the
        // canonical pattern (`sub-XX_..._eeg.fdt`). When the named
        // filename 404s, retry with the same basename as the .set but
        // `.fdt` extension. Mirrors brainvision.js stale-DataFile fix.
        let effectiveFdtUrl = fdtUrl;
        let totalBytesFdt;
        try {
          totalBytesFdt = await HttpRange.probeLength(effectiveFdtUrl);
        } catch (probeErr) {
          const msg = probeErr && probeErr.message ? probeErr.message : String(probeErr);
          if (!/404|HTTP 4\d\d|Cannot determine length/.test(msg)) throw probeErr;
          const fallbackUrl = setUrl.replace(/\.set(\?|$)/i, '.fdt$1');
          if (fallbackUrl === fdtUrl) throw probeErr;
          try {
            totalBytesFdt = await HttpRange.probeLength(fallbackUrl);
            console.warn(
              `EEGLAB v5: CHAR sidecar "${namedFdt}" 404'd at ${fdtUrl}; ` +
              `falling back to .set-basename sibling ${fallbackUrl}.`
            );
            effectiveFdtUrl = fallbackUrl;
          } catch {
            throw new Error(
              `EEGLAB v5: cannot find .fdt sibling. Tried (1) ${fdtUrl} ` +
              `from CHAR="${namedFdt}", (2) ${fallbackUrl} from .set basename. ` +
              `Original error: ${msg}`,
            );
          }
        }
        if (totalBytesFdt % (nchan * BYTES_PER_SAMPLE) !== 0) {
          throw new Error(
            `.fdt size ${totalBytesFdt} is not a multiple of ` +
            `${nchan}×${BYTES_PER_SAMPLE} — channel count may be wrong`
          );
        }
        const nSamplesFdt = totalBytesFdt / (nchan * BYTES_PER_SAMPLE);
        const labels = ChannelLabels.fromMetaOr(meta, nchan);
        return {
          n_channels: nchan,
          n_samples: nSamplesFdt,
          sampling_frequency: fs,
          duration_s: nSamplesFdt / fs,
          bytes_per_sample: BYTES_PER_SAMPLE,
          url: effectiveFdtUrl,
          channel_labels: labels,
          bids_channels: meta.channels || null,
          readWindow: async (startSample, nSamplesWindow, opts) => {
            const win = ChannelBuffers.clampWindow(startSample, nSamplesWindow, nSamplesFdt);
            if (!win) return ChannelBuffers.empty(nchan);
            return readInterleavedWindow(effectiveFdtUrl, nchan, win.start, win.nWin, opts);
          },
        };
      }
      throw e;
    }

    const nbchan = eeg.nbchan;
    // Sidecar values are advisory: warn on mismatch, but trust the
    // .set (it's the actual on-disk header). When the sidecar is
    // absent entirely (standalone .set), there's nothing to warn about.
    if (nChannelsFromSidecar != null && nbchan !== nChannelsFromSidecar) {
      console.warn(
        `EEGLAB inline .set: nbchan=${nbchan} disagrees with _channels.tsv ` +
        `(${nChannelsFromSidecar}); trusting the .set.`
      );
    }
    if (fsFromSidecar != null && Math.abs(eeg.srate - fsFromSidecar) > 0.5) {
      console.warn(
        `EEGLAB inline .set: srate=${eeg.srate} Hz disagrees with _eeg.json ` +
        `(${fsFromSidecar} Hz); trusting the .set.`
      );
    }
    const fs = eeg.srate;

    // Convert non-Float32 inputs (int16 / int32 / double) up-front so
    // the source typed array can be GC'd. sliceColumnMajor would
    // promote to Float32 implicitly at element-assignment anyway, but
    // that path keeps both the source AND the destination buffers in
    // memory simultaneously during the slice; converting now bounds
    // peak memory to the destination size only.
    const data32 = eeg.dataClass === 'single' ? eeg.data : Float32Array.from(eeg.data);
    const nSamples = eeg.pnts * eeg.trials;
    const expectedLen = nbchan * nSamples;
    if (data32.length !== expectedLen) {
      throw new Error(
        `EEGLAB inline .set: data length ${data32.length} != nbchan(${nbchan}) × pnts(${eeg.pnts}) × trials(${eeg.trials})`
      );
    }
    const trialsHint = eeg.trials > 1 ? eeg.trials : null;
    if (trialsHint) {
      console.warn(
        `EEGLAB inline .set is epoched (${trialsHint} trials); v1 flattens to continuous.`
      );
    }
    const duration_s = nSamples / fs;

    // Channel labels: prefer the BIDS sidecar (gives types + units),
    // otherwise fall back to Ch1..ChN. Note: EEGLAB's EEG.chanlocs
    // struct-array carries real labels in MATLAB but the current
    // MatV5 parser only reads the first element of a struct array,
    // so we can't extract per-channel labels from there yet — tracked
    // as a follow-up. Defaulting to indexed labels lets standalone
    // .set files (no BIDS sidecar) at least open and render.
    const channelLabels = ChannelLabels.fromMetaOr(meta, nbchan);

    return {
      n_channels: nbchan,
      n_samples: nSamples,
      sampling_frequency: fs,
      duration_s,
      bytes_per_sample: 4,
      trials_hint: trialsHint,
      url: setUrl,
      channel_labels: channelLabels,
      bids_channels: meta.channels || null,
      readWindow: async (startSample, nSamplesWindow) => {
        const win = ChannelBuffers.clampWindow(startSample, nSamplesWindow, nSamples);
        if (!win) return ChannelBuffers.empty(nbchan);
        return sliceColumnMajor(data32, nbchan, win.start, win.nWin);
      },
    };
  }

  // Slice an in-memory column-major (channels-major) Float32 array.
  // Same memory layout as the de-interleaved .fdt path output: one
  // Float32Array per channel, allocated through ChannelBuffers.
  function sliceColumnMajor(flat, nChannels, startSample, nWin) {
    const out = ChannelBuffers.alloc(nChannels, nWin);
    for (let s = 0; s < nWin; s++) {
      const base = (startSample + s) * nChannels;
      for (let c = 0; c < nChannels; c++) {
        out[c][s] = flat[base + c];
      }
    }
    return out;
  }

  // Returns one Float32Array per channel as views over a single
  // backing buffer — the renderer can subscript them at draw time
  // without copying, and we get one allocation per pan instead of
  // n_channels small ones.
  async function readInterleavedWindow(url, nChannels, startSample, nWin, opts) {
    const byteStart = startSample * nChannels * BYTES_PER_SAMPLE;
    const expectedBytes = nWin * nChannels * BYTES_PER_SAMPLE;
    const buf = await HttpRange.rangeFetch(url, byteStart, byteStart + expectedBytes - 1, expectedBytes, opts);
    const interleaved = new Float32Array(buf);
    const out = ChannelBuffers.alloc(nChannels, nWin);
    ChannelDecode.deinterleaveInto(out, interleaved, nChannels, nWin, null);
    return out;
  }

  // Underscore prefix marks "stable for tests, not for production
  // callers". Production code consumes `open()` only.
  api._classifyDurationMismatch = classifyDurationMismatch;
  api._sliceColumnMajor = sliceColumnMajor;

  // Streaming decode for EEGLAB .fdt (channel-interleaved Float32).
  // Yields { firstSampleIdx, lastSampleIdx, channels } as bytes arrive.
  // Each chunk is decoded by de-interleaving complete frames (nCh * 4 bytes).
  // STREAM_BATCH_FRAMES controls how many frames to accumulate before yielding.
  const STREAM_BATCH_FRAMES = 512;

  async function* streamInterleavedWindow(url, nChannels, nSamples, startSample, nWinReq, opts) {
    const win = ChannelBuffers.clampWindow(startSample, nWinReq, nSamples);
    if (!win) return;
    const { start, nWin } = win;

    const byteStart = start * nChannels * BYTES_PER_SAMPLE;
    const expectedBytes = nWin * nChannels * BYTES_PER_SAMPLE;
    const frameSize = nChannels * BYTES_PER_SAMPLE;

    let leftover = new Uint8Array(0);
    let outSamples = 0;

    for await (const { bytes } of HttpRange.rangeFetchStreaming(
      url, byteStart, byteStart + expectedBytes - 1, opts
    )) {
      const boundary = StreamingUtils.decodeChunkBoundary(leftover, bytes, frameSize);
      leftover = boundary.leftover;
      const completeBytes = boundary.completeRecordBytes;
      const nFrames = Math.floor(completeBytes.length / frameSize);
      if (nFrames === 0) continue;

      // Decode in batches to limit memory pressure
      let fOff = 0;
      while (fOff < nFrames && outSamples < nWin) {
        const batchFrames = Math.min(STREAM_BATCH_FRAMES, nFrames - fOff, nWin - outSamples);
        const batchU8 = completeBytes.subarray(fOff * frameSize, (fOff + batchFrames) * frameSize);
        const interleaved = new Float32Array(batchU8.buffer, batchU8.byteOffset, batchFrames * nChannels);
        const out = ChannelBuffers.alloc(nChannels, batchFrames);
        ChannelDecode.deinterleaveInto(out, interleaved, nChannels, batchFrames, null);
        const firstSampleIdx = start + outSamples;
        const lastSampleIdx = firstSampleIdx + batchFrames - 1;
        outSamples += batchFrames;
        yield { firstSampleIdx, lastSampleIdx, channels: out };
        fOff += batchFrames;
      }
    }
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.EEGLABReader = api;
})();
