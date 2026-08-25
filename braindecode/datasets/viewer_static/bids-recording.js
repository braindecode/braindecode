/* ============================================================
   bids-recording.js — fetch BIDS sidecars for a single EEG run
   straight from a static URL (e.g. OpenNeuro S3) and produce a
   uniform metadata object the format-specific binary readers
   consume. No backend, no auth.

   Sidecars covered (BIDS appendix):
     <prefix>_eeg.json          recording metadata (SamplingFrequency, …)
     <prefix>_channels.tsv      per-channel name/type/units/status
     <prefix>_events.tsv        events, BIDS-canonical
     <prefix>_electrodes.tsv    3D positions (delegated to bids-loader.js)
     <prefix>_coordsystem.json  coordinate space (delegated)

   URL grammar (in resolveTargets):
     ?eeg=<https://…/<prefix>_eeg.<ext>>
       → derive every sidecar from the basename.
     ?dataset=ds00XXXX&sub=01&ses=01&task=rest&run=01&ext=set
       → assemble the OpenNeuro S3 URL from BIDS path conventions.
     ?demo=<fixture-id>
       → look up a small bundled fixture under test-data/.

   Inheritance principle (BIDS): JSON/TSV metadata may live at any
   directory level above the raw file, with deeper files overriding
   shallower ones, and entity prefixes may be progressively stripped.
   We walk the tree from deepest → root and take the first hit per
   sidecar. Bound the walk at 3 dir levels up (covers sub/ses/eeg →
   sub → root, the only legal BIDS depths) so a malformed URL can't
   wander into the bucket root. Documented gap: when the same sidecar
   exists at multiple levels we don't merge fields, we just take the
   deepest — full BIDS merge is out of scope for v1.
   ============================================================ */
(function () {
  'use strict';

  const api = {};

  // Sidecar fetching delegates to HttpRange so the inheritance walk
  // works uniformly over real HTTPS URLs (force-cache on immutable
  // OpenNeuro buckets) and drag-dropped local Blobs (registered with
  // a synthetic localdrop.invalid URL).
  const fetchTextOrNull = HttpRange.fetchTextOrNull;

  // ---- URL plumbing -------------------------------------------
  // Strip the trailing _{suffix}.<ext> off the filename so we get the
  // BIDS entity prefix the sidecars share, e.g.
  //   sub-01_ses-01_task-rest_run-01_eeg.set
  //     → sub-01_ses-01_task-rest_run-01
  // Supports _eeg, _ieeg, _emg, and other electrophysiology suffixes.
  // Returns { dir, prefix, ext, suffix } where dir always ends with '/'.
  api.parsePhysioUrl = function (physioUrl) {
    // CTF MEG directory bundle: URLs look like
    //   .../<entities>_meg.ds/<entities>_meg.meg4
    // We surface ext='ds' (the bundle is what the reader registers
    // against in READERS) but dir = the *parent meg directory* so
    // BIDS sidecar inheritance walks the meg/ → ses/ → sub/ → root
    // chain, never into the bundle. The bundle itself owns the
    // .res4/.meg4/.mrk/BadChannels siblings — those are resolved by
    // ctf.js, not the inheritance walker.
    const ctf = /^(.*\/)([^/]+?)_(eeg|ieeg|emg|meg|nirs)\.ds\/\2_\3\.meg4$/.exec(physioUrl);
    if (ctf) return { dir: ctf[1], prefix: ctf[2], suffix: ctf[3], ext: 'ds' };
    // MEF3 iEEG directory bundle: URLs look like
    //   .../<entities>_ieeg.mefd/<segment>/...
    // Mirrors the .ds/ handling above — surface ext='mefd', and keep
    // dir = the parent ieeg/ directory so BIDS sidecar inheritance
    // walks the canonical chain (never into the bundle).
    const mefd = /^(.*\/)([^/]+?)_(eeg|ieeg|emg|meg|nirs)\.mefd\//.exec(physioUrl);
    if (mefd) return { dir: mefd[1], prefix: mefd[2], suffix: mefd[3], ext: 'mefd' };
    // BTi/4D Neuroimaging MEG bundle: directory with NO extension. URLs
    // look like .../sub-XX_task-YYY_meg/config or .../c,rfDC (the binary
    // data file uses a comma-prefixed naming scheme). We surface ext='bti'
    // and dir = the BTi bundle directory itself (parent of the config
    // file). Sidecar inheritance walks up from there.
    const bti = /^(.*\/)([^/]+?)_(eeg|ieeg|emg|meg|nirs)\/(?:config|c,rf[A-Za-z0-9.]+)$/.exec(physioUrl);
    if (bti) return { dir: bti[1], prefix: bti[2], suffix: bti[3], ext: 'bti' };
    // Primary: BIDS canonical `<prefix>_{suffix}.<ext>` form.
    // Matches any suffix (eeg, ieeg, emg, meg, nirs, etc.)
    const m = /^(.*\/)([^/]+?)_(eeg|ieeg|emg|meg|nirs)\.([A-Za-z0-9+]+)$/.exec(physioUrl);
    if (m) return { dir: m[1], prefix: m[2], suffix: m[3], ext: m[4].toLowerCase() };
    // Fallback: a known format file that omits the BIDS suffix
    // (e.g. local test fixtures and demo files). Sidecar inheritance will
    // find nothing at these synthetic paths — the format reader extracts
    // everything it needs from the binary header.
    const KNOWN_EXT = /\.(edf|bdf|set|vhdr|fif|fiff|snirf|con|sqd|nwb|raw|kdf)$/i;
    const m2 = /^(.*\/)([^/]+)$/.exec(physioUrl);
    if (m2 && KNOWN_EXT.test(m2[2])) {
      const dot = m2[2].lastIndexOf('.');
      return { dir: m2[1], prefix: m2[2].slice(0, dot), suffix: 'eeg', ext: m2[2].slice(dot + 1).toLowerCase() };
    }
    throw new Error(`URL is not a BIDS *_{suffix}.<ext> path: ${physioUrl}`);
  };

  // Backward compatibility: parseEegUrl now delegates to parsePhysioUrl
  api.parseEegUrl = function (eegUrl) {
    return api.parsePhysioUrl(eegUrl);
  };

  // Read once at module load — ?direct=1 is a startup flag, not a
  // hot-toggle. globalThis.location is undefined in worker / Node tests.
  const _DIRECT_S3 =
    typeof globalThis.location !== 'undefined' &&
    new URLSearchParams(globalThis.location.search).has('direct');

  // BIDS-relative path of a _{suffix}.<ext> recording. Same shape across
  // OpenNeuro (gets a bucket prefixed) and NEMAR (used as the unique
  // bidspath filter against the eegdash records API). Lifted out so
  // both call sites stay in lockstep when BIDS path conventions evolve.
  // Suffix defaults to 'eeg' but supports 'ieeg', 'emg', 'meg', 'nirs', etc.
  //
  // BIDS entity order (per the BIDS spec, MUST appear in this order in
  // the filename): sub, ses, task, acq, ce, rec, dir, run, mod, echo,
  // flip, inv, mt, part, proc, hemi, space, split, recording, chunk.
  // We thread sub/ses/task/acq/run because those are the entities the
  // viewer's URL grammar accepts. acq is critical for iEEG datasets
  // like ds003688 (clinical vs research electrode banks) and for MEG
  // datasets that distinguish noise/empty-room vs subject recordings.
  function buildBidsRelpath(params, suffix) {
    const ds   = required(params, 'dataset');
    const sub  = required(params, 'sub');
    const ses  = params.ses || null;
    const task = params.task || null;
    const acq  = params.acq || null;
    const run  = params.run || null;
    const ext  = (params.ext || 'set').toLowerCase();
    const suf  = (suffix || 'eeg').toLowerCase();
    // Map suffix to BIDS datatype directory
    const datatypeMap = {
      'eeg': 'eeg',
      'ieeg': 'ieeg',
      'emg': 'emg',
      'meg': 'meg',
      'nirs': 'nirs'
    };
    const datatype = datatypeMap[suf] || suf;
    const segs = [ds, `sub-${sub}`];
    if (ses) segs.push(`ses-${ses}`);
    segs.push(datatype);
    let entities = `sub-${sub}`;
    if (ses)  entities += `_ses-${ses}`;
    if (task) entities += `_task-${task}`;
    if (acq)  entities += `_acq-${acq}`;
    if (run)  entities += `_run-${run}`;
    if (ext === 'ds') {
      // CTF directory-bundle: the URL the reader fetches is
      // `<entities>_meg.ds/<entities>_meg.meg4`, the actual binary
      // inside the bundle. The bundle directory and the inner file
      // share the same entity-prefixed basename, just with different
      // extensions (.ds for the directory, .meg4 for the binary).
      // This mirrors how mne-python's mne.io.read_raw_ctf opens the
      // .ds/ path and discovers .meg4/.res4 siblings.
      return `${segs.join('/')}/${entities}_${suf}.ds/${entities}_${suf}.meg4`;
    }
    return `${segs.join('/')}/${entities}_${suf}.${ext}`;
  }

  // BIDS path convention on OpenNeuro:
  //   <bucket>/<dataset>/sub-<X>/[ses-<Y>/]<datatype>/<entities>_{suffix}.<ext>
  // Used by ?dataset=&sub=&ses=&task=&run=&ext= form so eegdash dataset
  // pages can deep-link without spelling out the full S3 URL.
  // Supports ?suffix= parameter for ieeg, emg, meg, nirs (defaults to eeg).
  //
  // Default: route through cdn.eegdash.org — Cloudflare Worker proxy
  // that caches OpenNeuro S3 byte-ranges at the global edge.
  // Measured cold-cache vs raw S3 (see docs/streaming-and-cdn-study.md
  // and cdn-worker/): TTFB 41-61 ms vs 333-460 ms (~10× faster), total
  // 77-176 ms vs 946-2622 ms (~13× faster), throughput 6-14 MB/s vs
  // 0.4-1.1 MB/s (~10× higher) for the same byte ranges.
  //
  // Override with ?direct=1 to force raw S3 (debugging, or in case of
  // CDN outage / caching surprise).
  api.buildOpenNeuroEegUrl = function (params) {
    const bucket = _DIRECT_S3
      ? 'https://s3.amazonaws.com/openneuro.org'
      : 'https://cdn.eegdash.org';
    const suffix = params.suffix || 'eeg';
    return `${bucket}/${buildBidsRelpath(params, suffix)}`;
  };

  // Modality auto-detection: probe all 5 BIDS electrophysiology
  // datatypes in parallel and return the first one that exists. This
  // lets URLs like ?dataset=ds003688&sub=01&ses=iemu&task=film&
  // acq=clinical&run=1&ext=vhdr work for iEEG without the user needing
  // to know ?suffix=ieeg. Uses GET + Range: bytes=0-0 (1-byte probe)
  // rather than HEAD because some CDN configurations don't support
  // HEAD for byte-range-cached objects.
  //
  // Priority order on tie: eeg, ieeg, meg, emg, nirs. Returns the
  // suffix string ('eeg'|'ieeg'|'meg'|'emg'|'nirs') of the winner,
  // or null if all 5 return 404.
  const _CANDIDATE_SUFFIXES = ['eeg', 'ieeg', 'meg', 'emg', 'nirs'];
  api.discoverSuffix = async function (params) {
    const candidates = _CANDIDATE_SUFFIXES.map(suf => ({
      suf,
      url: api.buildOpenNeuroEegUrl({ ...params, suffix: suf }),
    }));
    // Race all 5 probes. Each probe resolves to { suf, ok } where ok
    // is true iff the response is 200/206. We can't use Promise.race
    // because that returns the FIRST to resolve regardless of result;
    // we want the first SUCCESSFUL one in priority order. So we
    // Promise.allSettled then walk the results in declared order.
    const results = await Promise.all(candidates.map(async ({ suf, url }) => {
      try {
        const res = await fetch(url, {
          method: 'GET',
          headers: { Range: 'bytes=0-0' },
        });
        // 200 (server ignored Range) or 206 (partial content) → exists.
        return { suf, ok: res.status === 200 || res.status === 206 };
      } catch {
        return { suf, ok: false };
      }
    }));
    const winner = results.find(r => r.ok);
    return winner ? winner.suf : null;
  };

  // Subject auto-detection: when ?sub= is omitted from the URL, walk
  // two sources in priority order to find a real subject ID. This
  // unblocks 4+ datasets identified in docs/audit-100-datasets-2026-05-21.md
  // that use non-standard subject IDs (sub-001, sub-hc1, sub-xp101,
  // sub-283, sub-0001) and would 404 against the `sub-01` default.
  //
  // Priority:
  //   1. participants.tsv  (fast: 1 fetch, ~10KB body, present in
  //      ~95% of OpenNeuro datasets per the audit). The first data
  //      row's column 0 is `participant_id`, formatted as `sub-XXX`.
  //   2. S3 ListObjectsV2 with prefix=<dataset>/sub-  (1 fetch, XML
  //      response, ~20 keys returned). Extract the first sub-<X>/
  //      segment we see.
  //
  // Returns the bare subject ID (no `sub-` prefix) so it can be passed
  // straight to buildBidsRelpath, which re-adds the prefix. Returns
  // null when both sources fail — caller surfaces a clear error.
  //
  // Mirrors the discoverSuffix pattern: fetch + Range-byte probe is
  // intentionally NOT used here because (a) participants.tsv is small
  // enough to fetch in full, and (b) S3 ListObjectsV2 doesn't honor
  // Range. We use full GETs.
  const _S3_LIST_BASE = 'https://s3.amazonaws.com/openneuro.org';
  // bucket selection lockstep with buildOpenNeuroEegUrl
  const _PARTICIPANTS_BUCKET = _DIRECT_S3
    ? 'https://s3.amazonaws.com/openneuro.org'
    : 'https://cdn.eegdash.org';
  api.discoverSubject = async function (params) {
    const ds = required(params, 'dataset');

    // 1. participants.tsv — primary source.
    try {
      const res = await fetch(`${_PARTICIPANTS_BUCKET}/${ds}/participants.tsv`);
      if (res.ok) {
        const text = await res.text();
        const lines = text.split('\n').filter(l => l.trim());
        if (lines.length >= 2) {
          const firstCol = lines[1].split('\t')[0].trim();
          const sub = firstCol.replace(/^sub-/, '');
          if (sub) return sub;
        }
      }
    } catch { /* fall through to S3 list */ }

    // 2. S3 ListObjectsV2 — fallback. Always hits raw S3 (the CDN
    // worker doesn't proxy ?list-type=2 requests, only object GETs).
    try {
      const url = `${_S3_LIST_BASE}?list-type=2&prefix=${encodeURIComponent(ds + '/sub-')}&max-keys=20`;
      const res = await fetch(url);
      if (res.ok) {
        const xml = await res.text();
        for (const m of xml.matchAll(/<Key>([^<]+)<\/Key>/g)) {
          const sm = /^[^/]+\/sub-([^/]+)\//.exec(m[1]);
          if (sm) return sm[1];
        }
      }
    } catch { /* return null */ }

    return null;
  };

  function required(params, key) {
    const v = params[key];
    if (v == null || v === '') throw new Error(`missing required URL param: ${key}`);
    return String(v);
  }

  // ---- NEMAR ---------------------------------------------------
  // NEMAR loader extracted to bids-recording/nemar.js (Lane E2). The
  // sub-module reads shared helpers (required, buildBidsRelpath,
  // eachInheritanceLevel, assembleRecordingMetadata, fetchWithRetry)
  // from globalThis.BIDSRecording._* at call time, and is mounted
  // back onto api (isNemarDatasetId, loadNemarRecording) at the end
  // of this IIFE so the public surface and resolveTargets call sites
  // are unchanged.

  // ---- BIDS inheritance walk ----------------------------------
  // Generate every prefix shape that could legitimately host metadata
  // for this recording under BIDS inheritance. Two chains:
  //   1. Progressively drop entities from the right (run → task → ses).
  //      e.g. sub-01_ses-1_task-rest_run-1 → sub-01_ses-1_task-rest →
  //           sub-01_ses-1 → sub-01.
  //   2. Drop the leading `sub-…` (which carries no underscore prefix
  //      so the right-side strip never reaches it), then progressively
  //      drop from the right. This is what makes us find `task-X_eeg.json`
  //      sitting at the dataset root — the BIDS-canonical place to put
  //      task-level metadata that applies to every subject. Without it we
  //      miss the most common inheritance shape on real OpenNeuro data
  //      (e.g. ds002336).
  // Most-specific first so the walk's first hit is the deepest match.
  function entityVariants(prefix) {
    const tokens = tokenizePrefix(prefix);
    const out = new Set();
    if (prefix) out.add(prefix);
    // Chain 1
    for (let i = tokens.length - 1; i > 0; i--) {
      out.add(tokens.slice(0, i).join('_'));
    }
    // Chain 2 (leading-sub stripped)
    if (tokens.length && tokens[0].startsWith('sub-')) {
      const noSub = tokens.slice(1);
      for (let i = noSub.length; i > 0; i--) {
        out.add(noSub.slice(0, i).join('_'));
      }
    }
    return [...out].sort((a, b) => b.split('_').length - a.split('_').length);
  }

  // Split `sub-01_ses-1_task-rest` into ['sub-01', 'ses-1', 'task-rest'].
  // Falls back to `_`-split for unusual prefixes that don't begin with
  // `sub-`; those are rare in BIDS but we don't want to drop them silently.
  function tokenizePrefix(prefix) {
    if (!prefix) return [];
    const tokens = [];
    let rest = prefix;
    const head = /^(sub-[^_]+)/.exec(rest);
    if (head) {
      tokens.push(head[0]);
      rest = rest.slice(head[0].length);
    }
    while (rest.length) {
      const m = /^_([a-z]+-[^_]+)/i.exec(rest);
      if (!m) break;
      tokens.push(m[1]);
      rest = rest.slice(m[0].length);
    }
    if (!tokens.length) return prefix.split('_').filter(Boolean);
    return tokens;
  }

  // Generates the BIDS-inheritance probe order — for each directory
  // level (run dir → ses → sub → root), the candidate paths in
  // priority order (most-specific entity-stripped variants first,
  // then the bare suffix). Shared by the network-fetching walker
  // (fetchInheritedSidecar) and the inline-map walker NEMAR uses
  // (pickInlineSidecar) so both honour the same shape.
  function* eachInheritanceLevel(dir, prefix, suffix) {
    const variants = entityVariants(prefix);
    const bare = suffix.startsWith('_') ? suffix.substring(1) : suffix;
    let here = dir;
    for (let level = 0; level < 4; level++) {
      const paths = variants.map(v => `${here}${v}${suffix}`);
      paths.push(`${here}${bare}`);
      yield { here, paths, variants, bare };
      const parent = here.replace(/[^/]+\/$/, '');
      if (!parent || parent === here) break;
      here = parent;
    }
  }

  // At each directory level, fan out independent network probes
  // across all candidate paths and take the first non-null hit
  // (priority order preserved by walking results in the same order).
  async function fetchInheritedSidecar(dir, prefix, suffix) {
    let lastVariants, lastBare;
    for (const { paths, variants, bare } of eachInheritanceLevel(dir, prefix, suffix)) {
      lastVariants = variants; lastBare = bare;
      const results = await Promise.all(paths.map(fetchTextOrNull));
      for (let i = 0; i < results.length; i++) {
        if (results[i] != null) return { text: results[i], url: paths[i] };
      }
    }
    // Last resort: ask the eegdash backend for the dataset's known
    // sidecar inventory. Catches paths our entity-variant generator
    // didn't predict (acquisition-level files, dataset-specific naming).
    return eegdashFallback(dir, prefix, suffix, lastVariants, lastBare);
  }

  // ---- eegdash fallback ---------------------------------------
  // The eegdash FastAPI service at data.eegdash.org indexes every
  // OpenNeuro EEG dataset, including a `storage.dep_keys` listing of
  // every dataset-root sidecar. When our inheritance walk turns up
  // nothing for a sidecar, we query that record once (cached) and
  // look for a key whose filename matches one of our prefix variants
  // + suffix. If found, fetch the sidecar from OpenNeuro at the path
  // the eegdash record points us to.
  const EEGDASH_BASE = 'https://data.eegdash.org';
  const _eegdashCache = new Map();             // datasetId → record | null

  function openNeuroDatasetId(dir) {
    // Match either the raw S3 bucket OR the CDN proxy in front of it
    // (cdn.eegdash.org is a transparent edge cache for the same paths).
    const m = /^https?:\/\/(?:s3\.amazonaws\.com\/openneuro\.org|cdn\.eegdash\.org)\/([^/]+)\//.exec(dir);
    return m ? m[1] : null;
  }

  // Cache the in-flight Promise, not just the resolved record, so the
  // five sidecars doing Promise.all of inheritance walks coalesce on
  // a single eegdash request instead of stampeding it five times.
  function eegdashDataset(datasetId) {
    if (_eegdashCache.has(datasetId)) return _eegdashCache.get(datasetId);
    const p = (async () => {
      try {
        const r = await fetchWithRetry(`${EEGDASH_BASE}/api/eegdash/datasets/${datasetId}`);
        if (!r.ok) return null;
        const json = await r.json();
        return json && json.data ? json.data : null;
      } catch (e) {
        // Network failure / CORS / DNS — silently skip, the OpenNeuro
        // walk has already had its chance and the format readers can
        // still fall back to the binary header.
        return null;
      }
    })();
    _eegdashCache.set(datasetId, p);
    return p;
  }

  async function eegdashFallback(dir, prefix, suffix, variants, bare) {
    const datasetId = openNeuroDatasetId(dir);
    if (!datasetId) return null;
    const record = await eegdashDataset(datasetId);
    const depKeys = record && record.storage && record.storage.dep_keys;
    if (!Array.isArray(depKeys) || !depKeys.length) return null;
    // Most-specific first: same priority as the inheritance walk.
    const wanted = variants.map(v => `${v}${suffix}`).concat([bare]);
    for (const filename of wanted) {
      const key = depKeys.find(k => k === filename || k.endsWith(`/${filename}`));
      if (!key) continue;
      const url = `https://s3.amazonaws.com/openneuro.org/${datasetId}/${key}`;
      const text = await fetchTextOrNull(url);
      if (text != null) return { text, url };
    }
    return null;
  }

  // ---- per-recording records API (fast path) ------------------
  // The eegdash backend exposes a per-RECORDING index at
  // /api/eegdash/records, distinct from the dataset record consumed by
  // eegdashFallback above. The dataset record's dep_keys was reduced to
  // top-level files (CHANGES/README), so it can no longer locate
  // sidecars — but each per-recording record still carries the resolved
  // SamplingFrequency plus the exact sidecar paths for THAT recording.
  // Querying it once lets us skip the BIDS-inheritance 404-walk
  // entirely; it's the same source the eegdash Python client reads. We
  // only attempt it for OpenNeuro S3 / CDN datasets — anything else
  // (local fixtures, NEMAR) falls through to the inheritance walk.
  const _eegdashRecordCache = new Map();   // `${dataset} ${relpath}` → record|null

  // Dataset-root URL prefix (…/<datasetId>/) for an OpenNeuro dir,
  // recognising the same buckets as openNeuroDatasetId. Returns null
  // when dir is not an OpenNeuro S3 / CDN path.
  function openNeuroRootUrl(dir) {
    const m = /^(https?:\/\/(?:s3\.amazonaws\.com\/openneuro\.org|cdn\.eegdash\.org)\/[^/]+\/)/.exec(dir);
    return m ? m[1] : null;
  }

  // Fetch the per-recording record (cached in-flight Promise so the
  // call coalesces and a 404 — unindexed dataset — is remembered as
  // null rather than re-probed). Never throws: any failure → null,
  // which sends loadRecordingMetadata down the inheritance-walk path.
  function fetchEegdashRecord(datasetId, relpath) {
    const cacheKey = `${datasetId} ${relpath}`;
    if (_eegdashRecordCache.has(cacheKey)) return _eegdashRecordCache.get(cacheKey);
    const p = (async () => {
      try {
        const filter = encodeURIComponent(JSON.stringify({ dataset: datasetId, bids_relpath: relpath }));
        const r = await fetchWithRetry(`${EEGDASH_BASE}/api/eegdash/records?filter=${filter}&limit=1`);
        if (!r.ok) return null;
        const json = await r.json();
        const rec = json && Array.isArray(json.data) ? json.data[0] : null;
        return rec || null;
      } catch (e) {
        return null;
      }
    })();
    _eegdashRecordCache.set(cacheKey, p);
    return p;
  }

  // Build the metadata bundle from a per-recording record: SamplingFrequency
  // comes straight from the record; the sidecars it lists in
  // storage.dep_keys are fetched at their exact paths (no inheritance
  // walk). Sidecars not listed stay null — the format reader still fills
  // channel labels + sfreq from the binary header downstream, exactly as
  // the walk path does when a sidecar is absent.
  async function loadFromEegdashRecord({ eegUrl, ext, dir, prefix, rootUrl, record }) {
    const depKeys = Array.isArray(record.storage?.dep_keys) ? record.storage.dep_keys : [];
    // Resolve each sidecar from its exact dep_keys path only — no inheritance
    // walk, so the open path stays fast. Sidecars absent from dep_keys stay
    // null: channels are recovered from the record's ch_names below, and
    // events/electrodes/coordsystem degrade gracefully (the format reader
    // still supplies channel labels + EDF/annotation events downstream).
    const fetchDep = async (suffix) => {
      const key = depKeys.find(k => k.endsWith(suffix));
      if (!key) return null;
      const url = `${rootUrl}${key}`;
      const text = await fetchTextOrNull(url);
      return text == null ? null : { text, url };
    };
    const [channels, events, electrodes, coordsystem] = await Promise.all([
      fetchDep('_channels.tsv'),
      fetchDep('_events.tsv'),
      fetchDep('_electrodes.tsv'),
      fetchDep('_coordsystem.json'),
    ]);
    const sfreq = record.sampling_frequency;
    const duration = (record.ntimes != null && sfreq) ? record.ntimes / sfreq : null;
    return assembleRecordingMetadata({
      eeg_url: eegUrl, ext, dir, prefix,
      hits: { eeg_json: null, channels, events, electrodes, coordsystem },
      recordMeta: { sampling_frequency: sfreq, recording_duration: duration, ch_names: record.ch_names },
    });
  }

  // ---- _eeg.json ----------------------------------------------
  // Required field per BIDS: SamplingFrequency. Everything else is
  // recorded for display but not load-bearing. We also pass through
  // unknown keys so dataset-specific extensions stay visible.
  api.parseEegJson = function (obj) {
    if (!obj || typeof obj !== 'object') throw new Error('_eeg.json is not an object');
    let fs = obj.SamplingFrequency;
    // Lenient: an invalid sidecar SamplingFrequency is NOT fatal. Most
    // readers can derive sfreq from the file itself (EEGLAB EEG.srate,
    // BrainVision SamplingInterval, EDF/BDF record duration). Treat the
    // sidecar value as a hint that may be overridden. Observed in the
    // wild on ds006466 where the sidecar value is `null`.
    if (fs != null && (!isFinite(fs) || fs <= 0)) {
      console.warn(
        `_eeg.json: SamplingFrequency is invalid (${fs}); will derive from file.`
      );
      fs = null;
    }
    return {
      sampling_frequency: fs,
      recording_duration: numericOrNull(obj.RecordingDuration),
      eeg_reference: obj.EEGReference || null,
      power_line_frequency: numericOrNull(obj.PowerLineFrequency),
      software_filters: obj.SoftwareFilters || null,
      manufacturer: obj.Manufacturer || null,
      raw: obj,
    };
  };

  function numericOrNull(v) {
    return (typeof v === 'number' && isFinite(v)) ? v : null;
  }

  // ---- _channels.tsv ------------------------------------------
  // Columns: name, type, units, status (status_description), low_cutoff,
  // high_cutoff, sampling_frequency are common. We tolerate column
  // reordering and extra columns (BIDS allows both).
  // The row order is the channel order in the binary file, which the
  // format readers MUST honour — never reorder during display.
  api.parseChannelsTsv = function (text) {
    const rows = parseTsv(text);
    if (rows.length < 2) throw new Error('_channels.tsv has no data rows');
    const header = rows[0].map(h => h.trim().toLowerCase());
    const idx = (k) => header.indexOf(k);
    const iName = idx('name'), iType = idx('type'), iUnits = idx('units');
    if (iName < 0) throw new Error('_channels.tsv missing required column: name');

    const iStatus = idx('status');
    const iLow  = idx('low_cutoff');
    const iHigh = idx('high_cutoff');
    const iFs   = idx('sampling_frequency');

    const channels = [];
    for (let i = 1; i < rows.length; i++) {
      const c = rows[i];
      const name = (c[iName] || '').trim();
      if (!name) continue;
      channels.push({
        index: channels.length,
        name,
        type:   iType   >= 0 ? bidsCell(c[iType])   : null,
        units:  iUnits  >= 0 ? bidsCell(c[iUnits])  : null,
        status: iStatus >= 0 ? (bidsCell(c[iStatus]) || 'good') : 'good',
        low_cutoff:  iLow  >= 0 ? parseFloatOrNull(c[iLow])  : null,
        high_cutoff: iHigh >= 0 ? parseFloatOrNull(c[iHigh]) : null,
        sampling_frequency: iFs >= 0 ? parseFloatOrNull(c[iFs]) : null,
      });
    }
    if (!channels.length) throw new Error('_channels.tsv produced zero channels');
    return channels;
  };

  // ---- _events.tsv --------------------------------------------
  // Required columns per BIDS: onset (s), duration (s). trial_type is
  // common and we promote it to the display label when present.
  api.parseEventsTsv = function (text) {
    const rows = parseTsv(text);
    if (rows.length < 2) return [];
    const header = rows[0].map(h => h.trim().toLowerCase());
    const idx = (k) => header.indexOf(k);
    const iOnset = idx('onset'), iDur = idx('duration');
    if (iOnset < 0) throw new Error('_events.tsv missing required column: onset');

    const iLabel = idx('trial_type') >= 0 ? idx('trial_type') : idx('value');
    const iSample = idx('sample');

    const events = [];
    for (let i = 1; i < rows.length; i++) {
      const c = rows[i];
      const onset = parseFloat(c[iOnset]);
      if (!isFinite(onset)) continue;
      events.push({
        onset,
        duration: iDur >= 0 ? (parseFloatOrNull(c[iDur]) || 0) : 0,
        label:    iLabel >= 0 ? bidsCell(c[iLabel]) : null,
        sample:   iSample >= 0 ? parseFloatOrNull(c[iSample]) : null,
      });
    }
    return events;
  };

  // ---- shared TSV plumbing ------------------------------------
  // BIDS specifies tab-separated, but real-world files (e.g. ds002336)
  // sometimes use multi-space alignment instead. Detect at file level:
  // if the header row has a tab, parse as TSV; otherwise fall back to
  // whitespace-splitting. Cells are also surface-trimmed and unquoted —
  // some sources wrap values like 'Fp1' in literal single quotes that
  // BIDS doesn't define but does see in the wild.
  function parseTsv(text) {
    const lines = text
      .split(/\r?\n/)
      .filter(l => l.length > 0 && !l.startsWith('#'));
    if (!lines.length) return [];
    const sep = lines[0].includes('\t') ? /\t/ : /\s+/;
    return lines.map(l => l.split(sep).map(stripQuotes));
  }

  function stripQuotes(s) {
    s = s.trim();
    if (s.length >= 2) {
      const q = s[0];
      if ((q === '"' || q === "'") && s[s.length - 1] === q) return s.slice(1, -1);
    }
    return s;
  }

  function bidsCell(v) {
    if (v == null) return null;
    const s = String(v).trim();
    if (!s || s.toLowerCase() === 'n/a') return null;
    return s;
  }

  function parseFloatOrNull(v) {
    if (v == null) return null;
    const s = String(v).trim();
    if (!s || s.toLowerCase() === 'n/a') return null;
    const n = parseFloat(s);
    return isFinite(n) ? n : null;
  }

  // ---- top-level loader ---------------------------------------
  // Fetches every BIDS sidecar that goes with a recording and returns
  // a single metadata bundle. Optional sidecars (electrodes, coordsys,
  // events) are absent → null fields, never an error. The required
  // sidecar is _eeg.json: without SamplingFrequency we can't render
  // a time axis at all, so we surface that as a hard failure.
  api.loadRecordingMetadata = async function (eegUrl) {
    const { dir, prefix, ext } = api.parseEegUrl(eegUrl);

    // Fast path: the per-recording records API returns the resolved
    // SamplingFrequency and the exact sidecar paths in a single request,
    // so we can skip the BIDS-inheritance 404-walk (which is especially
    // wasteful for MEG/iEEG, where the walk probes _eeg.json and never
    // finds it). Only for OpenNeuro S3/CDN datasets; falls through to the
    // walk when the dataset isn't indexed or the record carries no usable
    // sampling frequency.
    const rootUrl = openNeuroRootUrl(dir);
    if (rootUrl) {
      const datasetId = openNeuroDatasetId(dir);
      const relpath = eegUrl.split('?')[0].slice(rootUrl.length);
      const record = datasetId ? await fetchEegdashRecord(datasetId, relpath) : null;
      if (record && record.sampling_frequency != null) {
        return loadFromEegdashRecord({ eegUrl, ext, dir, prefix, rootUrl, record });
      }
    }

    // Fetch in parallel — each call walks the inheritance tree
    // independently, so a missing run-level file falls through to the
    // dataset root (BIDS principle). Fetches are tiny; the CORS
    // round-trip dominates so parallel is the right call.
    const [eeg_json, channels, events, electrodes, coordsystem] =
      await Promise.all([
        fetchInheritedSidecar(dir, prefix, '_eeg.json'),
        fetchInheritedSidecar(dir, prefix, '_channels.tsv'),
        fetchInheritedSidecar(dir, prefix, '_events.tsv'),
        fetchInheritedSidecar(dir, prefix, '_electrodes.tsv'),
        fetchInheritedSidecar(dir, prefix, '_coordsystem.json'),
      ]);
    if (eeg_json == null) {
      // Soft-required: format-specific readers (BrainVision .vhdr,
      // EDF header, EEGLAB .set) carry SamplingFrequency and channel
      // counts inline, so the binary reader can fill these in. We
      // pass a stub through and let the reader override.
      console.warn(`No _eeg.json found via BIDS inheritance for ${eegUrl}; deferring to format header.`);
    }
    return assembleRecordingMetadata({
      eeg_url: eegUrl, ext, dir, prefix,
      hits: { eeg_json, channels, events, electrodes, coordsystem },
    });
  };

  // Parses the five canonical BIDS sidecars from already-fetched
  // {text, url} hits (any may be null) into the metadata bundle the
  // viewer + format readers consume. Shared between the OpenNeuro
  // network walker (loadRecordingMetadata) and the NEMAR inline-map
  // walker (loadNemarRecording) — the only thing that varies between
  // them is *how* hits get materialised, not how they're parsed.
  function assembleRecordingMetadata({ eeg_url, ext, dir, prefix, hits, recordMeta = null }) {
    const { eeg_json: eegJsonHit, channels: channelsHit, events: eventsHit,
            electrodes: electrodesHit, coordsystem: coordSysHit } = hits;

    let eegJson;
    if (eegJsonHit) {
      try {
        eegJson = api.parseEegJson(JSON.parse(eegJsonHit.text));
      } catch (e) {
        throw new Error(`Bad _eeg.json at ${eegJsonHit.url}: ${e.message}`);
      }
    } else if (recordMeta) {
      // SamplingFrequency resolved by the eegdash records API — no
      // _eeg.json/_meg.json text was fetched. Remaining fields stay null
      // and are filled by the format reader or optional sidecars.
      eegJson = { sampling_frequency: recordMeta.sampling_frequency ?? null,
                  recording_duration: recordMeta.recording_duration ?? null,
                  eeg_reference: null, power_line_frequency: null,
                  software_filters: null, manufacturer: null, raw: {} };
    } else {
      eegJson = { sampling_frequency: null, recording_duration: null, eeg_reference: null,
                  power_line_frequency: null, software_filters: null, manufacturer: null, raw: {} };
    }

    let channels = channelsHit ? api.parseChannelsTsv(channelsHit.text) : null;
    // Fallback: with no _channels.tsv anywhere, synthesise a minimal channel
    // list from the record's ch_names so the channel panel still populates
    // (names only — type/units/status come from _channels.tsv when present).
    if (!channels && recordMeta && Array.isArray(recordMeta.ch_names) && recordMeta.ch_names.length) {
      channels = recordMeta.ch_names.map((name, index) => ({
        index, name, type: null, units: null, status: 'good',
        low_cutoff: null, high_cutoff: null, sampling_frequency: null,
      }));
    }
    const events = eventsHit ? api.parseEventsTsv(eventsHit.text) : [];

    let electrodes = null, coordsystem = null;
    if (electrodesHit && typeof BIDSLoader !== 'undefined') {
      try { electrodes = BIDSLoader.parseElectrodesTSV(electrodesHit.text); }
      catch (e) { console.warn(`electrodes.tsv unparseable, skipping: ${e.message}`); }
    }
    if (coordSysHit && typeof BIDSLoader !== 'undefined') {
      try { coordsystem = BIDSLoader.parseCoordsystem(coordSysHit.text); }
      catch (e) { console.warn(`coordsystem.json unparseable, skipping: ${e.message}`); }
    }

    return {
      eeg_url, ext, dir, prefix,
      eeg_json: eegJson,
      channels, events, electrodes, coordsystem,
      // Provenance: which key the walker found each sidecar at —
      // a real https URL for OpenNeuro, an `inline:<rawKey>` tag for
      // NEMAR. renderProvenance treats both as opaque labels.
      sidecar_sources: {
        eeg_json:    eegJsonHit?.url   ?? (recordMeta ? 'eegdash:record' : null),
        channels:    channelsHit?.url  ?? null,
        events:      eventsHit?.url    ?? null,
        electrodes:  electrodesHit?.url ?? null,
        coordsystem: coordSysHit?.url  ?? null,
      },
    };
  }

  // ---- internal helpers exposed for unit testing --------------
  // Underscore-prefixed: stable contract for the test suite, no
  // implicit promise to keep them across releases. Production code
  // should consume the public surface (loadRecordingMetadata, …).
  api._entityVariants  = entityVariants;
  api._tokenizePrefix  = tokenizePrefix;
  api._parseTsv        = parseTsv;

  // ---- URL parameter resolver ---------------------------------
  // Walks the URL params on page load and returns a normalized
  // descriptor the bootstrap code feeds into loadRecordingMetadata.
  // Returns null when no params are present (cold viewer state).
  // Allowed URL protocols for query-param-supplied recording URLs. The
  // viewer accepts http(s) only — data:/blob:/file:/javascript:/etc.
  // are rejected to prevent (a) cross-origin SSRF-from-victim where a
  // malicious link causes the browser to fetch an attacker URL with
  // the viewer's referer/cookies, and (b) data:/blob: payloads that
  // could bypass the format-reader's bounds checks.
  //
  // Fix A3 (HIGH): the previous gate accepted ANY string starting with
  // '/' (and not '//') as "same-origin relative" — including paths like
  // `/cdn-worker/.env` that read attacker-chosen same-origin files via
  // a single-click URL. It was also case-sensitive, letting `HTTP://`
  // and `JaVaScRiPt:` slip through `new URL().protocol` (which would
  // lowercase the scheme). The replacement resolves every input string
  // against the document baseURI and only accepts the resolved URL when
  // its protocol is http: or https: — relative paths inherit the
  // current origin and scheme, so they still work, but cannot be used
  // to smuggle non-http schemes.
  function isAllowedProtocol(urlString) {
    if (typeof urlString !== 'string' || urlString.length === 0) return false;
    // Disallow scheme-relative early — `//evil.com/x` would inherit the
    // current scheme but go to attacker origin.
    if (urlString.startsWith('//')) return false;
    let u;
    try {
      const base = (typeof globalThis !== 'undefined' && globalThis.location && globalThis.location.href)
        ? globalThis.location.href
        : 'https://example.invalid/';
      u = new URL(urlString, base);
    } catch {
      return false;
    }
    // After resolution, only http/https survive. URL.protocol is always
    // lowercase per spec, so this implicitly handles uppercase schemes.
    return u.protocol === 'http:' || u.protocol === 'https:';
  }
  api._isAllowedProtocol = isAllowedProtocol;

  // Network resilience: NEMAR's data.nemar.org occasionally returns 404
  // 'Version not published' on the latest manifest; OpenNeuro S3 returns
  // 503 under load. Wrap fetch() with bounded exponential backoff —
  // 3 retries (200, 400, 800 ms) on transient 5xx and on network errors.
  // 4xx (other than 429) is treated as terminal — the URL is wrong, no
  // point retrying.
  async function fetchWithRetry(url, opts) {
    const TRANSIENT = new Set([429, 502, 503, 504]);
    const delays = [200, 400, 800];
    let lastErr;
    for (let attempt = 0; attempt <= delays.length; attempt++) {
      try {
        const res = await fetch(url, opts);
        if (res.ok) return res;
        if (TRANSIENT.has(res.status) && attempt < delays.length) {
          await new Promise(r => setTimeout(r, delays[attempt]));
          continue;
        }
        // 4xx terminal — return the response so caller can decide
        // (parsePhysioUrl or the sidecar walk often expects 404).
        return res;
      } catch (e) {
        lastErr = e;
        if (attempt < delays.length) {
          await new Promise(r => setTimeout(r, delays[attempt]));
          continue;
        }
        throw e;
      }
    }
    throw lastErr || new Error('fetchWithRetry: unreachable');
  }
  api._fetchWithRetry = fetchWithRetry;  // exposed for tests

  api.resolveTargets = function (urlSearchParams) {
    const p = urlSearchParams;
    // Support ?eeg=, ?ieeg=, ?emg= parameters for direct URL loading
    for (const suffix of ['eeg', 'ieeg', 'emg', 'meg', 'nirs']) {
      if (p.has(suffix)) {
        const url = p.get(suffix);
        if (!isAllowedProtocol(url)) {
          throw new Error(`Invalid URL protocol in ?${suffix}=; only http(s) allowed.`);
        }
        return { kind: 'url', eeg_url: url };
      }
    }
    if (p.has('dataset')) {
      const ds = p.get('dataset');
      const params = {
        dataset: ds,
        sub:     p.get('sub'),
        ses:     p.get('ses'),
        task:    p.get('task'),
        // acq is the BIDS "acquisition" entity. Critical for iEEG
        // (clinical vs research electrode banks) and MEG (subject vs
        // empty-room). Threaded through buildBidsRelpath so the
        // filename gets the `_acq-<X>_` segment in the BIDS-required
        // position between _task- and _run-.
        acq:     p.get('acq'),
        run:     p.get('run'),
        ext:     p.get('ext'),
        // NEMAR only: pin a specific manifest version. The loader
        // defaults to 'latest' when undefined and validates the
        // shape (latest|vN.N.N) before constructing the manifest URL.
        version: p.get('version') || undefined,
      };
      // Suffix policy:
      //   ?suffix=<X> explicit → use that suffix directly (no probe).
      //   No ?suffix= → kind: 'bids-path-auto' → viewer probes all 5
      //   suffixes in parallel via api.discoverSuffix and picks the
      //   winner. ds003688 (iEEG film) is the canonical case: users
      //   shouldn't need to know whether a dataset is EEG / iEEG / MEG
      //   to load it.
      const explicitSuffix = p.get('suffix');
      if (explicitSuffix) params.suffix = explicitSuffix;
      // NEMAR (nm-prefixed) datasets resolve via the eegdash records
      // API instead of a direct bucket URL — git-annex SHA addressing.
      // NEMAR manifests enumerate subjects, so subject discovery is
      // not needed for the NEMAR branch — it falls through to nemar
      // regardless of whether sub is set.
      if (api.isNemarDatasetId(ds)) {
        // NEMAR loader expects params.suffix; fall back to 'eeg' for
        // back-compat with existing NEMAR URLs (which have always
        // assumed eeg).
        params.suffix = params.suffix || 'eeg';
        return { kind: 'nemar', nemar_params: params };
      }
      // Subject discovery: when ?sub= is omitted, defer building the
      // URL to boot(), which calls api.discoverSubject and then
      // re-enters the (sub-set) resolution path. Owns both discovery
      // passes — once the sub is known, modality discovery may also
      // need to run if ?suffix= was also omitted.
      if (!params.sub) {
        return { kind: 'bids-path-discover-sub', params };
      }
      if (!explicitSuffix) {
        return { kind: 'bids-path-auto', params };
      }
      return {
        kind: 'bids-path',
        eeg_url: api.buildOpenNeuroEegUrl(params),
      };
    }
    if (p.has('demo')) {
      return { kind: 'demo', demo_id: p.get('demo') };
    }
    return null;
  };

  // ---- NEMAR mount (Lane E2) ----------------------------------
  // Expose shared helpers under api._* so bids-recording/nemar.js can
  // reach them via globalThis.BIDSRecording._… at request time. The
  // underscore prefix matches existing test seams (api._fetchWithRetry
  // was already published above) and signals "internal but stable".
  api._required = required;
  api._buildBidsRelpath = buildBidsRelpath;
  api._eachInheritanceLevel = eachInheritanceLevel;
  api._assembleRecordingMetadata = assembleRecordingMetadata;

  // Resolve the NEMAR sub-module: browser side-loads bids-recording/nemar.js
  // BEFORE this file (see index.html); Node tests reach it via require().
  // Either way, we mount isNemarDatasetId + loadNemarRecording onto api so
  // the public surface (and unit-api-surface.test.mjs) stays unchanged.
  const _Nemar = (typeof globalThis !== 'undefined' && globalThis.BIDSRecordingNemar)
    || (typeof require !== 'undefined' ? require('./bids-recording/nemar.js') : null);
  if (_Nemar) {
    api.isNemarDatasetId = _Nemar.isNemarDatasetId;
    api.loadNemarRecording = _Nemar.loadNemarRecording;
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.BIDSRecording = api;
})();
