/* ============================================================
   bids-recording/nemar.js — NEMAR (nm/on/xx) recording loader,
   extracted from bids-recording.js so the parent stays under
   the readability threshold. (Lane E2.)

   NEMAR datasets are addressed by data.nemar.org, the public
   BIDS-shaped HTTPS API the nemar-cli backend exposes for every
   *published* dataset. One fetch of the per-version
     /<id>/<version>/manifest.json
   gives us every file in the dataset as
     { path, size, checksum, checksum_algorithm, url }
   — a flat enumeration covering both git-annex bytes (large EDF/
   BDF/.set/.vhdr+.eeg/...) and inline-git sidecars (small
   JSON/TSV).

   Two URL shapes appear in the manifest:
     - git-tree:  https://raw.githubusercontent.com/nemarDatasets/<id>/<tag>/<bidsPath>
                  Has CORS — fetched as-is.
     - annex S3:  https://nemar.s3.<region>.amazonaws.com/<id>/objects/<sha>?<presign>
                  No CORS on bare S3 → routed through the cdn-worker's
                  existing /<id>/objects/<sha> proxy. That URL is
                  content-addressed so it's infinitely cacheable; we
                  drop the per-request presigned query string entirely.

   The data.nemar.org route itself also lacks
   Access-Control-Allow-Origin (verified May 2026), so manifest.json
   is also fetched through the cdn-worker (/data/<id>/<ver>/manifest.json
   proxy).

   Pre-data.nemar.org we used the eegdash records API + git-annex SHA
   lookups; that path required eegdash to have ingested every record
   and never worked for on-prefixed / xx-prefixed mirrors. The
   manifest-driven path unifies the three dataset families.

   Module wiring: bids-recording.js mounts our exports onto its `api`
   object before publishing `globalThis.BIDSRecording`, and exposes its
   own shared helpers under `BIDSRecording._*` so we can call them at
   request time without a circular dependency at IIFE-init.
   ============================================================ */
'use strict';
(function () {
  // Lockstep with cdn-worker's VALID_NEMAR (objects/) and
  // VALID_NEMAR_API (/data/) regexes — a 5- or 7-digit id, or any
  // prefix outside {nm,on,xx}, would 404 against the worker.
  //   nm = native NEMAR ingest
  //   on = OpenNeuro mirror (added in nemar-cli sprint #514 May 2026
  //        once #516 unblocked their D1 metadata + LLM enrichment)
  //   xx = sandbox (used by the nemar-cli test suite)
  function isNemarDatasetId(id) {
    return typeof id === 'string' && /^(?:nm|on|xx)\d{6}$/.test(id);
  }

  const _NEMAR_FETCH_TIMEOUT_MS = 15000;
  // Reject manifests larger than this — typical real manifests are
  // ~100KB-2MB; anything orders of magnitude bigger is corrupt or
  // hostile and would OOM the browser tab if parsed.
  const _NEMAR_MANIFEST_MAX_BYTES = 32 * 1024 * 1024;
  // Hosts the viewer is willing to fetch NEMAR bytes from. Centralised
  // so a host change is a single-line edit and the worker / loader
  // can't drift out of lockstep.
  const _CDN_BASE = 'https://cdn.eegdash.org';
  const _NEMAR_DATA_BASE = 'https://data.nemar.org';
  // Versions accepted by data.nemar.org: 'latest' or 'vMAJOR.MINOR.PATCH'.
  // Lockstep with the cdn-worker's VALID_NEMAR_API version segment so
  // a value rejected here would also be rejected at the proxy.
  const _NEMAR_VERSION_SHAPE = /^(?:latest|v\d+\.\d+\.\d+)$/;

  // Manifest-listed `url` values must match one of these two shapes
  // AND name the requested dataset — anything else is a trust-boundary
  // failure (the manifest is server-provided; we refuse to fetch from
  // arbitrary hosts OR from a different dataset's storage). Both
  // regexes capture the dsId so the caller can cross-check.
  //   git-tree: $1 = repo-name (== dsId for native NEMAR repos)
  //   annex S3: $1 = bucket-key dataset id, $2 = SHA-key
  const _GIT_TREE_URL = /^https:\/\/raw\.githubusercontent\.com\/nemarDatasets\/([^/]+)\/[^/]+\//;
  const _ANNEX_S3_URL = /^https:\/\/nemar\.s3(?:\.[a-z0-9-]+)?\.amazonaws\.com\/([^/]+)\/objects\/([A-Za-z0-9._-]+)(?:\?|$)/;

  // Read once at module load — ?direct=1 is a startup flag, not a
  // hot-toggle. globalThis.location is undefined in worker / Node tests.
  // Mirrors the same compute in bids-recording.js so the two stay in
  // lockstep without inter-module coupling.
  const _DIRECT_S3 =
    typeof globalThis !== 'undefined' &&
    typeof globalThis.location !== 'undefined' &&
    new URLSearchParams(globalThis.location.search).has('direct');

  // ---- BIDSRecording bridge ------------------------------------
  // Resolve shared helpers from the parent module at call time (not
  // module-load time) so the load order — nemar.js BEFORE
  // bids-recording.js — keeps working: at IIFE-init the global is
  // undefined; by the time isNemarDatasetId / loadNemarRecording is
  // ever called from production code, bids-recording.js has published
  // its api object via globalThis.BIDSRecording.
  function _br() {
    const br = (typeof globalThis !== 'undefined') ? globalThis.BIDSRecording : null;
    if (!br) {
      throw new Error('BIDSRecording (bids-recording.js) must load before NEMAR functions are invoked');
    }
    return br;
  }

  // Transform a manifest entry's `url` into one the viewer can fetch
  // cross-origin. Returns null when the URL fails the trust-boundary
  // check (caller surfaces a clear error with context).
  function transformManifestUrl(entryUrl, dsId) {
    if (typeof entryUrl !== 'string' || !entryUrl) return null;
    const git = _GIT_TREE_URL.exec(entryUrl);
    if (git && git[1] === dsId) return entryUrl;
    const annex = _ANNEX_S3_URL.exec(entryUrl);
    if (annex && annex[1] === dsId) {
      // ?direct=1 keeps the presigned URL intact for Node tests, where
      // CORS doesn't apply and we want to bypass the cdn-worker entirely.
      if (_DIRECT_S3) return entryUrl;
      return `${_CDN_BASE}/${dsId}/objects/${annex[2]}`;
    }
    return null;
  }

  function nemarManifestUrl(dsId, version) {
    const ver = version || 'latest';
    if (!_NEMAR_VERSION_SHAPE.test(ver)) {
      throw new Error(
        `NEMAR version param "${ver}" is invalid — expected "latest" or "vMAJOR.MINOR.PATCH".`
      );
    }
    if (_DIRECT_S3) return `${_NEMAR_DATA_BASE}/${dsId}/${ver}/manifest.json`;
    return `${_CDN_BASE}/data/${dsId}/${ver}/manifest.json`;
  }

  // Single-shot loader for NEMAR recordings. Returns a metadata
  // bundle in the same shape as loadRecordingMetadata (so the rest
  // of the viewer is format-agnostic), with one NEMAR-specific
  // addition: meta.sibling_urls (filename → URL) so format readers
  // with split layouts (BrainVision .vhdr+.eeg, EEGLAB .set+.fdt)
  // can resolve the sibling without doing path arithmetic against
  // a SHA-keyed URL.
  // postMessage-safe: only plain JSON, no functions or closures.
  async function loadNemarRecording(params) {
    const BR = _br();
    const ds = BR._required(params, 'dataset');
    if (!isNemarDatasetId(ds)) {
      throw new Error(`not a NEMAR-style dataset id: ${ds}`);
    }
    const bidspath = BR._buildBidsRelpath(params);
    // Manifest paths are dataset-relative — strip the leading <id>/.
    const innerPath = bidspath.startsWith(`${ds}/`)
      ? bidspath.slice(ds.length + 1)
      : bidspath;

    const manifest = await fetchNemarManifest(ds, params.version);

    const lastSlash = innerPath.lastIndexOf('/');
    const dir = lastSlash >= 0 ? innerPath.slice(0, lastSlash + 1) : '';
    const basename = lastSlash >= 0 ? innerPath.slice(lastSlash + 1) : innerPath;
    const prefixMatch = /^(.+?)_(?:eeg|ieeg|emg|meg|nirs)\.[^.]+$/.exec(basename);
    const prefix = prefixMatch ? prefixMatch[1] : basename.replace(/\.[^.]+$/, '');
    const ext = (basename.slice(basename.lastIndexOf('.') + 1) ||
                 params.ext || 'set').toLowerCase();

    // Single pass over the manifest: build the path index AND collect
    // same-directory siblings. sibling_urls feeds BrainVision .vhdr
    // (which references .eeg by bare filename) and EEGLAB .set+.fdt;
    // restricting to the recording's exact dir prevents cross-subject
    // basename collisions that wider scopes would cause.
    const byPath = new Map();
    const sibling_urls = {};
    for (const e of manifest) {
      if (!e || typeof e.path !== 'string') continue;
      byPath.set(e.path, e);
      if (!e.path.startsWith(dir)) continue;
      const rest = e.path.slice(dir.length);
      if (!rest || rest.includes('/')) continue;
      const u = transformManifestUrl(e.url, ds);
      if (u) sibling_urls[rest] = u;
    }

    const rawEntry = byPath.get(innerPath);
    if (!rawEntry) {
      throw new Error(
        `NEMAR manifest has no entry for ${innerPath}. ` +
        `Check dataset/sub/ses/task/run/ext URL params match a published recording, ` +
        `or pin a specific version with ?version=vX.Y.Z.`
      );
    }
    const eegUrl = transformManifestUrl(rawEntry.url, ds);
    if (!eegUrl) {
      throw new Error(`NEMAR manifest url has unrecognised shape: ${rawEntry.url}`);
    }

    // Sidecars: BIDS-inheritance walk against the manifest (deepest →
    // root, entity-stripped variants), then a network fetch of the
    // first hit's text. The provenance label in sidecar_sources keeps
    // the per-entry URL so renderProvenance shows where the value
    // came from (raw.githubusercontent.com for git-tree, cdn.eegdash
    // .org for annex). assembleRecordingMetadata is shared with the
    // OpenNeuro path so the bundle's downstream shape stays uniform.
    const sidecarPlan = [
      ['eeg_json',    '_eeg.json'],
      ['channels',    '_channels.tsv'],
      ['events',      '_events.tsv'],
      ['electrodes',  '_electrodes.tsv'],
      ['coordsystem', '_coordsystem.json'],
    ];
    const results = await Promise.all(sidecarPlan.map(
      ([, suffix]) => fetchManifestSidecar(byPath, dir, prefix, suffix, ds)
    ));
    const hits = Object.fromEntries(sidecarPlan.map(([k], i) => [k, results[i]]));

    const meta = BR._assembleRecordingMetadata({ eeg_url: eegUrl, ext, dir, prefix, hits });
    meta.sibling_urls = sibling_urls;
    return meta;
  }

  async function fetchNemarManifest(ds, version) {
    const BR = _br();
    const url = nemarManifestUrl(ds, version);
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), _NEMAR_FETCH_TIMEOUT_MS);
    let resp;
    try {
      resp = await BR._fetchWithRetry(url, { signal: ctrl.signal });
    } catch (e) {
      const reason = e.name === 'AbortError'
        ? `timed out after ${_NEMAR_FETCH_TIMEOUT_MS}ms`
        : e.message;
      throw new Error(`NEMAR manifest unreachable for ${ds}: ${reason}`);
    } finally {
      clearTimeout(timer);
    }
    if (resp.status === 404) {
      throw new Error(
        `NEMAR manifest 404 for ${ds}: dataset is unpublished, private, ` +
        `or has no minted version yet.`
      );
    }
    if (!resp.ok) {
      throw new Error(`NEMAR manifest HTTP ${resp.status} for ${ds}`);
    }
    // Guard against runaway manifest bodies (corrupt/hostile) — the
    // browser would otherwise OOM on resp.json(). Real manifests cap
    // out in single-digit MB even for 500+ recording datasets.
    const lenHeader = resp.headers && resp.headers.get && resp.headers.get('content-length');
    const declaredLen = lenHeader ? Number(lenHeader) : NaN;
    if (Number.isFinite(declaredLen) && declaredLen > _NEMAR_MANIFEST_MAX_BYTES) {
      throw new Error(
        `NEMAR manifest for ${ds} is ${declaredLen} bytes — refusing to parse ` +
        `(cap is ${_NEMAR_MANIFEST_MAX_BYTES} bytes; likely corrupt upstream).`
      );
    }
    let manifest;
    try {
      manifest = await resp.json();
    } catch (e) {
      throw new Error(
        `NEMAR manifest for ${ds} is not valid JSON (${e.message}). ` +
        `Upstream may be misconfigured.`
      );
    }
    if (!Array.isArray(manifest)) {
      throw new Error(`NEMAR manifest for ${ds} is not a JSON array`);
    }
    return manifest;
  }

  // Walk the BIDS inheritance shape against the manifest (no extra
  // network probes — the manifest is the authoritative file index for
  // the version), then fetch the matching entry's text once. Returns
  // null when nothing matches OR when the matching entry's text fetch
  // fails — assembleRecordingMetadata tolerates nulls (warns + falls
  // back to format-header values).
  async function fetchManifestSidecar(byPath, dir, prefix, suffix, ds) {
    const BR = _br();
    for (const { paths } of BR._eachInheritanceLevel(dir, prefix, suffix)) {
      for (const p of paths) {
        const entry = byPath.get(p);
        if (!entry) continue;
        const u = transformManifestUrl(entry.url, ds);
        if (!u) continue;
        try {
          const text = await globalThis.HttpRange.fetchText(u);
          return { text, url: u };
        } catch (e) {
          console.warn(`NEMAR sidecar fetch failed for ${ds}/${p}: ${e.message}`);
          return null;
        }
      }
    }
    return null;
  }

  const api = {
    isNemarDatasetId,
    loadNemarRecording,
    // Module-private helpers exposed for unit testing. Underscore
    // prefix mirrors the bids-recording.js test-seam convention.
    _transformManifestUrl: transformManifestUrl,
    _nemarManifestUrl: nemarManifestUrl,
    _fetchNemarManifest: fetchNemarManifest,
    _fetchManifestSidecar: fetchManifestSidecar,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.BIDSRecordingNemar = api;
})();
