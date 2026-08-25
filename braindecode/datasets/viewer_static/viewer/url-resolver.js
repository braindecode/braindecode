/* ============================================================
   viewer/url-resolver.js — URL/BIDS target resolution ladder
   extracted from viewer.js so the parent file stays under the
   readability threshold. (Lane F1.)

   The factory `resolveAndLoad(params, deps)` takes already-parsed
   URLSearchParams (after applyEmbedMode) and dispatches to the
   appropriate loader based on the target kind that
   BIDSRecording.resolveTargets produces:

     - 'url' / 'bids-path'              -> deps.load(target.eeg_url)
     - 'bids-path-discover-sub'         -> probe subject + suffix,
                                            then deps.load(builtUrl)
     - 'bids-path-auto'                 -> probe suffix only,
                                            then deps.load(builtUrl)
     - 'nemar'                          -> deps.loadNemar(params)
     - 'demo'                           -> setStatus stub

   Caller supplies deps:
     - load(eegUrl)        — internal closure in viewer.js
     - loadNemar(params)   — internal closure in viewer.js
     - setStatus(text)     — UI hook, wraps status.textContent = ...

   The function returns a Promise that resolves after the dispatch
   decision is made (NOT after the dataset finishes loading — that
   is load()'s job). Async probes run inside try/catch so a network
   failure surfaces via setStatus rather than an unhandled rejection.
   ============================================================ */
'use strict';
(function () {
  async function resolveAndLoad(params, deps) {
    const { load, loadNemar, setStatus } = deps;
    const BIDS = globalThis.BIDSRecording;
    if (!BIDS) throw new Error('viewer/url-resolver: globalThis.BIDSRecording not loaded');

    const target = BIDS.resolveTargets(params);

    if (target?.kind === 'url') {
      load(target.eeg_url);
      return;
    }

    if (target?.kind === 'bids-path') {
      // Explicit ?sub= + ?suffix= form. If the built URL 404s we try
      // discoverSubject + rebuild to absorb caller sub-ID mismatches
      // (e.g. eegdash.org templates generating `sub-001` when the
      // dataset actually uses `sub-1`). Probe is a single 1-byte Range
      // GET — fast and cheap regardless of file size.
      const HttpRange = globalThis.HttpRange;
      let ok = false;
      if (HttpRange && HttpRange.probeLength) {
        try {
          await HttpRange.probeLength(target.eeg_url);
          ok = true;
        } catch (e) {
          ok = false;
        }
      }
      if (ok) {
        load(target.eeg_url);
        return;
      }
      // Fallback: re-resolve with the dataset's actual first subject.
      const realSub = await BIDS.discoverSubject(target.params).catch(() => null);
      if (realSub && realSub !== target.params.sub) {
        const retryUrl = BIDS.buildOpenNeuroEegUrl({ ...target.params, sub: realSub });
        setStatus(
          `Subject ${target.params.sub} not found in ${target.params.dataset}; ` +
          `using ${realSub} (the dataset's first subject) instead.`,
        );
        load(retryUrl);
        return;
      }
      // No alternative — surface the original URL anyway; the reader
      // will produce the right 404 error message.
      load(target.eeg_url);
      return;
    }

    if (target?.kind === 'bids-path-discover-sub') {
      // Subject not specified in URL — probe participants.tsv then
      // S3-list to find the first real subject ID. After the sub is
      // discovered, fall through to modality discovery (if ?suffix=
      // also omitted) or direct URL build. Mirrors bids-path-auto.
      setStatus('Detecting subject...');
      try {
        const sub = await BIDS.discoverSubject(target.params);
        if (!sub) {
          setStatus(
            `No subjects found at ${target.params.dataset}/ ` +
            `(participants.tsv missing and no sub-* directories listed). ` +
            `Try adding &sub=<id> to the URL.`,
          );
          return;
        }
        // Re-enter resolution with the discovered sub baked in.
        const resolvedParams = { ...target.params, sub };
        if (resolvedParams.suffix) {
          // Both sub (discovered) and suffix (explicit) — direct build.
          load(BIDS.buildOpenNeuroEegUrl(resolvedParams));
          return;
        }
        // sub discovered, suffix unknown — run modality probe.
        setStatus('Detecting modality...');
        const suf = await BIDS.discoverSuffix(resolvedParams);
        if (!suf) {
          setStatus(
            `No EEG/iEEG/MEG/EMG/NIRS recording found at ` +
            `${resolvedParams.dataset}/sub-${sub}/...`,
          );
          return;
        }
        load(BIDS.buildOpenNeuroEegUrl({ ...resolvedParams, suffix: suf }));
      } catch (err) {
        setStatus(`Subject probe failed: ${err.message || err}`);
      }
      return;
    }

    if (target?.kind === 'bids-path-auto') {
      // Modality not specified in URL — probe eeg/ieeg/meg/emg/nirs in
      // parallel + pick the first that exists. ~50-200 ms wall on
      // cdn.eegdash.org because all 5 probes race; bandwidth cost is
      // 5 × 1-byte range requests.
      setStatus('Detecting modality...');
      try {
        let suf = await BIDS.discoverSuffix(target.params);
        let resolvedSub = target.params.sub;
        if (!suf) {
          // Subject naming mismatch fallback. The caller's ?sub= might
          // not match the dataset's actual subject IDs (common: passing
          // `001` when the dataset uses `1`, or `01`; eegdash.org
          // documentation pages sometimes generate sub IDs from generic
          // templates that don't match the publisher's naming). Discover
          // the real first subject and retry the modality probe before
          // giving up. Mirrors the bids-path-discover-sub branch above.
          const realSub = await BIDS.discoverSubject(target.params).catch(() => null);
          if (realSub && realSub !== target.params.sub) {
            const retryParams = { ...target.params, sub: realSub };
            const retrySuf = await BIDS.discoverSuffix(retryParams);
            if (retrySuf) {
              setStatus(
                `Subject ${target.params.sub} not found in ${target.params.dataset}; ` +
                `using ${realSub} (the dataset's first subject) instead.`,
              );
              suf = retrySuf;
              resolvedSub = realSub;
            }
          }
        }
        if (!suf) {
          setStatus(
            `No EEG/iEEG/MEG/EMG/NIRS recording found at ` +
            `${target.params.dataset}/sub-${target.params.sub}/...`,
          );
          return;
        }
        const resolved = { ...target.params, sub: resolvedSub, suffix: suf };
        load(BIDS.buildOpenNeuroEegUrl(resolved));
      } catch (err) {
        setStatus(`Modality probe failed: ${err.message || err}`);
      }
      return;
    }

    if (target?.kind === 'nemar') {
      // NEMAR (nm/on/xx prefixes): meta is built from the per-version
      // data.nemar.org manifest.json fetched through the cdn-worker
      // CORS proxy. Same render path as load(), pre-resolved bundle.
      loadNemar(target.nemar_params);
      return;
    }

    if (target?.kind === 'demo') {
      setStatus(`demo loader (${target.demo_id}) not wired yet`);
      return;
    }
  }

  const api = { resolveAndLoad };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.ViewerUrlResolver = api;
})();
