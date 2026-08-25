/* ============================================================
   viewer/render-helpers.js — pure helpers that shape the data
   the TraceRenderer expects, extracted from viewer.js so the
   render-pipeline body stays under the readability threshold.
   (Lane F2.)

   Every helper here is pure: it reads from a deps object the
   caller constructs and returns a fresh value. No closures over
   viewer.js state. Tests can call these directly with a
   synthetic deps payload.
   ============================================================ */
'use strict';
(function () {
  // Build the drawOpts payload for TraceRenderer.draw().
  //
  // Inputs:
  //   channels      — Float32Array[] (already trimmed to visible window)
  //   startSample   — sample offset (start of the visible window)
  //   fs            — sampling frequency in Hz
  //   deps          — read-only references the caller closes over:
  //     metaChannels, typeColors, channelLabels, channelBadMask,
  //     metaEvents, view (start_sec/window_sec/gain/time_mode/
  //     channel_offset), readerInfo, isEmbedMode
  //
  // Output: plain object ready to pass into TraceRenderer.draw().
  function buildDrawOpts(channels, startSample, fs, deps) {
    const {
      metaChannels, typeColors, channelLabels, channelBadMask, metaEvents,
      view, readerInfo, isEmbedMode,
    } = deps;
    const channelColors = metaChannels && metaChannels.length
      ? metaChannels.map(ch => typeColors[(ch.type || 'MISC').toUpperCase()] || null)
      : null;
    return {
      channels,
      n_samples_visible: channels[0]?.length || 0,
      channel_labels: channelLabels,
      channel_types: metaChannels ? metaChannels.map(ch => (ch.type || '').toUpperCase()) : null,
      bad_mask: channelBadMask,
      channel_colors: channelColors,
      channel_offset: view.channel_offset,
      events: metaEvents,
      fs,
      start_sec: startSample / fs,
      gain: view.gain,
      time_mode: view.time_mode,
      recording_start_iso: readerInfo ? (readerInfo.recording_start_iso ?? null) : null,
      transparent: isEmbedMode,
    };
  }

  // Issue a single best-effort prefetch in the user's last pan direction.
  //
  // Inputs (deps object, read-only refs the caller closes over and
  // passes fresh on every call):
  //   readerInfo, view, lastPanDir,
  //   rpcIsIdle()       — gate so we don't queue behind foreground
  //   worker            — Worker instance (null in Node tests)
  //   workerFetchWindow — (s, n, signal) => Promise<channels>
  //   fallbackReader    — non-worker readWindow path
  //   readCache         — "start-n" → Promise<channels> Map (mutated)
  //   READ_CACHE_MAX    — LRU bound for readCache
  //   clampStartSamples — coerce a start-time seconds to an in-range sample idx
  //
  // Returns: void. The prefetch promise lives in readCache and is
  // discarded if it rejects (caller doesn't await — this is fire-and-forget).
  function prefetchNeighbours(deps) {
    const {
      readerInfo, view, lastPanDir,
      rpcIsIdle, worker, workerFetchWindow, fallbackReader,
      readCache, READ_CACHE_MAX, clampStartSamples,
    } = deps;
    if (!readerInfo) return;
    // Gate: if the worker has any in-flight FETCH_WINDOW, do NOT
    // queue prefetches behind it. The worker processes requests
    // serially; queuing 4 prefetches behind a foreground fetch
    // pushes the user's NEXT pan ~5 round-trips deep and produces
    // the multi-second lag the perf benchmark exposed.
    if (!rpcIsIdle()) return;
    const fs = readerInfo.sampling_frequency;
    const n = Math.round(view.window_sec * fs);
    // Single-target prefetch: the most likely next window in the
    // user's pan direction. Fan-out higher than 1 amplifies queue
    // contention without speeding up perception.
    const half = view.window_sec / 2;
    const targets = lastPanDir !== 0
      ? [clampStartSamples(view.start_sec + lastPanDir * half)]
      : [clampStartSamples(view.start_sec + half)];
    for (const s of targets) {
      const key = `${s}-${n}`;
      if (readCache.has(key)) continue;
      // No abort signal: prefetch is best-effort.
      let p;
      if (worker) {
        p = workerFetchWindow(s, n, null).catch(() => null);
      } else {
        p = fallbackReader.readWindow(s, n).catch(() => null);
      }
      readCache.set(key, p);
      while (readCache.size > READ_CACHE_MAX) {
        readCache.delete(readCache.keys().next().value);
      }
    }
  }

  const api = { buildDrawOpts, prefetchNeighbours };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.ViewerRenderHelpers = api;
})();
