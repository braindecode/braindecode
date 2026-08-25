/* ============================================================
   viewer/render-pipeline.js — render orchestration extracted
   from viewer.js so the parent file stays under the readability
   threshold. (Lane F4.)

   The factory `createRenderPipeline(ctx)` returns a pipeline
   object that owns the render lifecycle:
     - rAF coalescing (`pending`)
     - abort/cancellation (`inFlight`)
     - streaming-vs-non-streaming dispatch
     - generation-guarded cache writeback (`_cacheGen`)

   It does NOT own the underlying mutable state. The caller
   (viewer.js boot()) keeps that state in its own closure and
   exposes it through `ctx` as getter/setter properties so any
   external rebind (e.g. `readerInfo = ...` after load()) is
   visible to this module on the next render tick.

   Required ctx surface (getter + optional setter):
     READ-WRITE state (setter required):
       pending, inFlight,
       lastChannels, lastStartSec, lastWindowSec

     READ-ONLY refs (getter only — caller may rebind externally):
       readerInfo, metaChannels, metaEvents,
       channelLabels, channelBadMask, lastPointerEvent,
       _cacheGen

     READ-ONLY constants/values:
       view, typeColors, readCache, READ_CACHE_MAX,
       worker, tracesCanvas, isEmbedMode, status

     FUNCTIONS (read-only references):
       rpc                          // worker-rpc instance (currently unused
                                    //   from here but kept for parity)
       workerFetchWindow,
       workerFetchWindowStreaming,
       readCachedWindow,
       hasActiveFilters,
       buildDrawOpts,
       prefetchNeighbours,
       refreshCursor,
       updateGainReadout,
       el,                          // utility (el(tag, cls, ...))

     ENVIRONMENT (getters return globals; lets us avoid binding
     globalThis at module-init time which breaks Node tests):
       TraceRenderer,
       requestAnimationFrame
   ============================================================ */
'use strict';
(function () {
  function createRenderPipeline(ctx) {
    function requestRender() {
      if (ctx.pending) return;
      ctx.pending = ctx.requestAnimationFrame(async () => {
        ctx.pending = null;
        if (!ctx.readerInfo) return;
        if (ctx.inFlight) ctx.inFlight.abort();
        ctx.inFlight = new AbortController();
        const ctrl = ctx.inFlight;
        const fs = ctx.readerInfo.sampling_frequency;
        const startSample = Math.max(0,
          Math.min(ctx.readerInfo.n_samples - 1, Math.round(ctx.view.start_sec * fs)));
        const windowSamples = Math.min(
          ctx.readerInfo.n_samples - startSample,
          Math.round(ctx.view.window_sec * fs)
        );

        // Use streaming path for foreground pan on cache miss + no active filters.
        const cacheKey = `${startSample}-${windowSamples}`;
        const cacheHit = ctx.readCache.has(cacheKey);
        const filtersOn = ctx.hasActiveFilters();
        const useStreaming = ctx.worker && !cacheHit && !filtersOn;

        // Fast path: same window + filter state unchanged + lastChannels available.
        // Covers canvas resize (window.resize / ResizeObserver) where the data
        // window and filter state haven't changed. Guards on both filtersOn and
        // lastFiltersOn so enabling OR disabling a filter always goes through the
        // worker path instead of reusing stale (potentially filtered) lastChannels.
        const lastSample = ctx.lastStartSec != null
          ? Math.round(ctx.lastStartSec * fs) : -1;
        const lastWindow = ctx.lastWindowSec != null
          ? Math.round(ctx.lastWindowSec * fs) : -1;
        if (!filtersOn && !ctx.lastFiltersOn && ctx.lastChannels &&
            startSample === lastSample && windowSamples === lastWindow) {
          const drawOpts = ctx.buildDrawOpts(ctx.lastChannels, startSample, fs);
          ctx.TraceRenderer.draw(ctx.tracesCanvas, drawOpts);
          ctx.updateGainReadout();
          if (ctx.lastPointerEvent) ctx.refreshCursor();
          ctx.prefetchNeighbours();
          return;
        }
        // Snapshot the cache generation at render start. The streaming
        // writeback at the end of the for-await loop only commits to
        // readCache if this still matches _cacheGen — otherwise a
        // filter toggle that ran during the render would be silently
        // undone. (Sleuth finding B.)
        const startCacheGen = ctx._cacheGen;

        if (useStreaming) {
          // Streaming render: first chunk clears canvas and paints, subsequent chunks
          // do partial updates. This gives time-to-first-pixel before full window arrives.
          let firstChunk = true;
          let assembledChannels = null;
          let totalSamples = 0;

          try {
            for await (const chunk of ctx.workerFetchWindowStreaming(startSample, windowSamples, ctrl.signal)) {
              if (ctrl.signal.aborted) break;
              const { partial, channels: chunkChannels, sample_start, sample_end } = chunk;

              if (!assembledChannels) {
                assembledChannels = chunkChannels.map(() => new Float32Array(windowSamples));
                totalSamples = 0;
              }
              const chunkLen = chunkChannels[0].length;
              // Guard against buffer overflow (can happen on rapid abort+restart)
              if (totalSamples + chunkLen > windowSamples) break;
              for (let c = 0; c < assembledChannels.length; c++) {
                assembledChannels[c].set(chunkChannels[c], totalSamples);
              }
              totalSamples += chunkLen;

              // Determine which samples to paint in this partial step
              const visibleChannels = assembledChannels.map(ch => ch.subarray(0, totalSamples));
              const drawOpts = ctx.buildDrawOpts(visibleChannels, startSample, fs);
              // Always pass partial_fill so the renderer knows the FULL
              // window's sample count (needed to map the polyline to its
              // real x-band instead of stretching across plotW). The first
              // chunk additionally requests a full clear to wipe stale
              // pixels from any previously-painted (now superseded) window.
              drawOpts.partial_fill = {
                sample_start,
                sample_end,
                total_samples: windowSamples,
                full_clear: firstChunk,
              };
              // Defensive abort recheck immediately before paint. The top-of-
              // iteration check at the for-await guards against abort BETWEEN
              // chunks, but a chunk delivered via the iterator's `_queue` path
              // (rapid back-to-back enqueues) can be observed AFTER the
              // controller has already been replaced by a newer render. Without
              // this check the old stream paints one final stale frame on top
              // of the new stream's first chunk → brief one-frame ghost flash.
              if (ctrl.signal.aborted) break;
              ctx.TraceRenderer.draw(ctx.tracesCanvas, drawOpts);
              if (firstChunk) firstChunk = false;
              ctx.updateGainReadout();

              // If final chunk: update cursor cache and fire prefetch
              if (!partial) {
                ctx.lastChannels = visibleChannels;
                ctx.lastStartSec = startSample / fs;
                ctx.lastWindowSec = ctx.view.window_sec;
                ctx.lastFiltersOn = filtersOn;
                if (ctx.lastPointerEvent) ctx.refreshCursor();
                // Populate the main-thread read cache promise so future
                // FETCH_WINDOW requests hit immediately — BUT only if
                // no clearReadCache() ran since this render started. A
                // filter toggle, window-sec change, or drop-load fires
                // clearReadCache(); writing stale (pre-filter) data into
                // the freshly-cleared cache would poison every
                // subsequent foreground read for that window.
                // (Sleuth finding B.)
                if (ctx._cacheGen === startCacheGen) {
                  const channelsCopy = assembledChannels.map(ch => {
                    const a = new Float32Array(ch.length);
                    a.set(ch);
                    return a;
                  });
                  ctx.readCache.set(cacheKey, Promise.resolve(channelsCopy));
                  while (ctx.readCache.size > ctx.READ_CACHE_MAX) {
                    ctx.readCache.delete(ctx.readCache.keys().next().value);
                  }
                }
              }
            }
          } catch (err) {
            if (err.name === 'AbortError') {
              ctx.prefetchNeighbours();
              return;
            }
            ctx.status.replaceChildren(ctx.el('span', 'err', `read window failed: ${err.message}`));
            console.error(err);
            return;
          }
          ctx.prefetchNeighbours();
          return;
        }

        // Non-streaming path (cache hit, or filters active, or no worker)
        let channels;
        try {
          channels = await ctx.readCachedWindow(startSample, windowSamples, ctrl.signal);
        } catch (err) {
          if (err.name === 'AbortError') {
            // Render was superseded by a newer ArrowRight/pan, but still
            // fire prefetch so the next render (or the one that replaced
            // this one) benefits from a warm cache. Each abort = one
            // prefetch = one FETCH_WINDOW → keeps messages_sent climbing
            // reliably under rapid panning (needed for F07 stats test).
            ctx.prefetchNeighbours();
            return;
          }
          ctx.status.replaceChildren(ctx.el('span', 'err', `read window failed: ${err.message}`));
          console.error(err);
          return;
        }
        // Fire prefetch here, before the aborted-render check, so that
        // even cache-hit renders that are superseded (ctrl.signal.aborted)
        // still warm the cache for the user's continued panning.
        ctx.prefetchNeighbours();
        if (!channels || ctrl.signal.aborted) return;
        const drawStartSec = startSample / fs;
        // Build per-channel colour array from current type→colour mapping.
        const channelColors = ctx.metaChannels && ctx.metaChannels.length
          ? ctx.metaChannels.map(ch => ctx.typeColors[(ch.type || 'MISC').toUpperCase()] || null)
          : null;
        ctx.TraceRenderer.draw(ctx.tracesCanvas, {
          channels,
          n_samples_visible: channels[0]?.length || 0,
          channel_labels: ctx.channelLabels,
          channel_types: ctx.metaChannels ? ctx.metaChannels.map(ch => (ch.type || '').toUpperCase()) : null,
          bad_mask: ctx.channelBadMask,
          channel_colors: channelColors,
          channel_offset: ctx.view.channel_offset,
          events: ctx.metaEvents,
          fs,
          start_sec: drawStartSec,
          gain: ctx.view.gain,
          time_mode: ctx.view.time_mode,
          recording_start_iso: ctx.readerInfo ? (ctx.readerInfo.recording_start_iso ?? null) : null,
          transparent: ctx.isEmbedMode,
        });
        ctx.updateGainReadout();
        // Cache for cursor readout.
        ctx.lastChannels = channels;
        ctx.lastStartSec = drawStartSec;
        ctx.lastWindowSec = ctx.view.window_sec;
        ctx.lastFiltersOn = filtersOn;
        // If the pointer is already over the canvas, refresh the readout
        // now that we have data (the mouse may have moved before the first
        // render completed).
        if (ctx.lastPointerEvent) ctx.refreshCursor();
      });
    }

    return { requestRender };
  }

  const api = { createRenderPipeline };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.ViewerRenderPipeline = api;
})();
