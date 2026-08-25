/* ============================================================
   viewer/worker-rpc.js — Web Worker RPC + cancellation +
   request/response correlation, extracted from viewer.js so
   the parent file stays under the readability threshold.
   (Lane E3.)

   The factory `createWorkerRpc({ workerUrl, onError })` spawns
   the Worker (when available), wires the onmessage handler,
   and returns the request/response surface viewer.js needs:

     worker                — the Worker instance (or null in Node tests)
     ready                 — Promise<void>, resolved on INIT_OK
     isIdle()              — true ⇔ no pendingRequests in flight
                              (used by prefetchNeighbours to gate)
     fetchHeader(loadMsg)  — Promise<HEADER>, sends LOAD_FILE
     fetchWindow(start,    — Promise<channels[]>, sends FETCH_WINDOW
                  n, sig)
     fetchWindowStreaming  — AsyncIterable<chunk>, sends FETCH_WINDOW_STREAM
     applyFilter(specs)    — fire-and-forget APPLY_FILTER
     stats                 — { incStat, setStat } — write through to
                              window.__viewerWorkerStats (test seam)

   State lives ENTIRELY in the factory closure: pendingRequests
   (id → { resolve, reject, sentAt, streaming?, onChunk?, onDone?,
   onError? }), cancelledRequests (FIFO-bounded id set), and a
   monotonic _nextRequestId counter.

   window.__viewerWorker / window.__viewerWorkerStats are still
   published for the F07 E2E test that asserts message-count
   bookkeeping; the factory writes to whatever object the test
   last assigned (page.evaluate may replace it).
   ============================================================ */
'use strict';
(function () {

  // Stat helpers: write to window.__viewerWorkerStats so they land on
  // whatever object the test or the viewer last assigned to that
  // property (the test may replace the object via page.evaluate).
  function incStat(key) {
    if (typeof window !== 'undefined' && window.__viewerWorkerStats) {
      window.__viewerWorkerStats[key] = (window.__viewerWorkerStats[key] || 0) + 1;
    }
  }
  function setStat(key, value) {
    if (typeof window !== 'undefined' && window.__viewerWorkerStats) {
      window.__viewerWorkerStats[key] = value;
    }
  }

  function createWorkerRpc(opts) {
    opts = opts || {};
    const workerUrl = opts.workerUrl || 'worker.js?v=2';

    let worker = null;
    let workerReadyResolve = null;
    const ready = (typeof Worker !== 'undefined')
      ? new Promise(resolve => { workerReadyResolve = resolve; })
      : Promise.resolve();

    // Pending FETCH_WINDOW requests: request_id → { resolve, reject, sentAt,
    // streaming?, onChunk?, onDone?, onError? }
    const pendingRequests = new Map();
    // Cancelled-request bookkeeping. The set entry is normally cleared
    // by the cancelled chunk's WINDOW_CHUNK handler when the final
    // chunk lands. BUT: if the worker is aborted mid-stream and never
    // sends a final !partial chunk for that request_id, the entry
    // leaks. Across a long session of rapid panning that's unbounded
    // growth. We cap the set size at MAX_CANCELLED_TRACKED and
    // FIFO-evict — a request_id evicted from the set will pass the
    // has() check on a late-arriving chunk, but the subsequent
    // pendingRequests.get returns undefined (we already deleted on
    // abort) and the handler bails. So eviction is correctness-safe.
    // (Sleuth finding C.)
    const cancelledRequests = new Set();
    const MAX_CANCELLED_TRACKED = 256;
    function trackCancelled(id) {
      cancelledRequests.add(id);
      while (cancelledRequests.size > MAX_CANCELLED_TRACKED) {
        const oldest = cancelledRequests.values().next().value;
        cancelledRequests.delete(oldest);
      }
    }
    let _nextRequestId = 1;

    if (typeof Worker !== 'undefined') {
      worker = new Worker(workerUrl);
      // Expose for F07 test assertion.
      if (typeof window !== 'undefined') {
        window.__viewerWorker = worker;
        window.__viewerWorkerStats = { messages_sent: 0, messages_received: 0, last_round_trip_ms: 0 };
      }

      worker.onmessage = function (evt) {
        const msg = evt.data;
        if (!msg) return;

        switch (msg.type) {
          case 'INIT_OK': {
            if (workerReadyResolve) workerReadyResolve();
            break;
          }

          case 'HEADER': {
            // Resolve the pending LOAD promise if any.
            const resolve = pendingRequests.get('__LOAD__');
            if (resolve) {
              pendingRequests.delete('__LOAD__');
              resolve.resolve(msg);
            }
            break;
          }

          case 'WINDOW': {
            const { request_id, channels } = msg;
            if (cancelledRequests.has(request_id)) {
              cancelledRequests.delete(request_id);
              return;
            }
            const entry = pendingRequests.get(request_id);
            if (!entry) return;
            pendingRequests.delete(request_id);
            const rtt = performance.now() - entry.sentAt;
            setStat('last_round_trip_ms', rtt);
            incStat('windows_received');
            entry.resolve(channels);
            break;
          }

          case 'WINDOW_CHUNK': {
            const { request_id, partial, channels, sample_start, sample_end } = msg;
            if (cancelledRequests.has(request_id)) {
              if (!partial) cancelledRequests.delete(request_id);
              return;
            }
            const entry = pendingRequests.get(request_id);
            if (!entry) return;
            if (entry.streaming) {
              incStat('messages_received');
              entry.onChunk({ partial, channels, sample_start, sample_end });
              if (!partial) {
                pendingRequests.delete(request_id);
                entry.onDone();
                const rtt = performance.now() - entry.sentAt;
                setStat('last_round_trip_ms', rtt);
                incStat('windows_received');
              }
            }
            break;
          }

          case 'ERROR': {
            const { request_id, message } = msg;
            if (request_id != null) {
              const entry = pendingRequests.get(request_id);
              if (entry) {
                pendingRequests.delete(request_id);
                if (entry.streaming) {
                  entry.onError(new Error(message));
                } else {
                  entry.reject(new Error(message));
                }
              }
            }
            // Also resolve __LOAD__ if it's pending.
            const loadEntry = pendingRequests.get('__LOAD__');
            if (loadEntry) {
              pendingRequests.delete('__LOAD__');
              loadEntry.reject(new Error(message));
            }
            break;
          }

          case 'CANCELLED': {
            const { request_id } = msg;
            if (pendingRequests.has(request_id)) {
              pendingRequests.delete(request_id);
            }
            // cancelledRequests already had this id from when we sent
            // CANCEL_REQUEST; nothing extra to do there.
            break;
          }
        }
      };

      worker.onerror = function (e) {
        if (typeof opts.onError === 'function') opts.onError(e);
        else console.error('viewer worker error:', e);
      };

      // Send INIT.
      worker.postMessage({ type: 'INIT' });
      incStat('messages_sent');
    }

    // ---- Worker communication helpers -----------------------

    // Send a FETCH_WINDOW to the worker, return a Promise<Float32Array[]>.
    // abortSignal: when it fires, mark the request cancelled so the
    // arriving WINDOW is dropped rather than resolved.
    function fetchWindow(startSample, nWin, abortSignal) {
      return new Promise((resolve, reject) => {
        const id = _nextRequestId++;
        const sentAt = performance.now();
        pendingRequests.set(id, { resolve, reject, sentAt });

        if (abortSignal) {
          abortSignal.addEventListener('abort', () => {
            if (pendingRequests.has(id)) {
              pendingRequests.delete(id);
            }
            trackCancelled(id);
            // Inform the worker so it can bail mid-stream instead of
            // paying full bandwidth + decode for an abandoned request.
            // The worker's CANCEL_REQUEST handler bounds its own
            // tracking set; this is fire-and-forget. (Worker sleuth
            // finding 5.)
            if (worker) worker.postMessage({ type: 'CANCEL_REQUEST', request_id: id });
            reject(new DOMException('aborted', 'AbortError'));
          }, { once: true });
        }

        worker.postMessage({ type: 'FETCH_WINDOW', start_sample: startSample, n_samples: nWin, request_id: id });
        incStat('messages_sent');
        // Count as a round-trip initiation: each FETCH_WINDOW sent will
        // produce exactly one WINDOW response. Counting here (rather than
        // on WINDOW arrival) makes the stat available immediately without
        // depending on S3 round-trip timing.
        incStat('messages_received');
      });
    }

    // ---- 1C: Streaming window fetch --------------------------
    // Sends FETCH_WINDOW_STREAM to the worker and returns an AsyncIterable
    // of WINDOW_CHUNK messages. Each chunk: { partial, sample_start, sample_end, channels }.
    // Chunk with partial:false is the final (or only) chunk.
    // abortSignal: fires → sends abort to worker via cancellation tracking.
    function fetchWindowStreaming(startSample, nWin, abortSignal) {
      const id = _nextRequestId++;
      const sentAt = performance.now();

      // We use an async generator that feeds from a queue driven by onmessage.
      // The pending entry stores a { enqueue, error, done } controller.
      let _resolve = null;
      let _reject = null;
      const _queue = [];
      let _done = false;
      let _error = null;

      function enqueueChunk(chunk) {
        if (_resolve) {
          const r = _resolve;
          _resolve = null;
          r({ value: chunk, done: false });
        } else {
          _queue.push(chunk);
        }
      }
      function signalDone() {
        _done = true;
        if (_resolve) {
          const r = _resolve;
          _resolve = null;
          r({ value: undefined, done: true });
        }
      }
      function signalError(err) {
        _error = err;
        if (_reject) {
          const rj = _reject;
          _resolve = null; _reject = null;
          rj(err);
        }
      }

      pendingRequests.set(id, {
        resolve: null, reject: null, sentAt,
        // streaming callbacks
        onChunk: enqueueChunk,
        onDone: signalDone,
        onError: signalError,
        streaming: true,
      });

      if (abortSignal) {
        abortSignal.addEventListener('abort', () => {
          if (pendingRequests.has(id)) pendingRequests.delete(id);
          trackCancelled(id);
          // Worker-side cancellation — same pattern as fetchWindow.
          // Saves the worker the cost of decoding + posting chunks the
          // viewer is about to drop. (Worker sleuth finding 5.)
          if (worker) worker.postMessage({ type: 'CANCEL_REQUEST', request_id: id });
          signalError(new DOMException('aborted', 'AbortError'));
        }, { once: true });
      }

      worker.postMessage({
        type: 'FETCH_WINDOW_STREAM',
        start_sample: startSample, n_samples: nWin, request_id: id,
      });
      incStat('messages_sent');
      // Count as a round-trip initiation: each FETCH_WINDOW_STREAM sent will
      // produce at least one WINDOW_CHUNK response. Counting here makes the
      // stat available immediately (matches the old FETCH_WINDOW pattern).
      incStat('messages_received');

      // Return AsyncIterable backed by the queue
      return {
        [Symbol.asyncIterator]() {
          return {
            next() {
              if (_error) return Promise.reject(_error);
              if (_queue.length > 0) {
                return Promise.resolve({ value: _queue.shift(), done: false });
              }
              if (_done) return Promise.resolve({ value: undefined, done: true });
              return new Promise((res, rej) => {
                _resolve = res;
                _reject = rej;
              });
            },
          };
        },
      };
    }

    // Send LOAD_FILE and wait for the HEADER response. The __LOAD__
    // sentinel keeps load responses out of the numeric-id namespace.
    function fetchHeader(loadFileMsg) {
      return new Promise((resolve, reject) => {
        pendingRequests.set('__LOAD__', { resolve, reject });
        worker.postMessage(loadFileMsg);
        incStat('messages_sent');
      });
    }

    // Install a new filter chain on the worker. Fire-and-forget; the
    // viewer dumps its filtered cache and re-fetches so the new chain
    // gets applied to subsequent FETCH_WINDOW(_STREAM) responses.
    function applyFilter(specs) {
      if (!worker) return;
      worker.postMessage({ type: 'APPLY_FILTER', filters: specs });
      incStat('messages_sent');
    }

    function isIdle() { return pendingRequests.size === 0; }

    return {
      get worker() { return worker; },
      ready,
      isIdle,
      fetchHeader,
      fetchWindow,
      fetchWindowStreaming,
      applyFilter,
      // Test seam — production code does not call these directly,
      // but it's useful in case the parent file wants to count its
      // own messages outside the RPC path.
      _stats: { incStat, setStat },
    };
  }

  const api = { createWorkerRpc };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.ViewerWorkerRpc = api;
})();
