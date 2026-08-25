/* ============================================================
   formats/_http_range.js — HTTP Range helpers + local-Blob
   registry shared by every format reader.

   Drag-dropped files register against synthetic
   `https://localdrop.invalid/<filename>` URLs (.invalid is
   reserved by RFC 2606) so the existing URL math — parseEegUrl,
   `new URL(rel, base)` for sibling derivation, the inheritance
   walk — works unchanged for both real OpenNeuro URLs and
   in-memory drops.
   ============================================================ */
(function () {
  'use strict';

  const LOCAL_PREFIX = 'https://localdrop.invalid/';
  const _localBlobs = new Map();

  function isLocal(url) { return typeof url === 'string' && url.startsWith(LOCAL_PREFIX); }

  function registerLocal(filename, blob) {
    const url = `${LOCAL_PREFIX}${encodeURIComponent(filename)}`;
    _localBlobs.set(url, blob);
    return url;
  }

  function clearLocal() { _localBlobs.clear(); }

  // Transient 5xx retry wrapper. CDN-fronted buckets (Cloudflare in
  // front of S3, what cdn.eegdash.org runs) sometimes return 502/503/504
  // for valid URLs due to upstream pool flake — usually resolves on the
  // next request. The 648-dataset OpenNeuro audit hit 103 of these per
  // run. A small retry+backoff turns most of them into passes without
  // hurting the happy path (single fetch on 200/206).
  //
  // Retries on transient server-side conditions:
  //   - 502/503/504 — upstream pool flake (Cloudflare in front of S3)
  //   - 429 — rate limiting; Cloudflare returns this when the client
  //     exceeds the per-IP burst quota. Common during audit runs that
  //     touch 600+ URLs quickly. Honors Retry-After header when set.
  // NOT 4xx other than 429 (real "missing file" errors propagate).
  // NOT network errors (those usually mean the caller aborted).
  const RETRY_STATUSES = new Set([429, 502, 503, 504]);
  // Longer tail for 429 — rate-limit windows on Cloudflare are usually
  // 10 s; the schedule lets the last retry happen well after a typical
  // window expiry. 200/600/1500/4000 ms = up to 6 s of accumulated wait
  // before the 4th retry, which is enough for most edge-cache cooldowns.
  const RETRY_DELAYS_MS = [200, 600, 1500, 4000];

  async function fetchWithRetry(url, init) {
    let lastResponse = null;
    for (let attempt = 0; attempt <= RETRY_DELAYS_MS.length; attempt++) {
      const r = await fetch(url, init);
      if (!RETRY_STATUSES.has(r.status)) return r;
      // Honor Retry-After when the server tells us when to come back.
      // Cloudflare 429 responses sometimes include a delta-seconds value;
      // if it's set and reasonable, prefer it over our backoff schedule.
      const retryAfter = r.headers.get('retry-after');
      // Drain the body so the connection is freed before we retry.
      try { await r.body?.cancel(); } catch { /* ignore */ }
      lastResponse = r;
      if (attempt === RETRY_DELAYS_MS.length) break;
      let delay = RETRY_DELAYS_MS[attempt];
      if (retryAfter) {
        const parsed = Number(retryAfter);
        // Cap at 10 s — anything longer means the server is broken,
        // not just rate-limited, and we shouldn't block that long.
        if (Number.isFinite(parsed) && parsed > 0 && parsed <= 10) {
          delay = Math.max(delay, parsed * 1000);
        }
      }
      // Respect the original signal — if the caller aborts mid-retry,
      // surface the abort instead of finishing the backoff.
      const signal = init && init.signal;
      if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError');
      await new Promise((res, rej) => {
        const t = setTimeout(res, delay);
        if (signal) signal.addEventListener('abort', () => { clearTimeout(t); rej(new DOMException('aborted', 'AbortError')); }, { once: true });
      });
    }
    return lastResponse;
  }

  // HEAD first; some S3 access policies block HEAD outright, so
  // fall back to a zero-byte GET with Range and parse the
  // Content-Range total. Local blobs short-circuit to `.size`.
  async function probeLength(url) {
    if (isLocal(url)) {
      const blob = _localBlobs.get(url);
      if (!blob) throw new Error(`Local drop missing: ${url}`);
      return blob.size;
    }
    let r = await fetchWithRetry(url, { method: 'HEAD' });
    if (r.ok && r.headers.get('content-length')) {
      return Number(r.headers.get('content-length'));
    }
    r = await fetchWithRetry(url, { headers: { Range: 'bytes=0-0' } });
    if (r.status !== 206) {
      throw new Error(`Cannot determine length: HTTP ${r.status} for ${url}`);
    }
    const cr = r.headers.get('content-range');
    const m = cr && /\/(\d+)$/.exec(cr);
    if (!m) throw new Error(`Server returned no Content-Range total for ${url}`);
    return Number(m[1]);
  }

  // HEAD-avoidant variant: some CDNs (verified against cdn.eegdash.org
  // and OpenNeuro S3 fronted by Cloudflare; ds003694 2 GB FIFF, see
  // tests/evidence/streaming-large/cdn-head-poisons-range.md) cache the
  // HEAD response with the same cache key as a subsequent GET-with-Range
  // — the GET then gets served the cached full body (200 + entire file)
  // instead of a 206 + the requested range. Bypass HEAD by issuing a
  // 1-byte Range GET; the Content-Range total component (after `/`)
  // gives the same information without poisoning the cache. Falls back
  // to probeLength on any error / missing Content-Range (covers local
  // blobs, file://, and servers that don't honour Range).
  //
  // Promoted from formats/fiff.js + formats/eeglab.js (B4) where both
  // shipped character-identical local copies.
  async function probeLengthNoHead(url) {
    try {
      const res = await fetchWithRetry(url, { headers: { Range: 'bytes=0-0' } });
      if (res.status === 206) {
        const cr = res.headers.get('content-range');
        const m = cr && /\/(\d+)$/.exec(cr);
        if (m) {
          // Drain the 1-byte body so the connection slot is freed.
          await res.arrayBuffer();
          return Number(m[1]);
        }
      }
      // Drain any body anyway so the socket doesn't leak.
      await res.arrayBuffer().catch(() => null);
    } catch {
      // Fall through to legacy probeLength on any error.
    }
    return probeLength(url);
  }

  // Inclusive range. `expectedBytes`, when given, also throws if the
  // body length differs (typical sign of a CDN that ignored Range and
  // served the full file). `signal` cancels in-flight HTTP fetches —
  // when the user pans rapidly, the page aborts the prior request so
  // we don't waste bandwidth on results that will be discarded. The
  // local-blob path checks `signal.aborted` once before slicing so a
  // late synchronous read still bails after the URL was unregistered.
  // OpenNeuro S3 is per-connection bandwidth-throttled (~0.7 MB/s on a
  // single fetch), but HTTP/2 multiplexes parallel requests on the
  // same TCP connection so concurrent range fetches reach ~3-4 MB/s
  // total. Tiling above the threshold therefore gives a 4-5× wall-time
  // speedup on the same total bytes — see scripts/bench-fetch.mjs.
  //
  // 256 KiB threshold + 256 KiB tile size + 6 parallel fetches.
  // Empirical: live-probing OpenNeuro S3 (us-east-1, raw bucket,
  // HTTP/1.1 only — no HTTP/2 multiplexing) showed 8 × 128 KB
  // parallel range fetches complete in 770 ms total vs 2557 ms for
  // a single 1 MB fetch — 3.3× faster. The original 4 MiB threshold
  // was right for the 38 MB benchmark it was tuned against (where
  // tiling won 4-5×) but wrong for the typical sub-MB pan, where
  // the per-TCP bandwidth cap (~0.4 MB/s) makes a single fetch the
  // slow path even though the bytes are few.
  //
  // 6 parallel matches the browser's HTTP/1.1 connection-per-host
  // cap (Chrome/Firefox default). Going higher is wasted: the 7th+
  // fetch queues until a slot opens. See docs/streaming-and-cdn-study.md.
  const TILE_THRESHOLD_BYTES = 256 * 1024;
  const TILE_TARGET_BYTES    = 256 * 1024;
  const TILE_MAX_PARALLEL    = 6;

  async function rangeFetch(url, byteStart, byteEndInclusive, expectedBytes, opts) {
    // Security A4: reject pathological inputs at the perimeter.
    // - Negative byteStart would become `Range: bytes=-N-…`, which S3
    //   interprets as a SUFFIX RANGE (fetch the trailing N bytes of the
    //   object) — a reader handing us `start = sampleIdx * frameSize`
    //   with an underflowed sampleIdx would silently get end-of-file
    //   data instead of an error.
    // - Non-integer byteStart is also rejected so callers can't sneak
    //   in NaN / -0 / 1e30 / "0" / etc.
    if (!Number.isInteger(byteStart) || byteStart < 0) {
      throw new Error(`HttpRange: bad byteStart ${byteStart}`);
    }
    // end < start collapses to zero-length without hitting the wire.
    // Non-integer end is treated the same way (still zero-length).
    if (!Number.isInteger(byteEndInclusive) || byteEndInclusive < byteStart) {
      return new ArrayBuffer(0);
    }
    // Zero-length / inverted ranges short-circuit without hitting the
    // network (or the local registry) — every reader has a code path
    // where nSamplesWindow=0 makes the math collapse, and we'd rather
    // return an empty ArrayBuffer than send `Range: bytes=0--1`.
    if (expectedBytes === 0) {
      return new ArrayBuffer(0);
    }
    if (isLocal(url)) {
      return rangeFetchLocal(url, byteStart, byteEndInclusive, expectedBytes, opts);
    }
    const total = byteEndInclusive - byteStart + 1;
    if (total >= TILE_THRESHOLD_BYTES) {
      return rangeFetchTiled(url, byteStart, byteEndInclusive, total, opts);
    }
    return rangeFetchSingle(url, byteStart, byteEndInclusive, expectedBytes, opts);
  }

  async function rangeFetchLocal(url, byteStart, byteEndInclusive, expectedBytes, opts) {
    const signal = opts && opts.signal;
    if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError');
    const blob = _localBlobs.get(url);
    if (!blob) throw new Error(`Local drop missing: ${url}`);
    const buf = await blob.slice(byteStart, byteEndInclusive + 1).arrayBuffer();
    if (expectedBytes != null && buf.byteLength !== expectedBytes) {
      throw new Error(`Local slice returned ${buf.byteLength}B, expected ${expectedBytes}B.`);
    }
    return buf;
  }

  // Largest stream-and-slice we'll do when the CDN ignores Range. Above
  // this, we throw rather than burn bandwidth — caller can retry or
  // fall back. Raised 2026-05-22 from 200 MB → 1 GB symmetric with the
  // inline-data fallback cap (eeglab.js). The cost is bytes downloaded
  // to reach the offset; modern browsers handle 1 GB streams fine and
  // huge files where this cap fires are typically multi-GB MEG bundles
  // that the user explicitly opened (not a default landing recording).
  const RANGE_IGNORED_STREAM_CAP_BYTES = 1024 * 1024 * 1024;

  async function rangeFetchSingle(url, byteStart, byteEndInclusive, expectedBytes, opts) {
    const signal = opts && opts.signal;
    const r = await fetchWithRetry(url, {
      headers: { Range: `bytes=${byteStart}-${byteEndInclusive}` },
      signal,
    });
    if (r.status === 206) {
      // Honored — normal path.
      const buf = await r.arrayBuffer();
      if (expectedBytes != null && buf.byteLength !== expectedBytes) {
        throw new Error(
          `Range fetch returned ${buf.byteLength}B, expected ${expectedBytes}B ` +
          `(server may have ignored Range header).`
        );
      }
      return buf;
    }
    if (r.status === 200) {
      // Server ignored Range and is returning the full body. Observed
      // intermittently on cdn.eegdash.org for some .meg4 files (CTF MEG
      // bundles) — the same URL serves 206 most of the time but 200
      // some of the time, presumably depending on edge-cache state.
      //
      // Strategy: stream-read enough bytes to cover the requested range,
      // then cancel the rest. This wastes the prefix bytes [0, byteStart)
      // but avoids downloading the full file (which on a 1.1 GB .meg4
      // takes 100+ s wall and burns 1 GB egress). Gives correctness
      // even when the CDN cache misbehaves.
      const needed = byteEndInclusive + 1;
      if (needed > RANGE_IGNORED_STREAM_CAP_BYTES) {
        // Refuse to stream-and-slice for huge offsets — caller can
        // retry, fall back to a different URL, or surface the error.
        // Drain + close so the connection is released cleanly.
        try { await r.body.cancel(); } catch { /* ignore */ }
        throw new Error(
          `Range fetch ignored by server (HTTP 200 for Range request); ` +
          `would need to stream ${needed}B (cap ${RANGE_IGNORED_STREAM_CAP_BYTES}B) to slice ` +
          `requested range ${byteStart}-${byteEndInclusive}.`
        );
      }
      const reader = r.body.getReader();
      const chunks = [];
      let received = 0;
      try {
        while (received < needed) {
          if (signal && signal.aborted) {
            try { await reader.cancel(); } catch { /* ignore */ }
            throw new DOMException('aborted', 'AbortError');
          }
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          received += value.byteLength;
        }
      } finally {
        // Cancel the rest of the response so we don't waste bandwidth.
        try { await reader.cancel(); } catch { /* ignore */ }
      }
      if (received < byteStart) {
        throw new Error(
          `Range-ignored fallback: response body had only ${received}B, ` +
          `requested range starts at byteStart=${byteStart}.`
        );
      }
      // Concatenate the chunks and slice the requested window.
      const all = new Uint8Array(received);
      let off = 0;
      for (const c of chunks) { all.set(c, off); off += c.byteLength; }
      const sliceEnd = Math.min(received, byteEndInclusive + 1);
      const out = all.buffer.slice(byteStart, sliceEnd);
      if (expectedBytes != null && out.byteLength !== expectedBytes) {
        // The CDN's full-body response was shorter than expected —
        // surface this clearly rather than silently truncate.
        throw new Error(
          `Range-ignored fallback: sliced ${out.byteLength}B from ${received}B body, ` +
          `expected ${expectedBytes}B.`
        );
      }
      return out;
    }
    throw new Error(`Range fetch failed: HTTP ${r.status} for ${url}`);
  }

  // Split a big range into ≤ TILE_MAX_PARALLEL pieces and pull them all
  // at once. Same `signal` propagates so a pan-mid-flight aborts every
  // tile in flight; one rejection rejects the whole `Promise.all`.
  async function rangeFetchTiled(url, byteStart, byteEndInclusive, total, opts) {
    const nTiles = Math.min(TILE_MAX_PARALLEL, Math.ceil(total / TILE_TARGET_BYTES));
    const tileSize = Math.ceil(total / nTiles);
    const ranges = [];
    for (let i = 0; i < nTiles; i++) {
      const a = byteStart + i * tileSize;
      const b = Math.min(a + tileSize - 1, byteEndInclusive);
      ranges.push([a, b]);
    }
    const buffers = await Promise.all(
      ranges.map(([a, b]) => rangeFetchSingle(url, a, b, b - a + 1, opts))
    );
    const out = new Uint8Array(total);
    let off = 0;
    for (const buf of buffers) {
      out.set(new Uint8Array(buf), off);
      off += buf.byteLength;
    }
    return out.buffer;
  }

  // Single text-fetch entry point. With `allowMissing` the helper is
  // 404-tolerant — sidecars are often optional and we want a `null`
  // back so the inheritance walk can fall through, not an exception.
  // Without it, anything other than 2xx throws (used for the .vhdr,
  // which is required by definition).
  async function fetchText(url, { allowMissing = false } = {}) {
    if (isLocal(url)) {
      const blob = _localBlobs.get(url);
      if (blob) return blob.text();
      if (allowMissing) return null;
      throw new Error(`Local drop missing: ${url}`);
    }
    // force-cache: OpenNeuro / static BIDS buckets serve immutable
    // content, so the browser cache is a free win across pans.
    const r = await fetchWithRetry(url, { cache: 'force-cache' });
    // Optional-sidecar mode: treat ANY non-2xx as "missing" rather than
    // only 404. Previously a CDN 502 on a coordsystem.json or events.tsv
    // would kill the whole load even though those files aren't required
    // to render a recording. The 648-dataset audit had ~20 cases where
    // a transient sidecar 502 doomed an otherwise-perfectly-loadable
    // recording (counted as reader-rejected). Surface a warn so the
    // developer can investigate, but don't propagate the error.
    if (allowMissing && !r.ok) {
      if (r.status !== 404) {
        // 404 is the "BIDS inheritance walk hit a dead end" expected
        // case — silent. Anything else is noteworthy.
        console.warn(
          `Optional sidecar fetch returned HTTP ${r.status} for ${url}; ` +
          `treating as missing.`,
        );
      }
      return null;
    }
    if (!r.ok) throw new Error(`${r.status} ${r.statusText} fetching ${url}`);
    return r.text();
  }
  const fetchTextOrNull = (url) => fetchText(url, { allowMissing: true });

  // ---- Streaming fetch ----------------------------------------
  // Returns an AsyncIterable<{ offset, bytes: Uint8Array }> where
  // `offset` is 0-based within the requested range [byteStart, byteEnd].
  // Falls back to a single rangeFetch (yielding one chunk) for:
  //   - local blobs (synchronous slice, no streaming gain)
  //   - tiny ranges below STREAM_THRESHOLD (chunking overhead not worth it)
  // opts.signal aborts mid-stream cleanly.
  // Throws if total received bytes != requested length.
  const STREAM_THRESHOLD = 64 * 1024;  // 64 KiB — chunking tiny ranges is wasteful

  async function* rangeFetchStreaming(url, byteStart, byteEndInclusive, opts) {
    const total = byteEndInclusive - byteStart + 1;
    if (total <= 0) return;

    // For local blobs or small ranges, fall back to a single arraybuffer
    // chunk — no streaming benefit for tiny or synchronous sources.
    if (isLocal(url) || total < STREAM_THRESHOLD) {
      const buf = await rangeFetch(url, byteStart, byteEndInclusive, total, opts);
      yield { offset: 0, bytes: new Uint8Array(buf) };
      return;
    }

    const signal = opts && opts.signal;
    const r = await fetchWithRetry(url, {
      headers: { Range: `bytes=${byteStart}-${byteEndInclusive}` },
      signal,
    });
    if (r.status !== 206 && r.status !== 200) {
      throw new Error(`Range fetch (streaming) failed: HTTP ${r.status} for ${url}`);
    }

    const reader = r.body.getReader();
    let offset = 0;
    try {
      while (true) {
        // Check abort signal before each read
        if (signal && signal.aborted) {
          reader.cancel();
          throw new DOMException('aborted', 'AbortError');
        }
        const { done, value } = await reader.read();
        if (done) break;
        if (value && value.byteLength > 0) {
          yield { offset, bytes: value };
          offset += value.byteLength;
        }
      }
    } catch (e) {
      reader.cancel();
      throw e;
    }

    if (offset !== total) {
      throw new Error(
        `Streaming range fetch received ${offset}B, expected ${total}B ` +
        `(server may have ignored Range header or truncated response).`
      );
    }
  }

  /**
   * Fetch the entire body of `url` as an ArrayBuffer. Used by readers
   * that need the whole file in memory (FIFF, CTF .res4 + .meg4).
   * Includes a 1 GB cap to surface the bandwidth/memory ceiling
   * before the browser OOMs. Range: 0-N-1 is used (not a plain GET)
   * so this routes through the same cache as range fetches.
   *
   * @param {string} url
   * @param {object} [opts]
   * @param {number} [opts.maxBytes=1073741824] - hard cap, default 1 GiB
   * @returns {Promise<ArrayBuffer>}
   * @throws if Content-Length / probeLength exceeds maxBytes, or the
   *         response is non-2xx
   */
  async function fetchBuffer(url, opts = {}) {
    const maxBytes = opts.maxBytes ?? 1073741824;  // 1 GiB
    // Probe size first so we can fail-fast with a clear message.
    const size = await probeLength(url);
    if (size > maxBytes) {
      throw new Error(
        `fetchBuffer: ${url} is ${(size / 1024 / 1024).toFixed(0)} MB ` +
        `(exceeds ${(maxBytes / 1024 / 1024).toFixed(0)} MB cap); ` +
        `use range-based readWindow instead.`,
      );
    }
    // Range request for the whole body — routes through CDN range cache.
    return rangeFetch(url, 0, size - 1);
  }

  const api = {
    probeLength, probeLengthNoHead, rangeFetch, rangeFetchStreaming, fetchBuffer,
    fetchText, fetchTextOrNull,
    registerLocal, clearLocal,
    _STREAM_THRESHOLD: STREAM_THRESHOLD,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.HttpRange = api;
})();
