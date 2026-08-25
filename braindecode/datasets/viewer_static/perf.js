/* ============================================================
   perf.js — auto-run benchmark mode for the EEG viewer.
   Activated by ?perf=1 in the URL. On first idle after the
   recording loads, runs four micro-benchmarks and dumps a
   structured JSON report to console.log + window.__perfReport.

   The report is the input format the chrome-devtools-mcp runbook
   (docs/perf-runbook.md) consumes — same script can be driven by
   a human in a focused tab or by an automated MCP tool sequence.

   Production usage: load the viewer normally (no ?perf=1) — this
   script is a no-op when the URL flag is absent.
   ============================================================ */
(function () {
  'use strict';

  if (!new URLSearchParams(globalThis.location.search).has('perf')) return;

  // Phase budgets (data-viz / web-perf reviews — see docs/perf-runbook.md
  // for derivation). Reported alongside measurements so the dump is
  // self-judging.
  const BUDGETS = {
    cold_pan_rtt_ms:      { p95: 800 },
    warm_pan_rtt_ms:      { p95: 100 },
    filter_apply_rtt_ms:  { p95: 200 },
    pure_draw_ms:         { p95: 8   },
    heap_growth_mb:       {  max: 5  },
  };

  function stats(arr) {
    if (!arr || !arr.length) return null;
    const sorted = [...arr].sort((a, b) => a - b);
    const at = (q) => sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * q))];
    return {
      n: arr.length,
      min: +sorted[0].toFixed(2),
      p50: +at(0.50).toFixed(2),
      p95: +at(0.95).toFixed(2),
      p99: +at(0.99).toFixed(2),
      max: +sorted[sorted.length - 1].toFixed(2),
    };
  }

  function memMB() {
    return performance.memory
      ? +(performance.memory.usedJSHeapSize / 1048576).toFixed(2)
      : null;
  }

  function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

  // Wait until the worker is up AND the first render has landed (which
  // sets TraceRenderer.lastSlotMicrovolts to a positive value).
  async function waitForReady(timeoutMs) {
    const t0 = performance.now();
    while (performance.now() - t0 < timeoutMs) {
      if (globalThis.__viewerWorker
          && globalThis.TraceRenderer?.lastSlotMicrovolts > 0) {
        return true;
      }
      await delay(100);
    }
    return false;
  }

  // Tap into worker.postMessage / onmessage to record FETCH_WINDOW RTT.
  function instrumentWorker() {
    const w = globalThis.__viewerWorker;
    if (!w || w.__perfTapped) return;
    w.__perfTapped = true;
    const sendTimes = new Map();
    const log = globalThis.__perfRttBuf = [];
    const origPost = w.postMessage.bind(w);
    w.postMessage = function (msg, transfer) {
      if (msg && msg.type === 'FETCH_WINDOW') sendTimes.set(msg.request_id, performance.now());
      return transfer ? origPost(msg, transfer) : origPost(msg);
    };
    w.addEventListener('message', (e) => {
      const m = e.data;
      if (m && m.type === 'WINDOW' && sendTimes.has(m.request_id)) {
        log.push(performance.now() - sendTimes.get(m.request_id));
        sendTimes.delete(m.request_id);
      }
    });
  }

  // Instrument TraceRenderer.draw to record per-call duration.
  function instrumentDraw() {
    const tr = globalThis.TraceRenderer;
    if (!tr || tr.__perfTapped) return;
    tr.__perfTapped = true;
    const origDraw = tr.draw;
    const log = globalThis.__perfDrawBuf = [];
    tr.draw = function (...args) {
      const t0 = performance.now();
      const r = origDraw.apply(this, args);
      log.push(performance.now() - t0);
      return r;
    };
  }

  // 1) Cold-pan workload: 10 ArrowRights with 1500 ms gap so each
  //    round-trip is observed independently, no cache hits dominating.
  async function runColdPan() {
    const before = globalThis.__perfRttBuf.length;
    for (let i = 0; i < 10; i++) {
      globalThis.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight' }));
      await delay(1500);
    }
    return globalThis.__perfRttBuf.slice(before);
  }

  // 2) Warm-pan workload: 10 fast ArrowRights, prefetch should be hot.
  async function runWarmPan() {
    const before = globalThis.__perfRttBuf.length;
    for (let i = 0; i < 10; i++) {
      globalThis.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight' }));
      await delay(60);
    }
    await delay(2000);     // let the queue drain
    return globalThis.__perfRttBuf.slice(before);
  }

  // 3) Filter-apply workload: toggle HP / LP / Notch, each forces a
  //    cache invalidation + worker round-trip with filter applied.
  async function runFilterApply() {
    const before = globalThis.__perfRttBuf.length;
    const fire = (id, val) => {
      const node = document.getElementById(id);
      if (!node) return;
      if (node.type === 'checkbox') node.checked = val;
      else node.value = String(val);
      node.dispatchEvent(new Event('change'));
    };
    fire('filter-hp-enable', true);  await delay(1500);
    fire('filter-lp-enable', true);  await delay(1500);
    fire('filter-notch-enable', true); await delay(1500);
    fire('filter-notch-enable', false);
    fire('filter-lp-enable',    false);
    fire('filter-hp-enable',    false); await delay(1500);
    return globalThis.__perfRttBuf.slice(before);
  }

  // 4) Pure-draw benchmark: 100 successive direct calls to
  //    TraceRenderer.draw with the latest channels — bypasses worker
  //    and RAF, so isolates rasterization cost.
  async function runPureDraw() {
    // We need representative channel data; capture from the last
    // resolved cache entry. Easiest: trigger a render, then re-issue
    // draw() against the same buffers.
    const tr = globalThis.TraceRenderer;
    const beforeBuf = tr.__perfDrawBuf?.length || 0;
    // Find a cached window in the closure — not exposed, so synthesize.
    const fs = 250, n = 7500;     // 30 s window
    const nCh = 36;
    const channels = Array.from({length: nCh}, (_, c) => {
      const a = new Float32Array(n);
      for (let i = 0; i < n; i++) a[i] = Math.sin(i * 0.05 + c) * (10 + c * 0.2);
      return a;
    });
    const labels = channels.map((_, c) => 'C' + c);
    const types  = labels.map(() => 'EEG');
    const canvas = document.getElementById('traces');
    const opts = {
      channels, n_samples_visible: n, channel_labels: labels, channel_types: types,
      fs, start_sec: 0, gain: 1, time_mode: 'relative',
      events: Array.from({length: 5}, (_, i) => ({ onset: i * 5 + 1, label: 'evt' + i })),
    };
    for (let i = 0; i < 100; i++) {
      tr.draw(canvas, opts);
    }
    return globalThis.__perfDrawBuf.slice(beforeBuf);
  }

  function pickGate(stat, budget) {
    if (!stat || !budget) return null;
    const fail = [];
    if (budget.p95 != null && stat.p95 > budget.p95) fail.push(`p95 ${stat.p95} > budget ${budget.p95}`);
    if (budget.max != null && stat.max > budget.max) fail.push(`max ${stat.max} > budget ${budget.max}`);
    return fail.length ? { fail } : { pass: true };
  }

  async function run() {
    const ok = await waitForReady(60_000);
    if (!ok) {
      console.error('[perf] viewer never reached ready state');
      return;
    }
    instrumentWorker();
    instrumentDraw();

    const memBaseline = memMB();
    const t0 = performance.now();

    const cold   = await runColdPan();
    const memAfterCold = memMB();
    const warm   = await runWarmPan();
    const memAfterWarm = memMB();
    const filt   = await runFilterApply();
    const memAfterFilters = memMB();
    const pureDraw = await runPureDraw();
    const memAfterDraw = memMB();
    const totalMs = +(performance.now() - t0).toFixed(0);

    const phases = {
      cold_pan_rtt_ms:      stats(cold),
      warm_pan_rtt_ms:      stats(warm),
      filter_apply_rtt_ms:  stats(filt),
      pure_draw_ms:         stats(pureDraw),
    };
    const memory_mb = [
      { label: 'baseline',         mb: memBaseline },
      { label: 'after_cold_pan',   mb: memAfterCold },
      { label: 'after_warm_pan',   mb: memAfterWarm },
      { label: 'after_filters',    mb: memAfterFilters },
      { label: 'after_pure_draw',  mb: memAfterDraw },
    ];
    const heap_growth_mb = (memBaseline != null && memAfterDraw != null)
      ? +(memAfterDraw - memBaseline).toFixed(2)
      : null;
    const gates = {
      cold_pan_rtt_ms:     pickGate(phases.cold_pan_rtt_ms,     BUDGETS.cold_pan_rtt_ms),
      warm_pan_rtt_ms:     pickGate(phases.warm_pan_rtt_ms,     BUDGETS.warm_pan_rtt_ms),
      filter_apply_rtt_ms: pickGate(phases.filter_apply_rtt_ms, BUDGETS.filter_apply_rtt_ms),
      pure_draw_ms:        pickGate(phases.pure_draw_ms,        BUDGETS.pure_draw_ms),
      heap_growth_mb:      heap_growth_mb != null
                            ? pickGate({max: heap_growth_mb}, BUDGETS.heap_growth_mb)
                            : null,
    };

    const report = {
      ua: navigator.userAgent,
      ts: new Date().toISOString(),
      url: globalThis.location.href,
      total_ms: totalMs,
      n_channels: globalThis.TraceRenderer?.lastTotalChannels,
      slot_microvolts: globalThis.TraceRenderer?.lastSlotMicrovolts,
      phases,
      memory_mb,
      heap_growth_mb,
      budgets: BUDGETS,
      gates,
    };
    globalThis.__perfReport = report;
    console.log('[perf] report ready:', JSON.stringify(report, null, 2));

    // Render a small overlay so a human in a focused tab sees the result.
    const div = document.createElement('div');
    div.id = 'perf-overlay';
    div.style.cssText = `
      position: fixed; bottom: 12px; right: 12px; z-index: 9999;
      max-width: 480px; max-height: 60vh; overflow: auto;
      background: #fbfaf6; border: 1px solid #b5b8bd; border-radius: 6px;
      padding: 12px 14px; font: 11px 'IBM Plex Mono', monospace;
      color: #17181a; box-shadow: 0 6px 22px rgba(0,0,0,0.18);
      white-space: pre-wrap; line-height: 1.45;`;
    const fmt = JSON.stringify({phases, heap_growth_mb, gates}, null, 2);
    div.textContent = '[perf=1] report ready (also at window.__perfReport)\n\n' + fmt;
    document.body.appendChild(div);
  }

  // Kick off after the page bootstraps. requestIdleCallback isn't on
  // every browser; setTimeout is portable and we're in deferred-script
  // land anyway.
  setTimeout(run, 500);
})();
