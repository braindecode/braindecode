/* ============================================================
   traces.js — canvas trace renderer for the EEG viewer.

   The renderer is intentionally stateless: callers manage the
   view (start time, window width, gain) and pass per-channel
   Float32Arrays from whichever format reader is in play. We do
   per-channel DC removal and amplitude normalisation before
   drawing so the same code handles raw EEG (large DC offsets)
   and reference-removed EEG (already mean-zero).

   When the visible-sample count exceeds the plot width in
   pixels we decimate via block min/max — every output pixel
   gets a vertical line spanning the min and max of the samples
   that fall in its bucket. This preserves spikes faithfully and
   costs O(samples_visible) per draw, which is fine for typical
   windows (10 s × 1 kHz × 64 ch ≈ 640 k samples).
   ============================================================ */
(function () {
  'use strict';

  // Sub-modules (Lane E1). Browser side-loads them via <script> tags
  // before this file, exposing globalThis.TraceScaleBar /
  // .TraceEventMarkers. Node tests fall back to require() so
  // `require('../traces.js')` continues to work standalone.
  const TraceScaleBar = (typeof globalThis !== 'undefined' && globalThis.TraceScaleBar)
    || (typeof require !== 'undefined' ? require('./traces/scale-bar.js') : null);
  const TraceEventMarkers = (typeof globalThis !== 'undefined' && globalThis.TraceEventMarkers)
    || (typeof require !== 'undefined' ? require('./traces/event-markers.js') : null);
  // Local aliases — used by drawTimeAxis call sites + the legacy
  // _computeScaleBarGeometry test seam. Pulled out of the sub-module
  // so the rest of traces.js doesn't need to know it's been split.
  const niceRound = TraceScaleBar.niceRound;
  const formatScale = TraceScaleBar.formatScale;
  const drawScaleBar = TraceScaleBar.drawScaleBar;
  const drawEventMarkers = TraceEventMarkers.drawEventMarkers;

  // Plot-area inset; rest of the canvas is reserved for channel
  // labels (left), the time axis (bottom), and a vertical amplitude
  // scale bar (right). PAD_RIGHT was 12 before adding the scale bar
  // (data-viz review tier 1) — bumped to 70 to fit `[│] 100 µV`.
  const PAD_LEFT = 96;
  const PAD_RIGHT = 70;
  const PAD_TOP = 8;
  const PAD_BOTTOM = 28;

  // Minimum per-channel slot height in CSS pixels. When
  // n_channels × MIN_SLOT_PX exceeds plotH, the renderer paginates
  // the visible channels (caller controls offset via `opts.channel_offset`;
  // viewer.js wires PgUp/PgDn to scroll). 16 px keeps a single-pixel
  // trace plus 7 px of breathing room above and below.
  const MIN_SLOT_PX = 16;

  // Trace stroke widths, separate so we can keep each channel
  // legible on a HiDPI display without bleeding into neighbours.
  const TRACE_WIDTH_DEFAULT = 1.0;
  const TRACE_WIDTH_BAD = 1.4;

  // Palette aligned with the page's CSS custom properties so the canvas
  // reads as part of the same instrument as the surrounding chrome.
  // (Canvas can't consume CSS vars directly; values mirror the :root
  // declarations in styles.css.)
  const BG_COLOR      = '#fbfaf6';   // --surface (cream paper)
  const TRACE_COLOR   = '#0072B2';   // Okabe-Ito blue
  const BAD_COLOR     = '#D55E00';   // Okabe-Ito vermillion
  const BAD_SLOT_COLOR = '#c8c8c8';  // muted grey fill for bad-channel slot background (R=200, delta ≥ 50 vs BG)
  const AXIS_COLOR    = '#b5b8bd';   // --ink-3
  const SLOT_COLOR  = '#e8e5dc';   // --line-2 — hairlines between channels
  const LABEL_COLOR = '#3a3d42';   // --ink-2
  const LABEL_FONT  = "10.5px 'IBM Plex Mono', ui-monospace, Menlo, monospace";
  const AXIS_FONT   = "9.5px 'IBM Plex Mono', ui-monospace, Menlo, monospace";

  // Channel-type suffix in the row label. Suppressed for EEG (the
  // dominant type — would just clutter every row) and shown for
  // EOG/ECG/EMG/RESP/MISC/etc. so non-EEG rows are scannable.
  const TYPE_LABEL_COLOR = '#8b8e94';   // --muted in styles.css
  const TYPE_LABEL_FONT  = "8.5px 'IBM Plex Mono', ui-monospace, Menlo, monospace";

  // Event onset markers moved to traces/event-markers.js (Lane E1) —
  // the EVENT_LINE_COLOR / EVENT_LABEL_COLOR constants live there now.

  // Per-channel-type dash pattern. Redundant encoding for grayscale
  // print readability — colour alone collapses to mid-grey when
  // desaturated, dash patterns survive. EEG is solid (the default
  // case for almost every recording); the other types each get a
  // distinct rhythm.
  const TYPE_DASH = {
    EEG:  [],
    EOG:  [5, 2],
    ECG:  [2, 2],
    EMG:  [4, 1, 1, 1],
    RESP: [6, 3],
    TEMP: [3, 3],
    MISC: [],
  };

  // 6σ covers > 99.7% of any roughly normal-distributed channel,
  // and is comfortably larger than the ±3σ stddev display the
  // EEGLAB browser uses. Clipping above this is acceptable —
  // pathological samples (saturated electrodes) shouldn't compress
  // the rest of the trace.
  const STDDEV_FILL_FACTOR = 6;

  // Block min/max kicks in only when there are at least ~2 samples
  // per pixel; below that, drawing the polyline directly is sharper.
  const DECIMATE_RATIO = 2;

  // Cached canvas geometry. `clientWidth`/`clientHeight` reads force a
  // layout flush on every draw; we instead listen for resize once and
  // refresh on demand. The first draw always probes (no cache yet),
  // subsequent draws use the cached values until ResizeObserver fires.
  // Note: the ResizeObserver pins the canvas in memory through the
  // closure. Fine for the viewer's single-canvas use; if a future
  // page draws into multiple short-lived canvases, switch to an
  // explicit dispose() call when the canvas leaves the DOM.
  const _canvasDims = new WeakMap();   // canvas → { dpr, cssW, cssH }
  function deviceFitCanvas(canvas) {
    let dims = _canvasDims.get(canvas);
    if (!dims) {
      dims = { dpr: 1, cssW: 0, cssH: 0 };
      _canvasDims.set(canvas, dims);
      if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => { dims.cssW = 0; dims.cssH = 0; });
        ro.observe(canvas);
      }
    }
    if (!dims.cssW || !dims.cssH) {
      dims.dpr = window.devicePixelRatio || 1;
      dims.cssW = canvas.clientWidth;
      dims.cssH = canvas.clientHeight;
    }
    const w = Math.max(1, Math.round(dims.cssW * dims.dpr));
    const h = Math.max(1, Math.round(dims.cssH * dims.dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    return dims;
  }

  // meanStd: single-pass mean + std. The previous WeakMap cache keyed by
  // Float32Array identity had ~zero hit rate in practice — readWindow
  // returns fresh subarrays per pan, so references never repeat across
  // pans (only same-frame redraws would hit, and they're already fast).
  // The cache overhead (WeakMap get/set + object alloc) cost more than
  // it saved, so we dropped it. Hoisting `v = data[i]` halves the index
  // reads in the hot loop.
  function meanStd(data, n) {
    if (n <= 0) return { mean: 0, std: 0 };
    let s = 0, ss = 0;
    for (let i = 0; i < n; i++) {
      const v = data[i];
      s += v;
      ss += v * v;
    }
    const mean = s / n;
    const variance = Math.max(0, ss / n - mean * mean);
    return { mean, std: Math.sqrt(variance), n };
  }

  // When `transparent` is true, leave the canvas pixel buffer fully
  // alpha-zero so the host page (or surrounding stage CSS in embed
  // mode) bleeds through. Otherwise paint the cream paper BG that
  // matches the rest of the instrument.
  function clear(ctx, w, h, transparent) {
    ctx.clearRect(0, 0, w, h);
    if (!transparent) {
      ctx.fillStyle = BG_COLOR;
      ctx.fillRect(0, 0, w, h);
    }
  }

  // Hairline divider between each channel slot. Subtle enough not to
  // compete with the trace, present enough that the eye can tell where
  // one channel ends and the next begins on dense (64+ ch) caps.
  function drawSlotDividers(ctx, plotX0, plotX1, plotY0, slotH, nCh) {
    ctx.strokeStyle = SLOT_COLOR;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let c = 1; c < nCh; c++) {
      const y = Math.round(plotY0 + c * slotH) + 0.5;     // half-pixel snap
      ctx.moveTo(plotX0, y);
      ctx.lineTo(plotX1, y);
    }
    ctx.stroke();
  }

  function drawChannelLabels(ctx, labels, types, slotH, x, y0) {
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let c = 0; c < labels.length; c++) {
      const y = y0 + (c + 0.5) * slotH;
      const type = (types && types[c] || '').toUpperCase();
      const showType = type && type !== 'EEG';
      if (showType) {
        // Right-most: type chip in muted small mono.
        ctx.font = TYPE_LABEL_FONT;
        ctx.fillStyle = TYPE_LABEL_COLOR;
        ctx.fillText(type, x - 8, y + 0.5);
        const typeW = ctx.measureText(type).width;
        // Then: name immediately to the left of the type chip.
        ctx.font = LABEL_FONT;
        ctx.fillStyle = LABEL_COLOR;
        ctx.fillText(labels[c], x - 8 - typeW - 6, y);
      } else {
        ctx.font = LABEL_FONT;
        ctx.fillStyle = LABEL_COLOR;
        ctx.fillText(labels[c], x - 8, y);
      }
    }
  }

  // niceRound + formatScale + drawScaleBar moved to traces/scale-bar.js (Lane E1).
  // drawEventMarkers moved to traces/event-markers.js (Lane E1).
  // Both surface via the TraceScaleBar / TraceEventMarkers aliases at the
  // top of this IIFE; tests still hit `_niceRound` / `_formatScale` /
  // `_computeScaleBarGeometry` via the api{} re-exports at the bottom.

  // Format an absolute number of seconds-since-midnight as HH:MM:SS.
  // Works for recording offsets that may wrap past midnight.
  function secToHHMMSS(totalSec) {
    const s = Math.floor(totalSec) % 86400;
    const hh = String(Math.floor(s / 3600) % 24).padStart(2, '0');
    const mm = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
    const ss = String(s % 60).padStart(2, '0');
    return `${hh}:${mm}:${ss}`;
  }

  // Parse an ISO 8601 string "YYYY-MM-DDTHH:MM:SS" and return the
  // number of seconds since midnight (i.e. time-of-day portion only).
  // Returns null if the string is not parseable.
  function isoToSecOfDay(isoStr) {
    if (!isoStr) return null;
    const m = isoStr.match(/T(\d{2}):(\d{2}):(\d{2})/);
    if (!m) return null;
    return parseInt(m[1], 10) * 3600 + parseInt(m[2], 10) * 60 + parseInt(m[3], 10);
  }

  // Compute tick positions and labels; returns { ticks: [{t, label}], step }.
  // When time_mode === 'clock' AND recording_start_iso is set, labels are
  // HH:MM:SS; otherwise labels are relative numeric strings.
  function computeTimeTicks(t0Sec, t1Sec, time_mode, recording_start_iso) {
    const span = t1Sec - t0Sec;
    const niceSteps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60];
    const target = span / 7;
    let step = niceSteps[0];
    for (const s of niceSteps) if (s <= target) step = s;
    const first = Math.ceil(t0Sec / step) * step;

    const useClock = time_mode === 'clock' && !!recording_start_iso;
    const startSecOfDay = useClock ? isoToSecOfDay(recording_start_iso) : null;

    const ticks = [];
    for (let t = first; t <= t1Sec + 1e-9; t += step) {
      let label;
      if (useClock && startSecOfDay !== null) {
        label = secToHHMMSS(startSecOfDay + t);
      } else {
        label = t.toFixed(step >= 1 ? 0 : 2);
      }
      ticks.push({ t, label });
    }
    return { ticks, step, useClock };
  }

  function drawTimeAxis(ctx, x0, x1, y, t0Sec, t1Sec, time_mode, recording_start_iso) {
    ctx.strokeStyle = AXIS_COLOR;
    ctx.fillStyle = AXIS_COLOR;
    ctx.font = AXIS_FONT;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x0, y);
    ctx.lineTo(x1, y);
    ctx.stroke();

    const span = t1Sec - t0Sec;
    const { ticks, useClock, step } = computeTimeTicks(t0Sec, t1Sec, time_mode, recording_start_iso);

    // Minor ticks: 4 between each major (5 sub-divisions), no labels,
    // shorter (2 px vs 4 px). Visual scaffolding for fine time
    // discrimination without crowding the major-tick labels.
    const minorStep = step / 5;
    const firstMinor = Math.ceil(t0Sec / minorStep) * minorStep;
    ctx.beginPath();
    for (let t = firstMinor; t <= t1Sec + 1e-9; t += minorStep) {
      // Skip positions that coincide with a major tick.
      const r = (t / step);
      if (Math.abs(r - Math.round(r)) < 1e-6) continue;
      const x = x0 + ((t - t0Sec) / span) * (x1 - x0);
      ctx.moveTo(x, y);
      ctx.lineTo(x, y + 2);
    }
    ctx.stroke();

    ctx.beginPath();
    for (const { t, label } of ticks) {
      const x = x0 + ((t - t0Sec) / span) * (x1 - x0);
      ctx.moveTo(x, y);
      ctx.lineTo(x, y + 4);
      ctx.fillText(useClock ? label : label + ' s', x, y + 6);
    }
    ctx.stroke();

    return ticks.map(tk => tk.label);
  }

  // Module-scope scratch buffers reused across decimate calls. Each
  // pan touches every channel; allocating fresh `Float32Array(nPixels)`
  // pairs adds ~64×2KB×60fps of GC pressure that this avoids. Callers
  // must consume the result before the next decimate call (the renderer
  // does that inside one synchronous draw pass).
  let _scratchMn = new Float32Array(0);
  let _scratchMx = new Float32Array(0);
  function decimateMinMax(data, n, nPixels) {
    if (_scratchMn.length < nPixels) {
      _scratchMn = new Float32Array(nPixels);
      _scratchMx = new Float32Array(nPixels);
    }
    const mn = _scratchMn, mx = _scratchMx;
    if (n <= 0 || nPixels <= 0) return { mn, mx };
    const step = n / nPixels;
    let from = 0;
    for (let p = 0; p < nPixels; p++) {
      const to = (p === nPixels - 1) ? n : Math.floor((p + 1) * step);
      let lo = data[from], hi = lo;
      for (let i = from + 1; i < to; i++) {
        const v = data[i];
        if (v < lo) lo = v;
        else if (v > hi) hi = v;
      }
      mn[p] = lo; mx[p] = hi;
      from = to;
    }
    return { mn, mx };
  }

  function drawChannelDecimated(ctx, data, nVisible, plotX0, plotW, yCenter, vToPx) {
    const nPixels = Math.max(1, Math.floor(plotW));
    const { mn, mx } = decimateMinMax(data, nVisible, nPixels);
    ctx.beginPath();
    for (let p = 0; p < nPixels; p++) {
      const x = plotX0 + p;
      ctx.moveTo(x, yCenter - mx[p] * vToPx);
      ctx.lineTo(x, yCenter - mn[p] * vToPx);
    }
    ctx.stroke();
  }

  function drawChannelPolyline(ctx, data, nVisible, plotX0, plotW, yCenter, vToPx) {
    if (nVisible <= 0) return;
    const dx = plotW / Math.max(1, nVisible - 1);
    ctx.beginPath();
    ctx.moveTo(plotX0, yCenter - data[0] * vToPx);
    for (let s = 1; s < nVisible; s++) {
      ctx.lineTo(plotX0 + s * dx, yCenter - data[s] * vToPx);
    }
    ctx.stroke();
  }

  // Single entry point. `channels` is a per-channel typed-array
  // (any length ≥ nVisible); we read indices [0, nVisible).
  // `bad_mask` is an optional boolean[]; channels marked true render
  // in the highlight colour.
  //
  // 1C streaming: when `opts.partial_fill` is present:
  //   { sample_start, sample_end, total_samples }
  // ONLY the x-region corresponding to those samples is redrawn.
  // The rest of the canvas is left intact (stale content stays visible
  // until the full window arrives). The caller is responsible for
  // calling draw() without partial_fill for the first chunk (to clear
  // the old frame) and subsequent chunks with partial_fill.
  function draw(canvas, opts) {
    const { dpr, cssW, cssH } = deviceFitCanvas(canvas);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Partial-fill mode: only repaint the x-band for the new samples
    // (skip the full clear so already-painted x-bands stay intact).
    //
    // Exception: when the caller flags `full_clear: true` we DO clear the
    // whole canvas first — this is the streaming first-chunk case, where
    // the previous render's pixels would otherwise persist outside the
    // new chunk's band. The polyline x-mapping (downstream) still uses
    // `total_samples` so the partial data lands in its real x position.
    const partialFill = opts.partial_fill;
    // Tracks the x-band cleared on a partial_fill (non-full) chunk.
    // Passed to drawEventMarkers so it only redraws events whose line
    // or label intersects the band — avoiding alpha-compound "ghost
    // traces" on the semi-transparent event hairlines + labels when
    // the same event is re-drawn on every chunk of a streaming pan.
    let eventClearedBand = null;
    if (partialFill) {
      if (partialFill.full_clear) {
        clear(ctx, cssW, cssH, opts.transparent === true);
      } else {
        const allChannels0 = opts.channels;
        const nSamplesTotal = partialFill.total_samples || (allChannels0[0] ? allChannels0[0].length : 0);
        const plotX0 = PAD_LEFT;
        const plotX1 = cssW - PAD_RIGHT;
        const plotW = plotX1 - plotX0;
        if (nSamplesTotal > 0 && plotW > 0) {
          const xStart = Math.floor(plotX0 + (partialFill.sample_start / nSamplesTotal) * plotW);
          const xEnd = Math.ceil(plotX0 + ((partialFill.sample_end + 1) / nSamplesTotal) * plotW);
          const bandW = Math.max(1, xEnd - xStart);
          ctx.fillStyle = BG_COLOR;
          ctx.fillRect(xStart, 0, bandW, cssH);
          eventClearedBand = { xStart, xEnd };
        }
      }
    } else {
      clear(ctx, cssW, cssH, opts.transparent === true);
    }

    const allChannels = opts.channels;
    const totalCh = allChannels.length;
    if (!totalCh) return;
    const allLabels = opts.channel_labels || [];
    const allTypes  = opts.channel_types  || [];
    const allColors = opts.channel_colors || null;
    const allBad    = opts.bad_mask       || null;

    const t0 = opts.start_sec;
    const nSamplesVisible0 = Math.min(opts.n_samples_visible, allChannels[0].length);
    // Time axis and event positions are anchored to the FULL window even
    // during streaming, so ticks don't jitter as chunks arrive and so events
    // appear at their true t-coordinate. Without partial_fill the full
    // window equals the partial (chunk == window). With partial_fill we
    // trust `total_samples` from the caller.
    const nSamplesTotal = (opts.partial_fill && opts.partial_fill.total_samples)
      ? opts.partial_fill.total_samples
      : nSamplesVisible0;
    const t1 = t0 + nSamplesTotal / opts.fs;
    const gain = opts.gain ?? 1;

    const plotX0 = PAD_LEFT;
    const plotX1 = cssW - PAD_RIGHT;
    const plotY0 = PAD_TOP;
    const plotH  = cssH - PAD_TOP - PAD_BOTTOM;
    const plotW  = plotX1 - plotX0;
    if (plotW <= 4 || plotH <= 4) return;

    // Virtual scroll: when n_channels would force slots smaller than
    // MIN_SLOT_PX, paginate. Caller (viewer.js) drives `channel_offset`
    // via PgUp/PgDn so pages stay in sync with the user's intent.
    const maxVisible = Math.max(1, Math.floor(plotH / MIN_SLOT_PX));
    const offsetRaw = opts.channel_offset || 0;
    const offset = Math.max(0, Math.min(Math.max(0, totalCh - 1), offsetRaw));
    const visibleN = Math.min(maxVisible, totalCh - offset);
    const slice = (arr) => (arr ? arr.slice(offset, offset + visibleN) : arr);
    const channels = totalCh > maxVisible ? slice(allChannels) : allChannels;
    const labels   = totalCh > maxVisible ? slice(allLabels)   : allLabels;
    const types    = totalCh > maxVisible ? slice(allTypes)    : allTypes;
    const colors   = totalCh > maxVisible ? slice(allColors)   : allColors;
    const badMask  = totalCh > maxVisible ? slice(allBad)      : allBad;

    const nCh = channels.length;
    const nVisible = Math.min(opts.n_samples_visible, channels[0].length);
    const slotH = plotH / nCh;
    const halfSlotPx = slotH * 0.45;
    // When streaming, the chunk contains only `nVisible` of the eventual
    // `nSamplesTotal` samples. The polyline/decimator must map sample i to
    // x = plotX0 + (i / (nSamplesTotal-1)) * plotW so partial data stays in
    // its real x-band instead of stretching across the whole plot. Without
    // this remapping, every chunk paints a different stretched lookalike of
    // the final shape and the band-clear leaves ghost lines from prior
    // chunks — the visible "trace residue" reported during fast pan.
    const effectivePlotW = (nSamplesTotal > nVisible)
      ? plotW * (nVisible / nSamplesTotal)
      : plotW;

    drawSlotDividers(ctx, plotX0, plotX1, plotY0, slotH, nCh);
    drawChannelLabels(ctx, labels, types, slotH, plotX0, plotY0);
    const xLabels = drawTimeAxis(
      ctx, plotX0, plotX1, plotY0 + plotH + 4, t0, t1,
      opts.time_mode, opts.recording_start_iso
    );

    // Event onset markers: rendered BEFORE traces so they read as
    // background scaffolding (muted green hairlines) rather than as
    // additional data.
    drawEventMarkers(ctx, opts.events, t0, t1, plotX0, plotX1, plotY0, plotH, eventClearedBand);

    // Decimation threshold uses the *effective* plot width — samples-per-
    // pixel density is the same regardless of partial vs full (we just have
    // fewer samples mapped into fewer pixels), so the trigger stays accurate.
    const decimated = nVisible > effectivePlotW * DECIMATE_RATIO;
    const stds = [];
    for (let c = 0; c < nCh; c++) {
      const data = channels[c];
      const isBad = badMask ? badMask[c] === true : false;

      const { mean, std } = meanStd(data, nVisible);
      stds.push(std);
      // Empty channel guard: a flat line stays at center, scale stays
      // finite, no NaN propagation.
      const ampl = std > 0 ? std * STDDEV_FILL_FACTOR : 1;
      const vToPx = (halfSlotPx * gain) / (ampl / 2);

      const yCenter = plotY0 + (c + 0.5) * slotH;
      ctx.save();
      // Outer clip: confine traces to the plot region (no leak into
      // PAD_LEFT label gutter, time-axis area, or PAD_TOP). Inside the
      // plot region, traces are NOT clipped to their per-channel slot
      // — over-driven gain is allowed to bleed into adjacent slots so
      // the user keeps seeing signal shape rather than a flat saturation
      // line. Trade-off: at very high gain a noisy channel can briefly
      // overlap its neighbours; that's preferable to losing detail.
      ctx.beginPath();
      ctx.rect(plotX0, plotY0, plotW, plotH);
      ctx.clip();

      if (isBad) {
        ctx.fillStyle = BAD_SLOT_COLOR;
        ctx.fillRect(plotX0, plotY0 + c * slotH, plotW, slotH);
      }

      const yC = yCenter + mean * vToPx;
      const typeColor = (colors && colors[c]) ? colors[c] : TRACE_COLOR;
      ctx.strokeStyle = isBad ? BAD_COLOR : typeColor;
      ctx.lineWidth = isBad ? TRACE_WIDTH_BAD : TRACE_WIDTH_DEFAULT;
      // Dash pattern by channel type — redundant encoding for
      // grayscale-safe distinction (color alone collapses to mid-grey
      // when desaturated).
      const type = (types && types[c] || '').toUpperCase();
      ctx.setLineDash(TYPE_DASH[type] || []);
      if (decimated) {
        drawChannelDecimated(ctx, data, nVisible, plotX0, effectivePlotW, yC, vToPx);
      } else {
        drawChannelPolyline(ctx, data, nVisible, plotX0, effectivePlotW, yC, vToPx);
      }
      ctx.restore();
    }

    // Vertical amplitude scale bar in the right gutter. Use median std
    // across visible channels as a representative amplitude — robust
    // to one or two saturated electrodes that would otherwise dominate
    // the mean.
    const sortedStds = stds.filter(s => s > 0).sort((a, b) => a - b);
    const medianStd = sortedStds.length ? sortedStds[Math.floor(sortedStds.length / 2)] : 0;
    // slot_µV = how many µV does one slot height represent?
    // From `vToPx = (halfSlotPx * gain) / (ampl/2)` with halfSlotPx = 0.45*slotH and ampl = 6*std:
    //   slot_µV = slotH / vToPx = std * 6 / (gain * 0.9) ≈ 6.67 * std / gain
    const slotMicrovolts = medianStd > 0 ? (medianStd * STDDEV_FILL_FACTOR) / (gain * 0.9) : 0;
    drawScaleBar(ctx, slotH, slotMicrovolts, plotX1, plotY0, plotH);

    // Persist for tests + viewer.js (gain readout, virtual-scroll
    // PgUp/PgDn clamping).
    api.lastDrawnXLabels = xLabels;
    api.lastSlotMicrovolts = slotMicrovolts;
    api.lastMaxVisibleChannels = maxVisible;
    api.lastChannelOffset = offset;
    api.lastTotalChannels = totalCh;
    if (typeof window !== 'undefined' && window.TraceRenderer) {
      window.TraceRenderer.lastDrawnXLabels = xLabels;
      window.TraceRenderer.lastSlotMicrovolts = slotMicrovolts;
      window.TraceRenderer.lastMaxVisibleChannels = maxVisible;
      window.TraceRenderer.lastChannelOffset = offset;
      window.TraceRenderer.lastTotalChannels = totalCh;
    }
    if (typeof globalThis !== 'undefined' && globalThis.TraceRenderer) {
      globalThis.TraceRenderer.lastDrawnXLabels = xLabels;
      globalThis.TraceRenderer.lastSlotMicrovolts = slotMicrovolts;
      globalThis.TraceRenderer.lastMaxVisibleChannels = maxVisible;
      globalThis.TraceRenderer.lastChannelOffset = offset;
      globalThis.TraceRenderer.lastTotalChannels = totalCh;
    }
  }

  // PAD_* are exported so the page can compute drag-pixel-to-time
  // mapping against the same plot geometry the renderer uses. Without
  // this, drag math would have to duplicate the magic numbers.
  const api = {
    draw, decimateMinMax, meanStd,
    PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM, MIN_SLOT_PX,
    // Test-surface export: niceRound is module-private but its boundary
    // at v<=0 was mutation-blind (mutant 131). Promoting it here lets a
    // direct unit test pin the contract without exposing more surface
    // than necessary — the leading underscore marks it as debug-only.
    _niceRound: niceRound,
    // Iteration-3 debug exports: time-axis + scale-bar helpers were
    // major mutant-survivor clusters (lines 200-399 in mutation-survivors
    // 2026-05). Each is module-private; exposing under an _-prefix lets
    // direct unit tests pin the boundary contracts (HH:MM:SS wrap, ISO
    // parsing fallback, niceSteps table selection, µV/mV format split)
    // without widening the public surface.
    _secToHHMMSS: secToHHMMSS,
    _isoToSecOfDay: isoToSecOfDay,
    _computeTimeTicks: computeTimeTicks,
    _formatScale: formatScale,
    // Iteration-4 debug exports: pure-function siblings of drawScaleBar
    // and drawTimeAxis that return the *geometry* those routines would
    // compute, without touching ctx. They mirror the in-function math
    // line-for-line — used by tests/unit-traces-scalebar-axis.test.mjs
    // to pin position-to-µV mapping and tick layout against the
    // iteration-4 surviving-mutant clusters (lines 200-249 and 350-399
    // in docs/mutation-survivors-2026-05.md).
    _computeScaleBarGeometry(slotMicrovolts, slotH, plotX1, plotY0, plotHeight) {
      // Mirrors drawScaleBar at traces.js:217-225 (everything before ctx).
      // Param naming: plotHeight (not plotH) only to avoid a duplicate-name
      // SyntaxError under 'use strict'; the math matches drawScaleBar's
      // internal `plotH` variable exactly.
      if (!isFinite(slotMicrovolts) || slotMicrovolts <= 0) return null;
      const targetMv = niceRound(slotMicrovolts * 0.5);
      const px = (targetMv / slotMicrovolts) * slotH;
      if (!isFinite(px) || px < 8) return null;
      const x = plotX1 + 18;
      const yBottom = plotY0 + plotHeight - 12;
      const yTop = yBottom - px;
      return { targetMv, px, x, yBottom, yTop };
    },
    _computeTimeAxisLayout(x0, x1, t0Sec, t1Sec, time_mode, recording_start_iso) {
      // Mirrors drawTimeAxis at traces.js:333-374 *without* ctx calls.
      // Returns the major-tick and minor-tick positions drawTimeAxis would
      // produce. The minor-skip-at-major rule (Math.abs(r-Math.round(r))<1e-6)
      // matches the implementation exactly so the layout shim is the
      // ground truth for tests.
      const { ticks, step, useClock } = computeTimeTicks(t0Sec, t1Sec, time_mode, recording_start_iso);
      const span = t1Sec - t0Sec;
      if (span <= 0) return { major: [], minor: [], useClock, step };
      const major = ticks.map(({ t, label }) => ({
        t,
        label,
        x: x0 + ((t - t0Sec) / span) * (x1 - x0),
      }));
      const minorStep = step / 5;
      const firstMinor = Math.ceil(t0Sec / minorStep) * minorStep;
      const minor = [];
      for (let t = firstMinor; t <= t1Sec + 1e-9; t += minorStep) {
        const r = t / step;
        if (Math.abs(r - Math.round(r)) < 1e-6) continue;
        minor.push({
          t,
          x: x0 + ((t - t0Sec) / span) * (x1 - x0),
        });
      }
      return { major, minor, useClock, step };
    },
    lastDrawnXLabels: [],
    lastSlotMicrovolts: 0,
    lastMaxVisibleChannels: 0,
    lastChannelOffset: 0,
    lastTotalChannels: 0,
  };
  if (typeof window !== 'undefined') window.TraceRenderer = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.TraceRenderer = api;
})();
