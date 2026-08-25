/* ============================================================
   traces/event-markers.js — event-marker rendering extracted
   from traces.js so the parent file stays under the readability
   threshold. (Lane E1.)

   `drawEventMarkers` renders event onsets as muted vertical
   hairlines, labelled at the top edge. Filters to those falling
   in the visible window. Labels are dropped when they would
   collide (< 32 px apart) so a dense burst of events looks like
   a comb instead of a wall of text.

   `clearedBand` — when set to { xStart, xEnd }, ONLY draws events
   whose line or label falls inside that band. Used by streaming
   partial_fill draws so we don't redraw the same event over and
   over (the event colors are semi-transparent — successive draws
   at the same pixel compound the alpha, producing a darker-and-
   darker "ghost trace"). Events outside the band were drawn on
   the first chunk's full_clear pass and are still on the canvas.
   Label-width slack (~100 px) ensures we also redraw events whose
   line is just LEFT of the band but whose label extends INTO it.
   ============================================================ */
'use strict';
(function () {
  // Event onset markers — muted Okabe-Ito green so events read as
  // background scaffolding rather than as another data trace.
  // (Mirrored from traces.js — keep in sync if the palette shifts.)
  const EVENT_LINE_COLOR  = 'rgba(0, 158, 115, 0.30)';
  const EVENT_LABEL_COLOR = 'rgba(0, 110, 80, 0.95)';
  const AXIS_FONT = "9.5px 'IBM Plex Mono', ui-monospace, Menlo, monospace";

  // Pixel slack for the clearedBand left edge — covers events whose
  // line is just LEFT of the band but whose label extends INTO it.
  const EVENT_LABEL_MAX_PX = 100;

  function drawEventMarkers(ctx, events, t0, t1, plotX0, plotX1, plotY0, plotH, clearedBand) {
    if (!events || !events.length) return;
    const span = t1 - t0;
    if (span <= 0) return;
    const xFor = (ev) => plotX0 + ((ev.onset - t0) / span) * (plotX1 - plotX0);
    const inBand = clearedBand
      ? (ev) => {
          const x = xFor(ev);
          return x >= clearedBand.xStart - EVENT_LABEL_MAX_PX && x <= clearedBand.xEnd + 2;
        }
      : null;
    const visible = [];
    for (const ev of events) {
      if (ev.onset < t0 || ev.onset > t1) continue;
      if (inBand && !inBand(ev)) continue;
      visible.push(ev);
    }
    if (!visible.length) return;
    ctx.save();
    ctx.strokeStyle = EVENT_LINE_COLOR;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const ev of visible) {
      const x = Math.round(xFor(ev)) + 0.5;
      ctx.moveTo(x, plotY0);
      ctx.lineTo(x, plotY0 + plotH);
    }
    ctx.stroke();
    ctx.fillStyle = EVENT_LABEL_COLOR;
    ctx.font = AXIS_FONT;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    let lastLabelX = -100;
    for (const ev of visible) {
      const x = Math.round(xFor(ev));
      if (x - lastLabelX < 32) continue;
      if (ev.label) {
        ctx.fillText(String(ev.label).slice(0, 14), x + 3, plotY0 + 1);
        lastLabelX = x;
      }
    }
    ctx.restore();
  }

  const api = { drawEventMarkers, EVENT_LABEL_MAX_PX };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.TraceEventMarkers = api;
})();
