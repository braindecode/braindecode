/* ============================================================
   traces/scale-bar.js — amplitude scale-bar rendering helpers
   extracted from traces.js so the parent file stays under the
   ~600-line readability threshold. (Lane E1.)

   Three exports:
     - niceRound(v):    round up to a 1/2/5×10^N "human" value
     - formatScale(µV): µV/mV unit-aware string
     - drawScaleBar(...): vertical scale glyph in the right gutter

   Constants mirror the parent file's palette/font so the sub-
   module is self-contained — if styles.css's :root tokens
   change, update BOTH files.
   ============================================================ */
'use strict';
(function () {
  // Mirror traces.js palette / fonts (see styles.css :root).
  const LABEL_COLOR = '#3a3d42';   // --ink-2
  const AXIS_FONT   = "9.5px 'IBM Plex Mono', ui-monospace, Menlo, monospace";

  // Round a positive number up to a "nice" round value from the
  // 1/2/5×10^N family. Used by the amplitude scale bar so its label
  // is human-friendly (50/100/200/500 µV, never 173 µV).
  function niceRound(v) {
    if (v <= 0) return 1;
    const exp = Math.floor(Math.log10(v));
    const f = v / Math.pow(10, exp);
    const niceF = f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10;
    return niceF * Math.pow(10, exp);
  }

  // Format an amplitude scale value in human-readable units. EEG is
  // typically 1-500 µV; large drift channels can reach mV.
  function formatScale(microvolts) {
    if (microvolts < 1)    return microvolts.toFixed(2) + ' µV';
    if (microvolts < 1000) return Math.round(microvolts) + ' µV';
    return (microvolts / 1000).toFixed(1) + ' mV';
  }

  // Vertical amplitude scale bar in the right gutter. Picks a nice
  // round µV value that maps to ~50% of a slot height so the glyph
  // is visible without crowding adjacent slots.
  function drawScaleBar(ctx, slotH, slotMicrovolts, plotX1, plotY0, plotH) {
    if (!isFinite(slotMicrovolts) || slotMicrovolts <= 0) return;
    const targetMv = niceRound(slotMicrovolts * 0.5);
    const px = (targetMv / slotMicrovolts) * slotH;
    if (!isFinite(px) || px < 8) return;

    const x = plotX1 + 18;
    const yBottom = plotY0 + plotH - 12;
    const yTop = yBottom - px;

    ctx.save();
    ctx.strokeStyle = LABEL_COLOR;
    ctx.fillStyle = LABEL_COLOR;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x + 0.5, yTop);
    ctx.lineTo(x + 0.5, yBottom);
    ctx.moveTo(x - 3, yTop + 0.5);
    ctx.lineTo(x + 4, yTop + 0.5);
    ctx.moveTo(x - 3, yBottom + 0.5);
    ctx.lineTo(x + 4, yBottom + 0.5);
    ctx.stroke();
    ctx.font = AXIS_FONT;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(formatScale(targetMv), x + 8, (yTop + yBottom) / 2);
    ctx.restore();
  }

  const api = { niceRound, formatScale, drawScaleBar };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.TraceScaleBar = api;
})();
