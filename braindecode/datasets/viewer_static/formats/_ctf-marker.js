/* ============================================================
   formats/_ctf-marker.js — parse CTF MEG MarkerFile.mrk + BadChannels.

   MarkerFile.mrk text format (whitespace tolerant):
     PATH OF DATASET:
     /some/path/foo.ds

     NUMBER OF MARKERS:
     <N>

     [repeated N times:]
     CLASSGROUPID:
     0
     NAME:
     <label>
     COMMENT:
     <comment>
     COLOR:
     <colour>
     EDITABLE:
     Yes|No
     CLASSID:
     <id>
     NUMBER OF SAMPLES:
     <M>
     LIST OF SAMPLES:
     TRIAL NUMBER       TIME FROM SYNC POINT (in seconds)
                  +<trial>     +<seconds>
                  +<trial>     +<seconds>
                  ...

   We only extract (label, trial, onset) tuples — that's all the viewer
   needs to draw event markers. Other fields (CLASSGROUPID, COLOR, …)
   are deliberately dropped.

   BadChannels: plain text, one channel name per line. Blank lines and
   '#'-comment lines are skipped.
   ============================================================ */
(function () {
  'use strict';

  const api = {};

  /**
   * Parse a CTF MarkerFile.mrk into a flat list of point events.
   * @param {string} text - file contents.
   * @returns {Array<{ onset: number, duration: number, label: string|null, trial: number, sample: number|null }>}
   *   Always returns an array. `onset` is in seconds, `duration` is 0
   *   (CTF markers are point events). `sample` is null because the
   *   sample index requires knowing the sample rate; convert downstream.
   */
  api.parseMarkerFile = function (text) {
    if (typeof text !== 'string' || !text.length) return [];

    const lines = text.split(/\r?\n/);
    const events = [];

    let i = 0;
    while (i < lines.length) {
      // Look for the marker section: NAME: header followed by the
      // label on the next line, then a NUMBER OF SAMPLES / LIST OF
      // SAMPLES block. Each marker is independent.
      if (/^\s*NAME:\s*$/i.test(lines[i])) {
        const label = (lines[i + 1] || '').trim() || null;
        // Skip ahead until we find LIST OF SAMPLES (or a new NAME:
        // which means a malformed marker — just abandon it).
        let j = i + 2;
        while (j < lines.length && !/^\s*LIST OF SAMPLES:?\s*$/i.test(lines[j])) {
          if (/^\s*NAME:\s*$/i.test(lines[j])) { j = -1; break; }
          j++;
        }
        if (j < 0 || j >= lines.length) { i = i + 2; continue; }
        // Skip the header row (TRIAL NUMBER ... TIME ...).
        let k = j + 1;
        if (k < lines.length && /TRIAL NUMBER/i.test(lines[k])) k++;
        // Collect rows until a blank line or a new section header.
        while (k < lines.length) {
          const ln = lines[k];
          if (/^\s*$/.test(ln)) break;
          if (/^\s*[A-Z][A-Z\s]*:\s*$/.test(ln)) break;
          // Two whitespace-separated numbers: trial, onsetSeconds.
          const m = /([+-]?\d+(?:\.\d*)?)\s+([+-]?\d+(?:\.\d*)?)/.exec(ln);
          if (m) {
            const trial = parseInt(m[1], 10);
            const onset = parseFloat(m[2]);
            if (Number.isFinite(trial) && Number.isFinite(onset)) {
              events.push({ onset, duration: 0, label, trial, sample: null });
            }
          }
          k++;
        }
        i = k;
        continue;
      }
      i++;
    }

    return events;
  };

  /**
   * Parse the CTF BadChannels text file (one channel name per line).
   * Skips blank lines and lines starting with '#'.
   * @param {string} text
   * @returns {string[]}
   */
  api.parseBadChannels = function (text) {
    if (typeof text !== 'string' || !text.length) return [];
    return text.split(/\r?\n/)
      .map(l => l.trim())
      .filter(l => l.length > 0 && !l.startsWith('#'));
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.CTFMarker = api;
})();
