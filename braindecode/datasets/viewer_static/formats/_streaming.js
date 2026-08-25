/* ============================================================
   formats/_streaming.js — shared utilities for per-format
   streaming decode wrappers.

   All functions are pure (no I/O), designed for use in both
   the Web Worker and unit tests.
   ============================================================ */
(function () {
  'use strict';

  // Split `incoming` bytes into complete records of `recordSize` bytes,
  // prepending any `leftover` bytes from the previous chunk.
  // Returns { completeRecordBytes: Uint8Array, leftover: Uint8Array }
  // `completeRecordBytes` is a single flat buffer containing N complete
  // records (may be 0 bytes if there aren't enough bytes for even one).
  // `leftover` contains the trailing partial record bytes (0 .. recordSize-1).
  function decodeChunkBoundary(leftover, incoming, recordSize) {
    // Fast path: no leftover and incoming is record-aligned
    let combined;
    if (leftover.length === 0) {
      combined = incoming;
    } else {
      combined = new Uint8Array(leftover.length + incoming.length);
      combined.set(leftover, 0);
      combined.set(incoming, leftover.length);
    }

    const nCompleteRecords = Math.floor(combined.length / recordSize);
    const completedBytes = nCompleteRecords * recordSize;
    const completeRecordBytes = combined.subarray(0, completedBytes);
    const newLeftover = combined.subarray(completedBytes);

    return { completeRecordBytes, leftover: new Uint8Array(newLeftover) };
  }

  // Compute how many EDF records fit entirely before `startSample`
  // and what the in-record sample offset is within the first overlapping record.
  // Returns { firstRec, startOffsetInFirstRec }
  function edfRecordLayout(startSample, samplesPerRecord) {
    const firstRec = Math.floor(startSample / samplesPerRecord);
    const startOffsetInFirstRec = startSample - firstRec * samplesPerRecord;
    return { firstRec, startOffsetInFirstRec };
  }

  // How many records span a window of nWin samples starting at startSample
  // (already aligned to firstRec * samplesPerRecord)?
  function edfRecordCount(startSample, nWin, samplesPerRecord) {
    const end = startSample + nWin;
    const firstRec = Math.floor(startSample / samplesPerRecord);
    const lastRec = Math.ceil(end / samplesPerRecord);
    return lastRec - firstRec;
  }

  const api = { decodeChunkBoundary, edfRecordLayout, edfRecordCount };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.StreamingUtils = api;
})();
