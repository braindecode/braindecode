/* ============================================================
   formats/kriss.js — KRISS MEG stub reader for eegdash-viewer.

   STATUS: STUB READER (Tier 1).

   KRISS (Korea Research Institute of Standards and Science) is one of
   the seven MEG vendors named in the BIDS-MEG appendix. A recording
   produces a `.kdf` file plus the companion sidecars described in
     https://bids-specification.readthedocs.io/en/stable/appendices/meg-file-formats.html#kriss
       - sub-<label>[_ses-<label>]_headshape.txt
       - <basename>.chn      (MEG coil centre positions)
       - <basename>.trg      (event markers)
       - <basename>_digitizer.txt
   The filename convention is documented, but the .kdf binary layout
   is NOT. Cross-checked at authorship time:
     - MNE-Python's mne/io/ has no `kriss/` directory; its KIT reader
       (mne/io/kit/kit.py) makes no reference to KRISS magic bytes or
       a KRISS sub-format. (Verified via the GitHub contents API.)
     - FieldTrip's fileio/private has no `read_kriss_header.m` (returns
       404 from raw.githubusercontent.com).
     - The BIDS-MEG appendix's "KRISS" section documents only the file
       naming, not byte offsets.
   So neither of the two open-source MEG ecosystems ship a public KRISS
   reader as of the date this stub was written. The .kdf format appears
   to be vendor-internal — readable only by KRISS lab software.

   WHAT THIS STUB DOES:
   1. Detect a "KRISS-shaped" .kdf header (see isKrissShaped() — magic
      bytes + label substring within the first 64 B).
   2. If the header looks like KRISS → throw a SPECIFIC error telling
      the user the format is recognised but not yet implemented, with
      a pointer to file a request. (User-facing: clean error in the
      viewer's "this file couldn't be loaded" dialog.)
   3. If the header is clearly NOT KRISS → throw a DIFFERENT error so
      callers (and tests) can distinguish "wrong file given to KRISS
      reader" from "right file, wrong reader stage". This matters when
      a future viewer routing change accidentally hands an EDF or EEGLAB
      file to KrissReader.open — the error tells the routing layer
      exactly what happened.

   WHY A STUB IS VALUE-ADDED:
   - The viewer can now SHOW a recognisable "KRISS support pending"
     message instead of crashing with a cryptic decoding error when
     a .kdf URL arrives. (When viewer.js / worker.js are wired through
     in a follow-up PR; this lane deliberately doesn't touch them.)
   - The detection threshold is conservative enough that a real KRISS
     .kdf file (whenever we see one) would trigger the "not yet
     implemented" path, not the "this isn't KRISS" path — assuming the
     vendor used either "KDF" or "KRISS" as an ASCII signature in the
     first 64 bytes, which is the standard convention for MEG vendor
     headers (CTF .res4 starts with "RES4\x00", KIT .con uses a sysid
     string in its SYSTEM block, FIFF uses tagged blocks but lab files
     embed the lab name in the first KB).
   - Future Tier-2 work is well-scoped: replace the throw with an
     actual parser, keep the same isKrissShaped() detection, ship a
     real fixture.

   MAGIC-BYTE DETECTION (conservative — see isKrissShaped):
   - Bytes 0..3  ASCII "KDF\0" (the four-byte signature the synthetic
     fixture uses); OR
   - Any substring "KRISS" or "KDF" inside the first 64 bytes (case-
     sensitive — vendor labels are typically uppercase per FIFF / CTF
     convention).
   We deliberately accept either signal because we do NOT know which
   one the real vendor binary uses; rejecting on signature mismatch
   alone would create false negatives once a real file appears.
   ============================================================ */
(function () {
  'use strict';

  const api = {};

  // Number of bytes the reader peeks at to decide "is this KRISS-shaped?"
  // Big enough to catch a header label that's offset away from byte 0
  // (CTF uses offset 0; KIT puts its label at offset 12; FIFF tags
  // start at byte 0 too). 64 leaves room for either convention without
  // pulling so much data that the range fetch is wasteful.
  const HEADER_PROBE_BYTES = 64;

  // ASCII bytes for the two signature substrings we look for in the
  // first HEADER_PROBE_BYTES bytes of the file. Stored as Uint8Array
  // so byteIndexOf can do a plain byte comparison without re-encoding
  // on every detect() call.
  const SIG_KDF   = new Uint8Array([0x4b, 0x44, 0x46]);              // "KDF"
  const SIG_KRISS = new Uint8Array([0x4b, 0x52, 0x49, 0x53, 0x53]);  // "KRISS"

  // Stable error messages — referenced by tests + future viewer routing
  // logic. Keep these as strings (not Error subclasses) because the
  // viewer's error dialog currently does `instanceof Error` + .message
  // substring checks rather than custom-class checks. Same convention
  // as the rest of formats/*.js.
  const ERR_NOT_KRISS =
    'kriss: file does not appear to be a valid KRISS .kdf recording ' +
    '(no KRISS / KDF signature found in the first 64 bytes)';
  const ERR_NOT_IMPLEMENTED =
    'kriss: KRISS MEG (.kdf) support is not yet implemented in ' +
    'eegdash-viewer. The file header was recognised as KRISS-shaped, ' +
    'but the binary layout is vendor-internal and currently has no ' +
    'public specification. If you have a .kdf dataset you would like ' +
    'supported, please open an issue on the eegdash-viewer repository ' +
    'with a small sample file or pointer to its spec.';

  // ---- public API --------------------------------------------------

  /**
   * Synchronous detect-and-reject entry point for the KRISS .kdf stub
   * reader. Pairs with formats/kit.js api.read(buf) — present so unit
   * tests and a future drag-drop path can run without HTTP.
   *
   * Throws on every input. Two distinct error messages:
   *  - ERR_NOT_KRISS — buffer's first 64 B do NOT carry a KRISS / KDF
   *    signature, so the caller routed the wrong file here.
   *  - ERR_NOT_IMPLEMENTED — signature WAS found, but we cannot decode
   *    the body. This is the "we know it's KRISS, support pending"
   *    path the viewer will eventually surface as a user-facing
   *    message.
   *
   * @param {ArrayBuffer | Uint8Array} buf
   * @throws {Error} always — see message contract above.
   */
  api.read = function (buf) {
    if (!buf) {
      throw new Error(
        'kriss.read: buffer is required (got ' +
        (buf === null ? 'null' : typeof buf) + ')',
      );
    }
    const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
    if (u8.byteLength < 8) {
      // 8 bytes is the minimum the magic-byte check needs to be useful;
      // anything smaller can't carry an ASCII "KRISS" or "KDF\0"
      // signature in a recognisable position.
      throw new Error(
        `kriss.read: buffer too small (${u8.byteLength}B) — need at ` +
        `least 8B to inspect the header`,
      );
    }
    if (isKrissShaped(u8)) {
      throw new Error(ERR_NOT_IMPLEMENTED);
    }
    throw new Error(ERR_NOT_KRISS);
  };

  /**
   * Async stub for opening a KRISS .kdf recording over HTTP. Range-
   * fetches the first 16 KiB (per the task spec — that's well over the
   * 64 B detection window, which leaves headroom for a future Tier-2
   * implementation that needs to peek further into the file to read
   * the channel-count / sfreq fields).
   *
   * Throws ERR_NOT_KRISS or ERR_NOT_IMPLEMENTED — never returns a
   * reader-shaped object in Tier 1.
   *
   * @param {object} meta - { eeg_url: string, ... }
   * @throws {Error}
   */
  api.open = async function (meta) {
    const url = meta && (meta.eeg_url || meta.url);
    if (!url) throw new Error('kriss.open: meta.eeg_url is required');

    const HttpRange = globalThis.HttpRange;
    if (!HttpRange) throw new Error('kriss.open: globalThis.HttpRange missing');

    // Probe file size first so we don't issue a range request that
    // exceeds the file length on tiny .kdf files (a future real .kdf
    // could be huge, but a stub fixture is 1 KiB).
    const totalBytes = await HttpRange.probeLengthNoHead(url);
    if (totalBytes < 8) {
      throw new Error(
        `kriss.open: file too small (${totalBytes}B) — need at least ` +
        `8B to inspect the header`,
      );
    }

    // Fetch the first min(16 KiB, totalBytes) bytes. The 16 KiB upper
    // bound matches the task spec; we cap at totalBytes so the range
    // request doesn't go past end-of-file (some CDNs return 416 for
    // that, others clamp silently — defensively avoid both).
    const probeLen = Math.min(16 * 1024, totalBytes);
    const headerBuf = await HttpRange.rangeFetch(
      url, 0, probeLen - 1, probeLen,
    );
    const headerU8 = new Uint8Array(headerBuf);

    if (isKrissShaped(headerU8)) {
      throw new Error(ERR_NOT_IMPLEMENTED);
    }
    throw new Error(ERR_NOT_KRISS);
  };

  /**
   * Heuristic "does this look like a KRISS .kdf header?" check.
   *
   * Decision rule (conservative; OR of two independent signals):
   *   1. The 3-byte ASCII "KDF" appears anywhere in the first 64 B
   *      of the buffer, OR
   *   2. The 5-byte ASCII "KRISS" appears anywhere in the first 64 B
   *      of the buffer.
   * Both signals are case-sensitive and uppercase — matching the
   * convention used by every other open MEG format we know of (CTF
   * "RES4", FIFF tag block names, KIT's "kit" sysid string). Lower-
   * casing here would risk false positives against EEG/iEEG formats
   * that happen to spell "kdf" inside a text annotation block.
   *
   * @param {Uint8Array} u8
   * @returns {boolean}
   */
  function isKrissShaped(u8) {
    if (!(u8 instanceof Uint8Array)) {
      // Defensive — read()'s callers always pass a Uint8Array, but the
      // detector is exported via api.detect() so anyone calling it
      // directly gets a clear coercion.
      u8 = new Uint8Array(u8);
    }
    const probeEnd = Math.min(u8.byteLength, HEADER_PROBE_BYTES);
    if (probeEnd < 3) return false;
    const probe = u8.subarray(0, probeEnd);
    return byteIndexOf(probe, SIG_KDF) >= 0 ||
           byteIndexOf(probe, SIG_KRISS) >= 0;
  }

  /**
   * Find the first index `i` in haystack where haystack[i..i+needle.len]
   * matches needle. Returns -1 if absent.
   *
   * Uint8Array.prototype.indexOf only takes a scalar in some older
   * runtimes (Node ≥ 16 supports the typed-array form, but we keep
   * an explicit byte loop here for clarity + to avoid relying on
   * runtime-version quirks). Time is O(probeLen × needle.len) which
   * is fine at probeLen ≤ 64.
   */
  function byteIndexOf(haystack, needle) {
    if (needle.length === 0) return 0;
    const last = haystack.length - needle.length;
    if (last < 0) return -1;
    outer:
    for (let i = 0; i <= last; i++) {
      for (let j = 0; j < needle.length; j++) {
        if (haystack[i + j] !== needle[j]) continue outer;
      }
      return i;
    }
    return -1;
  }

  // Expose the detector + error strings on the api so tests can pin
  // both code paths without grepping for substrings. Internal-only —
  // viewer.js / worker.js use only api.open + api.read.
  api._detect = isKrissShaped;
  api._ERR_NOT_KRISS = ERR_NOT_KRISS;
  api._ERR_NOT_IMPLEMENTED = ERR_NOT_IMPLEMENTED;

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.KrissReader = api;
})();
