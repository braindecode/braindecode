/* ============================================================
   formats/_matv5.js — minimal MATLAB v5 / v6 .mat reader.
   Built for one job: extract the top-level numeric matrices an
   inline-data EEGLAB .set carries (`data`, `srate`, `nbchan`,
   `pnts`, optionally wrapped in a struct named `EEG`). Anything
   beyond that — sparse, cell arrays, complex, sub-systems, MAT
   v7.3 (HDF5) — is out of scope and intentionally rejected.

   Spec: https://www.mathworks.com/help/pdf_doc/pdf_doc/matfile_format.pdf

   File layout:
     [128-byte header]
       bytes 0..115: text description (ignored)
       bytes 116..123: subsystem offset (ignored — we don't follow it)
       bytes 124..125: version uint16 (must be 0x0100 for v5)
       bytes 126..127: endian uint16 (0x4D49 = 'IM' = little, 0x4949 = big)
     [data elements, each]:
       small format: u16 nbytes (>0) | u16 type | payload (≤ 4 bytes)
       long  format: u32 type        | u32 nbytes | payload, padded to 8 bytes

   Element types we care about:
     1 INT8 / 2 UINT8 / 3 INT16 / 4 UINT16 /
     5 INT32 / 6 UINT32 / 7 SINGLE / 9 DOUBLE
     14 MATRIX     — the only top-level wrapper for named variables
     15 COMPRESSED — zlib-deflated payload that decompresses to one MATRIX

   miMATRIX payload (sub-elements, in order):
     1. Array Flags  (UINT32 pair: low byte of first u32 = mxClass)
     2. Dimensions   (INT32 array, ndims entries)
     3. Array Name   (INT8 array, padded — the variable name)
     4. Real Data    (numeric type matching mxClass; column-major)
     [5. Imaginary Data — if complex flag set; rejected]

   For mxStruct (class 2), sub-elements 4..N are:
     4. Field Name Length (INT32, 1 entry — max field name length)
     5. Field Names       (INT8, nfields × maxLen, ASCII padded)
     6..6+nfields-1.      Each field as a nested miMATRIX, in order.

   Returned variable shape (one entry per top-level / struct field):
     { class, dims, data, name }
       class — 'int8'|'uint8'|'int16'|'uint16'|'int32'|'uint32'|
               'single'|'double'|'char'|'struct'
       dims  — number[]
       data  — TypedArray (numeric), string (char), or Map<string, Var> (struct)
   ============================================================ */
(function () {
  'use strict';

  const api = {};

  // miType → bytes per scalar. 0 = variable / complex (we skip).
  const TYPE_BYTES = {
    1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 4, 9: 8, 12: 8, 13: 8,
    14: 0, 15: 0, 16: 1, 17: 2, 18: 4,
  };
  const TYPE_NAME = {
    1: 'int8', 2: 'uint8', 3: 'int16', 4: 'uint16', 5: 'int32', 6: 'uint32',
    7: 'single', 9: 'double', 12: 'int64', 13: 'uint64',
    14: 'matrix', 15: 'compressed', 16: 'utf8', 17: 'utf16', 18: 'utf32',
  };

  // mxClass → preferred TypedArray ctor + miType.
  const CLASS_INFO = {
    6:  { name: 'double', ctor: Float64Array, miType: 9  },
    7:  { name: 'single', ctor: Float32Array, miType: 7  },
    8:  { name: 'int8',   ctor: Int8Array,    miType: 1  },
    9:  { name: 'uint8',  ctor: Uint8Array,   miType: 2  },
    10: { name: 'int16',  ctor: Int16Array,   miType: 3  },
    11: { name: 'uint16', ctor: Uint16Array,  miType: 4  },
    12: { name: 'int32',  ctor: Int32Array,   miType: 5  },
    13: { name: 'uint32', ctor: Uint32Array,  miType: 6  },
  };

  // Chops a flat ArrayBuffer (or Uint8Array view) into MAT v5 elements,
  // honouring the small/long element format and 8-byte padding.
  // Returns an iterator of { miType, payload: Uint8Array, payloadOffset }.
  //
  // When `opts.allowTruncated` is true and an element declares more
  // payload bytes than the buffer holds, we yield a partial-payload
  // entry flagged `truncated: true` and stop iteration. This is the
  // scanElements path: it only needs each element's ~50-byte header
  // (mxClass / dims / name / dataSubOffset/Bytes/MiType — all read
  // from the first sub-elements of the payload) and does not need
  // the tail bytes. The materializing path (parse) must NOT pass
  // this flag — its callers consume the full payload.
  function* iterElements(view, baseOffset, endOffset, opts) {
    const allowTruncated = !!(opts && opts.allowTruncated);
    let off = baseOffset;
    while (off + 8 <= endOffset) {
      const tag = view.getUint32(off, true);
      const smallNbytes = (tag >>> 16) & 0xffff;
      const smallType = tag & 0xffff;
      let miType, nbytes, payloadStart;
      if (smallNbytes !== 0 && smallNbytes <= 4) {
        miType = smallType;
        nbytes = smallNbytes;
        payloadStart = off + 4;
      } else {
        miType = tag;
        nbytes = view.getUint32(off + 4, true);
        payloadStart = off + 8;
      }
      if (payloadStart + nbytes > endOffset) {
        if (allowTruncated) {
          // Yield with whatever bytes ARE available, mark truncated.
          // Consumers that only need the header (e.g. scanElements)
          // can still extract name/dims/dataSubOffset from the partial
          // payload's first sub-elements.
          const available = Math.max(0, endOffset - payloadStart);
          yield {
            miType,
            payloadOffset: payloadStart,
            payload: new Uint8Array(view.buffer, view.byteOffset + payloadStart, available),
            truncated: true,
            declaredBytes: nbytes,
          };
          return;  // stop iteration — truncated element is the LAST one
        }
        throw new Error(`MAT element overruns container at ${off}: claims ${nbytes}B, only ${endOffset - payloadStart}B left`);
      }
      yield {
        miType,
        payloadOffset: payloadStart,
        // Slice via subarray so the payload aliases the same backing
        // buffer — no copy, fast for big numeric blocks.
        payload: new Uint8Array(view.buffer, view.byteOffset + payloadStart, nbytes),
      };
      // 8-byte alignment except for small-format elements (tag-included
      // length is already 8 bytes, no extra padding needed there).
      const consumed = (payloadStart - off) + nbytes;
      const padded = payloadStart === off + 4 ? 8 : 8 * Math.ceil(nbytes / 8);
      off = (payloadStart === off + 4)
        ? off + padded
        : payloadStart + padded;
    }
  }

  // Coerces a numeric payload of `miType` into an instance of the
  // class's preferred typed array. The on-disk type may differ from
  // the array's declared class (e.g. an int32 dims array stored as
  // miINT32 is fine, but EEGLAB sometimes saves an mxDOUBLE with a
  // miSINGLE payload to halve file size — we honour the on-disk type).
  function payloadAsTypedArray(elem) {
    const { miType, payload } = elem;
    const elemBytes = TYPE_BYTES[miType];
    if (!elemBytes) {
      throw new Error(`unsupported numeric miType ${miType} (${TYPE_NAME[miType] || '?'})`);
    }
    const length = payload.length / elemBytes;
    if (!Number.isInteger(length)) {
      throw new Error(`numeric payload length ${payload.length} not a multiple of ${elemBytes}B (miType ${miType})`);
    }
    // ArrayBuffer.isView rejects unaligned subarrays for Float64Array
    // construction — copy if the source isn't aligned to elemBytes.
    const aligned = (payload.byteOffset % elemBytes) === 0;
    if (aligned) {
      return makeTyped(miType, payload.buffer, payload.byteOffset, length);
    }
    const copy = new Uint8Array(payload.length);
    copy.set(payload);
    return makeTyped(miType, copy.buffer, 0, length);
  }

  function makeTyped(miType, buf, off, length) {
    switch (miType) {
      case 1:  return new Int8Array   (buf, off, length);
      case 2:  return new Uint8Array  (buf, off, length);
      case 3:  return new Int16Array  (buf, off, length);
      case 4:  return new Uint16Array (buf, off, length);
      case 5:  return new Int32Array  (buf, off, length);
      case 6:  return new Uint32Array (buf, off, length);
      case 7:  return new Float32Array(buf, off, length);
      case 9:  return new Float64Array(buf, off, length);
      case 16: return new Uint8Array  (buf, off, length);
      default: throw new Error(`makeTyped: unsupported miType ${miType}`);
    }
  }

  // Pulls the array-flags subelement (miUINT32, 8 bytes payload) and
  // decodes the mxClass + flag bits. Layout per spec:
  //   flags[0]: 16 reserved | 8 class | 8 (complex|global|logical|undef)
  //   flags[1]: nzmax (sparse only)
  function readArrayFlags(elem) {
    if (elem.miType !== 6) {
      throw new Error(`expected array flags (miUINT32), got miType ${elem.miType}`);
    }
    const v = new DataView(elem.payload.buffer, elem.payload.byteOffset, elem.payload.length);
    const word0 = v.getUint32(0, true);
    const mxClass = word0 & 0xff;
    const flags = (word0 >>> 8) & 0xff;
    return {
      mxClass,
      complex: !!(flags & 0x08),
      global:  !!(flags & 0x04),
      logical: !!(flags & 0x02),
    };
  }

  function readDims(elem) {
    if (elem.miType !== 5) {
      throw new Error(`expected dimensions (miINT32), got miType ${elem.miType}`);
    }
    return Array.from(payloadAsTypedArray(elem));
  }

  function readArrayName(elem) {
    // INT8 name, occasionally stored as small-format (≤4 chars).
    if (elem.miType !== 1 && elem.miType !== 2) {
      throw new Error(`expected array name (miINT8/UINT8), got miType ${elem.miType}`);
    }
    return new TextDecoder('ascii').decode(elem.payload).replace(/\0+$/, '');
  }

  // Decompresses a miCOMPRESSED element synchronously via Pako if
  // available, otherwise asynchronously via DecompressionStream.
  // Returns a Promise<Uint8Array>. Most modern EEGLAB exports are
  // uncompressed — this path only fires for older / explicit-deflate
  // saves, and we accept the async cost when it does.
  async function inflateZlib(payload) {
    if (typeof DecompressionStream === 'undefined') {
      throw new Error('miCOMPRESSED found but DecompressionStream is unavailable in this runtime');
    }
    const ds = new DecompressionStream('deflate');
    const writer = ds.writable.getWriter();
    writer.write(payload);
    writer.close();
    const reader = ds.readable.getReader();
    const chunks = [];
    let total = 0;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      chunks.push(value);
      total += value.length;
    }
    const out = new Uint8Array(total);
    let off = 0;
    for (const c of chunks) { out.set(c, off); off += c.length; }
    return out;
  }

  // Parses a single miMATRIX payload (already extracted from its tag).
  // Returns { name, class, dims, data, complex }.
  // Asynchronous because we may hit nested miCOMPRESSED elements.
  async function parseMatrix(payload) {
    const view = new DataView(payload.buffer, payload.byteOffset, payload.length);
    const subs = [];
    for (const elem of iterElements(view, 0, payload.length)) subs.push(elem);
    // Standard miMATRIX has 4 sub-elements: flags, dims, name, data.
    // Empty matrices (e.g. unset EEG.icaact, EEG.epoch when no ICA/epochs
    // ran) can be written with as few as 0 (completely empty payload)
    // through 2 (flags + dims). Observed on ds002181 where some nested
    // struct fields have 0 sub-elements. Be lenient: synthesize an
    // "empty placeholder" matrix that downstream consumers can no-op.
    if (subs.length === 0) {
      // Completely empty payload — return a null matrix that consumers
      // can ignore (it can't be EEG.data since that needs flags/dims).
      return { name: '', class: 'empty', dims: [0, 0], data: null };
    }
    if (subs.length < 3) {
      // Have flags ± dims but no name. Best-effort: synthesize empty
      // matrix of the declared class.
      const flagsOnly = readArrayFlags(subs[0]);
      const dimsOnly  = subs.length >= 2 ? readDims(subs[1]) : [0, 0];
      const infoEmpty = CLASS_INFO[flagsOnly.mxClass];
      const className = infoEmpty ? infoEmpty.name :
        (flagsOnly.mxClass === 2 ? 'struct' :
         flagsOnly.mxClass === 4 ? 'char' :
         flagsOnly.mxClass === 1 ? 'cell' : 'empty');
      const emptyData = infoEmpty ? new infoEmpty.ctor(0) :
                        flagsOnly.mxClass === 2 ? new Map() :
                        flagsOnly.mxClass === 4 ? '' : null;
      return { name: '', class: className, dims: dimsOnly, data: emptyData };
    }
    const flags = readArrayFlags(subs[0]);
    const dims  = readDims(subs[1]);
    const name  = readArrayName(subs[2]);

    if (flags.complex) {
      throw new Error(`complex matrix '${name}' not supported`);
    }

    const info = CLASS_INFO[flags.mxClass];
    const isStruct = flags.mxClass === 2;
    const isChar   = flags.mxClass === 4;
    const isCell   = flags.mxClass === 1;
    const isSparse = flags.mxClass === 5;
    const isObject = flags.mxClass === 3;

    if (isCell || isSparse || isObject) {
      // Returned with data:null so the caller can skip it gracefully
      // — we don't need cells / sparse / objects for inline EEGLAB.
      return { name, class: isCell ? 'cell' : isSparse ? 'sparse' : 'object', dims, data: null };
    }

    if (isChar) {
      // MAT v5 CHAR storage encoding is conveyed in subs[3].miType:
      //   miUINT8 (2)  / miINT8 (1)   → ASCII / Latin-1, 1 byte per char
      //   miUTF8 (16)                  → UTF-8 variable-width
      //   miUINT16 (4) / miINT16 (3)  → 2 bytes per char, codepoints
      //   miUTF16 (17)                 → UTF-16LE
      //   miUTF32 (18)                 → UTF-32LE
      // Older MATLAB writes ASCII-only filenames as miUINT16 with high
      // byte = 0 (observed on ds003078 EEG.data = "S_1_cond1_run1.fdt").
      // Decoding as ASCII produces "S\0_\01\0..." — strip nulls
      // afterwards so the filename is usable.
      // Empty CHAR (e.g. EEG.comments = '' on freshly created datasets).
      if (subs.length < 4 || !subs[3].payload) {
        return { name, class: 'char', dims, data: '' };
      }
      const miType = subs[3].miType;
      const payload = subs[3].payload;
      let text;
      if (miType === 17 /* miUTF16 */ || miType === 4 /* miUINT16 */ || miType === 3 /* miINT16 */) {
        // 2 bytes per code unit, little-endian.
        try {
          text = new TextDecoder('utf-16le').decode(payload);
        } catch {
          text = '';
        }
      } else if (miType === 18 /* miUTF32 */) {
        try {
          text = new TextDecoder('utf-32le').decode(payload);
        } catch {
          text = '';
        }
      } else if (miType === 16 /* miUTF8 */) {
        text = new TextDecoder('utf-8').decode(payload);
      } else {
        // miUINT8 / miINT8 / unknown — assume ASCII / Latin-1.
        text = new TextDecoder('ascii').decode(payload);
      }
      // Strip embedded NULs anywhere (pad-only padding sometimes appears
      // inside as well as at the end on some old writers).
      text = text.replace(/\0+/g, '').replace(/\s+$/, '');
      return { name, class: 'char', dims, data: text };
    }

    if (isStruct) {
      // sub[3] = field name length (INT32, scalar)
      // sub[4] = field names (INT8, nfields × fieldNameLen)
      // sub[5..] = each field as a nested miMATRIX
      //
      // Empty struct (no fields, dims=[0,0]) — some writers omit subs[3+]
      // entirely. Return an empty Map so consumers can no-op cleanly.
      if (subs.length < 5) {
        return { name, class: 'struct', dims, data: new Map() };
      }
      const fieldNameLen = readDims(subs[3])[0];
      const namesBlob = new TextDecoder('ascii').decode(subs[4].payload);
      const nfields = subs[4].payload.length / fieldNameLen;
      const fieldNames = [];
      for (let i = 0; i < nfields; i++) {
        fieldNames.push(namesBlob.slice(i * fieldNameLen, (i + 1) * fieldNameLen).replace(/\0+$/, ''));
      }
      // For struct arrays of size > 1 we'd see (nfields × prod(dims))
      // miMATRIX subs; we collapse to the first element only — EEGLAB
      // top-level structs are scalar (1×1) anyway.
      const fields = new Map();
      // Synthesize an empty placeholder for missing nested fields. Some
      // writers truncate struct subelement lists when trailing fields
      // are all-empty (observed on ds005185, ds005106 'labels' field of
      // EEG.chanlocs; ds005876 'theta' field). The reader doesn't need
      // these unused fields — the audit's earlier hard-reject was too
      // strict. We log so callers can diagnose if it matters.
      const emptyPlaceholder = () => ({ name: '', class: 'unknown', dims: [0, 0], data: [] });
      let synthesizedCount = 0;
      const synthesizedNames = [];
      for (let i = 0; i < nfields; i++) {
        const subMatrix = subs[5 + i];
        if (!subMatrix) {
          synthesizedCount++;
          synthesizedNames.push(fieldNames[i]);
          fields.set(fieldNames[i], emptyPlaceholder());
          continue;
        }
        // Each field is itself a nested matrix. Non-miMATRIX is malformed
        // — same lenient strategy: synthesize empty rather than throw.
        if (subMatrix.miType !== 14) {
          synthesizedCount++;
          synthesizedNames.push(fieldNames[i]);
          fields.set(fieldNames[i], emptyPlaceholder());
          continue;
        }
        fields.set(fieldNames[i], await parseMatrix(subMatrix.payload));
      }
      if (synthesizedCount > 0) {
        console.warn(
          `MAT v5: struct '${name}' has ${synthesizedCount}/${nfields} ` +
          `missing/malformed nested subelements — synthesized as empty: ` +
          `[${synthesizedNames.slice(0, 6).join(', ')}${synthesizedNames.length > 6 ? ', ...' : ''}].`
        );
      }
      return { name, class: 'struct', dims, data: fields };
    }

    if (!info) {
      throw new Error(`'${name}' has unsupported mxClass ${flags.mxClass}`);
    }

    // Empty numeric matrix (e.g. EEG.icaact = [] on datasets without ICA):
    // some writers omit subs[3] when prod(dims) === 0. Return a 0-length
    // typed array of the right class so downstream consumers can no-op.
    const expectedLen = dims.reduce((a, b) => a * b, 1);
    if (subs.length < 4 || !subs[3].payload) {
      if (expectedLen === 0) {
        // info.ctor is the constructor (Float32Array, Int16Array, etc).
        return { name, class: info.name, dims, data: new info.ctor(0) };
      }
      throw new Error(`'${name}': missing real-data sub-element but dims=${JSON.stringify(dims)} expects ${expectedLen} elements`);
    }
    const realData = subs[3];
    const typed = payloadAsTypedArray(realData);
    if (typed.length !== expectedLen) {
      throw new Error(`'${name}' data length ${typed.length} != prod(dims) ${expectedLen}`);
    }
    return { name, class: info.name, dims, data: typed };
  }

  // Sniffs the on-disk MAT version from the header. Returns one of
  // 'v5' (0x0100, the format this module reads), 'v7.3' (0x0200,
  // HDF5-backed — out of scope), or 'unknown'. The ASCII description
  // at offset 0 corroborates the version uint16: v5 files start with
  // 'MATLAB 5.0 MAT-file', v7.3 with 'MATLAB 7.3 MAT-file'.
  api.detectMatVersion = function (buffer) {
    const u8 = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
    if (u8.length < 128) return 'unknown';
    const dv = new DataView(u8.buffer, u8.byteOffset, 128);
    const version = dv.getUint16(124, true);
    if (version === 0x0100) return 'v5';
    if (version === 0x0200) return 'v7.3';
    return 'unknown';
  };

  // Top-level entry point. Accepts an ArrayBuffer or Uint8Array,
  // returns a Promise<Map<string, Var>> of named variables.
  api.parse = async function (buffer) {
    const u8 = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
    if (u8.length < 128) {
      throw new Error(`MAT file too short for header: ${u8.length}B`);
    }
    const header = new DataView(u8.buffer, u8.byteOffset, 128);
    // MAT v7.3 detection FIRST: byte 124-125 is the version field.
    // 0x0100 = v5 (this module). 0x0200 = v7.3 (HDF5-backed) — that
    // path is handled by formats/_mat73.js; the EEGLAB reader
    // dispatches the right parser via MatV5.detectMatVersion(). If
    // a caller bypasses that dispatch and feeds a v7.3 buffer in
    // directly, surface a precise hint about which module owns the
    // format rather than the legacy "only v5" message.
    const version = header.getUint16(124, true);
    if (version === 0x0200) {
      throw new Error(
        'MAT v7.3 (HDF5) detected. This file is parsed by Mat73 (HDF5 reader), ' +
        'not MatV5. Either call Mat73.parse(buffer) directly, or go through ' +
        'EEGLABReader.open() which dispatches on MatV5.detectMatVersion().',
      );
    }
    // Endian indicator: 0x4D49 ('IM' bytes 'M','I' → little-endian on disk).
    const endian = header.getUint16(126, true);
    if (endian !== 0x4d49) {
      throw new Error(`MAT file is not little-endian (endian indicator = 0x${endian.toString(16)})`);
    }
    if (version !== 0x0100) {
      throw new Error(`unsupported MAT version 0x${version.toString(16)} (only v5/v6 = 0x0100 supported)`);
    }

    const view = new DataView(u8.buffer, u8.byteOffset, u8.length);
    const vars = new Map();
    for (const elem of iterElements(view, 128, u8.length)) {
      let payload = elem.payload;
      let miType = elem.miType;
      if (miType === 15) {  // miCOMPRESSED
        payload = await inflateZlib(payload);
        // After inflation, the inner data is a single tagged element
        // (typically a miMATRIX). Re-walk so the inner type is honoured.
        const innerView = new DataView(payload.buffer, payload.byteOffset, payload.length);
        const inner = iterElements(innerView, 0, payload.length).next().value;
        if (!inner) continue;
        miType = inner.miType;
        payload = inner.payload;
      }
      if (miType !== 14) continue;  // skip non-matrix top-level elements
      const v = await parseMatrix(payload);
      if (v.name) vars.set(v.name, v);
    }
    return vars;
  };

  // Unwraps an EEGLAB inline-data .set into the fields the reader
  // needs. EEGLAB writes either a single struct named "EEG" or one
  // top-level variable per field — handle both.
  // Returns { data, srate, nbchan, pnts, trials } where data is a
  // typed array in column-major (channels, samples [, trials]) layout.
  // Throws if any required field is missing or the wrong shape.
  api.extractEegInline = function (vars) {
    // Helper: pick a field by name, falling back to EEG.<name> when
    // the file uses the wrapped layout.
    const eegStruct = vars.get('EEG');
    function field(name) {
      if (vars.has(name)) return vars.get(name);
      if (eegStruct && eegStruct.class === 'struct' && eegStruct.data.has(name)) {
        return eegStruct.data.get(name);
      }
      return null;
    }
    function scalar(name) {
      const v = field(name);
      if (!v || !v.data || !v.data.length) return null;
      return Number(v.data[0]);
    }

    const data = field('data');
    if (!data) {
      throw new Error('EEG inline-data .set missing `data` (or EEG.data) variable');
    }
    if (data.class === 'char') {
      // EEGLAB split-file convention: EEG.data is a CHAR string containing
      // the sibling .fdt filename instead of inline numeric data.
      // For v5, our parser already decoded the CHAR payload to a string
      // at parseTopElement (data.data is a JS string, not a typed array).
      // Surface as a sentinel so eeglab.js can route to the .fdt sibling.
      // Mirrors the v7.3 CHAR-sidecar handling in _mat73.js.
      const filename = String(data.data || '').replace(/\0+$/, '').trim();
      const err = new Error(
        `EEG.data is a CHAR sidecar filename ("${filename}"); ` +
        `the .set references an external .fdt rather than carrying inline samples`,
      );
      err.code = 'EEGLAB_DATA_IS_CHAR';
      err.fdtFilename = filename;
      throw err;
    }
    if (data.class !== 'single' && data.class !== 'double' && data.class !== 'int16' && data.class !== 'int32') {
      throw new Error(`EEG.data has unsupported numeric class '${data.class}' (need single/double/int16/int32)`);
    }
    if (data.dims.length < 2 || data.dims.length > 3) {
      throw new Error(`EEG.data must be 2D or 3D, got dims=[${data.dims.join(',')}]`);
    }
    const nbchan = scalar('nbchan') ?? data.dims[0];
    const pnts   = scalar('pnts')   ?? data.dims[1];
    const trials = scalar('trials') ?? (data.dims[2] || 1);
    const srate  = scalar('srate');
    if (!srate || !isFinite(srate) || srate <= 0) {
      throw new Error(`EEG.srate missing or invalid (got ${srate})`);
    }
    if (data.dims[0] !== nbchan) {
      throw new Error(`EEG.data dims[0]=${data.dims[0]} disagrees with nbchan=${nbchan}`);
    }
    return { data: data.data, srate, nbchan, pnts, trials, dataClass: data.class };
  };

  // Scan top-level MAT v5 elements WITHOUT materializing payloads.
  // For each miMATRIX element, peek at the first 4 sub-elements
  // (flags, dims, name, real-data tag) so callers know where the
  // real-data bytes live without reading them. This enables the
  // streaming inline-EEGLAB path: scan the first ~16 MB to find
  // EEG.data's payload offset, then range-fetch column slices on
  // demand.
  //
  // For miCOMPRESSED elements we surface the compressed envelope
  // metadata but DO NOT decompress; callers fall back to the full
  // parse() path when any compressed element is encountered.
  //
  // Returns an array of:
  //   {
  //     miType,                 // 14 (miMATRIX), 15 (miCOMPRESSED), or rare other
  //     elementOffset,          // absolute byte offset of element header
  //     payloadOffset,          // absolute byte offset of element payload
  //     payloadBytes,           // payload length before padding
  //     mxClass | null,
  //     dims    | null,         // number[]
  //     name    | null,         // string
  //     dataSubOffset  | null,  // absolute byte offset of real-data payload
  //     dataSubBytes   | null,
  //     dataSubMiType  | null,
  //   }
  api.scanElements = function (buffer) {
    const u8 = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
    if (u8.length < 128) {
      throw new Error(`scanElements: MAT file too short for header: ${u8.length}B`);
    }
    const header = new DataView(u8.buffer, u8.byteOffset, 128);
    const version = header.getUint16(124, true);
    if (version === 0x0200) {
      throw new Error('scanElements: MAT v7.3 (HDF5) — call Mat73.parse, not scanElements');
    }
    if (version !== 0x0100) {
      throw new Error(`scanElements: unsupported MAT version 0x${version.toString(16)}`);
    }

    const baseOff = u8.byteOffset;
    const fullView = new DataView(u8.buffer, baseOff, u8.length);
    const results = [];

    for (const elem of iterElements(fullView, 128, u8.length, { allowTruncated: true })) {
      // Reconstruct elementOffset from payloadOffset: small format puts
      // payload at off+4, long format at off+8. The tag header itself
      // can be re-read to determine which (small if upper-16 of the
      // first uint32 is in [1, 4]).
      const tagWord = fullView.getUint32(elem.payloadOffset - 4, true);
      const upper16 = (tagWord >>> 16) & 0xffff;
      const small = upper16 >= 1 && upper16 <= 4;
      const elementLocalOffset = small ? (elem.payloadOffset - 4) : (elem.payloadOffset - 8);

      const meta = {
        miType:         elem.miType,
        elementOffset:  baseOff + elementLocalOffset,
        payloadOffset:  baseOff + elem.payloadOffset,
        // When the top-level element was truncated by the probe buffer,
        // `payload.length` is the available slice — but the on-disk
        // element actually declares `declaredBytes`. Surface both so
        // callers can pick the right value.
        payloadBytes:   elem.truncated ? elem.declaredBytes : elem.payload.length,
        payloadTruncated: !!elem.truncated,
        mxClass:        null,
        dims:           null,
        name:           null,
        dataSubOffset:  null,
        dataSubBytes:   null,
        dataSubMiType:  null,
        dataSubTruncated: false,
      };

      if (elem.miType === 14) {
        // Peek at sub-elements 0..3 (flags, dims, name, realData).
        const subView = new DataView(elem.payload.buffer, elem.payload.byteOffset, elem.payload.length);
        const subs = [];
        try {
          for (const s of iterElements(subView, 0, elem.payload.length, { allowTruncated: true })) {
            subs.push(s);
            if (subs.length === 4) break;
          }
        } catch {
          // Sub-element walk failed → leave fields null; caller will
          // see {miType:14, mxClass:null} and treat as opaque.
        }
        if (subs.length >= 1 && subs[0].miType === 6) {
          // Read mxClass from low byte of first uint32.
          const sub0DV = new DataView(subs[0].payload.buffer, subs[0].payload.byteOffset, subs[0].payload.length);
          const word0 = sub0DV.getUint32(0, true);
          meta.mxClass = word0 & 0xff;
        }
        if (subs.length >= 2 && subs[1].miType === 5) {
          meta.dims = Array.from(new Int32Array(
            subs[1].payload.buffer.slice(
              subs[1].payload.byteOffset,
              subs[1].payload.byteOffset + subs[1].payload.length,
            ),
          ));
        }
        if (subs.length >= 3 && (subs[2].miType === 1 || subs[2].miType === 2)) {
          meta.name = new TextDecoder('ascii').decode(subs[2].payload).replace(/\0+$/, '');
        }
        if (subs.length >= 4) {
          // The real-data sub-element's payload starts at
          // subs[3].payloadOffset inside the parent matrix payload.
          // The absolute file offset is therefore:
          //   baseOff + elem.payloadOffset + subs[3].payloadOffset
          meta.dataSubOffset  = baseOff + elem.payloadOffset + subs[3].payloadOffset;
          // When the real-data sub-element is truncated (the probe buffer
          // cut off mid-payload), iterElements yields with `truncated:true`
          // and exposes the on-disk-declared length via `declaredBytes`.
          // Callers that want to range-fetch the full data MUST see the
          // declared size, not the truncated slice length.
          meta.dataSubBytes   = subs[3].truncated
            ? subs[3].declaredBytes
            : subs[3].payload.length;
          meta.dataSubMiType  = subs[3].miType;
          meta.dataSubTruncated = !!subs[3].truncated;
        }
      }
      results.push(meta);
    }
    return results;
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.MatV5 = api;
})();
