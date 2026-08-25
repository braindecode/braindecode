/* ============================================================
   formats/_mat73.js — minimal MATLAB v7.3 (HDF5) reader.

   Built for the same job _matv5.js handles for older .set files:
   extract the EEG struct fields an EEGLAB inline-data .set carries
   (`data`, `srate`, `nbchan`, `pnts`, `trials`) so the viewer can
   serve windows from the in-memory column-major typed array.

   Why a separate module instead of extending _matv5.js: MAT v7.3
   replaces the entire on-disk format with HDF5. The legacy 128-byte
   MAT header survives as a stub at the start of the file, but the
   real payload begins at byte 512 and is a stock HDF5 file.

   File layout:
     bytes   0..511  legacy MAT v5 stub header
                       0..115   description text
                       124..125 version uint16 = 0x0200
                       126..127 endian uint16  = 0x4D49 ('IM')
     bytes 512..N    standard HDF5 file (magic 0x894844460D0A1A0A)

   We delegate HDF5 walking to jsfive (vendored as `_jsfive.js`),
   then patch its DataObjects prototype at first use to support
   *compact* storage layout (layout_class == 0). MATLAB packs scalars
   like `EEG.srate` / `EEG.nbchan` using compact storage; vanilla
   jsfive throws on those. The patch is ~30 LOC, scoped to the one
   prototype method, and falls through to the original for the other
   layout classes (contiguous + chunked) which jsfive already covers.

   EEGLAB HDF5 layout we read:
     /EEG/srate    1x1 double, compact storage
     /EEG/nbchan   1x1 double, compact storage
     /EEG/pnts     1x1 double, compact storage
     /EEG/trials   1x1 double, compact storage
     /EEG/data     either:
                     (a) Char string (uint16 row vector) = sidecar
                         .fdt filename → inline path doesn't apply,
                         caller falls back to split .set+.fdt flow.
                     (b) Numeric (single|double) dataset shape
                         [pnts, nbchan] HDF5 row-major == [nbchan,
                         pnts] MATLAB column-major. Sequential bytes
                         are sample-interleaved (all channels at s=0,
                         then all channels at s=1, …) — same layout
                         as the .fdt blob, so existing sliceColumn-
                         Major in eeglab.js works unchanged.

   Returned shape from `parse()`: a Map<string, Var> matching the
   subset of _matv5.js's surface that `MatV5.extractEegInline()`
   consumes. We piggyback on that helper to keep the dispatch in
   formats/eeglab.js flat: the file-format check is the only branch.

   What we DON'T handle (deliberately):
     - Cell arrays (EEGLAB's `EEG.chanlocs` struct array is a cell
       of references in v7.3; per-channel labels stay Ch1..ChN from
       the existing fallback path).
     - Sparse, complex, object references in numeric data, VLEN data.
     - Compressed (deflate-filtered) HDF5 datasets — EEGLAB doesn't
       emit those for `data` in practice; jsfive's chunked path
       handles them transparently if it ever needs to.
   ============================================================ */
(function () {
  'use strict';

  const api = {};

  // jsfive resolves differently in Node (CJS via npm) and the
  // browser/worker (vendored IIFE attaches globalThis.hdf5). Tests
  // run under Node and pull in the CJS module; the deployed bundle
  // loads _jsfive.js first, which sets globalThis.hdf5.
  function getJsfive() {
    if (typeof globalThis !== 'undefined' && globalThis.hdf5) return globalThis.hdf5;
    if (typeof require !== 'undefined') {
      try { return require('jsfive'); } catch (_) { /* fall through */ }
    }
    throw new Error(
      'jsfive not available: include formats/_jsfive.js before ' +
      'formats/_mat73.js in the browser, or `npm install jsfive` ' +
      'for the Node tests.'
    );
  }

  // Has the compact-storage patch been installed on jsfive's
  // DataObjects prototype yet? Idempotent: we re-run on every
  // module load (cheap) and short-circuit if the prototype already
  // carries our marker. Multiple parse() calls share the same
  // patched prototype across the lifetime of the worker.
  let patched = false;
  const PATCH_MARKER = '__mat73_compact_patched';

  // Locate jsfive's DataObjects prototype by opening a tiny HDF5
  // file we ship alongside the reader — except we don't ship one,
  // so instead we reach it through a Dataset instance on the user's
  // file. The patch is therefore lazy: applyCompactStoragePatch()
  // is called every parse() and bails fast when already installed.
  function applyCompactStoragePatch(file) {
    if (patched) return;
    // Walk to any dataset under the root group so we can reach the
    // _dataobjects prototype. The Mat v7.3 wrapper guarantees at
    // least `/EEG/srate` exists for any EEGLAB export; we don't
    // *read* it (that would trip the compact check we're about
    // to install), only use it to grab the prototype.
    let probe = null;
    function findDataset(group, depth = 0) {
      if (probe || depth > 3) return;
      for (const k of group.keys || []) {
        if (probe) return;
        let child;
        try { child = group.get(k); } catch (_) { continue; }
        if (child && child._dataobjects) { probe = child; return; }
        if (child && child.keys) findDataset(child, depth + 1);
      }
    }
    findDataset(file);
    if (!probe) {
      throw new Error('MAT v7.3: no dataset found under root to patch jsfive prototype');
    }
    const proto = Object.getPrototypeOf(probe._dataobjects);
    if (proto[PATCH_MARKER]) { patched = true; return; }

    const original_get_data = proto.get_data;
    const original_get_props = proto._get_data_message_properties;

    // Override message-property parsing to accept layout_class 0
    // (compact). Vanilla jsfive `assert(layout_class in {1, 2})` in
    // v1/v2 layout, and silently passes layout_class==0 in v3/v4 to
    // get_data which then throws "Compact storage not implemented".
    // We surface a property_offset that points at the size+data
    // payload regardless of layout class, mirroring the spec.
    proto._get_data_message_properties = function (msg_offset) {
      const fh = this.fh;
      const dv = new DataView(fh);
      const version = dv.getUint8(msg_offset);
      let dims, layout_class, property_offset;
      if (version === 1 || version === 2) {
        // v1/v2: <BBB BI ...> = ver, dims, class, reserved-byte, reserved-int
        dims = dv.getUint8(msg_offset + 1);
        layout_class = dv.getUint8(msg_offset + 2);
        property_offset = msg_offset + 3 + 1 + 4;
        return [version, dims, layout_class, property_offset];
      }
      if (version === 3 || version === 4) {
        // v3/v4: <BB ...> = ver, class
        layout_class = dv.getUint8(msg_offset + 1);
        property_offset = msg_offset + 2;
        return [version, undefined, layout_class, property_offset];
      }
      // Fall through to the original for unknown versions so we
      // inherit jsfive's error message verbatim.
      return original_get_props.call(this, msg_offset);
    };

    proto.get_data = function () {
      // DATA_STORAGE_MSG_TYPE = 8 (per HDF5 spec). jsfive's source
      // hardcodes the same constant — no public export to pull from.
      const DATA_STORAGE_MSG_TYPE = 8;
      const msg = this.find_msg_type(DATA_STORAGE_MSG_TYPE)[0];
      const msg_offset = msg.get('offset_to_message');
      const [version, _dims, layout_class, property_offset] =
        this._get_data_message_properties(msg_offset);

      if (layout_class === 0) {
        // Compact storage: the dataset's entire payload is inline
        // in the layout message itself, immediately after a uint16
        // size prefix (v3/v4 only — v1/v2 compact is technically
        // legal but EEGLAB never emits it; surface a clear error).
        if (version !== 3 && version !== 4) {
          throw new Error('MAT v7.3 compact-storage layout v' + version + ' not supported');
        }
        const fh = this.fh;
        const dv = new DataView(fh);
        const size = dv.getUint16(property_offset, true);
        const data_offset = property_offset + 2;
        const dtype = this.dtype;
        // dtype strings look like '<f8', '<f4', '<i4', '<u2', etc.
        // jsfive returns them verbatim. Parse into a getter +
        // element-size pair so we can fill a JS array directly.
        if (typeof dtype !== 'string' || !/[<>=!@\|]?[iuf](\d+)$/.test(dtype) && !/[<>=!@\|]?u(\d+)$/.test(dtype)) {
          // For non-numeric (REFERENCE, STRING, compound) compact
          // storage we'd need a parallel path. None are observed in
          // EEGLAB's compact-stored scalars in the wild — fail loud.
          throw new Error('MAT v7.3 compact storage: unsupported dtype ' + JSON.stringify(dtype));
        }
        const m = dtype.match(/[<>=!@\|]?([iuf])(\d+)/);
        const fstr = m[1];
        const nbytes = parseInt(m[2], 10);
        const big_endian = /^[>=!]/.test(dtype);
        let getter;
        if (fstr === 'i') getter = 'getInt' + (nbytes * 8);
        else if (fstr === 'u') getter = 'getUint' + (nbytes * 8);
        else if (fstr === 'f') getter = 'getFloat' + (nbytes * 8);
        else throw new Error('MAT v7.3 compact storage: unhandled fstr ' + fstr);
        const count = size / nbytes;
        if (!Number.isInteger(count)) {
          throw new Error('MAT v7.3 compact storage: size ' + size + ' not a multiple of ' + nbytes);
        }
        const out = new Array(count);
        for (let i = 0; i < count; i++) {
          out[i] = dv[getter](data_offset + i * nbytes, !big_endian);
        }
        return out;
      }
      return original_get_data.call(this);
    };

    proto[PATCH_MARKER] = true;
    patched = true;
  }

  // True iff the buffer looks like a MAT v7.3 (HDF5) file: legacy
  // MAT header at byte 0 with version uint16 = 0x0200, AND HDF5
  // magic bytes at offset 512. A standalone HDF5 file (no MAT stub)
  // is *not* recognised here — that's not a MATLAB save and the
  // EEGLAB path wouldn't know what to do with it.
  api.isHdf5 = function (buffer) {
    const u8 = buffer instanceof Uint8Array
      ? buffer
      : new Uint8Array(buffer);
    if (u8.length < 520) return false;
    const dv = new DataView(u8.buffer, u8.byteOffset, 520);
    const version = dv.getUint16(124, true);
    if (version !== 0x0200) return false;
    // HDF5 superblock signature at offset 512: \x89 H D F \r \n \x1a \n
    return u8[512] === 0x89 && u8[513] === 0x48 && u8[514] === 0x44 &&
           u8[515] === 0x46 && u8[516] === 0x0d && u8[517] === 0x0a &&
           u8[518] === 0x1a && u8[519] === 0x0a;
  };

  // Convert a jsfive dtype string ('<f4', '<f8', '<i2', '<u2', …)
  // to the field names extractEegInline() consumes (matching _matv5
  // 's class names). We only need the four numeric kinds EEGLAB
  // actually writes for `EEG.data`; everything else returns null
  // and the caller raises a clearer error.
  function dtypeToClass(dtype) {
    if (dtype === '<f4' || dtype === '|f4' || dtype === 'float32') return 'single';
    if (dtype === '<f8' || dtype === '|f8' || dtype === 'float64') return 'double';
    if (dtype === '<i2' || dtype === '|i2') return 'int16';
    if (dtype === '<i4' || dtype === '|i4') return 'int32';
    return null;
  }

  // Take a jsfive plain-JS Array (returned by Dataset.value for non-
  // typed reads) and promote it to the canonical typed-array of the
  // right class. Callers downstream (eeglab.js sliceColumnMajor) only
  // index it, so any typed array works — picking the matching one
  // saves memory and skips an implicit float promotion.
  function toTypedArray(values, klass) {
    if (klass === 'single') return Float32Array.from(values);
    if (klass === 'double') return Float64Array.from(values);
    if (klass === 'int16')  return Int16Array.from(values);
    if (klass === 'int32')  return Int32Array.from(values);
    return Float32Array.from(values);
  }

  // Read a scalar EEG.field by name. Returns null if the field is
  // missing or empty (matching `MatV5.extractEegInline`'s `scalar`
  // helper, which uses it as a guard).
  function readScalar(eegGroup, name) {
    if (!eegGroup.keys.includes(name)) return null;
    const ds = eegGroup.get(name);
    if (!ds || !ds.shape) return null;
    const v = ds.value;
    if (!v || !v.length) return null;
    return Number(v[0]);
  }

  // Pull a CHAR field (HDF5 stores MATLAB strings as uint16 vectors
  // with MATLAB_class='char') out and decode it as UTF-16. Used to
  // detect when EEG.data is a sidecar filename (the .set+.fdt case)
  // rather than inline numeric data.
  function readCharField(eegGroup, name) {
    if (!eegGroup.keys.includes(name)) return null;
    const ds = eegGroup.get(name);
    if (!ds || !ds.shape) return null;
    const attrs = ds._dataobjects ? ds._dataobjects.get_attributes() : {};
    if (attrs.MATLAB_class !== 'char') return null;
    const codes = ds.value;
    if (!codes || !codes.length) return '';
    return String.fromCharCode(...codes);
  }

  // Parse a MAT v7.3 buffer and return a Map<string, Var> with the
  // same shape MatV5.parse() returns for the subset of fields
  // extractEegInline() consumes. We synthesise both the top-level
  // EEG struct entry AND the un-prefixed scalars (srate, nbchan,
  // pnts, trials) so extractEegInline's "wrapped vs flat" branch
  // works without any changes there.
  //
  // Throws if the file isn't a MAT v7.3 EEGLAB export or if
  // `/EEG/data` resolves to a sidecar filename string (caller is
  // expected to detect that path and switch to the split-file flow
  // before invoking parse — but if it doesn't, we surface a clear
  // error rather than returning garbage).
  api.parse = async function (buffer) {
    const u8 = buffer instanceof Uint8Array
      ? buffer
      : new Uint8Array(buffer);
    if (!api.isHdf5(u8)) {
      throw new Error('MAT v7.3 parse: buffer is not MAT v7.3 (HDF5)');
    }
    const hdf5Bytes = u8.buffer.slice(u8.byteOffset + 512, u8.byteOffset + u8.byteLength);
    const jsfive = getJsfive();
    const file = new jsfive.File(hdf5Bytes);
    applyCompactStoragePatch(file);

    // Two valid layouts at the root:
    //   1. /EEG group wrapping the fields (most common — what EEGLAB
    //      writes by default).
    //   2. Fields flat at root: /data, /srate, /nbchan, /pnts, /trials, …
    //      (observed on ds004105 / ds004118 / ds004121 / ds004122 /
    //      ds004123). MATLAB's `save(..., '-v7.3', '-struct', 'EEG')`
    //      with the `-struct` flag drops the wrapping group.
    // We detect by looking for the canonical EEG fields at root before
    // falling back to the wrapper path.
    let eeg;
    const flatRoot = ['data', 'srate', 'nbchan'].every(k => file.keys.includes(k));
    if (flatRoot) {
      eeg = file;  // jsfive root behaves like a group for these reads
    } else if (file.keys.includes('EEG')) {
      eeg = file.get('EEG');
      if (!eeg.keys || !eeg.keys.length) {
        throw new Error('MAT v7.3 parse: /EEG group is empty');
      }
    } else {
      throw new Error(
        'MAT v7.3 parse: no /EEG group at root and no flat /data,/srate,/nbchan layout ' +
        '(not an EEGLAB v7.3 save?)',
      );
    }

    const srate  = readScalar(eeg, 'srate');
    const nbchan = readScalar(eeg, 'nbchan');
    const pnts   = readScalar(eeg, 'pnts');
    const trials = readScalar(eeg, 'trials');

    // EEG.data inline-vs-sidecar branch. If MATLAB_class == 'char',
    // we're looking at a split .set+.fdt save and there's no inline
    // data here. Caller (eeglab.js) is already structured to take
    // the .fdt path when the sibling URL probe succeeds; we surface
    // a precise message so a misrouted caller fails clearly.
    const dataDs = eeg.keys.includes('data') ? eeg.get('data') : null;
    if (!dataDs) {
      throw new Error('MAT v7.3 parse: /EEG/data missing');
    }
    const dataAttrs = dataDs._dataobjects ? dataDs._dataobjects.get_attributes() : {};
    if (dataAttrs.MATLAB_class === 'char') {
      const fname = readCharField(eeg, 'data') || '';
      throw new Error(
        'MAT v7.3 parse: /EEG/data is a CHAR sidecar filename (' +
        JSON.stringify(fname) + '), not inline numeric data. ' +
        'This .set expects a sibling .fdt file alongside it.'
      );
    }
    const klass = dtypeToClass(dataDs.dtype);
    if (!klass) {
      throw new Error(
        'MAT v7.3 parse: /EEG/data has unsupported dtype ' +
        JSON.stringify(dataDs.dtype) +
        ' (need single/double/int16/int32)'
      );
    }

    // jsfive Dataset.value returns a plain JS Array for non-chunked
    // numeric storage and a typed array for chunked. Promote to the
    // canonical typed array up-front so downstream consumers (slice-
    // ColumnMajor in eeglab.js) get O(1) length + numeric indexing
    // without surprise.
    const rawValues = dataDs.value;
    const flat = toTypedArray(rawValues, klass);

    // HDF5 stores MATLAB matrices in row-major with dims reversed
    // from the MATLAB declaration. EEG.data is (nbchan, pnts[, trials])
    // in MATLAB, so HDF5 sees (trials, pnts, nbchan) or (pnts, nbchan).
    // The bytes on disk are the same column-major sequence MATLAB
    // wrote — which is what _matv5's sliceColumnMajor expects. So we
    // surface dims in MATLAB order to keep extractEegInline's shape
    // check happy.
    const hdfShape = dataDs.shape;
    let matlabDims;
    if (hdfShape.length === 2) {
      // HDF5 (pnts, nbchan) → MATLAB (nbchan, pnts)
      matlabDims = [hdfShape[1], hdfShape[0]];
    } else if (hdfShape.length === 3) {
      // HDF5 (trials, pnts, nbchan) → MATLAB (nbchan, pnts, trials)
      matlabDims = [hdfShape[2], hdfShape[1], hdfShape[0]];
    } else {
      throw new Error(
        'MAT v7.3 parse: /EEG/data must be 2D or 3D, got HDF5 shape [' +
        hdfShape.join(',') + ']'
      );
    }

    // Build the same { class, dims, data, name } shape MatV5.parse
    // returns, so MatV5.extractEegInline() can walk it without
    // knowing which reader produced it. The struct entry carries
    // the per-field map for the wrapped path; we also drop the
    // flat duplicates at the top level for the unwrapped path. Both
    // resolve to the same underlying Var objects to avoid double
    // memory for the data array.
    const mkScalar = (name, val) => val == null ? null : ({
      name,
      class: 'double',
      dims: [1, 1],
      data: new Float64Array([val]),
    });
    const dataVar = {
      name: 'data',
      class: klass,
      dims: matlabDims,
      data: flat,
    };

    const eegFields = new Map();
    eegFields.set('data', dataVar);
    if (srate  != null) eegFields.set('srate',  mkScalar('srate',  srate));
    if (nbchan != null) eegFields.set('nbchan', mkScalar('nbchan', nbchan));
    if (pnts   != null) eegFields.set('pnts',   mkScalar('pnts',   pnts));
    if (trials != null) eegFields.set('trials', mkScalar('trials', trials));

    const vars = new Map();
    // Wrapped path: extractEegInline first looks at vars.get('EEG').
    vars.set('EEG', {
      name: 'EEG',
      class: 'struct',
      dims: [1, 1],
      data: eegFields,
    });
    // Unwrapped path: if a caller bypasses the struct, the flat
    // scalars are still here. Same Var instances — no extra memory.
    for (const [k, v] of eegFields) vars.set(k, v);

    return vars;
  };

  // Quick boolean test exposed for unit tests (avoids constructing
  // a full File just to check whether a buffer is HDF5-backed).
  // Documented as an underscore so it doesn't appear in the public
  // api-surface snapshot.
  api._dtypeToClass = dtypeToClass;

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.Mat73 = api;
})();
