/* ============================================================
   formats/_h5-stream.js — focused, range-read-friendly HDF5
   reader for NWB-style files. NOT a general-purpose HDF5 parser.

   Why this exists
   ---------------
   jsfive (vendored in formats/_jsfive.js) requires the *entire*
   HDF5 buffer in memory — its `fh.slice(a, b)` is synchronous and
   `new DataView64(fh, 0)` constructs a DataView directly over the
   whole ArrayBuffer. That's fine for the small SNIRF / MAT v7.3
   / sub-200 MB NWB path, but it forces a 200 MB cap on NWB and
   makes multi-GB DANDI files unloadable.

   This module gives the NWB reader two new capabilities:

     1. Head-only metadata probe. For a file of any size we fetch
        the FIRST `HEAD_BUFFER_BYTES` (default 16 MB) and hand
        that prefix to jsfive. Provided the file's metadata
        (superblock, root group, object headers, chunk B-tree)
        lives in those first bytes — which is the layout h5py /
        pynwb produce for every NWB writer we've inspected (they
        always write metadata first, then append chunk payloads)
        — jsfive's metadata walk succeeds. We never call .value
        on the big dataset, so we never trip jsfive's whole-file
        slice. We only extract:
          shape, dtype, chunks, _chunk_address, filter_pipeline,
          electrodes/{label,id} (small datasets), sampling rate.

     2. Windowed chunk fetcher. Given the metadata above we walk
        the V1 chunk B-tree ourselves (see h5BTreeV1WalkChunks),
        select only the chunks intersecting [s_start, s_end),
        range-fetch each chunk's exact byte range over HTTP, run
        them through the gzip / shuffle filter pipeline that the
        existing jsfive `Filters` map already implements, and
        copy the intersecting sample cells into the per-channel
        output buffers.

   Scope (intentionally narrow)
   ----------------------------
   - Storage layouts: contiguous (layout class 1) and chunked
     (layout class 2). Compact (class 0) is rare and trivially
     contiguous-like; we fall through to whole-file jsfive for it.
   - Datatypes: little-endian fixed-width integer / float (i1/u1
     /i2/u2/i4/u4/i8/u8/f4/f8). NWB ElectricalSeries.data is
     overwhelmingly f4 / i2 / i4. Bigger / fancier (compound,
     VLEN, REFERENCE) → fall back to whole-file jsfive.
   - Filters: GZIP_DEFLATE (id=1) only. SHUFFLE / FLETCH32 are
     present in jsfive's Filters map and we re-use them via the
     `globalThis.hdf5.Filters` import (no code duplication). If
     a filter id we don't recognise appears we throw cleanly.
   - Chunk B-tree: V1 only (the default h5py / pynwb writes).
     V2 chunk indexes (HDF5 1.10+ "single chunk", "implicit",
     "fixed array", "extensible array", "B-tree v2") will be a
     follow-up.

   Anything outside this scope causes us to throw a clearly-
   labelled "unsupported by streaming reader; download in full or
   open with pynwb" error rather than corrupting data.
   ============================================================ */
(function () {
  'use strict';

  // Head buffer size. NWB metadata for every fixture and DANDI
  // file inspected lives in the first 1-2 MB; 16 MB is generous
  // and still small enough to fetch in a single round-trip on a
  // typical OpenNeuro / DANDI connection.
  //
  // If a file's metadata lives past 16 MB (rare; only seen for
  // append-mode NWB writers that grow the global heap past the
  // boundary) we surface an error and the caller falls back to
  // whole-file jsfive (capped at 200 MB) or refuses outright.
  const HEAD_BUFFER_BYTES = 16 * 1024 * 1024;

  // HDF5 signature: byte 0..7 == 89 48 44 46 0d 0a 1a 0a.
  function isHdf5(u8) {
    if (!u8 || u8.length < 8) return false;
    return u8[0] === 0x89 && u8[1] === 0x48 && u8[2] === 0x44 &&
           u8[3] === 0x46 && u8[4] === 0x0d && u8[5] === 0x0a &&
           u8[6] === 0x1a && u8[7] === 0x0a;
  }

  // jsfive resolver — same pattern as formats/nwb.js.
  function getJsfive() {
    if (typeof globalThis !== 'undefined' && globalThis.hdf5) return globalThis.hdf5;
    if (typeof require !== 'undefined') {
      try { return require('jsfive'); } catch (_) { /* fall-through */ }
    }
    throw new Error('jsfive not available — see formats/nwb.js for the loader contract');
  }

  // Map an h5py-style numpy dtype string (e.g. "<f4", ">i2",
  // "<u4") to a typed-array constructor + per-cell byte width.
  // Returns null for anything outside our scope so the caller can
  // fall through to the whole-file jsfive path cleanly.
  function dtypeToTypedArray(dtype) {
    if (typeof dtype !== 'string') return null;
    const m = dtype.match(/^([<>=!@|])?([iuf])(\d+)$/);
    if (!m) return null;
    const endian = m[1] || '<';
    const cls = m[2];
    const bytes = parseInt(m[3], 10);
    if (endian === '>' || endian === '!') return null;  // big-endian fall-through
    if (cls === 'f') {
      if (bytes === 4) return { ctor: Float32Array, getter: 'getFloat32', bytes: 4 };
      if (bytes === 8) return { ctor: Float64Array, getter: 'getFloat64', bytes: 8 };
    } else if (cls === 'i') {
      if (bytes === 1) return { ctor: Int8Array,   getter: 'getInt8',   bytes: 1 };
      if (bytes === 2) return { ctor: Int16Array,  getter: 'getInt16',  bytes: 2 };
      if (bytes === 4) return { ctor: Int32Array,  getter: 'getInt32',  bytes: 4 };
    } else if (cls === 'u') {
      if (bytes === 1) return { ctor: Uint8Array,  getter: 'getUint8',  bytes: 1 };
      if (bytes === 2) return { ctor: Uint16Array, getter: 'getUint16', bytes: 2 };
      if (bytes === 4) return { ctor: Uint32Array, getter: 'getUint32', bytes: 4 };
    }
    return null;
  }

  // Read the DATA_STORAGE_MSG_TYPE (type 8) from a jsfive
  // DataObjects to extract the layout. Returns one of:
  //   { layoutClass: 1, dataAddress }                  (contiguous)
  //   { layoutClass: 2, chunkAddress, chunks, dtype, filterPipeline }
  //   null                                              (unsupported)
  //
  // jsfive's Dataset wraps `_dataobjects` which already exposes
  // chunks / filter_pipeline / _chunk_address / shape / dtype for
  // the chunked case via getters — we just bridge to a flat shape
  // for our windowed reader.
  function extractLayoutFromDataset(dataset) {
    const dobj = dataset._dataobjects;
    if (!dobj) return null;
    const STORAGE_MSG_TYPE = 8;
    const storageMsg = dobj.msgs.find((m) => m.get('type') === STORAGE_MSG_TYPE);
    if (!storageMsg) return null;
    const msgOffset = storageMsg.get('offset_to_message');
    const fh = dobj.fh;
    // fh is the (head-buffer) ArrayBuffer jsfive parsed; we read
    // the layout byte-for-byte exactly the way jsfive's
    // _get_data_message_properties does.
    const view = new DataView(fh);
    const version = view.getUint8(msgOffset);
    let layoutClass, propertyOffset;
    if (version === 1 || version === 2) {
      // BB header gives dims + layout_class, then BI reserved, then data.
      layoutClass = view.getUint8(msgOffset + 2);
      propertyOffset = msgOffset + 3 + 5;  // BBB + BI
    } else if (version === 3 || version === 4) {
      layoutClass = view.getUint8(msgOffset + 1);
      propertyOffset = msgOffset + 2;  // BB
    } else {
      return null;
    }
    if (layoutClass === 1) {
      // Contiguous: <Q data_offset, <Q size (size is informational;
      // we compute it from shape * dtype.bytes).
      const lo = view.getUint32(propertyOffset, true);
      const hi = view.getUint32(propertyOffset + 4, true);
      const dataAddress = hi * 4294967296 + lo;
      return {
        layoutClass: 1,
        dataAddress,
        shape: dobj.shape,
        dtype: dobj.dtype,
      };
    }
    if (layoutClass === 2) {
      // Chunked: jsfive's getters already parsed everything.
      const chunks = dobj.chunks;        // [chunkRows, chunkCols]
      const chunkAddr = dobj._chunk_address;
      const filters = dobj.filter_pipeline;
      return {
        layoutClass: 2,
        chunkAddress: chunkAddr,
        chunks,
        shape: dobj.shape,
        dtype: dobj.dtype,
        filterPipeline: filters,
      };
    }
    return null;
  }

  // Round-trip helper for the head buffer probe. Returns
  // { file, fileBytes, headBytes, isComplete } where:
  //   - file: a jsfive File parsed over the head buffer
  //   - fileBytes: total file size from probeLength
  //   - headBytes: how many bytes we actually fetched
  //   - isComplete: true iff headBytes >= fileBytes (we have the
  //     whole file; caller may still want to use jsfive for
  //     value reads).
  async function probeHead(url) {
    const HttpRange = globalThis.HttpRange;
    if (!HttpRange) throw new Error('HttpRange shim missing — required for streaming HDF5');
    const fileBytes = await HttpRange.probeLength(url);
    const headBytes = Math.min(fileBytes, HEAD_BUFFER_BYTES);
    // rangeFetch is end-inclusive; subtract 1 so we ask for
    // [0, headBytes-1].
    const ab = await HttpRange.rangeFetch(url, 0, headBytes - 1, headBytes);
    const u8 = new Uint8Array(ab);
    if (!isHdf5(u8)) {
      throw new Error('HDF5 stream: head buffer does not start with HDF5 magic');
    }
    const jsfive = getJsfive();
    let file;
    try {
      file = new jsfive.File(ab);
    } catch (e) {
      // Re-throw with a hint that the file's metadata may live
      // past the head buffer (in which case the caller can bump
      // HEAD_BUFFER_BYTES or fall back to whole-file).
      throw new Error(
        'HDF5 stream: jsfive failed to parse the head buffer ' +
        '(' + (e && e.message ? e.message : String(e)) + '). ' +
        'The file may have metadata past the first ' +
        (HEAD_BUFFER_BYTES >>> 20) + ' MB or use an unsupported ' +
        'superblock variant.'
      );
    }
    return { file, fileBytes, headBytes, isComplete: headBytes >= fileBytes };
  }

  // ---------------------------------------------------------------
  // V1 raw-data chunk B-tree walker.
  //
  // HDF5 spec § III.A.2 ("Disk Format: Level 1A2 - Version 1
  // B-trees"). Layout matches jsfive's BTreeV1RawDataChunks:
  //
  //   node header: "TREE" sig (4B) + node_type (1B) +
  //     node_level (1B) + entries_used (2B) + left_sibling (8B)
  //     + right_sibling (8B) = 24 B total
  //   per entry: chunk_size (4B) + filter_mask (4B) +
  //     chunk_offset[dims] (8B each, dims = on-disk shape dims+1
  //     for the dtype dim) + chunk_address (8B)
  //
  // We collect all leaf-node keys+addresses (which is what the
  // construct_data_from_chunks loop does in jsfive) and return
  // a flat list of { offset: [d0, d1, ...], size, mask, address }
  // entries.
  //
  // Note: `dims` here is the number of CHUNK dimensions plus one
  // — that extra dim is the "element index in the chunk", always
  // zero for our datasets but always written. h5py writes
  // chunk_offset with dims+1 64-bit values.
  // ---------------------------------------------------------------
  const BTREE_NODE_HEADER_SIZE = 24;
  const BTREE_SIGNATURE = 0x45455254;  // "TREE" little-endian

  function readUint32LE(view, offset) { return view.getUint32(offset, true); }
  function readUint64LE(view, offset) {
    // jsfive's getUint64 via BigInt is correct but slow; for chunk
    // addresses (always < 2^53) Number-converting two uint32s is
    // safe and faster.
    const lo = view.getUint32(offset, true);
    const hi = view.getUint32(offset + 4, true);
    // 2^53 == 9007199254740992. Chunk addresses in practice are
    // always file offsets so hi rarely exceeds 2^20.
    return hi * 4294967296 + lo;
  }

  // pageReader: async function(addr, byteLen) → Promise<ArrayBuffer>
  // Returns just the leaf chunk records.
  async function readChunkBTree(pageReader, rootAddr, chunkDims) {
    // Per-entry size: 8 (size + mask) + (chunkDims) * 8 (chunk offset) + 8 (address)
    const entrySize = 8 + chunkDims * 8 + 8;

    async function walk(addr, level) {
      // First fetch the node header to learn entries_used and level.
      // We don't know the exact node size up-front so we fetch an
      // upper bound: 24-byte header + max possible entries per node.
      // jsfive uses 32-entries-per-node default; h5py uses 32 too.
      // Fetch header + (entrySize + entrySize) — we'll re-fetch if
      // the node turns out wider. In practice all nodes are small.
      const headerBuf = await pageReader(addr, BTREE_NODE_HEADER_SIZE);
      const hv = new DataView(headerBuf);
      if (hv.getUint32(0, true) !== BTREE_SIGNATURE) {
        throw new Error(
          'HDF5 stream: B-tree node at 0x' + addr.toString(16) +
          ' has bad signature 0x' + hv.getUint32(0, true).toString(16) +
          ' (expected "TREE")'
        );
      }
      const nodeType = hv.getUint8(4);
      const nodeLevel = hv.getUint8(5);
      const entriesUsed = hv.getUint16(6, true);
      if (nodeType !== 1) {
        throw new Error(
          'HDF5 stream: B-tree node type ' + nodeType +
          ' (expected 1 = raw-data chunks)'
        );
      }

      // Now fetch the full body containing the per-entry records.
      // V1 B-tree node total size: header + (entries+1) * key + entries * 8 (child address)
      // For type-1 nodes the per-entry "key" preceding each child
      // pointer is (4 size + 4 mask + dims*8 offset) and there is
      // one extra trailing key after the last child. So total node
      // payload after header = (entries+1) * key + entries * 8.
      const keySize = 8 + chunkDims * 8;
      const bodySize = (entriesUsed + 1) * keySize + entriesUsed * 8;
      const bodyBuf = await pageReader(addr + BTREE_NODE_HEADER_SIZE, bodySize);
      const bv = new DataView(bodyBuf);

      const results = [];
      let off = 0;
      for (let i = 0; i < entriesUsed; i++) {
        const chunkSize = readUint32LE(bv, off); off += 4;
        const filterMask = readUint32LE(bv, off); off += 4;
        const chunkOffset = new Array(chunkDims);
        for (let d = 0; d < chunkDims; d++) {
          chunkOffset[d] = readUint64LE(bv, off);
          off += 8;
        }
        const childAddr = readUint64LE(bv, off); off += 8;
        if (nodeLevel === 0) {
          results.push({
            offset: chunkOffset,
            size: chunkSize,
            mask: filterMask,
            address: childAddr,
          });
        } else {
          // Internal node — recurse into child.
          const childResults = await walk(childAddr, nodeLevel - 1);
          for (const r of childResults) results.push(r);
        }
      }
      // Trailing key after last entry (we ignore it; it just marks
      // the upper bound). Skip keySize bytes.
      return results;
    }
    return walk(rootAddr, 0);  // node_level read from header
  }

  // Compute which chunks (in B-tree leaf-record terms) intersect
  // the sample range [sStart, sEnd). Datasets are 2-D, shape
  // [nSamples, nChannels]; chunks are [chunkRows, chunkCols].
  // We assume chunkCols == nChannels (the canonical NWB layout —
  // chunks span the full channel axis). If not, callers must use
  // the whole-file path.
  //
  // Each chunk record's `offset` is [row, col, 0] (the trailing 0
  // is the dtype index, always 0). For NWB shape == [nSamples,
  // nChannels] we need rows; col should be 0.
  function pickChunksForWindow(chunkRecords, sStart, sEnd, chunkRows) {
    const sel = [];
    for (const c of chunkRecords) {
      const row = c.offset[0];
      const rowEnd = row + chunkRows;  // exclusive
      if (rowEnd <= sStart || row >= sEnd) continue;
      sel.push(c);
    }
    // Sort by row so the output buffer is filled in order (also
    // means range-fetches are issued in roughly sequential order,
    // friendlier to HTTP/2 server pipelining than random order).
    sel.sort((a, b) => a.offset[0] - b.offset[0]);
    return sel;
  }

  // Run a chunk through the filter pipeline. We re-use jsfive's
  // `Filters` map directly so the gzip + shuffle + fletch32
  // implementations stay in one place.
  function applyFilters(buf, filterPipeline, itemSize) {
    if (!filterPipeline || !filterPipeline.length) return buf;
    const jsfive = getJsfive();
    const Filters = jsfive.Filters;
    if (!Filters) throw new Error('HDF5 stream: jsfive.Filters missing');
    // Filters are applied in reverse-order of the pipeline on
    // read (last-on-write is first-on-read). jsfive does this
    // in _filter_chunk; we mirror that.
    let out = buf;
    for (let i = filterPipeline.length - 1; i >= 0; i--) {
      const f = filterPipeline[i];
      const id = f instanceof Map ? f.get('filter_id') : f.filter_id;
      const fn = Filters.get(id);
      if (!fn) {
        throw new Error(
          'HDF5 stream: unsupported filter id ' + id +
          ' (only GZIP / SHUFFLE / FLETCH32 supported)'
        );
      }
      out = fn(out, itemSize);
    }
    return out;
  }

  // Read a single windowed slice. metadata = {
  //   shape: [nSamples, nChannels],
  //   chunks: [chunkRows, chunkCols],
  //   chunkAddress: <root B-tree address>,
  //   filterPipeline: jsfive's filter_pipeline result (or null),
  //   dtype: numpy dtype string,
  // }
  // pageReader: async (addr, len) => ArrayBuffer
  //
  // Returns a flat sample-major Float32Array of shape
  // [(sEnd-sStart) * nChannels].
  async function readWindowChunked(metadata, sStart, sEnd, pageReader) {
    const [, nChannels] = metadata.shape;
    const [chunkRows, chunkCols] = metadata.chunks;
    if (chunkCols !== nChannels) {
      throw new Error(
        'HDF5 stream: chunk does not span full channel axis ' +
        '(chunkCols=' + chunkCols + ', nChannels=' + nChannels + '). ' +
        'Multi-tile-per-row chunking is not yet supported by the streaming path.'
      );
    }
    const dt = dtypeToTypedArray(metadata.dtype);
    if (!dt) {
      throw new Error('HDF5 stream: unsupported dtype "' + metadata.dtype + '"');
    }

    // chunk-B-tree dims count: chunks.length + 1 (extra element-index dim).
    const chunkBTreeDims = metadata.chunks.length + 1;
    const allChunks = await readChunkBTree(pageReader, metadata.chunkAddress, chunkBTreeDims);
    const needed = pickChunksForWindow(allChunks, sStart, sEnd, chunkRows);

    const nWin = sEnd - sStart;
    const out = new Float32Array(nWin * nChannels);

    // Fetch chunks in parallel — the underlying HttpRange already
    // batches with HTTP/2 multiplexing.
    const fetched = await Promise.all(needed.map(async (c) => {
      // Per-chunk raw bytes are `c.size`. For uncompressed chunks
      // the size would equal chunkRows*chunkCols*itemSize; for
      // gzipped chunks it's the compressed length.
      const compressed = await pageReader(c.address, c.size);
      const itemSize = dt.bytes;
      const decoded = applyFilters(compressed, metadata.filterPipeline, itemSize);
      return { chunk: c, buf: decoded };
    }));

    // Each chunk's decoded buffer has chunkRows * chunkCols cells
    // in row-major order: cell(r, c) at offset (r*chunkCols+c)*itemSize.
    // Copy the intersecting rows into `out`.
    for (const { chunk, buf } of fetched) {
      const view = new DataView(buf);
      const chunkRow0 = chunk.offset[0];
      const chunkRowMax = Math.min(chunkRows, metadata.shape[0] - chunkRow0);
      // overlap with [sStart, sEnd)
      const rStart = Math.max(sStart, chunkRow0);
      const rEnd = Math.min(sEnd, chunkRow0 + chunkRowMax);
      for (let r = rStart; r < rEnd; r++) {
        const inRow = r - chunkRow0;
        const outRow = r - sStart;
        const inBase = inRow * chunkCols * dt.bytes;
        const outBase = outRow * nChannels;
        for (let c = 0; c < nChannels; c++) {
          out[outBase + c] = view[dt.getter](inBase + c * dt.bytes, true);
        }
      }
    }
    return out;
  }

  // ---------------------------------------------------------------
  // Contiguous-storage window read. Much simpler — the data lives
  // at a single byte address (the dataset's `_data_offset`) and is
  // stored row-major. We range-fetch exactly the sample rows we
  // need and decode in place.
  //
  // metadata = { shape, dtype, dataAddress }
  // ---------------------------------------------------------------
  async function readWindowContiguous(metadata, sStart, sEnd, pageReader) {
    const [, nChannels] = metadata.shape;
    const dt = dtypeToTypedArray(metadata.dtype);
    if (!dt) throw new Error('HDF5 stream: unsupported dtype "' + metadata.dtype + '"');
    const rowBytes = nChannels * dt.bytes;
    const startByte = metadata.dataAddress + sStart * rowBytes;
    const lenBytes = (sEnd - sStart) * rowBytes;
    const ab = await pageReader(startByte, lenBytes);
    const view = new DataView(ab);
    const nWin = sEnd - sStart;
    const out = new Float32Array(nWin * nChannels);
    let off = 0;
    for (let s = 0; s < nWin; s++) {
      const base = s * nChannels;
      for (let c = 0; c < nChannels; c++) {
        out[base + c] = view[dt.getter](off, true);
        off += dt.bytes;
      }
    }
    return out;
  }

  // Make an HttpRange-backed page reader. Each call issues one
  // range fetch; rapid sequential calls coalesce via HTTP/2
  // multiplexing inside formats/_http_range.js.
  function makeHttpPageReader(url) {
    const HttpRange = globalThis.HttpRange;
    return async function (addr, len) {
      if (len <= 0) return new ArrayBuffer(0);
      return HttpRange.rangeFetch(url, addr, addr + len - 1, len);
    };
  }

  // ---------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------
  const api = {
    HEAD_BUFFER_BYTES,
    probeHead,
    extractLayoutFromDataset,
    readChunkBTree,
    pickChunksForWindow,
    readWindowChunked,
    readWindowContiguous,
    makeHttpPageReader,
    dtypeToTypedArray,
    _isHdf5: isHdf5,
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.H5Stream = api;
})();
