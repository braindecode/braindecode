var hdf5 = (() => {
  var __defProp = Object.defineProperty;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __hasOwnProp = Object.prototype.hasOwnProperty;
  var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
  var __export = (target, all) => {
    for (var name in all)
      __defProp(target, name, { get: all[name], enumerable: true });
  };
  var __copyProps = (to, from, except, desc) => {
    if (from && typeof from === "object" || typeof from === "function") {
      for (let key of __getOwnPropNames(from))
        if (!__hasOwnProp.call(to, key) && key !== except)
          __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
    }
    return to;
  };
  var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);
  var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);

  // index.js
  var index_exports = {};
  __export(index_exports, {
    Dataset: () => Dataset,
    File: () => File,
    Filters: () => Filters,
    Group: () => Group
  });

  // esm/core.js
  function _unpack_struct_from(structure, buf, offset = 0) {
    var output = /* @__PURE__ */ new Map();
    for (let [key, fmt] of structure.entries()) {
      let value = struct.unpack_from("<" + fmt, buf, offset);
      offset += struct.calcsize(fmt);
      if (value.length == 1) {
        value = value[0];
      }
      ;
      output.set(key, value);
    }
    return output;
  }
  function assert(thing) {
    if (!thing) {
      thing();
    }
  }
  function _structure_size(structure) {
    var fmt = "<" + Array.from(structure.values()).join("");
    return struct.calcsize(fmt);
  }
  function _padded_size(size, padding_multiple = 8) {
    return Math.ceil(size / padding_multiple) * padding_multiple;
  }
  var dtype_to_format = {
    "u": "Uint",
    "i": "Int",
    "f": "Float"
  };
  function dtype_getter(dtype_str) {
    var big_endian = struct._is_big_endian(dtype_str);
    var getter, nbytes;
    if (/S/.test(dtype_str)) {
      getter = "getString";
      nbytes = ((dtype_str.match(/S(\d*)/) || [])[1] || 1) | 0;
    } else {
      let [_, fstr, bytestr] = dtype_str.match(/[<>=!@]?(i|u|f)(\d*)/);
      nbytes = parseInt(bytestr || 4, 10);
      let nbits = nbytes * 8;
      getter = "get" + dtype_to_format[fstr] + nbits.toFixed();
    }
    return [getter, big_endian, nbytes];
  }
  var Struct = class {
    constructor() {
      this.big_endian = isBigEndian();
      this.getters = {
        "s": "getUint8",
        "b": "getInt8",
        "B": "getUint8",
        "h": "getInt16",
        "H": "getUint16",
        "i": "getInt32",
        "I": "getUint32",
        "l": "getInt32",
        "L": "getUint32",
        "q": "getInt64",
        "Q": "getUint64",
        "e": "getFloat16",
        "f": "getFloat32",
        "d": "getFloat64"
      };
      this.byte_lengths = {
        "s": 1,
        "b": 1,
        "B": 1,
        "h": 2,
        "H": 2,
        "i": 4,
        "I": 4,
        "l": 4,
        "L": 4,
        "q": 8,
        "Q": 8,
        "e": 2,
        "f": 4,
        "d": 8
      };
      let all_formats = Object.keys(this.byte_lengths).join("");
      this.fmt_size_regex = "(\\d*)([" + all_formats + "])";
    }
    calcsize(fmt) {
      var size = 0;
      var match;
      var regex = new RegExp(this.fmt_size_regex, "g");
      while ((match = regex.exec(fmt)) !== null) {
        let n = parseInt(match[1] || 1, 10);
        let f = match[2];
        let subsize = this.byte_lengths[f];
        size += n * subsize;
      }
      return size;
    }
    _is_big_endian(fmt) {
      var big_endian;
      if (/^</.test(fmt)) {
        big_endian = false;
      } else if (/^(!|>)/.test(fmt)) {
        big_endian = true;
      } else {
        big_endian = this.big_endian;
      }
      return big_endian;
    }
    unpack_from(fmt, buffer, offset) {
      var offset = Number(offset || 0);
      var view = new DataView64(buffer, 0);
      var output = [];
      var big_endian = this._is_big_endian(fmt);
      var match;
      var regex = new RegExp(this.fmt_size_regex, "g");
      while ((match = regex.exec(fmt)) !== null) {
        let n = parseInt(match[1] || 1, 10);
        let f = match[2];
        let getter = this.getters[f];
        let size = this.byte_lengths[f];
        if (f == "s") {
          output.push(new TextDecoder().decode(buffer.slice(offset, offset + n)));
          offset += n;
        } else {
          for (var i = 0; i < n; i++) {
            output.push(view[getter](offset, !big_endian));
            offset += size;
          }
        }
      }
      return output;
    }
  };
  var struct = new Struct();
  function isBigEndian() {
    const array = new Uint8Array(4);
    const view = new Uint32Array(array.buffer);
    return !((view[0] = 1) & array[0]);
  }
  var WARN_OVERFLOW = false;
  var MAX_INT64 = 1n << 63n - 1n;
  var MIN_INT64 = -1n << 63n;
  var MAX_UINT64 = 1n << 64n;
  var MIN_UINT64 = 0n;
  function decodeFloat16(low, high) {
    let sign = (high & 128) >> 7;
    let exponent = (high & 124) >> 2;
    let fraction = ((high & 3) << 8) + low;
    let magnitude;
    if (exponent == 31) {
      magnitude = fraction == 0 ? Infinity : NaN;
    } else if (exponent == 0) {
      magnitude = 2 ** -14 * (fraction / 1024);
    } else {
      magnitude = 2 ** (exponent - 15) * (1 + fraction / 1024);
    }
    return sign ? -magnitude : magnitude;
  }
  var DataView64 = class extends DataView {
    getFloat16(byteOffset, littlEndian) {
      let bytes = [this.getUint8(byteOffset), this.getUint8(byteOffset + 1)];
      if (!littlEndian) bytes.reverse();
      let [low, high] = bytes;
      return decodeFloat16(low, high);
    }
    getUint64(byteOffset, littleEndian) {
      const left = BigInt(this.getUint32(byteOffset, littleEndian));
      const right = BigInt(this.getUint32(byteOffset + 4, littleEndian));
      let combined = littleEndian ? left + (right << 32n) : (left << 32n) + right;
      if (WARN_OVERFLOW && (combined < MIN_UINT64 || combined > MAX_UINT64)) {
        console.warn(combined, "exceeds range of 64-bit unsigned int");
      }
      return Number(combined);
    }
    getInt64(byteOffset, littleEndian) {
      var low, high;
      if (littleEndian) {
        low = this.getUint32(byteOffset, true);
        high = this.getInt32(byteOffset + 4, true);
      } else {
        high = this.getInt32(byteOffset, false);
        low = this.getUint32(byteOffset + 4, false);
      }
      let combined = BigInt(low) + (BigInt(high) << 32n);
      if (WARN_OVERFLOW && (combined < MIN_INT64 || combined > MAX_INT64)) {
        console.warn(combined, "exceeds range of 64-bit signed int");
      }
      return Number(combined);
    }
    getString(byteOffset, littleEndian, length) {
      const str_buffer = this.buffer.slice(byteOffset, byteOffset + length);
      const decoder = new TextDecoder();
      return decoder.decode(str_buffer);
    }
    getVLENStruct(byteOffset, littleEndian, length) {
      let item_size = this.getUint32(byteOffset, littleEndian);
      let collection_address = this.getUint64(byteOffset + 4, littleEndian);
      let object_index = this.getUint32(byteOffset + 12, littleEndian);
      return [item_size, collection_address, object_index];
    }
  };
  function bitSize(integer) {
    return integer.toString(2).length;
  }
  function _unpack_integer(nbytes, fh, offset = 0, littleEndian = true) {
    let bytes = new Uint8Array(fh.slice(offset, offset + nbytes));
    if (!littleEndian) {
      bytes.reverse();
    }
    let integer = bytes.reduce((accumulator, currentValue, index) => accumulator + (currentValue << index * 8), 0);
    return integer;
  }

  // esm/datatype-msg.js
  var DatatypeMessage = class {
    //""" Representation of a HDF5 Datatype Message. """
    //# Contents and layout defined in IV.A.2.d.
    constructor(buf, offset) {
      this.buf = buf;
      this.offset = offset;
      this.dtype = this.determine_dtype();
    }
    determine_dtype() {
      let datatype_msg = _unpack_struct_from(DATATYPE_MSG, this.buf, this.offset);
      this.offset += DATATYPE_MSG_SIZE;
      let datatype_class = datatype_msg.get("class_and_version") & 15;
      if (datatype_class == DATATYPE_FIXED_POINT) {
        return this._determine_dtype_fixed_point(datatype_msg);
      } else if (datatype_class == DATATYPE_FLOATING_POINT) {
        return this._determine_dtype_floating_point(datatype_msg);
      } else if (datatype_class == DATATYPE_TIME) {
        throw "Time datatype class not supported.";
      } else if (datatype_class == DATATYPE_STRING) {
        return this._determine_dtype_string(datatype_msg);
      } else if (datatype_class == DATATYPE_BITFIELD) {
        throw "Bitfield datatype class not supported.";
      } else if (datatype_class == DATATYPE_OPAQUE) {
        throw "Opaque datatype class not supported.";
      } else if (datatype_class == DATATYPE_COMPOUND) {
        return this._determine_dtype_compound(datatype_msg);
      } else if (datatype_class == DATATYPE_REFERENCE) {
        return ["REFERENCE", datatype_msg.get("size")];
      } else if (datatype_class == DATATYPE_ENUMERATED) {
        return this.determine_dtype();
      } else if (datatype_class == DATATYPE_ARRAY) {
        throw "Array datatype class not supported.";
      } else if (datatype_class == DATATYPE_VARIABLE_LENGTH) {
        let vlen_type = this._determine_dtype_vlen(datatype_msg);
        if (vlen_type[0] == "VLEN_SEQUENCE") {
          let base_type = this.determine_dtype();
          vlen_type = ["VLEN_SEQUENCE", base_type];
        }
        return vlen_type;
      } else {
        throw "Invalid datatype class " + datatype_class;
      }
    }
    _determine_dtype_fixed_point(datatype_msg) {
      let length_in_bytes = datatype_msg.get("size");
      if (![1, 2, 4, 8].includes(length_in_bytes)) {
        throw "Unsupported datatype size";
      }
      let signed = datatype_msg.get("class_bit_field_0") & 8;
      var dtype_char;
      if (signed > 0) {
        dtype_char = "i";
      } else {
        dtype_char = "u";
      }
      let byte_order = datatype_msg.get("class_bit_field_0") & 1;
      var byte_order_char;
      if (byte_order == 0) {
        byte_order_char = "<";
      } else {
        byte_order_char = ">";
      }
      this.offset += 4;
      return byte_order_char + dtype_char + length_in_bytes.toFixed();
    }
    _determine_dtype_floating_point(datatype_msg) {
      let length_in_bytes = datatype_msg.get("size");
      if (![1, 2, 4, 8].includes(length_in_bytes)) {
        throw "Unsupported datatype size";
      }
      let dtype_char = "f";
      let byte_order = datatype_msg.get("class_bit_field_0") & 1;
      var byte_order_char;
      if (byte_order == 0) {
        byte_order_char = "<";
      } else {
        byte_order_char = ">";
      }
      this.offset += 12;
      return byte_order_char + dtype_char + length_in_bytes.toFixed();
    }
    _determine_dtype_string(datatype_msg) {
      return "S" + datatype_msg.get("size").toFixed();
    }
    _determine_dtype_vlen(datatype_msg) {
      let vlen_type = datatype_msg.get("class_bit_field_0") & 1;
      if (vlen_type != 1) {
        return ["VLEN_SEQUENCE", 0, 0];
      }
      let padding_type = datatype_msg.get("class_bit_field_0") >> 4;
      let character_set = datatype_msg.get("class_bit_field_1") & 1;
      return ["VLEN_STRING", padding_type, character_set];
    }
    _determine_dtype_compound(datatype_msg) {
      throw "Compound type not yet implemented!";
    }
  };
  var DATATYPE_MSG = /* @__PURE__ */ new Map([
    ["class_and_version", "B"],
    ["class_bit_field_0", "B"],
    ["class_bit_field_1", "B"],
    ["class_bit_field_2", "B"],
    ["size", "I"]
  ]);
  var DATATYPE_MSG_SIZE = _structure_size(DATATYPE_MSG);
  var COMPOUND_PROP_DESC_V1 = /* @__PURE__ */ new Map([
    ["offset", "I"],
    ["dimensionality", "B"],
    ["reserved_0", "B"],
    ["reserved_1", "B"],
    ["reserved_2", "B"],
    ["permutation", "I"],
    ["reserved_3", "I"],
    ["dim_size_1", "I"],
    ["dim_size_2", "I"],
    ["dim_size_3", "I"],
    ["dim_size_4", "I"]
  ]);
  var COMPOUND_PROP_DESC_V1_SIZE = _structure_size(COMPOUND_PROP_DESC_V1);
  var DATATYPE_FIXED_POINT = 0;
  var DATATYPE_FLOATING_POINT = 1;
  var DATATYPE_TIME = 2;
  var DATATYPE_STRING = 3;
  var DATATYPE_BITFIELD = 4;
  var DATATYPE_OPAQUE = 5;
  var DATATYPE_COMPOUND = 6;
  var DATATYPE_REFERENCE = 7;
  var DATATYPE_ENUMERATED = 8;
  var DATATYPE_VARIABLE_LENGTH = 9;
  var DATATYPE_ARRAY = 10;

  // node_modules/pako/dist/pako.esm.mjs
  var Z_FIXED$1 = 4;
  var Z_BINARY = 0;
  var Z_TEXT = 1;
  var Z_UNKNOWN$1 = 2;
  function zero$1(buf) {
    let len = buf.length;
    while (--len >= 0) {
      buf[len] = 0;
    }
  }
  var STORED_BLOCK = 0;
  var STATIC_TREES = 1;
  var DYN_TREES = 2;
  var MIN_MATCH$1 = 3;
  var MAX_MATCH$1 = 258;
  var LENGTH_CODES$1 = 29;
  var LITERALS$1 = 256;
  var L_CODES$1 = LITERALS$1 + 1 + LENGTH_CODES$1;
  var D_CODES$1 = 30;
  var BL_CODES$1 = 19;
  var HEAP_SIZE$1 = 2 * L_CODES$1 + 1;
  var MAX_BITS$1 = 15;
  var Buf_size = 16;
  var MAX_BL_BITS = 7;
  var END_BLOCK = 256;
  var REP_3_6 = 16;
  var REPZ_3_10 = 17;
  var REPZ_11_138 = 18;
  var extra_lbits = (
    /* extra bits for each length code */
    new Uint8Array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0])
  );
  var extra_dbits = (
    /* extra bits for each distance code */
    new Uint8Array([0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13])
  );
  var extra_blbits = (
    /* extra bits for each bit length code */
    new Uint8Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 7])
  );
  var bl_order = new Uint8Array([16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]);
  var DIST_CODE_LEN = 512;
  var static_ltree = new Array((L_CODES$1 + 2) * 2);
  zero$1(static_ltree);
  var static_dtree = new Array(D_CODES$1 * 2);
  zero$1(static_dtree);
  var _dist_code = new Array(DIST_CODE_LEN);
  zero$1(_dist_code);
  var _length_code = new Array(MAX_MATCH$1 - MIN_MATCH$1 + 1);
  zero$1(_length_code);
  var base_length = new Array(LENGTH_CODES$1);
  zero$1(base_length);
  var base_dist = new Array(D_CODES$1);
  zero$1(base_dist);
  function StaticTreeDesc(static_tree, extra_bits, extra_base, elems, max_length) {
    this.static_tree = static_tree;
    this.extra_bits = extra_bits;
    this.extra_base = extra_base;
    this.elems = elems;
    this.max_length = max_length;
    this.has_stree = static_tree && static_tree.length;
  }
  var static_l_desc;
  var static_d_desc;
  var static_bl_desc;
  function TreeDesc(dyn_tree, stat_desc) {
    this.dyn_tree = dyn_tree;
    this.max_code = 0;
    this.stat_desc = stat_desc;
  }
  var d_code = (dist) => {
    return dist < 256 ? _dist_code[dist] : _dist_code[256 + (dist >>> 7)];
  };
  var put_short = (s, w) => {
    s.pending_buf[s.pending++] = w & 255;
    s.pending_buf[s.pending++] = w >>> 8 & 255;
  };
  var send_bits = (s, value, length) => {
    if (s.bi_valid > Buf_size - length) {
      s.bi_buf |= value << s.bi_valid & 65535;
      put_short(s, s.bi_buf);
      s.bi_buf = value >> Buf_size - s.bi_valid;
      s.bi_valid += length - Buf_size;
    } else {
      s.bi_buf |= value << s.bi_valid & 65535;
      s.bi_valid += length;
    }
  };
  var send_code = (s, c, tree) => {
    send_bits(
      s,
      tree[c * 2],
      tree[c * 2 + 1]
      /*.Len*/
    );
  };
  var bi_reverse = (code, len) => {
    let res = 0;
    do {
      res |= code & 1;
      code >>>= 1;
      res <<= 1;
    } while (--len > 0);
    return res >>> 1;
  };
  var bi_flush = (s) => {
    if (s.bi_valid === 16) {
      put_short(s, s.bi_buf);
      s.bi_buf = 0;
      s.bi_valid = 0;
    } else if (s.bi_valid >= 8) {
      s.pending_buf[s.pending++] = s.bi_buf & 255;
      s.bi_buf >>= 8;
      s.bi_valid -= 8;
    }
  };
  var gen_bitlen = (s, desc) => {
    const tree = desc.dyn_tree;
    const max_code = desc.max_code;
    const stree = desc.stat_desc.static_tree;
    const has_stree = desc.stat_desc.has_stree;
    const extra = desc.stat_desc.extra_bits;
    const base = desc.stat_desc.extra_base;
    const max_length = desc.stat_desc.max_length;
    let h;
    let n, m;
    let bits;
    let xbits;
    let f;
    let overflow = 0;
    for (bits = 0; bits <= MAX_BITS$1; bits++) {
      s.bl_count[bits] = 0;
    }
    tree[s.heap[s.heap_max] * 2 + 1] = 0;
    for (h = s.heap_max + 1; h < HEAP_SIZE$1; h++) {
      n = s.heap[h];
      bits = tree[tree[n * 2 + 1] * 2 + 1] + 1;
      if (bits > max_length) {
        bits = max_length;
        overflow++;
      }
      tree[n * 2 + 1] = bits;
      if (n > max_code) {
        continue;
      }
      s.bl_count[bits]++;
      xbits = 0;
      if (n >= base) {
        xbits = extra[n - base];
      }
      f = tree[n * 2];
      s.opt_len += f * (bits + xbits);
      if (has_stree) {
        s.static_len += f * (stree[n * 2 + 1] + xbits);
      }
    }
    if (overflow === 0) {
      return;
    }
    do {
      bits = max_length - 1;
      while (s.bl_count[bits] === 0) {
        bits--;
      }
      s.bl_count[bits]--;
      s.bl_count[bits + 1] += 2;
      s.bl_count[max_length]--;
      overflow -= 2;
    } while (overflow > 0);
    for (bits = max_length; bits !== 0; bits--) {
      n = s.bl_count[bits];
      while (n !== 0) {
        m = s.heap[--h];
        if (m > max_code) {
          continue;
        }
        if (tree[m * 2 + 1] !== bits) {
          s.opt_len += (bits - tree[m * 2 + 1]) * tree[m * 2];
          tree[m * 2 + 1] = bits;
        }
        n--;
      }
    }
  };
  var gen_codes = (tree, max_code, bl_count) => {
    const next_code = new Array(MAX_BITS$1 + 1);
    let code = 0;
    let bits;
    let n;
    for (bits = 1; bits <= MAX_BITS$1; bits++) {
      code = code + bl_count[bits - 1] << 1;
      next_code[bits] = code;
    }
    for (n = 0; n <= max_code; n++) {
      let len = tree[n * 2 + 1];
      if (len === 0) {
        continue;
      }
      tree[n * 2] = bi_reverse(next_code[len]++, len);
    }
  };
  var tr_static_init = () => {
    let n;
    let bits;
    let length;
    let code;
    let dist;
    const bl_count = new Array(MAX_BITS$1 + 1);
    length = 0;
    for (code = 0; code < LENGTH_CODES$1 - 1; code++) {
      base_length[code] = length;
      for (n = 0; n < 1 << extra_lbits[code]; n++) {
        _length_code[length++] = code;
      }
    }
    _length_code[length - 1] = code;
    dist = 0;
    for (code = 0; code < 16; code++) {
      base_dist[code] = dist;
      for (n = 0; n < 1 << extra_dbits[code]; n++) {
        _dist_code[dist++] = code;
      }
    }
    dist >>= 7;
    for (; code < D_CODES$1; code++) {
      base_dist[code] = dist << 7;
      for (n = 0; n < 1 << extra_dbits[code] - 7; n++) {
        _dist_code[256 + dist++] = code;
      }
    }
    for (bits = 0; bits <= MAX_BITS$1; bits++) {
      bl_count[bits] = 0;
    }
    n = 0;
    while (n <= 143) {
      static_ltree[n * 2 + 1] = 8;
      n++;
      bl_count[8]++;
    }
    while (n <= 255) {
      static_ltree[n * 2 + 1] = 9;
      n++;
      bl_count[9]++;
    }
    while (n <= 279) {
      static_ltree[n * 2 + 1] = 7;
      n++;
      bl_count[7]++;
    }
    while (n <= 287) {
      static_ltree[n * 2 + 1] = 8;
      n++;
      bl_count[8]++;
    }
    gen_codes(static_ltree, L_CODES$1 + 1, bl_count);
    for (n = 0; n < D_CODES$1; n++) {
      static_dtree[n * 2 + 1] = 5;
      static_dtree[n * 2] = bi_reverse(n, 5);
    }
    static_l_desc = new StaticTreeDesc(static_ltree, extra_lbits, LITERALS$1 + 1, L_CODES$1, MAX_BITS$1);
    static_d_desc = new StaticTreeDesc(static_dtree, extra_dbits, 0, D_CODES$1, MAX_BITS$1);
    static_bl_desc = new StaticTreeDesc(new Array(0), extra_blbits, 0, BL_CODES$1, MAX_BL_BITS);
  };
  var init_block = (s) => {
    let n;
    for (n = 0; n < L_CODES$1; n++) {
      s.dyn_ltree[n * 2] = 0;
    }
    for (n = 0; n < D_CODES$1; n++) {
      s.dyn_dtree[n * 2] = 0;
    }
    for (n = 0; n < BL_CODES$1; n++) {
      s.bl_tree[n * 2] = 0;
    }
    s.dyn_ltree[END_BLOCK * 2] = 1;
    s.opt_len = s.static_len = 0;
    s.sym_next = s.matches = 0;
  };
  var bi_windup = (s) => {
    if (s.bi_valid > 8) {
      put_short(s, s.bi_buf);
    } else if (s.bi_valid > 0) {
      s.pending_buf[s.pending++] = s.bi_buf;
    }
    s.bi_buf = 0;
    s.bi_valid = 0;
  };
  var smaller = (tree, n, m, depth) => {
    const _n2 = n * 2;
    const _m2 = m * 2;
    return tree[_n2] < tree[_m2] || tree[_n2] === tree[_m2] && depth[n] <= depth[m];
  };
  var pqdownheap = (s, tree, k) => {
    const v = s.heap[k];
    let j = k << 1;
    while (j <= s.heap_len) {
      if (j < s.heap_len && smaller(tree, s.heap[j + 1], s.heap[j], s.depth)) {
        j++;
      }
      if (smaller(tree, v, s.heap[j], s.depth)) {
        break;
      }
      s.heap[k] = s.heap[j];
      k = j;
      j <<= 1;
    }
    s.heap[k] = v;
  };
  var compress_block = (s, ltree, dtree) => {
    let dist;
    let lc;
    let sx = 0;
    let code;
    let extra;
    if (s.sym_next !== 0) {
      do {
        dist = s.pending_buf[s.sym_buf + sx++] & 255;
        dist += (s.pending_buf[s.sym_buf + sx++] & 255) << 8;
        lc = s.pending_buf[s.sym_buf + sx++];
        if (dist === 0) {
          send_code(s, lc, ltree);
        } else {
          code = _length_code[lc];
          send_code(s, code + LITERALS$1 + 1, ltree);
          extra = extra_lbits[code];
          if (extra !== 0) {
            lc -= base_length[code];
            send_bits(s, lc, extra);
          }
          dist--;
          code = d_code(dist);
          send_code(s, code, dtree);
          extra = extra_dbits[code];
          if (extra !== 0) {
            dist -= base_dist[code];
            send_bits(s, dist, extra);
          }
        }
      } while (sx < s.sym_next);
    }
    send_code(s, END_BLOCK, ltree);
  };
  var build_tree = (s, desc) => {
    const tree = desc.dyn_tree;
    const stree = desc.stat_desc.static_tree;
    const has_stree = desc.stat_desc.has_stree;
    const elems = desc.stat_desc.elems;
    let n, m;
    let max_code = -1;
    let node2;
    s.heap_len = 0;
    s.heap_max = HEAP_SIZE$1;
    for (n = 0; n < elems; n++) {
      if (tree[n * 2] !== 0) {
        s.heap[++s.heap_len] = max_code = n;
        s.depth[n] = 0;
      } else {
        tree[n * 2 + 1] = 0;
      }
    }
    while (s.heap_len < 2) {
      node2 = s.heap[++s.heap_len] = max_code < 2 ? ++max_code : 0;
      tree[node2 * 2] = 1;
      s.depth[node2] = 0;
      s.opt_len--;
      if (has_stree) {
        s.static_len -= stree[node2 * 2 + 1];
      }
    }
    desc.max_code = max_code;
    for (n = s.heap_len >> 1; n >= 1; n--) {
      pqdownheap(s, tree, n);
    }
    node2 = elems;
    do {
      n = s.heap[
        1
        /*SMALLEST*/
      ];
      s.heap[
        1
        /*SMALLEST*/
      ] = s.heap[s.heap_len--];
      pqdownheap(
        s,
        tree,
        1
        /*SMALLEST*/
      );
      m = s.heap[
        1
        /*SMALLEST*/
      ];
      s.heap[--s.heap_max] = n;
      s.heap[--s.heap_max] = m;
      tree[node2 * 2] = tree[n * 2] + tree[m * 2];
      s.depth[node2] = (s.depth[n] >= s.depth[m] ? s.depth[n] : s.depth[m]) + 1;
      tree[n * 2 + 1] = tree[m * 2 + 1] = node2;
      s.heap[
        1
        /*SMALLEST*/
      ] = node2++;
      pqdownheap(
        s,
        tree,
        1
        /*SMALLEST*/
      );
    } while (s.heap_len >= 2);
    s.heap[--s.heap_max] = s.heap[
      1
      /*SMALLEST*/
    ];
    gen_bitlen(s, desc);
    gen_codes(tree, max_code, s.bl_count);
  };
  var scan_tree = (s, tree, max_code) => {
    let n;
    let prevlen = -1;
    let curlen;
    let nextlen = tree[0 * 2 + 1];
    let count = 0;
    let max_count = 7;
    let min_count = 4;
    if (nextlen === 0) {
      max_count = 138;
      min_count = 3;
    }
    tree[(max_code + 1) * 2 + 1] = 65535;
    for (n = 0; n <= max_code; n++) {
      curlen = nextlen;
      nextlen = tree[(n + 1) * 2 + 1];
      if (++count < max_count && curlen === nextlen) {
        continue;
      } else if (count < min_count) {
        s.bl_tree[curlen * 2] += count;
      } else if (curlen !== 0) {
        if (curlen !== prevlen) {
          s.bl_tree[curlen * 2]++;
        }
        s.bl_tree[REP_3_6 * 2]++;
      } else if (count <= 10) {
        s.bl_tree[REPZ_3_10 * 2]++;
      } else {
        s.bl_tree[REPZ_11_138 * 2]++;
      }
      count = 0;
      prevlen = curlen;
      if (nextlen === 0) {
        max_count = 138;
        min_count = 3;
      } else if (curlen === nextlen) {
        max_count = 6;
        min_count = 3;
      } else {
        max_count = 7;
        min_count = 4;
      }
    }
  };
  var send_tree = (s, tree, max_code) => {
    let n;
    let prevlen = -1;
    let curlen;
    let nextlen = tree[0 * 2 + 1];
    let count = 0;
    let max_count = 7;
    let min_count = 4;
    if (nextlen === 0) {
      max_count = 138;
      min_count = 3;
    }
    for (n = 0; n <= max_code; n++) {
      curlen = nextlen;
      nextlen = tree[(n + 1) * 2 + 1];
      if (++count < max_count && curlen === nextlen) {
        continue;
      } else if (count < min_count) {
        do {
          send_code(s, curlen, s.bl_tree);
        } while (--count !== 0);
      } else if (curlen !== 0) {
        if (curlen !== prevlen) {
          send_code(s, curlen, s.bl_tree);
          count--;
        }
        send_code(s, REP_3_6, s.bl_tree);
        send_bits(s, count - 3, 2);
      } else if (count <= 10) {
        send_code(s, REPZ_3_10, s.bl_tree);
        send_bits(s, count - 3, 3);
      } else {
        send_code(s, REPZ_11_138, s.bl_tree);
        send_bits(s, count - 11, 7);
      }
      count = 0;
      prevlen = curlen;
      if (nextlen === 0) {
        max_count = 138;
        min_count = 3;
      } else if (curlen === nextlen) {
        max_count = 6;
        min_count = 3;
      } else {
        max_count = 7;
        min_count = 4;
      }
    }
  };
  var build_bl_tree = (s) => {
    let max_blindex;
    scan_tree(s, s.dyn_ltree, s.l_desc.max_code);
    scan_tree(s, s.dyn_dtree, s.d_desc.max_code);
    build_tree(s, s.bl_desc);
    for (max_blindex = BL_CODES$1 - 1; max_blindex >= 3; max_blindex--) {
      if (s.bl_tree[bl_order[max_blindex] * 2 + 1] !== 0) {
        break;
      }
    }
    s.opt_len += 3 * (max_blindex + 1) + 5 + 5 + 4;
    return max_blindex;
  };
  var send_all_trees = (s, lcodes, dcodes, blcodes) => {
    let rank2;
    send_bits(s, lcodes - 257, 5);
    send_bits(s, dcodes - 1, 5);
    send_bits(s, blcodes - 4, 4);
    for (rank2 = 0; rank2 < blcodes; rank2++) {
      send_bits(s, s.bl_tree[bl_order[rank2] * 2 + 1], 3);
    }
    send_tree(s, s.dyn_ltree, lcodes - 1);
    send_tree(s, s.dyn_dtree, dcodes - 1);
  };
  var detect_data_type = (s) => {
    let block_mask = 4093624447;
    let n;
    for (n = 0; n <= 31; n++, block_mask >>>= 1) {
      if (block_mask & 1 && s.dyn_ltree[n * 2] !== 0) {
        return Z_BINARY;
      }
    }
    if (s.dyn_ltree[9 * 2] !== 0 || s.dyn_ltree[10 * 2] !== 0 || s.dyn_ltree[13 * 2] !== 0) {
      return Z_TEXT;
    }
    for (n = 32; n < LITERALS$1; n++) {
      if (s.dyn_ltree[n * 2] !== 0) {
        return Z_TEXT;
      }
    }
    return Z_BINARY;
  };
  var static_init_done = false;
  var _tr_init$1 = (s) => {
    if (!static_init_done) {
      tr_static_init();
      static_init_done = true;
    }
    s.l_desc = new TreeDesc(s.dyn_ltree, static_l_desc);
    s.d_desc = new TreeDesc(s.dyn_dtree, static_d_desc);
    s.bl_desc = new TreeDesc(s.bl_tree, static_bl_desc);
    s.bi_buf = 0;
    s.bi_valid = 0;
    init_block(s);
  };
  var _tr_stored_block$1 = (s, buf, stored_len, last) => {
    send_bits(s, (STORED_BLOCK << 1) + (last ? 1 : 0), 3);
    bi_windup(s);
    put_short(s, stored_len);
    put_short(s, ~stored_len);
    if (stored_len) {
      s.pending_buf.set(s.window.subarray(buf, buf + stored_len), s.pending);
    }
    s.pending += stored_len;
  };
  var _tr_align$1 = (s) => {
    send_bits(s, STATIC_TREES << 1, 3);
    send_code(s, END_BLOCK, static_ltree);
    bi_flush(s);
  };
  var _tr_flush_block$1 = (s, buf, stored_len, last) => {
    let opt_lenb, static_lenb;
    let max_blindex = 0;
    if (s.level > 0) {
      if (s.strm.data_type === Z_UNKNOWN$1) {
        s.strm.data_type = detect_data_type(s);
      }
      build_tree(s, s.l_desc);
      build_tree(s, s.d_desc);
      max_blindex = build_bl_tree(s);
      opt_lenb = s.opt_len + 3 + 7 >>> 3;
      static_lenb = s.static_len + 3 + 7 >>> 3;
      if (static_lenb <= opt_lenb) {
        opt_lenb = static_lenb;
      }
    } else {
      opt_lenb = static_lenb = stored_len + 5;
    }
    if (stored_len + 4 <= opt_lenb && buf !== -1) {
      _tr_stored_block$1(s, buf, stored_len, last);
    } else if (s.strategy === Z_FIXED$1 || static_lenb === opt_lenb) {
      send_bits(s, (STATIC_TREES << 1) + (last ? 1 : 0), 3);
      compress_block(s, static_ltree, static_dtree);
    } else {
      send_bits(s, (DYN_TREES << 1) + (last ? 1 : 0), 3);
      send_all_trees(s, s.l_desc.max_code + 1, s.d_desc.max_code + 1, max_blindex + 1);
      compress_block(s, s.dyn_ltree, s.dyn_dtree);
    }
    init_block(s);
    if (last) {
      bi_windup(s);
    }
  };
  var _tr_tally$1 = (s, dist, lc) => {
    s.pending_buf[s.sym_buf + s.sym_next++] = dist;
    s.pending_buf[s.sym_buf + s.sym_next++] = dist >> 8;
    s.pending_buf[s.sym_buf + s.sym_next++] = lc;
    if (dist === 0) {
      s.dyn_ltree[lc * 2]++;
    } else {
      s.matches++;
      dist--;
      s.dyn_ltree[(_length_code[lc] + LITERALS$1 + 1) * 2]++;
      s.dyn_dtree[d_code(dist) * 2]++;
    }
    return s.sym_next === s.sym_end;
  };
  var _tr_init_1 = _tr_init$1;
  var _tr_stored_block_1 = _tr_stored_block$1;
  var _tr_flush_block_1 = _tr_flush_block$1;
  var _tr_tally_1 = _tr_tally$1;
  var _tr_align_1 = _tr_align$1;
  var trees = {
    _tr_init: _tr_init_1,
    _tr_stored_block: _tr_stored_block_1,
    _tr_flush_block: _tr_flush_block_1,
    _tr_tally: _tr_tally_1,
    _tr_align: _tr_align_1
  };
  var adler32 = (adler, buf, len, pos) => {
    let s1 = adler & 65535 | 0, s2 = adler >>> 16 & 65535 | 0, n = 0;
    while (len !== 0) {
      n = len > 2e3 ? 2e3 : len;
      len -= n;
      do {
        s1 = s1 + buf[pos++] | 0;
        s2 = s2 + s1 | 0;
      } while (--n);
      s1 %= 65521;
      s2 %= 65521;
    }
    return s1 | s2 << 16 | 0;
  };
  var adler32_1 = adler32;
  var makeTable = () => {
    let c, table = [];
    for (var n = 0; n < 256; n++) {
      c = n;
      for (var k = 0; k < 8; k++) {
        c = c & 1 ? 3988292384 ^ c >>> 1 : c >>> 1;
      }
      table[n] = c;
    }
    return table;
  };
  var crcTable = new Uint32Array(makeTable());
  var crc32 = (crc, buf, len, pos) => {
    const t = crcTable;
    const end = pos + len;
    crc ^= -1;
    for (let i = pos; i < end; i++) {
      crc = crc >>> 8 ^ t[(crc ^ buf[i]) & 255];
    }
    return crc ^ -1;
  };
  var crc32_1 = crc32;
  var messages = {
    2: "need dictionary",
    /* Z_NEED_DICT       2  */
    1: "stream end",
    /* Z_STREAM_END      1  */
    0: "",
    /* Z_OK              0  */
    "-1": "file error",
    /* Z_ERRNO         (-1) */
    "-2": "stream error",
    /* Z_STREAM_ERROR  (-2) */
    "-3": "data error",
    /* Z_DATA_ERROR    (-3) */
    "-4": "insufficient memory",
    /* Z_MEM_ERROR     (-4) */
    "-5": "buffer error",
    /* Z_BUF_ERROR     (-5) */
    "-6": "incompatible version"
    /* Z_VERSION_ERROR (-6) */
  };
  var constants$2 = {
    /* Allowed flush values; see deflate() and inflate() below for details */
    Z_NO_FLUSH: 0,
    Z_PARTIAL_FLUSH: 1,
    Z_SYNC_FLUSH: 2,
    Z_FULL_FLUSH: 3,
    Z_FINISH: 4,
    Z_BLOCK: 5,
    Z_TREES: 6,
    /* Return codes for the compression/decompression functions. Negative values
    * are errors, positive values are used for special but normal events.
    */
    Z_OK: 0,
    Z_STREAM_END: 1,
    Z_NEED_DICT: 2,
    Z_ERRNO: -1,
    Z_STREAM_ERROR: -2,
    Z_DATA_ERROR: -3,
    Z_MEM_ERROR: -4,
    Z_BUF_ERROR: -5,
    //Z_VERSION_ERROR: -6,
    /* compression levels */
    Z_NO_COMPRESSION: 0,
    Z_BEST_SPEED: 1,
    Z_BEST_COMPRESSION: 9,
    Z_DEFAULT_COMPRESSION: -1,
    Z_FILTERED: 1,
    Z_HUFFMAN_ONLY: 2,
    Z_RLE: 3,
    Z_FIXED: 4,
    Z_DEFAULT_STRATEGY: 0,
    /* Possible values of the data_type field (though see inflate()) */
    Z_BINARY: 0,
    Z_TEXT: 1,
    //Z_ASCII:                1, // = Z_TEXT (deprecated)
    Z_UNKNOWN: 2,
    /* The deflate compression method */
    Z_DEFLATED: 8
    //Z_NULL:                 null // Use -1 or null inline, depending on var type
  };
  var { _tr_init, _tr_stored_block, _tr_flush_block, _tr_tally, _tr_align } = trees;
  var {
    Z_NO_FLUSH: Z_NO_FLUSH$2,
    Z_PARTIAL_FLUSH,
    Z_FULL_FLUSH: Z_FULL_FLUSH$1,
    Z_FINISH: Z_FINISH$3,
    Z_BLOCK: Z_BLOCK$1,
    Z_OK: Z_OK$3,
    Z_STREAM_END: Z_STREAM_END$3,
    Z_STREAM_ERROR: Z_STREAM_ERROR$2,
    Z_DATA_ERROR: Z_DATA_ERROR$2,
    Z_BUF_ERROR: Z_BUF_ERROR$1,
    Z_DEFAULT_COMPRESSION: Z_DEFAULT_COMPRESSION$1,
    Z_FILTERED,
    Z_HUFFMAN_ONLY,
    Z_RLE,
    Z_FIXED,
    Z_DEFAULT_STRATEGY: Z_DEFAULT_STRATEGY$1,
    Z_UNKNOWN,
    Z_DEFLATED: Z_DEFLATED$2
  } = constants$2;
  var MAX_MEM_LEVEL = 9;
  var MAX_WBITS$1 = 15;
  var DEF_MEM_LEVEL = 8;
  var LENGTH_CODES = 29;
  var LITERALS = 256;
  var L_CODES = LITERALS + 1 + LENGTH_CODES;
  var D_CODES = 30;
  var BL_CODES = 19;
  var HEAP_SIZE = 2 * L_CODES + 1;
  var MAX_BITS = 15;
  var MIN_MATCH = 3;
  var MAX_MATCH = 258;
  var MIN_LOOKAHEAD = MAX_MATCH + MIN_MATCH + 1;
  var PRESET_DICT = 32;
  var INIT_STATE = 42;
  var GZIP_STATE = 57;
  var EXTRA_STATE = 69;
  var NAME_STATE = 73;
  var COMMENT_STATE = 91;
  var HCRC_STATE = 103;
  var BUSY_STATE = 113;
  var FINISH_STATE = 666;
  var BS_NEED_MORE = 1;
  var BS_BLOCK_DONE = 2;
  var BS_FINISH_STARTED = 3;
  var BS_FINISH_DONE = 4;
  var OS_CODE = 3;
  var err = (strm, errorCode) => {
    strm.msg = messages[errorCode];
    return errorCode;
  };
  var rank = (f) => {
    return f * 2 - (f > 4 ? 9 : 0);
  };
  var zero = (buf) => {
    let len = buf.length;
    while (--len >= 0) {
      buf[len] = 0;
    }
  };
  var slide_hash = (s) => {
    let n, m;
    let p;
    let wsize = s.w_size;
    n = s.hash_size;
    p = n;
    do {
      m = s.head[--p];
      s.head[p] = m >= wsize ? m - wsize : 0;
    } while (--n);
    n = wsize;
    p = n;
    do {
      m = s.prev[--p];
      s.prev[p] = m >= wsize ? m - wsize : 0;
    } while (--n);
  };
  var HASH_ZLIB = (s, prev, data) => (prev << s.hash_shift ^ data) & s.hash_mask;
  var HASH = HASH_ZLIB;
  var flush_pending = (strm) => {
    const s = strm.state;
    let len = s.pending;
    if (len > strm.avail_out) {
      len = strm.avail_out;
    }
    if (len === 0) {
      return;
    }
    strm.output.set(s.pending_buf.subarray(s.pending_out, s.pending_out + len), strm.next_out);
    strm.next_out += len;
    s.pending_out += len;
    strm.total_out += len;
    strm.avail_out -= len;
    s.pending -= len;
    if (s.pending === 0) {
      s.pending_out = 0;
    }
  };
  var flush_block_only = (s, last) => {
    _tr_flush_block(s, s.block_start >= 0 ? s.block_start : -1, s.strstart - s.block_start, last);
    s.block_start = s.strstart;
    flush_pending(s.strm);
  };
  var put_byte = (s, b) => {
    s.pending_buf[s.pending++] = b;
  };
  var putShortMSB = (s, b) => {
    s.pending_buf[s.pending++] = b >>> 8 & 255;
    s.pending_buf[s.pending++] = b & 255;
  };
  var read_buf = (strm, buf, start, size) => {
    let len = strm.avail_in;
    if (len > size) {
      len = size;
    }
    if (len === 0) {
      return 0;
    }
    strm.avail_in -= len;
    buf.set(strm.input.subarray(strm.next_in, strm.next_in + len), start);
    if (strm.state.wrap === 1) {
      strm.adler = adler32_1(strm.adler, buf, len, start);
    } else if (strm.state.wrap === 2) {
      strm.adler = crc32_1(strm.adler, buf, len, start);
    }
    strm.next_in += len;
    strm.total_in += len;
    return len;
  };
  var longest_match = (s, cur_match) => {
    let chain_length = s.max_chain_length;
    let scan = s.strstart;
    let match;
    let len;
    let best_len = s.prev_length;
    let nice_match = s.nice_match;
    const limit = s.strstart > s.w_size - MIN_LOOKAHEAD ? s.strstart - (s.w_size - MIN_LOOKAHEAD) : 0;
    const _win = s.window;
    const wmask = s.w_mask;
    const prev = s.prev;
    const strend = s.strstart + MAX_MATCH;
    let scan_end1 = _win[scan + best_len - 1];
    let scan_end = _win[scan + best_len];
    if (s.prev_length >= s.good_match) {
      chain_length >>= 2;
    }
    if (nice_match > s.lookahead) {
      nice_match = s.lookahead;
    }
    do {
      match = cur_match;
      if (_win[match + best_len] !== scan_end || _win[match + best_len - 1] !== scan_end1 || _win[match] !== _win[scan] || _win[++match] !== _win[scan + 1]) {
        continue;
      }
      scan += 2;
      match++;
      do {
      } while (_win[++scan] === _win[++match] && _win[++scan] === _win[++match] && _win[++scan] === _win[++match] && _win[++scan] === _win[++match] && _win[++scan] === _win[++match] && _win[++scan] === _win[++match] && _win[++scan] === _win[++match] && _win[++scan] === _win[++match] && scan < strend);
      len = MAX_MATCH - (strend - scan);
      scan = strend - MAX_MATCH;
      if (len > best_len) {
        s.match_start = cur_match;
        best_len = len;
        if (len >= nice_match) {
          break;
        }
        scan_end1 = _win[scan + best_len - 1];
        scan_end = _win[scan + best_len];
      }
    } while ((cur_match = prev[cur_match & wmask]) > limit && --chain_length !== 0);
    if (best_len <= s.lookahead) {
      return best_len;
    }
    return s.lookahead;
  };
  var fill_window = (s) => {
    const _w_size = s.w_size;
    let n, more, str;
    do {
      more = s.window_size - s.lookahead - s.strstart;
      if (s.strstart >= _w_size + (_w_size - MIN_LOOKAHEAD)) {
        s.window.set(s.window.subarray(_w_size, _w_size + _w_size - more), 0);
        s.match_start -= _w_size;
        s.strstart -= _w_size;
        s.block_start -= _w_size;
        if (s.insert > s.strstart) {
          s.insert = s.strstart;
        }
        slide_hash(s);
        more += _w_size;
      }
      if (s.strm.avail_in === 0) {
        break;
      }
      n = read_buf(s.strm, s.window, s.strstart + s.lookahead, more);
      s.lookahead += n;
      if (s.lookahead + s.insert >= MIN_MATCH) {
        str = s.strstart - s.insert;
        s.ins_h = s.window[str];
        s.ins_h = HASH(s, s.ins_h, s.window[str + 1]);
        while (s.insert) {
          s.ins_h = HASH(s, s.ins_h, s.window[str + MIN_MATCH - 1]);
          s.prev[str & s.w_mask] = s.head[s.ins_h];
          s.head[s.ins_h] = str;
          str++;
          s.insert--;
          if (s.lookahead + s.insert < MIN_MATCH) {
            break;
          }
        }
      }
    } while (s.lookahead < MIN_LOOKAHEAD && s.strm.avail_in !== 0);
  };
  var deflate_stored = (s, flush) => {
    let min_block = s.pending_buf_size - 5 > s.w_size ? s.w_size : s.pending_buf_size - 5;
    let len, left, have, last = 0;
    let used = s.strm.avail_in;
    do {
      len = 65535;
      have = s.bi_valid + 42 >> 3;
      if (s.strm.avail_out < have) {
        break;
      }
      have = s.strm.avail_out - have;
      left = s.strstart - s.block_start;
      if (len > left + s.strm.avail_in) {
        len = left + s.strm.avail_in;
      }
      if (len > have) {
        len = have;
      }
      if (len < min_block && (len === 0 && flush !== Z_FINISH$3 || flush === Z_NO_FLUSH$2 || len !== left + s.strm.avail_in)) {
        break;
      }
      last = flush === Z_FINISH$3 && len === left + s.strm.avail_in ? 1 : 0;
      _tr_stored_block(s, 0, 0, last);
      s.pending_buf[s.pending - 4] = len;
      s.pending_buf[s.pending - 3] = len >> 8;
      s.pending_buf[s.pending - 2] = ~len;
      s.pending_buf[s.pending - 1] = ~len >> 8;
      flush_pending(s.strm);
      if (left) {
        if (left > len) {
          left = len;
        }
        s.strm.output.set(s.window.subarray(s.block_start, s.block_start + left), s.strm.next_out);
        s.strm.next_out += left;
        s.strm.avail_out -= left;
        s.strm.total_out += left;
        s.block_start += left;
        len -= left;
      }
      if (len) {
        read_buf(s.strm, s.strm.output, s.strm.next_out, len);
        s.strm.next_out += len;
        s.strm.avail_out -= len;
        s.strm.total_out += len;
      }
    } while (last === 0);
    used -= s.strm.avail_in;
    if (used) {
      if (used >= s.w_size) {
        s.matches = 2;
        s.window.set(s.strm.input.subarray(s.strm.next_in - s.w_size, s.strm.next_in), 0);
        s.strstart = s.w_size;
        s.insert = s.strstart;
      } else {
        if (s.window_size - s.strstart <= used) {
          s.strstart -= s.w_size;
          s.window.set(s.window.subarray(s.w_size, s.w_size + s.strstart), 0);
          if (s.matches < 2) {
            s.matches++;
          }
          if (s.insert > s.strstart) {
            s.insert = s.strstart;
          }
        }
        s.window.set(s.strm.input.subarray(s.strm.next_in - used, s.strm.next_in), s.strstart);
        s.strstart += used;
        s.insert += used > s.w_size - s.insert ? s.w_size - s.insert : used;
      }
      s.block_start = s.strstart;
    }
    if (s.high_water < s.strstart) {
      s.high_water = s.strstart;
    }
    if (last) {
      return BS_FINISH_DONE;
    }
    if (flush !== Z_NO_FLUSH$2 && flush !== Z_FINISH$3 && s.strm.avail_in === 0 && s.strstart === s.block_start) {
      return BS_BLOCK_DONE;
    }
    have = s.window_size - s.strstart;
    if (s.strm.avail_in > have && s.block_start >= s.w_size) {
      s.block_start -= s.w_size;
      s.strstart -= s.w_size;
      s.window.set(s.window.subarray(s.w_size, s.w_size + s.strstart), 0);
      if (s.matches < 2) {
        s.matches++;
      }
      have += s.w_size;
      if (s.insert > s.strstart) {
        s.insert = s.strstart;
      }
    }
    if (have > s.strm.avail_in) {
      have = s.strm.avail_in;
    }
    if (have) {
      read_buf(s.strm, s.window, s.strstart, have);
      s.strstart += have;
      s.insert += have > s.w_size - s.insert ? s.w_size - s.insert : have;
    }
    if (s.high_water < s.strstart) {
      s.high_water = s.strstart;
    }
    have = s.bi_valid + 42 >> 3;
    have = s.pending_buf_size - have > 65535 ? 65535 : s.pending_buf_size - have;
    min_block = have > s.w_size ? s.w_size : have;
    left = s.strstart - s.block_start;
    if (left >= min_block || (left || flush === Z_FINISH$3) && flush !== Z_NO_FLUSH$2 && s.strm.avail_in === 0 && left <= have) {
      len = left > have ? have : left;
      last = flush === Z_FINISH$3 && s.strm.avail_in === 0 && len === left ? 1 : 0;
      _tr_stored_block(s, s.block_start, len, last);
      s.block_start += len;
      flush_pending(s.strm);
    }
    return last ? BS_FINISH_STARTED : BS_NEED_MORE;
  };
  var deflate_fast = (s, flush) => {
    let hash_head;
    let bflush;
    for (; ; ) {
      if (s.lookahead < MIN_LOOKAHEAD) {
        fill_window(s);
        if (s.lookahead < MIN_LOOKAHEAD && flush === Z_NO_FLUSH$2) {
          return BS_NEED_MORE;
        }
        if (s.lookahead === 0) {
          break;
        }
      }
      hash_head = 0;
      if (s.lookahead >= MIN_MATCH) {
        s.ins_h = HASH(s, s.ins_h, s.window[s.strstart + MIN_MATCH - 1]);
        hash_head = s.prev[s.strstart & s.w_mask] = s.head[s.ins_h];
        s.head[s.ins_h] = s.strstart;
      }
      if (hash_head !== 0 && s.strstart - hash_head <= s.w_size - MIN_LOOKAHEAD) {
        s.match_length = longest_match(s, hash_head);
      }
      if (s.match_length >= MIN_MATCH) {
        bflush = _tr_tally(s, s.strstart - s.match_start, s.match_length - MIN_MATCH);
        s.lookahead -= s.match_length;
        if (s.match_length <= s.max_lazy_match && s.lookahead >= MIN_MATCH) {
          s.match_length--;
          do {
            s.strstart++;
            s.ins_h = HASH(s, s.ins_h, s.window[s.strstart + MIN_MATCH - 1]);
            hash_head = s.prev[s.strstart & s.w_mask] = s.head[s.ins_h];
            s.head[s.ins_h] = s.strstart;
          } while (--s.match_length !== 0);
          s.strstart++;
        } else {
          s.strstart += s.match_length;
          s.match_length = 0;
          s.ins_h = s.window[s.strstart];
          s.ins_h = HASH(s, s.ins_h, s.window[s.strstart + 1]);
        }
      } else {
        bflush = _tr_tally(s, 0, s.window[s.strstart]);
        s.lookahead--;
        s.strstart++;
      }
      if (bflush) {
        flush_block_only(s, false);
        if (s.strm.avail_out === 0) {
          return BS_NEED_MORE;
        }
      }
    }
    s.insert = s.strstart < MIN_MATCH - 1 ? s.strstart : MIN_MATCH - 1;
    if (flush === Z_FINISH$3) {
      flush_block_only(s, true);
      if (s.strm.avail_out === 0) {
        return BS_FINISH_STARTED;
      }
      return BS_FINISH_DONE;
    }
    if (s.sym_next) {
      flush_block_only(s, false);
      if (s.strm.avail_out === 0) {
        return BS_NEED_MORE;
      }
    }
    return BS_BLOCK_DONE;
  };
  var deflate_slow = (s, flush) => {
    let hash_head;
    let bflush;
    let max_insert;
    for (; ; ) {
      if (s.lookahead < MIN_LOOKAHEAD) {
        fill_window(s);
        if (s.lookahead < MIN_LOOKAHEAD && flush === Z_NO_FLUSH$2) {
          return BS_NEED_MORE;
        }
        if (s.lookahead === 0) {
          break;
        }
      }
      hash_head = 0;
      if (s.lookahead >= MIN_MATCH) {
        s.ins_h = HASH(s, s.ins_h, s.window[s.strstart + MIN_MATCH - 1]);
        hash_head = s.prev[s.strstart & s.w_mask] = s.head[s.ins_h];
        s.head[s.ins_h] = s.strstart;
      }
      s.prev_length = s.match_length;
      s.prev_match = s.match_start;
      s.match_length = MIN_MATCH - 1;
      if (hash_head !== 0 && s.prev_length < s.max_lazy_match && s.strstart - hash_head <= s.w_size - MIN_LOOKAHEAD) {
        s.match_length = longest_match(s, hash_head);
        if (s.match_length <= 5 && (s.strategy === Z_FILTERED || s.match_length === MIN_MATCH && s.strstart - s.match_start > 4096)) {
          s.match_length = MIN_MATCH - 1;
        }
      }
      if (s.prev_length >= MIN_MATCH && s.match_length <= s.prev_length) {
        max_insert = s.strstart + s.lookahead - MIN_MATCH;
        bflush = _tr_tally(s, s.strstart - 1 - s.prev_match, s.prev_length - MIN_MATCH);
        s.lookahead -= s.prev_length - 1;
        s.prev_length -= 2;
        do {
          if (++s.strstart <= max_insert) {
            s.ins_h = HASH(s, s.ins_h, s.window[s.strstart + MIN_MATCH - 1]);
            hash_head = s.prev[s.strstart & s.w_mask] = s.head[s.ins_h];
            s.head[s.ins_h] = s.strstart;
          }
        } while (--s.prev_length !== 0);
        s.match_available = 0;
        s.match_length = MIN_MATCH - 1;
        s.strstart++;
        if (bflush) {
          flush_block_only(s, false);
          if (s.strm.avail_out === 0) {
            return BS_NEED_MORE;
          }
        }
      } else if (s.match_available) {
        bflush = _tr_tally(s, 0, s.window[s.strstart - 1]);
        if (bflush) {
          flush_block_only(s, false);
        }
        s.strstart++;
        s.lookahead--;
        if (s.strm.avail_out === 0) {
          return BS_NEED_MORE;
        }
      } else {
        s.match_available = 1;
        s.strstart++;
        s.lookahead--;
      }
    }
    if (s.match_available) {
      bflush = _tr_tally(s, 0, s.window[s.strstart - 1]);
      s.match_available = 0;
    }
    s.insert = s.strstart < MIN_MATCH - 1 ? s.strstart : MIN_MATCH - 1;
    if (flush === Z_FINISH$3) {
      flush_block_only(s, true);
      if (s.strm.avail_out === 0) {
        return BS_FINISH_STARTED;
      }
      return BS_FINISH_DONE;
    }
    if (s.sym_next) {
      flush_block_only(s, false);
      if (s.strm.avail_out === 0) {
        return BS_NEED_MORE;
      }
    }
    return BS_BLOCK_DONE;
  };
  var deflate_rle = (s, flush) => {
    let bflush;
    let prev;
    let scan, strend;
    const _win = s.window;
    for (; ; ) {
      if (s.lookahead <= MAX_MATCH) {
        fill_window(s);
        if (s.lookahead <= MAX_MATCH && flush === Z_NO_FLUSH$2) {
          return BS_NEED_MORE;
        }
        if (s.lookahead === 0) {
          break;
        }
      }
      s.match_length = 0;
      if (s.lookahead >= MIN_MATCH && s.strstart > 0) {
        scan = s.strstart - 1;
        prev = _win[scan];
        if (prev === _win[++scan] && prev === _win[++scan] && prev === _win[++scan]) {
          strend = s.strstart + MAX_MATCH;
          do {
          } while (prev === _win[++scan] && prev === _win[++scan] && prev === _win[++scan] && prev === _win[++scan] && prev === _win[++scan] && prev === _win[++scan] && prev === _win[++scan] && prev === _win[++scan] && scan < strend);
          s.match_length = MAX_MATCH - (strend - scan);
          if (s.match_length > s.lookahead) {
            s.match_length = s.lookahead;
          }
        }
      }
      if (s.match_length >= MIN_MATCH) {
        bflush = _tr_tally(s, 1, s.match_length - MIN_MATCH);
        s.lookahead -= s.match_length;
        s.strstart += s.match_length;
        s.match_length = 0;
      } else {
        bflush = _tr_tally(s, 0, s.window[s.strstart]);
        s.lookahead--;
        s.strstart++;
      }
      if (bflush) {
        flush_block_only(s, false);
        if (s.strm.avail_out === 0) {
          return BS_NEED_MORE;
        }
      }
    }
    s.insert = 0;
    if (flush === Z_FINISH$3) {
      flush_block_only(s, true);
      if (s.strm.avail_out === 0) {
        return BS_FINISH_STARTED;
      }
      return BS_FINISH_DONE;
    }
    if (s.sym_next) {
      flush_block_only(s, false);
      if (s.strm.avail_out === 0) {
        return BS_NEED_MORE;
      }
    }
    return BS_BLOCK_DONE;
  };
  var deflate_huff = (s, flush) => {
    let bflush;
    for (; ; ) {
      if (s.lookahead === 0) {
        fill_window(s);
        if (s.lookahead === 0) {
          if (flush === Z_NO_FLUSH$2) {
            return BS_NEED_MORE;
          }
          break;
        }
      }
      s.match_length = 0;
      bflush = _tr_tally(s, 0, s.window[s.strstart]);
      s.lookahead--;
      s.strstart++;
      if (bflush) {
        flush_block_only(s, false);
        if (s.strm.avail_out === 0) {
          return BS_NEED_MORE;
        }
      }
    }
    s.insert = 0;
    if (flush === Z_FINISH$3) {
      flush_block_only(s, true);
      if (s.strm.avail_out === 0) {
        return BS_FINISH_STARTED;
      }
      return BS_FINISH_DONE;
    }
    if (s.sym_next) {
      flush_block_only(s, false);
      if (s.strm.avail_out === 0) {
        return BS_NEED_MORE;
      }
    }
    return BS_BLOCK_DONE;
  };
  function Config(good_length, max_lazy, nice_length, max_chain, func) {
    this.good_length = good_length;
    this.max_lazy = max_lazy;
    this.nice_length = nice_length;
    this.max_chain = max_chain;
    this.func = func;
  }
  var configuration_table = [
    /*      good lazy nice chain */
    new Config(0, 0, 0, 0, deflate_stored),
    /* 0 store only */
    new Config(4, 4, 8, 4, deflate_fast),
    /* 1 max speed, no lazy matches */
    new Config(4, 5, 16, 8, deflate_fast),
    /* 2 */
    new Config(4, 6, 32, 32, deflate_fast),
    /* 3 */
    new Config(4, 4, 16, 16, deflate_slow),
    /* 4 lazy matches */
    new Config(8, 16, 32, 32, deflate_slow),
    /* 5 */
    new Config(8, 16, 128, 128, deflate_slow),
    /* 6 */
    new Config(8, 32, 128, 256, deflate_slow),
    /* 7 */
    new Config(32, 128, 258, 1024, deflate_slow),
    /* 8 */
    new Config(32, 258, 258, 4096, deflate_slow)
    /* 9 max compression */
  ];
  var lm_init = (s) => {
    s.window_size = 2 * s.w_size;
    zero(s.head);
    s.max_lazy_match = configuration_table[s.level].max_lazy;
    s.good_match = configuration_table[s.level].good_length;
    s.nice_match = configuration_table[s.level].nice_length;
    s.max_chain_length = configuration_table[s.level].max_chain;
    s.strstart = 0;
    s.block_start = 0;
    s.lookahead = 0;
    s.insert = 0;
    s.match_length = s.prev_length = MIN_MATCH - 1;
    s.match_available = 0;
    s.ins_h = 0;
  };
  function DeflateState() {
    this.strm = null;
    this.status = 0;
    this.pending_buf = null;
    this.pending_buf_size = 0;
    this.pending_out = 0;
    this.pending = 0;
    this.wrap = 0;
    this.gzhead = null;
    this.gzindex = 0;
    this.method = Z_DEFLATED$2;
    this.last_flush = -1;
    this.w_size = 0;
    this.w_bits = 0;
    this.w_mask = 0;
    this.window = null;
    this.window_size = 0;
    this.prev = null;
    this.head = null;
    this.ins_h = 0;
    this.hash_size = 0;
    this.hash_bits = 0;
    this.hash_mask = 0;
    this.hash_shift = 0;
    this.block_start = 0;
    this.match_length = 0;
    this.prev_match = 0;
    this.match_available = 0;
    this.strstart = 0;
    this.match_start = 0;
    this.lookahead = 0;
    this.prev_length = 0;
    this.max_chain_length = 0;
    this.max_lazy_match = 0;
    this.level = 0;
    this.strategy = 0;
    this.good_match = 0;
    this.nice_match = 0;
    this.dyn_ltree = new Uint16Array(HEAP_SIZE * 2);
    this.dyn_dtree = new Uint16Array((2 * D_CODES + 1) * 2);
    this.bl_tree = new Uint16Array((2 * BL_CODES + 1) * 2);
    zero(this.dyn_ltree);
    zero(this.dyn_dtree);
    zero(this.bl_tree);
    this.l_desc = null;
    this.d_desc = null;
    this.bl_desc = null;
    this.bl_count = new Uint16Array(MAX_BITS + 1);
    this.heap = new Uint16Array(2 * L_CODES + 1);
    zero(this.heap);
    this.heap_len = 0;
    this.heap_max = 0;
    this.depth = new Uint16Array(2 * L_CODES + 1);
    zero(this.depth);
    this.sym_buf = 0;
    this.lit_bufsize = 0;
    this.sym_next = 0;
    this.sym_end = 0;
    this.opt_len = 0;
    this.static_len = 0;
    this.matches = 0;
    this.insert = 0;
    this.bi_buf = 0;
    this.bi_valid = 0;
  }
  var deflateStateCheck = (strm) => {
    if (!strm) {
      return 1;
    }
    const s = strm.state;
    if (!s || s.strm !== strm || s.status !== INIT_STATE && //#ifdef GZIP
    s.status !== GZIP_STATE && //#endif
    s.status !== EXTRA_STATE && s.status !== NAME_STATE && s.status !== COMMENT_STATE && s.status !== HCRC_STATE && s.status !== BUSY_STATE && s.status !== FINISH_STATE) {
      return 1;
    }
    return 0;
  };
  var deflateResetKeep = (strm) => {
    if (deflateStateCheck(strm)) {
      return err(strm, Z_STREAM_ERROR$2);
    }
    strm.total_in = strm.total_out = 0;
    strm.data_type = Z_UNKNOWN;
    const s = strm.state;
    s.pending = 0;
    s.pending_out = 0;
    if (s.wrap < 0) {
      s.wrap = -s.wrap;
    }
    s.status = //#ifdef GZIP
    s.wrap === 2 ? GZIP_STATE : (
      //#endif
      s.wrap ? INIT_STATE : BUSY_STATE
    );
    strm.adler = s.wrap === 2 ? 0 : 1;
    s.last_flush = -2;
    _tr_init(s);
    return Z_OK$3;
  };
  var deflateReset = (strm) => {
    const ret = deflateResetKeep(strm);
    if (ret === Z_OK$3) {
      lm_init(strm.state);
    }
    return ret;
  };
  var deflateSetHeader = (strm, head) => {
    if (deflateStateCheck(strm) || strm.state.wrap !== 2) {
      return Z_STREAM_ERROR$2;
    }
    strm.state.gzhead = head;
    return Z_OK$3;
  };
  var deflateInit2 = (strm, level, method, windowBits, memLevel, strategy) => {
    if (!strm) {
      return Z_STREAM_ERROR$2;
    }
    let wrap = 1;
    if (level === Z_DEFAULT_COMPRESSION$1) {
      level = 6;
    }
    if (windowBits < 0) {
      wrap = 0;
      windowBits = -windowBits;
    } else if (windowBits > 15) {
      wrap = 2;
      windowBits -= 16;
    }
    if (memLevel < 1 || memLevel > MAX_MEM_LEVEL || method !== Z_DEFLATED$2 || windowBits < 8 || windowBits > 15 || level < 0 || level > 9 || strategy < 0 || strategy > Z_FIXED || windowBits === 8 && wrap !== 1) {
      return err(strm, Z_STREAM_ERROR$2);
    }
    if (windowBits === 8) {
      windowBits = 9;
    }
    const s = new DeflateState();
    strm.state = s;
    s.strm = strm;
    s.status = INIT_STATE;
    s.wrap = wrap;
    s.gzhead = null;
    s.w_bits = windowBits;
    s.w_size = 1 << s.w_bits;
    s.w_mask = s.w_size - 1;
    s.hash_bits = memLevel + 7;
    s.hash_size = 1 << s.hash_bits;
    s.hash_mask = s.hash_size - 1;
    s.hash_shift = ~~((s.hash_bits + MIN_MATCH - 1) / MIN_MATCH);
    s.window = new Uint8Array(s.w_size * 2);
    s.head = new Uint16Array(s.hash_size);
    s.prev = new Uint16Array(s.w_size);
    s.lit_bufsize = 1 << memLevel + 6;
    s.pending_buf_size = s.lit_bufsize * 4;
    s.pending_buf = new Uint8Array(s.pending_buf_size);
    s.sym_buf = s.lit_bufsize;
    s.sym_end = (s.lit_bufsize - 1) * 3;
    s.level = level;
    s.strategy = strategy;
    s.method = method;
    return deflateReset(strm);
  };
  var deflateInit = (strm, level) => {
    return deflateInit2(strm, level, Z_DEFLATED$2, MAX_WBITS$1, DEF_MEM_LEVEL, Z_DEFAULT_STRATEGY$1);
  };
  var deflate$2 = (strm, flush) => {
    if (deflateStateCheck(strm) || flush > Z_BLOCK$1 || flush < 0) {
      return strm ? err(strm, Z_STREAM_ERROR$2) : Z_STREAM_ERROR$2;
    }
    const s = strm.state;
    if (!strm.output || strm.avail_in !== 0 && !strm.input || s.status === FINISH_STATE && flush !== Z_FINISH$3) {
      return err(strm, strm.avail_out === 0 ? Z_BUF_ERROR$1 : Z_STREAM_ERROR$2);
    }
    const old_flush = s.last_flush;
    s.last_flush = flush;
    if (s.pending !== 0) {
      flush_pending(strm);
      if (strm.avail_out === 0) {
        s.last_flush = -1;
        return Z_OK$3;
      }
    } else if (strm.avail_in === 0 && rank(flush) <= rank(old_flush) && flush !== Z_FINISH$3) {
      return err(strm, Z_BUF_ERROR$1);
    }
    if (s.status === FINISH_STATE && strm.avail_in !== 0) {
      return err(strm, Z_BUF_ERROR$1);
    }
    if (s.status === INIT_STATE && s.wrap === 0) {
      s.status = BUSY_STATE;
    }
    if (s.status === INIT_STATE) {
      let header = Z_DEFLATED$2 + (s.w_bits - 8 << 4) << 8;
      let level_flags = -1;
      if (s.strategy >= Z_HUFFMAN_ONLY || s.level < 2) {
        level_flags = 0;
      } else if (s.level < 6) {
        level_flags = 1;
      } else if (s.level === 6) {
        level_flags = 2;
      } else {
        level_flags = 3;
      }
      header |= level_flags << 6;
      if (s.strstart !== 0) {
        header |= PRESET_DICT;
      }
      header += 31 - header % 31;
      putShortMSB(s, header);
      if (s.strstart !== 0) {
        putShortMSB(s, strm.adler >>> 16);
        putShortMSB(s, strm.adler & 65535);
      }
      strm.adler = 1;
      s.status = BUSY_STATE;
      flush_pending(strm);
      if (s.pending !== 0) {
        s.last_flush = -1;
        return Z_OK$3;
      }
    }
    if (s.status === GZIP_STATE) {
      strm.adler = 0;
      put_byte(s, 31);
      put_byte(s, 139);
      put_byte(s, 8);
      if (!s.gzhead) {
        put_byte(s, 0);
        put_byte(s, 0);
        put_byte(s, 0);
        put_byte(s, 0);
        put_byte(s, 0);
        put_byte(s, s.level === 9 ? 2 : s.strategy >= Z_HUFFMAN_ONLY || s.level < 2 ? 4 : 0);
        put_byte(s, OS_CODE);
        s.status = BUSY_STATE;
        flush_pending(strm);
        if (s.pending !== 0) {
          s.last_flush = -1;
          return Z_OK$3;
        }
      } else {
        put_byte(
          s,
          (s.gzhead.text ? 1 : 0) + (s.gzhead.hcrc ? 2 : 0) + (!s.gzhead.extra ? 0 : 4) + (!s.gzhead.name ? 0 : 8) + (!s.gzhead.comment ? 0 : 16)
        );
        put_byte(s, s.gzhead.time & 255);
        put_byte(s, s.gzhead.time >> 8 & 255);
        put_byte(s, s.gzhead.time >> 16 & 255);
        put_byte(s, s.gzhead.time >> 24 & 255);
        put_byte(s, s.level === 9 ? 2 : s.strategy >= Z_HUFFMAN_ONLY || s.level < 2 ? 4 : 0);
        put_byte(s, s.gzhead.os & 255);
        if (s.gzhead.extra && s.gzhead.extra.length) {
          put_byte(s, s.gzhead.extra.length & 255);
          put_byte(s, s.gzhead.extra.length >> 8 & 255);
        }
        if (s.gzhead.hcrc) {
          strm.adler = crc32_1(strm.adler, s.pending_buf, s.pending, 0);
        }
        s.gzindex = 0;
        s.status = EXTRA_STATE;
      }
    }
    if (s.status === EXTRA_STATE) {
      if (s.gzhead.extra) {
        let beg = s.pending;
        let left = (s.gzhead.extra.length & 65535) - s.gzindex;
        while (s.pending + left > s.pending_buf_size) {
          let copy = s.pending_buf_size - s.pending;
          s.pending_buf.set(s.gzhead.extra.subarray(s.gzindex, s.gzindex + copy), s.pending);
          s.pending = s.pending_buf_size;
          if (s.gzhead.hcrc && s.pending > beg) {
            strm.adler = crc32_1(strm.adler, s.pending_buf, s.pending - beg, beg);
          }
          s.gzindex += copy;
          flush_pending(strm);
          if (s.pending !== 0) {
            s.last_flush = -1;
            return Z_OK$3;
          }
          beg = 0;
          left -= copy;
        }
        let gzhead_extra = new Uint8Array(s.gzhead.extra);
        s.pending_buf.set(gzhead_extra.subarray(s.gzindex, s.gzindex + left), s.pending);
        s.pending += left;
        if (s.gzhead.hcrc && s.pending > beg) {
          strm.adler = crc32_1(strm.adler, s.pending_buf, s.pending - beg, beg);
        }
        s.gzindex = 0;
      }
      s.status = NAME_STATE;
    }
    if (s.status === NAME_STATE) {
      if (s.gzhead.name) {
        let beg = s.pending;
        let val;
        do {
          if (s.pending === s.pending_buf_size) {
            if (s.gzhead.hcrc && s.pending > beg) {
              strm.adler = crc32_1(strm.adler, s.pending_buf, s.pending - beg, beg);
            }
            flush_pending(strm);
            if (s.pending !== 0) {
              s.last_flush = -1;
              return Z_OK$3;
            }
            beg = 0;
          }
          if (s.gzindex < s.gzhead.name.length) {
            val = s.gzhead.name.charCodeAt(s.gzindex++) & 255;
          } else {
            val = 0;
          }
          put_byte(s, val);
        } while (val !== 0);
        if (s.gzhead.hcrc && s.pending > beg) {
          strm.adler = crc32_1(strm.adler, s.pending_buf, s.pending - beg, beg);
        }
        s.gzindex = 0;
      }
      s.status = COMMENT_STATE;
    }
    if (s.status === COMMENT_STATE) {
      if (s.gzhead.comment) {
        let beg = s.pending;
        let val;
        do {
          if (s.pending === s.pending_buf_size) {
            if (s.gzhead.hcrc && s.pending > beg) {
              strm.adler = crc32_1(strm.adler, s.pending_buf, s.pending - beg, beg);
            }
            flush_pending(strm);
            if (s.pending !== 0) {
              s.last_flush = -1;
              return Z_OK$3;
            }
            beg = 0;
          }
          if (s.gzindex < s.gzhead.comment.length) {
            val = s.gzhead.comment.charCodeAt(s.gzindex++) & 255;
          } else {
            val = 0;
          }
          put_byte(s, val);
        } while (val !== 0);
        if (s.gzhead.hcrc && s.pending > beg) {
          strm.adler = crc32_1(strm.adler, s.pending_buf, s.pending - beg, beg);
        }
      }
      s.status = HCRC_STATE;
    }
    if (s.status === HCRC_STATE) {
      if (s.gzhead.hcrc) {
        if (s.pending + 2 > s.pending_buf_size) {
          flush_pending(strm);
          if (s.pending !== 0) {
            s.last_flush = -1;
            return Z_OK$3;
          }
        }
        put_byte(s, strm.adler & 255);
        put_byte(s, strm.adler >> 8 & 255);
        strm.adler = 0;
      }
      s.status = BUSY_STATE;
      flush_pending(strm);
      if (s.pending !== 0) {
        s.last_flush = -1;
        return Z_OK$3;
      }
    }
    if (strm.avail_in !== 0 || s.lookahead !== 0 || flush !== Z_NO_FLUSH$2 && s.status !== FINISH_STATE) {
      let bstate = s.level === 0 ? deflate_stored(s, flush) : s.strategy === Z_HUFFMAN_ONLY ? deflate_huff(s, flush) : s.strategy === Z_RLE ? deflate_rle(s, flush) : configuration_table[s.level].func(s, flush);
      if (bstate === BS_FINISH_STARTED || bstate === BS_FINISH_DONE) {
        s.status = FINISH_STATE;
      }
      if (bstate === BS_NEED_MORE || bstate === BS_FINISH_STARTED) {
        if (strm.avail_out === 0) {
          s.last_flush = -1;
        }
        return Z_OK$3;
      }
      if (bstate === BS_BLOCK_DONE) {
        if (flush === Z_PARTIAL_FLUSH) {
          _tr_align(s);
        } else if (flush !== Z_BLOCK$1) {
          _tr_stored_block(s, 0, 0, false);
          if (flush === Z_FULL_FLUSH$1) {
            zero(s.head);
            if (s.lookahead === 0) {
              s.strstart = 0;
              s.block_start = 0;
              s.insert = 0;
            }
          }
        }
        flush_pending(strm);
        if (strm.avail_out === 0) {
          s.last_flush = -1;
          return Z_OK$3;
        }
      }
    }
    if (flush !== Z_FINISH$3) {
      return Z_OK$3;
    }
    if (s.wrap <= 0) {
      return Z_STREAM_END$3;
    }
    if (s.wrap === 2) {
      put_byte(s, strm.adler & 255);
      put_byte(s, strm.adler >> 8 & 255);
      put_byte(s, strm.adler >> 16 & 255);
      put_byte(s, strm.adler >> 24 & 255);
      put_byte(s, strm.total_in & 255);
      put_byte(s, strm.total_in >> 8 & 255);
      put_byte(s, strm.total_in >> 16 & 255);
      put_byte(s, strm.total_in >> 24 & 255);
    } else {
      putShortMSB(s, strm.adler >>> 16);
      putShortMSB(s, strm.adler & 65535);
    }
    flush_pending(strm);
    if (s.wrap > 0) {
      s.wrap = -s.wrap;
    }
    return s.pending !== 0 ? Z_OK$3 : Z_STREAM_END$3;
  };
  var deflateEnd = (strm) => {
    if (deflateStateCheck(strm)) {
      return Z_STREAM_ERROR$2;
    }
    const status = strm.state.status;
    strm.state = null;
    return status === BUSY_STATE ? err(strm, Z_DATA_ERROR$2) : Z_OK$3;
  };
  var deflateSetDictionary = (strm, dictionary) => {
    let dictLength = dictionary.length;
    if (deflateStateCheck(strm)) {
      return Z_STREAM_ERROR$2;
    }
    const s = strm.state;
    const wrap = s.wrap;
    if (wrap === 2 || wrap === 1 && s.status !== INIT_STATE || s.lookahead) {
      return Z_STREAM_ERROR$2;
    }
    if (wrap === 1) {
      strm.adler = adler32_1(strm.adler, dictionary, dictLength, 0);
    }
    s.wrap = 0;
    if (dictLength >= s.w_size) {
      if (wrap === 0) {
        zero(s.head);
        s.strstart = 0;
        s.block_start = 0;
        s.insert = 0;
      }
      let tmpDict = new Uint8Array(s.w_size);
      tmpDict.set(dictionary.subarray(dictLength - s.w_size, dictLength), 0);
      dictionary = tmpDict;
      dictLength = s.w_size;
    }
    const avail = strm.avail_in;
    const next = strm.next_in;
    const input = strm.input;
    strm.avail_in = dictLength;
    strm.next_in = 0;
    strm.input = dictionary;
    fill_window(s);
    while (s.lookahead >= MIN_MATCH) {
      let str = s.strstart;
      let n = s.lookahead - (MIN_MATCH - 1);
      do {
        s.ins_h = HASH(s, s.ins_h, s.window[str + MIN_MATCH - 1]);
        s.prev[str & s.w_mask] = s.head[s.ins_h];
        s.head[s.ins_h] = str;
        str++;
      } while (--n);
      s.strstart = str;
      s.lookahead = MIN_MATCH - 1;
      fill_window(s);
    }
    s.strstart += s.lookahead;
    s.block_start = s.strstart;
    s.insert = s.lookahead;
    s.lookahead = 0;
    s.match_length = s.prev_length = MIN_MATCH - 1;
    s.match_available = 0;
    strm.next_in = next;
    strm.input = input;
    strm.avail_in = avail;
    s.wrap = wrap;
    return Z_OK$3;
  };
  var deflateInit_1 = deflateInit;
  var deflateInit2_1 = deflateInit2;
  var deflateReset_1 = deflateReset;
  var deflateResetKeep_1 = deflateResetKeep;
  var deflateSetHeader_1 = deflateSetHeader;
  var deflate_2$1 = deflate$2;
  var deflateEnd_1 = deflateEnd;
  var deflateSetDictionary_1 = deflateSetDictionary;
  var deflateInfo = "pako deflate (from Nodeca project)";
  var deflate_1$2 = {
    deflateInit: deflateInit_1,
    deflateInit2: deflateInit2_1,
    deflateReset: deflateReset_1,
    deflateResetKeep: deflateResetKeep_1,
    deflateSetHeader: deflateSetHeader_1,
    deflate: deflate_2$1,
    deflateEnd: deflateEnd_1,
    deflateSetDictionary: deflateSetDictionary_1,
    deflateInfo
  };
  var _has = (obj, key) => {
    return Object.prototype.hasOwnProperty.call(obj, key);
  };
  var assign = function(obj) {
    const sources = Array.prototype.slice.call(arguments, 1);
    while (sources.length) {
      const source = sources.shift();
      if (!source) {
        continue;
      }
      if (typeof source !== "object") {
        throw new TypeError(source + "must be non-object");
      }
      for (const p in source) {
        if (_has(source, p)) {
          obj[p] = source[p];
        }
      }
    }
    return obj;
  };
  var flattenChunks = (chunks) => {
    let len = 0;
    for (let i = 0, l = chunks.length; i < l; i++) {
      len += chunks[i].length;
    }
    const result = new Uint8Array(len);
    for (let i = 0, pos = 0, l = chunks.length; i < l; i++) {
      let chunk = chunks[i];
      result.set(chunk, pos);
      pos += chunk.length;
    }
    return result;
  };
  var common = {
    assign,
    flattenChunks
  };
  var STR_APPLY_UIA_OK = true;
  try {
    String.fromCharCode.apply(null, new Uint8Array(1));
  } catch (__) {
    STR_APPLY_UIA_OK = false;
  }
  var _utf8len = new Uint8Array(256);
  for (let q = 0; q < 256; q++) {
    _utf8len[q] = q >= 252 ? 6 : q >= 248 ? 5 : q >= 240 ? 4 : q >= 224 ? 3 : q >= 192 ? 2 : 1;
  }
  _utf8len[254] = _utf8len[254] = 1;
  var string2buf = (str) => {
    if (typeof TextEncoder === "function" && TextEncoder.prototype.encode) {
      return new TextEncoder().encode(str);
    }
    let buf, c, c2, m_pos, i, str_len = str.length, buf_len = 0;
    for (m_pos = 0; m_pos < str_len; m_pos++) {
      c = str.charCodeAt(m_pos);
      if ((c & 64512) === 55296 && m_pos + 1 < str_len) {
        c2 = str.charCodeAt(m_pos + 1);
        if ((c2 & 64512) === 56320) {
          c = 65536 + (c - 55296 << 10) + (c2 - 56320);
          m_pos++;
        }
      }
      buf_len += c < 128 ? 1 : c < 2048 ? 2 : c < 65536 ? 3 : 4;
    }
    buf = new Uint8Array(buf_len);
    for (i = 0, m_pos = 0; i < buf_len; m_pos++) {
      c = str.charCodeAt(m_pos);
      if ((c & 64512) === 55296 && m_pos + 1 < str_len) {
        c2 = str.charCodeAt(m_pos + 1);
        if ((c2 & 64512) === 56320) {
          c = 65536 + (c - 55296 << 10) + (c2 - 56320);
          m_pos++;
        }
      }
      if (c < 128) {
        buf[i++] = c;
      } else if (c < 2048) {
        buf[i++] = 192 | c >>> 6;
        buf[i++] = 128 | c & 63;
      } else if (c < 65536) {
        buf[i++] = 224 | c >>> 12;
        buf[i++] = 128 | c >>> 6 & 63;
        buf[i++] = 128 | c & 63;
      } else {
        buf[i++] = 240 | c >>> 18;
        buf[i++] = 128 | c >>> 12 & 63;
        buf[i++] = 128 | c >>> 6 & 63;
        buf[i++] = 128 | c & 63;
      }
    }
    return buf;
  };
  var buf2binstring = (buf, len) => {
    if (len < 65534) {
      if (buf.subarray && STR_APPLY_UIA_OK) {
        return String.fromCharCode.apply(null, buf.length === len ? buf : buf.subarray(0, len));
      }
    }
    let result = "";
    for (let i = 0; i < len; i++) {
      result += String.fromCharCode(buf[i]);
    }
    return result;
  };
  var buf2string = (buf, max) => {
    const len = max || buf.length;
    if (typeof TextDecoder === "function" && TextDecoder.prototype.decode) {
      return new TextDecoder().decode(buf.subarray(0, max));
    }
    let i, out;
    const utf16buf = new Array(len * 2);
    for (out = 0, i = 0; i < len; ) {
      let c = buf[i++];
      if (c < 128) {
        utf16buf[out++] = c;
        continue;
      }
      let c_len = _utf8len[c];
      if (c_len > 4) {
        utf16buf[out++] = 65533;
        i += c_len - 1;
        continue;
      }
      c &= c_len === 2 ? 31 : c_len === 3 ? 15 : 7;
      while (c_len > 1 && i < len) {
        c = c << 6 | buf[i++] & 63;
        c_len--;
      }
      if (c_len > 1) {
        utf16buf[out++] = 65533;
        continue;
      }
      if (c < 65536) {
        utf16buf[out++] = c;
      } else {
        c -= 65536;
        utf16buf[out++] = 55296 | c >> 10 & 1023;
        utf16buf[out++] = 56320 | c & 1023;
      }
    }
    return buf2binstring(utf16buf, out);
  };
  var utf8border = (buf, max) => {
    max = max || buf.length;
    if (max > buf.length) {
      max = buf.length;
    }
    let pos = max - 1;
    while (pos >= 0 && (buf[pos] & 192) === 128) {
      pos--;
    }
    if (pos < 0) {
      return max;
    }
    if (pos === 0) {
      return max;
    }
    return pos + _utf8len[buf[pos]] > max ? pos : max;
  };
  var strings = {
    string2buf,
    buf2string,
    utf8border
  };
  function ZStream() {
    this.input = null;
    this.next_in = 0;
    this.avail_in = 0;
    this.total_in = 0;
    this.output = null;
    this.next_out = 0;
    this.avail_out = 0;
    this.total_out = 0;
    this.msg = "";
    this.state = null;
    this.data_type = 2;
    this.adler = 0;
  }
  var zstream = ZStream;
  var toString$1 = Object.prototype.toString;
  var {
    Z_NO_FLUSH: Z_NO_FLUSH$1,
    Z_SYNC_FLUSH,
    Z_FULL_FLUSH,
    Z_FINISH: Z_FINISH$2,
    Z_OK: Z_OK$2,
    Z_STREAM_END: Z_STREAM_END$2,
    Z_DEFAULT_COMPRESSION,
    Z_DEFAULT_STRATEGY,
    Z_DEFLATED: Z_DEFLATED$1
  } = constants$2;
  function Deflate$1(options) {
    this.options = common.assign({
      level: Z_DEFAULT_COMPRESSION,
      method: Z_DEFLATED$1,
      chunkSize: 16384,
      windowBits: 15,
      memLevel: 8,
      strategy: Z_DEFAULT_STRATEGY
    }, options || {});
    let opt = this.options;
    if (opt.raw && opt.windowBits > 0) {
      opt.windowBits = -opt.windowBits;
    } else if (opt.gzip && opt.windowBits > 0 && opt.windowBits < 16) {
      opt.windowBits += 16;
    }
    this.err = 0;
    this.msg = "";
    this.ended = false;
    this.chunks = [];
    this.strm = new zstream();
    this.strm.avail_out = 0;
    let status = deflate_1$2.deflateInit2(
      this.strm,
      opt.level,
      opt.method,
      opt.windowBits,
      opt.memLevel,
      opt.strategy
    );
    if (status !== Z_OK$2) {
      throw new Error(messages[status]);
    }
    if (opt.header) {
      deflate_1$2.deflateSetHeader(this.strm, opt.header);
    }
    if (opt.dictionary) {
      let dict;
      if (typeof opt.dictionary === "string") {
        dict = strings.string2buf(opt.dictionary);
      } else if (toString$1.call(opt.dictionary) === "[object ArrayBuffer]") {
        dict = new Uint8Array(opt.dictionary);
      } else {
        dict = opt.dictionary;
      }
      status = deflate_1$2.deflateSetDictionary(this.strm, dict);
      if (status !== Z_OK$2) {
        throw new Error(messages[status]);
      }
      this._dict_set = true;
    }
  }
  Deflate$1.prototype.push = function(data, flush_mode) {
    const strm = this.strm;
    const chunkSize = this.options.chunkSize;
    let status, _flush_mode;
    if (this.ended) {
      return false;
    }
    if (flush_mode === ~~flush_mode) _flush_mode = flush_mode;
    else _flush_mode = flush_mode === true ? Z_FINISH$2 : Z_NO_FLUSH$1;
    if (typeof data === "string") {
      strm.input = strings.string2buf(data);
    } else if (toString$1.call(data) === "[object ArrayBuffer]") {
      strm.input = new Uint8Array(data);
    } else {
      strm.input = data;
    }
    strm.next_in = 0;
    strm.avail_in = strm.input.length;
    for (; ; ) {
      if (strm.avail_out === 0) {
        strm.output = new Uint8Array(chunkSize);
        strm.next_out = 0;
        strm.avail_out = chunkSize;
      }
      if ((_flush_mode === Z_SYNC_FLUSH || _flush_mode === Z_FULL_FLUSH) && strm.avail_out <= 6) {
        this.onData(strm.output.subarray(0, strm.next_out));
        strm.avail_out = 0;
        continue;
      }
      status = deflate_1$2.deflate(strm, _flush_mode);
      if (status === Z_STREAM_END$2) {
        if (strm.next_out > 0) {
          this.onData(strm.output.subarray(0, strm.next_out));
        }
        status = deflate_1$2.deflateEnd(this.strm);
        this.onEnd(status);
        this.ended = true;
        return status === Z_OK$2;
      }
      if (strm.avail_out === 0) {
        this.onData(strm.output);
        continue;
      }
      if (_flush_mode > 0 && strm.next_out > 0) {
        this.onData(strm.output.subarray(0, strm.next_out));
        strm.avail_out = 0;
        continue;
      }
      if (strm.avail_in === 0) break;
    }
    return true;
  };
  Deflate$1.prototype.onData = function(chunk) {
    this.chunks.push(chunk);
  };
  Deflate$1.prototype.onEnd = function(status) {
    if (status === Z_OK$2) {
      this.result = common.flattenChunks(this.chunks);
    }
    this.chunks = [];
    this.err = status;
    this.msg = this.strm.msg;
  };
  function deflate$1(input, options) {
    const deflator = new Deflate$1(options);
    deflator.push(input, true);
    if (deflator.err) {
      throw deflator.msg || messages[deflator.err];
    }
    return deflator.result;
  }
  function deflateRaw$1(input, options) {
    options = options || {};
    options.raw = true;
    return deflate$1(input, options);
  }
  function gzip$1(input, options) {
    options = options || {};
    options.gzip = true;
    return deflate$1(input, options);
  }
  var Deflate_1$1 = Deflate$1;
  var deflate_2 = deflate$1;
  var deflateRaw_1$1 = deflateRaw$1;
  var gzip_1$1 = gzip$1;
  var constants$1 = constants$2;
  var deflate_1$1 = {
    Deflate: Deflate_1$1,
    deflate: deflate_2,
    deflateRaw: deflateRaw_1$1,
    gzip: gzip_1$1,
    constants: constants$1
  };
  var BAD$1 = 16209;
  var TYPE$1 = 16191;
  var inffast = function inflate_fast(strm, start) {
    let _in;
    let last;
    let _out;
    let beg;
    let end;
    let dmax;
    let wsize;
    let whave;
    let wnext;
    let s_window;
    let hold;
    let bits;
    let lcode;
    let dcode;
    let lmask;
    let dmask;
    let here;
    let op;
    let len;
    let dist;
    let from;
    let from_source;
    let input, output;
    const state = strm.state;
    _in = strm.next_in;
    input = strm.input;
    last = _in + (strm.avail_in - 5);
    _out = strm.next_out;
    output = strm.output;
    beg = _out - (start - strm.avail_out);
    end = _out + (strm.avail_out - 257);
    dmax = state.dmax;
    wsize = state.wsize;
    whave = state.whave;
    wnext = state.wnext;
    s_window = state.window;
    hold = state.hold;
    bits = state.bits;
    lcode = state.lencode;
    dcode = state.distcode;
    lmask = (1 << state.lenbits) - 1;
    dmask = (1 << state.distbits) - 1;
    top:
      do {
        if (bits < 15) {
          hold += input[_in++] << bits;
          bits += 8;
          hold += input[_in++] << bits;
          bits += 8;
        }
        here = lcode[hold & lmask];
        dolen:
          for (; ; ) {
            op = here >>> 24;
            hold >>>= op;
            bits -= op;
            op = here >>> 16 & 255;
            if (op === 0) {
              output[_out++] = here & 65535;
            } else if (op & 16) {
              len = here & 65535;
              op &= 15;
              if (op) {
                if (bits < op) {
                  hold += input[_in++] << bits;
                  bits += 8;
                }
                len += hold & (1 << op) - 1;
                hold >>>= op;
                bits -= op;
              }
              if (bits < 15) {
                hold += input[_in++] << bits;
                bits += 8;
                hold += input[_in++] << bits;
                bits += 8;
              }
              here = dcode[hold & dmask];
              dodist:
                for (; ; ) {
                  op = here >>> 24;
                  hold >>>= op;
                  bits -= op;
                  op = here >>> 16 & 255;
                  if (op & 16) {
                    dist = here & 65535;
                    op &= 15;
                    if (bits < op) {
                      hold += input[_in++] << bits;
                      bits += 8;
                      if (bits < op) {
                        hold += input[_in++] << bits;
                        bits += 8;
                      }
                    }
                    dist += hold & (1 << op) - 1;
                    if (dist > dmax) {
                      strm.msg = "invalid distance too far back";
                      state.mode = BAD$1;
                      break top;
                    }
                    hold >>>= op;
                    bits -= op;
                    op = _out - beg;
                    if (dist > op) {
                      op = dist - op;
                      if (op > whave) {
                        if (state.sane) {
                          strm.msg = "invalid distance too far back";
                          state.mode = BAD$1;
                          break top;
                        }
                      }
                      from = 0;
                      from_source = s_window;
                      if (wnext === 0) {
                        from += wsize - op;
                        if (op < len) {
                          len -= op;
                          do {
                            output[_out++] = s_window[from++];
                          } while (--op);
                          from = _out - dist;
                          from_source = output;
                        }
                      } else if (wnext < op) {
                        from += wsize + wnext - op;
                        op -= wnext;
                        if (op < len) {
                          len -= op;
                          do {
                            output[_out++] = s_window[from++];
                          } while (--op);
                          from = 0;
                          if (wnext < len) {
                            op = wnext;
                            len -= op;
                            do {
                              output[_out++] = s_window[from++];
                            } while (--op);
                            from = _out - dist;
                            from_source = output;
                          }
                        }
                      } else {
                        from += wnext - op;
                        if (op < len) {
                          len -= op;
                          do {
                            output[_out++] = s_window[from++];
                          } while (--op);
                          from = _out - dist;
                          from_source = output;
                        }
                      }
                      while (len > 2) {
                        output[_out++] = from_source[from++];
                        output[_out++] = from_source[from++];
                        output[_out++] = from_source[from++];
                        len -= 3;
                      }
                      if (len) {
                        output[_out++] = from_source[from++];
                        if (len > 1) {
                          output[_out++] = from_source[from++];
                        }
                      }
                    } else {
                      from = _out - dist;
                      do {
                        output[_out++] = output[from++];
                        output[_out++] = output[from++];
                        output[_out++] = output[from++];
                        len -= 3;
                      } while (len > 2);
                      if (len) {
                        output[_out++] = output[from++];
                        if (len > 1) {
                          output[_out++] = output[from++];
                        }
                      }
                    }
                  } else if ((op & 64) === 0) {
                    here = dcode[(here & 65535) + (hold & (1 << op) - 1)];
                    continue dodist;
                  } else {
                    strm.msg = "invalid distance code";
                    state.mode = BAD$1;
                    break top;
                  }
                  break;
                }
            } else if ((op & 64) === 0) {
              here = lcode[(here & 65535) + (hold & (1 << op) - 1)];
              continue dolen;
            } else if (op & 32) {
              state.mode = TYPE$1;
              break top;
            } else {
              strm.msg = "invalid literal/length code";
              state.mode = BAD$1;
              break top;
            }
            break;
          }
      } while (_in < last && _out < end);
    len = bits >> 3;
    _in -= len;
    bits -= len << 3;
    hold &= (1 << bits) - 1;
    strm.next_in = _in;
    strm.next_out = _out;
    strm.avail_in = _in < last ? 5 + (last - _in) : 5 - (_in - last);
    strm.avail_out = _out < end ? 257 + (end - _out) : 257 - (_out - end);
    state.hold = hold;
    state.bits = bits;
    return;
  };
  var MAXBITS = 15;
  var ENOUGH_LENS$1 = 852;
  var ENOUGH_DISTS$1 = 592;
  var CODES$1 = 0;
  var LENS$1 = 1;
  var DISTS$1 = 2;
  var lbase = new Uint16Array([
    /* Length codes 257..285 base */
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    15,
    17,
    19,
    23,
    27,
    31,
    35,
    43,
    51,
    59,
    67,
    83,
    99,
    115,
    131,
    163,
    195,
    227,
    258,
    0,
    0
  ]);
  var lext = new Uint8Array([
    /* Length codes 257..285 extra */
    16,
    16,
    16,
    16,
    16,
    16,
    16,
    16,
    17,
    17,
    17,
    17,
    18,
    18,
    18,
    18,
    19,
    19,
    19,
    19,
    20,
    20,
    20,
    20,
    21,
    21,
    21,
    21,
    16,
    72,
    78
  ]);
  var dbase = new Uint16Array([
    /* Distance codes 0..29 base */
    1,
    2,
    3,
    4,
    5,
    7,
    9,
    13,
    17,
    25,
    33,
    49,
    65,
    97,
    129,
    193,
    257,
    385,
    513,
    769,
    1025,
    1537,
    2049,
    3073,
    4097,
    6145,
    8193,
    12289,
    16385,
    24577,
    0,
    0
  ]);
  var dext = new Uint8Array([
    /* Distance codes 0..29 extra */
    16,
    16,
    16,
    16,
    17,
    17,
    18,
    18,
    19,
    19,
    20,
    20,
    21,
    21,
    22,
    22,
    23,
    23,
    24,
    24,
    25,
    25,
    26,
    26,
    27,
    27,
    28,
    28,
    29,
    29,
    64,
    64
  ]);
  var inflate_table = (type, lens, lens_index, codes, table, table_index, work, opts) => {
    const bits = opts.bits;
    let len = 0;
    let sym = 0;
    let min = 0, max = 0;
    let root = 0;
    let curr = 0;
    let drop = 0;
    let left = 0;
    let used = 0;
    let huff = 0;
    let incr;
    let fill;
    let low;
    let mask;
    let next;
    let base = null;
    let match;
    const count = new Uint16Array(MAXBITS + 1);
    const offs = new Uint16Array(MAXBITS + 1);
    let extra = null;
    let here_bits, here_op, here_val;
    for (len = 0; len <= MAXBITS; len++) {
      count[len] = 0;
    }
    for (sym = 0; sym < codes; sym++) {
      count[lens[lens_index + sym]]++;
    }
    root = bits;
    for (max = MAXBITS; max >= 1; max--) {
      if (count[max] !== 0) {
        break;
      }
    }
    if (root > max) {
      root = max;
    }
    if (max === 0) {
      table[table_index++] = 1 << 24 | 64 << 16 | 0;
      table[table_index++] = 1 << 24 | 64 << 16 | 0;
      opts.bits = 1;
      return 0;
    }
    for (min = 1; min < max; min++) {
      if (count[min] !== 0) {
        break;
      }
    }
    if (root < min) {
      root = min;
    }
    left = 1;
    for (len = 1; len <= MAXBITS; len++) {
      left <<= 1;
      left -= count[len];
      if (left < 0) {
        return -1;
      }
    }
    if (left > 0 && (type === CODES$1 || max !== 1)) {
      return -1;
    }
    offs[1] = 0;
    for (len = 1; len < MAXBITS; len++) {
      offs[len + 1] = offs[len] + count[len];
    }
    for (sym = 0; sym < codes; sym++) {
      if (lens[lens_index + sym] !== 0) {
        work[offs[lens[lens_index + sym]]++] = sym;
      }
    }
    if (type === CODES$1) {
      base = extra = work;
      match = 20;
    } else if (type === LENS$1) {
      base = lbase;
      extra = lext;
      match = 257;
    } else {
      base = dbase;
      extra = dext;
      match = 0;
    }
    huff = 0;
    sym = 0;
    len = min;
    next = table_index;
    curr = root;
    drop = 0;
    low = -1;
    used = 1 << root;
    mask = used - 1;
    if (type === LENS$1 && used > ENOUGH_LENS$1 || type === DISTS$1 && used > ENOUGH_DISTS$1) {
      return 1;
    }
    for (; ; ) {
      here_bits = len - drop;
      if (work[sym] + 1 < match) {
        here_op = 0;
        here_val = work[sym];
      } else if (work[sym] >= match) {
        here_op = extra[work[sym] - match];
        here_val = base[work[sym] - match];
      } else {
        here_op = 32 + 64;
        here_val = 0;
      }
      incr = 1 << len - drop;
      fill = 1 << curr;
      min = fill;
      do {
        fill -= incr;
        table[next + (huff >> drop) + fill] = here_bits << 24 | here_op << 16 | here_val | 0;
      } while (fill !== 0);
      incr = 1 << len - 1;
      while (huff & incr) {
        incr >>= 1;
      }
      if (incr !== 0) {
        huff &= incr - 1;
        huff += incr;
      } else {
        huff = 0;
      }
      sym++;
      if (--count[len] === 0) {
        if (len === max) {
          break;
        }
        len = lens[lens_index + work[sym]];
      }
      if (len > root && (huff & mask) !== low) {
        if (drop === 0) {
          drop = root;
        }
        next += min;
        curr = len - drop;
        left = 1 << curr;
        while (curr + drop < max) {
          left -= count[curr + drop];
          if (left <= 0) {
            break;
          }
          curr++;
          left <<= 1;
        }
        used += 1 << curr;
        if (type === LENS$1 && used > ENOUGH_LENS$1 || type === DISTS$1 && used > ENOUGH_DISTS$1) {
          return 1;
        }
        low = huff & mask;
        table[low] = root << 24 | curr << 16 | next - table_index | 0;
      }
    }
    if (huff !== 0) {
      table[next + huff] = len - drop << 24 | 64 << 16 | 0;
    }
    opts.bits = root;
    return 0;
  };
  var inftrees = inflate_table;
  var CODES = 0;
  var LENS = 1;
  var DISTS = 2;
  var {
    Z_FINISH: Z_FINISH$1,
    Z_BLOCK,
    Z_TREES,
    Z_OK: Z_OK$1,
    Z_STREAM_END: Z_STREAM_END$1,
    Z_NEED_DICT: Z_NEED_DICT$1,
    Z_STREAM_ERROR: Z_STREAM_ERROR$1,
    Z_DATA_ERROR: Z_DATA_ERROR$1,
    Z_MEM_ERROR: Z_MEM_ERROR$1,
    Z_BUF_ERROR,
    Z_DEFLATED
  } = constants$2;
  var HEAD = 16180;
  var FLAGS = 16181;
  var TIME = 16182;
  var OS = 16183;
  var EXLEN = 16184;
  var EXTRA = 16185;
  var NAME = 16186;
  var COMMENT = 16187;
  var HCRC = 16188;
  var DICTID = 16189;
  var DICT = 16190;
  var TYPE = 16191;
  var TYPEDO = 16192;
  var STORED = 16193;
  var COPY_ = 16194;
  var COPY = 16195;
  var TABLE = 16196;
  var LENLENS = 16197;
  var CODELENS = 16198;
  var LEN_ = 16199;
  var LEN = 16200;
  var LENEXT = 16201;
  var DIST = 16202;
  var DISTEXT = 16203;
  var MATCH = 16204;
  var LIT = 16205;
  var CHECK = 16206;
  var LENGTH = 16207;
  var DONE = 16208;
  var BAD = 16209;
  var MEM = 16210;
  var SYNC = 16211;
  var ENOUGH_LENS = 852;
  var ENOUGH_DISTS = 592;
  var MAX_WBITS = 15;
  var DEF_WBITS = MAX_WBITS;
  var zswap32 = (q) => {
    return (q >>> 24 & 255) + (q >>> 8 & 65280) + ((q & 65280) << 8) + ((q & 255) << 24);
  };
  function InflateState() {
    this.strm = null;
    this.mode = 0;
    this.last = false;
    this.wrap = 0;
    this.havedict = false;
    this.flags = 0;
    this.dmax = 0;
    this.check = 0;
    this.total = 0;
    this.head = null;
    this.wbits = 0;
    this.wsize = 0;
    this.whave = 0;
    this.wnext = 0;
    this.window = null;
    this.hold = 0;
    this.bits = 0;
    this.length = 0;
    this.offset = 0;
    this.extra = 0;
    this.lencode = null;
    this.distcode = null;
    this.lenbits = 0;
    this.distbits = 0;
    this.ncode = 0;
    this.nlen = 0;
    this.ndist = 0;
    this.have = 0;
    this.next = null;
    this.lens = new Uint16Array(320);
    this.work = new Uint16Array(288);
    this.lendyn = null;
    this.distdyn = null;
    this.sane = 0;
    this.back = 0;
    this.was = 0;
  }
  var inflateStateCheck = (strm) => {
    if (!strm) {
      return 1;
    }
    const state = strm.state;
    if (!state || state.strm !== strm || state.mode < HEAD || state.mode > SYNC) {
      return 1;
    }
    return 0;
  };
  var inflateResetKeep = (strm) => {
    if (inflateStateCheck(strm)) {
      return Z_STREAM_ERROR$1;
    }
    const state = strm.state;
    strm.total_in = strm.total_out = state.total = 0;
    strm.msg = "";
    if (state.wrap) {
      strm.adler = state.wrap & 1;
    }
    state.mode = HEAD;
    state.last = 0;
    state.havedict = 0;
    state.flags = -1;
    state.dmax = 32768;
    state.head = null;
    state.hold = 0;
    state.bits = 0;
    state.lencode = state.lendyn = new Int32Array(ENOUGH_LENS);
    state.distcode = state.distdyn = new Int32Array(ENOUGH_DISTS);
    state.sane = 1;
    state.back = -1;
    return Z_OK$1;
  };
  var inflateReset = (strm) => {
    if (inflateStateCheck(strm)) {
      return Z_STREAM_ERROR$1;
    }
    const state = strm.state;
    state.wsize = 0;
    state.whave = 0;
    state.wnext = 0;
    return inflateResetKeep(strm);
  };
  var inflateReset2 = (strm, windowBits) => {
    let wrap;
    if (inflateStateCheck(strm)) {
      return Z_STREAM_ERROR$1;
    }
    const state = strm.state;
    if (windowBits < 0) {
      wrap = 0;
      windowBits = -windowBits;
    } else {
      wrap = (windowBits >> 4) + 5;
      if (windowBits < 48) {
        windowBits &= 15;
      }
    }
    if (windowBits && (windowBits < 8 || windowBits > 15)) {
      return Z_STREAM_ERROR$1;
    }
    if (state.window !== null && state.wbits !== windowBits) {
      state.window = null;
    }
    state.wrap = wrap;
    state.wbits = windowBits;
    return inflateReset(strm);
  };
  var inflateInit2 = (strm, windowBits) => {
    if (!strm) {
      return Z_STREAM_ERROR$1;
    }
    const state = new InflateState();
    strm.state = state;
    state.strm = strm;
    state.window = null;
    state.mode = HEAD;
    const ret = inflateReset2(strm, windowBits);
    if (ret !== Z_OK$1) {
      strm.state = null;
    }
    return ret;
  };
  var inflateInit = (strm) => {
    return inflateInit2(strm, DEF_WBITS);
  };
  var virgin = true;
  var lenfix;
  var distfix;
  var fixedtables = (state) => {
    if (virgin) {
      lenfix = new Int32Array(512);
      distfix = new Int32Array(32);
      let sym = 0;
      while (sym < 144) {
        state.lens[sym++] = 8;
      }
      while (sym < 256) {
        state.lens[sym++] = 9;
      }
      while (sym < 280) {
        state.lens[sym++] = 7;
      }
      while (sym < 288) {
        state.lens[sym++] = 8;
      }
      inftrees(LENS, state.lens, 0, 288, lenfix, 0, state.work, { bits: 9 });
      sym = 0;
      while (sym < 32) {
        state.lens[sym++] = 5;
      }
      inftrees(DISTS, state.lens, 0, 32, distfix, 0, state.work, { bits: 5 });
      virgin = false;
    }
    state.lencode = lenfix;
    state.lenbits = 9;
    state.distcode = distfix;
    state.distbits = 5;
  };
  var updatewindow = (strm, src, end, copy) => {
    let dist;
    const state = strm.state;
    if (state.window === null) {
      state.wsize = 1 << state.wbits;
      state.wnext = 0;
      state.whave = 0;
      state.window = new Uint8Array(state.wsize);
    }
    if (copy >= state.wsize) {
      state.window.set(src.subarray(end - state.wsize, end), 0);
      state.wnext = 0;
      state.whave = state.wsize;
    } else {
      dist = state.wsize - state.wnext;
      if (dist > copy) {
        dist = copy;
      }
      state.window.set(src.subarray(end - copy, end - copy + dist), state.wnext);
      copy -= dist;
      if (copy) {
        state.window.set(src.subarray(end - copy, end), 0);
        state.wnext = copy;
        state.whave = state.wsize;
      } else {
        state.wnext += dist;
        if (state.wnext === state.wsize) {
          state.wnext = 0;
        }
        if (state.whave < state.wsize) {
          state.whave += dist;
        }
      }
    }
    return 0;
  };
  var inflate$2 = (strm, flush) => {
    let state;
    let input, output;
    let next;
    let put;
    let have, left;
    let hold;
    let bits;
    let _in, _out;
    let copy;
    let from;
    let from_source;
    let here = 0;
    let here_bits, here_op, here_val;
    let last_bits, last_op, last_val;
    let len;
    let ret;
    const hbuf = new Uint8Array(4);
    let opts;
    let n;
    const order = (
      /* permutation of code lengths */
      new Uint8Array([16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15])
    );
    if (inflateStateCheck(strm) || !strm.output || !strm.input && strm.avail_in !== 0) {
      return Z_STREAM_ERROR$1;
    }
    state = strm.state;
    if (state.mode === TYPE) {
      state.mode = TYPEDO;
    }
    put = strm.next_out;
    output = strm.output;
    left = strm.avail_out;
    next = strm.next_in;
    input = strm.input;
    have = strm.avail_in;
    hold = state.hold;
    bits = state.bits;
    _in = have;
    _out = left;
    ret = Z_OK$1;
    inf_leave:
      for (; ; ) {
        switch (state.mode) {
          case HEAD:
            if (state.wrap === 0) {
              state.mode = TYPEDO;
              break;
            }
            while (bits < 16) {
              if (have === 0) {
                break inf_leave;
              }
              have--;
              hold += input[next++] << bits;
              bits += 8;
            }
            if (state.wrap & 2 && hold === 35615) {
              if (state.wbits === 0) {
                state.wbits = 15;
              }
              state.check = 0;
              hbuf[0] = hold & 255;
              hbuf[1] = hold >>> 8 & 255;
              state.check = crc32_1(state.check, hbuf, 2, 0);
              hold = 0;
              bits = 0;
              state.mode = FLAGS;
              break;
            }
            if (state.head) {
              state.head.done = false;
            }
            if (!(state.wrap & 1) || /* check if zlib header allowed */
            (((hold & 255) << 8) + (hold >> 8)) % 31) {
              strm.msg = "incorrect header check";
              state.mode = BAD;
              break;
            }
            if ((hold & 15) !== Z_DEFLATED) {
              strm.msg = "unknown compression method";
              state.mode = BAD;
              break;
            }
            hold >>>= 4;
            bits -= 4;
            len = (hold & 15) + 8;
            if (state.wbits === 0) {
              state.wbits = len;
            }
            if (len > 15 || len > state.wbits) {
              strm.msg = "invalid window size";
              state.mode = BAD;
              break;
            }
            state.dmax = 1 << state.wbits;
            state.flags = 0;
            strm.adler = state.check = 1;
            state.mode = hold & 512 ? DICTID : TYPE;
            hold = 0;
            bits = 0;
            break;
          case FLAGS:
            while (bits < 16) {
              if (have === 0) {
                break inf_leave;
              }
              have--;
              hold += input[next++] << bits;
              bits += 8;
            }
            state.flags = hold;
            if ((state.flags & 255) !== Z_DEFLATED) {
              strm.msg = "unknown compression method";
              state.mode = BAD;
              break;
            }
            if (state.flags & 57344) {
              strm.msg = "unknown header flags set";
              state.mode = BAD;
              break;
            }
            if (state.head) {
              state.head.text = hold >> 8 & 1;
            }
            if (state.flags & 512 && state.wrap & 4) {
              hbuf[0] = hold & 255;
              hbuf[1] = hold >>> 8 & 255;
              state.check = crc32_1(state.check, hbuf, 2, 0);
            }
            hold = 0;
            bits = 0;
            state.mode = TIME;
          /* falls through */
          case TIME:
            while (bits < 32) {
              if (have === 0) {
                break inf_leave;
              }
              have--;
              hold += input[next++] << bits;
              bits += 8;
            }
            if (state.head) {
              state.head.time = hold;
            }
            if (state.flags & 512 && state.wrap & 4) {
              hbuf[0] = hold & 255;
              hbuf[1] = hold >>> 8 & 255;
              hbuf[2] = hold >>> 16 & 255;
              hbuf[3] = hold >>> 24 & 255;
              state.check = crc32_1(state.check, hbuf, 4, 0);
            }
            hold = 0;
            bits = 0;
            state.mode = OS;
          /* falls through */
          case OS:
            while (bits < 16) {
              if (have === 0) {
                break inf_leave;
              }
              have--;
              hold += input[next++] << bits;
              bits += 8;
            }
            if (state.head) {
              state.head.xflags = hold & 255;
              state.head.os = hold >> 8;
            }
            if (state.flags & 512 && state.wrap & 4) {
              hbuf[0] = hold & 255;
              hbuf[1] = hold >>> 8 & 255;
              state.check = crc32_1(state.check, hbuf, 2, 0);
            }
            hold = 0;
            bits = 0;
            state.mode = EXLEN;
          /* falls through */
          case EXLEN:
            if (state.flags & 1024) {
              while (bits < 16) {
                if (have === 0) {
                  break inf_leave;
                }
                have--;
                hold += input[next++] << bits;
                bits += 8;
              }
              state.length = hold;
              if (state.head) {
                state.head.extra_len = hold;
              }
              if (state.flags & 512 && state.wrap & 4) {
                hbuf[0] = hold & 255;
                hbuf[1] = hold >>> 8 & 255;
                state.check = crc32_1(state.check, hbuf, 2, 0);
              }
              hold = 0;
              bits = 0;
            } else if (state.head) {
              state.head.extra = null;
            }
            state.mode = EXTRA;
          /* falls through */
          case EXTRA:
            if (state.flags & 1024) {
              copy = state.length;
              if (copy > have) {
                copy = have;
              }
              if (copy) {
                if (state.head) {
                  len = state.head.extra_len - state.length;
                  if (!state.head.extra) {
                    state.head.extra = new Uint8Array(state.head.extra_len);
                  }
                  state.head.extra.set(
                    input.subarray(
                      next,
                      // extra field is limited to 65536 bytes
                      // - no need for additional size check
                      next + copy
                    ),
                    /*len + copy > state.head.extra_max - len ? state.head.extra_max : copy,*/
                    len
                  );
                }
                if (state.flags & 512 && state.wrap & 4) {
                  state.check = crc32_1(state.check, input, copy, next);
                }
                have -= copy;
                next += copy;
                state.length -= copy;
              }
              if (state.length) {
                break inf_leave;
              }
            }
            state.length = 0;
            state.mode = NAME;
          /* falls through */
          case NAME:
            if (state.flags & 2048) {
              if (have === 0) {
                break inf_leave;
              }
              copy = 0;
              do {
                len = input[next + copy++];
                if (state.head && len && state.length < 65536) {
                  state.head.name += String.fromCharCode(len);
                }
              } while (len && copy < have);
              if (state.flags & 512 && state.wrap & 4) {
                state.check = crc32_1(state.check, input, copy, next);
              }
              have -= copy;
              next += copy;
              if (len) {
                break inf_leave;
              }
            } else if (state.head) {
              state.head.name = null;
            }
            state.length = 0;
            state.mode = COMMENT;
          /* falls through */
          case COMMENT:
            if (state.flags & 4096) {
              if (have === 0) {
                break inf_leave;
              }
              copy = 0;
              do {
                len = input[next + copy++];
                if (state.head && len && state.length < 65536) {
                  state.head.comment += String.fromCharCode(len);
                }
              } while (len && copy < have);
              if (state.flags & 512 && state.wrap & 4) {
                state.check = crc32_1(state.check, input, copy, next);
              }
              have -= copy;
              next += copy;
              if (len) {
                break inf_leave;
              }
            } else if (state.head) {
              state.head.comment = null;
            }
            state.mode = HCRC;
          /* falls through */
          case HCRC:
            if (state.flags & 512) {
              while (bits < 16) {
                if (have === 0) {
                  break inf_leave;
                }
                have--;
                hold += input[next++] << bits;
                bits += 8;
              }
              if (state.wrap & 4 && hold !== (state.check & 65535)) {
                strm.msg = "header crc mismatch";
                state.mode = BAD;
                break;
              }
              hold = 0;
              bits = 0;
            }
            if (state.head) {
              state.head.hcrc = state.flags >> 9 & 1;
              state.head.done = true;
            }
            strm.adler = state.check = 0;
            state.mode = TYPE;
            break;
          case DICTID:
            while (bits < 32) {
              if (have === 0) {
                break inf_leave;
              }
              have--;
              hold += input[next++] << bits;
              bits += 8;
            }
            strm.adler = state.check = zswap32(hold);
            hold = 0;
            bits = 0;
            state.mode = DICT;
          /* falls through */
          case DICT:
            if (state.havedict === 0) {
              strm.next_out = put;
              strm.avail_out = left;
              strm.next_in = next;
              strm.avail_in = have;
              state.hold = hold;
              state.bits = bits;
              return Z_NEED_DICT$1;
            }
            strm.adler = state.check = 1;
            state.mode = TYPE;
          /* falls through */
          case TYPE:
            if (flush === Z_BLOCK || flush === Z_TREES) {
              break inf_leave;
            }
          /* falls through */
          case TYPEDO:
            if (state.last) {
              hold >>>= bits & 7;
              bits -= bits & 7;
              state.mode = CHECK;
              break;
            }
            while (bits < 3) {
              if (have === 0) {
                break inf_leave;
              }
              have--;
              hold += input[next++] << bits;
              bits += 8;
            }
            state.last = hold & 1;
            hold >>>= 1;
            bits -= 1;
            switch (hold & 3) {
              case 0:
                state.mode = STORED;
                break;
              case 1:
                fixedtables(state);
                state.mode = LEN_;
                if (flush === Z_TREES) {
                  hold >>>= 2;
                  bits -= 2;
                  break inf_leave;
                }
                break;
              case 2:
                state.mode = TABLE;
                break;
              case 3:
                strm.msg = "invalid block type";
                state.mode = BAD;
            }
            hold >>>= 2;
            bits -= 2;
            break;
          case STORED:
            hold >>>= bits & 7;
            bits -= bits & 7;
            while (bits < 32) {
              if (have === 0) {
                break inf_leave;
              }
              have--;
              hold += input[next++] << bits;
              bits += 8;
            }
            if ((hold & 65535) !== (hold >>> 16 ^ 65535)) {
              strm.msg = "invalid stored block lengths";
              state.mode = BAD;
              break;
            }
            state.length = hold & 65535;
            hold = 0;
            bits = 0;
            state.mode = COPY_;
            if (flush === Z_TREES) {
              break inf_leave;
            }
          /* falls through */
          case COPY_:
            state.mode = COPY;
          /* falls through */
          case COPY:
            copy = state.length;
            if (copy) {
              if (copy > have) {
                copy = have;
              }
              if (copy > left) {
                copy = left;
              }
              if (copy === 0) {
                break inf_leave;
              }
              output.set(input.subarray(next, next + copy), put);
              have -= copy;
              next += copy;
              left -= copy;
              put += copy;
              state.length -= copy;
              break;
            }
            state.mode = TYPE;
            break;
          case TABLE:
            while (bits < 14) {
              if (have === 0) {
                break inf_leave;
              }
              have--;
              hold += input[next++] << bits;
              bits += 8;
            }
            state.nlen = (hold & 31) + 257;
            hold >>>= 5;
            bits -= 5;
            state.ndist = (hold & 31) + 1;
            hold >>>= 5;
            bits -= 5;
            state.ncode = (hold & 15) + 4;
            hold >>>= 4;
            bits -= 4;
            if (state.nlen > 286 || state.ndist > 30) {
              strm.msg = "too many length or distance symbols";
              state.mode = BAD;
              break;
            }
            state.have = 0;
            state.mode = LENLENS;
          /* falls through */
          case LENLENS:
            while (state.have < state.ncode) {
              while (bits < 3) {
                if (have === 0) {
                  break inf_leave;
                }
                have--;
                hold += input[next++] << bits;
                bits += 8;
              }
              state.lens[order[state.have++]] = hold & 7;
              hold >>>= 3;
              bits -= 3;
            }
            while (state.have < 19) {
              state.lens[order[state.have++]] = 0;
            }
            state.lencode = state.lendyn;
            state.lenbits = 7;
            opts = { bits: state.lenbits };
            ret = inftrees(CODES, state.lens, 0, 19, state.lencode, 0, state.work, opts);
            state.lenbits = opts.bits;
            if (ret) {
              strm.msg = "invalid code lengths set";
              state.mode = BAD;
              break;
            }
            state.have = 0;
            state.mode = CODELENS;
          /* falls through */
          case CODELENS:
            while (state.have < state.nlen + state.ndist) {
              for (; ; ) {
                here = state.lencode[hold & (1 << state.lenbits) - 1];
                here_bits = here >>> 24;
                here_op = here >>> 16 & 255;
                here_val = here & 65535;
                if (here_bits <= bits) {
                  break;
                }
                if (have === 0) {
                  break inf_leave;
                }
                have--;
                hold += input[next++] << bits;
                bits += 8;
              }
              if (here_val < 16) {
                hold >>>= here_bits;
                bits -= here_bits;
                state.lens[state.have++] = here_val;
              } else {
                if (here_val === 16) {
                  n = here_bits + 2;
                  while (bits < n) {
                    if (have === 0) {
                      break inf_leave;
                    }
                    have--;
                    hold += input[next++] << bits;
                    bits += 8;
                  }
                  hold >>>= here_bits;
                  bits -= here_bits;
                  if (state.have === 0) {
                    strm.msg = "invalid bit length repeat";
                    state.mode = BAD;
                    break;
                  }
                  len = state.lens[state.have - 1];
                  copy = 3 + (hold & 3);
                  hold >>>= 2;
                  bits -= 2;
                } else if (here_val === 17) {
                  n = here_bits + 3;
                  while (bits < n) {
                    if (have === 0) {
                      break inf_leave;
                    }
                    have--;
                    hold += input[next++] << bits;
                    bits += 8;
                  }
                  hold >>>= here_bits;
                  bits -= here_bits;
                  len = 0;
                  copy = 3 + (hold & 7);
                  hold >>>= 3;
                  bits -= 3;
                } else {
                  n = here_bits + 7;
                  while (bits < n) {
                    if (have === 0) {
                      break inf_leave;
                    }
                    have--;
                    hold += input[next++] << bits;
                    bits += 8;
                  }
                  hold >>>= here_bits;
                  bits -= here_bits;
                  len = 0;
                  copy = 11 + (hold & 127);
                  hold >>>= 7;
                  bits -= 7;
                }
                if (state.have + copy > state.nlen + state.ndist) {
                  strm.msg = "invalid bit length repeat";
                  state.mode = BAD;
                  break;
                }
                while (copy--) {
                  state.lens[state.have++] = len;
                }
              }
            }
            if (state.mode === BAD) {
              break;
            }
            if (state.lens[256] === 0) {
              strm.msg = "invalid code -- missing end-of-block";
              state.mode = BAD;
              break;
            }
            state.lenbits = 9;
            opts = { bits: state.lenbits };
            ret = inftrees(LENS, state.lens, 0, state.nlen, state.lencode, 0, state.work, opts);
            state.lenbits = opts.bits;
            if (ret) {
              strm.msg = "invalid literal/lengths set";
              state.mode = BAD;
              break;
            }
            state.distbits = 6;
            state.distcode = state.distdyn;
            opts = { bits: state.distbits };
            ret = inftrees(DISTS, state.lens, state.nlen, state.ndist, state.distcode, 0, state.work, opts);
            state.distbits = opts.bits;
            if (ret) {
              strm.msg = "invalid distances set";
              state.mode = BAD;
              break;
            }
            state.mode = LEN_;
            if (flush === Z_TREES) {
              break inf_leave;
            }
          /* falls through */
          case LEN_:
            state.mode = LEN;
          /* falls through */
          case LEN:
            if (have >= 6 && left >= 258) {
              strm.next_out = put;
              strm.avail_out = left;
              strm.next_in = next;
              strm.avail_in = have;
              state.hold = hold;
              state.bits = bits;
              inffast(strm, _out);
              put = strm.next_out;
              output = strm.output;
              left = strm.avail_out;
              next = strm.next_in;
              input = strm.input;
              have = strm.avail_in;
              hold = state.hold;
              bits = state.bits;
              if (state.mode === TYPE) {
                state.back = -1;
              }
              break;
            }
            state.back = 0;
            for (; ; ) {
              here = state.lencode[hold & (1 << state.lenbits) - 1];
              here_bits = here >>> 24;
              here_op = here >>> 16 & 255;
              here_val = here & 65535;
              if (here_bits <= bits) {
                break;
              }
              if (have === 0) {
                break inf_leave;
              }
              have--;
              hold += input[next++] << bits;
              bits += 8;
            }
            if (here_op && (here_op & 240) === 0) {
              last_bits = here_bits;
              last_op = here_op;
              last_val = here_val;
              for (; ; ) {
                here = state.lencode[last_val + ((hold & (1 << last_bits + last_op) - 1) >> last_bits)];
                here_bits = here >>> 24;
                here_op = here >>> 16 & 255;
                here_val = here & 65535;
                if (last_bits + here_bits <= bits) {
                  break;
                }
                if (have === 0) {
                  break inf_leave;
                }
                have--;
                hold += input[next++] << bits;
                bits += 8;
              }
              hold >>>= last_bits;
              bits -= last_bits;
              state.back += last_bits;
            }
            hold >>>= here_bits;
            bits -= here_bits;
            state.back += here_bits;
            state.length = here_val;
            if (here_op === 0) {
              state.mode = LIT;
              break;
            }
            if (here_op & 32) {
              state.back = -1;
              state.mode = TYPE;
              break;
            }
            if (here_op & 64) {
              strm.msg = "invalid literal/length code";
              state.mode = BAD;
              break;
            }
            state.extra = here_op & 15;
            state.mode = LENEXT;
          /* falls through */
          case LENEXT:
            if (state.extra) {
              n = state.extra;
              while (bits < n) {
                if (have === 0) {
                  break inf_leave;
                }
                have--;
                hold += input[next++] << bits;
                bits += 8;
              }
              state.length += hold & (1 << state.extra) - 1;
              hold >>>= state.extra;
              bits -= state.extra;
              state.back += state.extra;
            }
            state.was = state.length;
            state.mode = DIST;
          /* falls through */
          case DIST:
            for (; ; ) {
              here = state.distcode[hold & (1 << state.distbits) - 1];
              here_bits = here >>> 24;
              here_op = here >>> 16 & 255;
              here_val = here & 65535;
              if (here_bits <= bits) {
                break;
              }
              if (have === 0) {
                break inf_leave;
              }
              have--;
              hold += input[next++] << bits;
              bits += 8;
            }
            if ((here_op & 240) === 0) {
              last_bits = here_bits;
              last_op = here_op;
              last_val = here_val;
              for (; ; ) {
                here = state.distcode[last_val + ((hold & (1 << last_bits + last_op) - 1) >> last_bits)];
                here_bits = here >>> 24;
                here_op = here >>> 16 & 255;
                here_val = here & 65535;
                if (last_bits + here_bits <= bits) {
                  break;
                }
                if (have === 0) {
                  break inf_leave;
                }
                have--;
                hold += input[next++] << bits;
                bits += 8;
              }
              hold >>>= last_bits;
              bits -= last_bits;
              state.back += last_bits;
            }
            hold >>>= here_bits;
            bits -= here_bits;
            state.back += here_bits;
            if (here_op & 64) {
              strm.msg = "invalid distance code";
              state.mode = BAD;
              break;
            }
            state.offset = here_val;
            state.extra = here_op & 15;
            state.mode = DISTEXT;
          /* falls through */
          case DISTEXT:
            if (state.extra) {
              n = state.extra;
              while (bits < n) {
                if (have === 0) {
                  break inf_leave;
                }
                have--;
                hold += input[next++] << bits;
                bits += 8;
              }
              state.offset += hold & (1 << state.extra) - 1;
              hold >>>= state.extra;
              bits -= state.extra;
              state.back += state.extra;
            }
            if (state.offset > state.dmax) {
              strm.msg = "invalid distance too far back";
              state.mode = BAD;
              break;
            }
            state.mode = MATCH;
          /* falls through */
          case MATCH:
            if (left === 0) {
              break inf_leave;
            }
            copy = _out - left;
            if (state.offset > copy) {
              copy = state.offset - copy;
              if (copy > state.whave) {
                if (state.sane) {
                  strm.msg = "invalid distance too far back";
                  state.mode = BAD;
                  break;
                }
              }
              if (copy > state.wnext) {
                copy -= state.wnext;
                from = state.wsize - copy;
              } else {
                from = state.wnext - copy;
              }
              if (copy > state.length) {
                copy = state.length;
              }
              from_source = state.window;
            } else {
              from_source = output;
              from = put - state.offset;
              copy = state.length;
            }
            if (copy > left) {
              copy = left;
            }
            left -= copy;
            state.length -= copy;
            do {
              output[put++] = from_source[from++];
            } while (--copy);
            if (state.length === 0) {
              state.mode = LEN;
            }
            break;
          case LIT:
            if (left === 0) {
              break inf_leave;
            }
            output[put++] = state.length;
            left--;
            state.mode = LEN;
            break;
          case CHECK:
            if (state.wrap) {
              while (bits < 32) {
                if (have === 0) {
                  break inf_leave;
                }
                have--;
                hold |= input[next++] << bits;
                bits += 8;
              }
              _out -= left;
              strm.total_out += _out;
              state.total += _out;
              if (state.wrap & 4 && _out) {
                strm.adler = state.check = /*UPDATE_CHECK(state.check, put - _out, _out);*/
                state.flags ? crc32_1(state.check, output, _out, put - _out) : adler32_1(state.check, output, _out, put - _out);
              }
              _out = left;
              if (state.wrap & 4 && (state.flags ? hold : zswap32(hold)) !== state.check) {
                strm.msg = "incorrect data check";
                state.mode = BAD;
                break;
              }
              hold = 0;
              bits = 0;
            }
            state.mode = LENGTH;
          /* falls through */
          case LENGTH:
            if (state.wrap && state.flags) {
              while (bits < 32) {
                if (have === 0) {
                  break inf_leave;
                }
                have--;
                hold += input[next++] << bits;
                bits += 8;
              }
              if (state.wrap & 4 && hold !== (state.total & 4294967295)) {
                strm.msg = "incorrect length check";
                state.mode = BAD;
                break;
              }
              hold = 0;
              bits = 0;
            }
            state.mode = DONE;
          /* falls through */
          case DONE:
            ret = Z_STREAM_END$1;
            break inf_leave;
          case BAD:
            ret = Z_DATA_ERROR$1;
            break inf_leave;
          case MEM:
            return Z_MEM_ERROR$1;
          case SYNC:
          /* falls through */
          default:
            return Z_STREAM_ERROR$1;
        }
      }
    strm.next_out = put;
    strm.avail_out = left;
    strm.next_in = next;
    strm.avail_in = have;
    state.hold = hold;
    state.bits = bits;
    if (state.wsize || _out !== strm.avail_out && state.mode < BAD && (state.mode < CHECK || flush !== Z_FINISH$1)) {
      if (updatewindow(strm, strm.output, strm.next_out, _out - strm.avail_out)) ;
    }
    _in -= strm.avail_in;
    _out -= strm.avail_out;
    strm.total_in += _in;
    strm.total_out += _out;
    state.total += _out;
    if (state.wrap & 4 && _out) {
      strm.adler = state.check = /*UPDATE_CHECK(state.check, strm.next_out - _out, _out);*/
      state.flags ? crc32_1(state.check, output, _out, strm.next_out - _out) : adler32_1(state.check, output, _out, strm.next_out - _out);
    }
    strm.data_type = state.bits + (state.last ? 64 : 0) + (state.mode === TYPE ? 128 : 0) + (state.mode === LEN_ || state.mode === COPY_ ? 256 : 0);
    if ((_in === 0 && _out === 0 || flush === Z_FINISH$1) && ret === Z_OK$1) {
      ret = Z_BUF_ERROR;
    }
    return ret;
  };
  var inflateEnd = (strm) => {
    if (inflateStateCheck(strm)) {
      return Z_STREAM_ERROR$1;
    }
    let state = strm.state;
    if (state.window) {
      state.window = null;
    }
    strm.state = null;
    return Z_OK$1;
  };
  var inflateGetHeader = (strm, head) => {
    if (inflateStateCheck(strm)) {
      return Z_STREAM_ERROR$1;
    }
    const state = strm.state;
    if ((state.wrap & 2) === 0) {
      return Z_STREAM_ERROR$1;
    }
    state.head = head;
    head.done = false;
    return Z_OK$1;
  };
  var inflateSetDictionary = (strm, dictionary) => {
    const dictLength = dictionary.length;
    let state;
    let dictid;
    let ret;
    if (inflateStateCheck(strm)) {
      return Z_STREAM_ERROR$1;
    }
    state = strm.state;
    if (state.wrap !== 0 && state.mode !== DICT) {
      return Z_STREAM_ERROR$1;
    }
    if (state.mode === DICT) {
      dictid = 1;
      dictid = adler32_1(dictid, dictionary, dictLength, 0);
      if (dictid !== state.check) {
        return Z_DATA_ERROR$1;
      }
    }
    ret = updatewindow(strm, dictionary, dictLength, dictLength);
    if (ret) {
      state.mode = MEM;
      return Z_MEM_ERROR$1;
    }
    state.havedict = 1;
    return Z_OK$1;
  };
  var inflateReset_1 = inflateReset;
  var inflateReset2_1 = inflateReset2;
  var inflateResetKeep_1 = inflateResetKeep;
  var inflateInit_1 = inflateInit;
  var inflateInit2_1 = inflateInit2;
  var inflate_2$1 = inflate$2;
  var inflateEnd_1 = inflateEnd;
  var inflateGetHeader_1 = inflateGetHeader;
  var inflateSetDictionary_1 = inflateSetDictionary;
  var inflateInfo = "pako inflate (from Nodeca project)";
  var inflate_1$2 = {
    inflateReset: inflateReset_1,
    inflateReset2: inflateReset2_1,
    inflateResetKeep: inflateResetKeep_1,
    inflateInit: inflateInit_1,
    inflateInit2: inflateInit2_1,
    inflate: inflate_2$1,
    inflateEnd: inflateEnd_1,
    inflateGetHeader: inflateGetHeader_1,
    inflateSetDictionary: inflateSetDictionary_1,
    inflateInfo
  };
  function GZheader() {
    this.text = 0;
    this.time = 0;
    this.xflags = 0;
    this.os = 0;
    this.extra = null;
    this.extra_len = 0;
    this.name = "";
    this.comment = "";
    this.hcrc = 0;
    this.done = false;
  }
  var gzheader = GZheader;
  var toString = Object.prototype.toString;
  var {
    Z_NO_FLUSH,
    Z_FINISH,
    Z_OK,
    Z_STREAM_END,
    Z_NEED_DICT,
    Z_STREAM_ERROR,
    Z_DATA_ERROR,
    Z_MEM_ERROR
  } = constants$2;
  function Inflate$1(options) {
    this.options = common.assign({
      chunkSize: 1024 * 64,
      windowBits: 15,
      to: ""
    }, options || {});
    const opt = this.options;
    if (opt.raw && opt.windowBits >= 0 && opt.windowBits < 16) {
      opt.windowBits = -opt.windowBits;
      if (opt.windowBits === 0) {
        opt.windowBits = -15;
      }
    }
    if (opt.windowBits >= 0 && opt.windowBits < 16 && !(options && options.windowBits)) {
      opt.windowBits += 32;
    }
    if (opt.windowBits > 15 && opt.windowBits < 48) {
      if ((opt.windowBits & 15) === 0) {
        opt.windowBits |= 15;
      }
    }
    this.err = 0;
    this.msg = "";
    this.ended = false;
    this.chunks = [];
    this.strm = new zstream();
    this.strm.avail_out = 0;
    let status = inflate_1$2.inflateInit2(
      this.strm,
      opt.windowBits
    );
    if (status !== Z_OK) {
      throw new Error(messages[status]);
    }
    this.header = new gzheader();
    inflate_1$2.inflateGetHeader(this.strm, this.header);
    if (opt.dictionary) {
      if (typeof opt.dictionary === "string") {
        opt.dictionary = strings.string2buf(opt.dictionary);
      } else if (toString.call(opt.dictionary) === "[object ArrayBuffer]") {
        opt.dictionary = new Uint8Array(opt.dictionary);
      }
      if (opt.raw) {
        status = inflate_1$2.inflateSetDictionary(this.strm, opt.dictionary);
        if (status !== Z_OK) {
          throw new Error(messages[status]);
        }
      }
    }
  }
  Inflate$1.prototype.push = function(data, flush_mode) {
    const strm = this.strm;
    const chunkSize = this.options.chunkSize;
    const dictionary = this.options.dictionary;
    let status, _flush_mode, last_avail_out;
    if (this.ended) return false;
    if (flush_mode === ~~flush_mode) _flush_mode = flush_mode;
    else _flush_mode = flush_mode === true ? Z_FINISH : Z_NO_FLUSH;
    if (toString.call(data) === "[object ArrayBuffer]") {
      strm.input = new Uint8Array(data);
    } else {
      strm.input = data;
    }
    strm.next_in = 0;
    strm.avail_in = strm.input.length;
    for (; ; ) {
      if (strm.avail_out === 0) {
        strm.output = new Uint8Array(chunkSize);
        strm.next_out = 0;
        strm.avail_out = chunkSize;
      }
      status = inflate_1$2.inflate(strm, _flush_mode);
      if (status === Z_NEED_DICT && dictionary) {
        status = inflate_1$2.inflateSetDictionary(strm, dictionary);
        if (status === Z_OK) {
          status = inflate_1$2.inflate(strm, _flush_mode);
        } else if (status === Z_DATA_ERROR) {
          status = Z_NEED_DICT;
        }
      }
      while (strm.avail_in > 0 && status === Z_STREAM_END && strm.state.wrap > 0 && data[strm.next_in] !== 0) {
        inflate_1$2.inflateReset(strm);
        status = inflate_1$2.inflate(strm, _flush_mode);
      }
      switch (status) {
        case Z_STREAM_ERROR:
        case Z_DATA_ERROR:
        case Z_NEED_DICT:
        case Z_MEM_ERROR:
          this.onEnd(status);
          this.ended = true;
          return false;
      }
      last_avail_out = strm.avail_out;
      if (strm.next_out) {
        if (strm.avail_out === 0 || status === Z_STREAM_END) {
          if (this.options.to === "string") {
            let next_out_utf8 = strings.utf8border(strm.output, strm.next_out);
            let tail = strm.next_out - next_out_utf8;
            let utf8str = strings.buf2string(strm.output, next_out_utf8);
            strm.next_out = tail;
            strm.avail_out = chunkSize - tail;
            if (tail) strm.output.set(strm.output.subarray(next_out_utf8, next_out_utf8 + tail), 0);
            this.onData(utf8str);
          } else {
            this.onData(strm.output.length === strm.next_out ? strm.output : strm.output.subarray(0, strm.next_out));
          }
        }
      }
      if (status === Z_OK && last_avail_out === 0) continue;
      if (status === Z_STREAM_END) {
        status = inflate_1$2.inflateEnd(this.strm);
        this.onEnd(status);
        this.ended = true;
        return true;
      }
      if (strm.avail_in === 0) break;
    }
    return true;
  };
  Inflate$1.prototype.onData = function(chunk) {
    this.chunks.push(chunk);
  };
  Inflate$1.prototype.onEnd = function(status) {
    if (status === Z_OK) {
      if (this.options.to === "string") {
        this.result = this.chunks.join("");
      } else {
        this.result = common.flattenChunks(this.chunks);
      }
    }
    this.chunks = [];
    this.err = status;
    this.msg = this.strm.msg;
  };
  function inflate$1(input, options) {
    const inflator = new Inflate$1(options);
    inflator.push(input);
    if (inflator.err) throw inflator.msg || messages[inflator.err];
    return inflator.result;
  }
  function inflateRaw$1(input, options) {
    options = options || {};
    options.raw = true;
    return inflate$1(input, options);
  }
  var Inflate_1$1 = Inflate$1;
  var inflate_2 = inflate$1;
  var inflateRaw_1$1 = inflateRaw$1;
  var ungzip$1 = inflate$1;
  var constants = constants$2;
  var inflate_1$1 = {
    Inflate: Inflate_1$1,
    inflate: inflate_2,
    inflateRaw: inflateRaw_1$1,
    ungzip: ungzip$1,
    constants
  };
  var { Deflate, deflate, deflateRaw, gzip } = deflate_1$1;
  var { Inflate, inflate, inflateRaw, ungzip } = inflate_1$1;
  var inflate_1 = inflate;

  // esm/filters.js
  var zlib_decompress = function(buf, itemsize) {
    let input_array = new Uint8Array(buf);
    return inflate_1(input_array).buffer;
  };
  var unshuffle = function(buf, itemsize) {
    let buffer_size = buf.byteLength;
    let unshuffled_view = new Uint8Array(buffer_size);
    let step = Math.floor(buffer_size / itemsize);
    let shuffled_view = new DataView(buf);
    for (var j = 0; j < itemsize; j++) {
      for (var i = 0; i < step; i++) {
        unshuffled_view[j + i * itemsize] = shuffled_view.getUint8(j * step + i);
      }
    }
    return unshuffled_view.buffer;
  };
  var fletch32 = function(buf, itemsize) {
    _verify_fletcher32(buf);
    return buf.slice(0, -4);
  };
  function _verify_fletcher32(chunk_buffer) {
    var odd_chunk_buffer = chunk_buffer.byteLength % 2 != 0;
    var data_length = chunk_buffer.byteLength - 4;
    var view = new DataView(chunk_buffer);
    var sum1 = 0;
    var sum2 = 0;
    for (var offset = 0; offset < data_length - 1; offset += 2) {
      let datum = view.getUint16(offset, true);
      sum1 = (sum1 + datum) % 65535;
      sum2 = (sum2 + sum1) % 65535;
    }
    if (odd_chunk_buffer) {
      let datum = view.getUint8(data_length - 1);
      sum1 = (sum1 + datum) % 65535;
      sum2 = (sum2 + sum1) % 65535;
    }
    var [ref_sum1, ref_sum2] = struct.unpack_from(">HH", chunk_buffer, data_length);
    ref_sum1 = ref_sum1 % 65535;
    ref_sum2 = ref_sum2 % 65535;
    if (sum1 != ref_sum1 || sum2 != ref_sum2) {
      throw 'ValueError("fletcher32 checksum invalid")';
    }
    return true;
  }
  var GZIP_DEFLATE_FILTER = 1;
  var SHUFFLE_FILTER = 2;
  var FLETCH32_FILTER = 3;
  var Filters = /* @__PURE__ */ new Map([
    [GZIP_DEFLATE_FILTER, zlib_decompress],
    [SHUFFLE_FILTER, unshuffle],
    [FLETCH32_FILTER, fletch32]
  ]);

  // esm/btree.js
  var AbstractBTree = class {
    //B_LINK_NODE = null;
    //NODE_TYPE = null;
    constructor(fh, offset) {
      this.fh = fh;
      this.offset = offset;
      this.depth = null;
    }
    init() {
      this.all_nodes = /* @__PURE__ */ new Map();
      this._read_root_node();
      this._read_children();
    }
    _read_children() {
      let node_level = this.depth;
      while (node_level > 0) {
        for (var parent_node of this.all_nodes.get(node_level)) {
          for (var child_addr of parent_node.get("addresses")) {
            this._add_node(this._read_node(child_addr, node_level - 1));
          }
        }
        node_level--;
      }
    }
    _read_root_node() {
      let root_node = this._read_node(this.offset, null);
      this._add_node(root_node);
      this.depth = root_node.get("node_level");
    }
    _add_node(node2) {
      let node_level = node2.get("node_level");
      if (this.all_nodes.has(node_level)) {
        this.all_nodes.get(node_level).push(node2);
      } else {
        this.all_nodes.set(node_level, [node2]);
      }
    }
    _read_node(offset, node_level) {
      node = this._read_node_header(offset, node_level);
      node.set("keys", []);
      node.set("addresses", []);
      return node;
    }
    _read_node_header(offset) {
      throw "NotImplementedError: must define _read_node_header in implementation class";
    }
  };
  var BTreeV1 = class extends AbstractBTree {
    constructor() {
      super(...arguments);
      /*
      """
      HDF5 version 1 B-Tree.
      """
      */
      __publicField(this, "B_LINK_NODE", /* @__PURE__ */ new Map([
        ["signature", "4s"],
        ["node_type", "B"],
        ["node_level", "B"],
        ["entries_used", "H"],
        ["left_sibling", "Q"],
        // 8 byte addressing
        ["right_sibling", "Q"]
        // 8 byte addressing
      ]));
    }
    _read_node_header(offset, node_level) {
      let node2 = _unpack_struct_from(this.B_LINK_NODE, this.fh, offset);
      if (node_level != null) {
        if (node2.get("node_level") != node_level) {
          throw "node level does not match";
        }
      }
      return node2;
    }
  };
  var BTreeV1Groups = class extends BTreeV1 {
    constructor(fh, offset) {
      super(fh, offset);
      /*
      """
      HDF5 version 1 B-Tree storing group nodes (type 0).
      """
      */
      __publicField(this, "NODE_TYPE", 0);
      this.init();
    }
    _read_node(offset, node_level) {
      let node2 = this._read_node_header(offset, node_level);
      offset += _structure_size(this.B_LINK_NODE);
      let keys = [];
      let addresses = [];
      let entries_used = node2.get("entries_used");
      for (var i = 0; i < entries_used; i++) {
        let key = struct.unpack_from("<Q", this.fh, offset)[0];
        offset += 8;
        let address = struct.unpack_from("<Q", this.fh, offset)[0];
        offset += 8;
        keys.push(key);
        addresses.push(address);
      }
      keys.push(struct.unpack_from("<Q", this.fh, offset)[0]);
      node2.set("keys", keys);
      node2.set("addresses", addresses);
      return node2;
    }
    symbol_table_addresses() {
      var all_address = [];
      var root_nodes = this.all_nodes.get(0);
      for (var node2 of root_nodes) {
        all_address = all_address.concat(node2.get("addresses"));
      }
      return all_address;
    }
  };
  var BTreeV1RawDataChunks = class extends BTreeV1 {
    constructor(fh, offset, dims) {
      super(fh, offset);
      /*
      HDF5 version 1 B-Tree storing raw data chunk nodes (type 1).
      */
      __publicField(this, "NODE_TYPE", 1);
      this.dims = dims;
      this.init();
    }
    _read_node(offset, node_level) {
      let node2 = this._read_node_header(offset, node_level);
      offset += _structure_size(this.B_LINK_NODE);
      var keys = [];
      var addresses = [];
      let entries_used = node2.get("entries_used");
      for (var i = 0; i < entries_used; i++) {
        let [chunk_size, filter_mask] = struct.unpack_from("<II", this.fh, offset);
        offset += 8;
        let fmt = "<" + this.dims.toFixed() + "Q";
        let fmt_size = struct.calcsize(fmt);
        let chunk_offset = struct.unpack_from(fmt, this.fh, offset);
        offset += fmt_size;
        let chunk_address = struct.unpack_from("<Q", this.fh, offset)[0];
        offset += 8;
        keys.push(/* @__PURE__ */ new Map([
          ["chunk_size", chunk_size],
          ["filter_mask", filter_mask],
          ["chunk_offset", chunk_offset]
        ]));
        addresses.push(chunk_address);
      }
      node2.set("keys", keys);
      node2.set("addresses", addresses);
      return node2;
    }
    construct_data_from_chunks(chunk_shape, data_shape, dtype, filter_pipeline) {
      var true_dtype;
      var item_getter, item_big_endian, item_size;
      if (dtype instanceof Array) {
        true_dtype = dtype;
        let dtype_class = dtype[0];
        if (dtype_class == "REFERENCE") {
          let size = dtype[1];
          if (size != 8) {
            throw "NotImplementedError('Unsupported Reference type')";
          }
          var dtype = "<u8";
          item_getter = "getUint64";
          item_big_endian = false;
          item_size = 8;
        } else if (dtype_class == "VLEN_STRING" || dtype_class == "VLEN_SEQUENCE") {
          item_getter = "getVLENStruct";
          item_big_endian = false;
          item_size = 16;
        } else {
          throw "NotImplementedError('datatype not implemented')";
        }
      } else {
        true_dtype = null;
        [item_getter, item_big_endian, item_size] = dtype_getter(dtype);
      }
      var data_size = data_shape.reduce(function(a, b) {
        return a * b;
      }, 1);
      var chunk_size = chunk_shape.reduce(function(a, b) {
        return a * b;
      }, 1);
      let dims = data_shape.length;
      var current_stride = 1;
      var chunk_strides = chunk_shape.slice().map(function(d2) {
        let s = current_stride;
        current_stride *= d2;
        return s;
      });
      var current_stride = 1;
      var data_strides = data_shape.slice().reverse().map(function(d2) {
        let s = current_stride;
        current_stride *= d2;
        return s;
      }).reverse();
      var data = new Array(data_size);
      let chunk_buffer_size = chunk_size * item_size;
      for (var node2 of this.all_nodes.get(0)) {
        let node_keys = node2.get("keys");
        let node_addresses = node2.get("addresses");
        let nkeys = node_keys.length;
        for (var ik = 0; ik < nkeys; ik++) {
          let node_key = node_keys[ik];
          let addr = node_addresses[ik];
          var chunk_buffer;
          if (filter_pipeline == null) {
            chunk_buffer = this.fh.slice(addr, addr + chunk_buffer_size);
          } else {
            chunk_buffer = this.fh.slice(addr, addr + node_key.get("chunk_size"));
            let filter_mask = node_key.get("filter_mask");
            chunk_buffer = this._filter_chunk(
              chunk_buffer,
              filter_mask,
              filter_pipeline,
              item_size
            );
          }
          var chunk_offset = node_key.get("chunk_offset").slice();
          var apos = chunk_offset.slice();
          var cpos = apos.map(function() {
            return 0;
          });
          var cview = new DataView64(chunk_buffer);
          for (var ci = 0; ci < chunk_size; ci++) {
            for (var d = dims - 1; d >= 0; d--) {
              if (cpos[d] >= chunk_shape[d]) {
                cpos[d] = 0;
                apos[d] = chunk_offset[d];
                if (d > 0) {
                  cpos[d - 1] += 1;
                  apos[d - 1] += 1;
                }
              } else {
                break;
              }
            }
            let inbounds = apos.slice(0, -1).every(function(p, d2) {
              return p < data_shape[d2];
            });
            if (inbounds) {
              let cb_offset = ci * item_size;
              let datum = cview[item_getter](cb_offset, !item_big_endian, item_size);
              let ai = apos.slice(0, -1).reduce(function(prev, curr, index) {
                return curr * data_strides[index] + prev;
              }, 0);
              data[ai] = datum;
            }
            cpos[dims - 1] += 1;
            apos[dims - 1] += 1;
          }
        }
      }
      return data;
    }
    _filter_chunk(chunk_buffer, filter_mask, filter_pipeline, itemsize) {
      let num_filters = filter_pipeline.length;
      let buf = chunk_buffer.slice();
      for (var filter_index = num_filters - 1; filter_index >= 0; filter_index--) {
        if (filter_mask & 1 << filter_index) {
          continue;
        }
        let pipeline_entry = filter_pipeline[filter_index];
        let filter_id = pipeline_entry.get("filter_id");
        let client_data = pipeline_entry.get("client_data");
        if (Filters.has(filter_id)) {
          buf = Filters.get(filter_id)(buf, itemsize, client_data);
        } else {
          throw 'NotImplementedError("Filter with id:' + filter_id.toFixed() + ' not supported")';
        }
      }
      return buf;
    }
  };
  var BTreeV2 = class extends AbstractBTree {
    constructor(fh, offset) {
      super(fh, offset);
      /*
      HDF5 version 2 B-Tree.
      */
      // III.A.2. Disk Format: Level 1A2 - Version 2 B-trees
      __publicField(this, "B_TREE_HEADER", /* @__PURE__ */ new Map([
        ["signature", "4s"],
        ["version", "B"],
        ["node_type", "B"],
        ["node_size", "I"],
        ["record_size", "H"],
        ["depth", "H"],
        ["split_percent", "B"],
        ["merge_percent", "B"],
        ["root_address", "Q"],
        // 8 byte addressing
        ["root_nrecords", "H"],
        ["total_nrecords", "Q"]
        // 8 byte addressing
      ]));
      __publicField(this, "B_LINK_NODE", /* @__PURE__ */ new Map([
        ["signature", "4s"],
        ["version", "B"],
        ["node_type", "B"]
      ]));
      this.init();
    }
    _read_root_node() {
      let h = this._read_tree_header(this.offset);
      this.address_formats = this._calculate_address_formats(h);
      this.header = h;
      this.depth = h.get("depth");
      let address = [h.get("root_address"), h.get("root_nrecords"), h.get("total_nrecords")];
      let root_node = this._read_node(address, this.depth);
      this._add_node(root_node);
    }
    _read_tree_header(offset) {
      let header = _unpack_struct_from(this.B_TREE_HEADER, this.fh, this.offset);
      return header;
    }
    _calculate_address_formats(header) {
      let node_size = header.get("node_size");
      let record_size = header.get("record_size");
      let nrecords_max = 0;
      let ntotalrecords_max = 0;
      let address_formats = /* @__PURE__ */ new Map();
      let max_depth = header.get("depth");
      for (var node_level = 0; node_level <= max_depth; node_level++) {
        let offset_fmt = "";
        let num1_fmt = "";
        let num2_fmt = "";
        let offset_size, num1_size, num2_size;
        if (node_level == 0) {
          offset_size = 0;
          num1_size = 0;
          num2_size = 0;
        } else if (node_level == 1) {
          offset_size = 8;
          offset_fmt = "<Q";
          num1_size = this._required_bytes(nrecords_max);
          num1_fmt = this._int_format(num1_size);
          num2_size = 0;
        } else {
          offset_size = 8;
          offset_fmt = "<Q";
          num1_size = this._required_bytes(nrecords_max);
          num1_fmt = this._int_format(num1_size);
          num2_size = this._required_bytes(ntotalrecords_max);
          num2_fmt = this._int_format(num2_size);
        }
        address_formats.set(node_level, [
          offset_size,
          num1_size,
          num2_size,
          offset_fmt,
          num1_fmt,
          num2_fmt
        ]);
        if (node_level < max_depth) {
          let addr_size = offset_size + num1_size + num2_size;
          nrecords_max = this._nrecords_max(node_size, record_size, addr_size);
          if (ntotalrecords_max > 0) {
            ntotalrecords_max *= nrecords_max;
          } else {
            ntotalrecords_max = nrecords_max;
          }
        }
      }
      return address_formats;
    }
    _nrecords_max(node_size, record_size, addr_size) {
      return Math.floor((node_size - 10 - addr_size) / (record_size + addr_size));
    }
    _required_bytes(integer) {
      return Math.ceil(bitSize(integer) / 8);
    }
    _int_format(bytelength) {
      return ["<B", "<H", "<I", "<Q"][bytelength - 1];
    }
    _read_node(address, node_level) {
      let [offset, nrecords, ntotalrecords] = address;
      let node2 = this._read_node_header(offset, node_level);
      offset += _structure_size(this.B_LINK_NODE);
      let record_size = this.header.get("record_size");
      let keys = [];
      for (let i = 0; i < nrecords; i++) {
        let record = this._parse_record(this.fh, offset, record_size);
        offset += record_size;
        keys.push(record);
      }
      let addresses = [];
      let fmts = this.address_formats.get(node_level);
      if (node_level != 0) {
        let [offset_size, num1_size, num2_size, offset_fmt, num1_fmt, num2_fmt] = fmts;
        for (let j = 0; j <= nrecords; j++) {
          let address_offset = struct.unpack_from(offset_fmt, this.fh, offset)[0];
          offset += offset_size;
          let num1 = struct.unpack_from(num1_fmt, this.fh, offset)[0];
          offset += num1_size;
          let num2 = num1;
          if (num2_size > 0) {
            num2 = struct.unpack_from(num2_fmt, this.fh, offset)[0];
            offset += num2_size;
          }
          addresses.push([address_offset, num1, num2]);
        }
      }
      node2.set("keys", keys);
      node2.set("addresses", addresses);
      return node2;
    }
    _read_node_header(offset, node_level) {
      let node2 = _unpack_struct_from(this.B_LINK_NODE, this.fh, offset);
      if (node_level > 0) {
      } else {
      }
      node2.set("node_level", node_level);
      return node2;
    }
    *iter_records() {
      for (let nodelist of this.all_nodes.values()) {
        for (let node2 of nodelist) {
          for (let key of node2.get("keys")) {
            yield key;
          }
        }
      }
    }
    _parse_record(record) {
      throw "NotImplementedError";
    }
  };
  var BTreeV2GroupNames = class extends BTreeV2 {
    constructor() {
      super(...arguments);
      /*
      HDF5 version 2 B-Tree storing group names (type 5).
      */
      __publicField(this, "NODE_TYPE", 5);
    }
    _parse_record(buf, offset, size) {
      let namehash = struct.unpack_from("<I", buf, offset)[0];
      offset += 4;
      return /* @__PURE__ */ new Map([["namehash", namehash], ["heapid", buf.slice(offset, offset + 7)]]);
    }
  };
  var BTreeV2GroupOrders = class extends BTreeV2 {
    constructor() {
      super(...arguments);
      /*
      HDF5 version 2 B-Tree storing group creation orders (type 6).
      */
      __publicField(this, "NODE_TYPE", 6);
    }
    _parse_record(buf, offset, size) {
      let creationorder = struct.unpack_from("<Q", buf, offset)[0];
      offset += 8;
      return /* @__PURE__ */ new Map([["creationorder", creationorder], ["heapid", buf.slice(offset, offset + 7)]]);
    }
  };

  // esm/misc-low-level.js
  var SuperBlock = class {
    constructor(fh, offset) {
      let version_hint = struct.unpack_from("<B", fh, offset + 8)[0];
      var contents;
      if (version_hint == 0) {
        contents = _unpack_struct_from(SUPERBLOCK_V0, fh, offset);
        this._end_of_sblock = offset + SUPERBLOCK_V0_SIZE;
      } else if (version_hint == 2 || version_hint == 3) {
        contents = _unpack_struct_from(SUPERBLOCK_V2_V3, fh, offset);
        this._end_of_sblock = offset + SUPERBLOCK_V2_V3_SIZE;
      } else {
        throw "unsupported superblock version: " + version_hint.toFixed();
      }
      if (contents.get("format_signature") != FORMAT_SIGNATURE) {
        throw "Incorrect file signature: " + contents.get("format_signature");
      }
      if (contents.get("offset_size") != 8 || contents.get("length_size") != 8) {
        throw "File uses non-64-bit addressing";
      }
      this.version = contents.get("superblock_version");
      this._contents = contents;
      this._root_symbol_table = null;
      this._fh = fh;
    }
    get offset_to_dataobjects() {
      if (this.version == 0) {
        var sym_table = new SymbolTable(this._fh, this._end_of_sblock, true);
        this._root_symbol_table = sym_table;
        return sym_table.group_offset;
      } else if (this.version == 2 || this.version == 3) {
        return this._contents.get("root_group_address");
      } else {
        throw "Not implemented version = " + this.version.toFixed();
      }
    }
  };
  var Heap = class {
    /*
    """
    HDF5 local heap.
    """
    */
    constructor(fh, offset) {
      let local_heap = _unpack_struct_from(LOCAL_HEAP, fh, offset);
      assert(local_heap.get("signature") == "HEAP");
      assert(local_heap.get("version") == 0);
      let data_offset = local_heap.get("address_of_data_segment");
      let heap_data = fh.slice(data_offset, data_offset + local_heap.get("data_segment_size"));
      local_heap.set("heap_data", heap_data);
      this._contents = local_heap;
      this.data = heap_data;
    }
    get_object_name(offset) {
      let end = new Uint8Array(this.data).indexOf(0, offset);
      let name_size = end - offset;
      let name = struct.unpack_from("<" + name_size.toFixed() + "s", this.data, offset)[0];
      return name;
    }
  };
  var SymbolTable = class {
    /*
    """
    HDF5 Symbol Table.
    """
    */
    constructor(fh, offset, root = false) {
      var node2;
      if (root) {
        node2 = /* @__PURE__ */ new Map([["symbols", 1]]);
      } else {
        node2 = _unpack_struct_from(SYMBOL_TABLE_NODE, fh, offset);
        if (node2.get("signature") != "SNOD") {
          throw "incorrect node type";
        }
        offset += SYMBOL_TABLE_NODE_SIZE;
      }
      var entries = [];
      var n_symbols = node2.get("symbols");
      for (var i = 0; i < n_symbols; i++) {
        entries.push(_unpack_struct_from(SYMBOL_TABLE_ENTRY, fh, offset));
        offset += SYMBOL_TABLE_ENTRY_SIZE;
      }
      if (root) {
        this.group_offset = entries[0].get("object_header_address");
      }
      this.entries = entries;
      this._contents = node2;
    }
    assign_name(heap) {
      this.entries.forEach(function(entry) {
        let offset = entry.get("link_name_offset");
        let link_name = heap.get_object_name(offset);
        entry.set("link_name", link_name);
      });
    }
    get_links(heap) {
      var links = {};
      this.entries.forEach(function(e) {
        let cache_type = e.get("cache_type");
        let link_name = e.get("link_name");
        if (cache_type == 0 || cache_type == 1) {
          links[link_name] = e.get("object_header_address");
        } else if (cache_type == 2) {
          let scratch = e.get("scratch");
          let buf = new ArrayBuffer(4);
          let bufView = new Uint8Array(buf);
          for (var i = 0; i < 4; i++) {
            bufView[i] = scratch.charCodeAt(i);
          }
          let offset = struct.unpack_from("<I", buf, 0)[0];
          links[link_name] = heap.get_object_name(offset);
        }
      });
      return links;
    }
  };
  var GlobalHeap = class {
    /*
    HDF5 Global Heap collection.
    */
    constructor(fh, offset) {
      let header = _unpack_struct_from(GLOBAL_HEAP_HEADER, fh, offset);
      offset += GLOBAL_HEAP_HEADER_SIZE;
      let heap_data_size = header.get("collection_size") - GLOBAL_HEAP_HEADER_SIZE;
      let heap_data = fh.slice(offset, offset + heap_data_size);
      this.heap_data = heap_data;
      this._header = header;
      this._objects = null;
    }
    get objects() {
      if (this._objects == null) {
        this._objects = /* @__PURE__ */ new Map();
        var offset = 0;
        while (offset <= this.heap_data.byteLength - GLOBAL_HEAP_OBJECT_SIZE) {
          let info = _unpack_struct_from(
            GLOBAL_HEAP_OBJECT,
            this.heap_data,
            offset
          );
          if (info.get("object_index") == 0) {
            break;
          }
          offset += GLOBAL_HEAP_OBJECT_SIZE;
          let obj_data = this.heap_data.slice(offset, offset + info.get("object_size"));
          this._objects.set(info.get("object_index"), obj_data);
          offset += _padded_size(info.get("object_size"));
        }
      }
      return this._objects;
    }
  };
  var FractalHeap = class {
    /*
    HDF5 Fractal Heap.
    */
    constructor(fh, offset) {
      this.fh = fh;
      let header = _unpack_struct_from(FRACTAL_HEAP_HEADER, fh, offset);
      offset += _structure_size(FRACTAL_HEAP_HEADER);
      assert(header.get("signature") == "FRHP");
      assert(header.get("version") == 0);
      if (header.get("filter_info_size") > 0) {
        throw "Filter info size not supported on FractalHeap";
      }
      if (header.get("btree_address_huge_objects") == UNDEFINED_ADDRESS) {
        header.set("btree_address_huge_objects", null);
      } else {
        throw "Huge objects not implemented in FractalHeap";
      }
      if (header.get("root_block_address") == UNDEFINED_ADDRESS) {
        header.set("root_block_address", null);
      }
      let nbits = header.get("log2_maximum_heap_size");
      let block_offset_size = this._min_size_nbits(nbits);
      let h = /* @__PURE__ */ new Map([
        ["signature", "4s"],
        ["version", "B"],
        ["heap_header_adddress", "Q"],
        //['block_offset', `${block_offset_size}s`]
        ["block_offset", `${block_offset_size}B`]
        //this._int_format(block_offset_size)]
      ]);
      this.indirect_block_header = new Map(h);
      this.indirect_block_header_size = _structure_size(h);
      if ((header.get("flags") & 2) == 2) {
        h.set("checksum", "I");
      }
      this.direct_block_header = h;
      this.direct_block_header_size = _structure_size(h);
      let maximum_dblock_size = header.get("maximum_direct_block_size");
      this._managed_object_offset_size = this._min_size_nbits(nbits);
      let value = Math.min(maximum_dblock_size, header.get("max_managed_object_size"));
      this._managed_object_length_size = this._min_size_integer(value);
      let start_block_size = header.get("starting_block_size");
      let table_width = header.get("table_width");
      if (!(start_block_size > 0)) {
        throw "Starting block size == 0 not implemented";
      }
      let log2_maximum_dblock_size = Number(Math.floor(Math.log2(maximum_dblock_size)));
      assert(1n << BigInt(log2_maximum_dblock_size) == maximum_dblock_size);
      let log2_start_block_size = Number(Math.floor(Math.log2(start_block_size)));
      assert(1n << BigInt(log2_start_block_size) == start_block_size);
      this._max_direct_nrows = log2_maximum_dblock_size - log2_start_block_size + 2;
      let log2_table_width = Math.floor(Math.log2(table_width));
      assert(1 << log2_table_width == table_width);
      this._indirect_nrows_sub = log2_table_width + log2_start_block_size - 1;
      this.header = header;
      this.nobjects = header.get("managed_object_count") + header.get("huge_object_count") + header.get("tiny_object_count");
      let managed = [];
      let root_address = header.get("root_block_address");
      let nrows = 0;
      if (root_address != null) {
        nrows = header.get("indirect_current_rows_count");
      }
      if (nrows > 0) {
        for (let data of this._iter_indirect_block(fh, root_address, nrows)) {
          managed.push(data);
        }
      } else {
        let data = this._read_direct_block(fh, root_address, start_block_size);
        managed.push(data);
      }
      let data_size = managed.reduce((p, c) => p + c.byteLength, 0);
      let combined = new Uint8Array(data_size);
      let moffset = 0;
      managed.forEach((m) => {
        combined.set(new Uint8Array(m), moffset);
        moffset += m.byteLength;
      });
      this.managed = combined.buffer;
    }
    _read_direct_block(fh, offset, block_size) {
      let data = fh.slice(offset, offset + block_size);
      let header = _unpack_struct_from(this.direct_block_header, data);
      assert(header.get("signature") == "FHDB");
      return data;
    }
    get_data(heapid) {
      let firstbyte = struct.unpack_from("<B", heapid, 0)[0];
      let reserved = firstbyte & 15;
      let idtype = firstbyte >> 4 & 3;
      let version = firstbyte >> 6;
      let data_offset = 1;
      if (idtype == 0) {
        assert(version == 0);
        let nbytes = this._managed_object_offset_size;
        let offset = _unpack_integer(nbytes, heapid, data_offset);
        data_offset += nbytes;
        nbytes = this._managed_object_length_size;
        let size = _unpack_integer(nbytes, heapid, data_offset);
        return this.managed.slice(offset, offset + size);
      } else if (idtype == 1) {
        throw "tiny objectID not supported in FractalHeap";
      } else if (idtype == 2) {
        throw "huge objectID not supported in FractalHeap";
      } else {
        throw "unknown objectID type in FractalHeap";
      }
    }
    _min_size_integer(integer) {
      return this._min_size_nbits(bitSize(integer));
    }
    _min_size_nbits(nbits) {
      return Math.ceil(nbits / 8);
    }
    *_iter_indirect_block(fh, offset, nrows) {
      let header = _unpack_struct_from(this.indirect_block_header, fh, offset);
      offset += this.indirect_block_header_size;
      assert(header.get("signature") == "FHIB");
      let block_offset_bytes = header.get("block_offset");
      let block_offset = block_offset_bytes.reduce((p, c, i) => p + (c << i * 8), 0);
      header.set("block_offset", block_offset);
      let [ndirect, nindirect] = this._indirect_info(nrows);
      let direct_blocks = [];
      for (let i = 0; i < ndirect; i++) {
        let address = struct.unpack_from("<Q", fh, offset)[0];
        offset += 8;
        if (address == UNDEFINED_ADDRESS) {
          break;
        }
        let block_size = this._calc_block_size(i);
        direct_blocks.push([address, block_size]);
      }
      let indirect_blocks = [];
      for (let i = ndirect; i < ndirect + nindirect; i++) {
        let address = struct.unpack_from("<Q", fh, offset)[0];
        offset += 8;
        if (address == UNDEFINED_ADDRESS) {
          break;
        }
        let block_size = this._calc_block_size(i);
        let nrows2 = this._iblock_nrows_from_block_size(block_size);
        indirect_blocks.push([address, nrows2]);
      }
      for (let [address, block_size] of direct_blocks) {
        let obj = this._read_direct_block(fh, address, block_size);
        yield obj;
      }
      for (let [address, nrows2] of indirect_blocks) {
        for (let obj of this._iter_indirect_block(fh, address, nrows2)) {
          yield obj;
        }
      }
    }
    _calc_block_size(iblock) {
      let row = Math.floor(iblock / this.header.get("table_width"));
      return 2 ** Math.max(row - 1, 0) * this.header.get("starting_block_size");
    }
    _iblock_nrows_from_block_size(block_size) {
      let log2_block_size = Math.floor(Math.log2(block_size));
      assert(2 ** log2_block_size == block_size);
      return log2_block_size - this._indirect_nrows_sub;
    }
    _indirect_info(nrows) {
      let table_width = this.header.get("table_width");
      let nobjects = nrows * table_width;
      let ndirect_max = this._max_direct_nrows * table_width;
      let ndirect, nindirect;
      if (nrows <= ndirect_max) {
        ndirect = nobjects;
        nindirect = 0;
      } else {
        ndirect = ndirect_max;
        nindirect = nobjects - ndirect_max;
      }
      return [ndirect, nindirect];
    }
    _int_format(bytelength) {
      return ["B", "H", "I", "Q"][bytelength - 1];
    }
  };
  var FORMAT_SIGNATURE = struct.unpack_from("8s", new Uint8Array([137, 72, 68, 70, 13, 10, 26, 10]).buffer)[0];
  var UNDEFINED_ADDRESS = struct.unpack_from("<Q", new Uint8Array([255, 255, 255, 255, 255, 255, 255, 255]).buffer)[0];
  var SUPERBLOCK_V0 = /* @__PURE__ */ new Map([
    ["format_signature", "8s"],
    ["superblock_version", "B"],
    ["free_storage_version", "B"],
    ["root_group_version", "B"],
    ["reserved_0", "B"],
    ["shared_header_version", "B"],
    ["offset_size", "B"],
    // assume 8
    ["length_size", "B"],
    // assume 8
    ["reserved_1", "B"],
    ["group_leaf_node_k", "H"],
    ["group_internal_node_k", "H"],
    ["file_consistency_flags", "L"],
    ["base_address_lower", "Q"],
    // assume 8 byte addressing
    ["free_space_address", "Q"],
    // assume 8 byte addressing
    ["end_of_file_address", "Q"],
    ["driver_information_address", "Q"]
    // assume 8 byte addressing
  ]);
  var SUPERBLOCK_V0_SIZE = _structure_size(SUPERBLOCK_V0);
  var SUPERBLOCK_V2_V3 = /* @__PURE__ */ new Map([
    ["format_signature", "8s"],
    ["superblock_version", "B"],
    ["offset_size", "B"],
    ["length_size", "B"],
    ["file_consistency_flags", "B"],
    ["base_address", "Q"],
    // assume 8 byte addressing
    ["superblock_extension_address", "Q"],
    // assume 8 byte addressing
    ["end_of_file_address", "Q"],
    // assume 8 byte addressing
    ["root_group_address", "Q"],
    // assume 8 byte addressing
    ["superblock_checksum", "I"]
  ]);
  var SUPERBLOCK_V2_V3_SIZE = _structure_size(SUPERBLOCK_V2_V3);
  var SYMBOL_TABLE_ENTRY = /* @__PURE__ */ new Map([
    ["link_name_offset", "Q"],
    // 8 byte address
    ["object_header_address", "Q"],
    ["cache_type", "I"],
    ["reserved", "I"],
    ["scratch", "16s"]
  ]);
  var SYMBOL_TABLE_ENTRY_SIZE = _structure_size(SYMBOL_TABLE_ENTRY);
  var SYMBOL_TABLE_NODE = /* @__PURE__ */ new Map([
    ["signature", "4s"],
    ["version", "B"],
    ["reserved_0", "B"],
    ["symbols", "H"]
  ]);
  var SYMBOL_TABLE_NODE_SIZE = _structure_size(SYMBOL_TABLE_NODE);
  var LOCAL_HEAP = /* @__PURE__ */ new Map([
    ["signature", "4s"],
    ["version", "B"],
    ["reserved", "3s"],
    ["data_segment_size", "Q"],
    // 8 byte size of lengths
    ["offset_to_free_list", "Q"],
    // 8 bytes size of lengths
    ["address_of_data_segment", "Q"]
    // 8 byte addressing
  ]);
  var GLOBAL_HEAP_HEADER = /* @__PURE__ */ new Map([
    ["signature", "4s"],
    ["version", "B"],
    ["reserved", "3s"],
    ["collection_size", "Q"]
  ]);
  var GLOBAL_HEAP_HEADER_SIZE = _structure_size(GLOBAL_HEAP_HEADER);
  var GLOBAL_HEAP_OBJECT = /* @__PURE__ */ new Map([
    ["object_index", "H"],
    ["reference_count", "H"],
    ["reserved", "I"],
    ["object_size", "Q"]
    // 8 byte addressing,
  ]);
  var GLOBAL_HEAP_OBJECT_SIZE = _structure_size(GLOBAL_HEAP_OBJECT);
  var FRACTAL_HEAP_HEADER = /* @__PURE__ */ new Map([
    ["signature", "4s"],
    ["version", "B"],
    ["object_index_size", "H"],
    ["filter_info_size", "H"],
    ["flags", "B"],
    ["max_managed_object_size", "I"],
    ["next_huge_object_index", "Q"],
    // 8 byte addressing
    ["btree_address_huge_objects", "Q"],
    // 8 byte addressing
    ["managed_freespace_size", "Q"],
    // 8 byte addressing
    ["freespace_manager_address", "Q"],
    // 8 byte addressing
    ["managed_space_size", "Q"],
    // 8 byte addressing
    ["managed_alloc_size", "Q"],
    // 8 byte addressing
    ["next_directblock_iterator_address", "Q"],
    // 8 byte addressing
    ["managed_object_count", "Q"],
    // 8 byte addressing
    ["huge_objects_total_size", "Q"],
    // 8 byte addressing
    ["huge_object_count", "Q"],
    // 8 byte addressing
    ["tiny_objects_total_size", "Q"],
    // 8 byte addressing
    ["tiny_object_count", "Q"],
    // 8 byte addressing
    ["table_width", "H"],
    ["starting_block_size", "Q"],
    // 8 byte addressing
    ["maximum_direct_block_size", "Q"],
    // 8 byte addressing
    ["log2_maximum_heap_size", "H"],
    ["indirect_starting_rows_count", "H"],
    ["root_block_address", "Q"],
    // 8 byte addressing
    ["indirect_current_rows_count", "H"]
  ]);

  // esm/dataobjects.js
  var DataObjects = class {
    /*
    """
    HDF5 DataObjects.
    """
    */
    constructor(fh, offset) {
      let version_hint = struct.unpack_from("<B", fh, offset)[0];
      if (version_hint == 1) {
        var [msgs, msg_data, header] = this._parse_v1_objects(fh, offset);
      } else if (version_hint == "O".charCodeAt(0)) {
        var [msgs, msg_data, header] = this._parse_v2_objects(fh, offset);
      } else {
        throw "InvalidHDF5File('unknown Data Object Header')";
      }
      this.fh = fh;
      this.msgs = msgs;
      this.msg_data = msg_data;
      this.offset = offset;
      this._global_heaps = {};
      this._header = header;
      this._filter_pipeline = null;
      this._chunk_params_set = false;
      this._chunks = null;
      this._chunk_dims = null;
      this._chunk_address = null;
    }
    get dtype() {
      let msg = this.find_msg_type(DATATYPE_MSG_TYPE)[0];
      let msg_offset = msg.get("offset_to_message");
      return new DatatypeMessage(this.fh, msg_offset).dtype;
    }
    get chunks() {
      this._get_chunk_params();
      return this._chunks;
    }
    get shape() {
      let msg = this.find_msg_type(DATASPACE_MSG_TYPE)[0];
      let msg_offset = msg.get("offset_to_message");
      return determine_data_shape(this.fh, msg_offset);
    }
    get filter_pipeline() {
      if (this._filter_pipeline != null) {
        return this._filter_pipeline;
      }
      let filter_msgs = this.find_msg_type(DATA_STORAGE_FILTER_PIPELINE_MSG_TYPE);
      if (!filter_msgs.length) {
        this._filter_pipeline = null;
        return this._filter_pipeline;
      }
      var offset = filter_msgs[0].get("offset_to_message");
      let [version, nfilters] = struct.unpack_from("<BB", this.fh, offset);
      offset += struct.calcsize("<BB");
      var filters = [];
      if (version == 1) {
        let [res0, res1] = struct.unpack_from("<HI", this.fh, offset);
        offset += struct.calcsize("<HI");
        for (var _ = 0; _ < nfilters; _++) {
          let filter_info = _unpack_struct_from(
            FILTER_PIPELINE_DESCR_V1,
            this.fh,
            offset
          );
          offset += FILTER_PIPELINE_DESCR_V1_SIZE;
          let padded_name_length = _padded_size(filter_info.get("name_length"), 8);
          let fmt = "<" + padded_name_length.toFixed() + "s";
          let filter_name = struct.unpack_from(fmt, this.fh, offset)[0];
          filter_info.set("filter_name", filter_name);
          offset += padded_name_length;
          fmt = "<" + filter_info.get("client_data_values").toFixed() + "I";
          let client_data = struct.unpack_from(fmt, this.fh, offset);
          filter_info.set("client_data", client_data);
          offset += 4 * filter_info.get("client_data_values");
          if (filter_info.get("client_data_values") % 2) {
            offset += 4;
          }
          filters.push(filter_info);
        }
      } else if (version == 2) {
        for (let nf = 0; nf < nfilters; nf++) {
          let filter_info = /* @__PURE__ */ new Map();
          let buf = this.fh;
          let filter_id = struct.unpack_from("<H", buf, offset)[0];
          offset += 2;
          filter_info.set("filter_id", filter_id);
          let name_length = 0;
          if (filter_id > 255) {
            name_length = struct.unpack_from("<H", buf, offset)[0];
            offset += 2;
          }
          let flags = struct.unpack_from("<H", buf, offset)[0];
          offset += 2;
          let optional = (flags & 1) > 0;
          filter_info.set("optional", optional);
          let num_client_values = struct.unpack_from("<H", buf, offset)[0];
          offset += 2;
          let name;
          if (name_length > 0) {
            name = struct.unpack_from(`${name_length}s`, buf, offset)[0];
            offset += name_length;
          }
          filter_info.set("name", name);
          let client_values = struct.unpack_from(`<${num_client_values}i`, buf, offset);
          offset += 4 * num_client_values;
          filter_info.set("client_data", client_values);
          filter_info.set("client_data_values", num_client_values);
          filters.push(filter_info);
        }
      } else {
        throw `version ${version} is not supported`;
      }
      this._filter_pipeline = filters;
      return this._filter_pipeline;
    }
    find_msg_type(msg_type) {
      return this.msgs.filter(function(m) {
        return m.get("type") == msg_type;
      });
    }
    get_attributes() {
      let attrs = {};
      let attr_msgs = this.find_msg_type(ATTRIBUTE_MSG_TYPE);
      for (let msg of attr_msgs) {
        let offset = msg.get("offset_to_message");
        let [name, value] = this.unpack_attribute(offset);
        attrs[name] = value;
      }
      return attrs;
    }
    get fillvalue() {
      let msg = this.find_msg_type(FILLVALUE_MSG_TYPE)[0];
      var offset = msg.get("offset_to_message");
      var is_defined;
      let version = struct.unpack_from("<B", this.fh, offset)[0];
      var info, size, fillvalue;
      if (version == 1 || version == 2) {
        info = _unpack_struct_from(FILLVAL_MSG_V1V2, this.fh, offset);
        offset += FILLVAL_MSG_V1V2_SIZE;
        is_defined = info.get("fillvalue_defined");
      } else if (version == 3) {
        info = _unpack_struct_from(FILLVAL_MSG_V3, this.fh, offset);
        offset += FILLVAL_MSG_V3_SIZE;
        is_defined = info.get("flags") & 32;
      } else {
        throw 'InvalidHDF5File("Unknown fillvalue msg version: "' + String(version);
      }
      if (is_defined) {
        size = struct.unpack_from("<I", this.fh, offset)[0];
        offset += 4;
      } else {
        size = 0;
      }
      if (size) {
        let [getter, big_endian, size2] = dtype_getter(this.dtype);
        let payload_view = new DataView64(this.fh);
        fillvalue = payload_view[getter](offset, !big_endian, size2);
      } else {
        fillvalue = 0;
      }
      return fillvalue;
    }
    unpack_attribute(offset) {
      let version = struct.unpack_from("<B", this.fh, offset)[0];
      var attr_map, padding_multiple;
      if (version == 1) {
        attr_map = _unpack_struct_from(
          ATTR_MSG_HEADER_V1,
          this.fh,
          offset
        );
        assert(attr_map.get("version") == 1);
        offset += ATTR_MSG_HEADER_V1_SIZE;
        padding_multiple = 8;
      } else if (version == 3) {
        attr_map = _unpack_struct_from(
          ATTR_MSG_HEADER_V3,
          this.fh,
          offset
        );
        assert(attr_map.get("version") == 3);
        offset += ATTR_MSG_HEADER_V3_SIZE;
        padding_multiple = 1;
      } else {
        throw "unsupported attribute message version: " + version;
      }
      let name_size = attr_map.get("name_size");
      let name = struct.unpack_from("<" + name_size.toFixed() + "s", this.fh, offset)[0];
      name = name.replace(/\x00$/, "");
      offset += _padded_size(name_size, padding_multiple);
      var dtype;
      try {
        dtype = new DatatypeMessage(this.fh, offset).dtype;
      } catch (e) {
        console.warn("Attribute " + name + " type not implemented, set to null.");
        return [name, null];
      }
      offset += _padded_size(attr_map.get("datatype_size"), padding_multiple);
      let shape = this.determine_data_shape(this.fh, offset);
      let items = shape.reduce(function(a, b) {
        return a * b;
      }, 1);
      offset += _padded_size(attr_map.get("dataspace_size"), padding_multiple);
      var value = this._attr_value(dtype, this.fh, items, offset);
      if (shape.length == 0) {
        value = value[0];
      } else {
      }
      return [name, value];
    }
    determine_data_shape(buf, offset) {
      let version = struct.unpack_from("<B", buf, offset)[0];
      var header;
      if (version == 1) {
        header = _unpack_struct_from(DATASPACE_MSG_HEADER_V1, buf, offset);
        assert(header.get("version") == 1);
        offset += DATASPACE_MSG_HEADER_V1_SIZE;
      } else if (version == 2) {
        header = _unpack_struct_from(DATASPACE_MSG_HEADER_V2, buf, offset);
        assert(header.get("version") == 2);
        offset += DATASPACE_MSG_HEADER_V2_SIZE;
      } else {
        throw "unknown dataspace message version";
      }
      let ndims = header.get("dimensionality");
      let dim_sizes = struct.unpack_from("<" + ndims.toFixed() + "Q", buf, offset);
      return dim_sizes;
    }
    _attr_value(dtype, buf, count, offset) {
      var value = new Array(count);
      if (dtype instanceof Array) {
        let dtype_class = dtype[0];
        for (var i = 0; i < count; i++) {
          if (dtype_class == "VLEN_STRING") {
            let character_set = dtype[2];
            var [vlen, vlen_data] = this._vlen_size_and_data(buf, offset);
            const encoding = character_set == 0 ? "ascii" : "utf-8";
            const decoder = new TextDecoder(encoding);
            value[i] = decoder.decode(vlen_data);
            offset += 16;
          } else if (dtype_class == "REFERENCE") {
            var address = struct.unpack_from("<Q", buf, offset);
            value[i] = address;
            offset += 8;
          } else if (dtype_class == "VLEN_SEQUENCE") {
            let base_dtype = dtype[1];
            var [vlen, vlen_data] = this._vlen_size_and_data(buf, offset);
            value[i] = this._attr_value(base_dtype, vlen_data, vlen, 0);
            offset += 16;
          } else {
            throw "NotImplementedError";
          }
        }
      } else {
        let [getter, big_endian, size] = dtype_getter(dtype);
        let view = new DataView64(buf, 0);
        for (var i = 0; i < count; i++) {
          value[i] = view[getter](offset, !big_endian, size);
          offset += size;
        }
      }
      return value;
    }
    _vlen_size_and_data(buf, offset) {
      let vlen_size = struct.unpack_from("<I", buf, offset)[0];
      let gheap_id = _unpack_struct_from(GLOBAL_HEAP_ID, buf, offset + 4);
      let gheap_address = gheap_id.get("collection_address");
      assert(gheap_id.get("collection_address") < Number.MAX_SAFE_INTEGER);
      var gheap;
      if (!(gheap_address in this._global_heaps)) {
        gheap = new GlobalHeap(this.fh, gheap_address);
        this._global_heaps[gheap_address] = gheap;
      }
      gheap = this._global_heaps[gheap_address];
      let vlen_data = gheap.objects.get(gheap_id.get("object_index"));
      return [vlen_size, vlen_data];
    }
    _parse_v1_objects(buf, offset) {
      let header = _unpack_struct_from(OBJECT_HEADER_V1, buf, offset);
      assert(header.get("version") == 1);
      let total_header_messages = header.get("total_header_messages");
      var block_size = header.get("object_header_size");
      var block_offset = offset + _structure_size(OBJECT_HEADER_V1);
      var msg_data = buf.slice(block_offset, block_offset + block_size);
      var object_header_blocks = [[block_offset, block_size]];
      var current_block = 0;
      var local_offset = 0;
      var msgs = new Array(total_header_messages);
      for (var i = 0; i < total_header_messages; i++) {
        if (local_offset >= block_size) {
          [block_offset, block_size] = object_header_blocks[++current_block];
          local_offset = 0;
        }
        let msg = _unpack_struct_from(HEADER_MSG_INFO_V1, buf, block_offset + local_offset);
        let offset_to_message = block_offset + local_offset + HEADER_MSG_INFO_V1_SIZE;
        msg.set("offset_to_message", offset_to_message);
        if (msg.get("type") == OBJECT_CONTINUATION_MSG_TYPE) {
          var [fh_off, size] = struct.unpack_from("<QQ", buf, offset_to_message);
          object_header_blocks.push([fh_off, size]);
        }
        local_offset += HEADER_MSG_INFO_V1_SIZE + msg.get("size");
        msgs[i] = msg;
      }
      return [msgs, msg_data, header];
    }
    _parse_v2_objects(buf, offset) {
      var [header, creation_order_size, block_offset] = this._parse_v2_header(buf, offset);
      offset = block_offset;
      var msgs = [];
      var block_size = header.get("size_of_chunk_0");
      var msg_data = buf.slice(offset, offset += block_size);
      var object_header_blocks = [[block_offset, block_size]];
      var current_block = 0;
      var local_offset = 0;
      while (true) {
        if (local_offset >= block_size - HEADER_MSG_INFO_V2_SIZE) {
          let next_block = object_header_blocks[++current_block];
          if (next_block == null) {
            break;
          }
          [block_offset, block_size] = next_block;
          local_offset = 0;
        }
        let msg = _unpack_struct_from(HEADER_MSG_INFO_V2, buf, block_offset + local_offset);
        let offset_to_message = block_offset + local_offset + HEADER_MSG_INFO_V2_SIZE + creation_order_size;
        msg.set("offset_to_message", offset_to_message);
        if (msg.get("type") == OBJECT_CONTINUATION_MSG_TYPE) {
          var [fh_off, size] = struct.unpack_from("<QQ", buf, offset_to_message);
          object_header_blocks.push([fh_off + 4, size - 4]);
        }
        local_offset += HEADER_MSG_INFO_V2_SIZE + msg.get("size") + creation_order_size;
        msgs.push(msg);
      }
      return [msgs, msg_data, header];
    }
    _parse_v2_header(buf, offset) {
      let header = _unpack_struct_from(OBJECT_HEADER_V2, buf, offset);
      var creation_order_size;
      offset += _structure_size(OBJECT_HEADER_V2);
      assert(header.get("version") == 2);
      if (header.get("flags") & 4) {
        creation_order_size = 2;
      } else {
        creation_order_size = 0;
      }
      assert((header.get("flags") & 16) == 0);
      if (header.get("flags") & 32) {
        let times = struct.unpack_from("<4I", buf, offset);
        offset += 16;
        header.set("access_time", times[0]);
        header.set("modification_time", times[1]);
        header.set("change_time", times[2]);
        header.set("birth_time", times[3]);
      }
      let chunk_fmt = ["<B", "<H", "<I", "<Q"][header.get("flags") & 3];
      header.set("size_of_chunk_0", struct.unpack_from(chunk_fmt, buf, offset)[0]);
      offset += struct.calcsize(chunk_fmt);
      return [header, creation_order_size, offset];
    }
    get_links() {
      return Object.fromEntries(this.iter_links());
    }
    *iter_links() {
      for (let msg of this.msgs) {
        if (msg.get("type") == SYMBOL_TABLE_MSG_TYPE) {
          yield* this._iter_links_from_symbol_tables(msg);
        } else if (msg.get("type") == LINK_MSG_TYPE) {
          yield this._get_link_from_link_msg(msg);
        } else if (msg.get("type") == LINK_INFO_MSG_TYPE) {
          yield* this._iter_link_from_link_info_msg(msg);
        }
      }
    }
    *_iter_links_from_symbol_tables(sym_tbl_msg) {
      assert(sym_tbl_msg.get("size") == 16);
      let data = _unpack_struct_from(
        // NOTE: using this.fh instead of this.msg_data - needs to be fixed in py file?
        SYMBOL_TABLE_MSG,
        this.fh,
        sym_tbl_msg.get("offset_to_message")
      );
      yield* this._iter_links_btree_v1(data.get("btree_address"), data.get("heap_address"));
    }
    *_iter_links_btree_v1(btree_address, heap_address) {
      let btree = new BTreeV1Groups(this.fh, btree_address);
      let heap = new Heap(this.fh, heap_address);
      for (let symbol_table_address of btree.symbol_table_addresses()) {
        let table = new SymbolTable(this.fh, symbol_table_address);
        table.assign_name(heap);
        yield* Object.entries(table.get_links(heap));
      }
    }
    _get_link_from_link_msg(link_msg) {
      let offset = link_msg.get("offset_to_message");
      return this._decode_link_msg(this.fh, offset)[1];
      ;
    }
    _decode_link_msg(data, offset) {
      let [version, flags] = struct.unpack_from("<BB", data, offset);
      offset += 2;
      assert(version == 1);
      let size_of_length_of_link_name = 2 ** (flags & 3);
      let link_type_field_present = (flags & 2 ** 3) > 0;
      let link_name_character_set_field_present = (flags & 2 ** 4) > 0;
      let ordered = (flags & 2 ** 2) > 0;
      let link_type;
      if (link_type_field_present) {
        link_type = struct.unpack_from("<B", data, offset)[0];
        offset += 1;
      } else {
        link_type = 0;
      }
      assert([0, 1].includes(link_type));
      let creationorder;
      if (ordered) {
        creationorder = struct.unpack_from("<Q", data, offset)[0];
        offset += 8;
      }
      let link_name_character_set = 0;
      if (link_name_character_set_field_present) {
        link_name_character_set = struct.unpack_from("<B", data, offset)[0];
        offset += 1;
      }
      let encoding = link_name_character_set == 0 ? "ascii" : "utf-8";
      let name_size_fmt = ["<B", "<H", "<I", "<Q"][flags & 3];
      let name_size = struct.unpack_from(name_size_fmt, data, offset)[0];
      offset += size_of_length_of_link_name;
      let name = new TextDecoder(encoding).decode(data.slice(offset, offset + name_size));
      offset += name_size;
      let address;
      if (link_type == 0) {
        address = struct.unpack_from("<Q", data, offset)[0];
      } else if (link_type == 1) {
        let length_of_soft_link_value = struct.unpack_from("<H", data, offset)[0];
        offset += 2;
        address = new TextDecoder(encoding).decode(data.slice(offset, offset + length_of_soft_link_value));
      }
      return [creationorder, [name, address]];
    }
    *_iter_link_from_link_info_msg(info_msg) {
      let offset = info_msg.get("offset_to_message");
      let data = this._decode_link_info_msg(this.fh, offset);
      let heap_address = data.get("heap_address");
      let name_btree_address = data.get("name_btree_address");
      let order_btree_address = data.get("order_btree_address");
      if (name_btree_address != null) {
        yield* this._iter_links_btree_v2(name_btree_address, order_btree_address, heap_address);
      }
    }
    *_iter_links_btree_v2(name_btree_address, order_btree_address, heap_address) {
      let heap = new FractalHeap(this.fh, heap_address);
      let btree;
      const ordered = order_btree_address != null;
      if (ordered) {
        btree = new BTreeV2GroupOrders(this.fh, order_btree_address);
      } else {
        btree = new BTreeV2GroupNames(this.fh, name_btree_address);
      }
      let items = /* @__PURE__ */ new Map();
      for (let record of btree.iter_records()) {
        let data = heap.get_data(record.get("heapid"));
        let [creationorder, item] = this._decode_link_msg(data, 0);
        const key = ordered ? creationorder : item[0];
        items.set(key, item);
      }
      let sorted_keys = Array.from(items.keys()).sort();
      for (let key of sorted_keys) {
        yield items.get(key);
      }
    }
    _decode_link_info_msg(data, offset) {
      let [version, flags] = struct.unpack_from("<BB", data, offset);
      assert(version == 0);
      offset += 2;
      if ((flags & 1) > 0) {
        offset += 8;
      }
      let fmt = (flags & 2) > 0 ? LINK_INFO_MSG2 : LINK_INFO_MSG1;
      let link_info = _unpack_struct_from(fmt, data, offset);
      let output = /* @__PURE__ */ new Map();
      for (let [k, v] of link_info.entries()) {
        output.set(k, v == UNDEFINED_ADDRESS2 ? null : v);
      }
      return output;
    }
    get is_dataset() {
      return this.find_msg_type(DATASPACE_MSG_TYPE).length > 0;
    }
    /**
     * Return the data pointed to in the DataObject
     *
     * @returns {Array}
     * @memberof DataObjects
     */
    get_data() {
      let msg = this.find_msg_type(DATA_STORAGE_MSG_TYPE)[0];
      let msg_offset = msg.get("offset_to_message");
      var [version, dims, layout_class, property_offset] = this._get_data_message_properties(msg_offset);
      if (layout_class == 0) {
        throw "Compact storage of DataObject not implemented";
      } else if (layout_class == 1) {
        return this._get_contiguous_data(property_offset);
      } else if (layout_class == 2) {
        return this._get_chunked_data(msg_offset);
      }
    }
    _get_data_message_properties(msg_offset) {
      let dims, layout_class, property_offset;
      let [version, arg1, arg2] = struct.unpack_from(
        "<BBB",
        this.fh,
        msg_offset
      );
      if (version == 1 || version == 2) {
        dims = arg1;
        layout_class = arg2;
        property_offset = msg_offset;
        property_offset += struct.calcsize("<BBB");
        property_offset += struct.calcsize("<BI");
        assert(layout_class == 1 || layout_class == 2);
      } else if (version == 3 || version == 4) {
        layout_class = arg1;
        property_offset = msg_offset;
        property_offset += struct.calcsize("<BB");
      }
      assert(version >= 1 && version <= 4);
      return [version, dims, layout_class, property_offset];
    }
    _get_contiguous_data(property_offset) {
      let [data_offset] = struct.unpack_from("<Q", this.fh, property_offset);
      if (data_offset == UNDEFINED_ADDRESS2) {
        let size = this.shape.reduce(function(a, b) {
          return a * b;
        }, 1);
        return new Array(size);
      }
      var fullsize = this.shape.reduce(function(a, b) {
        return a * b;
      }, 1);
      if (!(this.dtype instanceof Array)) {
        let dtype = this.dtype;
        if (/[<>=!@\|]?(i|u|f|S)(\d*)/.test(dtype)) {
          let [item_getter, item_is_big_endian, item_size] = dtype_getter(dtype);
          let output = new Array(fullsize);
          let view = new DataView64(this.fh);
          for (var i = 0; i < fullsize; i++) {
            output[i] = view[item_getter](data_offset + i * item_size, !item_is_big_endian, item_size);
          }
          return output;
        } else {
          throw "not Implemented - no proper dtype defined";
        }
      } else {
        let dtype_class = this.dtype[0];
        if (dtype_class == "REFERENCE") {
          let size = this.dtype[1];
          if (size != 8) {
            throw "NotImplementedError('Unsupported Reference type')";
          }
          let ref_addresses = this.fh.slice(data_offset, data_offset + fullsize);
          return ref_addresses;
        } else if (dtype_class == "VLEN_STRING") {
          let character_set = this.dtype[2];
          const encoding = character_set == 0 ? "ascii" : "utf-8";
          const decoder = new TextDecoder(encoding);
          var value = [];
          for (var i = 0; i < fullsize; i++) {
            const [vlen, vlen_data] = this._vlen_size_and_data(this.fh, data_offset);
            value[i] = decoder.decode(vlen_data);
            data_offset += 16;
          }
          return value;
        } else {
          throw "NotImplementedError('datatype not implemented')";
        }
      }
    }
    _get_chunked_data(offset) {
      this._get_chunk_params();
      if (this._chunk_address == UNDEFINED_ADDRESS2) {
        return [];
      }
      var chunk_btree = new BTreeV1RawDataChunks(
        this.fh,
        this._chunk_address,
        this._chunk_dims
      );
      let data = chunk_btree.construct_data_from_chunks(
        this.chunks,
        this.shape,
        this.dtype,
        this.filter_pipeline
      );
      if (this.dtype instanceof Array && /^VLEN/.test(this.dtype[0])) {
        let dtype_class = this.dtype[0];
        for (var i = 0; i < data.length; i++) {
          let [item_size, gheap_address, object_index] = data[i];
          var gheap;
          if (!(gheap_address in this._global_heaps)) {
            gheap = new GlobalHeap(this.fh, gheap_address);
            this._global_heaps[gheap_address] = gheap;
          } else {
            gheap = this._global_heaps[gheap_address];
          }
          let vlen_data = gheap.objects.get(object_index);
          if (dtype_class == "VLEN_STRING") {
            let character_set = this.dtype[2];
            const encoding = character_set == 0 ? "ascii" : "utf-8";
            const decoder = new TextDecoder(encoding);
            data[i] = decoder.decode(vlen_data);
          }
        }
      }
      return data;
    }
    _get_chunk_params() {
      if (this._chunk_params_set) {
        return;
      }
      this._chunk_params_set = true;
      var msg = this.find_msg_type(DATA_STORAGE_MSG_TYPE)[0];
      var offset = msg.get("offset_to_message");
      var [version, dims, layout_class, property_offset] = this._get_data_message_properties(offset);
      if (layout_class != 2) {
        return;
      }
      var data_offset;
      if (version == 1 || version == 2) {
        var address = struct.unpack_from("<Q", this.fh, property_offset)[0];
        data_offset = property_offset + struct.calcsize("<Q");
      } else if (version == 3) {
        var [dims, address] = struct.unpack_from(
          "<BQ",
          this.fh,
          property_offset
        );
        data_offset = property_offset + struct.calcsize("<BQ");
      }
      assert(version >= 1 && version <= 3);
      var fmt = "<" + (dims - 1).toFixed() + "I";
      var chunk_shape = struct.unpack_from(fmt, this.fh, data_offset);
      this._chunks = chunk_shape;
      this._chunk_dims = dims;
      this._chunk_address = address;
      return;
    }
  };
  function determine_data_shape(buf, offset) {
    let version = struct.unpack_from("<B", buf, offset)[0];
    var header;
    if (version == 1) {
      header = _unpack_struct_from(DATASPACE_MSG_HEADER_V1, buf, offset);
      assert(header.get("version") == 1);
      offset += DATASPACE_MSG_HEADER_V1_SIZE;
    } else if (version == 2) {
      header = _unpack_struct_from(DATASPACE_MSG_HEADER_V2, buf, offset);
      assert(header.get("version") == 2);
      offset += DATASPACE_MSG_HEADER_V2_SIZE;
    } else {
      throw "InvalidHDF5File('unknown dataspace message version')";
    }
    let ndims = header.get("dimensionality");
    let dim_sizes = struct.unpack_from("<" + (ndims * 2).toFixed() + "I", buf, offset);
    return dim_sizes.filter(function(s, i) {
      return i % 2 == 0;
    });
  }
  var UNDEFINED_ADDRESS2 = struct.unpack_from("<Q", new Uint8Array([255, 255, 255, 255, 255, 255, 255, 255]).buffer);
  var GLOBAL_HEAP_ID = /* @__PURE__ */ new Map([
    ["collection_address", "Q"],
    //# 8 byte addressing,
    ["object_index", "I"]
  ]);
  var GLOBAL_HEAP_ID_SIZE = _structure_size(GLOBAL_HEAP_ID);
  var ATTR_MSG_HEADER_V1 = /* @__PURE__ */ new Map([
    ["version", "B"],
    ["reserved", "B"],
    ["name_size", "H"],
    ["datatype_size", "H"],
    ["dataspace_size", "H"]
  ]);
  var ATTR_MSG_HEADER_V1_SIZE = _structure_size(ATTR_MSG_HEADER_V1);
  var ATTR_MSG_HEADER_V3 = /* @__PURE__ */ new Map([
    ["version", "B"],
    ["flags", "B"],
    ["name_size", "H"],
    ["datatype_size", "H"],
    ["dataspace_size", "H"],
    ["character_set_encoding", "B"]
  ]);
  var ATTR_MSG_HEADER_V3_SIZE = _structure_size(ATTR_MSG_HEADER_V3);
  var OBJECT_HEADER_V1 = /* @__PURE__ */ new Map([
    ["version", "B"],
    ["reserved", "B"],
    ["total_header_messages", "H"],
    ["object_reference_count", "I"],
    ["object_header_size", "I"],
    ["padding", "I"]
  ]);
  var OBJECT_HEADER_V2 = /* @__PURE__ */ new Map([
    ["signature", "4s"],
    ["version", "B"],
    ["flags", "B"]
    //# Access time (optional)
    //# Modification time (optional)
    //# Change time (optional)
    //# Birth time (optional)
    //# Maximum # of compact attributes
    //# Maximum # of dense attributes
    //# Size of Chunk #0
  ]);
  var DATASPACE_MSG_HEADER_V1 = /* @__PURE__ */ new Map([
    ["version", "B"],
    ["dimensionality", "B"],
    ["flags", "B"],
    ["reserved_0", "B"],
    ["reserved_1", "I"]
  ]);
  var DATASPACE_MSG_HEADER_V1_SIZE = _structure_size(DATASPACE_MSG_HEADER_V1);
  var DATASPACE_MSG_HEADER_V2 = /* @__PURE__ */ new Map([
    ["version", "B"],
    ["dimensionality", "B"],
    ["flags", "B"],
    ["type", "B"]
  ]);
  var DATASPACE_MSG_HEADER_V2_SIZE = _structure_size(DATASPACE_MSG_HEADER_V2);
  var HEADER_MSG_INFO_V1 = /* @__PURE__ */ new Map([
    ["type", "H"],
    ["size", "H"],
    ["flags", "B"],
    ["reserved", "3s"]
  ]);
  var HEADER_MSG_INFO_V1_SIZE = _structure_size(HEADER_MSG_INFO_V1);
  var HEADER_MSG_INFO_V2 = /* @__PURE__ */ new Map([
    ["type", "B"],
    ["size", "H"],
    ["flags", "B"]
  ]);
  var HEADER_MSG_INFO_V2_SIZE = _structure_size(HEADER_MSG_INFO_V2);
  var SYMBOL_TABLE_MSG = /* @__PURE__ */ new Map([
    ["btree_address", "Q"],
    //# 8 bytes addressing
    ["heap_address", "Q"]
    //# 8 byte addressing
  ]);
  var LINK_INFO_MSG1 = /* @__PURE__ */ new Map([
    ["heap_address", "Q"],
    // 8 byte addressing
    ["name_btree_address", "Q"]
    // 8 bytes addressing
  ]);
  var LINK_INFO_MSG2 = /* @__PURE__ */ new Map([
    ["heap_address", "Q"],
    // 8 byte addressing
    ["name_btree_address", "Q"],
    // 8 bytes addressing
    ["order_btree_address", "Q"]
    // 8 bytes addressing
  ]);
  var FILLVAL_MSG_V1V2 = /* @__PURE__ */ new Map([
    ["version", "B"],
    ["space_allocation_time", "B"],
    ["fillvalue_write_time", "B"],
    ["fillvalue_defined", "B"]
  ]);
  var FILLVAL_MSG_V1V2_SIZE = _structure_size(FILLVAL_MSG_V1V2);
  var FILLVAL_MSG_V3 = /* @__PURE__ */ new Map([
    ["version", "B"],
    ["flags", "B"]
  ]);
  var FILLVAL_MSG_V3_SIZE = _structure_size(FILLVAL_MSG_V3);
  var FILTER_PIPELINE_DESCR_V1 = /* @__PURE__ */ new Map([
    ["filter_id", "H"],
    ["name_length", "H"],
    ["flags", "H"],
    ["client_data_values", "H"]
  ]);
  var FILTER_PIPELINE_DESCR_V1_SIZE = _structure_size(FILTER_PIPELINE_DESCR_V1);
  var DATASPACE_MSG_TYPE = 1;
  var LINK_INFO_MSG_TYPE = 2;
  var DATATYPE_MSG_TYPE = 3;
  var FILLVALUE_MSG_TYPE = 5;
  var LINK_MSG_TYPE = 6;
  var DATA_STORAGE_MSG_TYPE = 8;
  var DATA_STORAGE_FILTER_PIPELINE_MSG_TYPE = 11;
  var ATTRIBUTE_MSG_TYPE = 12;
  var OBJECT_CONTINUATION_MSG_TYPE = 16;
  var SYMBOL_TABLE_MSG_TYPE = 17;

  // esm/high-level.js
  var Group = class _Group {
    /*
      An HDF5 Group which may hold attributes, datasets, or other groups.
      Attributes
      ----------
      attrs : dict
          Attributes for this group.
      name : str
          Full path to this group.
      file : File
          File instance where this group resides.
      parent : Group
          Group instance containing this group.
    */
    /**
     *
     *
     * @memberof Group
     * @member {Group|File} parent;
     * @member {File} file;
     * @member {string} name;
     * @member {DataObjects} _dataobjects;
     * @member {Object} _attrs;
     * @member {Array<string>} _keys;
     */
    // parent;
    // file;
    // name;
    // _links;
    // _dataobjects;
    // _attrs;
    // _keys;
    /**
     * 
     * @param {string} name 
     * @param {DataObjects} dataobjects 
     * @param {Group} [parent] 
     * @param {boolean} [getterProxy=false]
     * @returns {Group}
     */
    constructor(name, dataobjects, parent, getterProxy = false) {
      if (parent == null) {
        this.parent = this;
        this.file = this;
      } else {
        this.parent = parent;
        this.file = parent.file;
      }
      this.name = name;
      this._links = dataobjects.get_links();
      this._dataobjects = dataobjects;
      this._attrs = null;
      this._keys = null;
      if (getterProxy) {
        return new Proxy(this, groupGetHandler);
      }
    }
    get keys() {
      if (this._keys == null) {
        this._keys = Object.keys(this._links);
      }
      return this._keys.slice();
    }
    get values() {
      return this.keys.map((k) => this.get(k));
    }
    length() {
      return this.keys.length;
    }
    _dereference(ref) {
      if (!ref) {
        throw "cannot deference null reference";
      }
      let obj = this.file._get_object_by_address(ref);
      if (obj == null) {
        throw "reference not found in file";
      }
      return obj;
    }
    get(y) {
      if (typeof y == "number") {
        return this._dereference(y);
      }
      var path = normpath(y);
      if (path == "/") {
        return this.file;
      }
      if (path == ".") {
        return this;
      }
      if (/^\//.test(path)) {
        return this.file.get(path.slice(1));
      }
      if (posix_dirname(path) != "") {
        var [next_obj, additional_obj] = path.split(/\/(.*)/);
      } else {
        var next_obj = path;
        var additional_obj = ".";
      }
      if (!(next_obj in this._links)) {
        throw next_obj + " not found in group";
      }
      var obj_name = normpath(this.name + "/" + next_obj);
      let link_target = this._links[next_obj];
      if (typeof link_target == "string") {
        try {
          return this.get(link_target);
        } catch (error) {
          return null;
        }
      }
      var dataobjs = new DataObjects(this.file._fh, link_target);
      if (dataobjs.is_dataset) {
        if (additional_obj != ".") {
          throw obj_name + " is a dataset, not a group";
        }
        return new Dataset(obj_name, dataobjs, this);
      } else {
        var new_group = new _Group(obj_name, dataobjs, this);
        return new_group.get(additional_obj);
      }
    }
    visit(func) {
      return this.visititems((name, obj) => func(name));
    }
    visititems(func) {
      var root_name_length = this.name.length;
      if (!/\/$/.test(this.name)) {
        root_name_length += 1;
      }
      var queue = this.values.slice();
      while (queue) {
        let obj = queue.shift();
        if (queue.length == 1) console.log(obj);
        let name = obj.name.slice(root_name_length);
        let ret = func(name, obj);
        if (ret != null) {
          return ret;
        }
        if (obj instanceof _Group) {
          queue = queue.concat(obj.values);
        }
      }
      return null;
    }
    get attrs() {
      if (this._attrs == null) {
        this._attrs = this._dataobjects.get_attributes();
      }
      return this._attrs;
    }
  };
  var groupGetHandler = {
    get: function(target, prop, receiver) {
      if (prop in target) {
        return target[prop];
      }
      return target.get(prop);
    }
  };
  var File = class extends Group {
    /*
    Open a HDF5 file.
    Note in addition to having file specific methods the File object also
    inherit the full interface of **Group**.
    File is also a context manager and therefore supports the with statement.
    Files opened by the class will be closed after the with block, file-like
    object are not closed.
    Parameters
    ----------
    filename : str or file-like
        Name of file (string or unicode) or file like object which has read
        and seek methods which behaved like a Python file object.
    Attributes
    ----------
    filename : str
        Name of the file on disk, None if not available.
    mode : str
        String indicating that the file is open readonly ("r").
    userblock_size : int
        Size of the user block in bytes (currently always 0).
    */
    constructor(fh, filename) {
      var superblock = new SuperBlock(fh, 0);
      var offset = superblock.offset_to_dataobjects;
      var dataobjects = new DataObjects(fh, offset);
      super("/", dataobjects, null);
      this.parent = this;
      this._fh = fh;
      this.filename = filename || "";
      this.file = this;
      this.mode = "r";
      this.userblock_size = 0;
    }
    _get_object_by_address(obj_addr) {
      if (this._dataobjects.offset == obj_addr) {
        return this;
      }
      return this.visititems(
        (y) => {
          y._dataobjects.offset == obj_addr ? y : null;
        }
      );
    }
  };
  var Dataset = class extends Array {
    /*
    A HDF5 Dataset containing an n-dimensional array and meta-data attributes.
    Attributes
    ----------
    shape : tuple
        Dataset dimensions.
    dtype : dtype
        Dataset's type.
    size : int
        Total number of elements in the dataset.
    chunks : tuple or None
        Chunk shape, or NOne is chunked storage not used.
    compression : str or None
        Compression filter used on dataset.  None if compression is not enabled
        for this dataset.
    compression_opts : dict or None
        Options for the compression filter.
    scaleoffset : dict or None
        Setting for the HDF5 scale-offset filter, or None if scale-offset
        compression is not used for this dataset.
    shuffle : bool
        Whether the shuffle filter is applied for this dataset.
    fletcher32 : bool
        Whether the Fletcher32 checksumming is enabled for this dataset.
    fillvalue : float or None
        Value indicating uninitialized portions of the dataset. None is no fill
        values has been defined.
    dim : int
        Number of dimensions.
    dims : None
        Dimension scales.
    attrs : dict
        Attributes for this dataset.
    name : str
        Full path to this dataset.
    file : File
        File instance where this dataset resides.
    parent : Group
        Group instance containing this dataset.
    */
    /**
     *
     *
     * @memberof Dataset
     * @member {Group|File} parent;
     * @member {File} file;
     * @member {string} name;
     * @member {DataObjects} _dataobjects;
     * @member {Object} _attrs;
     * @member {string} _astype;
     */
    // parent;
    // file;
    // name;
    // _dataobjects;
    // _attrs;
    // _astype;
    constructor(name, dataobjects, parent) {
      super();
      this.parent = parent;
      this.file = parent.file;
      this.name = name;
      this._dataobjects = dataobjects;
      this._attrs = null;
      this._astype = null;
    }
    get value() {
      var data = this._dataobjects.get_data();
      if (this._astype == null) {
        return data;
      }
      return data.astype(this._astype);
    }
    get shape() {
      return this._dataobjects.shape;
    }
    get attrs() {
      return this._dataobjects.get_attributes();
    }
    get dtype() {
      return this._dataobjects.dtype;
    }
    get fillvalue() {
      return this._dataobjects.fillvalue;
    }
  };
  function posix_dirname(p) {
    let sep = "/";
    let i = p.lastIndexOf(sep) + 1;
    let head = p.slice(0, i);
    let all_sep = new RegExp("^" + sep + "+$");
    let end_sep = new RegExp(sep + "$");
    if (head && !all_sep.test(head)) {
      head = head.replace(end_sep, "");
    }
    return head;
  }
  function normpath(path) {
    return path.replace(/\/(\/)+/g, "/");
  }
  return __toCommonJS(index_exports);
})();
/*! Bundled license information:

pako/dist/pako.esm.mjs:
  (*! pako 2.1.0 https://github.com/nodeca/pako @license (MIT AND Zlib) *)
*/

// ─────────────────────────────────────────────────────────────────
// Local additions — keep at end of file.
//
// 1. Strip the sourceMappingURL pragma to avoid 404s in the deployed
//    site (the .map file is not vendored). The vendor build's pragma
//    above is harmless when the map is absent but pollutes DevTools.
// 2. Mirror `var hdf5 = …` onto `globalThis.hdf5` explicitly. In a
//    browser document `var` already attaches to window, but inside a
//    classic Web Worker (where worker.js imports us via importScripts)
//    we want the same name resolution path the page uses.
//    `formats/_mat73.js` consumes the global from there.
// ─────────────────────────────────────────────────────────────────
if (typeof globalThis !== 'undefined' && typeof hdf5 !== 'undefined') {
  globalThis.hdf5 = hdf5;
}
