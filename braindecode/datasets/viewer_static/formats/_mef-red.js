/* ============================================================
   formats/_mef-red.js — MEF3 RED (Range Encoded Differential)
   block codec. JS port of meflib's RED_decode (+ RED_encode_exec
   for test-fixture generation).

   Spec source: msel-source/meflib (Apache 2.0).
     https://github.com/msel-source/meflib (meflib/meflib.c)

   For the full bit-level spec see formats/_mef-red-spec.md. Every
   non-trivial decision below cross-references the corresponding
   meflib.c line number so a future maintainer can audit against
   upstream without re-reading C.

   What this module provides:
     - parseBlockHeader(blockBytes, blockOffset)     -> header struct
     - decodeBlock(blockBytes, blockOffset, {validateCrc?})
                                                     -> Int32Array
     - encodeBlock(samples, {scaleFactor?, startTime?, discontinuity?,
                             pad8?, validateCrc?})
                                                     -> Uint8Array
     - the constants (RED_BLOCK_HEADER_BYTES, sentinel values, ...)

   The codec is round-trip-safe: encodeBlock(samples) → decodeBlock →
   samples is verified in tests/unit-mef-red.test.mjs.

   What this module does NOT do:
     - AES decryption (encrypted MEF3 blocks throw).
     - Lossy compression modes that need normality testing (encoder
       only supports lossless + fixed scale factor — same on-disk
       layout as upstream, but we don't expose the lossy chooser).
   ============================================================ */
(function () {
  'use strict';

  // ---- Constants — meflib.h L1035-1071 -----------------------------
  const RED_BLOCK_HEADER_BYTES         = 304;
  const RED_BLOCK_STATISTICS_OFFSET    = 48;
  const RED_BLOCK_STATISTICS_BYTES     = 256;

  // Block-header field offsets (meflib.h L1037-1050)
  const OFF_BLOCK_CRC                  = 0;     // ui4
  const OFF_FLAGS                      = 4;     // ui1
  const OFF_DETREND_SLOPE              = 16;    // sf4
  const OFF_DETREND_INTERCEPT          = 20;    // sf4
  const OFF_SCALE_FACTOR               = 24;    // sf4
  const OFF_DIFFERENCE_BYTES           = 28;    // ui4
  const OFF_NUMBER_OF_SAMPLES          = 32;    // ui4
  const OFF_BLOCK_BYTES                = 36;    // ui4
  const OFF_START_TIME                 = 40;    // si8

  // Flag masks (meflib.h L1054-1056)
  const RED_DISCONTINUITY_MASK         = 0x01;
  const RED_LEVEL_1_ENCRYPTION_MASK    = 0x02;
  const RED_LEVEL_2_ENCRYPTION_MASK    = 0x04;

  // Range coder constants (meflib.h L1066-1071)
  const TOP_VALUE                      = 0x80000000 >>> 0;
  const TOP_VALUE_MINUS_1              = 0x7FFFFFFF >>> 0;
  const CARRY_CHECK                    = 0x7F800000 >>> 0;
  const SHIFT_BITS                     = 23;
  const EXTRA_BITS                     = 7;
  const BOTTOM_VALUE                   = 0x00800000 >>> 0;

  // Sentinel sample values (meflib.h L1059-1063) — exposed for callers
  // that want to filter them out before plotting.
  const RED_NAN                        = -2147483648;   // 0x80000000
  const RED_NEGATIVE_INFINITY          = -2147483647;   // 0x80000001
  const RED_POSITIVE_INFINITY          = 2147483647;    // 0x7FFFFFFF

  // Pad byte at end of block to reach 8-byte alignment (meflib.h
  // PAD_BYTE_VALUE — defined elsewhere in meflib but matches what
  // RED_encode_exec writes at L7008).
  const PAD_BYTE_VALUE                 = 0x7e;

  // ---- Koopman32 CRC (shared with _mef-segment.js) -----------------
  // We re-derive locally so this module is independent of load order.
  // Table verbatim from CRC_KOOPMAN32_KEY (meflib.h L1217-1224).
  const CRC_TABLE = new Uint32Array([
    0x00000000, 0x9695C4CA, 0xFB4839C9, 0x6DDDFD03,
    0x20F3C3CF, 0xB6660705, 0xDBBBFA06, 0x4D2E3ECC,
    0x41E7879E, 0xD7724354, 0xBAAFBE57, 0x2C3A7A9D,
    0x61144451, 0xF781809B, 0x9A5C7D98, 0x0CC9B952,
    0x83CF0F3C, 0x155ACBF6, 0x788736F5, 0xEE12F23F,
    0xA33CCCF3, 0x35A90839, 0x5874F53A, 0xCEE131F0,
    0xC22888A2, 0x54BD4C68, 0x3960B16B, 0xAFF575A1,
    0xE2DB4B6D, 0x744E8FA7, 0x199372A4, 0x8F06B66E,
    0xD1FDAE25, 0x47686AEF, 0x2AB597EC, 0xBC205326,
    0xF10E6DEA, 0x679BA920, 0x0A465423, 0x9CD390E9,
    0x901A29BB, 0x068FED71, 0x6B521072, 0xFDC7D4B8,
    0xB0E9EA74, 0x267C2EBE, 0x4BA1D3BD, 0xDD341777,
    0x5232A119, 0xC4A765D3, 0xA97A98D0, 0x3FEF5C1A,
    0x72C162D6, 0xE454A61C, 0x89895B1F, 0x1F1C9FD5,
    0x13D52687, 0x8540E24D, 0xE89D1F4E, 0x7E08DB84,
    0x3326E548, 0xA5B32182, 0xC86EDC81, 0x5EFB184B,
    0x7598EC17, 0xE30D28DD, 0x8ED0D5DE, 0x18451114,
    0x556B2FD8, 0xC3FEEB12, 0xAE231611, 0x38B6D2DB,
    0x347F6B89, 0xA2EAAF43, 0xCF375240, 0x59A2968A,
    0x148CA846, 0x82196C8C, 0xEFC4918F, 0x79515545,
    0xF657E32B, 0x60C227E1, 0x0D1FDAE2, 0x9B8A1E28,
    0xD6A420E4, 0x4031E42E, 0x2DEC192D, 0xBB79DDE7,
    0xB7B064B5, 0x2125A07F, 0x4CF85D7C, 0xDA6D99B6,
    0x9743A77A, 0x01D663B0, 0x6C0B9EB3, 0xFA9E5A79,
    0xA4654232, 0x32F086F8, 0x5F2D7BFB, 0xC9B8BF31,
    0x849681FD, 0x12034537, 0x7FDEB834, 0xE94B7CFE,
    0xE582C5AC, 0x73170166, 0x1ECAFC65, 0x885F38AF,
    0xC5710663, 0x53E4C2A9, 0x3E393FAA, 0xA8ACFB60,
    0x27AA4D0E, 0xB13F89C4, 0xDCE274C7, 0x4A77B00D,
    0x07598EC1, 0x91CC4A0B, 0xFC11B708, 0x6A8473C2,
    0x664DCA90, 0xF0D80E5A, 0x9D05F359, 0x0B903793,
    0x46BE095F, 0xD02BCD95, 0xBDF63096, 0x2B63F45C,
    0xEB31D82E, 0x7DA41CE4, 0x1079E1E7, 0x86EC252D,
    0xCBC21BE1, 0x5D57DF2B, 0x308A2228, 0xA61FE6E2,
    0xAAD65FB0, 0x3C439B7A, 0x519E6679, 0xC70BA2B3,
    0x8A259C7F, 0x1CB058B5, 0x716DA5B6, 0xE7F8617C,
    0x68FED712, 0xFE6B13D8, 0x93B6EEDB, 0x05232A11,
    0x480D14DD, 0xDE98D017, 0xB3452D14, 0x25D0E9DE,
    0x2919508C, 0xBF8C9446, 0xD2516945, 0x44C4AD8F,
    0x09EA9343, 0x9F7F5789, 0xF2A2AA8A, 0x64376E40,
    0x3ACC760B, 0xAC59B2C1, 0xC1844FC2, 0x57118B08,
    0x1A3FB5C4, 0x8CAA710E, 0xE1778C0D, 0x77E248C7,
    0x7B2BF195, 0xEDBE355F, 0x8063C85C, 0x16F60C96,
    0x5BD8325A, 0xCD4DF690, 0xA0900B93, 0x3605CF59,
    0xB9037937, 0x2F96BDFD, 0x424B40FE, 0xD4DE8434,
    0x99F0BAF8, 0x0F657E32, 0x62B88331, 0xF42D47FB,
    0xF8E4FEA9, 0x6E713A63, 0x03ACC760, 0x953903AA,
    0xD8173D66, 0x4E82F9AC, 0x235F04AF, 0xB5CAC065,
    0x9EA93439, 0x083CF0F3, 0x65E10DF0, 0xF374C93A,
    0xBE5AF7F6, 0x28CF333C, 0x4512CE3F, 0xD3870AF5,
    0xDF4EB3A7, 0x49DB776D, 0x24068A6E, 0xB2934EA4,
    0xFFBD7068, 0x6928B4A2, 0x04F549A1, 0x92608D6B,
    0x1D663B05, 0x8BF3FFCF, 0xE62E02CC, 0x70BBC606,
    0x3D95F8CA, 0xAB003C00, 0xC6DDC103, 0x504805C9,
    0x5C81BC9B, 0xCA147851, 0xA7C98552, 0x315C4198,
    0x7C727F54, 0xEAE7BB9E, 0x873A469D, 0x11AF8257,
    0x4F549A1C, 0xD9C15ED6, 0xB41CA3D5, 0x2289671F,
    0x6FA759D3, 0xF9329D19, 0x94EF601A, 0x027AA4D0,
    0x0EB31D82, 0x9826D948, 0xF5FB244B, 0x636EE081,
    0x2E40DE4D, 0xB8D51A87, 0xD508E784, 0x439D234E,
    0xCC9B9520, 0x5A0E51EA, 0x37D3ACE9, 0xA1466823,
    0xEC6856EF, 0x7AFD9225, 0x17206F26, 0x81B5ABEC,
    0x8D7C12BE, 0x1BE9D674, 0x76342B77, 0xE0A1EFBD,
    0xAD8FD171, 0x3B1A15BB, 0x56C7E8B8, 0xC0522C72,
  ]);

  function crcCalculate(bytes, start, end) {
    if (start == null) start = 0;
    if (end == null) end = bytes.length;
    let crc = 0xFFFFFFFF;
    for (let i = start; i < end; i++) {
      crc = ((crc >>> 8) ^ CRC_TABLE[(crc ^ bytes[i]) & 0xFF]) >>> 0;
    }
    return crc >>> 0;
  }

  // ---- Half-away-from-zero rounding with sentinel clamps -----------
  // RED_round (meflib.c L7322-7334). Pure-JS port: every si4 produced
  // by the decoder eventually passes through this.
  function redRound(val) {
    if (val >= 0) {
      val += 0.5;
      if (val >= RED_POSITIVE_INFINITY) return RED_POSITIVE_INFINITY;
    } else {
      val -= 0.5;
      if (val <= RED_NEGATIVE_INFINITY) return RED_NEGATIVE_INFINITY;
    }
    // Truncate toward zero, matching C (si4) cast.
    return val | 0;
  }

  // ---- Block header parsing ----------------------------------------
  /**
   * Read a RED block header from a byte buffer. `blockOffset` is the
   * offset of the first byte of THIS block (the byte where block_CRC
   * lives) within `bytes`.
   *
   * @param {Uint8Array} bytes
   * @param {number} [blockOffset=0]
   * @returns {{
   *   block_crc: number, flags: number,
   *   detrend_slope: number, detrend_intercept: number,
   *   scale_factor: number,
   *   difference_bytes: number, number_of_samples: number,
   *   block_bytes: number, start_time_low: number, start_time_high: number,
   *   statistics: Uint8Array,
   *   discontinuity: boolean,
   *   encrypted: boolean,
   * }}
   */
  function parseBlockHeader(bytes, blockOffset) {
    if (blockOffset == null) blockOffset = 0;
    if (bytes.length - blockOffset < RED_BLOCK_HEADER_BYTES) {
      throw new Error(
        `mef-red: parseBlockHeader needs ${RED_BLOCK_HEADER_BYTES} bytes ` +
        `at offset ${blockOffset}, got ${bytes.length - blockOffset}`,
      );
    }
    const dv = new DataView(bytes.buffer, bytes.byteOffset + blockOffset, RED_BLOCK_HEADER_BYTES);
    const flags = dv.getUint8(OFF_FLAGS);
    const encrypted =
      (flags & RED_LEVEL_1_ENCRYPTION_MASK) !== 0 ||
      (flags & RED_LEVEL_2_ENCRYPTION_MASK) !== 0;
    return {
      block_crc:          dv.getUint32(OFF_BLOCK_CRC,         true),
      flags,
      detrend_slope:      dv.getFloat32(OFF_DETREND_SLOPE,    true),
      detrend_intercept:  dv.getFloat32(OFF_DETREND_INTERCEPT,true),
      scale_factor:       dv.getFloat32(OFF_SCALE_FACTOR,     true),
      difference_bytes:   dv.getUint32(OFF_DIFFERENCE_BYTES,  true),
      number_of_samples:  dv.getUint32(OFF_NUMBER_OF_SAMPLES, true),
      block_bytes:        dv.getUint32(OFF_BLOCK_BYTES,       true),
      // 64-bit start_time exposed as two 32-bit halves so callers don't
      // pay for a BigInt unless they need full microsecond precision.
      // Most timing math is done at the segment / channel level.
      start_time_low:     dv.getUint32(OFF_START_TIME,        true),
      start_time_high:    dv.getInt32(OFF_START_TIME + 4,     true),
      statistics:         bytes.subarray(
        blockOffset + RED_BLOCK_STATISTICS_OFFSET,
        blockOffset + RED_BLOCK_STATISTICS_OFFSET + RED_BLOCK_STATISTICS_BYTES,
      ),
      discontinuity: (flags & RED_DISCONTINUITY_MASK) !== 0,
      encrypted,
    };
  }

  /**
   * Decode one RED block. Pure port of meflib's RED_decode
   * (meflib.c L6639-6770) for the unencrypted, lossless / fixed-scale
   * path.
   *
   * @param {Uint8Array} bytes - buffer containing the block
   * @param {number} [blockOffset=0] - offset of the block within `bytes`
   * @param {object} [opts]
   * @param {boolean} [opts.validateCrc=true] - verify block CRC32 before
   *   decoding. Disable only for fuzz / forensic decode.
   * @returns {{ samples: Int32Array, header: object }}
   */
  function decodeBlock(bytes, blockOffset, opts) {
    if (blockOffset == null) blockOffset = 0;
    const validateCrc = !opts || opts.validateCrc !== false;

    const header = parseBlockHeader(bytes, blockOffset);

    if (header.encrypted) {
      throw new Error(
        `mef-red: block at offset ${blockOffset} is encrypted ` +
        `(flags=0x${header.flags.toString(16)}); ` +
        `encrypted MEF3 is not supported`,
      );
    }
    if (header.block_bytes < RED_BLOCK_HEADER_BYTES) {
      throw new Error(
        `mef-red: block_bytes=${header.block_bytes} < ` +
        `RED_BLOCK_HEADER_BYTES (${RED_BLOCK_HEADER_BYTES})`,
      );
    }
    if (bytes.length - blockOffset < header.block_bytes) {
      throw new Error(
        `mef-red: block at offset ${blockOffset} declares block_bytes=` +
        `${header.block_bytes} but only ${bytes.length - blockOffset} ` +
        `bytes remain in buffer`,
      );
    }

    if (validateCrc) {
      // meflib.c L6655 — CRC covers bytes [4..block_bytes) of the block.
      const computed = crcCalculate(
        bytes, blockOffset + 4, blockOffset + header.block_bytes,
      );
      if (computed !== header.block_crc) {
        throw new Error(
          `mef-red: block CRC mismatch at offset ${blockOffset}: ` +
          `stored=0x${header.block_crc.toString(16)} computed=0x${computed.toString(16)}`,
        );
      }
    }

    // Edge case: zero-sample block — encoder L6871-6880 writes an empty
    // body (block_bytes = RED_BLOCK_HEADER_BYTES, no payload). Return
    // an empty Int32Array.
    if (header.number_of_samples === 0) {
      return { samples: new Int32Array(0), header };
    }

    // ---- Build cumulative_counts (meflib.c L6703-6708) -------------
    const statistics = header.statistics;
    const cumulative_counts = new Uint32Array(RED_BLOCK_STATISTICS_BYTES + 1);
    {
      let acc = 0;
      cumulative_counts[0] = 0;
      for (let i = 0; i < RED_BLOCK_STATISTICS_BYTES; i++) {
        acc += statistics[i];
        cumulative_counts[i + 1] = acc;
      }
    }
    const scaled_total_counts = cumulative_counts[RED_BLOCK_STATISTICS_BYTES];
    if (scaled_total_counts === 0) {
      // All-zero statistics with N samples > 0 means the block is malformed
      // (no symbol could ever be decoded). meflib would loop forever; we
      // throw a clean error.
      throw new Error(
        `mef-red: block at offset ${blockOffset} has all-zero statistics ` +
        `but number_of_samples=${header.number_of_samples}`,
      );
    }

    // ---- Range decode payload → difference buffer ------------------
    // diff_buffer is sized worst-case: 1 (synthetic -128) + difference_bytes
    // symbols emitted. Encoder bumps difference_bytes by 1 so this is
    // 1 + (real_payload_bytes + 1) = real_payload_bytes + 2 — those
    // 2 trailing bytes are range-coder flush state, harmless if we
    // stop output at number_of_samples.
    const diff_buffer = new Int8Array(1 + header.difference_bytes);
    diff_buffer[0] = -128;   // synthetic keysample flag (meflib.c L6711)
    let diff_p = 1;

    // Position into the raw block payload. ib_p = pointer into `bytes`.
    const payload_start = blockOffset + RED_BLOCK_HEADER_BYTES;
    const payload_end   = blockOffset + header.block_bytes;
    let ib_p = payload_start;
    if (ib_p >= payload_end) {
      throw new Error(
        `mef-red: empty payload but number_of_samples=${header.number_of_samples}`,
      );
    }
    let in_byte = bytes[ib_p++];
    // Use unsigned semantics throughout. (low_bound >>> 0) coerces JS
    // bitwise results back to ui4 — without this they become si4 and
    // arithmetic flips sign for the top half of the range.
    let low_bound = (in_byte >>> (8 - EXTRA_BITS)) >>> 0;
    let range = (1 << EXTRA_BITS) >>> 0;

    for (let i = 0; i < header.difference_bytes; i++) {
      // ---- Renormalise (meflib.c L6719-6730) -----------------------
      while (range <= BOTTOM_VALUE) {
        low_bound =
          (((low_bound << 8) >>> 0) | ((in_byte << EXTRA_BITS) & 0xff)) >>> 0;
        if (ib_p <= payload_end - 1) {
          in_byte = bytes[ib_p++];
        } else {
          in_byte = 0;  // bounds-safe zero pad (L6726)
        }
        low_bound = (low_bound | (in_byte >>> (8 - EXTRA_BITS))) >>> 0;
        range = (range * 256) >>> 0;   // range <<= 8 (ui4)
      }

      // ---- Symbol lookup (meflib.c L6731-6739) ---------------------
      // Use Math.floor for ui4 division — the operands can exceed 2^31
      // but the quotient fits comfortably. (range <= 2^31, total <= 65280,
      // so range_per_count fits in ui4.)
      const range_per_count = Math.floor(range / scaled_total_counts);
      const temp = Math.floor(low_bound / range_per_count);
      const cc = (temp >= scaled_total_counts ? scaled_total_counts - 1 : temp);

      let symbol;
      if (cc > cumulative_counts[128]) {
        // High branch: scan down from 256 — find largest k with
        // cumulative_counts[k] <= cc. C: `for (p--p; *p > cc;);
        // symbol = p - base`.
        let k = 256;
        while (cumulative_counts[k - 1] > cc) k--;
        symbol = k - 1;
      } else {
        // Low branch: scan up from 0 — find largest k with
        // cumulative_counts[k] <= cc.
        let k = 0;
        while (cumulative_counts[k + 1] <= cc) k++;
        symbol = k;
      }

      // ---- Narrow interval to selected symbol (meflib.c L6740-6745) -
      low_bound = (low_bound - range_per_count * cumulative_counts[symbol]) >>> 0;
      if (symbol < 255) {
        range = (range_per_count * statistics[symbol]) >>> 0;
      } else {
        range = (range - range_per_count * cumulative_counts[symbol]) >>> 0;
      }

      diff_buffer[diff_p++] = (symbol > 127 ? symbol - 256 : symbol);
    }

    // ---- Difference buffer → samples (meflib.c L6748-6759) ---------
    const out = new Int32Array(header.number_of_samples);
    let p = 0;
    let current_val = 0;
    // Re-view diff_buffer as bytes for keysample reconstruction
    const diff_u8 = new Uint8Array(diff_buffer.buffer, diff_buffer.byteOffset, diff_buffer.length);
    for (let i = 0; i < header.number_of_samples; i++) {
      if (diff_buffer[p] === -128) {
        // Keysample: next 4 bytes are an int32 LE.
        p++;
        if (p + 4 > diff_buffer.length) {
          throw new Error(
            `mef-red: ran out of difference bytes while reading keysample ` +
            `(sample ${i}/${header.number_of_samples})`,
          );
        }
        // Read 4 LE bytes as si4. Avoid DataView so we keep the
        // existing Uint8Array view we already built.
        current_val =
          (diff_u8[p] |
           (diff_u8[p + 1] << 8) |
           (diff_u8[p + 2] << 16) |
           (diff_u8[p + 3] << 24)) | 0;
        p += 4;
      } else {
        if (p >= diff_buffer.length) {
          throw new Error(
            `mef-red: ran out of difference bytes (sample ${i}/${header.number_of_samples})`,
          );
        }
        current_val = (current_val + diff_buffer[p]) | 0;
        p++;
      }
      out[i] = current_val;
    }

    // ---- Unscale (meflib.c L6762-6763 → L7464-7480) ----------------
    if (header.scale_factor > 1.0) {
      const sf = header.scale_factor;
      for (let i = 0; i < out.length; i++) {
        out[i] = redRound(out[i] * sf);
      }
    }

    // ---- Retrend (meflib.c L6766-6767 → L7294-7319) ---------------
    if (header.detrend_slope !== 0.0 || header.detrend_intercept !== 0.0) {
      const m = header.detrend_slope;
      const b = header.detrend_intercept;
      for (let i = 0; i < out.length; i++) {
        // c = i+1 in the C loop (it preincrements c=0 at the top)
        out[i] = redRound(out[i] + m * (i + 1) + b);
      }
    }

    return { samples: out, header };
  }

  // ==================================================================
  // ENCODE — needed for test fixtures. Implements RED_encode_exec
  // (meflib.c L6848-7049) for the unencrypted, lossless / fixed-scale-
  // factor path. We do NOT implement RED_encode_lossy, RED_detrend,
  // or the normality dispatch — fixtures pass raw samples through
  // unmodified.
  // ==================================================================

  /**
   * Encode a block of int32 samples into a single RED block byte
   * array, including header + payload + 8-byte alignment padding.
   *
   * @param {Int32Array | number[]} samples
   * @param {object} [opts]
   * @param {number} [opts.scaleFactor=1.0] - lossless = 1.0
   * @param {boolean} [opts.discontinuity=false] - sets RED_DISCONTINUITY_MASK
   * @param {number} [opts.startTimeLow=0] - low 32 bits of start_time μUTC
   * @param {number} [opts.startTimeHigh=0] - high 32 bits (signed)
   * @returns {Uint8Array} a single complete RED block
   */
  function encodeBlock(samples, opts) {
    opts = opts || {};
    const scaleFactor   = opts.scaleFactor != null ? opts.scaleFactor : 1.0;
    const discontinuity = !!opts.discontinuity;
    const startTimeLow  = (opts.startTimeLow  || 0) >>> 0;
    const startTimeHigh = (opts.startTimeHigh || 0) | 0;
    const N = samples.length;

    if (N === 0) {
      // Empty block: header only. Match encoder L6871-6880.
      const buf = new Uint8Array(RED_BLOCK_HEADER_BYTES);
      const dv = new DataView(buf.buffer);
      dv.setUint32(OFF_BLOCK_BYTES, RED_BLOCK_HEADER_BYTES, true);
      dv.setUint8(OFF_FLAGS, discontinuity ? RED_DISCONTINUITY_MASK : 0);
      dv.setUint32(OFF_START_TIME, startTimeLow, true);
      dv.setInt32(OFF_START_TIME + 4, startTimeHigh, true);
      const crc = crcCalculate(buf, 4, RED_BLOCK_HEADER_BYTES);
      dv.setUint32(OFF_BLOCK_CRC, crc, true);
      return buf;
    }

    // ---- Apply scale (lossy: input //= scaleFactor) ----------------
    const src = new Int32Array(N);
    if (scaleFactor > 1.0) {
      for (let i = 0; i < N; i++) {
        src[i] = redRound(samples[i] / scaleFactor);
      }
    } else {
      for (let i = 0; i < N; i++) src[i] = samples[i] | 0;
    }

    // ---- Build difference buffer (meflib.c L6890-6905) --------------
    // First 4 bytes: keysample (LE int32). Subsequent: signed-byte
    // diffs, with -128 + 4-byte keysample restart whenever |diff| > 127.
    // Worst case: 4 + 5*(N-1) bytes.
    const diff_buffer = new Int8Array(4 + 5 * Math.max(N - 1, 0));
    const diff_u8 = new Uint8Array(diff_buffer.buffer, diff_buffer.byteOffset, diff_buffer.length);
    let dp = 0;
    // Encode keysample 0
    const v0 = src[0] | 0;
    diff_u8[dp++] = v0       & 0xff;
    diff_u8[dp++] = (v0 >> 8) & 0xff;
    diff_u8[dp++] = (v0 >> 16) & 0xff;
    diff_u8[dp++] = (v0 >> 24) & 0xff;
    let prev = v0;
    for (let i = 1; i < N; i++) {
      const cur = src[i] | 0;
      const diff = cur - prev;
      if (diff > 127 || diff < -127) {
        diff_buffer[dp++] = -128;
        diff_u8[dp++] = cur       & 0xff;
        diff_u8[dp++] = (cur >> 8) & 0xff;
        diff_u8[dp++] = (cur >> 16) & 0xff;
        diff_u8[dp++] = (cur >> 24) & 0xff;
      } else {
        diff_buffer[dp++] = diff;
      }
      prev = cur;
    }
    const real_diff_bytes = dp;

    // ---- Build statistics histogram (meflib.c L6908-6929) ----------
    const counts = new Uint32Array(RED_BLOCK_STATISTICS_BYTES);
    for (let i = 0; i < real_diff_bytes; i++) counts[diff_u8[i]]++;
    let maxCount = counts[0];
    for (let i = 1; i < RED_BLOCK_STATISTICS_BYTES; i++) {
      if (counts[i] > maxCount) maxCount = counts[i];
    }
    const statistics = new Uint8Array(RED_BLOCK_STATISTICS_BYTES);
    if (maxCount > 255) {
      const stats_scale = 254.999999999 / maxCount;
      for (let i = 0; i < RED_BLOCK_STATISTICS_BYTES; i++) {
        statistics[i] = counts[i] ? Math.ceil(counts[i] * stats_scale) : 0;
      }
    } else {
      for (let i = 0; i < RED_BLOCK_STATISTICS_BYTES; i++) {
        statistics[i] = counts[i];
      }
    }

    // Cumulative counts (same as decoder)
    const cumulative_counts = new Uint32Array(RED_BLOCK_STATISTICS_BYTES + 1);
    {
      let acc = 0;
      cumulative_counts[0] = 0;
      for (let i = 0; i < RED_BLOCK_STATISTICS_BYTES; i++) {
        acc += statistics[i];
        cumulative_counts[i + 1] = acc;
      }
    }
    const scaled_total_counts = cumulative_counts[RED_BLOCK_STATISTICS_BYTES];

    // ---- Range encode (meflib.c L6939-6995) ------------------------
    // Output buffer: worst case the encoded payload is roughly
    // difference_bytes + a few flush bytes. Allocate generously:
    // RED_MAX_DIFFERENCE_BYTES + RED_BLOCK_HEADER_BYTES (meflib.h L1101-1102).
    const max_payload = real_diff_bytes + 8;
    const out = new Uint8Array(RED_BLOCK_HEADER_BYTES + max_payload + 16);
    // The encoder's "compressed_buffer_p" starts at HEADER_BYTES - 1
    // (saving last statistics byte across the write); we replicate
    // that exactly to match upstream byte-for-byte.
    let cp = RED_BLOCK_HEADER_BYTES - 1;
    // We'll write the statistics block (incl. last byte) into the header
    // region after encoding completes.
    let low_bound = 0 >>> 0;
    let out_byte = 0;
    let underflow_bytes = 0;
    let range = TOP_VALUE;

    for (let i = 0; i < real_diff_bytes; i++) {
      // Renormalise (encoder version, meflib.c L6945-6960)
      while (range <= BOTTOM_VALUE) {
        if (low_bound < CARRY_CHECK) {
          out[cp++] = out_byte & 0xff;
          for (; underflow_bytes; underflow_bytes--) out[cp++] = 0xff;
          out_byte = (low_bound >>> SHIFT_BITS) & 0xff;
        } else if (low_bound & TOP_VALUE) {
          out[cp++] = (out_byte + 1) & 0xff;
          for (; underflow_bytes; underflow_bytes--) out[cp++] = 0x00;
          out_byte = (low_bound >>> SHIFT_BITS) & 0xff;
        } else {
          underflow_bytes++;
        }
        range = (range * 256) >>> 0;
        low_bound = ((low_bound << 8) >>> 0) & TOP_VALUE_MINUS_1;
      }
      const r = Math.floor(range / scaled_total_counts);
      const sym = diff_u8[i];
      const add = (r * cumulative_counts[sym]) >>> 0;
      low_bound = (low_bound + add) >>> 0;
      if (sym < 255) {
        range = (r * statistics[sym]) >>> 0;
      } else {
        range = (range - add) >>> 0;
      }
    }

    // Trailing flush — see L6967-6982 then L6983-6994
    while (range <= BOTTOM_VALUE) {
      if (low_bound < CARRY_CHECK) {
        out[cp++] = out_byte & 0xff;
        for (; underflow_bytes; underflow_bytes--) out[cp++] = 0xff;
        out_byte = (low_bound >>> SHIFT_BITS) & 0xff;
      } else if (low_bound & TOP_VALUE) {
        out[cp++] = (out_byte + 1) & 0xff;
        for (; underflow_bytes; underflow_bytes--) out[cp++] = 0x00;
        out_byte = (low_bound >>> SHIFT_BITS) & 0xff;
      } else {
        underflow_bytes++;
      }
      range = (range * 256) >>> 0;
      low_bound = ((low_bound << 8) >>> 0) & TOP_VALUE_MINUS_1;
    }
    const temp = ((low_bound >>> SHIFT_BITS) + 1) >>> 0;
    if (temp > 0xff) {
      out[cp++] = (out_byte + 1) & 0xff;
      for (; underflow_bytes; underflow_bytes--) out[cp++] = 0x00;
    } else {
      out[cp++] = out_byte & 0xff;
      for (; underflow_bytes; underflow_bytes--) out[cp++] = 0xff;
    }
    out[cp++] = temp & 0xff;
    out[cp++] = 0;   // L6994 — terminator zero byte

    // L7000: difference_bytes carries the +1 for the synthetic
    // keysample flag the decoder writes.
    const difference_bytes_field = (real_diff_bytes + 1) >>> 0;

    // L7003-7010 — pad block to 8-byte alignment
    let block_bytes = cp;
    let extra = block_bytes % 8;
    if (extra) {
      extra = 8 - extra;
      for (let i = 0; i < extra; i++) out[cp++] = PAD_BYTE_VALUE;
      block_bytes += extra;
    }

    // ---- Write header (with statistics, then CRC) ------------------
    // Copy statistics into the header region BEFORE the encoder's
    // "last statistics byte" overwrite quirk is reproduced. Actually
    // we kept it simpler: the encoder smashed out_byte over the last
    // stats byte during encoding, so we write statistics LAST, knowing
    // statistics[255] may or may not have been "saved+restored" — but
    // the encoder explicitly preserves it (L6997: `*last_byte_ptr = last_byte_val`).
    // We never touched bytes < HEADER_BYTES-1 during encoding (we
    // started cp at HEADER_BYTES-1 but only WROTE on the increments,
    // and the first increment is preceded by either a write or a check
    // that doesn't actually persist a garbage byte until renormalisation
    // triggers — at which point HEADER_BYTES-1 gets overwritten with a
    // real codec byte). Restoring it now puts the original statistics
    // byte back exactly as the C encoder does.
    out.set(statistics, RED_BLOCK_STATISTICS_OFFSET);

    // Header scalar fields
    const dv = new DataView(out.buffer, 0, RED_BLOCK_HEADER_BYTES);
    dv.setUint8(OFF_FLAGS, discontinuity ? RED_DISCONTINUITY_MASK : 0);
    // detrend_slope, detrend_intercept default 0.0 (zeroed buffer).
    dv.setFloat32(OFF_SCALE_FACTOR, scaleFactor, true);
    dv.setUint32(OFF_DIFFERENCE_BYTES, difference_bytes_field, true);
    dv.setUint32(OFF_NUMBER_OF_SAMPLES, N, true);
    dv.setUint32(OFF_BLOCK_BYTES, block_bytes, true);
    dv.setUint32(OFF_START_TIME, startTimeLow, true);
    dv.setInt32(OFF_START_TIME + 4, startTimeHigh, true);

    // Trim to exact block_bytes length and compute CRC over [4..end).
    const block = out.subarray(0, block_bytes);
    const crc = crcCalculate(block, 4, block_bytes);
    dv.setUint32(OFF_BLOCK_CRC, crc, true);

    return block;
  }

  const api = {
    RED_BLOCK_HEADER_BYTES,
    RED_BLOCK_STATISTICS_BYTES,
    RED_DISCONTINUITY_MASK,
    RED_LEVEL_1_ENCRYPTION_MASK,
    RED_LEVEL_2_ENCRYPTION_MASK,
    RED_NAN,
    RED_NEGATIVE_INFINITY,
    RED_POSITIVE_INFINITY,
    PAD_BYTE_VALUE,
    crcCalculate,
    redRound,
    parseBlockHeader,
    decodeBlock,
    encodeBlock,
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.MefRed = api;
})();
