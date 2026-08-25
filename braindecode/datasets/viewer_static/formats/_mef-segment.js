/* ============================================================
   formats/_mef-segment.js — MEF3 (Multiscale Electrophysiology
   Format v3) universal-header + metadata parsing helpers.

   Spec source: msel-source/meflib (Apache 2.0).
     https://github.com/msel-source/meflib (master/meflib/meflib.h)
   All byte offsets are documented inline against the constant
   names from meflib.h so a maintainer can cross-check against
   upstream.

   What this module provides:
     - File-type magic codes (`tmet`, `tdat`, `tidx`) for sniffing.
     - Universal header parser (1024 bytes, LE).
     - Time-series Section 2 parser (sampling_frequency, n_samples).
     - Time-series index entry parser (56 bytes per entry).
     - Koopman32 CRC table + CRC_calculate(buf) compatible with
       meflib's CRC_calculate (init=0xFFFFFFFF, no final XOR).

   What this module does NOT do:
     - Decryption (the viewer only supports unencrypted MEF3).
     - RED block decoding (Tier 3 of the spec — out of scope).
     - Section 1 / Section 3 parsing (recording start time and
       discretionary metadata aren't needed for the viewer's
       read window contract).

   IMPORTANT: All multi-byte values in MEF3 files are
   LITTLE-ENDIAN (UNIVERSAL_HEADER_BYTE_ORDER_CODE = 1 in every
   in-the-wild file — meflib.h L141 #define MEF_LITTLE_ENDIAN 1).
   ============================================================ */
(function () {
  'use strict';

  // ---- File-type magic codes (offset 8 of every MEF3 file) ----------
  // From meflib.h:
  //   TIME_SERIES_METADATA_FILE_TYPE_STRING   "tmet"  (LE code 0x74656d74)
  //   TIME_SERIES_DATA_FILE_TYPE_STRING       "tdat"  (LE code 0x74616474)
  //   TIME_SERIES_INDICES_FILE_TYPE_STRING    "tidx"  (LE code 0x78646974)
  //   TIME_SERIES_CHANNEL_DIRECTORY_TYPE_STRING "timd"
  //   SESSION_DIRECTORY_TYPE_STRING           "mefd"
  // Magic is stored as 4 ASCII bytes followed by a null terminator;
  // we match on the 4-char prefix.
  const MAGIC = {
    TMET: 'tmet',
    TDAT: 'tdat',
    TIDX: 'tidx',
  };

  // ---- Universal Header offsets (meflib.h L309-347) -----------------
  // The universal header sits at offset 0 of every .tmet / .tdat /
  // .tidx file. It is exactly 1024 bytes. We only read the fields the
  // viewer needs; the rest are skipped.
  const UH_BYTES                       = 1024;
  const UH_HEADER_CRC_OFFSET           = 0;     // ui4 — CRC of bytes [4..1024)
  const UH_BODY_CRC_OFFSET             = 4;     // ui4 — CRC of bytes [1024..EOF)
  const UH_FILE_TYPE_OFFSET            = 8;     // ascii[4] (no null required)
  const UH_MEF_VERSION_MAJOR_OFFSET    = 13;    // ui1
  const UH_MEF_VERSION_MINOR_OFFSET    = 14;    // ui1
  const UH_BYTE_ORDER_CODE_OFFSET      = 15;    // ui1 (1 = LE)
  const UH_START_TIME_OFFSET           = 16;    // si8 — μUTC
  const UH_END_TIME_OFFSET             = 24;    // si8 — μUTC
  const UH_NUMBER_OF_ENTRIES_OFFSET    = 32;    // si8
  const UH_MAXIMUM_ENTRY_SIZE_OFFSET   = 40;    // si8
  const UH_SEGMENT_NUMBER_OFFSET       = 48;    // si4
  const UH_CHANNEL_NAME_OFFSET         = 52;    // utf8 up to 256 bytes (null-terminated)
  const UH_CHANNEL_NAME_BYTES          = 256;
  const UH_SESSION_NAME_OFFSET         = 308;
  const UH_LEVEL_1_PASSWORD_VALIDATION_OFFSET = 868;  // 16 bytes — zero = no password set
  const UH_LEVEL_2_PASSWORD_VALIDATION_OFFSET = 884;  // 16 bytes — zero = no password set

  // ---- Metadata section offsets (meflib.h L353-422) -----------------
  // The metadata file (.tmet) layout is:
  //   [0, 1024)            Universal Header
  //   [1024, 2560)         METADATA_SECTION_1 (1536 bytes)
  //   [2560, 13312)        METADATA_SECTION_2 (10752 bytes — for time series
  //                         channels; video channels reuse the same region
  //                         with a different layout)
  //   [13312, 16384)       METADATA_SECTION_3 (3072 bytes)
  // METADATA_SECTION_1 stores the encryption levels for sections 2+3.
  const METADATA_SECTION_1_OFFSET                  = UH_BYTES;       // = 1024
  const METADATA_SECTION_2_ENCRYPTION_OFFSET       = 1024;           // within sec1
  const METADATA_SECTION_3_ENCRYPTION_OFFSET       = 1025;           // within sec1
  const METADATA_SECTION_2_OFFSET                  = 2560;

  // Time series Section 2 fields we read (meflib.h L378-419). Offsets
  // are RELATIVE to the start of section 2 (i.e. file offset 2560).
  const TSM2_SAMPLING_FREQUENCY_OFFSET    = 8720 - METADATA_SECTION_2_OFFSET;  // sf8 → 6160
  const TSM2_NUMBER_OF_SAMPLES_OFFSET     = 8920 - METADATA_SECTION_2_OFFSET;  // si8 → 6360
  const TSM2_NUMBER_OF_BLOCKS_OFFSET      = 8928 - METADATA_SECTION_2_OFFSET;  // si8 → 6368
  const TSM2_MAXIMUM_BLOCK_BYTES_OFFSET   = 8936 - METADATA_SECTION_2_OFFSET;  // si8 → 6376
  const TSM2_MAXIMUM_BLOCK_SAMPLES_OFFSET = 8944 - METADATA_SECTION_2_OFFSET;  // ui4 → 6384

  // ---- Time series index entry (meflib.h L499-521) ------------------
  // The .tidx file contains, after its 1024-byte universal header, a
  // densely-packed array of TIME_SERIES_INDEX entries (56 bytes each).
  // Each entry points at one RED block in the .tdat file.
  const TSI_ENTRY_BYTES                = 56;
  const TSI_FILE_OFFSET_OFFSET         = 0;     // si8 — byte offset within .tdat
  const TSI_START_TIME_OFFSET          = 8;     // si8 — μUTC
  const TSI_START_SAMPLE_OFFSET        = 16;    // si8
  const TSI_NUMBER_OF_SAMPLES_OFFSET   = 24;    // ui4
  const TSI_BLOCK_BYTES_OFFSET         = 28;    // ui4
  const TSI_RED_BLOCK_FLAGS_OFFSET     = 44;    // ui1

  // ---- Encryption levels (meflib.h L181-194) ------------------------
  // We only support recordings where Section 2 + Section 3 are
  // unencrypted (level 0). Anything else gets rejected up-front so the
  // viewer surfaces a clean error instead of decoding garbage.
  const NO_ENCRYPTION                  = 0;

  // ---- Koopman32 CRC table (meflib.h L1166-1224) --------------------
  // Verbatim from CRC_KOOPMAN32_KEY. meflib's CRC_calculate is:
  //   crc = 0xFFFFFFFF
  //   for each byte b: crc = (crc >>> 8) ^ table[(crc ^ b) & 0xFF]
  // No final XOR. Returned as a ui4 (32-bit unsigned).
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

  /**
   * Compute the Koopman32 CRC over a byte range. Direct port of
   * meflib's CRC_calculate (meflib.c L1618). Returns an unsigned 32-bit
   * integer (uses >>> 0 to coerce off the 53-bit JS number representation
   * back to ui4 semantics).
   *
   * @param {Uint8Array} bytes
   * @param {number} [start=0]
   * @param {number} [end=bytes.length]
   * @returns {number} CRC as ui4 (0..0xFFFFFFFF)
   */
  function crcCalculate(bytes, start, end) {
    if (start == null) start = 0;
    if (end == null) end = bytes.length;
    let crc = 0xFFFFFFFF;
    for (let i = start; i < end; i++) {
      crc = ((crc >>> 8) ^ CRC_TABLE[(crc ^ bytes[i]) & 0xFF]) >>> 0;
    }
    return crc >>> 0;
  }

  /**
   * Read a 64-bit signed integer from a DataView. JavaScript numbers
   * lose precision above 2^53; for MEF fields that fit comfortably in
   * 53 bits (n_samples, n_blocks, segment start_sample), this is safe.
   * For μUTC timestamps we'd technically want BigInt — but the viewer's
   * sample-index math never exceeds 2^53, so this approximation is
   * sufficient for Tier 1 metadata parsing.
   *
   * @param {DataView} dv
   * @param {number} offset
   * @returns {number}
   */
  function readSi8(dv, offset) {
    // BigInt-aware read — Node and modern browsers expose getBigInt64.
    // Convert to a Number for downstream arithmetic; n_samples values
    // we care about are well under 2^53.
    const big = dv.getBigInt64(offset, /*littleEndian=*/ true);
    // Number(BigInt) silently truncates to Number — this is the
    // documented JS conversion, and is the right move at our scale.
    return Number(big);
  }

  /**
   * Read a UTF-8 string from a fixed-size byte region. Stops at the
   * first null byte (matching the C `strlen` semantics meflib uses
   * when serialising channel_name / session_name).
   *
   * @param {Uint8Array} bytes
   * @param {number} offset
   * @param {number} maxLen
   * @returns {string}
   */
  function readUtf8Cstring(bytes, offset, maxLen) {
    let end = offset;
    const stop = Math.min(offset + maxLen, bytes.length);
    while (end < stop && bytes[end] !== 0) end++;
    // TextDecoder is universally available in Node 12+ and modern browsers.
    const slice = bytes.subarray(offset, end);
    return new TextDecoder('utf-8', { fatal: false }).decode(slice);
  }

  /**
   * Parse a MEF3 universal header (1024 bytes at offset 0 of every
   * .tmet/.tdat/.tidx).
   *
   * @param {ArrayBuffer | Uint8Array} buf
   * @returns {{
   *   header_crc: number,
   *   body_crc: number,
   *   file_type: string,
   *   mef_version_major: number,
   *   mef_version_minor: number,
   *   byte_order_code: number,
   *   start_time: number,
   *   end_time: number,
   *   number_of_entries: number,
   *   maximum_entry_size: number,
   *   segment_number: number,
   *   channel_name: string,
   *   session_name: string,
   *   level_1_encrypted: boolean,
   *   level_2_encrypted: boolean,
   * }}
   * @throws {Error} on truncated buffer or unknown byte-order code.
   */
  function parseUniversalHeader(buf) {
    const bytes = buf instanceof Uint8Array
      ? buf
      : new Uint8Array(buf, 0, Math.min(buf.byteLength, UH_BYTES));
    if (bytes.length < UH_BYTES) {
      throw new Error(
        `mef: universal header needs ${UH_BYTES} bytes, got ${bytes.length}`,
      );
    }
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

    const file_type = readUtf8Cstring(bytes, UH_FILE_TYPE_OFFSET, 4);
    const byte_order_code = bytes[UH_BYTE_ORDER_CODE_OFFSET];
    // We only support little-endian MEF3 — every file in the wild is LE
    // (MEF_LITTLE_ENDIAN=1). A big-endian file would require a different
    // DataView call sites; reject up-front rather than misdecode.
    if (byte_order_code !== 1) {
      throw new Error(
        `mef: byte_order_code=${byte_order_code} — only little-endian (1) supported`,
      );
    }

    // Encryption is signalled by a non-zero "password validation field"
    // at offsets 868/884 (16 bytes each). If those regions are all zero,
    // no password has been set and Section 2 + Section 3 will be in the
    // clear. We check the first byte as a fast path; a fully-zero region
    // means no encryption.
    let level_1_encrypted = false;
    let level_2_encrypted = false;
    for (let i = 0; i < 16; i++) {
      if (bytes[UH_LEVEL_1_PASSWORD_VALIDATION_OFFSET + i] !== 0) {
        level_1_encrypted = true;
        break;
      }
    }
    for (let i = 0; i < 16; i++) {
      if (bytes[UH_LEVEL_2_PASSWORD_VALIDATION_OFFSET + i] !== 0) {
        level_2_encrypted = true;
        break;
      }
    }

    return {
      header_crc:         dv.getUint32(UH_HEADER_CRC_OFFSET, true),
      body_crc:           dv.getUint32(UH_BODY_CRC_OFFSET,   true),
      file_type,
      mef_version_major:  bytes[UH_MEF_VERSION_MAJOR_OFFSET],
      mef_version_minor:  bytes[UH_MEF_VERSION_MINOR_OFFSET],
      byte_order_code,
      start_time:         readSi8(dv, UH_START_TIME_OFFSET),
      end_time:           readSi8(dv, UH_END_TIME_OFFSET),
      number_of_entries:  readSi8(dv, UH_NUMBER_OF_ENTRIES_OFFSET),
      maximum_entry_size: readSi8(dv, UH_MAXIMUM_ENTRY_SIZE_OFFSET),
      segment_number:     dv.getInt32(UH_SEGMENT_NUMBER_OFFSET, true),
      channel_name:       readUtf8Cstring(bytes, UH_CHANNEL_NAME_OFFSET, UH_CHANNEL_NAME_BYTES),
      session_name:       readUtf8Cstring(bytes, UH_SESSION_NAME_OFFSET, UH_CHANNEL_NAME_BYTES),
      level_1_encrypted,
      level_2_encrypted,
    };
  }

  /**
   * Parse the time-series metadata file (.tmet). The file consists of
   * a universal header followed by Section 1 (encryption levels for
   * sec2+sec3) and Section 2 (the time-series fields the viewer needs).
   * Section 3 is ignored.
   *
   * Requires the FULL .tmet file in `buf` — it's only 16384 bytes
   * (UH 1024 + sec1 1536 + sec2 10752 + sec3 3072), so reading the
   * whole thing in one HTTP Range is cheap.
   *
   * @param {ArrayBuffer | Uint8Array} buf
   * @returns {{
   *   universal_header: object,
   *   sampling_frequency: number,
   *   n_samples: number,
   *   n_blocks: number,
   *   maximum_block_bytes: number,
   *   maximum_block_samples: number,
   *   section_2_encrypted: boolean,
   *   section_3_encrypted: boolean,
   * }}
   * @throws {Error} if buffer is too small, magic mismatches, or
   *   section 2 is encrypted (we don't decrypt).
   */
  function parseTmet(buf) {
    const bytes = buf instanceof Uint8Array
      ? buf
      : new Uint8Array(buf);
    // Section 2 ends at offset 2560 + 10752 = 13312 — that's the
    // minimum span we need to parse. We don't read sec3, so a file
    // truncated at 13312 still parses fine.
    const NEED = METADATA_SECTION_2_OFFSET + 10752;
    if (bytes.length < NEED) {
      throw new Error(
        `mef: .tmet needs at least ${NEED} bytes (UH + sec1 + sec2), ` +
        `got ${bytes.length}`,
      );
    }

    const uh = parseUniversalHeader(bytes);
    if (uh.file_type !== MAGIC.TMET) {
      throw new Error(
        `mef: .tmet magic mismatch — expected ${JSON.stringify(MAGIC.TMET)}, ` +
        `got ${JSON.stringify(uh.file_type)}`,
      );
    }

    // Section 1 encryption flags. meflib writes signed bytes here; the
    // convention is: positive value = encrypted at that level, 0 = no
    // encryption, negative = decrypted (only seen in-memory after
    // successful decryption).
    const sec2EncByte = new Int8Array(bytes.buffer, bytes.byteOffset + METADATA_SECTION_1_OFFSET + METADATA_SECTION_2_ENCRYPTION_OFFSET, 1)[0];
    const sec3EncByte = new Int8Array(bytes.buffer, bytes.byteOffset + METADATA_SECTION_1_OFFSET + METADATA_SECTION_3_ENCRYPTION_OFFSET, 1)[0];
    const section_2_encrypted = sec2EncByte > NO_ENCRYPTION;
    const section_3_encrypted = sec3EncByte > NO_ENCRYPTION;
    if (section_2_encrypted) {
      throw new Error(
        `mef: section 2 is encrypted (level ${sec2EncByte}) — encrypted MEF3 ` +
        `recordings are not supported by this reader`,
      );
    }

    const sec2 = METADATA_SECTION_2_OFFSET;
    const dv = new DataView(bytes.buffer, bytes.byteOffset + sec2, 10752);
    const sampling_frequency    = dv.getFloat64(TSM2_SAMPLING_FREQUENCY_OFFSET, true);
    const n_samples             = readSi8(dv, TSM2_NUMBER_OF_SAMPLES_OFFSET);
    const n_blocks              = readSi8(dv, TSM2_NUMBER_OF_BLOCKS_OFFSET);
    const maximum_block_bytes   = readSi8(dv, TSM2_MAXIMUM_BLOCK_BYTES_OFFSET);
    const maximum_block_samples = dv.getUint32(TSM2_MAXIMUM_BLOCK_SAMPLES_OFFSET, true);

    if (!Number.isFinite(sampling_frequency) || sampling_frequency <= 0) {
      throw new Error(`mef: invalid sampling_frequency ${sampling_frequency}`);
    }
    if (!Number.isInteger(n_samples) || n_samples < 0) {
      throw new Error(`mef: invalid n_samples ${n_samples}`);
    }

    return {
      universal_header: uh,
      sampling_frequency,
      n_samples,
      n_blocks,
      maximum_block_bytes,
      maximum_block_samples,
      section_2_encrypted,
      section_3_encrypted,
    };
  }

  /**
   * Parse one entry from a .tidx file (the per-block index table).
   * Used for navigation/seeking; not currently consumed by readWindow
   * because we don't decode RED blocks. Exposed for tests + future use.
   *
   * @param {DataView} dv
   * @param {number} base - byte offset of the entry within `dv`
   * @returns {{
   *   file_offset: number, start_time: number, start_sample: number,
   *   number_of_samples: number, block_bytes: number, red_block_flags: number,
   * }}
   */
  function parseTidxEntry(dv, base) {
    return {
      file_offset:       readSi8(dv, base + TSI_FILE_OFFSET_OFFSET),
      start_time:        readSi8(dv, base + TSI_START_TIME_OFFSET),
      start_sample:      readSi8(dv, base + TSI_START_SAMPLE_OFFSET),
      number_of_samples: dv.getUint32(base + TSI_NUMBER_OF_SAMPLES_OFFSET, true),
      block_bytes:       dv.getUint32(base + TSI_BLOCK_BYTES_OFFSET,       true),
      red_block_flags:   dv.getUint8(base + TSI_RED_BLOCK_FLAGS_OFFSET),
    };
  }

  /**
   * Validate the universal header CRC. meflib computes:
   *   header_CRC = CRC_calculate(buf + CRC_BYTES, UH_BYTES - CRC_BYTES)
   * i.e. over bytes [4..1024). The header's own CRC field (bytes [0..4))
   * is excluded.
   *
   * @param {Uint8Array} bytes - full 1024-byte universal header
   * @returns {{ stored: number, computed: number, valid: boolean }}
   */
  function validateUniversalHeaderCrc(bytes) {
    if (bytes.length < UH_BYTES) {
      throw new Error(`mef: validateUniversalHeaderCrc needs ${UH_BYTES} bytes`);
    }
    const dv = new DataView(bytes.buffer, bytes.byteOffset, UH_BYTES);
    const stored = dv.getUint32(UH_HEADER_CRC_OFFSET, true);
    const computed = crcCalculate(bytes, 4, UH_BYTES);
    return { stored, computed, valid: stored === computed };
  }

  const api = {
    MAGIC,
    UH_BYTES,
    METADATA_SECTION_2_OFFSET,
    TSI_ENTRY_BYTES,
    crcCalculate,
    parseUniversalHeader,
    parseTmet,
    parseTidxEntry,
    validateUniversalHeaderCrc,
    readUtf8Cstring,
    readSi8,
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.MefSegment = api;
})();
