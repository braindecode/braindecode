# MEF3 RED (Range Encoded Differential) — Bit-Level Spec

Authoritative source: msel-source/meflib (Apache 2.0).
Every constant + every byte/bit-level decision below cross-references
the file/line in upstream `meflib/meflib.c` and `meflib/meflib.h` so
the JS port can be audited without re-reading the C.

Local clone used for cross-checks: `/tmp/meflib/meflib/meflib/`.

---

## 1. Per-block on-disk layout

`.tdat` is `[ universal_header (1024 B) | block 0 | block 1 | ... ]`.

Each RED block is `block_bytes` long, 8-byte aligned (pad byte at end if needed —
encoder: meflib.c L7004-7010, pad value = `PAD_BYTE_VALUE` defined in meflib.h).

### Block header — 304 bytes (meflib.h L1132-1145, offsets at L1035-1051)

| Offset | Type | Field                | Notes                                                |
|-------:|------|----------------------|------------------------------------------------------|
| 0      | ui4  | block_CRC            | Koopman32 over bytes [4..block_bytes)                |
| 4      | ui1  | flags                | bit0 = discontinuity, bit1/2 = L1/L2 encryption      |
| 5      | ui1[3] | protected_region   | reserved                                              |
| 8      | ui1[8] | discretionary_region | reserved                                            |
| 16     | sf4  | detrend_slope        | added back during decode (RED_retrend)                |
| 20     | sf4  | detrend_intercept    | added back during decode                              |
| 24     | sf4  | scale_factor         | if > 1.0, output is multiplied (RED_unscale)          |
| 28     | ui4  | difference_bytes     | size of encoded payload (see §3)                      |
| 32     | ui4  | number_of_samples    | samples produced by this block                        |
| 36     | ui4  | block_bytes          | total block size on disk (incl header + padding)      |
| 40     | si8  | start_time           | μUTC                                                  |
| 48     | ui1[256] | statistics       | the 256 scaled symbol counts (CDF table source)      |

Right after the header (offset 304) begins the range-coded payload.

**Compression mode dispatch.** Decode only supports the unencrypted, lossless
or fixed-scale-factor mode produced by RED_encode_exec (meflib.c L6848). Lossy
modes also call RED_encode_exec internally, so the on-disk format is identical;
lossiness is purely the encoder choosing a non-1 scale_factor.

### Encryption (meflib.c L6663-6684)
- flags bit1 set → AES decrypt `statistics` field with L1 key.
- flags bit2 set → AES decrypt with L2 key.
- We do not support encrypted MEF3; throw a clean error if either bit is set.

### Discontinuity (meflib.c L6693-6696)
- flags bit0 set → `directives.discontinuity = true`. Caller-visible signal that
  the differential chain restarted at this block. Within a block the
  differential chain itself uses the keysample marker (§3).

---

## 2. CDF construction (meflib.c L6702-6708)

Given the 256-byte `statistics[]` (scaled symbol counts):

```
cumulative_counts[0]   = 0
cumulative_counts[i+1] = cumulative_counts[i] + statistics[i]   for i in 0..255
scaled_total_counts    = cumulative_counts[256]
```

`cumulative_counts` is the CDF used by the range decoder. All arithmetic is
ui4. `scaled_total_counts` is at most 256 * 255 = 65280, fits in ui4 fine.

---

## 3. Range decode payload → difference buffer (meflib.c L6710-6746)

Range coder constants (meflib.h L1066-1071):
```
TOP_VALUE      = 0x80000000   (unused in decode, used in encode renormalisation)
SHIFT_BITS     = 23
EXTRA_BITS     = 7
BOTTOM_VALUE   = 0x00800000   = 1<<23
```

### Initialisation
```
ib_p       = block + RED_BLOCK_HEADER_BYTES   (start of payload)
in_byte    = *ib_p++                          (first payload byte)
low_bound  = in_byte >> (8 - EXTRA_BITS)      = in_byte >> 1
range      = 1 << EXTRA_BITS                  = 128

diff_buffer[0] = -128            # synthetic keysample-flag (L6711)
diff_buffer_p  = diff_buffer + 1
```

### Per-symbol loop (runs `difference_bytes` times — L6718)

```
# Renormalise: shift in 8 more bits of payload until range > BOTTOM_VALUE
while range <= BOTTOM_VALUE:
    # Top bits of low_bound shift left; refill bottom with leftover 7 bits of
    # the previous in_byte then EXTRA_BITS=7 bits of the new in_byte.
    low_bound = (low_bound << 8) | ((in_byte << EXTRA_BITS) & 0xff)
    if (ib_p - block) <= (block_bytes - 1):
        in_byte = *ib_p++
    else:
        in_byte = 0     # bounds-safe zero pad (L6726)
    low_bound |= in_byte >> (8 - EXTRA_BITS)
    range <<= 8

# Symbol search
range_per_count = range // scaled_total_counts
temp            = low_bound // range_per_count
cc              = min(temp, scaled_total_counts - 1)

# Two-sided linear scan from the midpoint (L6733-6739):
if cc > cumulative_counts[128]:
    # scan downward from index 256 until cumulative_counts[k] <= cc
    k = 256
    while cumulative_counts[k-1] > cc: k -= 1
    symbol = k - 1     # see note below
else:
    # scan upward from index 0 until cumulative_counts[k] > cc
    k = 0
    while cumulative_counts[k+1] <= cc: k += 1
    symbol = k

# Narrow the interval to the symbol
low_bound -= range_per_count * cumulative_counts[symbol]
if symbol < 255:
    range = range_per_count * statistics[symbol]
else:
    range -= range_per_count * cumulative_counts[symbol]

diff_buffer[diff_buffer_p++] = symbol
```

**Symbol resolution detail.** In the C code (L6733-6739):

```c
if (cc > cumulative_counts[128]) {
    for (ui4_p1 = ui4_p2; *--ui4_p1 > cc;);  // ui4_p2 = cumulative_counts + 256
    symbol = ui4_p1 - cumulative_counts;
} else {
    for (ui4_p1 = cumulative_counts; *++ui4_p1 <= cc;);
    symbol = ui4_p1 - cumulative_counts - 1;
}
```

Decoded literally:
- **High branch**: pointer starts at `cumulative_counts + 256` and is *pre*-decremented
  while the dereferenced value is `> cc`. Loop exits with `*ui4_p1 <= cc`.
  `symbol = ui4_p1 - cumulative_counts` (zero-based index of the first cum-count ≤ cc).
- **Low branch**: pointer starts at `cumulative_counts` and is *pre*-incremented
  while the dereferenced value is `<= cc`. Loop exits with `*ui4_p1 > cc`.
  `symbol = ui4_p1 - cumulative_counts - 1` (zero-based index of the last cum-count ≤ cc).

Both branches return the symbol s where `cumulative_counts[s] <= cc < cumulative_counts[s+1]`,
which is the standard arithmetic-coder symbol lookup.

---

## 4. Difference buffer → samples (meflib.c L6748-6759)

The difference buffer holds at least 1 + difference_bytes signed bytes
(first is the synthetic -128 flag, then the encoded payload bytes).

```
current_val = undefined
for sample_idx in 0..number_of_samples-1:
    if diff_buffer[p] == -128:          # keysample marker
        p += 1
        current_val = read_int32_le(diff_buffer + p)
        p += 4
    else:
        current_val += diff_buffer[p]    # signed byte add
        p += 1
    output[sample_idx] = current_val
```

The first iteration always hits the keysample branch because of the synthetic
-128 written at L6711. Subsequent keysample markers (encoder: L6900) only
appear when an adjacent difference exceeded the int8 range [-127, 127].

---

## 5. Post-processing (meflib.c L6761-6767)

```
if block_header.scale_factor > 1.0:        # RED_unscale (L7464)
    for i: output[i] = round(output[i] * scale_factor)

if detrend_slope != 0.0 or detrend_intercept != 0.0:    # RED_retrend (L7294)
    m, b = detrend_slope, detrend_intercept
    for i in 0..N-1:
        output[i] = round(output[i] + m * (i+1) + b)
```

`round` here is `RED_round` (meflib.c L7322): half-away-from-zero rounding,
clamped to `[RED_NEGATIVE_INFINITY, RED_POSITIVE_INFINITY]` =
`[0x80000001, 0x7FFFFFFF]`. The sentinels `RED_NAN` = `0x80000000`,
`RED_NEGATIVE_INFINITY` = `0x80000001`, `RED_POSITIVE_INFINITY` = `0x7FFFFFFF`,
`RED_MAXIMUM_SAMPLE_VALUE` = `0x7FFFFFFE`, `RED_MINIMUM_SAMPLE_VALUE` = `0x80000002`
(meflib.h L1059-1063) carve out the legal sample range.

---

## 6. Block CRC validation (meflib.c L6653-6660)

```
expected = block.block_CRC
computed = Koopman32(block[CRC_BYTES..block_bytes])     # CRC_BYTES = 4
if expected != computed: return without decoding (log error)
```

We replicate this: throw on CRC mismatch by default; allow caller to skip
for debug. CRC table + algorithm: see `formats/_mef-segment.js`.

---

## 7. JS port deviations

1. **No AES.** Encrypted blocks → throw. (Production iEEG datasets we target
   ship unencrypted.)
2. **Lossy modes still decode the same way.** The encoder selects scale_factor;
   the on-disk format is identical to lossless. We honour scale_factor blindly.
3. **`difference_buffer` size bound.** C allocates a worst-case
   `RED_MAX_DIFFERENCE_BYTES(N) = N * 5` byte buffer (meflib.h L1101) — every
   sample could in theory be a 5-byte keysample restart. We do the same.
4. **`difference_bytes` includes the synthetic -128.** The encoder bumps it
   by 1 at L7000 so the on-disk count equals the byte count *with* the leading
   flag. The decoder reads `difference_bytes - 1` real bytes from the range
   coder — wait, no: the decoder reads `difference_bytes` from the range coder
   AND writes -128 first separately. So the decoded buffer is
   `1 + difference_bytes` bytes long. Confirmed by walking encode→decode:
   - encoder writes N actual bytes then bumps difference_bytes to N+1.
   - decoder writes 1 synthetic byte, then range-decodes N+1 = difference_bytes bytes.
   - But wait, that gives N+2 bytes. Re-read L6711 + L6718:
     - L6711: `*diff_buffer_p++ = -128;` (1 byte written)
     - L6718: loops `block_header->difference_bytes` times, each writing 1 byte.
     - Total bytes in diff_buffer: 1 + difference_bytes.
   - The encoder bumped difference_bytes by 1 (L7000). So the originally-
     emitted byte count was difference_bytes - 1, and 1 + difference_bytes
     = 1 + (real_count + 1) = real_count + 2. **That's 2 extra bytes vs the
     real encoded count.**
   - Inspection of the C output loop (L6751-6759): the loop consumes bytes from
     `si1_p1` until `number_of_samples` outputs are emitted. Trailing
     unconsumed bytes are harmless. The "extra" byte at the end of the decoded
     buffer is dummy padding from the range coder running one extra cycle to
     flush its state — exactly mirroring the encoder which emits 2 trailing
     bytes (L6993-6994) for the same purpose.
   - **Conclusion**: decode literally `difference_bytes` symbols from the
     range coder (after writing the synthetic -128), then walk the buffer
     consuming exactly enough to produce N samples. The trailing 1-2 bytes
     are ignored.

5. **CRC validation is mandatory by default** — meflib makes it opt-in via
   `MEF_globals->CRC_mode`, but in JS we want crashes loud and early.

---

## 8. Cross-references summary

| Concept                  | meflib.c lines |
|--------------------------|----------------|
| RED_decode entry         | L6639-6770     |
| CDF cumulative scan      | L6702-6708     |
| Range decoder init       | L6713-6716     |
| Renormalisation          | L6719-6730     |
| Symbol search            | L6731-6740     |
| Interval narrowing       | L6740-6745     |
| Diff buffer → samples    | L6748-6759     |
| RED_unscale              | L7464-7480     |
| RED_retrend              | L7294-7319     |
| RED_round                | L7322-7334     |
| RED_encode_exec (verify) | L6848-7049     |
| Block header constants   | L1035-1051     |
| Range coder constants    | L1066-1071     |

| Concept                  | meflib.h lines |
|--------------------------|----------------|
| RED_BLOCK_HEADER struct  | L1132-1145     |
| RED_PROCESSING_STRUCT    | L1167-1181     |
| RED_DISCONTINUITY_MASK   | L1054          |
| Sentinel sample values   | L1059-1063     |
