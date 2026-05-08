#ifndef PTX_UTILS_H
#define PTX_UTILS_H

#include <cuda_fp16.h>
#include <cstdint>

// =====================================================================
// PTX Bit-Manipulation Macros for 2-Bit Ternary Weight Decompression
// =====================================================================
//
// Encoding: 00 -> 0 (zero), 01 -> +1 (positive), 10 -> -1 (negative)
// Each uint32_t holds 16 weights: [w0:lsb | w1 | w2 | ... | w15:msb]
// =====================================================================

// PRMT (Permute Bytes): Rearrange bytes within a register
// Useful for byte-level repositioning before bit extraction
#define PTX_PRMT(out, a, b, sel)                                               \
    asm volatile("prmt.b32 %0, %1, %2, %3;"                                    \
                 : "=r"(out) : "r"(a), "r"(b), "r"(sel))

// LOP3.LUT: 3-input logical operation via 8-bit truth table
// dst = LUT(a, b, c) where LUT is the truth table byte
// For ternary extraction: mask bits and classify simultaneously
#define PTX_LOP3_LUT(dst, a, b, c, lut)                                        \
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;"                                \
                 : "=r"(dst) : "r"(a), "r"(b), "r"(c), "r"(lut))

// BFE (Bit Field Extract): Extract bits [pos, pos+len)
#define PTX_BFE(dst, src, pos, len)                                            \
    asm volatile("bfe.u32 %0, %1, %2, %3;"                                     \
                 : "=r"(dst) : "r"(src), "r"(pos), "r"(len))

// BFI (Bit Field Insert): Insert bits into a register
#define PTX_BFI(dst, src, base, pos, len)                                      \
    asm volatile("bfi.b32 %0, %1, %2, %3, %4;"                                 \
                 : "=r"(dst) : "r"(src), "r"(base), "r"(pos), "r"(len))

// SHL (Shift Left)
#define PTX_SHL(dst, src, shift)                                               \
    asm volatile("shl.b32 %0, %1, %2;"                                         \
                 : "=r"(dst) : "r"(src), "r"(shift))

// SHR (Shift Right)
#define PTX_SHR(dst, src, shift)                                               \
    asm volatile("shr.b32 %0, %1, %2;"                                         \
                 : "=r"(dst) : "r"(src), "r"(shift))

// SELP (Select with Predicate): Conditional select
#define PTX_SELP(dst, a, b, pred)                                              \
    asm volatile("{ .reg .pred p; setp.ne.b32 p, %3, 0; selp.b32 %0, %1, %2, p; }" \
                 : "=r"(dst) : "r"(a), "r"(b), "r"(pred))

// =====================================================================
// LOP3.LUT Truth Table Constants
// =====================================================================

// LUT for extracting sign bit (bit 1) from 2-bit weight:
//   input bits: (weight, mask_sign, 0)
//   output: sign_bit where weight & mask_sign != 0
#define LUT_EXTRACT_SIGN   0xA0   // (a & b)

// LUT for extracting magnitude bit (bit 0) from 2-bit weight:
//   input bits: (weight, mask_mag, 0)
//   output: mag_bit where weight & mask_mag != 0
#define LUT_EXTRACT_MAG    0xC0   // (a & b)

// LUT for computing nonzero flag: sign OR magnitude
//   output: 1 if any bit set in the 2-bit field
#define LUT_NONZERO        0xEC   // (a | b) masked

// LUT for AND-NOT: mask application
#define LUT_ANDNOT         0x0C   // a & ~b

// =====================================================================
// Inline PTX: Unpack 16 ternary weights from a single uint32_t
// =====================================================================
//
// Strategy:
//   1. Use PRMT to align bytes for convenient extraction
//   2. Process 4 weights per iteration using mask+shift
//   3. For each 2-bit weight: extract sign (bit1) and magnitude (bit0)
//   4. Zero-weight detection: both bits zero -> skip
//
// Output arrays (must be allocated by caller):
//   signs[16]: 0 or 1 (1 means negative)
//   mags[16]:  0 or 1 (1 means non-zero)
// =====================================================================

__device__ __forceinline__ void unpack_16_ternary_ptx(
    uint32_t packed,
    uint32_t* __restrict__ signs,
    uint32_t* __restrict__ mags
) {
    // Byte-permute for alignment: ensure bytes are in natural order
    uint32_t aligned;
    PTX_PRMT(aligned, packed, 0, 0x3210);

    // Masks for extracting individual 2-bit fields
    // Process in groups of 4 weights (8 bits = 1 byte)
    const uint32_t mask_2bit = 0x03;  // isolate 2-bit field

    // Unroll: extract all 16 weights
    // Each weight is at bit position (i*2)
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint32_t bits;
        uint32_t pos = i * 2;

        // Use BFE to extract 2 bits at position pos
        PTX_BFE(bits, aligned, pos, 2);

        // sign = bit 1 of the 2-bit field (means negative)
        uint32_t sign_bit;
        PTX_BFE(sign_bit, bits, 1, 1);

        // magnitude = bit 0 of the 2-bit field (means non-zero)
        uint32_t mag_bit;
        PTX_BFE(mag_bit, bits, 0, 1);

        signs[i] = sign_bit;
        mags[i] = mag_bit;
    }
}

// =====================================================================
// Inline PTX: Vectorized unpack of 4 weights simultaneously
// =====================================================================
// Uses LOP3.LUT to classify 4 weights in parallel via bitmask operations
// More efficient than per-weight extraction for the critical inner loop

__device__ __forceinline__ void unpack_4_ternary_vectorized(
    uint32_t byte_val,    // 8 bits containing 4 weights
    uint32_t& sign_mask,  // output: 4-bit mask of sign bits
    uint32_t& mag_mask    // output: 4-bit mask of magnitude bits
) {
    // byte_val layout: [w0:w1:w2:w3] each 2 bits
    // Extract odd bits (sign) and even bits (magnitude)

    // Sign mask: bits at positions 1,3,5,7
    uint32_t sign_pattern = 0xAA;  // 10101010
    PTX_LOP3_LUT(sign_mask, byte_val, sign_pattern, 0, LUT_EXTRACT_SIGN);

    // Magnitude mask: bits at positions 0,2,4,6
    uint32_t mag_pattern = 0x55;   // 01010101
    PTX_LOP3_LUT(mag_mask, byte_val, mag_pattern, 0, LUT_EXTRACT_MAG);

    // Normalize: shift sign bits right by 1 so they align with magnitude positions
    uint32_t normalized_sign;
    PTX_SHR(normalized_sign, sign_mask, 1);

    sign_mask = normalized_sign;
    // mag_mask already in correct positions
}

// =====================================================================
// Inline PTX: Zero-gate conditional accumulation
// =====================================================================
// Branchless: if mag_bit=0, result is zero (skip); if sign_bit=1, negate

__device__ __forceinline__ half2 zero_gate_accumulate_ptx(
    half2 accumulator,
    half2 activation,
    uint32_t sign_bit,
    uint32_t mag_bit
) {
    // Create mask from mag_bit: 0xFFFFFFFF if non-zero, 0x00000000 if zero
    uint32_t nz_mask = (uint32_t)(-(int32_t)mag_bit);

    // Create mask from sign_bit: 0xFFFFFFFF if negative, 0x00000000 if positive
    uint32_t neg_mask = (uint32_t)(-(int32_t)sign_bit);

    // Reinterpret half2 as uint32_t for bitwise operations
    unsigned short a_lo = __half_as_ushort(activation.x);
    unsigned short a_hi = __half_as_ushort(activation.y);
    uint32_t act_bits = ((uint32_t)a_hi << 16) | (uint32_t)a_lo;

    // Apply sign: if negative, XOR with sign mask to negate (FP16 sign flip)
    // For half2: sign bit is bit 15 of each 16-bit lane
    uint32_t sign_flip_mask = neg_mask & 0x80008000u;
    uint32_t signed_act;
    PTX_LOP3_LUT(signed_act, act_bits, sign_flip_mask, 0, 0x6C);  // XOR

    // Apply zero-gate: AND with nz_mask
    uint32_t gated_act;
    PTX_LOP3_LUT(gated_act, signed_act, nz_mask, 0, LUT_EXTRACT_SIGN); // a & b

    // Reinterpret back to half2 and add to accumulator
    half2 contribution;
    contribution.x = __ushort_as_half((unsigned short)(gated_act & 0xFFFF));
    contribution.y = __ushort_as_half((unsigned short)(gated_act >> 16));
    half2 result = __hadd2(accumulator, contribution);
    return result;
}

#endif // PTX_UTILS_H
