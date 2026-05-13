// Copyright (C) 2025 ternary-zero contributors
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
#define PTX_PRMT(out, a, b, sel)                                               \
    asm volatile("prmt.b32 %0, %1, %2, %3;"                                    \
                 : "=r"(out) : "r"(a), "r"(b), "r"(sel))

// LOP3.LUT: 3-input logical operation via 8-bit truth table
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

#ifdef __CUDA_ARCH__

// =====================================================================
// LOP3.LUT Truth Table Constants
// =====================================================================

#define LUT_EXTRACT_SIGN   0xA0
#define LUT_EXTRACT_MAG    0xC0
#define LUT_NONZERO        0xEC
#define LUT_ANDNOT         0x0C

// =====================================================================
// Inline PTX: Zero-gate conditional accumulation
// =====================================================================

__device__ __forceinline__ half2 zero_gate_accumulate_ptx(
    half2 accumulator,
    half2 activation,
    uint32_t sign_bit,
    uint32_t mag_bit
) {
    uint32_t nonzero = sign_bit | mag_bit;
    uint32_t nz_mask = (uint32_t)(-(int32_t)nonzero);
    uint32_t neg_mask = (uint32_t)(-(int32_t)sign_bit);

    unsigned short a_lo = __half_as_ushort(activation.x);
    unsigned short a_hi = __half_as_ushort(activation.y);
    uint32_t act_bits = ((uint32_t)a_hi << 16) | (uint32_t)a_lo;

    uint32_t sign_flip_mask = neg_mask & 0x80008000u;
    uint32_t signed_act;
    PTX_LOP3_LUT(signed_act, act_bits, sign_flip_mask, 0, 0x6C);

    uint32_t gated_act;
    PTX_LOP3_LUT(gated_act, signed_act, nz_mask, 0, LUT_EXTRACT_SIGN);

    half2 contribution;
    contribution.x = __ushort_as_half((unsigned short)(gated_act & 0xFFFF));
    contribution.y = __ushort_as_half((unsigned short)(gated_act >> 16));
    half2 result = __hadd2(accumulator, contribution);
    return result;
}

#endif // __CUDA_ARCH__

#endif // PTX_UTILS_H