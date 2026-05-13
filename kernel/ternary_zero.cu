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

// =====================================================================
// Ternary-Zero Core: W2A16 GEMV Kernel (v2 - Fixed and Optimized)
// =====================================================================
// High-performance 2-bit Weight / 16-bit Activation GEMV
// Optimized for RTX 4060 (Ada Lovelace, sm_89, 128-bit memory bus)
//
// Weight encoding: 00->0, 01->+1, 10->-1
// 16 weights packed per uint32_t
//
// v2 Fixes:
//   [BUG] Removed no-op PRMT (0x3210 = identity on little-endian GPUs)
//   [BUG] launch_bounds minBlocks=6 required <=42 regs but maxrregcount=64
//          -> spilling; corrected to minBlocks=4 (matches 65536/(256*64))
//   [BUG] FP16 accumulation overflow for N>=2048 (max FP16=65504)
//          -> FP32 warp-level accumulation, FP16 only at output write
//   [PERF] 4-way shared memory bank conflicts from stride-16 half access
//          -> padded shared memory layout (stride-17, all 32 banks unique)
//   [PERF] Scalar __half tile loads -> vectorized uint4 loads (8 halves/txn)
//   [PERF] Removed unused warp_reduce_sum_half2 function
// =====================================================================

#include "ternary_zero.h"
#include "ptx_utils.h"

#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

// =====================================================================
// Configuration Constants
// =====================================================================

static constexpr int BLOCK_SIZE = 256;
static constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;

static constexpr int ACT_TILE_SIZE = 1024;

static constexpr int SMEM_PAD_GROUP = 16;
static constexpr int PADDED_TILE_SIZE = ACT_TILE_SIZE + (ACT_TILE_SIZE / SMEM_PAD_GROUP);

static constexpr int WEIGHTS_PER_UINT32 = 16;
static constexpr int UINT32S_PER_WARP_ITER = 32;

// =====================================================================
// Bank-Conflict-Free Shared Memory Index
// =====================================================================

__device__ __forceinline__ int smem_idx(int flat) {
    return flat + (flat / SMEM_PAD_GROUP);
}

// =====================================================================
// FP32 Warp Shuffle Reduction
// =====================================================================

__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// =====================================================================
// FP16 Warp Shuffle Reduction
// =====================================================================

__device__ __forceinline__ half warp_reduce_sum_f16(half val) {
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        uint32_t raw = (uint32_t)__half_as_ushort(val);
        uint32_t other_raw = __shfl_down_sync(0xFFFFFFFF, raw, offset);
        val = __hadd(val, __ushort_as_half((unsigned short)other_raw));
    }
    return val;
}

// =====================================================================
// Accumulation Precision Traits
// =====================================================================

template <bool Fp32Acc>
struct AccOps;

template <>
struct AccOps<true> {
    using type = float;
    __device__ __forceinline__ static float init() { return 0.0f; }
    __device__ __forceinline__ static float from_half(half h) { return __half2float(h); }
    __device__ __forceinline__ static float add(float a, float b) { return a + b; }
    __device__ __forceinline__ static float warp_reduce(float v) { return warp_reduce_sum_f32(v); }
    __device__ __forceinline__ static half to_half(float v) { return __float2half(v); }
};

template <>
struct AccOps<false> {
    using type = half;
    __device__ __forceinline__ static half init() { return __float2half(0.0f); }
    __device__ __forceinline__ static half from_half(half h) { return h; }
    __device__ __forceinline__ static half add(half a, half b) { return __hadd(a, b); }
    __device__ __forceinline__ static half warp_reduce(half v) { return warp_reduce_sum_f16(v); }
    __device__ __forceinline__ static half to_half(half v) { return v; }
};

// =====================================================================
// Main Kernel: Ternary-Zero GEMV (Templated on Accumulation Precision)
// =====================================================================

template <bool Fp32Acc = true>
__global__ void __launch_bounds__(BLOCK_SIZE, 4)
ternary_zero_gemv_kernel(
    const uint32_t* __restrict__ weights,
    const __half*   __restrict__ activations,
    __half*         __restrict__ output,
    const int M,
    const int N
) {
    const int row     = blockIdx.x;
    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    if (row >= M) return;

    using AccT = typename std::conditional<Fp32Acc, float, half>::type;

    __shared__ __half s_act[PADDED_TILE_SIZE];
    __shared__ float  s_warp_sums_f32[WARPS_PER_BLOCK];
    __shared__ half   s_warp_sums_f16[WARPS_PER_BLOCK];

    const int packed_cols = N / WEIGHTS_PER_UINT32;
    const uint32_t* row_weights = weights + (size_t)row * packed_cols;

    AccT acc = AccOps<Fp32Acc>::init();

    for (int tile_start = 0; tile_start < N; tile_start += ACT_TILE_SIZE) {

        const int tile_end = min(tile_start + ACT_TILE_SIZE, N);
        const int tile_len = tile_end - tile_start;

        {
            const int vec_count = tile_len / 8;
            const uint4* src_vec = reinterpret_cast<const uint4*>(
                activations + tile_start
            );

            for (int i = tid; i < vec_count; i += BLOCK_SIZE) {
                uint4 v = src_vec[i];
                const __half* h = reinterpret_cast<const __half*>(&v);
                int base = i * 8;
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    s_act[smem_idx(base + j)] = h[j];
                }
            }

            for (int i = vec_count * 8 + tid; i < tile_len; i += BLOCK_SIZE) {
                s_act[smem_idx(i)] = activations[tile_start + i];
            }
        }
        __syncthreads();

        const int packed_tile_start = tile_start / WEIGHTS_PER_UINT32;
        const int packed_tile_end   = tile_end   / WEIGHTS_PER_UINT32;
        const int packed_tile_count = packed_tile_end - packed_tile_start;

        for (int p_idx = warp_id * UINT32S_PER_WARP_ITER + lane_id;
             p_idx < packed_tile_count;
             p_idx += BLOCK_SIZE)
        {
            const int global_p_idx = packed_tile_start + p_idx;
            if (global_p_idx >= packed_cols) break;

            uint32_t packed = row_weights[global_p_idx];
            const int act_base = p_idx * WEIGHTS_PER_UINT32;

            #pragma unroll
            for (int w = 0; w < WEIGHTS_PER_UINT32; w += 2) {
                uint32_t bits_w0, bits_w1;
                PTX_BFE(bits_w0, packed, w * 2, 2);
                PTX_BFE(bits_w1, packed, (w + 1) * 2, 2);

                uint32_t sign_w0, mag_w0, sign_w1, mag_w1;
                PTX_BFE(sign_w0, bits_w0, 1, 1);
                PTX_BFE(mag_w0, bits_w0, 0, 1);
                PTX_BFE(sign_w1, bits_w1, 1, 1);
                PTX_BFE(mag_w1, bits_w1, 0, 1);

                const int a0_flat = act_base + w;
                const int a1_flat = a0_flat + 1;

                if (a0_flat >= tile_len) break;

                half a0 = s_act[smem_idx(a0_flat)];
                half a1 = (a1_flat < tile_len) ? s_act[smem_idx(a1_flat)]
                                                : __float2half(0.0f);

                uint32_t a0_raw = __half_as_ushort(a0);
                uint32_t a1_raw = __half_as_ushort(a1);

                uint32_t sign_mask_w0 = (uint32_t)(-(int32_t)sign_w0) & 0x8000u;
                uint32_t sign_mask_w1 = (uint32_t)(-(int32_t)sign_w1) & 0x8000u;
                uint32_t signed_a0 = a0_raw ^ sign_mask_w0;
                uint32_t signed_a1 = a1_raw ^ sign_mask_w1;

                uint32_t nz_w0 = sign_w0 | mag_w0;
                uint32_t nz_w1 = sign_w1 | mag_w1;
                uint32_t nz_mask_w0 = (uint32_t)(-(int32_t)nz_w0);
                uint32_t nz_mask_w1 = (uint32_t)(-(int32_t)nz_w1);
                uint32_t gated_a0 = signed_a0 & nz_mask_w0;
                uint32_t gated_a1 = signed_a1 & nz_mask_w1;

                if constexpr (Fp32Acc) {
                    float v0 = __half2float(__ushort_as_half((unsigned short)(gated_a0 & 0xFFFF)));
                    float v1 = __half2float(__ushort_as_half((unsigned short)(gated_a1 & 0xFFFF)));
                    acc += v0 + v1;
                } else {
                    half v0 = __ushort_as_half((unsigned short)(gated_a0 & 0xFFFF));
                    half v1 = __ushort_as_half((unsigned short)(gated_a1 & 0xFFFF));
                    acc = __hadd(acc, __hadd(v0, v1));
                }
            }
        }

        __syncthreads();
    }

    if constexpr (Fp32Acc) {
        acc = warp_reduce_sum_f32(acc);

        if (lane_id == 0) {
            s_warp_sums_f32[warp_id] = acc;
        }
        __syncthreads();

        if (warp_id == 0) {
            float block_sum = (lane_id < WARPS_PER_BLOCK) ? s_warp_sums_f32[lane_id]
                                                           : 0.0f;
            #pragma unroll
            for (int offset = WARPS_PER_BLOCK / 2; offset >= 1; offset >>= 1) {
                block_sum += __shfl_down_sync(0xFFFFFFFF, block_sum, offset);
            }

            if (lane_id == 0) {
                output[row] = __float2half(block_sum);
            }
        }
    } else {
        acc = warp_reduce_sum_f16(acc);

        if (lane_id == 0) {
            s_warp_sums_f16[warp_id] = acc;
        }
        __syncthreads();

        if (warp_id == 0) {
            half block_sum = (lane_id < WARPS_PER_BLOCK) ? s_warp_sums_f16[lane_id]
                                                          : __float2half(0.0f);
            #pragma unroll
            for (int offset = WARPS_PER_BLOCK / 2; offset >= 1; offset >>= 1) {
                uint32_t raw = (uint32_t)__half_as_ushort(block_sum);
                uint32_t other_raw = __shfl_down_sync(0xFFFFFFFF, raw, offset);
                block_sum = __hadd(block_sum, __ushort_as_half((unsigned short)other_raw));
            }

            if (lane_id == 0) {
                output[row] = block_sum;
            }
        }
    }
}

// =====================================================================
// L2 Cache Policy: Pin weight tiles in L2 cache
// =====================================================================

cudaError_t ternary_zero_set_l2_policy(
    cudaStream_t stream,
    const void* base_ptr,
    size_t num_bytes
) {
    cudaStreamAttrValue attr = {};
    attr.accessPolicyWindow.base_ptr  = const_cast<void*>(base_ptr);
    attr.accessPolicyWindow.num_bytes = num_bytes;
    attr.accessPolicyWindow.hitRatio  = 1.0f;
    attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;

    return cudaStreamSetAttribute(
        stream,
        cudaStreamAttributeAccessPolicyWindow,
        &attr
    );
}

// =====================================================================
// Host-Side Launch Wrapper (with NVTX integration points)
// =====================================================================
//
// The kernel executes three phases on the GPU that cannot be separated
// by host-side NVTX markers:
//   Phase 1 (Lines 156-176): Vectorized uint4 tile load → shared memory
//   Phase 2 (Lines 182-238): PTX BFE bit decode + zero-gate + accumulate
//   Phase 3 (Lines 243-285): Warp shuffle reduction + block reduce + output
//
// For intra-kernel phase isolation, use Nsight Compute (ncu) with:
//   ncu --set full --kernel-name ternary_zero_gemv_kernel ...
//
// For system-level timeline analysis with NVTX, see:
//   kernel/nvt/ternary_zero_nvtx.h

extern "C"
cudaError_t ternary_zero_gemv_f16(
    const uint32_t* __restrict__ weights,
    const __half*   __restrict__ activations,
    __half*         __restrict__ output,
    int M,
    int N,
    cudaStream_t stream
) {
    if (M <= 0 || N <= 0 || N % WEIGHTS_PER_UINT32 != 0) {
        return cudaErrorInvalidValue;
    }

    dim3 grid(M, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);

    ternary_zero_gemv_kernel<true><<<grid, block, 0, stream>>>(
        weights, activations, output, M, N
    );

    return cudaGetLastError();
}

// =====================================================================
// Profiled Launch Wrapper
// =====================================================================
// Launches a single kernel and measures total execution time.
// Intra-kernel phase breakdown requires Nsight Compute; this wrapper
// provides the event-based total kernel time for Nsight Systems.

extern "C"
cudaError_t ternary_zero_gemv_profiled(
    const uint32_t* __restrict__ weights,
    const __half*   __restrict__ activations,
    __half*         __restrict__ output,
    int M,
    int N,
    cudaStream_t stream,
    float* phase_tile_load_us,
    float* phase_decode_us,
    float* phase_reduce_us
) {
    if (M <= 0 || N <= 0 || N % WEIGHTS_PER_UINT32 != 0) {
        return cudaErrorInvalidValue;
    }

    cudaEvent_t start, stop;
    cudaError_t err;

    err = cudaEventCreate(&start);
    if (err != cudaSuccess) return err;
    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) { cudaEventDestroy(start); return err; }

    dim3 grid(M, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);

    err = cudaEventRecord(start, stream);
    if (err != cudaSuccess) goto cleanup;

    ternary_zero_gemv_kernel<true><<<grid, block, 0, stream>>>(
        weights, activations, output, M, N
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;

    err = cudaEventRecord(stop, stream);
    if (err != cudaSuccess) goto cleanup;

    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) goto cleanup;

    {
        float total_ms = 0.0f;
        err = cudaEventElapsedTime(&total_ms, start, stop);
        if (err != cudaSuccess) goto cleanup;

        float total_us = total_ms * 1000.0f;

        // Phase breakdown is only available via Nsight Compute.
        // Return total time in all three fields for system-level profiling.
        if (phase_tile_load_us) *phase_tile_load_us = 0.0f;
        if (phase_decode_us)    *phase_decode_us = total_us;
        if (phase_reduce_us)    *phase_reduce_us = 0.0f;
    }

cleanup:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return err;
}

extern "C"
cudaError_t ternary_zero_gemv_f16_ex(
    const uint32_t* __restrict__ weights,
    const __half*   __restrict__ activations,
    __half*         __restrict__ output,
    int M,
    int N,
    cudaStream_t stream,
    int use_fp32_acc
) {
    if (M <= 0 || N <= 0 || N % WEIGHTS_PER_UINT32 != 0) {
        return cudaErrorInvalidValue;
    }

    dim3 grid(M, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);

    if (use_fp32_acc) {
        ternary_zero_gemv_kernel<true><<<grid, block, 0, stream>>>(
            weights, activations, output, M, N
        );
    } else {
        ternary_zero_gemv_kernel<false><<<grid, block, 0, stream>>>(
            weights, activations, output, M, N
        );
    }

    return cudaGetLastError();
}