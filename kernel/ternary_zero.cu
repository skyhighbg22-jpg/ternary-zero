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
// Main Kernel: Ternary-Zero GEMV
// =====================================================================

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

    __shared__ __half s_act[PADDED_TILE_SIZE];
    __shared__ float  s_warp_sums[WARPS_PER_BLOCK];

    const int packed_cols = N / WEIGHTS_PER_UINT32;
    const uint32_t* row_weights = weights + (size_t)row * packed_cols;

    float acc = 0.0f;

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

                float v0 = __half2float(__ushort_as_half((unsigned short)(gated_a0 & 0xFFFF)));
                float v1 = __half2float(__ushort_as_half((unsigned short)(gated_a1 & 0xFFFF)));
                acc += v0 + v1;
            }
        }

        __syncthreads();
    }

    acc = warp_reduce_sum_f32(acc);

    if (lane_id == 0) {
        s_warp_sums[warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = (lane_id < WARPS_PER_BLOCK) ? s_warp_sums[lane_id]
                                                       : 0.0f;
        #pragma unroll
        for (int offset = WARPS_PER_BLOCK / 2; offset >= 1; offset >>= 1) {
            block_sum += __shfl_down_sync(0xFFFFFFFF, block_sum, offset);
        }

        if (lane_id == 0) {
            output[row] = __float2half(block_sum);
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
// Host-Side Launch Wrapper
// =====================================================================

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

    ternary_zero_gemv_kernel<<<grid, block, 0, stream>>>(
        weights, activations, output, M, N
    );

    return cudaGetLastError();
}