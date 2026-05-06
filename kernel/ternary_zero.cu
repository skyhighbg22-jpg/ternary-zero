// =====================================================================
// Ternary-Zero Core: W2A16 GEMV Kernel
// =====================================================================
// High-performance 2-bit Weight / 16-bit Activation GEMV
// Optimized for RTX 4060 (Ada Lovelace, sm_89, 128-bit memory bus)
//
// Weight encoding: 00->0, 01->+1, 10->-1
// 16 weights packed per uint32_t
// =====================================================================

#include "ternary_zero.h"
#include "ptx_utils.h"

#include <cuda_fp16.h>
#include <cstdint>

// =====================================================================
// Configuration Constants
// =====================================================================

// Threads per block
static constexpr int BLOCK_SIZE = 256;

// Number of warps per block
static constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;

// Activation tile size loaded into shared memory per iteration
// 1024 half values = 2048 bytes = 2 KB shared memory
static constexpr int ACT_TILE_SIZE = 1024;

// Each uint32_t packs 16 ternary weights
static constexpr int WEIGHTS_PER_UINT32 = 16;

// Each warp processes this many uint32_t per iteration
// warp_lane reads consecutive uint32_t for coalesced access
static constexpr int UINT32S_PER_WARP_ITER = 32;

// Total weights processed per warp per iteration
static constexpr int WEIGHTS_PER_WARP_ITER = UINT32S_PER_WARP_ITER * WEIGHTS_PER_UINT32;

// =====================================================================
// Shared Memory Layout
// =====================================================================
// s_act[TILE_N]:  Activation tile (FP16)
// s_warp_sums[WARPS_PER_BLOCK]: Partial sums from each warp (FP16x2)
// =====================================================================

// =====================================================================
// PTX Warp Shuffle Reduction
// =====================================================================

__device__ __forceinline__ half2 warp_reduce_sum_half2(half2 val) {
    // Butterfly reduction within a 32-thread warp
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        half2 other;
        other.x = __shfl_down_sync(0xFFFFFFFF, val.x, offset);
        other.y = __shfl_down_sync(0xFFFFFFFF, val.y, offset);
        val = __hadd2(val, other);
    }
    return val;
}

// =====================================================================
// Main Kernel: Ternary-Zero GEMV
// =====================================================================
//
// Grid:  (M, 1, 1)
// Block: (BLOCK_SIZE, 1, 1)
//
// Each block computes one output row: output[row] = W[row,:] * x
// =====================================================================

__global__ void __launch_bounds__(BLOCK_SIZE, 6)
ternary_zero_gemv_kernel(
    const uint32_t* __restrict__ weights,       // [M x (N/16)] packed
    const __half*   __restrict__ activations,    // [N]
    __half*         __restrict__ output,         // [M]
    const int M,
    const int N
) {
    // =================================================================
    // Thread/Warp identification
    // =================================================================
    const int row       = blockIdx.x;                  // output row
    const int tid       = threadIdx.x;                 // thread within block
    const int warp_id   = tid / 32;                    // warp index
    const int lane_id   = tid % 32;                    // lane within warp

    if (row >= M) return;

    // =================================================================
    // Shared memory allocation
    // =================================================================
    __shared__ __half  s_act[ACT_TILE_SIZE];           // activation tile
    __shared__ half2   s_warp_sums[WARPS_PER_BLOCK];   // per-warp partial sums

    // =================================================================
    // Pointers for this row
    // =================================================================
    const int packed_cols = N / WEIGHTS_PER_UINT32;    // number of uint32_t per row
    const uint32_t* row_weights = weights + (size_t)row * packed_cols;

    // =================================================================
    // Accumulator: half2 for vectorized SIMD accumulation
    // We process two FP16 values simultaneously
    // =================================================================
    half2 acc = __float2half2_rn(0.0f);

    // =================================================================
    // Main loop: iterate over activation tiles
    // =================================================================
    for (int tile_start = 0; tile_start < N; tile_start += ACT_TILE_SIZE) {

        // =============================================================
        // Cooperative loading of activation tile into shared memory
        // =============================================================
        const int tile_end = min(tile_start + ACT_TILE_SIZE, N);
        const int tile_len = tile_end - tile_start;

        // Each thread loads multiple half values
        for (int i = tid; i < tile_len; i += BLOCK_SIZE) {
            s_act[i] = activations[tile_start + i];
        }
        __syncthreads();

        // =============================================================
        // Compute: process weights against this activation tile
        // =============================================================
        // Weight index range for this tile:
        //   weights at positions [tile_start, tile_end) map to
        //   packed indices [tile_start/16, tile_end/16)
        const int packed_tile_start = tile_start / WEIGHTS_PER_UINT32;
        const int packed_tile_end   = tile_end / WEIGHTS_PER_UINT32;
        const int packed_tile_count = packed_tile_end - packed_tile_start;

        // Each warp processes a stripe of packed uint32_t values
        for (int p_idx = warp_id * UINT32S_PER_WARP_ITER + lane_id;
             p_idx < packed_tile_count;
             p_idx += BLOCK_SIZE)
        {
            // Global packed weight index
            const int global_p_idx = packed_tile_start + p_idx;

            // Bounds check
            if (global_p_idx >= packed_cols) break;

            // =========================================================
            // PTX Load: coalesced 32-bit load from global memory
            // =========================================================
            uint32_t packed = row_weights[global_p_idx];

            // =========================================================
            // PTX Decompression: extract 16 ternary weights
            // =========================================================
            // PRMT byte-permute for alignment
            uint32_t aligned;
            PTX_PRMT(aligned, packed, 0, 0x3210);

            // Base activation index for this packed word within the tile
            const int act_base = p_idx * WEIGHTS_PER_UINT32;

            // =========================================================
            // Zero-Gate Accumulation: process 16 weights
            // =========================================================
            // Unroll the 16-weight decompression and accumulation
            // Process 2 weights per iteration for half2 vectorization
            #pragma unroll
            for (int w = 0; w < WEIGHTS_PER_UINT32; w += 2) {
                // Extract 2-bit weight w
                uint32_t bits_w0, bits_w1;
                PTX_BFE(bits_w0, aligned, w * 2, 2);
                PTX_BFE(bits_w1, aligned, (w + 1) * 2, 2);

                // Decode w0: sign = bit1, magnitude = bit0
                uint32_t sign_w0, mag_w0;
                PTX_BFE(sign_w0, bits_w0, 1, 1);
                PTX_BFE(mag_w0, bits_w0, 0, 1);

                // Decode w1
                uint32_t sign_w1, mag_w1;
                PTX_BFE(sign_w1, bits_w1, 1, 1);
                PTX_BFE(mag_w1, bits_w1, 0, 1);

                // Load activations from shared memory
                const int a0_idx = act_base + w;
                const int a1_idx = a0_idx + 1;

                // Bounds check within tile
                if (a0_idx >= tile_len) break;

                half a0 = s_act[a0_idx];
                half a1 = (a1_idx < tile_len) ? s_act[a1_idx] : __float2half(0.0f);

                // =====================================================
                // Zero-Gate Branchless Accumulation
                // =====================================================
                // Apply sign: negate if sign_bit=1 (using FP16 sign flip)
                // Apply zero-gate: zero-out if mag_bit=0

                // For a0:
                //   If mag=0: contribution = 0 (zero-gate skip)
                //   If mag=1 and sign=0: contribution = +a0
                //   If mag=1 and sign=1: contribution = -a0

                // Branchless via predicated sign flip + zero mask
                uint32_t a0_raw = __half_as_ushort(a0);
                uint32_t a1_raw = __half_as_ushort(a1);

                // Flip sign bit if sign=1
                uint32_t sign_mask_w0 = (uint32_t)(-(int32_t)sign_w0) & 0x8000u;
                uint32_t sign_mask_w1 = (uint32_t)(-(int32_t)sign_w1) & 0x8000u;

                uint32_t signed_a0 = a0_raw ^ sign_mask_w0;
                uint32_t signed_a1 = a1_raw ^ sign_mask_w1;

                // Zero-gate: AND with 0xFFFF if non-zero, 0x0000 if zero
                uint32_t nz_mask_w0 = (uint32_t)(-(int32_t)mag_w0);
                uint32_t nz_mask_w1 = (uint32_t)(-(int32_t)mag_w1);

                uint32_t gated_a0 = signed_a0 & nz_mask_w0;
                uint32_t gated_a1 = signed_a1 & nz_mask_w1;

                // Reinterpret as half and add to accumulator
                half2 contribution;
                contribution.x = __ushort_as_half((unsigned short)(gated_a0 & 0xFFFF));
                contribution.y = __ushort_as_half((unsigned short)(gated_a1 & 0xFFFF));

                acc = __hadd2(acc, contribution);
            }
        }

        __syncthreads();
    }

    // =================================================================
    // Warp-Level Reduction
    // =================================================================
    // First reduce the half2 accumulator to a single half
    half partial_sum = __hadd(acc.x, acc.y);

    // Warp-level butterfly reduction using __shfl_down_sync
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        half other = __shfl_down_sync(0xFFFFFFFF, partial_sum, offset);
        partial_sum = __hadd(partial_sum, other);
    }

    // First lane of each warp writes to shared memory
    if (lane_id == 0) {
        s_warp_sums[warp_id] = make_half2(partial_sum, __float2half(0.0f));
    }
    __syncthreads();

    // =================================================================
    // Block-Level Reduction (first warp reduces across all warps)
    // =================================================================
    if (warp_id == 0) {
        half block_sum;
        if (lane_id < WARPS_PER_BLOCK) {
            block_sum = s_warp_sums[lane_id].x;
        } else {
            block_sum = __float2half(0.0f);
        }

        // Reduce the 8 warp sums
        #pragma unroll
        for (int offset = WARPS_PER_BLOCK / 2; offset >= 1; offset >>= 1) {
            half other = __shfl_down_sync(0xFFFFFFFF, block_sum, offset);
            block_sum = __hadd(block_sum, other);
        }

        // Thread 0 writes the final result
        if (lane_id == 0) {
            output[row] = block_sum;
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

    // Grid: one block per output row
    dim3 grid(M, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);

    // Shared memory: activation tile + warp sums
    // Dynamically allocated would be needed for variable tile sizes,
    // but we use a fixed tile so static shared memory suffices.

    ternary_zero_gemv_kernel<<<grid, block, 0, stream>>>(
        weights, activations, output, M, N
    );

    return cudaGetLastError();
}
