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

#ifndef TERNARY_ZERO_H
#define TERNARY_ZERO_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// =====================================================================
// Ternary-Zero GEMV Kernel: W2A16 (2-bit Weight, 16-bit Activation)
// =====================================================================
//
// Computes: output[m] = sum_n( decode_ternary(weights[m,n]) * activations[n] )
//
// Weight encoding (2-bit packed into uint32_t):
//   00 -> 0  (zero-gate: skip accumulation)
//   01 -> +1 (add activation)
//   10 -> -1 (subtract activation)
//
// Args:
//   weights     - Packed 2-bit weights [M x ceil(N/16)], column-major tiled
//   activations - FP16 activation vector [N]
//   output      - FP16 output vector [M]
//   M           - Number of output rows
//   N           - Number of input features (must be multiple of 16)
//
// Returns: cudaSuccess on success
// =====================================================================
cudaError_t ternary_zero_gemv_f16(
    const uint32_t* __restrict__ weights,
    const __half*   __restrict__ activations,
    __half*         __restrict__ output,
    int M,
    int N,
    cudaStream_t stream
);

cudaError_t ternary_zero_gemv_f16_ex(
    const uint32_t* __restrict__ weights,
    const __half*   __restrict__ activations,
    __half*         __restrict__ output,
    int M,
    int N,
    cudaStream_t stream,
    int use_fp32_acc
);

// =====================================================================
// L2 Cache Policy: Pin weight tiles in L2 for repeated access
// =====================================================================
cudaError_t ternary_zero_set_l2_policy(
    cudaStream_t stream,
    const void* base_ptr,
    size_t num_bytes
);

// =====================================================================
// Advanced L2 Persistence (l2_persist.cu)
// =====================================================================

// Pin a single layer's weights in L2 cache with automatic hit ratio.
cudaError_t l2_persist_single_layer(
    cudaStream_t stream,
    const uint32_t* weights,
    int M, int N,
    float* hit_ratio_out,
    size_t* bytes_pinned_out
);

// FFN two-phase L2 pinning for Llama-2-7B.
cudaError_t l2_persist_ffn_phase1_gate_up(
    cudaStream_t stream,
    const uint32_t* gate_weights,
    const uint32_t* up_weights,
    int intermediate_size,
    int hidden_size
);
cudaError_t l2_persist_ffn_phase2_down(
    cudaStream_t stream,
    const uint32_t* down_weights,
    int hidden_size,
    int intermediate_size
);

// Tiled L2 pinning for layers exceeding L2 capacity.
cudaError_t l2_persist_tiled(
    cudaStream_t stream,
    const uint32_t* weights,
    int M, int N,
    int tile_index,
    int* tile_count_out,
    int* tile_rows_out
);

// Print L2 analysis for a given layer shape.
void l2_print_analysis(int M, int N, const char* label);

// =====================================================================
// NVTX-Profiled Launch Wrapper
// =====================================================================
// Launches the kernel and records per-phase timing via CUDA events.
// For full NVTX instrumentation (H2D, L2 policy, kernel, D2H),
// include kernel/nvt/ternary_zero_nvtx.h instead.
//
// Args:
//   phase_tile_load_us  - [out] shared memory tile load time (us)
//   phase_decode_us     - [out] bit decode + accumulate time (us)
//   phase_reduce_us     - [out] warp+block reduction + output time (us)
//
// Note: Requires Nsight Compute for per-phase GPU-side timing.
// This API records a single kernel event pair and returns 0 for all
// phases when run without ncu. Use ncu --export sqlite to populate.
// =====================================================================
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
);

#ifdef __cplusplus
}
#endif

#endif // TERNARY_ZERO_H
