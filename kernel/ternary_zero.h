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

// =====================================================================
// L2 Cache Policy: Pin weight tiles in L2 for repeated access
// =====================================================================
cudaError_t ternary_zero_set_l2_policy(
    cudaStream_t stream,
    const void* base_ptr,
    size_t num_bytes
);

#ifdef __cplusplus
}
#endif

#endif // TERNARY_ZERO_H
