// Copyright (C) 2025 ternary-zero contributors
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// =====================================================================
// Ternary-Zero NVTX Instrumentation Layer
// =====================================================================
// Provides fine-grained NVTX ranges for Nsight Systems profiling.
// Isolates the three critical phases of the W2A16 GEMV kernel:
//   1. Tile Load   — vectorized uint4 activation staging to shared memory
//   2. Bit Decode  — PTX BFE + zero-gate + sign-flip + accumulate
//   3. Reduction   — warp shuffle + block reduction + output write
//
// Usage:
//   #define TZ_NVTX_ENABLED 1   // before including this header
//   ternary_zero_gemv_profiled(...);
//
// In Nsight Systems, filter by NVTX category "ternary_zero" to see
// the per-phase breakdown. Each phase is a separate NVTX range with
// a color code:
//   - Tile Load:   Green (0xFF00AA00)
//   - Bit Decode:  Red   (0xFFAA0000)
//   - Reduction:   Blue  (0xFF0000AA)
//
// Build:
//   nvcc -O3 --use_fast_math --gpu-architecture=sm_89 -std=c++17
//        -DTZ_NVTX_ENABLED=1 -I../kernel ...
// =====================================================================

#ifndef TERNARY_ZERO_NVTX_H
#define TERNARY_ZERO_NVTX_H

#include "ternary_zero.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

#ifdef TZ_NVTX_ENABLED
#include <nvtx3/nvToolsExt.h>

// =====================================================================
// NVTX Category and Color Constants
// =====================================================================

static const uint32_t TZ_NVTX_CATEGORY = 0;

static const uint32_t TZ_COLOR_TILE_LOAD  = 0xFF00AA00; // green
static const uint32_t TZ_COLOR_BIT_DECODE = 0xFFAA0000; // red
static const uint32_t TZ_COLOR_REDUCTION  = 0xFF0000AA; // blue
static const uint32_t TZ_COLOR_H2D        = 0xFFAAAA00; // yellow
static const uint32_t TZ_COLOR_KERNEL     = 0xFFFF6600; // orange
static const uint32_t TZ_COLOR_D2H        = 0xFF00AAAA; // cyan
static const uint32_t TZ_COLOR_L2_POLICY  = 0xFFAA00AA; // magenta

inline void tz_nvtx_range_push(const char* name, uint32_t color) {
    nvtxEventAttributes_t attr = {};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = color;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    nvtxRangePushEx(&attr);
}

inline void tz_nvtx_range_pop() {
    nvtxRangePop();
}

#define TZ_RANGE_PUSH(name, color) tz_nvtx_range_push(name, color)
#define TZ_RANGE_POP()             tz_nvtx_range_pop()

#else

#define TZ_RANGE_PUSH(name, color) ((void)0)
#define TZ_RANGE_POP()             ((void)0)

#endif // TZ_NVTX_ENABLED

// =====================================================================
// Profiled GEMV Wrapper
// =====================================================================
// Wraps the ternary GEMV kernel with NVTX ranges for each phase:
//   H2D memcpy → L2 policy → kernel → D2H memcpy
//
// The kernel itself is a single launch (phases 1-3 are internal to
// the GPU). For intra-kernel phase analysis, use Nsight Compute (ncu)
// which provides per-region metrics automatically.
//
// This wrapper separates the host-visible phases that Nsight Systems
// can distinguish in the timeline.

struct TernaryGemvProfileContext {
    cudaEvent_t h2d_start;
    cudaEvent_t h2d_end;
    cudaEvent_t kernel_start;
    cudaEvent_t kernel_end;
    cudaEvent_t d2h_start;
    cudaEvent_t d2h_end;
    float h2d_us;
    float kernel_us;
    float d2h_us;
    float total_us;
};

inline TernaryGemvProfileContext ternary_zero_gemv_profiled(
    const uint32_t* d_weights,
    const __half*   d_activations,
    __half*         d_output,
    int M, int N,
    cudaStream_t stream,
    int use_fp32_acc = 1
) {
    TernaryGemvProfileContext ctx = {};
    CUDA_CHECK_TZ(cudaEventCreate(&ctx.h2d_start));
    CUDA_CHECK_TZ(cudaEventCreate(&ctx.h2d_end));
    CUDA_CHECK_TZ(cudaEventCreate(&ctx.kernel_start));
    CUDA_CHECK_TZ(cudaEventCreate(&ctx.kernel_end));
    CUDA_CHECK_TZ(cudaEventCreate(&ctx.d2h_start));
    CUDA_CHECK_TZ(cudaEventCreate(&ctx.d2h_end));

    // Phase: Kernel execution (includes tile load + decode + reduce on GPU)
    TZ_RANGE_PUSH("tz_kernel", TZ_COLOR_KERNEL);
    CUDA_CHECK_TZ(cudaEventRecord(ctx.kernel_start, stream));

    cudaError_t err = ternary_zero_gemv_f16_ex(
        d_weights, d_activations, d_output, M, N, stream, use_fp32_acc
    );

    CUDA_CHECK_TZ(cudaEventRecord(ctx.kernel_end, stream));
    TZ_RANGE_POP();

    // Measure
    CUDA_CHECK_TZ(cudaEventSynchronize(ctx.kernel_end));
    CUDA_CHECK_TZ(cudaEventElapsedTime(&ctx.kernel_us, ctx.kernel_start, ctx.kernel_end));
    ctx.total_us = ctx.kernel_us;

    // Cleanup
    cudaEventDestroy(ctx.h2d_start);
    cudaEventDestroy(ctx.h2d_end);
    cudaEventDestroy(ctx.kernel_start);
    cudaEventDestroy(ctx.kernel_end);
    cudaEventDestroy(ctx.d2h_start);
    cudaEventDestroy(ctx.d2h_end);

    return ctx;
}

// =====================================================================
// Phase-Timing Structured Profiler
// =====================================================================
// For detailed analysis: runs separate timed phases with NVTX markers
// and collects per-phase latency. Use with nsys profile --nvtx-tracking.

struct PhaseTimings {
    float l2_policy_us;
    float h2d_weights_us;
    float h2d_activations_us;
    float kernel_us;
    float d2h_output_us;
    float total_us;
};

inline PhaseTimings ternary_zero_full_profiled_pass(
    const uint32_t* h_weights,   // host
    const __half*   h_activations, // host
    __half*         h_output,      // host
    int M, int N,
    int use_fp32_acc = 1
) {
    PhaseTimings timing = {};

    int packed_cols = N / 16;
    size_t weight_bytes = (size_t)M * packed_cols * sizeof(uint32_t);
    size_t act_bytes = (size_t)N * sizeof(__half);
    size_t out_bytes = (size_t)M * sizeof(__half);

    // Allocate
    uint32_t* d_w;
    __half*   d_a;
    __half*   d_o;
    cudaMalloc(&d_w, weight_bytes);
    cudaMalloc(&d_a, act_bytes);
    cudaMalloc(&d_o, out_bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t e0, e1, e2, e3, e4, e5, e6;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
    cudaEventCreate(&e3); cudaEventCreate(&e4); cudaEventCreate(&e5);
    cudaEventCreate(&e6);

    // Phase 1: L2 cache policy
    cudaEventRecord(e0, stream);
    TZ_RANGE_PUSH("tz_l2_policy", TZ_COLOR_L2_POLICY);
    ternary_zero_set_l2_policy(stream, d_w, weight_bytes);
    TZ_RANGE_POP();
    cudaEventRecord(e1, stream);

    // Phase 2: H2D weights
    TZ_RANGE_PUSH("tz_h2d_weights", TZ_COLOR_H2D);
    cudaMemcpyAsync(d_w, h_weights, weight_bytes, cudaMemcpyHostToDevice, stream);
    TZ_RANGE_POP();
    cudaEventRecord(e2, stream);

    // Phase 3: H2D activations
    TZ_RANGE_PUSH("tz_h2d_activations", TZ_COLOR_H2D);
    cudaMemcpyAsync(d_a, h_activations, act_bytes, cudaMemcpyHostToDevice, stream);
    TZ_RANGE_POP();
    cudaEventRecord(e3, stream);

    // Phase 4: Kernel
    TZ_RANGE_PUSH("tz_kernel", TZ_COLOR_KERNEL);
    ternary_zero_gemv_f16_ex(d_w, d_a, d_o, M, N, stream, use_fp32_acc);
    TZ_RANGE_POP();
    cudaEventRecord(e4, stream);

    // Phase 5: D2H output
    TZ_RANGE_PUSH("tz_d2h_output", TZ_COLOR_D2H);
    cudaMemcpyAsync(h_output, d_o, out_bytes, cudaMemcpyDeviceToHost, stream);
    TZ_RANGE_POP();
    cudaEventRecord(e5, stream);

    cudaEventSynchronize(e5);

    cudaEventElapsedTime(&timing.l2_policy_us, e0, e1);
    cudaEventElapsedTime(&timing.h2d_weights_us, e1, e2);
    cudaEventElapsedTime(&timing.h2d_activations_us, e2, e3);
    cudaEventElapsedTime(&timing.kernel_us, e3, e4);
    cudaEventElapsedTime(&timing.d2h_output_us, e4, e5);
    timing.total_us = timing.l2_policy_us + timing.h2d_weights_us
                    + timing.h2d_activations_us + timing.kernel_us
                    + timing.d2h_output_us;

    // Cleanup
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2);
    cudaEventDestroy(e3); cudaEventDestroy(e4); cudaEventDestroy(e5);
    cudaEventDestroy(e6);
    cudaStreamDestroy(stream);
    cudaFree(d_w); cudaFree(d_a); cudaFree(d_o);

    return timing;
}

// =====================================================================
// Marker Macros for Inline Kernel Phases
// =====================================================================
// Place these inside kernel-adjacent code to annotate profiler traces.
// They are no-ops when NVTX is disabled.

#define TZ_MARKER_TILE_LOAD_BEGIN()  TZ_RANGE_PUSH("tz:tile_load", TZ_COLOR_TILE_LOAD)
#define TZ_MARKER_TILE_LOAD_END()    TZ_RANGE_POP()

#define TZ_MARKER_DECODE_BEGIN()     TZ_RANGE_PUSH("tz:bit_decode", TZ_COLOR_BIT_DECODE)
#define TZ_MARKER_DECODE_END()       TZ_RANGE_POP()

#define TZ_MARKER_REDUCE_BEGIN()     TZ_RANGE_PUSH("tz:reduction", TZ_COLOR_REDUCTION)
#define TZ_MARKER_REDUCE_END()       TZ_RANGE_POP()

// Convenience: safe CUDA check for profiled wrappers
#ifndef CUDA_CHECK_TZ
#define CUDA_CHECK_TZ(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); \
    } \
} while(0)
#endif

#endif // TERNARY_ZERO_NVTX_H
