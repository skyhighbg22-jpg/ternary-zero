// Copyright (C) 2025 ternary-zero contributors
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// =====================================================================
// L2 Cache Persistence Manager for RTX 4060 (Ada Lovelace, sm_89)
// =====================================================================
//
// RTX 4060 specifications:
//   L2 Cache:     32 MB (33,554,432 bytes)
//   Memory Bus:   128-bit GDDR6
//   Peak BW:      ~272 GB/s (17 Gbps × 128-bit / 8)
//   SM Count:     24
//
// Target: Pin ternary weights for Llama-2-7B FFN layer in L2 cache.
//
// FFN Layer dimensions (Llama-2-7B):
//   gate_proj:  11008 × 4096 (W2: 11008 × 256 packed u32)
//   up_proj:    11008 × 4096 (W2: 11008 × 256 packed u32)
//   down_proj:   4096 × 11008 (W2: 4096 × 688 packed u32)
//
// Memory footprint (2-bit packed, 16 weights per uint32_t):
//   gate_proj: 11008 × (4096/16) × 4 = 11008 × 256 × 4 = 11,272,192 B ≈ 10.75 MB
//   up_proj:   11008 × (4096/16) × 4 = 11008 × 256 × 4 = 11,272,192 B ≈ 10.75 MB
//   down_proj: 4096 × (11008/16) × 4 = 4096 × 688 × 4  = 11,272,192 B ≈ 10.75 MB
//
//   Total FFN: 33,816,576 B ≈ 32.25 MB (barely exceeds 32 MB L2)
//
// Strategy:
//   - gate_proj and up_proj are accessed sequentially during the FFN up-projection.
//     Combined: 21.5 MB — fits in 32 MB L2 with 10.5 MB headroom for activations.
//   - down_proj (10.75 MB) is accessed after the nonlinearity, so it can replace
//     the up-projection weights in L2.
//   - Two-phase L2 pinning: phase 1 pins gate+up, phase 2 pins down.
//   - For single-layer decode (M=1), only one GEMV is active at a time,
//     so any single projection (10.75 MB) always fits.
//
// cudaAccessPolicyWindow constraints (Ada):
//   - max num_bytes: L2 cache size (32 MB)
//   - hitRatio: fraction of L2 reserved for the window [0.0, 1.0]
//   - Window must be aligned to 32 bytes (L2 sector size)
//   - Only one active window per stream (last set wins)
//
// Build:
//   nvcc -O3 --use_fast_math --gpu-architecture=sm_89 -std=c++17
//        -c -I../kernel l2_persist.cu
// =====================================================================

#include "ternary_zero.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

// =====================================================================
// RTX 4060 L2 Cache Constants
// =====================================================================

static constexpr size_t RTX4060_L2_CACHE_BYTES    = 32 * 1024 * 1024;  // 32 MB
static constexpr size_t L2_SECTOR_SIZE             = 32;                 // 32-byte sectors
static constexpr float  DEFAULT_HIT_RATIO           = 1.0f;
static constexpr float  CONSERVED_HIT_RATIO         = 0.85f;

// =====================================================================
// Ternary Weight Layer Descriptor
// =====================================================================

struct TernaryLayer {
    const uint32_t* device_ptr;   // packed 2-bit weights on device
    int M;                        // output features (rows)
    int N;                        // input features (columns)
    size_t packed_cols;           // N / 16
    size_t byte_size;             // M * packed_cols * sizeof(uint32_t)
};

static TernaryLayer make_ternary_layer(const uint32_t* ptr, int M, int N) {
    TernaryLayer layer;
    layer.device_ptr = ptr;
    layer.M = M;
    layer.N = N;
    layer.packed_cols = N / 16;
    layer.byte_size = (size_t)M * layer.packed_cols * sizeof(uint32_t);
    return layer;
}

// Align byte size up to L2 sector boundary
static size_t align_to_l2_sector(size_t bytes) {
    return (bytes + L2_SECTOR_SIZE - 1) & ~(L2_SECTOR_SIZE - 1);
}

// =====================================================================
// L2 Persistence Policy Application
// =====================================================================

static cudaError_t apply_l2_policy(
    cudaStream_t stream,
    const void* base_ptr,
    size_t num_bytes,
    float hit_ratio
) {
    cudaStreamAttrValue attr = {};
    attr.accessPolicyWindow.base_ptr  = const_cast<void*>(base_ptr);
    attr.accessPolicyWindow.num_bytes = num_bytes;
    attr.accessPolicyWindow.hitRatio  = hit_ratio;
    attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;

    return cudaStreamSetAttribute(
        stream,
        cudaStreamAttributeAccessPolicyWindow,
        &attr
    );
}

// Clear L2 persistence policy (revert to default)
static cudaError_t clear_l2_policy(cudaStream_t stream) {
    cudaStreamAttrValue attr = {};
    attr.accessPolicyWindow.base_ptr  = nullptr;
    attr.accessPolicyWindow.num_bytes = 0;
    attr.accessPolicyWindow.hitRatio  = 0.0f;
    attr.accessPolicyWindow.hitProp   = cudaAccessPropertyNormal;
    attr.accessPolicyWindow.missProp  = cudaAccessPropertyNormal;

    return cudaStreamSetAttribute(
        stream,
        cudaStreamAttributeAccessPolicyWindow,
        &attr
    );
}

// =====================================================================
// L2 Fit Analysis
// =====================================================================

enum class L2FitResult {
    FITS_ALONE,          // layer fits in L2 by itself
    FITS_WITH_HEADROOM,  // fits with room for activations
    DOES_NOT_FIT,        // exceeds L2 capacity
};

struct L2Analysis {
    L2FitResult fit;
    size_t layer_bytes;
    size_t l2_bytes;
    float utilization_pct;
    float recommended_hit_ratio;
    int recommended_tile_count;
};

static L2Analysis analyze_l2_fit(const TernaryLayer& layer) {
    L2Analysis analysis = {};
    analysis.layer_bytes = align_to_l2_sector(layer.byte_size);
    analysis.l2_bytes = RTX4060_L2_CACHE_BYTES;
    analysis.utilization_pct = (float)analysis.layer_bytes / (float)analysis.l2_bytes * 100.0f;

    if (analysis.layer_bytes <= RTX4060_L2_CACHE_BYTES * 0.6f) {
        analysis.fit = L2FitResult::FITS_WITH_HEADROOM;
        analysis.recommended_hit_ratio = DEFAULT_HIT_RATIO;
        analysis.recommended_tile_count = 1;
    } else if (analysis.layer_bytes <= RTX4060_L2_CACHE_BYTES) {
        analysis.fit = L2FitResult::FITS_ALONE;
        analysis.recommended_hit_ratio = CONSERVED_HIT_RATIO;
        analysis.recommended_tile_count = 1;
    } else {
        analysis.fit = L2FitResult::DOES_NOT_FIT;
        analysis.recommended_hit_ratio = DEFAULT_HIT_RATIO;
        // Tile count: ceil(layer_bytes / (L2 * 0.6))
        size_t tile_budget = (size_t)(RTX4060_L2_CACHE_BYTES * 0.6);
        analysis.recommended_tile_count = (int)((analysis.layer_bytes + tile_budget - 1) / tile_budget);
    }

    return analysis;
}

// =====================================================================
// Single-Layer L2 Pin (for M=1 decode, one GEMV at a time)
// =====================================================================

extern "C"
cudaError_t l2_persist_single_layer(
    cudaStream_t stream,
    const uint32_t* weights,
    int M, int N,
    float* hit_ratio_out,
    size_t* bytes_pinned_out
) {
    TernaryLayer layer = make_ternary_layer(weights, M, N);
    L2Analysis analysis = analyze_l2_fit(layer);

    if (analysis.fit == L2FitResult::DOES_NOT_FIT) {
        fprintf(stderr, "WARNING: Layer %dx%d (%zu bytes) exceeds L2 cache (%zu bytes). "
                "Consider tiling.\n", M, N, analysis.layer_bytes, RTX4060_L2_CACHE_BYTES);
    }

    cudaError_t err = apply_l2_policy(
        stream, weights, analysis.layer_bytes,
        analysis.recommended_hit_ratio
    );

    if (hit_ratio_out) *hit_ratio_out = analysis.recommended_hit_ratio;
    if (bytes_pinned_out) *bytes_pinned_out = analysis.layer_bytes;

    return err;
}

// =====================================================================
// FFN Two-Phase L2 Pin (Llama-2-7B)
// =====================================================================
//
// The FFN block executes: gate_proj(x) * up_proj(x) -> silu -> down_proj
//
// Phase 1 (stream A): Pin gate_proj + up_proj (21.5 MB < 32 MB)
// Phase 2 (stream B): Pin down_proj (10.75 MB)
//
// For M=1 decode, a single layer at a time always fits, so we use
// a simpler single-phase approach and switch the policy between
// forward sub-steps.

struct FFNLayers {
    TernaryLayer gate_proj;  // [intermediate_size × hidden_size]
    TernaryLayer up_proj;    // [intermediate_size × hidden_size]
    TernaryLayer down_proj;  // [hidden_size × intermediate_size]
};

extern "C"
cudaError_t l2_persist_ffn_phase1_gate_up(
    cudaStream_t stream,
    const uint32_t* gate_weights,
    const uint32_t* up_weights,
    int intermediate_size,
    int hidden_size
) {
    TernaryLayer gate = make_ternary_layer(gate_weights, intermediate_size, hidden_size);
    TernaryLayer up   = make_ternary_layer(up_weights, intermediate_size, hidden_size);

    size_t combined = align_to_l2_sector(gate.byte_size + up.byte_size);

    if (combined <= RTX4060_L2_CACHE_BYTES) {
        // Both fit: pin gate_proj first (contiguous window), up_proj is
        // in a separate allocation so we pin the larger one.
        // cudaAccessPolicyWindow only supports one contiguous range,
        // so we pin each individually with a shared hit budget.
        float per_layer_ratio = (float)combined / (float)RTX4060_L2_CACHE_BYTES;

        cudaError_t err = apply_l2_policy(
            stream, gate.device_ptr, align_to_l2_sector(gate.byte_size),
            per_layer_ratio > 0.8f ? CONSERVED_HIT_RATIO : DEFAULT_HIT_RATIO
        );
        if (err != cudaSuccess) return err;

        // Note: Only one window is active per stream. For two allocations,
        // we'd need two streams or accept that one window covers the primary.
        // In practice, pin the gate_proj (first access) and rely on L2
        // temporal locality for up_proj.
    }

    return cudaSuccess;
}

extern "C"
cudaError_t l2_persist_ffn_phase2_down(
    cudaStream_t stream,
    const uint32_t* down_weights,
    int hidden_size,
    int intermediate_size
) {
    TernaryLayer down = make_ternary_layer(down_weights, hidden_size, intermediate_size);
    return apply_l2_policy(
        stream, down.device_ptr, align_to_l2_sector(down.byte_size),
        DEFAULT_HIT_RATIO
    );
}

// =====================================================================
// Tiled L2 Pinning for Large Layers
// =====================================================================
// For layers exceeding L2 capacity, split into row tiles and pin
// each tile before its GEMV launch. Requires tiled kernel invocation.

struct L2TilePlan {
    int tile_count;
    int rows_per_tile;
    int last_tile_rows;
    size_t tile_bytes;
    size_t total_bytes;
};

static L2TilePlan compute_tile_plan(const TernaryLayer& layer, float budget_fraction = 0.6f) {
    L2TilePlan plan = {};
    plan.total_bytes = align_to_l2_sector(layer.byte_size);

    size_t budget = (size_t)(RTX4060_L2_CACHE_BYTES * budget_fraction);
    size_t bytes_per_row = layer.packed_cols * sizeof(uint32_t);

    plan.rows_per_tile = (int)(budget / bytes_per_row);
    if (plan.rows_per_tile <= 0) plan.rows_per_tile = 1;
    if (plan.rows_per_tile > layer.M) plan.rows_per_tile = layer.M;

    plan.tile_count = (layer.M + plan.rows_per_tile - 1) / plan.rows_per_tile;
    plan.last_tile_rows = layer.M - (plan.tile_count - 1) * plan.rows_per_tile;
    plan.tile_bytes = align_to_l2_sector((size_t)plan.rows_per_tile * bytes_per_row);

    return plan;
}

extern "C"
cudaError_t l2_persist_tiled(
    cudaStream_t stream,
    const uint32_t* weights,
    int M, int N,
    int tile_index,       // which tile to pin (0-based)
    int* tile_count_out,  // [out] total tiles
    int* tile_rows_out    // [out] rows in this tile
) {
    TernaryLayer layer = make_ternary_layer(weights, M, N);
    L2TilePlan plan = compute_tile_plan(layer);

    if (tile_index < 0 || tile_index >= plan.tile_count) {
        return cudaErrorInvalidValue;
    }

    int rows = (tile_index < plan.tile_count - 1)
        ? plan.rows_per_tile
        : (int)plan.last_tile_rows;

    size_t offset = (size_t)tile_index * plan.rows_per_tile * layer.packed_cols * sizeof(uint32_t);
    size_t tile_bytes = align_to_l2_sector((size_t)rows * layer.packed_cols * sizeof(uint32_t));

    if (tile_count_out) *tile_count_out = plan.tile_count;
    if (tile_rows_out) *tile_rows_out = rows;

    return apply_l2_policy(
        stream,
        (const uint8_t*)weights + offset,
        tile_bytes,
        DEFAULT_HIT_RATIO
    );
}

// =====================================================================
// Diagnostic: Print L2 Analysis
// =====================================================================

extern "C"
void l2_print_analysis(int M, int N, const char* label) {
    TernaryLayer dummy = {};
    dummy.M = M;
    dummy.N = N;
    dummy.packed_cols = N / 16;
    dummy.byte_size = (size_t)M * dummy.packed_cols * sizeof(uint32_t);

    L2Analysis analysis = analyze_l2_fit(dummy);
    L2TilePlan plan = compute_tile_plan(dummy);

    printf("=== L2 Cache Analysis: %s (%dx%d) ===\n", label ? label : "layer", M, N);
    printf("  Weight bytes:       %zu (%.2f MB)\n",
           dummy.byte_size, (float)dummy.byte_size / (1024.0f * 1024.0f));
    printf("  Aligned bytes:      %zu (%.2f MB)\n",
           analysis.layer_bytes, (float)analysis.layer_bytes / (1024.0f * 1024.0f));
    printf("  L2 capacity:        %zu (%.2f MB)\n",
           analysis.l2_bytes, (float)analysis.l2_bytes / (1024.0f * 1024.0f));
    printf("  L2 utilization:     %.1f%%\n", analysis.utilization_pct);
    printf("  Fit result:         %s\n",
           analysis.fit == L2FitResult::FITS_WITH_HEADROOM ? "FITS (with headroom)" :
           analysis.fit == L2FitResult::FITS_ALONE ? "FITS (tight)" : "DOES NOT FIT");
    printf("  Recommended ratio:  %.2f\n", analysis.recommended_hit_ratio);
    printf("  Tile count:         %d\n", plan.tile_count);
    printf("  Rows per tile:      %d\n", plan.rows_per_tile);
    printf("  Tile bytes:         %zu (%.2f MB)\n",
           plan.tile_bytes, (float)plan.tile_bytes / (1024.0f * 1024.0f));
    printf("\n");
}

// =====================================================================
// Standalone: Print L2 analysis for Llama-2-7B FFN layers
// =====================================================================

#ifdef L2_PERSIST_MAIN

int main() {
    printf("========================================================\n");
    printf("  L2 Cache Persistence Analysis: RTX 4060 (32 MB L2)\n");
    printf("========================================================\n\n");

    // Llama-2-7B FFN
    l2_print_analysis(11008, 4096, "Llama-7B gate_proj (11008x4096)");
    l2_print_analysis(11008, 4096, "Llama-7B up_proj (11008x4096)");
    l2_print_analysis(4096, 11008, "Llama-7B down_proj (4096x11008)");

    // Combined gate+up
    printf("=== Combined: gate_proj + up_proj ===\n");
    size_t combined = 2 * (size_t)11008 * (4096 / 16) * sizeof(uint32_t);
    printf("  Combined bytes: %zu (%.2f MB)\n", combined, (float)combined / (1024.0f * 1024.0f));
    printf("  L2 utilization: %.1f%%\n", (float)combined / (32.0f * 1024.0f * 1024.0f) * 100.0f);
    printf("  Fits in L2: %s\n\n", combined <= 32 * 1024 * 1024 ? "YES" : "NO");

    // Llama-3.2-1B FFN
    l2_print_analysis(8192, 2048, "Llama-1B gate_proj (8192x2048)");
    l2_print_analysis(2048, 8192, "Llama-1B down_proj (2048x8192)");

    // Llama-3-8B FFN
    l2_print_analysis(14336, 4096, "Llama-8B gate_proj (14336x4096)");

    return 0;
}

#endif // L2_PERSIST_MAIN
