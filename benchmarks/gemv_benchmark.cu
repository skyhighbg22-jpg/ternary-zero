// =====================================================================
// Ternary-Zero GEMV Benchmark: Custom Kernel vs cuBLAS FP16
// =====================================================================
// Standalone CUDA benchmark for rigorous comparative analysis.
// Compares the custom W2A16 ternary GEMV kernel against cuBLAS GemmEx.
//
// Build:
//   nvcc -O3 --use_fast_math -std=c++17 --gpu-architecture=sm_89
//        -I../kernel -o gemv_benchmark.exe gemv_benchmark.cu
//        -lcublas -lcudart_static
// =====================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <functional>
#include <nvtx3/nvToolsExt.h>

#include "ternary_zero.h"

// =====================================================================
// Error Checking Macros
// =====================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t stat = (call); \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", \
                __FILE__, __LINE__, (int)stat); \
        exit(1); \
    } \
} while(0)

// =====================================================================
// Configuration
// =====================================================================

struct BenchConfig {
    int M;
    int N;
    int warmup_iters;
    int bench_iters;
};

// =====================================================================
// Ternary Weight Generation and Packing
// =====================================================================

static void generate_ternary_weights(std::vector<int8_t>& weights, int M, int N,
                                     float sparsity = 0.33f) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    weights.resize(M * N);
    for (auto& w : weights) {
        float r = dist(rng);
        if (r < sparsity) {
            w = 0;
        } else if (r < sparsity + (1.0f - sparsity) / 2.0f) {
            w = 1;
        } else {
            w = -1;
        }
    }
}

static std::vector<uint32_t> pack_ternary(const std::vector<int8_t>& weights,
                                          int M, int N) {
    int packed_cols = N / 16;
    std::vector<uint32_t> packed(M * packed_cols, 0);

    for (int row = 0; row < M; row++) {
        for (int pc = 0; pc < packed_cols; pc++) {
            uint32_t word = 0;
            for (int w = 0; w < 16; w++) {
                int8_t val = weights[row * N + pc * 16 + w];
                uint32_t bits = (val == 0) ? 0u : (val == 1) ? 1u : 2u;
                word |= bits << (w * 2);
            }
            packed[row * packed_cols + pc] = word;
        }
    }
    return packed;
}

static std::vector<__half> ternary_to_fp16(const std::vector<int8_t>& weights) {
    std::vector<__half> fp16(weights.size());
    for (size_t i = 0; i < weights.size(); i++) {
        fp16[i] = __float2half((float)weights[i]);
    }
    return fp16;
}

// =====================================================================
// Timing Utilities
// =====================================================================

struct TimingResult {
    float avg_ms;
    float min_ms;
    float max_ms;
    float std_ms;
};

static TimingResult measure_kernel(cudaStream_t stream, int warmup, int iters,
                                   std::function<void()> launch_fn) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; i++) {
        launch_fn();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> times(iters);
    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        launch_fn();
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float sum = 0, min_v = times[0], max_v = times[0];
    for (float t : times) {
        sum += t;
        min_v = std::min(min_v, t);
        max_v = std::max(max_v, t);
    }
    float avg = sum / iters;

    float sq_diff_sum = 0;
    for (float t : times) {
        sq_diff_sum += (t - avg) * (t - avg);
    }

    return {avg, min_v, max_v, std::sqrt(sq_diff_sum / iters)};
}

// =====================================================================
// Correctness Verification
// =====================================================================

static bool verify_results(const std::vector<__half>& custom_out,
                           const std::vector<__half>& cublas_out,
                           int M, float rtol = 5e-2f, float atol = 5e-2f) {
    int mismatches = 0;
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;

    for (int i = 0; i < M; i++) {
        float cv = __half2float(custom_out[i]);
        float bv = __half2float(cublas_out[i]);
        float abs_err = std::fabs(cv - bv);
        float denom = std::fabs(bv) + 1e-8f;
        float rel_err = abs_err / denom;

        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);

        if (abs_err > atol && rel_err > rtol) {
            mismatches++;
        }
    }

    printf("  Correctness: %s (%d/%d match, max_abs_err=%.6f, max_rel_err=%.6f)\n",
           mismatches == 0 ? "PASS" : "FAIL",
           M - mismatches, M, max_abs_err, max_rel_err);

    return mismatches == 0;
}

// =====================================================================
// Benchmark: Custom Ternary GEMV
// =====================================================================

struct CustomBenchResult {
    TimingResult timing;
    float bandwidth_GBps;
    float compute_GFLOPS;
};

static CustomBenchResult bench_custom_kernel(
    const uint32_t* d_weights,
    const __half* d_activations,
    __half* d_output,
    int M, int N,
    cudaStream_t stream,
    int warmup, int iters
) {
    int packed_cols = N / 16;
    size_t weight_bytes = (size_t)M * packed_cols * sizeof(uint32_t);
    size_t act_bytes = (size_t)N * sizeof(__half);
    size_t out_bytes = (size_t)M * sizeof(__half);

    auto timing = measure_kernel(stream, warmup, iters, [&]() {
        nvtxRangePushA("ternary_gemv");
        ternary_zero_gemv_f16(d_weights, d_activations, d_output, M, N, stream);
        nvtxRangePop();
    });

    size_t total_bytes = weight_bytes + act_bytes + out_bytes;
    float bandwidth = (float)total_bytes / (timing.avg_ms * 1e-3f) / 1e9f;

    float flops = 2.0f * M * N * 0.67f;
    float gflops = flops / (timing.avg_ms * 1e-3f) / 1e9f;

    return {timing, bandwidth, gflops};
}

// =====================================================================
// Benchmark: cuBLAS FP16 GEMV (via GemmEx with N=1)
// =====================================================================

struct CuBLASBenchResult {
    TimingResult timing;
    float bandwidth_GBps;
    float compute_GFLOPS;
};

static CuBLASBenchResult bench_cublas_hgemv(
    cublasHandle_t handle,
    const __half* d_weights_fp16,
    const __half* d_activations,
    __half* d_output,
    int M, int N,
    cudaStream_t stream,
    int warmup, int iters
) {
    size_t weight_bytes = (size_t)M * N * sizeof(__half);
    size_t act_bytes = (size_t)N * sizeof(__half);
    size_t out_bytes = (size_t)M * sizeof(__half);

    __half alpha = __float2half(1.0f);
    __half beta  = __float2half(0.0f);

    // cuBLAS GemmEx: y = op(A) * x
    // A is [N x M] col-major (row-major M x N), op(A) = A^T = [M x N]
    // x is [N x 1], y is [M x 1]
    // Result: y = A^T * x = W * x (row-major GEMV)
    auto timing = measure_kernel(stream, warmup, iters, [&]() {
        nvtxRangePushA("cublas_gemv");
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_T,           // transpose A (row-major -> col-major)
            CUBLAS_OP_N,           // no transpose on x
            M, 1, N,               // output_dim, 1, input_dim
            &alpha,
            d_weights_fp16, CUDA_R_16F, N,  // A, type, lda
            d_activations, CUDA_R_16F, N,   // B, type, ldb
            &beta,
            d_output, CUDA_R_16F, M,        // C, type, ldc
            CUBLAS_COMPUTE_16F,             // compute in FP16
            CUBLAS_GEMM_DEFAULT             // auto algorithm
        ));
        nvtxRangePop();
    });

    size_t total_bytes = weight_bytes + act_bytes + out_bytes;
    float bandwidth = (float)total_bytes / (timing.avg_ms * 1e-3f) / 1e9f;

    float flops = 2.0f * M * N;
    float gflops = flops / (timing.avg_ms * 1e-3f) / 1e9f;

    return {timing, bandwidth, gflops};
}

// =====================================================================
// GPU Info
// =====================================================================

static void print_gpu_info() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int mem_clock_khz = 0;
    int bus_width = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, 0));

    printf("=== GPU Configuration ===\n");
    printf("  Device:           %s\n", prop.name);
    printf("  SM Count:         %d\n", prop.multiProcessorCount);
    printf("  Max Threads/SM:   %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  L2 Cache Size:    %d KB\n", prop.l2CacheSize / 1024);
    printf("  Shared Mem/Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Registers/Block:  %d\n", prop.regsPerBlock);
    printf("  Memory Clock:     %d MHz\n", mem_clock_khz / 1000);
    printf("  Memory Bus Width: %d bits\n", bus_width);
    printf("  Peak Mem BW:      %.1f GB/s\n",
           2.0 * mem_clock_khz * 1e3 * (bus_width / 8) / 1e9);
    printf("  Compute Cap:      %d.%d\n", prop.major, prop.minor);
    printf("\n");
}

// =====================================================================
// Main Benchmark Suite
// =====================================================================

int main(int argc, char** argv) {
    printf("========================================================\n");
    printf("  Ternary-Zero GEMV Benchmark: Custom vs cuBLAS FP16\n");
    printf("========================================================\n\n");

    print_gpu_info();

    int sm_clock = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_clock, cudaDevAttrClockRate, 0));
    printf("=== Clock Stability ===\n");
    printf("  Current SM clock: %d MHz\n", sm_clock / 1000);
    printf("  For fixed clocks, run as admin:\n");
    printf("    nvidia-smi -lgc %d,%d\n", sm_clock/1000, sm_clock/1000);
    printf("\n");

    std::vector<BenchConfig> configs = {
        {1,    768,  100,   1000},
        {1,    1024, 100,   1000},
        {1,    2048, 100,   1000},
        {1,    4096, 100,   1000},
        {16,   768,  50,    500},
        {16,   1024, 50,    500},
        {16,   2048, 50,    500},
        {16,   4096, 50,    500},
        {64,   1024, 20,    200},
        {64,   4096, 20,    200},
        {256,  1024, 10,    100},
        {256,  4096, 10,    100},
    };

    if (argc >= 3) {
        int M = atoi(argv[1]);
        int N = atoi(argv[2]);
        int warmup = argc > 3 ? atoi(argv[3]) : 100;
        int iters  = argc > 4 ? atoi(argv[4]) : 1000;
        configs = {{M, N, warmup, iters}};
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    printf("=== Benchmark Results ===\n\n");
    printf("%-6s %-6s | %10s %10s %10s | %10s %10s %10s | %8s %8s %8s\n",
           "M", "N",
           "Custom(ms)", "Cust-std", "Cust-min",
           "cuBLAS(ms)", "cuBL-std", "cuBL-min",
           "Speedup", "Cust-BW", "cuBL-BW");
    printf("%s\n", std::string(105, '-').c_str());

    float total_custom_ms = 0, total_cublas_ms = 0;
    int all_pass = 1;

    for (const auto& cfg : configs) {
        int M = cfg.M;
        int N = cfg.N;

        if (N % 16 != 0) {
            printf("Skipping M=%d N=%d (N must be multiple of 16)\n", M, N);
            continue;
        }

        std::vector<int8_t> ternary_weights;
        generate_ternary_weights(ternary_weights, M, N, 0.33f);

        auto packed = pack_ternary(ternary_weights, M, N);
        auto fp16_weights = ternary_to_fp16(ternary_weights);

        std::mt19937 rng(123);
        std::normal_distribution<float> act_dist(0.0f, 1.0f);
        std::vector<__half> activations(N);
        for (auto& a : activations) {
            a = __float2half(act_dist(rng));
        }

        uint32_t* d_packed;
        __half*   d_fp16_weights;
        __half*   d_activations;
        __half*   d_output_custom;
        __half*   d_output_cublas;

        int packed_cols = N / 16;
        CUDA_CHECK(cudaMalloc(&d_packed, (size_t)M * packed_cols * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_fp16_weights, (size_t)M * N * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_activations, (size_t)N * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_output_custom, (size_t)M * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_output_cublas, (size_t)M * sizeof(__half)));

        CUDA_CHECK(cudaMemcpy(d_packed, packed.data(),
                              packed.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_fp16_weights, fp16_weights.data(),
                              fp16_weights.size() * sizeof(__half),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_activations, activations.data(),
                              N * sizeof(__half),
                              cudaMemcpyHostToDevice));

        ternary_zero_set_l2_policy(stream, d_packed,
                                   (size_t)M * packed_cols * sizeof(uint32_t));

        auto custom_result = bench_custom_kernel(
            d_packed, d_activations, d_output_custom,
            M, N, stream, cfg.warmup_iters, cfg.bench_iters
        );

        auto cublas_result = bench_cublas_hgemv(
            handle, d_fp16_weights, d_activations, d_output_cublas,
            M, N, stream, cfg.warmup_iters, cfg.bench_iters
        );

        std::vector<__half> h_custom(M), h_cublas(M);
        CUDA_CHECK(cudaMemcpy(h_custom.data(), d_output_custom,
                              M * sizeof(__half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_cublas.data(), d_output_cublas,
                              M * sizeof(__half), cudaMemcpyDeviceToHost));

        bool pass = verify_results(h_custom, h_cublas, M);
        if (!pass) all_pass = 0;

        float speedup = cublas_result.timing.avg_ms / custom_result.timing.avg_ms;

        printf("%-6d %-6d | %10.4f %10.4f %10.4f | %10.4f %10.4f %10.4f | %7.2fx %7.1f %7.1f\n",
               M, N,
               custom_result.timing.avg_ms,
               custom_result.timing.std_ms,
               custom_result.timing.min_ms,
               cublas_result.timing.avg_ms,
               cublas_result.timing.std_ms,
               cublas_result.timing.min_ms,
               speedup,
               custom_result.bandwidth_GBps,
               cublas_result.bandwidth_GBps);

        total_custom_ms += custom_result.timing.avg_ms;
        total_cublas_ms += cublas_result.timing.avg_ms;

        CUDA_CHECK(cudaFree(d_packed));
        CUDA_CHECK(cudaFree(d_fp16_weights));
        CUDA_CHECK(cudaFree(d_activations));
        CUDA_CHECK(cudaFree(d_output_custom));
        CUDA_CHECK(cudaFree(d_output_cublas));
    }

    printf("\n=== Summary ===\n");
    printf("  Total custom time:  %.2f ms\n", total_custom_ms);
    printf("  Total cuBLAS time:  %.2f ms\n", total_cublas_ms);
    if (total_custom_ms > 0) {
        printf("  Overall speedup:    %.2fx\n", total_cublas_ms / total_custom_ms);
    }
    printf("  Correctness:        %s\n", all_pass ? "ALL PASS" : "SOME FAILURES");

    printf("\n=== Key Observations ===\n");
    printf("  - Custom kernel uses 2-bit packed weights (16x memory reduction)\n");
    printf("  - cuBLAS uses standard FP16 weights (full 16-bit per weight)\n");
    printf("  - For M=1 (single-token decode), only 1 block launched\n");

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());

    return all_pass ? 0 : 1;
}