// =====================================================================
// Ternary-Zero Baseline Comparison Harness
// =====================================================================
// M=1 (batch-1 decode) GEMV: Ternary-Zero W2A16 vs cuBLAS FP16 vs
// simulated GGUF/AutoGPTQ INT4 dequant GEMV.
//
// Measures: latency floor (us), memory bandwidth utilization (GB/s),
// and bytes transferred per kernel invocation.
//
// Build:
//   nvcc -O3 --use_fast_math -std=c++17 --gpu-architecture=sm_89
//        -I../kernel -o baseline_comparison.exe baseline_comparison.cu
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
// Error Checking
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
    const char* label;
};

struct TimingResult {
    float avg_us;
    float min_us;
    float max_us;
    float std_us;
    float p50_us;
    float p95_us;
    float p99_us;
};

// =====================================================================
// GPU Info
// =====================================================================

struct GpuInfo {
    char name[256];
    int sm_count;
    int l2_size_kb;
    int mem_clock_mhz;
    int bus_width_bits;
    float peak_bw_gbps;
    float compute_cap_major;
    float compute_cap_minor;
};

static GpuInfo get_gpu_info() {
    GpuInfo info = {};
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    strncpy(info.name, prop.name, 255);
    info.sm_count = prop.multiProcessorCount;
    info.l2_size_kb = prop.l2CacheSize / 1024;
    info.compute_cap_major = prop.major;
    info.compute_cap_minor = prop.minor;

    int mem_clock_khz = 0, bus_width = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, 0));
    info.mem_clock_mhz = mem_clock_khz / 1000;
    info.bus_width_bits = bus_width;
    info.peak_bw_gbps = 2.0f * mem_clock_khz * 1e3f * (bus_width / 8) / 1e9f;

    return info;
}

// =====================================================================
// Ternary Weight Generation
// =====================================================================

static void generate_ternary_weights(std::vector<int8_t>& weights, int M, int N,
                                     float sparsity = 0.33f) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    weights.resize(M * N);
    for (auto& w : weights) {
        float r = dist(rng);
        if (r < sparsity) w = 0;
        else if (r < sparsity + (1.0f - sparsity) / 2.0f) w = 1;
        else w = -1;
    }
}

static std::vector<uint32_t> pack_ternary(const std::vector<int8_t>& weights, int M, int N) {
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
    for (size_t i = 0; i < weights.size(); i++)
        fp16[i] = __float2half((float)weights[i]);
    return fp16;
}

// Generate 4-bit quantized weights (simulating GGUF Q4_0 / AutoGPTQ W4A16)
static std::vector<uint8_t> generate_int4_weights(int M, int N) {
    int packed_per_byte = 2;
    int packed_bytes = (M * N) / packed_per_byte;
    std::vector<uint8_t> packed(packed_bytes);
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(0, 15);
    for (auto& b : packed) {
        uint8_t lo = (uint8_t)dist(rng);
        uint8_t hi = (uint8_t)dist(rng);
        b = lo | (hi << 4);
    }
    return packed;
}

// =====================================================================
// Timing
// =====================================================================

static TimingResult measure_kernel(cudaStream_t stream, int warmup, int iters,
                                   std::function<void()> launch_fn) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; i++) launch_fn();
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

    std::sort(times.begin(), times.end());
    float sum = 0;
    for (float t : times) sum += t;
    float avg = sum / iters;
    float sq = 0;
    for (float t : times) sq += (t - avg) * (t - avg);

    return {
        avg * 1000.0f,           // avg_us
        times.front() * 1000.0f, // min_us
        times.back() * 1000.0f,  // max_us
        sqrtf(sq / iters) * 1000.0f, // std_us
        times[iters / 2] * 1000.0f,  // p50_us
        times[(int)(iters * 0.95)] * 1000.0f, // p95_us
        times[(int)(iters * 0.99)] * 1000.0f, // p99_us
    };
}

// =====================================================================
// Benchmark: Ternary-Zero W2A16 GEMV
// =====================================================================

struct KernelResult {
    TimingResult timing;
    float bandwidth_GBps;
    float peak_bw_pct;
    size_t bytes_transferred;
};

static KernelResult bench_ternary_zero(
    const uint32_t* d_weights, const __half* d_act, __half* d_out,
    int M, int N, cudaStream_t stream, int warmup, int iters
) {
    int packed_cols = N / 16;
    size_t weight_bytes = (size_t)M * packed_cols * sizeof(uint32_t);
    size_t act_bytes = (size_t)N * sizeof(__half);
    size_t out_bytes = (size_t)M * sizeof(__half);
    size_t total_bytes = weight_bytes + act_bytes + out_bytes;

    auto timing = measure_kernel(stream, warmup, iters, [&]() {
        nvtxRangePushA("ternary_zero_gemv");
        ternary_zero_gemv_f16(d_weights, d_act, d_out, M, N, stream);
        nvtxRangePop();
    });

    GpuInfo info = get_gpu_info();
    float bw = (float)total_bytes / (timing.avg_us * 1e-6f) / 1e9f;
    float peak_pct = (bw / info.peak_bw_gbps) * 100.0f;

    return {timing, bw, peak_pct, total_bytes};
}

// =====================================================================
// Benchmark: cuBLAS FP16 GEMV (GGUF/AutoGPTQ proxy for FP16 baseline)
// =====================================================================

static KernelResult bench_cublas_fp16(
    cublasHandle_t handle, const __half* d_fp16_weights,
    const __half* d_act, __half* d_out,
    int M, int N, cudaStream_t stream, int warmup, int iters
) {
    size_t weight_bytes = (size_t)M * N * sizeof(__half);
    size_t act_bytes = (size_t)N * sizeof(__half);
    size_t out_bytes = (size_t)M * sizeof(__half);
    size_t total_bytes = weight_bytes + act_bytes + out_bytes;

    __half alpha = __float2half(1.0f);
    __half beta  = __float2half(0.0f);

    auto timing = measure_kernel(stream, warmup, iters, [&]() {
        nvtxRangePushA("cublas_fp16_gemv");
        CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            M, 1, N, &alpha,
            d_fp16_weights, CUDA_R_16F, N,
            d_act, CUDA_R_16F, N,
            &beta, d_out, CUDA_R_16F, M,
            CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT
        ));
        nvtxRangePop();
    });

    GpuInfo info = get_gpu_info();
    float bw = (float)total_bytes / (timing.avg_us * 1e-6f) / 1e9f;
    float peak_pct = (bw / info.peak_bw_gbps) * 100.0f;

    return {timing, bw, peak_pct, total_bytes};
}

// =====================================================================
// Simulated INT4 Dequant GEMV Kernel
// =====================================================================
// Models the GGUF Q4_0 / AutoGPTQ W4A16 decode path:
//   1. Load packed 4-bit weights from DRAM
//   2. Dequantize to FP16 in registers
//   3. FMA with FP16 activations
//   4. Warp reduction
//
// This is a representative kernel — real GGUF/AutoGPTQ kernels use
// block-wise quantization with per-block scales, but the memory
// access pattern and arithmetic density are equivalent.

__global__ void int4_dequant_gemv_kernel(
    const uint8_t* __restrict__ int4_weights,  // 4-bit packed, 2 per byte
    const __half*  __restrict__ activations,
    __half*        __restrict__ output,
    const __half*  __restrict__ scales,         // per-block scale (FP16)
    int M, int N, int block_size  // block_size = 32 (typical Q4_0)
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= M) return;

    extern __shared__ __half s_act[];
    const int tile_size = 1024;

    float acc = 0.0f;
    const int num_blocks = N / block_size;
    const int bytes_per_block = block_size / 2;  // 4-bit = 2 per byte

    for (int tile_start = 0; tile_start < N; tile_start += tile_size) {
        // Load activations into shared memory
        for (int i = tid; i < tile_size && (tile_start + i) < N; i += blockDim.x) {
            s_act[i] = activations[tile_start + i];
        }
        __syncthreads();

        const int tile_end = min(tile_start + tile_size, N);

        for (int col = tile_start + tid; col < tile_end; col += blockDim.x) {
            int block_idx = col / block_size;
            int col_in_block = col % block_size;

            // Load packed 4-bit weight
            int byte_offset = row * num_blocks * bytes_per_block
                            + block_idx * bytes_per_block
                            + col_in_block / 2;
            uint8_t packed = int4_weights[byte_offset];
            uint8_t nibble = (col_in_block % 2 == 0) ? (packed & 0x0F) : (packed >> 4);

            // Dequantize: w = (nibble - 8) * scale
            float scale = __half2float(scales[row * num_blocks + block_idx]);
            float w = ((int)nibble - 8) * scale;

            float a = __half2float(s_act[col - tile_start]);
            acc += w * a;
        }
        __syncthreads();
    }

    // Warp reduction
    for (int offset = 16; offset >= 1; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);

    __shared__ float s_sums[8]; // 256/32 = 8 warps
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) s_sums[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = (lane_id < 8) ? s_sums[lane_id] : 0.0f;
        for (int offset = 4; offset >= 1; offset >>= 1)
            block_sum += __shfl_down_sync(0xFFFFFFFF, block_sum, offset);
        if (lane_id == 0)
            output[row] = __float2half(block_sum);
    }
}

static KernelResult bench_int4_dequant(
    const uint8_t* d_int4_weights, const __half* d_act, __half* d_out,
    const __half* d_scales,
    int M, int N, cudaStream_t stream, int warmup, int iters
) {
    int block_size = 32;
    size_t weight_bytes = (size_t)M * (N / block_size) * (block_size / 2);
    size_t scale_bytes = (size_t)M * (N / block_size) * sizeof(__half);
    size_t act_bytes = (size_t)N * sizeof(__half);
    size_t out_bytes = (size_t)M * sizeof(__half);
    size_t total_bytes = weight_bytes + scale_bytes + act_bytes + out_bytes;

    dim3 grid(M);
    dim3 block(256);
    int smem_bytes = 1024 * sizeof(__half);

    auto timing = measure_kernel(stream, warmup, iters, [&]() {
        nvtxRangePushA("int4_dequant_gemv");
        int4_dequant_gemv_kernel<<<grid, block, smem_bytes, stream>>>(
            d_int4_weights, d_act, d_out, d_scales, M, N, block_size
        );
        nvtxRangePop();
    });

    GpuInfo info = get_gpu_info();
    float bw = (float)total_bytes / (timing.avg_us * 1e-6f) / 1e9f;
    float peak_pct = (bw / info.peak_bw_gbps) * 100.0f;

    return {timing, bw, peak_pct, total_bytes};
}

// =====================================================================
// Output Formatting
// =====================================================================

static void print_header() {
    printf("%-6s %-6s | %-18s | %-18s | %-18s | %-8s %-8s %-8s\n",
           "M", "N",
           "Ternary-Zero", "cuBLAS FP16", "INT4 Dequant",
           "TZ-vs-FP16", "TZ-vs-INT4", "FP16-BW%");
    printf("%-6s %-6s | %-8s %-8s | %-8s %-8s | %-8s %-8s | %-8s %-8s %-8s\n",
           "", "", "avg(us)", "BW(GB/s)",
           "avg(us)", "BW(GB/s)",
           "avg(us)", "BW(GB/s)",
           "speedup", "speedup", "");
    printf("%s\n", std::string(120, '-').c_str());
}

static void print_result(int M, int N,
                         const KernelResult& tz, const KernelResult& fp16,
                         const KernelResult& int4) {
    float speedup_vs_fp16 = fp16.timing.avg_us / tz.timing.avg_us;
    float speedup_vs_int4 = int4.timing.avg_us / tz.timing.avg_us;

    printf("%-6d %-6d | %8.2f %8.1f | %8.2f %8.1f | %8.2f %8.1f | %7.2fx %7.2fx %7.1f\n",
           M, N,
           tz.timing.avg_us, tz.bandwidth_GBps,
           fp16.timing.avg_us, fp16.bandwidth_GBps,
           int4.timing.avg_us, int4.bandwidth_GBps,
           speedup_vs_fp16, speedup_vs_int4, fp16.peak_bw_pct);
}

static void print_latency_detail(const char* label, const KernelResult& r) {
    printf("  %-20s | avg=%7.2f  min=%7.2f  p50=%7.2f  p95=%7.2f  p99=%7.2f  std=%7.2f us\n",
           label, r.timing.avg_us, r.timing.min_us, r.timing.p50_us,
           r.timing.p95_us, r.timing.p99_us, r.timing.std_us);
    printf("  %-20s | BW=%7.1f GB/s (%.1f%% peak)  bytes=%zu\n",
           "", r.bandwidth_GBps, r.peak_bw_pct, r.bytes_transferred);
}

// =====================================================================
// Main
// =====================================================================

int main(int argc, char** argv) {
    printf("========================================================\n");
    printf("  Ternary-Zero Baseline Comparison: M=1 Decode GEMV\n");
    printf("  Target: RTX 4060 (sm_89, 32MB L2, 272 GB/s peak)\n");
    printf("========================================================\n\n");

    GpuInfo info = get_gpu_info();
    printf("=== GPU ===\n");
    printf("  %s\n", info.name);
    printf("  SMs: %d  L2: %d KB  BW: %.1f GB/s  Compute: %.0f.%.0f\n\n",
           info.sm_count, info.l2_size_kb, info.peak_bw_gbps,
           info.compute_cap_major, info.compute_cap_minor);

    // Shapes: real FFN layer dimensions from Llama-family models
    std::vector<BenchConfig> configs = {
        // Llama-3.2-1B: hidden=2048, intermediate=8192
        {1, 2048,  200, 5000, "Llama-1B hidden"},
        {1, 8192,  200, 5000, "Llama-1B FFN up"},
        {1, 2048,  200, 5000, "Llama-1B FFN down"},

        // Llama-2-7B: hidden=4096, intermediate=11008
        {1, 4096,  200, 5000, "Llama-7B hidden"},
        {1, 11008, 100, 3000, "Llama-7B FFN up"},
        {1, 4096,  200, 5000, "Llama-7B FFN down"},

        // Sweeps
        {1, 768,   200, 5000, "BERT-base"},
        {1, 1024,  200, 5000, "small"},
        {1, 4096,  200, 5000, "medium"},
        {1, 11008, 100, 3000, "large (11008)"},
        {1, 14336, 100, 3000, "Llama-3 14336"},
    };

    if (argc >= 3) {
        int M = atoi(argv[1]);
        int N = atoi(argv[2]);
        int warmup = argc > 3 ? atoi(argv[3]) : 200;
        int iters  = argc > 4 ? atoi(argv[4]) : 5000;
        configs = {{M, N, warmup, iters, "custom"}};
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    print_header();

    for (const auto& cfg : configs) {
        int M = cfg.M;
        int N = cfg.N;
        if (N % 16 != 0) { printf("Skip M=%d N=%d\n", M, N); continue; }

        // Ternary weights
        std::vector<int8_t> ternary_weights;
        generate_ternary_weights(ternary_weights, M, N, 0.33f);
        auto packed = pack_ternary(ternary_weights, M, N);
        auto fp16_weights = ternary_to_fp16(ternary_weights);

        // INT4 weights
        auto int4_weights = generate_int4_weights(M, N);
        int block_size = 32;
        int num_blocks = N / block_size;
        std::vector<__half> scales(M * num_blocks);
        std::mt19937 rng(77);
        std::uniform_real_distribution<float> sdist(0.01f, 0.1f);
        for (auto& s : scales) s = __float2half(sdist(rng));

        // Activations
        std::vector<__half> activations(N);
        std::normal_distribution<float> adist(0.0f, 1.0f);
        for (auto& a : activations) a = __float2half(adist(rng));

        // Device allocations
        uint32_t* d_packed;
        __half*   d_fp16_w;
        __half*   d_act;
        __half*   d_out_tz;
        __half*   d_out_fp16;
        __half*   d_out_int4;
        uint8_t*  d_int4_w;
        __half*   d_scales;

        int packed_cols = N / 16;
        CUDA_CHECK(cudaMalloc(&d_packed, (size_t)M * packed_cols * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_fp16_w, (size_t)M * N * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_act, (size_t)N * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_out_tz, (size_t)M * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_out_fp16, (size_t)M * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_out_int4, (size_t)M * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_int4_w, int4_weights.size()));
        CUDA_CHECK(cudaMalloc(&d_scales, scales.size() * sizeof(__half)));

        CUDA_CHECK(cudaMemcpy(d_packed, packed.data(),
                              packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_fp16_w, fp16_weights.data(),
                              fp16_weights.size() * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_act, activations.data(),
                              N * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_int4_w, int4_weights.data(),
                              int4_weights.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_scales, scales.data(),
                              scales.size() * sizeof(__half), cudaMemcpyHostToDevice));

        // L2 persist policy for ternary weights
        ternary_zero_set_l2_policy(stream, d_packed,
                                   (size_t)M * packed_cols * sizeof(uint32_t));

        // Run benchmarks
        auto tz = bench_ternary_zero(d_packed, d_act, d_out_tz,
                                     M, N, stream, cfg.warmup_iters, cfg.bench_iters);
        auto fp16 = bench_cublas_fp16(handle, d_fp16_w, d_act, d_out_fp16,
                                      M, N, stream, cfg.warmup_iters, cfg.bench_iters);
        auto int4 = bench_int4_dequant(d_int4_w, d_act, d_out_int4, d_scales,
                                       M, N, stream, cfg.warmup_iters, cfg.bench_iters);

        print_result(M, N, tz, fp16, int4);

        CUDA_CHECK(cudaFree(d_packed));
        CUDA_CHECK(cudaFree(d_fp16_w));
        CUDA_CHECK(cudaFree(d_act));
        CUDA_CHECK(cudaFree(d_out_tz));
        CUDA_CHECK(cudaFree(d_out_fp16));
        CUDA_CHECK(cudaFree(d_out_int4));
        CUDA_CHECK(cudaFree(d_int4_w));
        CUDA_CHECK(cudaFree(d_scales));
    }

    // Detailed latency breakdown for the primary shape
    printf("\n=== Detailed Latency Breakdown (M=1, N=4096) ===\n");
    {
        int M = 1, N = 4096;
        std::vector<int8_t> tw; generate_ternary_weights(tw, M, N, 0.33f);
        auto pk = pack_ternary(tw, M, N);
        auto fw = ternary_to_fp16(tw);
        std::vector<__half> act(N);
        std::mt19937 rng(42);
        std::normal_distribution<float> ad(0.0f, 1.0f);
        for (auto& a : act) a = __float2half(ad(rng));

        uint32_t* d_pk; __half* d_fw; __half* d_act; __half* d_tz; __half* d_fp;
        int pc = N / 16;
        CUDA_CHECK(cudaMalloc(&d_pk, (size_t)M * pc * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_fw, (size_t)M * N * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_act, (size_t)N * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_tz, (size_t)M * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_fp, (size_t)M * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(d_pk, pk.data(), pk.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_fw, fw.data(), fw.size() * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_act, act.data(), N * sizeof(__half), cudaMemcpyHostToDevice));
        ternary_zero_set_l2_policy(stream, d_pk, (size_t)M * pc * sizeof(uint32_t));

        auto tz_r = bench_ternary_zero(d_pk, d_act, d_tz, M, N, stream, 500, 10000);
        auto fp_r = bench_cublas_fp16(handle, d_fw, d_act, d_fp, M, N, stream, 500, 10000);

        print_latency_detail("Ternary-Zero", tz_r);
        print_latency_detail("cuBLAS FP16", fp_r);

        printf("\n  Latency floor (Ternary-Zero): %.2f us\n", tz_r.timing.min_us);
        printf("  Bandwidth utilization: %.1f / %.1f GB/s (%.1f%%)\n",
               tz_r.bandwidth_GBps, info.peak_bw_gbps, tz_r.peak_bw_pct);

        CUDA_CHECK(cudaFree(d_pk)); CUDA_CHECK(cudaFree(d_fw));
        CUDA_CHECK(cudaFree(d_act)); CUDA_CHECK(cudaFree(d_tz)); CUDA_CHECK(cudaFree(d_fp));
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
