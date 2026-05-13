# Ternary-Zero Architecture

## 1. System Overview

Ternary-Zero is a W2A16 (2-bit Weight, 16-bit Activation) inference engine for ternary-weight neural networks. Weights are constrained to {-1, 0, +1} and packed 16 per `uint32_t`. Activations remain FP16. The GEMV kernel is multiply-free: each output element reduces to a sequence of add/subtract operations gated by zero-skips.

```
+------------------------------------------------------------------+
|                     Python Layer (ternary_zero/)                  |
|  BitLinear.forward()  |  ternary_quantize()  |  Tensor class      |
|  autograd (STE)       |  pack/unpack utils   |  NumPy/PyTorch     |
+-----------------------------+------------------------------------+
                              | PyO3 / numpy bindings
+-----------------------------v------------------------------------+
|                     Rust Layer (src/)                             |
|  BitLinear struct  |  CudaMemoryPool  |  PendingResult           |
|  GpuBuffer (RAII)  |  PinnedHostBuf   |  CudaStream / CudaEvent  |
|  pack_ternary_to_u32  |  STE quantize  |  error handling          |
+-----------------------------+------------------------------------+
                              | FFI (extern "C")
+-----------------------------v------------------------------------+
|                     CUDA Layer (kernel/)                          |
|  ternary_zero_gemv_kernel  |  PTX BFE extraction                 |
|  stride-17 shared memory   |  vectorized uint4 loads             |
|  branchless zero-gating    |  FP32 warp reduction                |
|  L2 cache pinning          |  sm_89 targeted                     |
+-----------------------------+------------------------------------+
                              | nvcc compilation
+-----------------------------v------------------------------------+
|                     Build Layer (build.rs)                        |
|  nvcc detection  |  CUDA_HOME/CUDA_PATH resolution               |
|  sm_89 arch flag |  static lib linking  |  cpu-only fallback     |
+------------------------------------------------------------------+
```

### Layer Responsibilities

| Layer | Language | Role |
|-------|----------|------|
| Python | Python/NumPy | Module API, autograd STE, quantization utilities, Tensor abstraction |
| Rust | Rust + PyO3 | RAII GPU memory, pool allocator, FFI bindings, pack/unpack, async pipeline |
| CUDA | CUDA C++ + PTX | GEMV kernel, L2 cache policy, bank-conflict-free shared memory |
| Build | Rust build.rs | nvcc invocation, static lib creation, platform-specific linking |

---

## 2. Bit Packing Format

### 2.1 Ternary Encoding

Each weight takes one of three values, encoded in 2 bits:

```
Value   Binary   Hex
-----   ------   ---
  0       00      0x0    (zero-gate: skip accumulation)
 +1       01      0x1    (add activation)
 -1       10      0x2    (subtract activation)
  ?       11      0x3    (INVALID - never produced)
```

### 2.2 Packing Layout

16 weights are packed into a single `uint32_t` using LSB-first ordering:

```
uint32_t word
+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
|31 |29 |27 |25 |23 |21 |19 |17 |15 |13 |11 | 9 | 7 | 5 | 3 | 1 |
|30 |28 |26 |24 |22 |20 |18 |16 |14 |12 |10 | 8 | 6 | 4 | 2 | 0 |
+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
|w15|w14|w13|w12|w11|w10|w9 |w8 |w7 |w6 |w5 |w4 |w3 |w2 |w1 |w0 |
 MSB                                                        LSB

Bit extraction formula for weight w[i]:
  bits = (word >> (i * 2)) & 0b11
```

### 2.3 Weight Matrix Memory Layout

For an M x N weight matrix (M output rows, N input columns, N divisible by 16):

```
Logical view (M=4, N=32):
+--------------------------------------------------------------+
| Row 0: w[0,0]  w[0,1]  w[0,2]  ...  w[0,31]                |
| Row 1: w[1,0]  w[1,1]  w[1,2]  ...  w[1,31]                |
| Row 2: w[2,0]  w[2,1]  w[2,2]  ...  w[2,31]                |
| Row 3: w[3,0]  w[3,1]  w[3,2]  ...  w[3,31]                |
+--------------------------------------------------------------+

Packed view (4 rows x 2 uint32_t per row):
+--------------------------------------------------+
| word[0]: w[0,0..15]  | word[1]: w[0,16..31]  |  <- Row 0
| word[2]: w[1,0..15]  | word[3]: w[1,16..31]  |  <- Row 1
| word[4]: w[2,0..15]  | word[5]: w[2,16..31]  |  <- Row 2
| word[6]: w[3,0..15]  | word[7]: w[3,16..31]  |  <- Row 3
+--------------------------------------------------+

packed_cols = N / 16
Total packed words = M x packed_cols
```

### 2.4 Compression Ratios

```
Format       Bits/Weight   Weights/32b Word   Ratio vs FP32   Ratio vs FP16
------       -----------   ----------------   -------------   -------------
FP32              32               1              1.0x             0.5x
FP16              16               2              2.0x             1.0x
INT8               8               4              4.0x             2.0x
INT4               4               8              8.0x             4.0x
Ternary (W2)       2              16             16.0x             8.0x
```

---

## 3. Quantization Pipeline

### 3.1 Forward: Ternary Quantization with STE

The quantization function maps FP16/FP32 weights to {-1, 0, +1} using a threshold derived from the weight statistics.

```
Input:  W (FP16 weight matrix), alpha (scaling hyperparameter, typically 0.5-0.7)

Step 1: Compute threshold
  mean_abs = (1/MN) * sum(|W[i]|)
  threshold = alpha * mean_abs

Step 2: Quantize each weight
            +-- +1    if W[i] > threshold
  Q[i] = --+  0    if |W[i]| <= threshold
            +-- -1    if W[i] < -threshold

Step 3: Compute scale factor (for dequantization)
  scale = mean(|W[i]|) for all i where Q[i] != 0
  (falls back to 1.0 if all weights quantize to zero)
```

Implemented in `src/ste.rs:29` (`ternary_quantize_ste`) and `ternary_zero/quantize.py:12` (`ternary_quantize`).

### 3.2 Backward: Straight-Through Estimator

The quantization function is non-differentiable. STE approximates the gradient by passing it through as if quantization were the identity function within a clipping range:

```
Forward:   q = Quantize(w)
Backward:  dL/dw ~ dL/dq * 1_{|w/scale| <= 1}

Weight gradient (ste_backward_weights):
  For each output m, input n:
    grad_w[m,n] = grad_out[m] x activation[n]    if |raw_w[m,n]/scale| <= 1
    grad_w[m,n] = 0                                otherwise

Activation gradient (ste_backward_activations):
  For each input n:
    grad_act[n] = sum_m (grad_out[m] x ternary_w[m,n] x scale)
```

Implemented in `src/ste.rs:100` and `src/ste.rs:140`.

### 3.3 End-to-End Quantization Flow

```
Training:
  +-------------+     +--------------+     +---------------+     +------------+
  | FP32 Weights |---->|  Quantize    |---->| Ternary x     |---->|  Linear    |
  | (learnable)  |     |  (STE fwd)   |     |  scale (deq)  |     |  (GEMV)    |
  +------+------+     +--------------+     +---------------+     +------+-----+
         |                                                              |
         |              +--------------+     +---------------+         |
         +--------------<-- Weight     |<---->  STE          |<--------+
                        |  Update      |     |  Backward     |
                        +--------------+     +---------------+

Inference:
  +-------------+     +--------------+     +---------------+
  | Pre-quantized|---->|  Pack to     |---->|  CUDA GEMV    |----> Output
  | {-1,0,+1}   |     |  uint32_t    |     |  (multiply-   |
  | + scale      |     |  (16 per w)  |     |   free)       |
  +--------------+     +--------------+     +---------------+
```

---

## 4. CUDA Kernel Architecture

### 4.1 Kernel Overview

The `ternary_zero_gemv_kernel` (`kernel/ternary_zero.cu:67`) computes `output[m] = sum_n(decode(W[m,n]) x A[n])` where each decoded weight is -1, 0, or +1. This reduces a standard GEMV to add/subtract operations with zero-gating -- no multiplications required.

```
Configuration:
  BLOCK_SIZE         = 256 threads
  WARPS_PER_BLOCK    = 8 (256/32)
  ACT_TILE_SIZE      = 1024 elements
  WEIGHTS_PER_UINT32 = 16

Launch: <<<dim3(M), dim3(256)>>>
  One block per output row.
  Each block cooperatively tiles across N input features.
```

### 4.2 Shared Memory Staging with Stride-17 Padding

Activations are loaded into shared memory before the compute loop. A stride-17 padding scheme eliminates 4-way bank conflicts that would occur with stride-16 (natural) access.

```
Physical shared memory: __half s_act[1088]   (1024 + 1024/16 x 1 padding)

Bank conflict analysis (32 banks, 4-byte bank width, 2 half elements per bank):

Without padding (stride-16, natural):
  Threads 0 and 16 both access bank 0   -> 2-way conflict
  Threads 1 and 17 both access bank 1   -> 2-way conflict
  ... (16 pairs of conflicts across 32 banks)

With padding (stride-17):
  +---------+-----------+----------+-------+
  | Logical | Physical  | Offset   | Bank  |
  +---------+-----------+----------+-------+
  |    0    |     0     |    0     |   0   |
  |    1    |     1     |    0     |   1   |
  |   ...   |    ...    |   ...    |  ...  |
  |   15    |    15     |    0     |  15   |
  |   16    |    17     |    1     |   1   |  (not bank 0!)
  |   17    |    18     |    1     |   2   |
  |   ...   |    ...    |   ...    |  ...  |
  |   31    |    32     |    1     |  16   |
  +---------+-----------+----------+-------+

Index function: smem_idx(flat) = flat + (flat / 16)
  Each group of 16 halfs starts at a different bank offset.
  Within any 32-thread warp, all 32 banks are accessed uniquely.
```

### 4.3 Vectorized uint4 Loads

Activation tiles are loaded 8 half-precision elements per transaction using `uint4` (128-bit) loads:

```
const uint4* src_vec = reinterpret_cast<const uint4*>(activations + tile_start);

for (int i = tid; i < vec_count; i += BLOCK_SIZE) {
    uint4 v = src_vec[i];               // 128-bit load: 8 x FP16
    const __half* h = reinterpret_cast<const __half*>(&v);
    for (int j = 0; j < 8; j++)
        s_act[smem_idx(base + j)] = h[j];
}

Scalar fallback for tail elements (tile_len % 8 != 0):
for (int i = vec_count * 8 + tid; i < tile_len; i += BLOCK_SIZE)
    s_act[smem_idx(i)] = activations[tile_start + i];
```

This issues one 128-bit transaction per thread instead of eight 16-bit transactions, maximizing memory throughput.

### 4.4 PTX BFE for 2-Bit Extraction

The kernel uses inline PTX `bfe.u32` (Bit Field Extract) to pull 2-bit weight fields from packed `uint32_t` words. This maps to a single hardware instruction on sm_89:

```
PTX_BFE(bits_w0, packed, w * 2, 2);      // Extract 2 bits at position w*2
PTX_BFE(sign_w0, bits_w0, 1, 1);         // Extract sign bit (bit 1)
PTX_BFE(mag_w0,  bits_w0, 0, 1);         // Extract magnitude bit (bit 0)

Encoding extraction:
  bits  sign  mag   meaning
  00     0     0    -> weight = 0   (zero)
  01     0     1    -> weight = +1  (positive)
  10     1     0    -> weight = -1  (negative)
  11     1     1    -> INVALID (never produced)
```

Defined as macros in `kernel/ptx_utils.h:26`.

### 4.5 Branchless Zero-Gating via Sign/Magnitude Masking

The kernel avoids branches entirely. Weight decoding and gating are performed through bitwise operations:

```
// Step 1: Sign-flip the activation (XOR with sign-derived mask)
sign_mask = (uint32_t)(-(int32_t)sign) & 0x8000u;
  // sign=1 (negative): sign_mask = 0x8000  (FP16 sign bit)
  // sign=0 (positive): sign_mask = 0x0000  (no-op)
signed_a  = activation_raw ^ sign_mask;
  // XOR with 0x8000 flips FP16 sign bit -> negation

// Step 2: Zero-gate (AND with nonzero-derived mask)
nz        = sign | mag;
  // 00 -> nz=0 (zero weight)
  // 01 -> nz=1 (positive)
  // 10 -> nz=1 (negative)
nz_mask   = (uint32_t)(-(int32_t)nz);
  // nz=1 -> 0xFFFFFFFF (all bits set)
  // nz=0 -> 0x00000000 (all bits clear)
gated_a   = signed_a & nz_mask;
  // AND with 0xFFFFFFFF -> pass through
  // AND with 0x00000000 -> zero out

// Step 3: Convert back to float and accumulate
acc += __half2float(gated_a);
```

Complete truth table:

```
Weight   bits   sign   mag   nz    nz_mask      sign_mask   signed_a   gated_a
------   ----   ----   ---   --    -------      ---------   --------   -------
  0       00      0     0    0    0x00000000    0x0000      +act       0x0000 (0.0)
 +1       01      0     1    1    0xFFFFFFFF    0x0000      +act       +act
 -1       10      1     0    1    0xFFFFFFFF    0x8000      -act       -act
  ?       11      1     1    1    0xFFFFFFFF    0x8000      -act       -act (INVALID)
```

### 4.6 FP32 Warp-Level Reduction

Accumulation uses FP32 to avoid overflow (FP16 max = 65504, which overflows for N>=2048). Warp reduction uses butterfly shuffles:

```
__device__ float warp_reduce_sum_f32(float val) {
    for (int offset = 16; offset >= 1; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}
```

Reduction tree for 8 warps (256 threads):

```
Phase 1: Per-warp reduction (butterfly shuffle within each warp)
  Thread 0..31 in warp k: shuffle tree reduces to thread 0
  Thread 0 writes warp_sum[k] to s_warp_sums[k]

Phase 2: Cross-warp reduction (warp 0 only)
  s_warp_sums[0..7] loaded by threads 0..7
  Shuffle tree within warp 0:
    offset=4: [0]+[4], [1]+[5], [2]+[6], [3]+[7]
    offset=2: [01]+[23], [45]+[67]
    offset=1: [0123]+[4567]
  Thread 0 writes final sum to output[row] as FP16

Warp 0 --+
Warp 1 --+
Warp 2 --+
Warp 3 --+---> s_warp_sums[0..7] ---> Warp 0 shuffle ---> output[row]
Warp 4 --+
Warp 5 --+
Warp 6 --+
Warp 7 --+
```

### 4.7 L2 Cache Pinning Strategy

Weight data is pinned in the L2 cache using CUDA's access policy window API (`kernel/ternary_zero.cu:201`). This is critical because:

- Weights are read repeatedly across kernel invocations (same matrix, different input vectors).
- L2 residency is controlled per-stream via `cudaStreamSetAttribute`.
- `hitRatio=1.0` with `cudaAccessPropertyPersisting` marks the entire weight allocation as persisting, while misses fall back to `cudaAccessPropertyStreaming`.

```
cudaStreamAttrValue attr = {};
attr.accessPolicyWindow.base_ptr  = weight_ptr;
attr.accessPolicyWindow.num_bytes = weight_bytes;
attr.accessPolicyWindow.hitRatio  = 1.0f;
attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
```

### 4.8 Kernel Data Flow Diagram

```
Global Memory                     Shared Memory                 Registers
------------                     ---------------                 ---------

activations[N] --uint4 loads--> s_act[1088]                     (tile buffer)
                                (stride-17 padded)

weights[row*N/16] --> (global) --------------------------------> packed word

                                                     PTX_BFE --> sign, mag
                                                     bitwise --> nz_mask, sign_mask
                                                     XOR/AND --> gated activation

                                                     FP32 add -> acc (per-thread)

                                                     warp_reduce_sum_f32
                                                     shfl_down tree

output[row] <-------------------------------------------------------------------- block_sum (FP16)
```

---

## 5. Rust Backend

### 5.1 Module Structure

```
src/
+-- lib.rs          PyO3 module definition, Python-facing functions
+-- bitlinear.rs    GpuBuffer, CudaMemoryPool, BitLinear, pack/unpack
+-- ffi.rs          CUDA runtime FFI declarations, error enum
+-- ste.rs          STE quantization, dequantization, gradient functions
+-- error.rs        Unified TernaryError type
```

### 5.2 GpuBuffer -- Unique Ownership

`GpuBuffer<T>` (`src/bitlinear.rs:22`) is a RAII wrapper around a single CUDA device allocation:

```
GpuBuffer<T>
+-- ptr: *mut T          (device pointer from cudaMalloc)
+-- len: usize           (element count)
+-- Drop -> cudaFree     (automatic cleanup)
+-- Send                 (ownership transfer safe)
+-- !Sync                (concurrent &GpuBuffer could race on memcpy)
```

Supports both synchronous and async memory transfers:
- `copy_from_host()` / `copy_to_host()` -- blocking `cudaMemcpy`
- `copy_from_host_async()` / `copy_to_host_async()` -- stream-ordered `cudaMemcpyAsync`

### 5.3 CudaMemoryPool -- Free-List Sub-Allocator

`CudaMemoryPool` (`src/bitlinear.rs:232`) eliminates per-call `cudaMalloc`/`cudaFree` overhead in hot inference paths:

```
CudaMemoryPool
+-- inner: Arc<Mutex<Vec<PoolSlot>>>
|       +-- PoolSlot { ptr, byte_size }
|
+-- alloc<T>(count) -> PooledGpuBuffer<T>
|       +-- Search free list for slot with byte_size >= requested
|       +-- Found -> reuse slot (swap_remove from Vec)
|       +-- Not found -> cudaMalloc fresh allocation
|
+-- Drop
        +-- Arc strong count = 1 -> drain and cudaFree all slots
        +-- Fallback -> drain locked slots and cudaFree

PooledGpuBuffer<T>
+-- ptr, len, byte_size
+-- pool: Weak<Mutex<Vec<PoolSlot>>>
+-- Drop
|       +-- Pool alive (Weak upgrades) -> return slot to free list
|       +-- Pool gone (Weak fails) -> cudaFree directly
+-- Send
```

### 5.4 PinnedHostBuffer -- Page-Locked for Async DMA

`PinnedHostBuffer<T>` (`src/bitlinear.rs:436`) wraps `cudaMallocHost`/`cudaFreeHost` for page-locked host memory. Required for:

- Non-blocking `cudaMemcpyAsync` D2H transfers
- DMA engine access without CPU staging through OS page tables
- Used exclusively by `PendingResult` for async output reads

### 5.5 CudaStream and CudaEvent -- RAII Wrappers

```
CudaStream (src/bitlinear.rs:511)
+-- new()          -> cudaStreamCreate
+-- synchronize()  -> cudaStreamSynchronize
+-- raw()          -> cudaStream_t
+-- Drop           -> cudaStreamDestroy
+-- Send, !Sync

CudaEvent (src/bitlinear.rs:549)
+-- new()          -> cudaEventCreate
+-- record(stream) -> cudaEventRecord
+-- synchronize()  -> cudaEventSynchronize
+-- is_ready()     -> cudaEventQuery (non-blocking poll)
+-- Drop           -> cudaEventDestroy
+-- Send
```

### 5.6 PendingResult -- Async Forward with Event Polling

`PendingResult` (`src/bitlinear.rs:600`) is the return type of `forward_async()`. It holds all resources needed for an in-flight kernel:

```
PendingResult
+-- host_buffer: PinnedHostBuffer<u16>     (D2H destination)
+-- event: CudaEvent                        (recorded after D2H copy)
+-- activation_buffer: PooledGpuBuffer<u16> (kept alive for kernel reads)
+-- m: usize                                (output dimension)
|
+-- is_ready() -> bool       (non-blocking: cudaEventQuery)
+-- get_output() -> Vec<f16> (blocking: cudaEventSynchronize + read)
+-- Drop -> event.synchronize() (ensures GPU work completes before pool reclaim)
```

---

## 6. Execution Flow

### 6.1 Synchronous Forward Pass

```
Python                          Rust (BitLinear::forward)          CUDA Runtime
------                          --------------------------          -----------
BitLinear.forward(x) ------->
                                pool.alloc<u16>(N)
                                  +-- free list hit? -> reuse
                                  +-- miss -> cudaMalloc ----------> alloc N x 2 bytes

                                Convert f16 -> u16 bits
                                d_act.copy_from_host() -----------> cudaMemcpy H2D

                                ffi::ternary_zero_gemv_f16() -----> kernel<<<M,256>>>
                                  +-- tile activations into smem
                                  +-- PTX BFE decode weights
                                  +-- branchless zero-gate accumulate
                                  +-- warp reduce -> output[row]

                                output_buffer.copy_to_host() -----> cudaMemcpy D2H
                                  (implicit stream sync)

                                Convert u16 -> f16
return Vec<f16> <---------------
```

### 6.2 Asynchronous Forward Pass

```
Python                          Rust (BitLinear::forward_async)    CUDA Stream
------                          -------------------------------    -----------
forward_async(x) --------->
                                pool.alloc<u16>(N)
                                d_act.copy_from_host_async() -----> [H2D memcpy]

                                ffi::ternary_zero_gemv_f16() -----> [kernel launch]

                                PinnedHostBuffer::alloc(M)
                                cudaMemcpyAsync D2H --------------> [D2H memcpy]

                                event.record(stream) -------------> [event record]

return PendingResult <----------

   ... host work continues ...

pending.is_ready() ----------> cudaEventQuery -------------------> (non-blocking)
return bool <-------------------

pending.get_output() ---------> event.synchronize() -------------> (blocks until done)
                                read pinned_buffer as Vec<f16>
return output <-----------------
```

### 6.3 BitLinear Layer Construction

```
BitLinear::new(M, N)
+-- Validate N % 16 == 0 (required for 2-bit packing)
+-- packed_weights = GpuBuffer<u32>::alloc(M x N/16)  (weight storage)
+-- output_buffer  = GpuBuffer<u16>::alloc(M)          (output staging)
+-- pool = CudaMemoryPool::new()                       (activation allocator)
+-- stream = CudaStream::new()                         (execution stream)
+-- l2_pinned = false

BitLinear::load_weights(ternary_weights)
+-- pack_ternary_to_u32(weights, N)  ->  Vec<u32>
+-- packed_weights.copy_from_host(&packed)

BitLinear::pin_weights_in_l2()
+-- ternary_zero_set_l2_policy(stream, ptr, bytes)
+-- l2_pinned = true
```

---

## 7. Why W2A16 Matters

### 7.1 Compression

Ternary quantization achieves extreme compression by reducing each weight to 2 bits:

```
                    Memory per weight    Compression vs FP32    Compression vs FP16
                    -----------------    -------------------    -------------------
FP32 baseline            32 bits              1.0x                    0.5x
FP16                     16 bits              2.0x                    1.0x
INT8                      8 bits              4.0x                    2.0x
W2A16 (ternary)           2 bits             16.0x                    8.0x

Example: 7B parameter model
  FP32 weights:  28.0 GB
  FP16 weights:  14.0 GB
  INT8 weights:   7.0 GB
  W2A16 weights:  1.75 GB    <- fits in 2 GB VRAM with headroom
```

### 7.2 Memory-Bandwidth Bound Inference

LLM inference is memory-bandwidth bound, but the right analytical frame is a roofline model rather than a single bandwidth ratio. For each generated token, the weight matrix must be read from the cache hierarchy or DRAM, and the achieved throughput is capped by both arithmetic intensity and occupancy.

Let operational intensity be

$$I = \frac{\text{useful operations}}{\text{bytes transferred}}$$

and throughput be approximated by

$$P = \min\left(P_{\text{peak}} \cdot Occ \cdot \eta_{\text{warp}}, \; I \cdot BW_{\text{eff}}\right)$$

where $Occ$ is the occupancy ceiling, $\eta_{\text{warp}}$ is warp efficiency, and $BW_{\text{eff}}$ includes DRAM, L2, and shared-memory reuse. For ternary GEMV, useful work scales with non-zero accumulations while bytes are dominated by packed weights plus activations and output.

```
Standard GEMV (FP16):  output[m] = sum_n W[m,n] x A[n]
  - 2*MN FLOPs
  - 2*MN bytes weight read (FP16)
  - Arithmetic intensity = 1 FLOP/byte (weight-dominant idealization)

Ternary GEMV (W2A16):  output[m] = sum_n decode(W[m,n]) x A[n]
  - (1-rho0)*MN useful accumulations
  - MN/4 bytes weight read (2-bit packed, 16 per uint32_t)
  - Higher arithmetic intensity, but still memory-bound for practical decode kernels
  - Actual speedup is limited by decode throughput, occupancy, and cache reuse

Roofline reasoning:
  If DRAM bandwidth is B bytes/sec and overheads are negligible:
    FP16 GEMV time  ~ 2MN / B
    W2A16 GEMV time ~ (MN/4) / B
  In practice, add T_decode + T_reduce + T_sync and apply occupancy limits.

  On RTX 4060 (272 GB/s bandwidth):
    FP16  7B model decode: 14 GB / 272 GB/s   ~ 51 ms/token (weight loading only)
    W2A16 7B model decode: 1.75 GB / 272 GB/s ~ 6.4 ms/token (weight loading only)
```

This is why the kernel is strongest in the batch-size-1 decode regime: the bandwidth term improves by 8x relative to FP16, but the realized speedup depends on whether decode and occupancy stay below the memory ceiling.

### 7.3 Multiply-Free GEMV

The ternary encoding eliminates all multiplications from the inner loop. Each decoded weight contributes one of three operations:

```
Weight =  0  ->  skip entirely     (branchless: AND with 0x00000000 mask)
Weight = +1  ->  add activation    (sign bit 0: activation unchanged)
Weight = -1  ->  subtract activation (sign bit 1: XOR with 0x8000 to flip FP16 sign)
```

On modern GPUs, FP16 multiply throughput is the same as add throughput, so the multiply-free property alone doesn't save cycles. The win comes from the combination of:

1. **16x less weight memory** to fetch from DRAM
2. **Zero-gating** skips ~30-50% of additions (typical sparsity after ternary quantization)
3. **2-bit extraction** is a single `bfe.u32` instruction -- no decode table needed

---

## 8. Current Limitations

### 8.1 CUDA Kernel Targets sm_89 Only

The kernel is compiled with `--gpu-architecture=sm_89` (Ada Lovelace, RTX 4060/4070/4090). PTX instructions like `bfe.u32` are available on all sm_50+ GPUs, but the register allocation (`-maxrregcount=64`), launch bounds (`__launch_bounds__(256, 4)`), and shared memory sizing are tuned for sm_89's resource constraints. Running on other architectures would require re-tuning these parameters.

### 8.2 No Kernel Fusion

The current design launches a standalone GEMV kernel. There is no fusion with:
- Bias addition
- Activation functions (ReLU, GELU, SiLU)
- Layer normalization
- Residual connections

Each of these would require a separate kernel launch and global memory round-trip. Fusion would reduce latency by keeping intermediate results in registers/shared memory.

### 8.3 No Multi-GPU Support

All operations target a single CUDA device. There is no:
- Weight tensor parallelism across GPUs
- NCCL collective communication
- Peer-to-peer memory access
- Pipeline parallelism across device boundaries

### 8.4 No INT4 Support

The system is exclusively W2 (2-bit ternary). INT4 quantization (GPTQ, AWQ-style) would provide a different accuracy/compression tradeoff but is not implemented. The packing infrastructure (`pack_ternary_to_u32`) is specific to the 2-bit ternary encoding and cannot be repurposed for INT4.

### 8.5 Python Autograd Overhead for Training

Training uses Python-level autograd with STE. The backward pass (`ste_backward_weights`, `ste_backward_activations`) runs on the CPU through the Rust FFI, processing FP16 data element-by-element. This is suitable for research/prototyping but not competitive with native CUDA training kernels. The per-element loop in `ste_backward_weights` (`src/ste.rs:122-134`) has O(M x N) complexity with no parallelism beyond what the Python caller provides.

### 8.6 Theoretical GEMV Estimates Not Validated on Hardware

The roofline estimates in Section 7 are theoretical and should be read as ceilings, not promises. Actual performance depends on:
- DRAM bandwidth utilization efficiency (typically 60-80% of peak)
- L2 cache hit rate for weight reuse across tokens
- Occupancy limited by register pressure (64 regs/thread, 4 blocks/SM)
- Shared memory bank conflict residual despite stride-17 padding
- Warp scheduling efficiency with the branchless zero-gate pattern
- Decode throughput for `BFE` / `PRMT` / `LOP3` instruction mix

No profiling data or benchmark results are included in this documentation. The `gemv_bench` criterion benchmark referenced in `Cargo.toml` would provide empirical measurements.

### 8.7 Additional Limitations

- **N must be a multiple of 16**: Required by the 16-weights-per-uint32_t packing. Non-multiple-of-16 input dimensions require padding.
- **Single-precision warp reduction only**: The kernel accumulates in FP32 but writes FP16 output. For very large N (>=65536), even FP32 may accumulate rounding error.
- **No weight-only quantization for activations**: Activations remain FP16. W4A8 or similar schemes are not supported.
- **Synchronous path uses null-stream semantics**: The sync `forward()` relies on implicit synchronization through blocking `cudaMemcpy` on the null stream, but the kernel is launched on a non-default stream. This works because blocking streams synchronize with all preceding work, but it's an implicit coupling.
