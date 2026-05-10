use crate::error::TernaryError;
use crate::ffi::{
    self, cudaEvent_t, cudaStream_t, CUDA_MEMCPY_DEVICE_TO_HOST, CUDA_MEMCPY_HOST_TO_DEVICE,
};
use half::f16;
use std::os::raw::c_void;
use std::ptr;
use std::sync::{Arc, Mutex, Weak};

// =====================================================================
// GPU Buffer: RAII Wrapper for CUDA Device Memory
// =====================================================================
//
// Owns a unique allocation on the device. Allocations and deallocations
// go through the CUDA runtime (cudaMalloc / cudaFree).
//
// Thread safety:
//   - Send: safe to transfer ownership to another thread.
//   - NOT Sync: concurrent &GpuBuffer access could race on host-side
//     memcpy destinations. Use Mutex<GpuBuffer> if shared access is needed.

pub struct GpuBuffer<T: Copy> {
    ptr: *mut T,
    len: usize,
}

impl<T: Copy> GpuBuffer<T> {
    pub fn alloc(len: usize) -> Result<Self, TernaryError> {
        if len == 0 {
            return Ok(Self {
                ptr: ptr::null_mut(),
                len: 0,
            });
        }

        let mut dev_ptr: *mut c_void = ptr::null_mut();
        let elem_size = std::mem::size_of::<T>();
        let size = len
            .checked_mul(elem_size)
            .ok_or_else(|| TernaryError::Overflow {
                context: format!("GpuBuffer::alloc({} * {} bytes)", len, elem_size),
            })?;

        let err = unsafe { ffi::cudaMalloc(&mut dev_ptr as *mut *mut c_void, size) };
        err.to_result()?;

        Ok(Self {
            ptr: dev_ptr as *mut T,
            len,
        })
    }

    pub fn copy_from_host(&mut self, data: &[T]) -> Result<(), TernaryError> {
        if data.len() != self.len {
            return Err(TernaryError::DimensionMismatch {
                expected: self.len,
                actual: data.len(),
                context: "GpuBuffer::copy_from_host".into(),
            });
        }

        if self.len == 0 {
            return Ok(());
        }

        let size = self.byte_size();
        let err = unsafe {
            ffi::cudaMemcpy(
                self.ptr as *mut c_void,
                data.as_ptr() as *const c_void,
                size,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        err.to_result()?;
        Ok(())
    }

    pub fn copy_from_host_async(
        &mut self,
        data: &[T],
        stream: &CudaStream,
    ) -> Result<(), TernaryError> {
        if data.len() != self.len {
            return Err(TernaryError::DimensionMismatch {
                expected: self.len,
                actual: data.len(),
                context: "GpuBuffer::copy_from_host_async".into(),
            });
        }

        if self.len == 0 {
            return Ok(());
        }

        let size = self.byte_size();
        let err = unsafe {
            ffi::cudaMemcpyAsync(
                self.ptr as *mut c_void,
                data.as_ptr() as *const c_void,
                size,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                stream.raw(),
            )
        };
        err.to_result()?;
        Ok(())
    }

    pub fn copy_to_host(&self, data: &mut [T]) -> Result<(), TernaryError> {
        if data.len() != self.len {
            return Err(TernaryError::DimensionMismatch {
                expected: self.len,
                actual: data.len(),
                context: "GpuBuffer::copy_to_host".into(),
            });
        }

        if self.len == 0 {
            return Ok(());
        }

        let size = self.byte_size();
        let err = unsafe {
            ffi::cudaMemcpy(
                data.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                size,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        err.to_result()?;
        Ok(())
    }

    pub fn copy_to_host_async(
        &self,
        data: &mut [T],
        stream: &CudaStream,
    ) -> Result<(), TernaryError> {
        if data.len() != self.len {
            return Err(TernaryError::DimensionMismatch {
                expected: self.len,
                actual: data.len(),
                context: "GpuBuffer::copy_to_host_async".into(),
            });
        }

        if self.len == 0 {
            return Ok(());
        }

        let size = self.byte_size();
        let err = unsafe {
            ffi::cudaMemcpyAsync(
                data.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                size,
                CUDA_MEMCPY_DEVICE_TO_HOST,
                stream.raw(),
            )
        };
        err.to_result()?;
        Ok(())
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn byte_size(&self) -> usize {
        self.len.saturating_mul(std::mem::size_of::<T>())
    }
}

impl<T: Copy> Drop for GpuBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ffi::cudaFree(self.ptr as *mut c_void);
            }
        }
    }
}

// Safety: GpuBuffer uniquely owns device memory. CUDA runtime memory
// operations (cudaMalloc, cudaFree, cudaMemcpy) are thread-safe.
// Send is safe: ownership can transfer between threads.
// Sync is intentionally NOT implemented: shared &GpuBuffer across
// threads could race on host-side memcpy destinations without
// external synchronization.
unsafe impl<T: Copy> Send for GpuBuffer<T> {}

// =====================================================================
// CUDA Memory Pool: Pre-Allocated Device Memory Sub-Allocator
// =====================================================================
//
// Maintains a thread-safe free-list of CUDA device memory blocks.
// When a PooledGpuBuffer is dropped its slot is returned to the pool
// instead of calling cudaFree, eliminating per-call allocation overhead
// in hot inference paths.
//
// Design:
//   - Pool holds Arc<Mutex<Vec<PoolSlot>>>; buffers hold Weak<...>.
//   - On pool drop: unwrap Arc (always succeeds—buffers only hold Weak),
//     drain and cudaFree every slot.
//   - On buffer drop after pool is gone: Weak upgrade fails, so the
//     buffer frees its own allocation directly.

struct PoolSlot {
    ptr: *mut c_void,
    byte_size: usize,
}

// Safety: PoolSlot is only accessed through Arc<Mutex<Vec<PoolSlot>>>.
// The mutex provides the necessary synchronization for thread safety.
unsafe impl Send for PoolSlot {}
unsafe impl Sync for PoolSlot {}

pub struct CudaMemoryPool {
    inner: Option<Arc<Mutex<Vec<PoolSlot>>>>,
}

impl Default for CudaMemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaMemoryPool {
    pub fn new() -> Self {
        Self {
            inner: Some(Arc::new(Mutex::new(Vec::new()))),
        }
    }

    pub fn alloc<T: Copy>(&self, count: usize) -> Result<PooledGpuBuffer<T>, TernaryError> {
        let elem_size = std::mem::size_of::<T>();
        let byte_size = count
            .checked_mul(elem_size)
            .ok_or_else(|| TernaryError::Overflow {
                context: format!("CudaMemoryPool::alloc({} * {} bytes)", count, elem_size),
            })?;

        let inner = self.inner.as_ref().unwrap();
        let weak = Arc::downgrade(inner);

        if byte_size == 0 {
            return Ok(PooledGpuBuffer {
                ptr: ptr::null_mut(),
                len: 0,
                byte_size: 0,
                pool: weak,
            });
        }

        // Attempt to reuse a previously freed block that is large enough.
        {
            let mut free = inner.lock().unwrap();
            if let Some(idx) = free.iter().position(|s| s.byte_size >= byte_size) {
                let slot = free.swap_remove(idx);
                return Ok(PooledGpuBuffer {
                    ptr: slot.ptr as *mut T,
                    len: count,
                    byte_size: slot.byte_size,
                    pool: weak,
                });
            }
        }

        // No suitable free block — allocate fresh device memory.
        let mut dev_ptr: *mut c_void = ptr::null_mut();
        unsafe { ffi::cudaMalloc(&mut dev_ptr as *mut *mut c_void, byte_size) }.to_result()?;

        Ok(PooledGpuBuffer {
            ptr: dev_ptr as *mut T,
            len: count,
            byte_size,
            pool: weak,
        })
    }
}

impl Drop for CudaMemoryPool {
    fn drop(&mut self) {
        if let Some(arc) = self.inner.take() {
            match Arc::try_unwrap(arc) {
                Ok(mutex) => {
                    if let Ok(slots) = mutex.into_inner() {
                        for slot in slots {
                            unsafe {
                                ffi::cudaFree(slot.ptr);
                            }
                        }
                    }
                }
                Err(arc) => {
                    // Fallback: drain free slots even if something holds a strong ref.
                    if let Ok(mut slots) = arc.lock() {
                        for slot in slots.drain(..) {
                            unsafe {
                                ffi::cudaFree(slot.ptr);
                            }
                        }
                    }
                }
            }
        }
    }
}

// =====================================================================
// Pooled GPU Buffer: Pool-Managed Device Memory
// =====================================================================

pub struct PooledGpuBuffer<T: Copy> {
    ptr: *mut T,
    len: usize,
    byte_size: usize,
    pool: Weak<Mutex<Vec<PoolSlot>>>,
}

impl<T: Copy> PooledGpuBuffer<T> {
    pub fn copy_from_host(&mut self, data: &[T]) -> Result<(), TernaryError> {
        if data.len() != self.len {
            return Err(TernaryError::DimensionMismatch {
                expected: self.len,
                actual: data.len(),
                context: "PooledGpuBuffer::copy_from_host".into(),
            });
        }
        if self.len == 0 {
            return Ok(());
        }
        let err = unsafe {
            ffi::cudaMemcpy(
                self.ptr as *mut c_void,
                data.as_ptr() as *const c_void,
                self.byte_size,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        err.to_result()?;
        Ok(())
    }

    pub fn copy_from_host_async(
        &mut self,
        data: &[T],
        stream: &CudaStream,
    ) -> Result<(), TernaryError> {
        if data.len() != self.len {
            return Err(TernaryError::DimensionMismatch {
                expected: self.len,
                actual: data.len(),
                context: "PooledGpuBuffer::copy_from_host_async".into(),
            });
        }
        if self.len == 0 {
            return Ok(());
        }
        let err = unsafe {
            ffi::cudaMemcpyAsync(
                self.ptr as *mut c_void,
                data.as_ptr() as *const c_void,
                self.byte_size,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                stream.raw(),
            )
        };
        err.to_result()?;
        Ok(())
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn byte_size(&self) -> usize {
        self.byte_size
    }
}

impl<T: Copy> Drop for PooledGpuBuffer<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        if let Some(pool) = self.pool.upgrade() {
            // Pool alive — return slot for reuse.
            if let Ok(mut slots) = pool.lock() {
                slots.push(PoolSlot {
                    ptr: self.ptr as *mut c_void,
                    byte_size: self.byte_size,
                });
            }
        } else {
            // Pool dropped — free directly.
            unsafe {
                ffi::cudaFree(self.ptr as *mut c_void);
            }
        }
    }
}

unsafe impl<T: Copy> Send for PooledGpuBuffer<T> {}

// =====================================================================
// Pinned Host Buffer: Page-Locked Host Memory for Async DMA
// =====================================================================

pub struct PinnedHostBuffer<T: Copy> {
    ptr: *mut T,
    len: usize,
}

impl<T: Copy> PinnedHostBuffer<T> {
    pub fn alloc(len: usize) -> Result<Self, TernaryError> {
        if len == 0 {
            return Ok(Self {
                ptr: ptr::null_mut(),
                len: 0,
            });
        }
        let mut host_ptr: *mut c_void = ptr::null_mut();
        let size =
            len.checked_mul(std::mem::size_of::<T>())
                .ok_or_else(|| TernaryError::Overflow {
                    context: "PinnedHostBuffer::alloc".into(),
                })?;
        unsafe { ffi::cudaMallocHost(&mut host_ptr as *mut *mut c_void, size) }.to_result()?;
        Ok(Self {
            ptr: host_ptr as *mut T,
            len,
        })
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    pub fn as_slice(&self) -> &[T] {
        if self.ptr.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.ptr.is_null() {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T: Copy> Drop for PinnedHostBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ffi::cudaFreeHost(self.ptr as *mut c_void);
            }
        }
    }
}

unsafe impl<T: Copy> Send for PinnedHostBuffer<T> {}

// =====================================================================
// CUDA Stream: RAII Wrapper
// =====================================================================

pub struct CudaStream(pub cudaStream_t);

impl CudaStream {
    pub fn new() -> Result<Self, TernaryError> {
        let mut stream: cudaStream_t = ptr::null_mut();
        unsafe { ffi::cudaStreamCreate(&mut stream) }.to_result()?;
        Ok(Self(stream))
    }

    pub fn synchronize(&self) -> Result<(), TernaryError> {
        unsafe { ffi::cudaStreamSynchronize(self.0) }.to_result()?;
        Ok(())
    }

    pub fn raw(&self) -> cudaStream_t {
        self.0
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                ffi::cudaStreamDestroy(self.0);
            }
        }
    }
}

// Safety: safe to transfer ownership between threads.
// NOT Sync: concurrent stream operations without external
// synchronization could produce non-deterministic ordering.
unsafe impl Send for CudaStream {}

// =====================================================================
// CUDA Event: RAII Wrapper for Stream Synchronization
// =====================================================================

pub struct CudaEvent(cudaEvent_t);

impl CudaEvent {
    pub fn new() -> Result<Self, TernaryError> {
        let mut event: cudaEvent_t = ptr::null_mut();
        unsafe { ffi::cudaEventCreate(&mut event) }.to_result()?;
        Ok(Self(event))
    }

    pub fn record(&self, stream: &CudaStream) -> Result<(), TernaryError> {
        unsafe { ffi::cudaEventRecord(self.0, stream.raw()) }.to_result()?;
        Ok(())
    }

    pub fn synchronize(&self) -> Result<(), TernaryError> {
        unsafe { ffi::cudaEventSynchronize(self.0) }.to_result()?;
        Ok(())
    }

    pub fn is_ready(&self) -> bool {
        let err = unsafe { ffi::cudaEventQuery(self.0) };
        err == ffi::CudaError::Success
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                ffi::cudaEventDestroy(self.0);
            }
        }
    }
}

unsafe impl Send for CudaEvent {}

// =====================================================================
// Pending Async Result
// =====================================================================
//
// Returned by `BitLinear::forward_async`.  Holds:
//   - A pinned host buffer receiving the D2H copy.
//   - A CUDA event recorded after the D2H memcpy.
//   - The activation pool buffer (kept alive so the kernel can read it).
//
// The caller can poll `is_ready()` without blocking, or consume with
// `get_output()` which synchronizes and returns the result vector.
// Drop synchronizes the event to guarantee GPU work completes before
// device buffers are returned to the pool.

pub struct PendingResult {
    host_buffer: PinnedHostBuffer<u16>,
    event: CudaEvent,
    #[allow(dead_code)] // held for RAII: prevents pool from reclaiming during kernel execution
    activation_buffer: PooledGpuBuffer<u16>,
    m: usize,
}

impl PendingResult {
    /// Non-blocking check: has the GPU finished all work?
    pub fn is_ready(&self) -> bool {
        self.event.is_ready()
    }

    /// Synchronize and consume the result, returning the FP16 output.
    pub fn get_output(self) -> Result<Vec<f16>, TernaryError> {
        self.event.synchronize()?;
        let slice = self.host_buffer.as_slice();
        Ok(slice.iter().map(|&bits| f16::from_bits(bits)).collect())
    }

    pub fn len(&self) -> usize {
        self.m
    }

    pub fn is_empty(&self) -> bool {
        self.m == 0
    }
}

impl Drop for PendingResult {
    fn drop(&mut self) {
        let _ = self.event.synchronize();
    }
}

// =====================================================================
// BitLinear Layer
// =====================================================================

pub struct BitLinear {
    packed_weights: GpuBuffer<u32>,
    output_buffer: GpuBuffer<u16>,
    pool: CudaMemoryPool,
    stream: CudaStream,
    m: usize,
    n: usize,
    l2_pinned: bool,
}

impl BitLinear {
    /// Create a new BitLinear layer.
    ///
    /// # Arguments
    /// * `m` - Number of output features (rows of weight matrix)
    /// * `n` - Number of input features (columns, must be multiple of 16)
    pub fn new(m: usize, n: usize) -> Result<Self, TernaryError> {
        if !n.is_multiple_of(16) {
            return Err(TernaryError::Validation {
                message: format!("N must be a multiple of 16 for 2-bit packing, got {}", n),
            });
        }
        if m == 0 {
            return Err(TernaryError::Validation {
                message: "M must be > 0".into(),
            });
        }
        if n == 0 {
            return Err(TernaryError::Validation {
                message: "N must be > 0".into(),
            });
        }

        let packed_cols = n / 16;
        let packed_weights = GpuBuffer::alloc(m * packed_cols)?;
        let output_buffer = GpuBuffer::alloc(m)?;
        let pool = CudaMemoryPool::new();
        let stream = CudaStream::new()?;

        Ok(Self {
            packed_weights,
            output_buffer,
            pool,
            stream,
            m,
            n,
            l2_pinned: false,
        })
    }

    /// Load ternary weights {-1, 0, 1} into the layer.
    pub fn load_weights(&mut self, ternary_weights: &[i8]) -> Result<(), TernaryError> {
        if ternary_weights.len() != self.m * self.n {
            return Err(TernaryError::DimensionMismatch {
                expected: self.m * self.n,
                actual: ternary_weights.len(),
                context: "BitLinear::load_weights".into(),
            });
        }

        let packed = pack_ternary_to_u32(ternary_weights, self.n)?;
        self.packed_weights.copy_from_host(&packed)
    }

    /// Pin weight data in L2 cache for repeated access patterns.
    pub fn pin_weights_in_l2(&mut self) -> Result<(), TernaryError> {
        let byte_size = self.packed_weights.byte_size();
        unsafe {
            ffi::ternary_zero_set_l2_policy(
                self.stream.raw(),
                self.packed_weights.as_ptr() as *const c_void,
                byte_size,
            )
        }
        .to_result()?;
        self.l2_pinned = true;
        Ok(())
    }

    /// Synchronous forward pass.
    ///
    /// Allocates the activation buffer from the memory pool (amortised
    /// zero-cost after the first call), launches the GEMV kernel, and
    /// reads the result back to host memory.
    ///
    /// No explicit `stream.synchronize()` is needed: the synchronous
    /// `cudaMemcpy` D2H copy implicitly waits for all preceding stream
    /// work (including the kernel) via null-stream synchronisation
    /// semantics of blocking streams.
    pub fn forward(&mut self, activations: &[f16]) -> Result<Vec<f16>, TernaryError> {
        if activations.len() != self.n {
            return Err(TernaryError::DimensionMismatch {
                expected: self.n,
                actual: activations.len(),
                context: "BitLinear::forward".into(),
            });
        }

        let mut d_act = self.pool.alloc::<u16>(self.n)?;
        let act_u16: Vec<u16> = activations.iter().map(|h| h.to_bits()).collect();
        d_act.copy_from_host(&act_u16)?;

        unsafe {
            ffi::ternary_zero_gemv_f16(
                self.packed_weights.as_ptr(),
                d_act.as_ptr(),
                self.output_buffer.as_mut_ptr(),
                self.m as i32,
                self.n as i32,
                self.stream.raw(),
            )
        }
        .to_result()?;

        // cudaMemcpy (blocking stream 0) implicitly synchronises with
        // all blocking streams, so the kernel completes before this read.
        let mut output_u16 = vec![0u16; self.m];
        self.output_buffer.copy_to_host(&mut output_u16)?;

        Ok(output_u16
            .iter()
            .map(|&bits| f16::from_bits(bits))
            .collect())
    }

    /// Asynchronous forward pass.
    ///
    /// Pipelines H2D copy → kernel → D2H copy on the CUDA stream
    /// without any host-side synchronisation.  Returns a `PendingResult`
    /// that can be polled (`is_ready()`) or consumed (`get_output()`).
    ///
    /// The activation buffer is held by the `PendingResult` guard to
    /// prevent the pool from reusing it while the kernel is in-flight.
    pub fn forward_async(&mut self, activations: &[f16]) -> Result<PendingResult, TernaryError> {
        if activations.len() != self.n {
            return Err(TernaryError::DimensionMismatch {
                expected: self.n,
                actual: activations.len(),
                context: "BitLinear::forward_async".into(),
            });
        }

        let mut d_act = self.pool.alloc::<u16>(self.n)?;
        let act_u16: Vec<u16> = activations.iter().map(|h| h.to_bits()).collect();
        d_act.copy_from_host_async(&act_u16, &self.stream)?;

        unsafe {
            ffi::ternary_zero_gemv_f16(
                self.packed_weights.as_ptr(),
                d_act.as_ptr(),
                self.output_buffer.as_mut_ptr(),
                self.m as i32,
                self.n as i32,
                self.stream.raw(),
            )
        }
        .to_result()?;

        // Async D2H copy into pinned host memory — no host blocking.
        let mut pinned_output = PinnedHostBuffer::alloc(self.m)?;
        unsafe {
            ffi::cudaMemcpyAsync(
                pinned_output.as_mut_ptr() as *mut c_void,
                self.output_buffer.as_ptr() as *const c_void,
                self.output_buffer.byte_size(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
                self.stream.raw(),
            )
        }
        .to_result()?;

        // Record event after the D2H copy so we can poll for completion.
        let event = CudaEvent::new()?;
        event.record(&self.stream)?;

        Ok(PendingResult {
            host_buffer: pinned_output,
            event,
            activation_buffer: d_act,
            m: self.m,
        })
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.m, self.n)
    }
}

// =====================================================================
// Ternary Packing Utilities
// =====================================================================

/// Pack ternary weights {-1, 0, 1} into 2-bit packed uint32_t format.
///
/// Encoding: -1 -> 10, 0 -> 00, +1 -> 01
/// 16 weights per uint32_t, LSB-first packing.
pub fn pack_ternary_to_u32(weights: &[i8], n: usize) -> Result<Vec<u32>, TernaryError> {
    if !n.is_multiple_of(16) {
        return Err(TernaryError::Validation {
            message: format!("N must be multiple of 16, got {}", n),
        });
    }
    let total = weights.len();
    if !total.is_multiple_of(n) {
        return Err(TernaryError::Validation {
            message: format!("Weight length {} must be multiple of N={}", total, n),
        });
    }
    let m = total / n;
    let packed_cols = n / 16;
    let mut packed = vec![0u32; m * packed_cols];

    for row in 0..m {
        for pc in 0..packed_cols {
            let mut word: u32 = 0;
            for w in 0..16 {
                let idx = row * n + pc * 16 + w;
                let val = weights[idx];
                let bits: u32 = match val {
                    0 => 0b00,
                    1 => 0b01,
                    -1 => 0b10,
                    _ => return Err(TernaryError::InvalidTernaryValue(val)),
                };
                word |= bits << (w * 2);
            }
            packed[row * packed_cols + pc] = word;
        }
    }

    Ok(packed)
}

/// Unpack uint32_t packed ternary weights back to {-1, 0, 1}.
pub fn unpack_u32_to_ternary(packed: &[u32], n: usize) -> Result<Vec<i8>, TernaryError> {
    if n == 0 || !n.is_multiple_of(16) {
        return Err(TernaryError::Validation {
            message: format!("N must be a positive multiple of 16, got {}", n),
        });
    }
    let packed_cols = n / 16;
    if !packed.len().is_multiple_of(packed_cols) {
        return Err(TernaryError::Validation {
            message: format!(
                "Packed length {} must be multiple of N/16={}",
                packed.len(),
                packed_cols
            ),
        });
    }
    let m = packed.len() / packed_cols;
    let mut weights = vec![0i8; m * n];

    for row in 0..m {
        for pc in 0..packed_cols {
            let word = packed[row * packed_cols + pc];
            for w in 0..16 {
                let bits = (word >> (w * 2)) & 0b11;
                let val = match bits {
                    0b00 => 0i8,
                    0b01 => 1i8,
                    0b10 => -1i8,
                    _ => return Err(TernaryError::InvalidBitPattern(bits)),
                };
                weights[row * n + pc * 16 + w] = val;
            }
        }
    }

    Ok(weights)
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        let weights: Vec<i8> = vec![1, 0, -1, 1, 0, 0, -1, 1, 0, 1, -1, 0, 1, -1, 0, 1];
        let n = 16;
        let packed = pack_ternary_to_u32(&weights, n).unwrap();
        assert_eq!(packed.len(), 1);
        let unpacked = unpack_u32_to_ternary(&packed, n).unwrap();
        assert_eq!(weights, unpacked);
    }

    #[test]
    fn test_pack_multiple_rows() {
        let mut weights = Vec::new();
        for _ in 0..64 {
            weights.push(if rand_bit() { 1 } else { 0 });
        }
        let packed = pack_ternary_to_u32(&weights, 32).unwrap();
        assert_eq!(packed.len(), 4);
        let unpacked = unpack_u32_to_ternary(&packed, 32).unwrap();
        assert_eq!(weights, unpacked);
    }

    #[test]
    fn test_invalid_ternary_value() {
        let mut weights = vec![0i8; 16];
        weights[2] = 2; // invalid ternary value
        let result = pack_ternary_to_u32(&weights, 16);
        assert!(result.is_err());
        match result.unwrap_err() {
            TernaryError::InvalidTernaryValue(2) => {}
            other => panic!("Expected InvalidTernaryValue(2), got {:?}", other),
        }
    }

    #[test]
    fn test_invalid_bit_pattern() {
        let packed = vec![0b11u32];
        let result = unpack_u32_to_ternary(&packed, 16);
        assert!(result.is_err());
        match result.unwrap_err() {
            TernaryError::InvalidBitPattern(3) => {}
            other => panic!("Expected InvalidBitPattern(3), got {:?}", other),
        }
    }

    #[test]
    fn test_dimension_mismatch() {
        let weights = vec![0i8, 1, -1]; // not multiple of 16
        let result = pack_ternary_to_u32(&weights, 16);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_display() {
        let err = TernaryError::InvalidTernaryValue(42);
        assert!(format!("{}", err).contains("42"));

        let err = TernaryError::DimensionMismatch {
            expected: 16,
            actual: 3,
            context: "test".into(),
        };
        assert!(format!("{}", err).contains("expected 16"));
        assert!(format!("{}", err).contains("got 3"));
    }

    fn rand_bit() -> bool {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let mut h = DefaultHasher::new();
        COUNTER.fetch_add(1, Ordering::Relaxed).hash(&mut h);
        h.finish() % 2 == 0
    }
}
