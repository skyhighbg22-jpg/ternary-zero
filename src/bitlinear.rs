use crate::ffi::{
    self, CudaError, CUDA_MEMCPY_DEVICE_TO_HOST, CUDA_MEMCPY_HOST_TO_DEVICE,
    cudaStream_t,
};
use half::f16;
use std::os::raw::c_void;
use std::ptr;

// =====================================================================
// GPU Buffer: RAII Wrapper for CUDA Device Memory
// =====================================================================

pub struct GpuBuffer<T: Copy> {
    ptr: *mut T,
    len: usize,
}

impl<T: Copy> GpuBuffer<T> {
    /// Allocate `len` elements on the GPU
    pub fn alloc(len: usize) -> Result<Self, CudaError> {
        if len == 0 {
            return Ok(Self {
                ptr: ptr::null_mut(),
                len: 0,
            });
        }

        let mut dev_ptr: *mut c_void = ptr::null_mut();
        let size = len * std::mem::size_of::<T>();

        let err = unsafe { ffi::cudaMalloc(&mut dev_ptr as *mut *mut c_void, size) };
        err.to_result()?;

        Ok(Self {
            ptr: dev_ptr as *mut T,
            len,
        })
    }

    /// Copy data from host slice to GPU
    pub fn copy_from_host(&mut self, data: &[T]) -> Result<(), CudaError> {
        assert_eq!(
            data.len(),
            self.len,
            "copy_from_host: data length {} != buffer length {}",
            data.len(),
            self.len
        );

        if self.len == 0 {
            return Ok(());
        }

        let size = self.len * std::mem::size_of::<T>();
        let err = unsafe {
            ffi::cudaMemcpy(
                self.ptr as *mut c_void,
                data.as_ptr() as *const c_void,
                size,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        err.to_result()
    }

    /// Copy data from GPU to host slice
    pub fn copy_to_host(&self, data: &mut [T]) -> Result<(), CudaError> {
        assert_eq!(
            data.len(),
            self.len,
            "copy_to_host: data length {} != buffer length {}",
            data.len(),
            self.len
        );

        if self.len == 0 {
            return Ok(());
        }

        let size = self.len * std::mem::size_of::<T>();
        let err = unsafe {
            ffi::cudaMemcpy(
                data.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                size,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        err.to_result()
    }

    /// Get raw device pointer
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Get raw mutable device pointer
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Number of elements
    pub fn len(&self) -> usize {
        self.len
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Size in bytes
    pub fn byte_size(&self) -> usize {
        self.len * std::mem::size_of::<T>()
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

// Safety: GpuBuffer owns unique device memory and the CUDA runtime
// is thread-safe for memory operations from different host threads.
unsafe impl<T: Copy> Send for GpuBuffer<T> {}
unsafe impl<T: Copy> Sync for GpuBuffer<T> {}

// =====================================================================
// CUDA Stream: RAII Wrapper
// =====================================================================

pub struct CudaStream(pub cudaStream_t);

impl CudaStream {
    pub fn new() -> Result<Self, CudaError> {
        let mut stream: cudaStream_t = ptr::null_mut();
        let err = unsafe { ffi::cudaStreamCreate(&mut stream) };
        err.to_result()?;
        Ok(Self(stream))
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        let err = unsafe { ffi::cudaStreamSynchronize(self.0) };
        err.to_result()
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

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

// =====================================================================
// BitLinear Layer
// =====================================================================
//
// Encapsulates ternary-weight GEMV with GPU buffer management.
// Stores packed 2-bit weights and manages the forward pass lifecycle.

pub struct BitLinear {
    packed_weights: GpuBuffer<u32>,
    output_buffer: GpuBuffer<u16>,
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
    /// * `n` - Number of input features (columns of weight matrix, must be multiple of 16)
    pub fn new(m: usize, n: usize) -> Result<Self, CudaError> {
        assert!(n % 16 == 0, "N must be a multiple of 16 for 2-bit packing, got {}", n);
        assert!(m > 0, "M must be > 0");
        assert!(n > 0, "N must be > 0");

        let packed_cols = n / 16;
        let packed_weights = GpuBuffer::alloc(m * packed_cols)?;
        let output_buffer = GpuBuffer::alloc(m)?;
        let stream = CudaStream::new()?;

        Ok(Self {
            packed_weights,
            output_buffer,
            stream,
            m,
            n,
            l2_pinned: false,
        })
    }

    /// Load ternary weights {-1, 0, 1} into the layer.
    ///
    /// # Arguments
    /// * `ternary_weights` - Flat array of i8 values in {-1, 0, 1}, length M*N
    pub fn load_weights(&mut self, ternary_weights: &[i8]) -> Result<(), CudaError> {
        assert_eq!(
            ternary_weights.len(),
            self.m * self.n,
            "Weight array length {} != M*N = {}",
            ternary_weights.len(),
            self.m * self.n
        );

        let packed = pack_ternary_to_u32(ternary_weights, self.n);
        self.packed_weights.copy_from_host(&packed)
    }

    /// Set L2 cache persistence policy for weight data.
    /// Pins weight tiles in L2 for repeated access patterns.
    pub fn pin_weights_in_l2(&mut self) -> Result<(), CudaError> {
        let byte_size = self.packed_weights.byte_size();
        let err = unsafe {
            ffi::ternary_zero_set_l2_policy(
                self.stream.raw(),
                self.packed_weights.as_ptr() as *const c_void,
                byte_size,
            )
        };
        err.to_result()?;
        self.l2_pinned = true;
        Ok(())
    }

    /// Forward pass: compute output = W_ternary * activations
    ///
    /// # Arguments
    /// * `activations` - FP16 activation vector of length N
    ///
    /// # Returns
    /// Vec of FP16 output values of length M
    pub fn forward(&mut self, activations: &[f16]) -> Result<Vec<f16>, CudaError> {
        assert_eq!(
            activations.len(),
            self.n,
            "Activation length {} != N = {}",
            activations.len(),
            self.n
        );

        // Upload activations to GPU
        let mut d_act: GpuBuffer<u16> = GpuBuffer::alloc(self.n)?;
        let act_u16: Vec<u16> = activations.iter().map(|h| h.to_bits()).collect();
        d_act.copy_from_host(&act_u16)?;

        // Launch kernel
        let err = unsafe {
            ffi::ternary_zero_gemv_f16(
                self.packed_weights.as_ptr(),
                d_act.as_ptr(),
                self.output_buffer.as_mut_ptr(),
                self.m as i32,
                self.n as i32,
                self.stream.raw(),
            )
        };
        err.to_result()?;

        // Synchronize and read back
        self.stream.synchronize()?;

        let mut output_u16 = vec![0u16; self.m];
        self.output_buffer.copy_to_host(&mut output_u16)?;

        let output: Vec<f16> = output_u16.iter().map(|&bits| f16::from_bits(bits)).collect();
        Ok(output)
    }

    /// Get layer dimensions (M, N)
    pub fn dimensions(&self) -> (usize, usize) {
        (self.m, self.n)
    }
}

// =====================================================================
// Ternary Packing Utility
// =====================================================================

/// Pack ternary weights {-1, 0, 1} into 2-bit packed uint32_t format.
///
/// Encoding: -1 -> 10, 0 -> 00, +1 -> 01
/// 16 weights per uint32_t, LSB-first packing.
///
/// # Arguments
/// * `weights` - Flat array of i8 values in {-1, 0, 1}, length M*N
/// * `n` - Number of columns (input features)
///
/// # Returns
/// Packed uint32_t array of length M * (N/16)
pub fn pack_ternary_to_u32(weights: &[i8], n: usize) -> Vec<u32> {
    assert!(n % 16 == 0, "N must be multiple of 16");
    let total = weights.len();
    assert!(total % n == 0, "Weight length must be multiple of N");
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
                    _ => panic!("Invalid ternary value: {}. Must be -1, 0, or 1", val),
                };
                word |= bits << (w * 2);
            }
            packed[row * packed_cols + pc] = word;
        }
    }

    packed
}

/// Unpack uint32_t packed ternary weights back to {-1, 0, 1}.
pub fn unpack_u32_to_ternary(packed: &[u32], n: usize) -> Vec<i8> {
    let packed_cols = n / 16;
    assert!(packed.len() % packed_cols == 0);
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
                    _ => panic!("Invalid 2-bit pattern: {:02b}", bits),
                };
                weights[row * n + pc * 16 + w] = val;
            }
        }
    }

    weights
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
        let packed = pack_ternary_to_u32(&weights, n);
        assert_eq!(packed.len(), 1);
        let unpacked = unpack_u32_to_ternary(&packed, n);
        assert_eq!(weights, unpacked);
    }

    #[test]
    fn test_pack_multiple_rows() {
        // 2 rows, 32 columns each
        let mut weights = Vec::new();
        for _ in 0..64 {
            weights.push(if rand_bit() { 1 } else { 0 });
        }
        let packed = pack_ternary_to_u32(&weights, 32);
        assert_eq!(packed.len(), 4); // 2 rows * (32/16) = 4
        let unpacked = unpack_u32_to_ternary(&packed, 32);
        assert_eq!(weights, unpacked);
    }

    fn rand_bit() -> bool {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        static mut COUNTER: u64 = 0;
        let mut h = DefaultHasher::new();
        unsafe {
            COUNTER += 1;
            COUNTER.hash(&mut h);
        }
        h.finish() % 2 == 0
    }
}
