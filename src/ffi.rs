#![allow(non_camel_case_types)]

use std::fmt;
use std::os::raw::{c_int, c_void};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaError {
    Success = 0,
    InvalidValue = 1,
    MemoryAllocation = 2,
    InitializationError = 3,
    CudartUnloading = 4,
    ProfilerDisabled = 5,
    ProfilerNotInitialized = 6,
    ProfilerAlreadyStarted = 7,
    ProfilerAlreadyStopped = 8,
    InvalidDevice = 100,
    InvalidImage = 200,
    InvalidContext = 201,
    ContextAlreadyCurrent = 202,
    MapFailed = 205,
    UnmapFailed = 206,
    ArrayIsMapped = 207,
    AlreadyMapped = 208,
    NoBinaryForGpu = 209,
    AlreadyAcquired = 210,
    NotMapped = 211,
    NotMappedAsArray = 212,
    NotMappedAsPointer = 213,
    EccUncorrectable = 214,
    UnsupportedLimit = 215,
    ContextAlreadyInUse = 216,
    PeerAccessUnsupported = 217,
    InvalidPtx = 218,
    InvalidGraphicsContext = 219,
    StartupFailure = 127,
    ApiFailureBase = 10000,
    Unknown = 99999,
}

impl CudaError {
    pub fn to_result(self) -> Result<(), CudaError> {
        if self == CudaError::Success {
            Ok(())
        } else {
            Err(self)
        }
    }

    pub fn description(&self) -> &str {
        match self {
            CudaError::Success => "no error",
            CudaError::InvalidValue => "invalid argument",
            CudaError::MemoryAllocation => "out of memory",
            CudaError::InitializationError => "initialization error",
            CudaError::CudartUnloading => "CUDA runtime shutting down",
            CudaError::ProfilerDisabled => "profiler disabled",
            CudaError::ProfilerNotInitialized => "profiler not initialized",
            CudaError::ProfilerAlreadyStarted => "profiler already started",
            CudaError::ProfilerAlreadyStopped => "profiler already stopped",
            CudaError::InvalidDevice => "invalid device ordinal",
            CudaError::InvalidImage => "invalid kernel image",
            CudaError::InvalidContext => "invalid device context",
            CudaError::ContextAlreadyCurrent => "context already current",
            CudaError::MapFailed => "mapping failed",
            CudaError::UnmapFailed => "unmapping failed",
            CudaError::ArrayIsMapped => "array is mapped",
            CudaError::AlreadyMapped => "resource already mapped",
            CudaError::NoBinaryForGpu => "no binary for GPU",
            CudaError::AlreadyAcquired => "resource already acquired",
            CudaError::NotMapped => "resource not mapped",
            CudaError::NotMappedAsArray => "not mapped as array",
            CudaError::NotMappedAsPointer => "not mapped as pointer",
            CudaError::EccUncorrectable => "uncorrectable ECC error",
            CudaError::UnsupportedLimit => "unsupported limit",
            CudaError::ContextAlreadyInUse => "context already in use",
            CudaError::PeerAccessUnsupported => "peer access unsupported",
            CudaError::InvalidPtx => "invalid PTX",
            CudaError::InvalidGraphicsContext => "invalid graphics context",
            CudaError::StartupFailure => "CUDA driver startup failure",
            CudaError::ApiFailureBase => "API failure base",
            CudaError::Unknown => "unknown error",
        }
    }
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CUDA error {:?} (code {}): {}", self, *self as i32, self.description())
    }
}

impl std::error::Error for CudaError {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUstream_st {
    _private: [u8; 0],
}

pub type cudaStream_t = *mut CUstream_st;

// Copy kind constants
pub const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

// Stream attribute enum value for access policy window
pub const CUDA_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW: c_int = 1;

// Access property types
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum cudaAccessProperty {
    Normal = 0,
    Persisting = 1,
    Streaming = 2,
    Num,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cudaAccessPolicyWindow {
    pub base_ptr: *mut c_void,
    pub num_bytes: usize,
    pub hit_ratio: f32,
    pub hit_prop: cudaAccessProperty,
    pub miss_prop: cudaAccessProperty,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union cudaStreamAttrValue {
    pub access_policy_window: cudaAccessPolicyWindow,
    _reserved: [u8; 64],
}

// =====================================================================
// External CUDA Runtime Functions
// =====================================================================

#[link(name = "cudart_static")]
extern "C" {
    pub fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> CudaError;
    pub fn cudaFree(dev_ptr: *mut c_void) -> CudaError;
    pub fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
    ) -> CudaError;
    pub fn cudaStreamCreate(stream: *mut cudaStream_t) -> CudaError;
    pub fn cudaStreamDestroy(stream: cudaStream_t) -> CudaError;
    pub fn cudaStreamSynchronize(stream: cudaStream_t) -> CudaError;
    pub fn cudaStreamSetAttribute(
        stream: cudaStream_t,
        attr: c_int,
        value: *const cudaStreamAttrValue,
    ) -> CudaError;
    pub fn cudaGetLastError() -> CudaError;
    pub fn cudaPeekAtLastError() -> CudaError;
    pub fn cudaGetErrorString(error: CudaError) -> *const i8;
}

// =====================================================================
// External Ternary-Zero Kernel Function
// =====================================================================

extern "C" {
    pub fn ternary_zero_gemv_f16(
        weights: *const u32,
        activations: *const u16, // __half represented as u16
        output: *mut u16,
        m: c_int,
        n: c_int,
        stream: cudaStream_t,
    ) -> CudaError;

    pub fn ternary_zero_set_l2_policy(
        stream: cudaStream_t,
        base_ptr: *const c_void,
        num_bytes: usize,
    ) -> CudaError;
}

// =====================================================================
// Helper: Get CUDA error string
// =====================================================================

pub fn cuda_error_string(err: CudaError) -> String {
    unsafe {
        let ptr = cudaGetErrorString(err);
        if ptr.is_null() {
            return format!("Unknown CUDA error ({:?})", err);
        }
        std::ffi::CStr::from_ptr(ptr)
            .to_string_lossy()
            .into_owned()
    }
}
