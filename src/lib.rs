#![allow(clippy::useless_conversion)]

pub mod bitlinear;
pub mod error;
pub mod ffi;
pub mod ste;

pub use bitlinear::{pack_ternary_to_u32, unpack_u32_to_ternary};
#[cfg(not(no_cuda))]
pub use bitlinear::{
    BitLinear, CudaEvent, CudaMemoryPool, CudaStream, GpuBuffer, PendingResult, PinnedHostBuffer,
    PooledGpuBuffer,
};
pub use error::TernaryError;
#[cfg(not(no_cuda))]
pub use ffi::cuda_error_string;
pub use ffi::CudaError;
pub use ste::{
    dequantize_ternary, ste_backward_activations, ste_backward_weights, ternary_quantize_fixed,
    ternary_quantize_ste,
};

#[cfg(not(no_cuda))]
use half::f16;
use ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
#[cfg(not(no_cuda))]
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[cfg(not(no_cuda))]
impl From<CudaError> for PyErr {
    fn from(err: CudaError) -> PyErr {
        PyRuntimeError::new_err(format!("{}", err))
    }
}

// =====================================================================
// Python Module: _core
// =====================================================================

#[pyfunction]
fn pack_ternary_to_u32_py<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray1<'py, i8>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let w = weights.as_slice()?;
    let packed = pack_ternary_to_u32(w, n)?;
    Ok(PyArray1::from_vec_bound(py, packed))
}

#[pyfunction]
fn unpack_u32_to_ternary_py<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray1<'py, u32>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    let p = packed.as_slice()?;
    let weights = unpack_u32_to_ternary(p, n)?;
    Ok(PyArray1::from_vec_bound(py, weights))
}

#[pyfunction]
fn ternary_quantize_ste_py<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray1<'py, f32>,
    alpha: f32,
) -> PyResult<(Bound<'py, PyArray1<i8>>, f32)> {
    let w_f32 = weights.as_slice()?;
    let w_f16: Vec<half::f16> = w_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
    let (ternary, scale) = ternary_quantize_ste(&w_f16, alpha)?;
    Ok((PyArray1::from_vec_bound(py, ternary), scale))
}

#[pyfunction]
fn ternary_quantize_fixed_py<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray1<'py, f32>,
    threshold: f32,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    let w_f32 = weights.as_slice()?;
    let w_f16: Vec<half::f16> = w_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
    let ternary = ternary_quantize_fixed(&w_f16, threshold);
    Ok(PyArray1::from_vec_bound(py, ternary))
}

#[pyfunction]
fn dequantize_ternary_py<'py>(
    py: Python<'py>,
    ternary_weights: PyReadonlyArray1<'py, i8>,
    scale: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let tw = ternary_weights.as_slice()?;
    let f16_vals = dequantize_ternary(tw, scale);
    let f32_vals: Vec<f32> = f16_vals.iter().map(|h| h.to_f32()).collect();
    Ok(PyArray1::from_vec_bound(py, f32_vals))
}

#[pyfunction]
fn ste_backward_weights_py<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray1<'py, f32>,
    activations: PyReadonlyArray1<'py, f32>,
    raw_weights: PyReadonlyArray1<'py, f32>,
    scale: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let go: Vec<half::f16> = grad_output
        .as_slice()?
        .iter()
        .map(|&v| half::f16::from_f32(v))
        .collect();
    let act: Vec<half::f16> = activations
        .as_slice()?
        .iter()
        .map(|&v| half::f16::from_f32(v))
        .collect();
    let rw: Vec<half::f16> = raw_weights
        .as_slice()?
        .iter()
        .map(|&v| half::f16::from_f32(v))
        .collect();
    let grad_f16 = ste_backward_weights(&go, &act, &rw, scale)?;
    let grad_f32: Vec<f32> = grad_f16.iter().map(|h| h.to_f32()).collect();
    Ok(PyArray1::from_vec_bound(py, grad_f32))
}

#[pyfunction]
fn ste_backward_activations_py<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray1<'py, f32>,
    ternary_weights: PyReadonlyArray1<'py, i8>,
    scale: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let go: Vec<half::f16> = grad_output
        .as_slice()?
        .iter()
        .map(|&v| half::f16::from_f32(v))
        .collect();
    let tw = ternary_weights.as_slice()?;
    let grad_f16 = ste_backward_activations(&go, tw, scale)?;
    let grad_f32: Vec<f32> = grad_f16.iter().map(|h| h.to_f32()).collect();
    Ok(PyArray1::from_vec_bound(py, grad_f32))
}

#[pyfunction]
fn ternary_gemv_cpu<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray1<'py, i8>,
    activations: PyReadonlyArray1<'py, f32>,
    m: usize,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let w = weights.as_slice()?;
    let act = activations.as_slice()?;
    if w.len() != m * n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "weights length {} != M*N = {}",
            w.len(),
            m * n
        )));
    }
    if act.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "activations length {} != N = {}",
            act.len(),
            n
        )));
    }

    let mut output = vec![0.0f32; m];
    for row in 0..m {
        let row_start = row * n;
        let w_row = &w[row_start..row_start + n];
        let mut sum = 0.0f32;
        for (wi, ai) in w_row.iter().zip(act.iter()) {
            sum += *wi as f32 * *ai;
        }
        output[row] = sum;
    }
    Ok(PyArray1::from_vec_bound(py, output))
}

#[pyfunction]
fn ternary_gemm_cpu<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray2<'py, i8>,
    activations: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let w = weights.as_array();
    let act = activations.as_array();
    let m = w.shape()[0];
    let k = w.shape()[1];
    if act.shape()[0] != k {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "inner dimensions must match: weights K={} vs activations K={}",
            k,
            act.shape()[0]
        )));
    }
    let n = act.shape()[1];

    let mut result = Array2::<f32>::zeros((m, n));
    for mi in 0..m {
        for ki in 0..k {
            let wv = w[[mi, ki]] as f32;
            if wv == 0.0 {
                continue;
            }
            for ni in 0..n {
                result[[mi, ni]] += wv * act[[ki, ni]];
            }
        }
    }
    Ok(PyArray2::from_owned_array_bound(py, result))
}

// =====================================================================
// High-Performance Packed Ternary CPU GEMV
// =====================================================================
//
// Operates directly on packed u32 weights (matching the CUDA kernel's
// input format: 16 ternary weights per u32, encoding 00=0, 01=+1, 10=-1).
// Uses a branchless 4-entry lookup table for O(1) ternary decode per
// 2-bit field, eliminating branches and maximizing instruction-level
// parallelism.
//
// This is the critical CPU fallback path for CI/CD runners without GPUs.

/// Branchless ternary decode lookup table.
/// Indexed by 2-bit field: 00->0.0, 01->+1.0, 10->-1.0, 11->0.0 (invalid, treated as zero).
static TERNARY_LUT: [f32; 4] = [0.0, 1.0, -1.0, 0.0];

#[allow(dead_code)]
#[inline(always)]
fn decode_packed_ternary_f32(packed: u32, idx: usize) -> f32 {
    TERNARY_LUT[((packed >> (idx * 2)) & 0b11) as usize]
}

/// High-performance CPU GEMV using packed ternary u32 weights.
///
/// Computes: output[m] = sum_n(decode_ternary(packed_weights[m, n/16]) * activations[n])
///
/// # Arguments
/// * `packed_weights` - Flat array of packed u32 [M * (N/16)]
/// * `activations`   - FP32 activation vector [N]
/// * `m`             - Number of output rows
/// * `n`             - Number of input features (must be multiple of 16)
#[pyfunction]
fn ternary_gemv_cpu_packed<'py>(
    py: Python<'py>,
    packed_weights: PyReadonlyArray1<'py, u32>,
    activations: PyReadonlyArray1<'py, f32>,
    m: usize,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    if !n.is_multiple_of(16) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "N must be multiple of 16 for packed format, got {}",
            n
        )));
    }
    let pw = packed_weights.as_slice()?;
    let act = activations.as_slice()?;
    let packed_cols = n / 16;
    let expected_len = m * packed_cols;
    if pw.len() != expected_len {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "packed_weights length {} != M*(N/16) = {}",
            pw.len(),
            expected_len
        )));
    }
    if act.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "activations length {} != N = {}",
            act.len(),
            n
        )));
    }

    let mut output = vec![0.0f32; m];

    for row in 0..m {
        let row_start = row * packed_cols;
        let mut sum = 0.0f32;

        for pc in 0..packed_cols {
            let word = pw[row_start + pc];
            let act_base = pc * 16;

            let mut w0 = TERNARY_LUT[(word & 0b11) as usize];
            let mut w1 = TERNARY_LUT[((word >> 2) & 0b11) as usize];
            let mut w2 = TERNARY_LUT[((word >> 4) & 0b11) as usize];
            let mut w3 = TERNARY_LUT[((word >> 6) & 0b11) as usize];
            let mut w4 = TERNARY_LUT[((word >> 8) & 0b11) as usize];
            let mut w5 = TERNARY_LUT[((word >> 10) & 0b11) as usize];
            let mut w6 = TERNARY_LUT[((word >> 12) & 0b11) as usize];
            let mut w7 = TERNARY_LUT[((word >> 14) & 0b11) as usize];
            let mut w8 = TERNARY_LUT[((word >> 16) & 0b11) as usize];
            let mut w9 = TERNARY_LUT[((word >> 18) & 0b11) as usize];
            let mut w10 = TERNARY_LUT[((word >> 20) & 0b11) as usize];
            let mut w11 = TERNARY_LUT[((word >> 22) & 0b11) as usize];
            let mut w12 = TERNARY_LUT[((word >> 24) & 0b11) as usize];
            let mut w13 = TERNARY_LUT[((word >> 26) & 0b11) as usize];
            let mut w14 = TERNARY_LUT[((word >> 28) & 0b11) as usize];
            let mut w15 = TERNARY_LUT[((word >> 30) & 0b11) as usize];

            // Branchless zero-skip: multiply by weight value directly.
            // The LUT returns 0.0 for zero weights, so the multiply
            // naturally zeros out the contribution without branching.
            w0 *= act[act_base];
            w1 *= act[act_base + 1];
            w2 *= act[act_base + 2];
            w3 *= act[act_base + 3];
            w4 *= act[act_base + 4];
            w5 *= act[act_base + 5];
            w6 *= act[act_base + 6];
            w7 *= act[act_base + 7];
            w8 *= act[act_base + 8];
            w9 *= act[act_base + 9];
            w10 *= act[act_base + 10];
            w11 *= act[act_base + 11];
            w12 *= act[act_base + 12];
            w13 *= act[act_base + 13];
            w14 *= act[act_base + 14];
            w15 *= act[act_base + 15];

            // Pairwise reduction to minimize FP rounding error
            let s0 = w0 + w1;
            let s1 = w2 + w3;
            let s2 = w4 + w5;
            let s3 = w6 + w7;
            let s4 = w8 + w9;
            let s5 = w10 + w11;
            let s6 = w12 + w13;
            let s7 = w14 + w15;

            let t0 = s0 + s1;
            let t1 = s2 + s3;
            let t2 = s4 + s5;
            let t3 = s6 + s7;

            let u0 = t0 + t1;
            let u1 = t2 + t3;

            sum += u0 + u1;
        }

        output[row] = sum;
    }

    Ok(PyArray1::from_vec_bound(py, output))
}

/// High-performance CPU GEMM using packed ternary u32 weights.
///
/// Computes: output[m, n] = sum_k(decode_ternary(packed_weights[m, k/16]) * activations[k, n])
///
/// # Arguments
/// * `packed_weights` - Flat array of packed u32 [M * (K/16)]
/// * `activations`   - FP32 activation matrix [K, N] (row-major)
/// * `m`             - Number of output rows
/// * `k`             - Number of input features (must be multiple of 16)
/// * `n`             - Number of output columns
#[pyfunction]
fn ternary_gemm_cpu_packed<'py>(
    py: Python<'py>,
    packed_weights: PyReadonlyArray1<'py, u32>,
    activations: PyReadonlyArray2<'py, f32>,
    m: usize,
    k: usize,
    n: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if !k.is_multiple_of(16) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "K must be multiple of 16 for packed format, got {}",
            k
        )));
    }
    let pw = packed_weights.as_slice()?;
    let act = activations.as_array();
    let packed_cols = k / 16;
    let expected_pw_len = m * packed_cols;
    if pw.len() != expected_pw_len {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "packed_weights length {} != M*(K/16) = {}",
            pw.len(),
            expected_pw_len
        )));
    }
    if act.shape() != [k, n] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "activations shape [{}, {}] != [{}, {}]",
            act.shape()[0],
            act.shape()[1],
            k,
            n
        )));
    }

    let mut result = Array2::<f32>::zeros((m, n));

    for row in 0..m {
        let row_start = row * packed_cols;
        for pc in 0..packed_cols {
            let word = pw[row_start + pc];
            let k_base = pc * 16;

            for w_idx in 0..16 {
                let bits = (word >> (w_idx * 2)) & 0b11;
                let w_val = TERNARY_LUT[bits as usize];
                if w_val == 0.0 {
                    continue;
                }
                let ki = k_base + w_idx;
                for ni in 0..n {
                    result[[row, ni]] += w_val * act[[ki, ni]];
                }
            }
        }
    }

    Ok(PyArray2::from_owned_array_bound(py, result))
}

// =====================================================================
// One-Shot GPU GEMV (for Python BitLinear Inference)
// =====================================================================
//
// Allocates GPU buffers, copies data, launches the ternary GEMV kernel,
// and returns the result. For repeated inference calls, the Python layer
// should cache packed weights and call this per-input.

#[cfg(not(no_cuda))]
#[pyfunction]
#[pyo3(signature = (packed_weights, activations, m, n, use_fp32_acc=true))]
fn ternary_gemv_gpu<'py>(
    py: Python<'py>,
    packed_weights: PyReadonlyArray1<'py, u32>,
    activations: PyReadonlyArray1<'py, f32>,
    m: usize,
    n: usize,
    use_fp32_acc: bool,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    if !n.is_multiple_of(16) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "N must be multiple of 16, got {}",
            n
        )));
    }
    let pw = packed_weights.as_slice()?;
    let act = activations.as_slice()?;
    let packed_cols = n / 16;
    let expected_pw = m * packed_cols;
    if pw.len() != expected_pw {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "packed_weights length {} != M*(N/16) = {}",
            pw.len(),
            expected_pw
        )));
    }
    if act.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "activations length {} != N = {}",
            act.len(),
            n
        )));
    }

    let stream = CudaStream::new().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA stream creation failed: {}", e))
    })?;
    let mut d_weights = GpuBuffer::<u32>::alloc(expected_pw).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("GPU alloc weights failed: {}", e))
    })?;
    let mut d_act = GpuBuffer::<u16>::alloc(n).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("GPU alloc activations failed: {}", e))
    })?;
    let mut d_output = GpuBuffer::<u16>::alloc(m).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("GPU alloc output failed: {}", e))
    })?;

    d_weights.copy_from_host(pw).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("H2D weights failed: {}", e))
    })?;
    let act_u16: Vec<u16> = act.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    d_act.copy_from_host(&act_u16).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("H2D activations failed: {}", e))
    })?;

    unsafe {
        ffi::ternary_zero_gemv_f16_ex(
            d_weights.as_ptr(),
            d_act.as_ptr(),
            d_output.as_mut_ptr(),
            m as std::os::raw::c_int,
            n as std::os::raw::c_int,
            stream.raw(),
            use_fp32_acc as std::os::raw::c_int,
        )
    }
    .to_result()
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Kernel launch failed: {}", e))
    })?;

    let mut output_u16 = vec![0u16; m];
    d_output.copy_to_host(&mut output_u16).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("D2H copy failed: {}", e))
    })?;

    let output_f32: Vec<f32> = output_u16
        .iter()
        .map(|&bits| f16::from_bits(bits).to_f32())
        .collect();
    Ok(PyArray1::from_vec_bound(py, output_f32))
}

// =====================================================================
// GPU Kernel Benchmarking Harness (cudaEvent Precision Timing)
// =====================================================================
//
// Uses CUDA event pairs for sub-microsecond kernel execution timing.
// Eliminates host-side overhead from the measurement by recording
// events directly on the GPU timeline.
//
// Returns a Python dict with: min_us, max_us, mean_us, median_us,
// p95_us, num_iterations, m, n, and computed throughput in GFLOP/s.

#[cfg(not(no_cuda))]
#[pyfunction]
#[pyo3(signature = (packed_weights, activations, m, n, warmup=10, iterations=100, use_fp32_acc=true))]
fn benchmark_kernel_gpu<'py>(
    py: Python<'py>,
    packed_weights: PyReadonlyArray1<'py, u32>,
    activations: PyReadonlyArray1<'py, f32>,
    m: usize,
    n: usize,
    warmup: usize,
    iterations: usize,
    use_fp32_acc: bool,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use pyo3::types::PyDict;

    if !n.is_multiple_of(16) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "N must be multiple of 16, got {}",
            n
        )));
    }

    let pw = packed_weights.as_slice()?;
    let act = activations.as_slice()?;
    let packed_cols = n / 16;
    let expected_pw = m * packed_cols;
    if pw.len() != expected_pw {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "packed_weights length {} != M*(N/16) = {}",
            pw.len(),
            expected_pw
        )));
    }
    if act.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "activations length {} != N = {}",
            act.len(),
            n
        )));
    }

    let stream = CudaStream::new().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create CUDA stream: {}", e))
    })?;

    let mut d_weights = GpuBuffer::<u32>::alloc(expected_pw).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to alloc GPU weights: {}", e))
    })?;
    let mut d_act = GpuBuffer::<u16>::alloc(n).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to alloc GPU activations: {}", e))
    })?;
    let mut d_output = GpuBuffer::<u16>::alloc(m).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to alloc GPU output: {}", e))
    })?;

    d_weights.copy_from_host(pw).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("H2D copy weights failed: {}", e))
    })?;

    let act_u16: Vec<u16> = act.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    d_act.copy_from_host(&act_u16).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("H2D copy activations failed: {}", e))
    })?;
    let mut d_act = GpuBuffer::<f32>::alloc(n).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to alloc GPU activations: {}", e))
    })?;
    let mut d_output = GpuBuffer::<f16>::alloc(m).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to alloc GPU output: {}", e))
    })?;

    d_weights.copy_from_host(pw).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("H2D copy weights failed: {}", e))
    })?;

    let act_u16: Vec<u16> = act.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    d_act.copy_from_host(&act_u16).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("H2D copy activations failed: {}", e))
    })?;

    let start_event = CudaEvent::new().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create start event: {}", e))
    })?;
    let end_event = CudaEvent::new().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create end event: {}", e))
    })?;

    for _ in 0..warmup {
        unsafe {
            ffi::ternary_zero_gemv_f16_ex(
                d_weights.as_ptr(),
                d_act.as_ptr(),
                d_output.as_mut_ptr(),
                m as std::os::raw::c_int,
                n as std::os::raw::c_int,
                stream.raw(),
                use_fp32_acc as std::os::raw::c_int,
            )
        }
        .to_result()
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Warmup kernel launch failed: {}", e))
        })?;
    }
    stream.synchronize().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Warmup sync failed: {}", e))
    })?;

    let mut timings_ms = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        start_event.record(&stream).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Event record failed: {}", e))
        })?;

        unsafe {
            ffi::ternary_zero_gemv_f16_ex(
                d_weights.as_ptr(),
                d_act.as_ptr(),
                d_output.as_mut_ptr(),
                m as std::os::raw::c_int,
                n as std::os::raw::c_int,
                stream.raw(),
                use_fp32_acc as std::os::raw::c_int,
            )
        }
        .to_result()
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Kernel launch failed: {}", e))
        })?;

        end_event.record(&stream).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Event record failed: {}", e))
        })?;
        end_event.synchronize().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Event sync failed: {}", e))
        })?;

        let mut elapsed_ms: f32 = 0.0;
        let err = unsafe {
            ffi::cudaEventElapsedTime(
                &mut elapsed_ms as *mut f32,
                start_event.raw(),
                end_event.raw(),
            )
        };
        if err != CudaError::Success {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "cudaEventElapsedTime failed: {}",
                err
            )));
        }
        timings_ms.push(elapsed_ms);
    }

    timings_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min_ms = timings_ms[0];
    let max_ms = timings_ms[timings_ms.len() - 1];
    let mean_ms: f32 = timings_ms.iter().sum::<f32>() / timings_ms.len() as f32;
    let median_ms = timings_ms[timings_ms.len() / 2];
    let p95_idx = ((timings_ms.len() as f32) * 0.95) as usize;
    let p95_ms = timings_ms[p95_idx.min(timings_ms.len() - 1)];

    let nnz_estimate = (m * n) as f32 * 0.5;
    let flops_per_iter = 2.0 * nnz_estimate;
    let mean_s = mean_ms / 1000.0;
    let gflops = if mean_s > 0.0 {
        flops_per_iter / mean_s / 1e9
    } else {
        0.0
    };

    let dict = PyDict::new(py);
    dict.set_item("min_us", min_ms * 1000.0)?;
    dict.set_item("max_us", max_ms * 1000.0)?;
    dict.set_item("mean_us", mean_ms * 1000.0)?;
    dict.set_item("median_us", median_ms * 1000.0)?;
    dict.set_item("p95_us", p95_ms * 1000.0)?;
    dict.set_item("min_ms", min_ms)?;
    dict.set_item("max_ms", max_ms)?;
    dict.set_item("mean_ms", mean_ms)?;
    dict.set_item("median_ms", median_ms)?;
    dict.set_item("p95_ms", p95_ms)?;
    dict.set_item("num_iterations", iterations)?;
    dict.set_item("warmup", warmup)?;
    dict.set_item("m", m)?;
    dict.set_item("n", n)?;
    dict.set_item("gflops", gflops)?;
    dict.set_item("use_fp32_acc", use_fp32_acc)?;
    Ok(dict)
}

#[cfg(not(no_cuda))]
#[pyfunction]
fn has_cuda() -> bool {
    let err = unsafe { ffi::cudaGetLastError() };
    err == CudaError::Success || err == CudaError::InitializationError
}

#[cfg(no_cuda)]
#[pyfunction]
fn has_cuda() -> bool {
    false
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pack_ternary_to_u32_py, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_u32_to_ternary_py, m)?)?;
    m.add_function(wrap_pyfunction!(ternary_quantize_ste_py, m)?)?;
    m.add_function(wrap_pyfunction!(ternary_quantize_fixed_py, m)?)?;
    m.add_function(wrap_pyfunction!(dequantize_ternary_py, m)?)?;
    m.add_function(wrap_pyfunction!(ste_backward_weights_py, m)?)?;
    m.add_function(wrap_pyfunction!(ste_backward_activations_py, m)?)?;
    m.add_function(wrap_pyfunction!(ternary_gemv_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(ternary_gemm_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(ternary_gemv_cpu_packed, m)?)?;
    m.add_function(wrap_pyfunction!(ternary_gemm_cpu_packed, m)?)?;
    #[cfg(not(no_cuda))]
    m.add_function(wrap_pyfunction!(ternary_gemv_gpu, m)?)?;
    #[cfg(not(no_cuda))]
    m.add_function(wrap_pyfunction!(benchmark_kernel_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(has_cuda, m)?)?;
    Ok(())
}
