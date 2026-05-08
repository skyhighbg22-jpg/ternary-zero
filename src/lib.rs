pub mod error;
pub mod ffi;
pub mod bitlinear;
pub mod ste;

pub use bitlinear::{
    BitLinear, CudaEvent, CudaMemoryPool, CudaStream, GpuBuffer, PendingResult,
    PinnedHostBuffer, PooledGpuBuffer, pack_ternary_to_u32, unpack_u32_to_ternary,
};
pub use error::TernaryError;
pub use ffi::{CudaError, cuda_error_string};
pub use ste::{
    dequantize_ternary, ste_backward_activations, ste_backward_weights,
    ternary_quantize_fixed, ternary_quantize_ste,
};

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::Array2;

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
    let go: Vec<half::f16> = grad_output.as_slice()?.iter().map(|&v| half::f16::from_f32(v)).collect();
    let act: Vec<half::f16> = activations.as_slice()?.iter().map(|&v| half::f16::from_f32(v)).collect();
    let rw: Vec<half::f16> = raw_weights.as_slice()?.iter().map(|&v| half::f16::from_f32(v)).collect();
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
    let go: Vec<half::f16> = grad_output.as_slice()?.iter().map(|&v| half::f16::from_f32(v)).collect();
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
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("weights length {} != M*N = {}", w.len(), m * n),
        ));
    }
    if act.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("activations length {} != N = {}", act.len(), n),
        ));
    }

    let mut output = vec![0.0f32; m];
    for row in 0..m {
        let mut sum = 0.0f32;
        for col in 0..n {
            sum += w[row * n + col] as f32 * act[col];
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
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("inner dimensions must match: weights K={} vs activations K={}", k, act.shape()[0]),
        ));
    }
    let n = act.shape()[1];

    let mut result = Array2::<f32>::zeros((m, n));
    for mi in 0..m {
        for ni in 0..n {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += w[[mi, ki]] as f32 * act[[ki, ni]];
            }
            result[[mi, ni]] = sum;
        }
    }
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
fn has_cuda() -> bool {
    let err = unsafe { ffi::cudaGetLastError() };
    err == CudaError::Success || err == CudaError::InitializationError
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
    m.add_function(wrap_pyfunction!(has_cuda, m)?)?;
    Ok(())
}
