use std::fmt;

use crate::ffi::CudaError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;

/// Unified error type for all ternary-zero operations.
///
/// Replaces panic-based validation with recoverable errors that are
/// safe for FFI and Python bindings.
#[derive(Debug, Clone)]
pub enum TernaryError {
    /// CUDA runtime error from device operations.
    Cuda(CudaError),
    /// Input validation failure (e.g., non-positive dimensions, bad alpha).
    Validation { message: String },
    /// Arithmetic overflow in size calculations.
    Overflow { context: String },
    /// Buffer dimension mismatch (expected vs actual with context).
    DimensionMismatch {
        expected: usize,
        actual: usize,
        context: String,
    },
    /// Ternary weight value outside {-1, 0, 1}.
    InvalidTernaryValue(i8),
    /// 2-bit pattern outside {00, 01, 10} during unpacking.
    InvalidBitPattern(u32),
}

impl fmt::Display for TernaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TernaryError::Cuda(e) => write!(f, "{}", e),
            TernaryError::Validation { message } => {
                write!(f, "validation error: {}", message)
            }
            TernaryError::Overflow { context } => {
                write!(f, "arithmetic overflow in {}", context)
            }
            TernaryError::DimensionMismatch {
                expected,
                actual,
                context,
            } => {
                write!(f, "{}: expected {}, got {}", context, expected, actual)
            }
            TernaryError::InvalidTernaryValue(v) => {
                write!(f, "invalid ternary value: {}. Must be -1, 0, or 1", v)
            }
            TernaryError::InvalidBitPattern(bits) => {
                write!(
                    f,
                    "invalid 2-bit pattern: {:02b}. Expected 00, 01, or 10",
                    bits
                )
            }
        }
    }
}

impl std::error::Error for TernaryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TernaryError::Cuda(e) => Some(e),
            _ => None,
        }
    }
}

impl From<CudaError> for TernaryError {
    fn from(err: CudaError) -> Self {
        TernaryError::Cuda(err)
    }
}

impl From<TernaryError> for PyErr {
    fn from(err: TernaryError) -> PyErr {
        PyRuntimeError::new_err(format!("{}", err))
    }
}
