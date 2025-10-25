use thiserror::Error;

#[cfg(feature = "python-bindings")]
use pyo3::PyErr;
#[cfg(feature = "python-bindings")]
use pyo3::exceptions::PyRuntimeError;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Casting from types failed")]
    CastFailed,
    #[error("No image data have been provided")]
    NoData,
    #[error("Dimensions between image and mask don't match.")]
    DimensionMismatch,
    #[error("Heap pop failed as it does not contain data.")]
    HeapDoesNotContainData,
    #[error("NDArray had an error during initializaiton of shape: {0}")]
    NDArray(#[from] ndarray::ShapeError),
    #[error("{0}")]
    Custom(String),
}

#[cfg(feature = "python-bindings")]
impl std::convert::From<Error> for PyErr {
    fn from(err: Error) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
