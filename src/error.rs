use thiserror::Error;

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

pub type Result<T> = std::result::Result<T, Error>;
