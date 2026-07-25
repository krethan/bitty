use thiserror::Error;

#[derive(Error, Debug)]
pub enum TensorError {
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),

    #[error("Invalid shape: {0}")]
    InvalidShape(String),

    #[error("Out of bounds access")]
    OutOfBounds,

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Index error: {0}")]
    IndexError(String),

    #[error("Conversion error: {0}")]
    ConversionError(String),
}
