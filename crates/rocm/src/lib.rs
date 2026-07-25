pub mod device;
pub mod memory;
pub mod ops;

pub use device::{Device, DeviceInfo};
pub use memory::GpuBuffer;
pub use ops::GpuOps;

#[derive(Debug, thiserror::Error)]
pub enum RocmError {
    #[error("ROCm not available (compile with `rocm` feature)")]
    NotAvailable,

    #[error("No ROCm devices found")]
    NoDevices,

    #[error("HIP error: {0}")]
    HipError(String),

    #[error("Memory allocation failed: {0} bytes")]
    AllocationFailed(usize),

    #[error("Kernel launch failed: {0}")]
    KernelLaunchFailed(String),

    #[error("Memory transfer failed: {0}")]
    TransferFailed(String),

    #[error("Device mismatch: operation requires same device")]
    DeviceMismatch,
}

pub type Result<T> = std::result::Result<T, RocmError>;

pub fn is_available() -> bool {
    #[cfg(feature = "rocm")]
    {
        device::detect_devices().is_ok()
    }
    #[cfg(not(feature = "rocm"))]
    {
        false
    }
}
