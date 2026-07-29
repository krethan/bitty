#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
mod scalar;

// On x86_64: avx2.rs has its own runtime is_x86_feature_detected! dispatch
// that falls back to scalar inline. On non-x86_64, just use scalar.
#[cfg(target_arch = "x86_64")]
pub use avx2::*;

#[cfg(not(target_arch = "x86_64"))]
pub use scalar::*;
