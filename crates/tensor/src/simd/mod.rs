#[cfg(target_arch = "x86_64")]
mod avx2;

#[cfg(target_arch = "x86_64")]
pub use avx2::*;

#[cfg(not(target_arch = "x86_64"))]
mod scalar;

#[cfg(not(target_arch = "x86_64"))]
pub use scalar::*;
