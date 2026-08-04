//! Bitty 1-bit training primitives.
//!
//! [`TernaryLoRA`] trains low-rank adapters whose weights stay in the repo's
//! packed ternary ("1-bit") format via straight-through gradients and
//! stochastic trit flips — with no F32 shadow weights in the training loop.

pub mod lora;
pub mod qat;
pub mod training;

pub use lora::{TernaryLoRA, TernaryLoRAConfig};
pub use qat::{QATConfig, QATModel, mean_sq_error, ste_grad};
pub use training::{StochasticFlip, TrainingConfig};
