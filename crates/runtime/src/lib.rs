pub mod attention;
pub mod bitlinear;
pub mod config;
pub mod gguf;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod layers;
pub mod loader;
pub mod model;
pub mod sampler;

#[cfg(feature = "gpu")]
pub use gpu::GpuContext;

#[cfg(not(feature = "gpu"))]
#[derive(Debug, Clone)]
pub struct GpuContext {
    device_id: i32,
}

#[cfg(not(feature = "gpu"))]
impl GpuContext {
    pub fn new(device_id: i32) -> Result<Self, String> {
        let _ = device_id;
        Err("GPU support not compiled (enable `gpu` feature)".into())
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }
}

pub use bitlinear::BitLinear;
pub use config::ModelConfig;
pub use loader::{load_safetensors_weights, LlamaWeightMapper, LoadingStats, SafeTensorsLoader, WeightTarget};
pub use model::Model;
pub use sampler::{Sampler, SamplingStrategy};
