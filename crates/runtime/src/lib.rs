pub mod attention;
pub mod bitlinear;
pub mod bittransformer;
pub mod config;
pub mod continuous;
pub mod gguf;
pub mod scheduler;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod layers;
pub mod loader;
pub mod model;
pub mod record;
pub mod sampler;
pub mod speculative;

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
pub use bittransformer::{BitAttention, BitTransformerLayer};
pub use config::{Activation, Architecture, ModelConfig, NormType};
pub use loader::{
    apply_weight_target, load_safetensors_weights, map_weight_for_architecture, CustomWeightMapper,
    GemmaWeightMapper, Gpt2WeightMapper, LlamaWeightMapper, LoadingStats, MistralWeightMapper,
    PhiWeightMapper, QwenWeightMapper, SafeTensorsLoader, WeightMapper, WeightTarget,
};
pub use model::Model;
pub use sampler::{Sampler, SamplingStrategy};