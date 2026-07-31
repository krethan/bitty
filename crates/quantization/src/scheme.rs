use bitllm_tensor::DType;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantScheme {
    Ternary,
}

impl QuantScheme {
    pub fn dtype(&self) -> DType {
        DType::BIT1
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantConfig {
    pub scheme: QuantScheme,
}

impl QuantConfig {
    pub fn ternary() -> Self {
        Self {
            scheme: QuantScheme::Ternary,
        }
    }

    pub fn target_dtype(&self) -> DType {
        DType::BIT1
    }

    pub fn compression_ratio(&self) -> f64 {
        32.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedTensor {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub scales: Vec<f32>,
    pub config: QuantConfig,
}

impl QuantizedTensor {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn nbytes(&self) -> usize {
        self.data.len()
    }

    pub fn compression_ratio(&self) -> f64 {
        let original_bytes = self.num_elements() * 4;
        let quantized_bytes = self.data.len() + self.scales.len() * 4;
        original_bytes as f64 / quantized_bytes.max(1) as f64
    }
}
