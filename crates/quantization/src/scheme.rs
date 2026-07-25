use bitllm_tensor::DType;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantScheme {
    AbsMax,
    Symmetric,
    Asymmetric,
    GroupSymmetric,
    GroupAsymmetric,
    Ternary,
}

impl QuantScheme {
    pub fn dtype(&self) -> DType {
        match self {
            QuantScheme::AbsMax | QuantScheme::Symmetric | QuantScheme::Asymmetric => DType::INT8,
            QuantScheme::GroupSymmetric | QuantScheme::GroupAsymmetric => DType::INT4,
            QuantScheme::Ternary => DType::BIT1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantConfig {
    pub scheme: QuantScheme,
    pub bits: usize,
    pub group_size: Option<usize>,
    pub symmetric: bool,
}

impl QuantConfig {
    pub fn int8() -> Self {
        Self {
            scheme: QuantScheme::AbsMax,
            bits: 8,
            group_size: None,
            symmetric: true,
        }
    }

    pub fn int4() -> Self {
        Self {
            scheme: QuantScheme::GroupSymmetric,
            bits: 4,
            group_size: Some(128),
            symmetric: true,
        }
    }

    pub fn int4_group(group_size: usize) -> Self {
        Self {
            scheme: QuantScheme::GroupSymmetric,
            bits: 4,
            group_size: Some(group_size),
            symmetric: true,
        }
    }

    pub fn ternary() -> Self {
        Self {
            scheme: QuantScheme::Ternary,
            bits: 1,
            group_size: None,
            symmetric: true,
        }
    }

    pub fn target_dtype(&self) -> DType {
        match self.bits {
            8 => DType::INT8,
            4 => DType::INT4,
            1 => DType::BIT1,
            _ => DType::F32,
        }
    }

    pub fn compression_ratio(&self) -> f64 {
        32.0 / self.bits as f64
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedTensor {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub scales: Vec<f32>,
    pub zeros: Option<Vec<f32>>,
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
        let quantized_bytes = self.data.len() + self.meta_bytes();
        original_bytes as f64 / quantized_bytes.max(1) as f64
    }

    fn meta_bytes(&self) -> usize {
        self.scales_bytes() + self.zeros.as_ref().map_or(0, |z| z.len() * 4)
    }

    fn scales_bytes(&self) -> usize {
        self.scales.len() * 4
    }
}
