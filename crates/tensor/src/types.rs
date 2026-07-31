use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    F32,
    BIT1,
}

impl DType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::BIT1 => 1,
        }
    }

    pub fn bit_width(&self) -> usize {
        match self {
            DType::F32 => 32,
            DType::BIT1 => 1,
        }
    }

    pub fn elems_per_byte(&self) -> usize {
        match self {
            DType::BIT1 => 8,
            _ => 1,
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, DType::BIT1)
    }

    pub fn is_float(&self) -> bool {
        matches!(self, DType::F32)
    }

    pub fn name(&self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::BIT1 => "bit1",
        }
    }
}
