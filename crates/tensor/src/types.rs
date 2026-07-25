use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    INT8,
    INT4,
    BIT1,
}

impl DType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::INT8 => 1,
            DType::INT4 => 1,
            DType::BIT1 => 1,
        }
    }

    pub fn bit_width(&self) -> usize {
        match self {
            DType::F32 => 32,
            DType::F16 => 16,
            DType::BF16 => 16,
            DType::INT8 => 8,
            DType::INT4 => 4,
            DType::BIT1 => 1,
        }
    }

    pub fn elems_per_byte(&self) -> usize {
        match self {
            DType::BIT1 => 8,
            DType::INT4 => 2,
            _ => 1,
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, DType::INT4 | DType::BIT1)
    }

    pub fn is_float(&self) -> bool {
        matches!(self, DType::F32 | DType::F16 | DType::BF16)
    }

    pub fn name(&self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::INT8 => "i8",
            DType::INT4 => "i4",
            DType::BIT1 => "bit1",
        }
    }
}
