use bitllm_tensor::DType;
use serde::{Deserialize, Serialize};
use std::mem::size_of;

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
    /// Fraction of the largest-magnitude weight elements kept exact (f32),
    /// stored separately in [`QuantizedTensor::outliers`] and added back as a
    /// dense correction at matmul time. `0.0` disables outlier channels.
    #[serde(default)]
    pub outlier_frac: f64,
    /// Number of consecutive weight columns (along the reduction dim `k`)
    /// that share a single ternary scale. `0` selects a single global scale
    /// (legacy behavior). Must be a multiple of 8 so groups never split a
    /// packed byte. Scales are shared across output rows, so a matrix with
    /// `k` columns has `ceil(k / group_size)` scales.
    #[serde(default)]
    pub group_size: usize,
    /// Per-token int8 activation quantization (W1A8) on the matmul path.
    /// This is the BitNet b1.58-style default: activations are quantized to
    /// int8 per row so the inner loop is integer-only (i32 accumulation).
    /// Set to `false` to keep the exact f32-activation kernel.
    #[serde(default = "default_a8")]
    pub a8: bool,
}

fn default_a8() -> bool {
    true
}

impl QuantConfig {
    pub fn ternary() -> Self {
        Self {
            scheme: QuantScheme::Ternary,
            outlier_frac: 0.0,
            group_size: 0,
            a8: true,
        }
    }

    /// Ternary quantization that keeps the top `outlier_frac` fraction of
    /// weights (ranked by |w|) in exact f32.
    pub fn ternary_with_outliers(outlier_frac: f64) -> Self {
        Self {
            scheme: QuantScheme::Ternary,
            outlier_frac,
            group_size: 0,
            a8: true,
        }
    }

    /// Group-wise ternary: every `group_size` consecutive columns along `k`
    /// get their own scale instead of a single global one. See
    /// [`QuantConfig::group_size`].
    pub fn ternary_grouped(group_size: usize) -> Self {
        Self {
            scheme: QuantScheme::Ternary,
            outlier_frac: 0.0,
            group_size,
            a8: true,
        }
    }

    /// Group-wise ternary with the top `outlier_frac` fraction of weights kept
    /// exact.
    pub fn ternary_grouped_with_outliers(outlier_frac: f64, group_size: usize) -> Self {
        Self {
            scheme: QuantScheme::Ternary,
            outlier_frac,
            group_size,
            a8: true,
        }
    }

    /// Return a copy with per-token int8 activations toggled to `enabled`.
    pub fn with_a8(mut self, enabled: bool) -> Self {
        self.a8 = enabled;
        self
    }

    /// Convenience: disable per-token int8 activations (exact f32 path).
    pub fn without_a8(self) -> Self {
        self.with_a8(false)
    }

    pub fn target_dtype(&self) -> DType {
        DType::BIT1
    }

    pub fn compression_ratio(&self) -> f64 {
        32.0
    }
}

/// Exact f32 values kept out of the packed ternary representation.
///
/// `indices` are flat offsets into the [n, k] weight tensor. The packed ternary
/// keeps the sign bit for these positions (there is no zero symbol in 1
/// bit/weight); the matmul subtracts the ternary contribution and adds the
/// exact value back, which is mathematically identical to zeroing the position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierMap {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedTensor {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub scales: Vec<f32>,
    pub config: QuantConfig,
    #[serde(default)]
    pub outliers: Option<OutlierMap>,
}

impl QuantizedTensor {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn nbytes(&self) -> usize {
        let outlier_bytes = self
            .outliers
            .as_ref()
            .map(|o| o.indices.len() * (size_of::<u32>() + size_of::<f32>()))
            .unwrap_or(0);
        self.data.len() + outlier_bytes
    }

    pub fn compression_ratio(&self) -> f64 {
        let original_bytes = self.num_elements() * 4;
        let quantized_bytes = self.nbytes() + self.scales.len() * 4;
        original_bytes as f64 / quantized_bytes.max(1) as f64
    }
}
