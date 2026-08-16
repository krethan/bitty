use bitllm_tensor::{DType, Tensor};
use std::cell::OnceCell;

use crate::GpuContext;

/// Add a 1-D bias `[n]` to every row of a `[m, n]` tensor in place.
/// Falls back to `add_assign` for already-compatible shapes.
fn add_bias_1d(result: &mut Tensor, bias: &Tensor) {
    let m = result.shape()[0];
    let n = result.shape().last().copied().unwrap_or(1);
    let b = bias.as_f32_slice();
    if bias.shape().len() == 1 && bias.shape()[0] == n {
        let out = result.as_f32_slice_mut();
        for row in 0..m {
            let base = row * n;
            for j in 0..n {
                out[base + j] += b[j];
            }
        }
    } else {
        result.add_assign(bias).unwrap();
    }
}

pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn from_f32(weight_data: &[f32], weight_shape: &[usize]) -> Self {
        let weight = Tensor::from_slice(weight_data, weight_shape);
        Self { weight, bias: None }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_gpu(input, None)
    }

    pub fn forward_gpu(&self, input: &Tensor, gpu: Option<&GpuContext>) -> Tensor {
        if let Some(ctx) = gpu {
            if input.is_gpu() || self.weight.is_gpu() {
                return self.forward_gpu_impl(input, ctx);
            }
        }
        self.forward_cpu(input)
    }

    fn forward_cpu(&self, input: &Tensor) -> Tensor {
        let m = input.shape()[0];
        let k = input.shape()[1];
        let n = self.weight.shape()[0];
        let mut result = Tensor::zeros(&[m, n], DType::F32);
        {
            let a = input.as_f32_slice();
            let b = self.weight.as_f32_slice();
            let out = result.as_f32_slice_mut();
            bitllm_tensor::simd::f32_matmul(a, b, out, m, k, n);
        }
        if let Some(ref bias) = self.bias {
            add_bias_1d(&mut result, bias);
        }
        result
    }

    #[cfg(feature = "gpu")]
    fn forward_gpu_impl(&self, input: &Tensor, ctx: &GpuContext) -> Tensor {
        let seq_len = input.shape()[0];
        let k = input.shape().last().copied().unwrap_or(1);
        let n = self.weight.shape()[0];
        let w_t = self.weight.transpose();
        let mut result = ctx
            .matmul_transposed(input, &w_t, seq_len, k, n)
            .unwrap_or_else(|e| {
                log::warn!("GPU matmul failed, falling back to CPU: {}", e);
                self.forward_cpu(input)
            });
        if let Some(ref bias) = self.bias {
            let mut out = result.clone();
            add_bias_1d(&mut out, bias);
            result = out;
        }
        result
    }

    #[cfg(not(feature = "gpu"))]
    fn forward_gpu_impl(&self, input: &Tensor, _ctx: &GpuContext) -> Tensor {
        self.forward_cpu(input)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NormKind {
    Rms,
    Layer,
}

#[derive(Clone)]
pub struct RmsNorm {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub eps: f32,
    pub kind: NormKind,
    /// Gemma-style one-centered norm: `out = x * rms * (1 + w)` instead of
    /// `x * rms * w`. Applies to all Gemma/Gemma-2 norms.
    pub one_centered: bool,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f32) -> Self {
        Self {
            weight,
            bias: None,
            eps,
            kind: NormKind::Rms,
            one_centered: false,
        }
    }

    /// One-centered (Gemma) RMSNorm.
    pub fn new_one_centered(weight: Tensor, eps: f32) -> Self {
        Self {
            weight,
            bias: None,
            eps,
            kind: NormKind::Rms,
            one_centered: true,
        }
    }

    /// A LayerNorm-capable norm (mean/variance subtraction plus bias).
    pub fn new_layer(weight: Tensor, bias: Option<Tensor>, eps: f32) -> Self {
        Self {
            weight,
            bias,
            eps,
            kind: NormKind::Layer,
            one_centered: false,
        }
    }

    pub fn from_f32(data: &[f32], shape: &[usize], eps: f32) -> Self {
        let weight = Tensor::from_slice(data, shape);
        Self {
            weight,
            bias: None,
            eps,
            kind: NormKind::Rms,
            one_centered: false,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_gpu(input, None)
    }

    pub fn forward_gpu(&self, input: &Tensor, gpu: Option<&GpuContext>) -> Tensor {
        #[cfg(feature = "gpu")]
        if let Some(ctx) = gpu {
            if self.kind == NormKind::Rms
                && !self.one_centered
                && (input.is_gpu() || self.weight.is_gpu())
            {
                return ctx
                    .rms_norm(input, &self.weight, self.eps)
                    .unwrap_or_else(|e| {
                        log::warn!("GPU rms_norm failed, falling back to CPU: {}", e);
                        self.forward_cpu(input)
                    });
            }
        }
        let _ = gpu;
        self.forward_cpu(input)
    }

    fn forward_cpu(&self, input: &Tensor) -> Tensor {
        match self.kind {
            NormKind::Rms => self.forward_rms(input),
            NormKind::Layer => self.forward_layer(input),
        }
    }

    fn forward_rms(&self, input: &Tensor) -> Tensor {
        let n = input.num_elements();
        let hidden = input.shape().last().copied().unwrap_or(n);

        let mut result = Tensor::zeros(input.shape(), DType::F32);
        let in_slice = input.as_f32_slice();
        let w_slice = self.weight.as_f32_slice();
        let out_slice = result.as_f32_slice_mut();

        let eps = self.eps;
        for i in 0..(n / hidden) {
            let row = &in_slice[i * hidden..][..hidden];
            let sum_sq = bitllm_tensor::simd::f32_dot(row, row);
            let inv_rms = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();
            let w_row = &w_slice[..hidden];
            let out_row = &mut out_slice[i * hidden..][..hidden];
            if self.one_centered {
                for j in 0..hidden {
                    out_row[j] = row[j] * (1.0 + w_row[j]) * inv_rms;
                }
            } else {
                bitllm_tensor::simd::f32_mul_scaled(row, w_row, inv_rms, out_row);
            }
        }

        result
    }

    fn forward_layer(&self, input: &Tensor) -> Tensor {
        let n = input.num_elements();
        let hidden = input.shape().last().copied().unwrap_or(n);

        let mut result = Tensor::zeros(input.shape(), DType::F32);
        let in_slice = input.as_f32_slice();
        let w_slice = self.weight.as_f32_slice();
        let b_slice = self.bias.as_ref().map(|b| b.as_f32_slice());
        let out_slice = result.as_f32_slice_mut();

        let eps = self.eps;
        for i in 0..(n / hidden) {
            let row = &in_slice[i * hidden..][..hidden];
            let mean = bitllm_tensor::simd::f32_sum(row) / hidden as f32;
            let mut var = 0.0f64;
            for &x in row {
                let d = (x - mean) as f64;
                var += d * d;
            }
            var /= hidden as f64;
            let inv_std = 1.0 / (var as f32 + eps).sqrt();
            let out_row = &mut out_slice[i * hidden..][..hidden];
            for (j, &x) in row.iter().enumerate() {
                out_row[j] = (x - mean) * inv_std * w_slice[j];
                if let Some(b) = b_slice {
                    out_row[j] += b[j];
                }
            }
        }

        result
    }
}

pub struct Embedding {
    pub weight: Tensor,
    pub vocab_size: usize,
    pub embed_dim: usize,
    cached_f32: OnceCell<Tensor>,
}

impl Embedding {
    pub fn new(weight: Tensor, vocab_size: usize, embed_dim: usize) -> Self {
        Self {
            weight,
            vocab_size,
            embed_dim,
            cached_f32: OnceCell::new(),
        }
    }

    pub fn from_f32(data: &[f32], vocab_size: usize, embed_dim: usize) -> Self {
        let weight = Tensor::from_slice(data, &[vocab_size, embed_dim]);
        Self {
            weight,
            vocab_size,
            embed_dim,
            cached_f32: OnceCell::new(),
        }
    }

    pub fn forward(&self, token_ids: &[u32]) -> Tensor {
        let seq_len = token_ids.len();
        let mut result = Tensor::zeros(&[seq_len, self.embed_dim], DType::F32);

        let w_slice = if self.weight.dtype() == DType::F32 {
            self.weight.as_f32_slice()
        } else {
            let f32_tensor = self.cached_f32.get_or_init(|| self.weight.to_f32());
            f32_tensor.as_f32_slice()
        };

        let out_slice = result.as_f32_slice_mut();

        for (pos, &token_id) in token_ids.iter().enumerate() {
            let id = token_id as usize;
            assert!(
                id < self.vocab_size,
                "token id {} out of vocab range {}",
                id,
                self.vocab_size
            );
            let src = &w_slice[id * self.embed_dim..(id + 1) * self.embed_dim];
            let dst = &mut out_slice[pos * self.embed_dim..(pos + 1) * self.embed_dim];
            dst.copy_from_slice(src);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_one_centered() {
        let input = Tensor::from_slice(&[3.0, 4.0, 5.0, 12.0], &[2, 2]);
        let eps = 1e-5;

        let norm = RmsNorm::new_one_centered(Tensor::ones(&[2], DType::F32), eps);
        let out = norm.forward(&input);

        let standard = RmsNorm::new(Tensor::ones(&[2], DType::F32), eps);
        let std_out = standard.forward(&input);

        let one = out.as_f32_slice();
        let std = std_out.as_f32_slice();
        for i in 0..4 {
            assert!(
                (one[i] - 2.0 * std[i]).abs() < 1e-5,
                "one-centered (w=1) scales by (1 + w) = 2"
            );
        }

        let zero = RmsNorm::new_one_centered(Tensor::zeros(&[2], DType::F32), eps);
        let z_out = zero.forward(&input);
        let zin = z_out.as_f32_slice();
        for i in 0..4 {
            assert!(
                (zin[i] - std[i]).abs() < 1e-5,
                "one-centered (w=0) equals standard RMSNorm"
            );
        }
    }
}
