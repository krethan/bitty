use bitllm_quantization::{absmax_dequantize, absmax_quantize, QuantConfig, QuantizedTensor};
use bitllm_tensor::simd;
use bitllm_tensor::DType;
use bitllm_tensor::Tensor;

use crate::GpuContext;

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
        let mut result = input.dot(&self.weight.transpose()).unwrap();
        if let Some(ref bias) = self.bias {
            result = result.add(bias).unwrap();
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
            result = ctx.add(&result, bias).unwrap_or_else(|e| {
                log::warn!("GPU add failed, falling back to CPU: {}", e);
                result.add(bias).unwrap()
            });
        }
        result
    }

    #[cfg(not(feature = "gpu"))]
    fn forward_gpu_impl(&self, input: &Tensor, _ctx: &GpuContext) -> Tensor {
        self.forward_cpu(input)
    }

    pub fn quantize_int8(&self) -> QuantizedLinear {
        let qt = absmax_quantize(&self.weight, &QuantConfig::int8());
        QuantizedLinear {
            weight: qt,
            bias: self.bias.clone(),
            config: QuantConfig::int8(),
        }
    }
}

pub struct QuantizedLinear {
    pub weight: QuantizedTensor,
    pub bias: Option<Tensor>,
    pub config: QuantConfig,
}

impl QuantizedLinear {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let w_dequant = absmax_dequantize(&self.weight);
        let mut result = input.dot(&w_dequant.transpose()).unwrap();
        if let Some(ref bias) = self.bias {
            result = result.add(bias).unwrap();
        }
        result
    }
}

pub struct RmsNorm {
    pub weight: Tensor,
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn from_f32(data: &[f32], shape: &[usize], eps: f32) -> Self {
        let weight = Tensor::from_slice(data, shape);
        Self { weight, eps }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_gpu(input, None)
    }

    pub fn forward_gpu(&self, input: &Tensor, gpu: Option<&GpuContext>) -> Tensor {
        #[cfg(feature = "gpu")]
        if let Some(ctx) = gpu {
            if input.is_gpu() || self.weight.is_gpu() {
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
        let n = input.num_elements();
        let hidden = input.shape().last().copied().unwrap_or(n);

        let mut result = Tensor::zeros(input.shape(), DType::F32);
        let in_f32 = input.to_f32();
        let in_slice = in_f32.as_f32_slice();
        let w_slice = self.weight.as_f32_slice();
        let out_slice = result.as_f32_slice_mut();

        let eps = self.eps;
        for i in 0..(n / hidden) {
            let row = &in_slice[i * hidden..][..hidden];
            let sum_sq: f32 = row.iter().map(|v| v * v).sum();
            let rms = (sum_sq / hidden as f32 + eps).sqrt();
            let inv_rms = 1.0 / rms;
            let w_row = &w_slice[..hidden];
            let out_row = &mut out_slice[i * hidden..][..hidden];
            simd::f32_mul(row, w_row, out_row);
            let tmp = out_row.to_vec();
            simd::f32_scale(&tmp, inv_rms, out_row);
        }

        result
    }
}

pub struct Embedding {
    pub weight: Tensor,
    pub vocab_size: usize,
    pub embed_dim: usize,
}

impl Embedding {
    pub fn new(weight: Tensor, vocab_size: usize, embed_dim: usize) -> Self {
        Self {
            weight,
            vocab_size,
            embed_dim,
        }
    }

    pub fn from_f32(data: &[f32], vocab_size: usize, embed_dim: usize) -> Self {
        let weight = Tensor::from_slice(data, &[vocab_size, embed_dim]);
        Self {
            weight,
            vocab_size,
            embed_dim,
        }
    }

    pub fn forward(&self, token_ids: &[u32]) -> Tensor {
        let seq_len = token_ids.len();
        let mut result = Tensor::zeros(&[seq_len, self.embed_dim], DType::F32);

        for (pos, &token_id) in token_ids.iter().enumerate() {
            let id = token_id as usize;
            assert!(
                id < self.vocab_size,
                "token id {} out of vocab range {}",
                id,
                self.vocab_size
            );
            for j in 0..self.embed_dim {
                let val = self.weight.get_flat_f32(id * self.embed_dim + j);
                result.set_flat_f32(pos * self.embed_dim + j, val);
            }
        }

        result
    }
}
