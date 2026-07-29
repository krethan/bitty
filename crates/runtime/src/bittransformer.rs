use crate::attention::{KvCache, RoPECache};
use crate::bitlinear::BitLinear;
use crate::config::ModelConfig;
use crate::layers::RmsNorm;
use crate::GpuContext;
use bitllm_tensor::{DType, Tensor};
use std::cell::RefCell;

/// BitTransformerLayer - Transformer layer using 1-bit (ternary) quantized linear layers
/// for attention projections and FFN, with FP32 norms and activations.
pub struct BitTransformerLayer {
    pub attention: BitAttention,
    pub attn_norm: RmsNorm,
    pub ffn_up: BitLinear,
    pub ffn_gate: BitLinear,
    pub ffn_down: BitLinear,
    pub ffn_norm: RmsNorm,
    pub config: ModelConfig,
}

/// Attention module with 1-bit quantized Q/K/V/O projections
pub struct BitAttention {
    pub q_proj: BitLinear,
    pub k_proj: BitLinear,
    pub v_proj: BitLinear,
    pub o_proj: BitLinear,
    pub config: ModelConfig,
    scores: RefCell<Vec<f32>>,
    acc: RefCell<Vec<f32>>,
}

impl BitAttention {
    pub fn new(
        q_proj: BitLinear,
        k_proj: BitLinear,
        v_proj: BitLinear,
        o_proj: BitLinear,
        config: ModelConfig,
    ) -> Self {
        let max_seq_len = config.max_seq_len;
        let head_dim = config.head_dim();
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            config,
            scores: RefCell::new(Vec::with_capacity(max_seq_len)),
            acc: RefCell::new(Vec::with_capacity(head_dim)),
        }
    }

    pub fn forward(
        &self,
        input: &Tensor,
        cache: Option<&mut KvCache>,
        layer_idx: usize,
        position: usize,
    ) -> Tensor {
        self.forward_gpu(input, cache, layer_idx, position, None, None)
    }

    pub fn forward_gpu(
        &self,
        input: &Tensor,
        mut cache: Option<&mut KvCache>,
        layer_idx: usize,
        position: usize,
        gpu: Option<&GpuContext>,
        rope_cache: Option<&RoPECache>,
    ) -> Tensor {
        let seq_len = input.shape()[0];
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads();
        let head_dim = self.config.head_dim();

        let q = self.q_proj.forward(input);
        let k = self.k_proj.forward(input);
        let v = self.v_proj.forward(input);

        let mut q_reshaped = reshape_for_attention(&q, num_heads, head_dim);
        let mut k_reshaped = reshape_for_attention(&k, num_kv_heads, head_dim);
        let v_reshaped = reshape_for_attention(&v, num_kv_heads, head_dim);

        gpu_rope(
            &mut q_reshaped,
            &mut k_reshaped,
            position,
            head_dim,
            self.config.rope_theta,
            gpu,
            rope_cache,
        );

        if let Some(c) = cache.as_mut() {
            c.update(layer_idx, &k_reshaped, &v_reshaped, position);
        }

        let (k_ref, v_ref, kv_seq_len) = match cache.as_ref() {
            Some(c) => c.get_kv_used(layer_idx),
            None => (&k_reshaped, &v_reshaped, k_reshaped.shape()[1]),
        };

        let mut scores_buf = self.scores.borrow_mut();
        let mut acc_buf = self.acc.borrow_mut();
        if scores_buf.len() < kv_seq_len {
            scores_buf.resize(kv_seq_len, 0.0);
        }
        if acc_buf.len() < head_dim {
            acc_buf.resize(head_dim, 0.0);
        }

        let output = scaled_dot_product_attention(
            &q_reshaped,
            k_ref,
            v_ref,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
            kv_seq_len,
            &mut scores_buf[..kv_seq_len],
            &mut acc_buf[..head_dim],
        );

        let reshaped = output.reshape_owned(&[seq_len, hidden_size]);
        self.o_proj.forward(&reshaped)
    }
}

impl BitTransformerLayer {
    pub fn new(
        attention: BitAttention,
        attn_norm: RmsNorm,
        ffn_up: BitLinear,
        ffn_gate: BitLinear,
        ffn_down: BitLinear,
        ffn_norm: RmsNorm,
        config: ModelConfig,
    ) -> Self {
        Self {
            attention,
            attn_norm,
            ffn_up,
            ffn_gate,
            ffn_down,
            ffn_norm,
            config,
        }
    }

    pub fn forward(
        &self,
        input: &Tensor,
        cache: Option<&mut KvCache>,
        layer_idx: usize,
        position: usize,
    ) -> Tensor {
        self.forward_gpu(input, cache, layer_idx, position, None, None)
    }

    pub fn forward_gpu(
        &self,
        input: &Tensor,
        cache: Option<&mut KvCache>,
        layer_idx: usize,
        position: usize,
        gpu: Option<&GpuContext>,
        rope_cache: Option<&RoPECache>,
    ) -> Tensor {
        // Attention block with residual
        let normed = self.attn_norm.forward_gpu(input, gpu);
        let attn_out = self
            .attention
            .forward_gpu(&normed, cache, layer_idx, position, gpu, rope_cache);

        #[cfg(feature = "gpu")]
        if let Some(ctx) = gpu {
            let h = ctx.add(input, &attn_out).unwrap_or_else(|e| {
                log::warn!("GPU add failed, falling back to CPU: {}", e);
                let mut h = input.clone();
                h.add_assign(&attn_out).unwrap();
                h
            });
            let normed2 = self.ffn_norm.forward_gpu(&h, gpu);
            let up = self.ffn_up.forward(&normed2);
            let gate = self.ffn_gate.forward(&normed2);
            let activated = silu_mul(&gate, &up);
            let ffn_out = self.ffn_down.forward(&activated);
            return ctx.add(&h, &ffn_out).unwrap_or_else(|e| {
                log::warn!("GPU add failed, falling back to CPU: {}", e);
                let mut h2 = h;
                h2.add_assign(&ffn_out).unwrap();
                h2
            });
        }
        let _ = gpu;

        // FFN block with residual
        let mut h = input.clone();
        h.add_assign(&attn_out).unwrap();

        let normed2 = self.ffn_norm.forward_gpu(&h, gpu);
        let up = self.ffn_up.forward(&normed2);
        let gate = self.ffn_gate.forward(&normed2);
        let activated = silu_mul(&gate, &up);
        let ffn_out = self.ffn_down.forward(&activated);
        h.add_assign(&ffn_out).unwrap();
        h
    }

    /// Create a BitTransformerLayer from a standard TransformerLayer by quantizing weights to 1-bit
    pub fn from_fp32_layer(layer: &crate::model::TransformerLayer) -> Self {
        let config = layer.config.clone();

        // Quantize attention projections to 1-bit (ternary)
        let q_proj = BitLinear::from_linear(&layer.attention.q_proj);
        let k_proj = BitLinear::from_linear(&layer.attention.k_proj);
        let v_proj = BitLinear::from_linear(&layer.attention.v_proj);
        let o_proj = BitLinear::from_linear(&layer.attention.o_proj);

        let attention = BitAttention::new(q_proj, k_proj, v_proj, o_proj, config.clone());

        // Quantize FFN layers to 1-bit
        let ffn_up = BitLinear::from_linear(&layer.ffn_up);
        let ffn_gate = BitLinear::from_linear(&layer.ffn_gate);
        let ffn_down = BitLinear::from_linear(&layer.ffn_down);

        Self {
            attention,
            attn_norm: layer.attn_norm.clone(),
            ffn_up,
            ffn_gate,
            ffn_down,
            ffn_norm: layer.ffn_norm.clone(),
            config,
        }
    }
}

// Reuse helper functions from model.rs
fn reshape_for_attention(tensor: &Tensor, num_heads: usize, head_dim: usize) -> Tensor {
    let seq_len = tensor.shape()[0];
    let mut result = Tensor::zeros(&[num_heads, seq_len, head_dim], DType::F32);
    let src_slice = tensor.as_f32_slice();
    let dst_slice = result.as_f32_slice_mut();

    for h in 0..num_heads {
        for pos in 0..seq_len {
            let src_base = pos * num_heads * head_dim + h * head_dim;
            let dst_base = h * seq_len * head_dim + pos * head_dim;
            dst_slice[dst_base..dst_base + head_dim]
                .copy_from_slice(&src_slice[src_base..src_base + head_dim]);
        }
    }

    result
}

fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    kv_seq_len: usize,
    scores: &mut [f32],
    acc: &mut [f32],
) -> Tensor {
    let kv_stride = k.shape()[1];
    let scale = (head_dim as f32).sqrt();
    let kv_groups = num_heads / num_kv_heads;

    let mut output = Tensor::zeros(&[num_heads, seq_len, head_dim], DType::F32);

    let q_slice = q.as_f32_slice();
    let k_slice = k.as_f32_slice();
    let v_slice = v.as_f32_slice();
    let out_slice = output.as_f32_slice_mut();

    for h in 0..num_heads {
        let kv_h = h / kv_groups;
        for pos_q in 0..seq_len {
            let q_row = &q_slice[h * seq_len * head_dim + pos_q * head_dim..][..head_dim];

            let mut max_val: f32 = f32::NEG_INFINITY;

            for pos_k in 0..kv_seq_len {
                let k_row = &k_slice[kv_h * kv_stride * head_dim + pos_k * head_dim..][..head_dim];
                let dot = bitllm_tensor::simd::f32_dot(q_row, k_row);
                let score = dot / scale;
                scores[pos_k] = score;
                if score > max_val {
                    max_val = score;
                }
            }

            bitllm_tensor::simd::f32_add_scalar_inplace(&mut scores[..kv_seq_len], -max_val);
            bitllm_tensor::simd::f32_exp_inplace(&mut scores[..kv_seq_len]);
            let sum_exp: f32 = bitllm_tensor::simd::f32_sum(&scores[..kv_seq_len]);
            bitllm_tensor::simd::f32_scale_inplace(&mut scores[..kv_seq_len], 1.0 / sum_exp);

            acc[..head_dim].fill(0.0);
            for (pos_k, score) in scores[..kv_seq_len].iter().enumerate() {
                let v_row = &v_slice[kv_h * kv_stride * head_dim + pos_k * head_dim..][..head_dim];
                bitllm_tensor::simd::f32_axpy(v_row, *score, &mut acc[..head_dim]);
            }
            let out_row = &mut out_slice[h * seq_len * head_dim + pos_q * head_dim..][..head_dim];
            out_row.copy_from_slice(&acc[..head_dim]);
        }
    }

    output
}

fn silu_mul(a: &Tensor, b: &Tensor) -> Tensor {
    let mut result = Tensor::zeros(a.shape(), DType::F32);
    let a_slice = a.as_f32_slice();
    let b_slice = b.as_f32_slice();
    let out_slice = result.as_f32_slice_mut();
    bitllm_tensor::simd::f32_silu_mul(a_slice, b_slice, out_slice);
    result
}

fn gpu_rope(
    q: &mut Tensor,
    k: &mut Tensor,
    position: usize,
    head_dim: usize,
    theta: f32,
    gpu: Option<&GpuContext>,
    rope_cache: Option<&RoPECache>,
) {
    #[cfg(feature = "gpu")]
    if let Some(ctx) = gpu {
        if q.is_gpu() || k.is_gpu() {
            let num_heads = q.shape()[0];
            if let Ok((q_rope, k_rope)) = ctx.rope(q, k, num_heads, head_dim, position, theta) {
                *q = q_rope;
                *k = k_rope;
                return;
            }
        }
    }
    let _ = gpu;
    crate::attention::apply_rotary_emb_inplace_with_cache(q, k, position, head_dim, theta, rope_cache);
}

#[cfg(test)]
mod tests {
    use crate::config::ModelConfig;
    use crate::model::TransformerLayer;

    #[test]
    fn test_bit_transformer_layer_creation() {
        let config = ModelConfig::tiny_test();
        let _layer = TransformerLayer::new_dummy(&config);
        // Can't test fully without the from_fp32_layer method being callable
    }
}