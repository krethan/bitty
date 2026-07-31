use crate::attention::{apply_rotary_emb_inplace_with_cache, KvCache, RoPECache};
use crate::bitlinear::BitLinear;
use crate::layers::RmsNorm;
use crate::config::ModelConfig;
use crate::model::TransformerLayer;
use crate::GpuContext;
use bitllm_tensor::{DType, Tensor};

/// BitTransformerLayer - Transformer layer using 1-bit quantized linear layers
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
}

impl BitAttention {
    pub fn new(
        q_proj: BitLinear,
        k_proj: BitLinear,
        v_proj: BitLinear,
        o_proj: BitLinear,
        config: ModelConfig,
    ) -> Self {
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
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
        let seq_len = input.shape()[0];
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads();
        let head_dim = self.config.head_dim();

        // Q, K, V projections using 1-bit linear layers
        let q = self.q_proj.forward(input);
        let k = self.k_proj.forward(input);
        let v = self.v_proj.forward(input);

        let mut q_reshaped = reshape_for_attention(&q, num_heads, head_dim);
        let mut k_reshaped = reshape_for_attention(&k, num_kv_heads, head_dim);
        let mut v_reshaped = reshape_for_attention(&v, num_kv_heads, head_dim);

        gpu_rope(
            &mut q_reshaped,
            &mut k_reshaped,
            position,
            head_dim,
            self.config.rope_theta,
            gpu,
            rope_cache,
        );

        if let Some(cache) = cache {
            cache.update(layer_idx, &k_reshaped, &v_reshaped, position);
            let (used_k, used_v, _) = cache.get_kv_used(layer_idx);
            k_reshaped = used_k.clone();
            v_reshaped = used_v.clone();
        }

        let output = scaled_dot_product_attention(
            &q_reshaped,
            &k_reshaped,
            &v_reshaped,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
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
        let residual = input.clone();
        let normed = self.attn_norm.forward_gpu(input, gpu);
        let attn_out = self
            .attention
            .forward_gpu(&normed, cache, layer_idx, position, gpu, rope_cache);
        let h = gpu_add(&residual, &attn_out, gpu);

        // FFN block with residual
        let residual2 = h.clone();
        let normed2 = self.ffn_norm.forward_gpu(&h, gpu);
        let up = self.ffn_up.forward(&normed2);
        let gate = self.ffn_gate.forward(&normed2);
        let gated = silu(&gate);
        let activated = hadamard(&gated, &up);
        let ffn_out = self.ffn_down.forward(&activated);

        gpu_add(&residual2, &ffn_out, gpu)
    }

    /// Create a BitTransformerLayer from an fp32 TransformerLayer by quantizing weights to 1-bit.
    pub fn from_fp32_layer(layer: &TransformerLayer) -> Self {
        let config = layer.config.clone();

        // Convert fp32 Linear layers to 1-bit BitLinear
        let q_proj = BitLinear::from_linear(&layer.attention.q_proj);
        let k_proj = BitLinear::from_linear(&layer.attention.k_proj);
        let v_proj = BitLinear::from_linear(&layer.attention.v_proj);
        let o_proj = BitLinear::from_linear(&layer.attention.o_proj);

        let attention = BitAttention::new(q_proj, k_proj, v_proj, o_proj, config.clone());

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

fn reshape_for_attention(tensor: &Tensor, num_heads: usize, head_dim: usize) -> Tensor {
    let seq_len = tensor.shape()[0];
    let mut result = Tensor::zeros(&[num_heads, seq_len, head_dim], DType::F32);
    let src = tensor.as_f32_slice();
    let dst = result.as_f32_slice_mut();

    for h in 0..num_heads {
        for pos in 0..seq_len {
            let src_base = pos * num_heads * head_dim + h * head_dim;
            let dst_base = h * seq_len * head_dim + pos * head_dim;
            dst[dst_base..dst_base + head_dim].copy_from_slice(&src[src_base..src_base + head_dim]);
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
) -> Tensor {
    let kv_seq_len = k.shape()[1];
    let scale = (head_dim as f32).sqrt();
    let kv_groups = num_heads / num_kv_heads;

    let mut output = Tensor::zeros(&[num_heads, seq_len, head_dim], DType::F32);

    let q_ptr = q.as_f32_slice();
    let k_ptr = k.as_f32_slice();
    let v_ptr = v.as_f32_slice();
    let out_ptr = output.as_f32_slice_mut();

    for h in 0..num_heads {
        let kv_h = h / kv_groups;
        for pos_q in 0..seq_len {
            let q_row = &q_ptr[h * seq_len * head_dim + pos_q * head_dim..][..head_dim];

            // Compute attention scores
            let mut scores = Vec::with_capacity(kv_seq_len);
            for pos_k in 0..kv_seq_len {
                let k_row = &k_ptr[kv_h * kv_seq_len * head_dim + pos_k * head_dim..][..head_dim];
                let dot = q_row.iter().zip(k_row.iter()).map(|(a, b)| a * b).sum::<f32>();
                scores.push(dot / scale);
            }

            // Numerically stable softmax
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();
            let inv_sum = 1.0 / sum_exp;
            for s in exp_scores.iter_mut() {
                *s *= inv_sum;
            }

            // Weighted sum of values
            let out_row = &mut out_ptr[h * seq_len * head_dim + pos_q * head_dim..][..head_dim];
            for d in 0..head_dim {
                out_row[d] = 0.0;
            }
            for (pos_k, weight) in exp_scores.iter().enumerate() {
                let v_row = &v_ptr[kv_h * kv_seq_len * head_dim + pos_k * head_dim..][..head_dim];
                for d in 0..head_dim {
                    out_row[d] += weight * v_row[d];
                }
            }
        }
    }

    output
}

#[inline]
fn silu(x: &Tensor) -> Tensor {
    let mut result = Tensor::zeros(x.shape(), DType::F32);
    let in_slice = x.as_f32_slice();
    let out_slice = result.as_f32_slice_mut();
    bitllm_tensor::simd::f32_silu(in_slice, out_slice);
    result
}

#[inline]
fn hadamard(a: &Tensor, b: &Tensor) -> Tensor {
    let mut result = Tensor::zeros(a.shape(), DType::F32);
    let a_slice = a.as_f32_slice();
    let b_slice = b.as_f32_slice();
    let out_slice = result.as_f32_slice_mut();
    bitllm_tensor::simd::f32_mul(a_slice, b_slice, out_slice);
    result
}

#[inline]
fn gpu_add(a: &Tensor, b: &Tensor, gpu: Option<&GpuContext>) -> Tensor {
    #[cfg(feature = "gpu")]
    if let Some(ctx) = gpu {
        return ctx.add(a, b).unwrap_or_else(|_| a.add(b).unwrap());
    }
    let _ = gpu;
    a.add(b).unwrap()
}

/// Apply RoPE embeddings, delegating to GPU if available, otherwise CPU fallback.
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
        let num_heads = q.shape()[0];
        if q.is_gpu() || k.is_gpu() {
            if let Ok((q_rope, k_rope)) = ctx.rope(q, k, num_heads, head_dim, position, theta) {
                *q = q_rope;
                *k = k_rope;
                return;
            }
        }
    }
    let _ = gpu;
    apply_rotary_emb_inplace_with_cache(q, k, position, head_dim, theta, rope_cache);
}