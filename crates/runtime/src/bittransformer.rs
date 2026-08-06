use crate::attention::{
    apply_rotary_emb_batch, apply_rotary_emb_inplace_with_cache, scaled_dot_product_attention_batched,
    KvCache, RoPECache,
};
use crate::bitlinear::BitLinear;
use crate::layers::RmsNorm;
use crate::config::ModelConfig;
use crate::model::TransformerLayer;
use crate::GpuContext;
use bitllm_quantization::QuantConfig;
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
    /// When set, the layer uses the BitNet b1.58 mixed-precision residual:
    /// the block input is `x - SubLN(x)` (bounded, for W1A8 stability) and
    /// the residual add stays in f32. The residual stream is never quantized.
    pub sub_ln: Option<RmsNorm>,
    pub config: ModelConfig,
}

/// Attention module with 1-bit quantized Q/K/V/O projections
pub struct BitAttention {
    pub q_proj: BitLinear,
    pub k_proj: BitLinear,
    pub v_proj: BitLinear,
    pub o_proj: BitLinear,
    /// Gemma-style per-head Q norm (RMSNorm over each head, applied before RoPE).
    pub q_norm: Option<RmsNorm>,
    /// Gemma-style per-head K norm (RMSNorm over each head, applied before RoPE).
    pub k_norm: Option<RmsNorm>,
    pub config: ModelConfig,
}

impl BitAttention {
    pub fn new(
        q_proj: BitLinear,
        k_proj: BitLinear,
        v_proj: BitLinear,
        o_proj: BitLinear,
        q_norm: Option<RmsNorm>,
        k_norm: Option<RmsNorm>,
        config: ModelConfig,
    ) -> Self {
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
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
        self.forward_gpu(input, cache, layer_idx, 0, position, None, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_gpu(
        &self,
        input: &Tensor,
        cache: Option<&mut KvCache>,
        layer_idx: usize,
        slot: usize,
        position: usize,
        gpu: Option<&GpuContext>,
        rope_cache: Option<&RoPECache>,
    ) -> Tensor {
        let seq_len = input.shape()[0];
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

        let output = if let Some(cache) = cache {
            cache.update(layer_idx, slot, &k_reshaped, &v_reshaped, position);
            let (used_k, used_v, kv_len) = cache.get_kv_used(layer_idx, slot);
            k_reshaped = used_k.clone();
            v_reshaped = used_v.clone();
            scaled_dot_product_attention(
                &q_reshaped,
                &k_reshaped,
                &v_reshaped,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                kv_len,
                slot,
                position,
                self.config.attn_logit_scale(),
                self.config.attn_logit_softcap(),
            )
        } else {
            scaled_dot_product_attention(
                &q_reshaped,
                &k_reshaped,
                &v_reshaped,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                k_reshaped.shape()[1],
                0,
                position,
                self.config.attn_logit_scale(),
                self.config.attn_logit_softcap(),
            )
        };

        let reshaped = crate::attention::sdp_output_to_hidden(&output, seq_len, num_heads, head_dim);
        self.o_proj.forward(&reshaped)
    }

    /// Batched decode: `input` is `[batch, hidden_size]` with one current token
    /// per batch slot; each row uses its own absolute `positions[b]` and cache
    /// slot.
    pub fn forward_batch(
        &self,
        input: &Tensor,
        mut cache: Option<&mut KvCache>,
        layer_idx: usize,
        positions: &[usize],
    ) -> Tensor {
        let batch = input.shape()[0];
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads();
        let head_dim = self.config.head_dim();
        let theta = self.config.rope_theta;

        let q = self.q_proj.forward(input);
        let k = self.k_proj.forward(input);
        let v = self.v_proj.forward(input);

        let mut q_reshaped = reshape_for_attention(&q, num_heads, head_dim);
        let mut k_reshaped = reshape_for_attention(&k, num_kv_heads, head_dim);
        let v_reshaped = reshape_for_attention(&v, num_kv_heads, head_dim);

        apply_rotary_emb_batch(&mut q_reshaped, positions, head_dim, theta, None);
        apply_rotary_emb_batch(&mut k_reshaped, positions, head_dim, theta, None);

        let output = match cache.as_mut() {
            Some(c) => {
                c.update_batch(layer_idx, &k_reshaped, &v_reshaped, positions);
                let kv_lens = c.kv_lens();
                let max_seq_len = c.k[layer_idx].shape()[2];
                let mut scores = vec![0.0f32; max_seq_len];
                let mut acc = vec![0.0f32; head_dim];
                scaled_dot_product_attention_batched(
                    &q_reshaped,
                    &c.k[layer_idx],
                    &c.v[layer_idx],
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    batch,
                    kv_lens,
                    None,
                    &mut scores,
                    &mut acc,
                    self.config.attn_logit_scale(),
                    self.config.attn_logit_softcap(),
                )
            }
            None => {
                let ones: Vec<usize> = vec![1; batch];
                let mut scores = vec![0.0f32; 1];
                let mut acc = vec![0.0f32; head_dim];
                scaled_dot_product_attention_batched(
                    &q_reshaped,
                    &k_reshaped,
                    &v_reshaped,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    batch,
                    &ones,
                    None,
                    &mut scores,
                    &mut acc,
                    self.config.attn_logit_scale(),
                    self.config.attn_logit_softcap(),
                )
            }
        };

        let reshaped = crate::attention::sdp_batched_output_to_hidden(&output, batch, num_heads, head_dim);
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
            sub_ln: None,
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
        self.forward_gpu(input, cache, layer_idx, 0, position, None, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_gpu(
        &self,
        input: &Tensor,
        cache: Option<&mut KvCache>,
        layer_idx: usize,
        slot: usize,
        position: usize,
        gpu: Option<&GpuContext>,
        rope_cache: Option<&RoPECache>,
    ) -> Tensor {
        // Attention block with residual. When SubLN is enabled the block input
        // is `x - RMSNorm(x)` (bounded activations for W1A8) and the residual
        // `x` stays in full precision — the mixed-precision residual path.
        let (residual, block_input) = match &self.sub_ln {
            Some(sub_ln) => {
                let normed = sub_ln.forward_gpu(input, gpu);
                let block_input = input.sub(&normed).unwrap();
                (input.clone(), block_input)
            }
            None => {
                let normed = self.attn_norm.forward_gpu(input, gpu);
                (input.clone(), normed)
            }
        };
        let attn_out = self
            .attention
            .forward_gpu(&block_input, cache, layer_idx, slot, position, gpu, rope_cache);
        let h = gpu_add(&residual, &attn_out, gpu);

        // FFN block with residual
        let (residual2, ffn_input) = match &self.sub_ln {
            Some(sub_ln) => {
                let normed = sub_ln.forward_gpu(&h, gpu);
                let ffn_input = h.sub(&normed).unwrap();
                (h, ffn_input)
            }
            None => {
                let normed2 = self.ffn_norm.forward_gpu(&h, gpu);
                (h, normed2)
            }
        };
        let up = self.ffn_up.forward(&ffn_input);
        let gate = self.ffn_gate.forward(&ffn_input);
        let gated = silu(&gate);
        let activated = hadamard(&gated, &up);
        let ffn_out = self.ffn_down.forward(&activated);

        gpu_add(&residual2, &ffn_out, gpu)
    }

    /// Batched decode layer forward: `input` is `[batch, hidden_size]` with one
    /// current token per batch slot.
    pub fn forward_batch_gpu(
        &self,
        input: &Tensor,
        cache: Option<&mut KvCache>,
        layer_idx: usize,
        positions: &[usize],
        gpu: Option<&GpuContext>,
    ) -> Tensor {
        // Attention block with residual (SubLN-aware, see `forward_gpu`).
        let (residual, block_input) = match &self.sub_ln {
            Some(sub_ln) => {
                let normed = sub_ln.forward_gpu(input, gpu);
                let block_input = input.sub(&normed).unwrap();
                (input.clone(), block_input)
            }
            None => {
                let normed = self.attn_norm.forward_gpu(input, gpu);
                (input.clone(), normed)
            }
        };
        let attn_out = self
            .attention
            .forward_batch(&block_input, cache, layer_idx, positions);
        let h = gpu_add(&residual, &attn_out, gpu);

        // FFN block with residual
        let (residual2, ffn_input) = match &self.sub_ln {
            Some(sub_ln) => {
                let normed = sub_ln.forward_gpu(&h, gpu);
                let ffn_input = h.sub(&normed).unwrap();
                (h, ffn_input)
            }
            None => {
                let normed2 = self.ffn_norm.forward_gpu(&h, gpu);
                (h, normed2)
            }
        };
        let up = self.ffn_up.forward(&ffn_input);
        let gate = self.ffn_gate.forward(&ffn_input);
        let gated = silu(&gate);
        let activated = hadamard(&gated, &up);
        let ffn_out = self.ffn_down.forward(&activated);

        gpu_add(&residual2, &ffn_out, gpu)
    }

    /// Create a BitTransformerLayer from an fp32 TransformerLayer by quantizing weights to 1-bit.
    pub fn from_fp32_layer(layer: &TransformerLayer) -> Self {
        Self::from_fp32_layer_q(layer, &QuantConfig::ternary())
    }

    /// Like [`from_fp32_layer`], honoring the config's outlier fraction.
    pub fn from_fp32_layer_q(layer: &TransformerLayer, config: &QuantConfig) -> Self {
        let layer_config = layer.config.clone();

        // Convert fp32 Linear layers to 1-bit BitLinear
        let q_proj = BitLinear::from_linear_with_config(&layer.attention.q_proj, config);
        let k_proj = BitLinear::from_linear_with_config(&layer.attention.k_proj, config);
        let v_proj = BitLinear::from_linear_with_config(&layer.attention.v_proj, config);
        let o_proj = BitLinear::from_linear_with_config(&layer.attention.o_proj, config);

        let attention = BitAttention::new(
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            layer.attention.q_norm.clone(),
            layer.attention.k_norm.clone(),
            layer_config.clone(),
        );

        let ffn_up = BitLinear::from_linear_with_config(&layer.ffn_up, config);
        let ffn_gate = BitLinear::from_linear_with_config(&layer.ffn_gate, config);
        let ffn_down = BitLinear::from_linear_with_config(&layer.ffn_down, config);

        Self {
            attention,
            attn_norm: layer.attn_norm.clone(),
            ffn_up,
            ffn_gate,
            ffn_down,
            ffn_norm: layer.ffn_norm.clone(),
            sub_ln: if layer_config.sub_ln {
                Some(RmsNorm::new(
                    Tensor::ones(&[layer_config.hidden_size], DType::F32),
                    layer_config.norm_eps,
                ))
            } else {
                None
            },
            config: layer_config,
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

#[allow(clippy::too_many_arguments)]
fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    kv_seq_len: usize,
    slot: usize,
    position: usize,
    attn_scale: f32,
    softcap: f32,
) -> Tensor {
    // Cache tensors are [batch, num_kv_heads, max_seq_len, head_dim]; a bare
    // (non-cached) k/v is [num_heads, seq_len, head_dim].
    let batch_layout = k.shape().len() == 4;
    let kv_stride = if batch_layout {
        k.shape()[2]
    } else {
        k.shape()[1]
    };
    let kv_groups = num_heads / num_kv_heads;

    let mut output = Tensor::zeros(&[num_heads, seq_len, head_dim], DType::F32);

    let q_ptr = q.as_f32_slice();
    let k_ptr = k.as_f32_slice();
    let v_ptr = v.as_f32_slice();
    let out_ptr = output.as_f32_slice_mut();

    for h in 0..num_heads {
        let kv_h = h / kv_groups;
        let head_base = if batch_layout {
            (slot * num_kv_heads + kv_h) * kv_stride * head_dim
        } else {
            kv_h * kv_stride * head_dim
        };
        for pos_q in 0..seq_len {
            let q_row = &q_ptr[h * seq_len * head_dim + pos_q * head_dim..][..head_dim];

            // Causal mask: a query may only attend to keys at or before its own
            // position. Cache keys are absolute; bare keys are block-relative.
            let max_k = if batch_layout {
                position + pos_q
            } else {
                pos_q
            };
            let attn_len = (max_k + 1).min(kv_seq_len);

            // Compute attention scores
            let mut scores = Vec::with_capacity(attn_len);
            for pos_k in 0..attn_len {
                let k_row = &k_ptr[head_base + pos_k * head_dim..][..head_dim];
                let dot = q_row.iter().zip(k_row.iter()).map(|(a, b)| a * b).sum::<f32>();
                let score = if softcap > 0.0 {
                    let s = dot * attn_scale;
                    softcap * (s / softcap).tanh()
                } else {
                    dot * attn_scale
                };
                scores.push(score);
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
                let v_row = &v_ptr[head_base + pos_k * head_dim..][..head_dim];
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Model;
    use crate::sampler::Sampler;

    /// Manual `x - RMSNorm(x)` with unit (ones) weights, for verifying the
    /// SubLN math against an independent implementation.
    fn manual_sub_ln(input: &Tensor, eps: f32) -> Tensor {
        let hidden = input.shape().last().copied().unwrap_or(input.num_elements());
        let slice = input.as_f32_slice();
        let mean_sq: f64 = slice.iter().map(|v| (v * v) as f64).sum::<f64>() / hidden as f64;
        let rms = (mean_sq + eps as f64).sqrt();
        let mut out = input.clone();
        let o = out.as_f32_slice_mut();
        for v in o.iter_mut() {
            *v = *v - (*v as f64 / rms) as f32;
        }
        out
    }

    #[test]
    fn test_sub_ln_math() {
        // The layer computes block input = x - RMSNorm(x).
        let hidden = 8;
        let data: Vec<f32> = (0..hidden).map(|i| (i as f32 - 3.0) * 0.5).collect();
        let input = Tensor::from_slice(&data, &[1, hidden]);
        let sub_ln = RmsNorm::new(Tensor::ones(&[hidden], DType::F32), 1e-5);
        let normed = sub_ln.forward(&input);
        let got = input.sub(&normed).unwrap();
        let expected = manual_sub_ln(&input, 1e-5);

        let g = got.as_f32_slice();
        let e = expected.as_f32_slice();
        for i in 0..hidden {
            assert!(
                (g[i] - e[i]).abs() < 1e-5,
                "i={}: got {} expected {}",
                i,
                g[i],
                e[i]
            );
        }
    }

    #[test]
    fn test_sub_ln_layer_forward_and_generate() {
        // Quantized model with the mixed-precision residual enabled must
        // generate through both the single-sequence and batched paths.
        let config = ModelConfig {
            sub_ln: true,
            ..ModelConfig::tiny_test()
        };
        let mut model = Model::new(config.clone());
        model.quantize_to_bit1();

        let layers = model.bit_layers.as_ref().unwrap();
        assert_eq!(layers.len(), config.num_layers);
        assert!(
            layers.iter().all(|l| l.sub_ln.is_some()),
            "every quantized layer should carry a SubLN norm when config.sub_ln"
        );

        let sampler = Sampler::greedy();
        let gen = model.generate(&[0, 1, 2], 5, &sampler);
        assert_eq!(gen.len(), 5);

        model.clear_cache();
        let p = [vec![0u32, 1], vec![2u32, 3]];
        let refs: Vec<&[u32]> = p.iter().map(|v| v.as_slice()).collect();
        let out = model.generate_batch(&refs, 4, &sampler, None);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 4);
        assert_eq!(out[1].len(), 4);
    }
}