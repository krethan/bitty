use crate::config::ModelConfig;
use crate::layers::Linear;
use crate::GpuContext;
use bitllm_tensor::simd;
use bitllm_tensor::{DType, Tensor};
use std::cell::RefCell;


/// Precomputed RoPE cos/sin table for efficient lookups.
pub struct RoPECache {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub theta: f32,
}

impl RoPECache {
    pub fn new(max_seq_len: usize, head_dim: usize, theta: f32) -> Self {
        let half = head_dim / 2;
        let mut cos = vec![0.0f32; max_seq_len * half];
        let mut sin = vec![0.0f32; max_seq_len * half];
        for pos in 0..max_seq_len {
            for i in 0..half {
                let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                cos[pos * half + i] = angle.cos();
                sin[pos * half + i] = angle.sin();
            }
        }
        Self { cos, sin, head_dim, max_seq_len, theta }
    }
}

pub struct KvCache {
    pub k: Vec<Tensor>,
    pub v: Vec<Tensor>,
    pub seq_len: usize,
}

impl KvCache {
    pub fn new(
        num_layers: usize,
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let k = (0..num_layers)
            .map(|_| Tensor::zeros(&[num_kv_heads, max_seq_len, head_dim], DType::F32))
            .collect();
        let v = (0..num_layers)
            .map(|_| Tensor::zeros(&[num_kv_heads, max_seq_len, head_dim], DType::F32))
            .collect();
        Self { k, v, seq_len: 0 }
    }

    pub fn update(&mut self, layer_idx: usize, new_k: &Tensor, new_v: &Tensor, position: usize) {
        let num_heads = new_k.shape()[0];
        let seq_len = new_k.shape()[1];
        let head_dim = new_k.shape()[2];
        let cache_seq_len = self.k[layer_idx].shape()[1];

        let k_data = new_k.as_f32_slice();
        let v_data = new_v.as_f32_slice();
        let cache_k = self.k[layer_idx].as_f32_slice_mut();
        let cache_v = self.v[layer_idx].as_f32_slice_mut();

        for h in 0..num_heads {
            for pos in 0..seq_len {
                let src_base = h * seq_len * head_dim + pos * head_dim;
                let dst_base = h * cache_seq_len * head_dim + (position + pos) * head_dim;
                cache_k[dst_base..dst_base + head_dim]
                    .copy_from_slice(&k_data[src_base..src_base + head_dim]);
                cache_v[dst_base..dst_base + head_dim]
                    .copy_from_slice(&v_data[src_base..src_base + head_dim]);
            }
        }
    }

    pub fn get_kv(&self, layer_idx: usize) -> (&Tensor, &Tensor) {
        (&self.k[layer_idx], &self.v[layer_idx])
    }

    /// Returns references to the KV cache tensors and the number of
    /// positions that are actually populated.  This avoids copying the
    /// entire cache on every attention call.
    pub fn get_kv_used(&self, layer_idx: usize) -> (&Tensor, &Tensor, usize) {
        let kv_len = self.seq_len.max(1);
        (&self.k[layer_idx], &self.v[layer_idx], kv_len)
    }

    pub fn get_seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn advance(&mut self, n: usize) {
        self.seq_len += n;
    }
}

pub struct Attention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub config: ModelConfig,
    scores: RefCell<Vec<f32>>,
    acc: RefCell<Vec<f32>>,
}

impl Attention {
    pub fn new(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
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
        self.forward_gpu(input, cache, layer_idx, position, None)
    }

    pub fn forward_gpu(
        &self,
        input: &Tensor,
        cache: Option<&mut KvCache>,
        layer_idx: usize,
        position: usize,
        gpu: Option<&GpuContext>,
    ) -> Tensor {
        self.forward_gpu_with_rope_cache(input, cache, layer_idx, position, gpu, None)
    }

    pub fn forward_gpu_with_rope_cache(
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

        let q = self.q_proj.forward_gpu(input, gpu);
        let k = self.k_proj.forward_gpu(input, gpu);
        let v = self.v_proj.forward_gpu(input, gpu);

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

        // Ensure scratch buffers are large enough
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
        self.o_proj
            .forward_gpu(&reshaped, gpu)
    }
}

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
                let dot = simd::f32_dot(q_row, k_row);
                let score = dot / scale;
                scores[pos_k] = score;
                if score > max_val {
                    max_val = score;
                }
            }

            simd::f32_add_scalar_inplace(&mut scores[..kv_seq_len], -max_val);
            simd::f32_exp_inplace(&mut scores[..kv_seq_len]);
            let sum_exp: f32 = simd::f32_sum(&scores[..kv_seq_len]);
            simd::f32_scale_inplace(&mut scores[..kv_seq_len], 1.0 / sum_exp);

            acc[..head_dim].fill(0.0);
            for (pos_k, score) in scores[..kv_seq_len].iter().enumerate() {
                let v_row = &v_slice[kv_h * kv_stride * head_dim + pos_k * head_dim..][..head_dim];
                simd::f32_axpy(v_row, *score, &mut acc[..head_dim]);
            }
            let out_row = &mut out_slice[h * seq_len * head_dim + pos_q * head_dim..][..head_dim];
            out_row.copy_from_slice(&acc[..head_dim]);
        }
    }

    output
}

pub fn apply_rotary_emb(x: &Tensor, position: usize, head_dim: usize, theta: f32) -> Tensor {
    apply_rotary_emb_with_cache(x, position, head_dim, theta, None)
}

pub fn apply_rotary_emb_with_cache(
    x: &Tensor,
    position: usize,
    head_dim: usize,
    theta: f32,
    cache: Option<&RoPECache>,
) -> Tensor {
    let seq_len = x.shape()[1];
    let num_heads = x.shape()[0];
    let half = head_dim / 2;

    let mut result = Tensor::zeros(&[num_heads, seq_len, head_dim], DType::F32);
    let x_slice = x.as_f32_slice();
    let out_slice = result.as_f32_slice_mut();

    let (cos_table, sin_table) = cache
        .filter(|c| c.head_dim == head_dim && c.theta == theta)
        .map(|c| (&c.cos[..], &c.sin[..]))
        .unwrap_or((&[], &[]));

    for h in 0..num_heads {
        for pos in 0..seq_len {
            let base = h * seq_len * head_dim + pos * head_dim;
            for i in 0..half {
                let (cos_val, sin_val) = if !cos_table.is_empty() {
                    let idx = (position + pos) * half + i;
                    (cos_table[idx], sin_table[idx])
                } else {
                    let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
                    let angle = (position + pos) as f32 * freq;
                    (angle.cos(), angle.sin())
                };

                let idx_even = base + 2 * i;
                let idx_odd = base + 2 * i + 1;

                let x_even = x_slice[idx_even];
                let x_odd = x_slice[idx_odd];

                out_slice[idx_even] = x_even * cos_val - x_odd * sin_val;
                out_slice[idx_odd] = x_even * sin_val + x_odd * cos_val;
            }
        }
    }

    result
}

pub fn apply_rotary_emb_inplace(
    q: &mut Tensor,
    k: &mut Tensor,
    position: usize,
    head_dim: usize,
    theta: f32,
) {
    apply_rotary_inplace_inner(q, position, head_dim, theta, None);
    apply_rotary_inplace_inner(k, position, head_dim, theta, None);
}

pub fn apply_rotary_emb_inplace_with_cache(
    q: &mut Tensor,
    k: &mut Tensor,
    position: usize,
    head_dim: usize,
    theta: f32,
    cache: Option<&RoPECache>,
) {
    apply_rotary_inplace_inner(q, position, head_dim, theta, cache);
    apply_rotary_inplace_inner(k, position, head_dim, theta, cache);
}

fn apply_rotary_inplace_inner(x: &mut Tensor, position: usize, head_dim: usize, theta: f32, cache: Option<&RoPECache>) {
    let seq_len = x.shape()[1];
    let num_heads = x.shape()[0];
    let half = head_dim / 2;
    let x_slice = x.as_f32_slice_mut();

    let (cos_table, sin_table) = cache
        .filter(|c| c.head_dim == head_dim && c.theta == theta)
        .map(|c| (&c.cos[..], &c.sin[..]))
        .unwrap_or((&[], &[]));

    for h in 0..num_heads {
        for pos in 0..seq_len {
            let base = h * seq_len * head_dim + pos * head_dim;
            for i in 0..half {
                let (cos_val, sin_val) = if !cos_table.is_empty() {
                    let idx = (position + pos) * half + i;
                    (cos_table[idx], sin_table[idx])
                } else {
                    let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
                    let angle = (position + pos) as f32 * freq;
                    (angle.cos(), angle.sin())
                };

                let idx_even = base + 2 * i;
                let idx_odd = base + 2 * i + 1;

                let x_even = x_slice[idx_even];
                let x_odd = x_slice[idx_odd];

                x_slice[idx_even] = x_even * cos_val - x_odd * sin_val;
                x_slice[idx_odd] = x_even * sin_val + x_odd * cos_val;
            }
        }
    }
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
    apply_rotary_emb_inplace_with_cache(q, k, position, head_dim, theta, rope_cache);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;

    #[test]
    fn test_rope_preserves_norm() {
        let head_dim = 8;
        let num_heads = 2;
        let seq_len = 4;
        let data: Vec<f32> = (0..(num_heads * seq_len * head_dim) as i32)
            .map(|i| i as f32 * 0.1)
            .collect();
        let x = Tensor::from_slice(&data, &[num_heads, seq_len, head_dim]);

        let input_norm: f32 = data.iter().map(|v| v * v).sum::<f32>().sqrt();

        let result = apply_rotary_emb(&x, 0, head_dim, 10000.0);
        let output_norm: f32 = (0..result.num_elements())
            .map(|i| result.get_flat_f32(i).powi(2))
            .sum::<f32>()
            .sqrt();

        let diff = (input_norm - output_norm).abs();
        assert!(
            diff < 1e-4,
            "RoPE changed vector norm: input={}, output={}",
            input_norm,
            output_norm
        );
    }

    #[test]
    fn test_rope_different_positions_give_different_outputs() {
        let head_dim = 8;
        let num_heads = 1;
        let seq_len = 1;
        let data = vec![1.0; head_dim];
        let x = Tensor::from_slice(&data, &[num_heads, seq_len, head_dim]);

        let r0 = apply_rotary_emb(&x, 0, head_dim, 10000.0);
        let r1 = apply_rotary_emb(&x, 1, head_dim, 10000.0);

        let mut differ = false;
        for i in 0..head_dim {
            if (r0.get_flat_f32(i) - r1.get_flat_f32(i)).abs() > 1e-6 {
                differ = true;
                break;
            }
        }
        assert!(
            differ,
            "Different positions should produce different outputs"
        );
    }

    #[test]
    fn test_rope_position_zero() {
        let head_dim = 4;
        let x = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], &[1, 1, head_dim]);
        let result = apply_rotary_emb(&x, 0, head_dim, 10000.0);

        let freq = 1.0 / 10000.0_f32.powf(0.0);
        let angle = 0.0 * freq;
        let cos_val = angle.cos();
        let sin_val = angle.sin();

        let expected_even = 1.0 * cos_val - 0.0 * sin_val;
        let expected_odd = 1.0 * sin_val + 0.0 * cos_val;
        let diff = (result.get_flat_f32(0) - expected_even).abs()
            + (result.get_flat_f32(1) - expected_odd).abs();
        assert!(diff < 1e-6, "RoPE at position 0 failed: diff={}", diff);
    }

    #[test]
    fn test_rope_inplace_matches_outplace() {
        let head_dim = 8;
        let num_heads = 2;
        let seq_len = 3;
        let data: Vec<f32> = (0..(num_heads * seq_len * head_dim) as i32)
            .map(|i| i as f32 * 0.1)
            .collect();
        let x = Tensor::from_slice(&data, &[num_heads, seq_len, head_dim]);

        let mut q = x.clone();
        let mut k = x.clone();
        apply_rotary_emb_inplace(&mut q, &mut k, 5, head_dim, 10000.0);

        let outplace_q = apply_rotary_emb(&x, 5, head_dim, 10000.0);

        for i in 0..outplace_q.num_elements() {
            let diff = (outplace_q.get_flat_f32(i) - q.get_flat_f32(i)).abs();
            assert!(
                diff < 1e-6,
                "Inplace/outplace mismatch at {}: outplace={}, inplace={}",
                i,
                outplace_q.get_flat_f32(i),
                q.get_flat_f32(i)
            );
        }
    }

    #[test]
    fn test_attention_with_rope() {
        let config = ModelConfig::tiny_test();
        let hidden = config.hidden_size;
        let head_dim = config.head_dim();

        let attention = Attention::new(
            Linear::new(Tensor::random(&[hidden, hidden], DType::F32), None),
            Linear::new(
                Tensor::random(&[hidden, config.num_kv_heads() * head_dim], DType::F32),
                None,
            ),
            Linear::new(
                Tensor::random(&[hidden, config.num_kv_heads() * head_dim], DType::F32),
                None,
            ),
            Linear::new(Tensor::random(&[hidden, hidden], DType::F32), None),
            config.clone(),
        );

        let input = Tensor::random(&[1, hidden], DType::F32);
        let mut cache = KvCache::new(
            config.num_layers,
            config.max_seq_len,
            config.num_kv_heads(),
            head_dim,
        );

        let output = attention.forward(&input, Some(&mut cache), 0, 0);
        assert_eq!(output.shape(), &[1, hidden]);

        let output2 = attention.forward(&input, Some(&mut cache), 0, 1);
        assert_eq!(output2.shape(), &[1, hidden]);

        assert_eq!(
            cache.get_seq_len(),
            0,
            "Attention does not advance cache - Model does"
        );
    }
}
