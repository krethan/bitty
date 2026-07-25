use crate::config::ModelConfig;
use crate::layers::Linear;
use crate::GpuContext;
use bitllm_tensor::simd;
use bitllm_tensor::{DType, Tensor};

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
        let head_dim = new_k.shape()[2];
        let cache_seq_len = self.k[layer_idx].shape()[1];

        for h in 0..num_heads {
            for d in 0..head_dim {
                let val_k = new_k.get_flat_f32(h * head_dim + d);
                self.k[layer_idx].set_flat_f32(
                    h * cache_seq_len * head_dim + position * head_dim + d,
                    val_k,
                );

                let val_v = new_v.get_flat_f32(h * head_dim + d);
                self.v[layer_idx].set_flat_f32(
                    h * cache_seq_len * head_dim + position * head_dim + d,
                    val_v,
                );
            }
        }
    }

    pub fn get_kv(&self, layer_idx: usize) -> (&Tensor, &Tensor) {
        (&self.k[layer_idx], &self.v[layer_idx])
    }

    pub fn get_kv_used(&self, layer_idx: usize) -> (Tensor, Tensor) {
        let kv_len = self.seq_len.max(1);
        let num_heads = self.k[layer_idx].shape()[0];
        let head_dim = self.k[layer_idx].shape()[2];
        let mut k_out = Tensor::zeros(&[num_heads, kv_len, head_dim], DType::F32);
        let mut v_out = Tensor::zeros(&[num_heads, kv_len, head_dim], DType::F32);

        for h in 0..num_heads {
            for pos in 0..kv_len {
                for d in 0..head_dim {
                    let src_idx = h * self.k[layer_idx].shape()[1] * head_dim + pos * head_dim + d;
                    let dst_idx = h * kv_len * head_dim + pos * head_dim + d;
                    k_out.set_flat_f32(dst_idx, self.k[layer_idx].get_flat_f32(src_idx));
                    v_out.set_flat_f32(dst_idx, self.v[layer_idx].get_flat_f32(src_idx));
                }
            }
        }

        (k_out, v_out)
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
}

impl Attention {
    pub fn new(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
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
        let mut v_reshaped = reshape_for_attention(&v, num_kv_heads, head_dim);

        gpu_rope(
            &mut q_reshaped,
            &mut k_reshaped,
            position,
            head_dim,
            self.config.rope_theta,
            gpu,
        );

        if let Some(cache) = cache {
            cache.update(layer_idx, &k_reshaped, &v_reshaped, position);
            let (used_k, used_v) = cache.get_kv_used(layer_idx);
            k_reshaped = used_k;
            v_reshaped = used_v;
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

        let output_flat = output.flatten();
        self.o_proj
            .forward_gpu(&output_flat.reshape(&[seq_len, hidden_size]), gpu)
    }
}

fn reshape_for_attention(tensor: &Tensor, num_heads: usize, head_dim: usize) -> Tensor {
    let seq_len = tensor.shape()[0];
    let mut result = Tensor::zeros(&[num_heads, seq_len, head_dim], DType::F32);

    for h in 0..num_heads {
        for pos in 0..seq_len {
            for d in 0..head_dim {
                let val = tensor.get_flat_f32(pos * num_heads * head_dim + h * head_dim + d);
                result.set_flat_f32(h * seq_len * head_dim + pos * head_dim + d, val);
            }
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

    let q_slice = q.as_f32_slice();
    let k_slice = k.as_f32_slice();
    let v_slice = v.as_f32_slice();
    let out_slice = output.as_f32_slice_mut();

    for h in 0..num_heads {
        let kv_h = h / kv_groups;
        for pos_q in 0..seq_len {
            let q_row = &q_slice[h * seq_len * head_dim + pos_q * head_dim..][..head_dim];

            let mut scores: Vec<f32> = Vec::with_capacity(kv_seq_len);
            let mut max_val: f32 = f32::NEG_INFINITY;

            for pos_k in 0..kv_seq_len {
                let k_row = &k_slice[kv_h * kv_seq_len * head_dim + pos_k * head_dim..][..head_dim];
                let dot = simd::f32_dot(q_row, k_row);
                let score = dot / scale;
                scores.push(score);
                if score > max_val {
                    max_val = score;
                }
            }

            let mut exp_scores = vec![0.0f32; kv_seq_len];
            simd::f32_exp(&scores, &mut exp_scores);
            let sum_exp: f32 = simd::f32_sum(&exp_scores);
            let inv_sum = 1.0 / sum_exp;

            for s in exp_scores.iter_mut() {
                *s *= inv_sum;
            }

            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (pos_k, score) in exp_scores.iter().enumerate() {
                    let v_val = v_slice[kv_h * kv_seq_len * head_dim + pos_k * head_dim + d];
                    acc += score * v_val;
                }
                out_slice[h * seq_len * head_dim + pos_q * head_dim + d] = acc;
            }
        }
    }

    output
}

pub fn apply_rotary_emb(x: &Tensor, position: usize, head_dim: usize, theta: f32) -> Tensor {
    let seq_len = x.shape()[1];
    let num_heads = x.shape()[0];
    let mut result = x.clone();

    for i in 0..(head_dim / 2) {
        let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);

        for h in 0..num_heads {
            for pos in 0..seq_len {
                let angle = (position + pos) as f32 * freq;
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                let idx_even = h * seq_len * head_dim + pos * head_dim + 2 * i;
                let idx_odd = h * seq_len * head_dim + pos * head_dim + 2 * i + 1;

                let x_even = x.get_flat_f32(idx_even);
                let x_odd = x.get_flat_f32(idx_odd);

                result.set_flat_f32(idx_even, x_even * cos_val - x_odd * sin_val);
                result.set_flat_f32(idx_odd, x_even * sin_val + x_odd * cos_val);
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
    apply_rotary_inplace_inner(q, position, head_dim, theta);
    apply_rotary_inplace_inner(k, position, head_dim, theta);
}

fn apply_rotary_inplace_inner(x: &mut Tensor, position: usize, head_dim: usize, theta: f32) {
    let seq_len = x.shape()[1];
    let num_heads = x.shape()[0];

    for i in 0..(head_dim / 2) {
        let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);

        for h in 0..num_heads {
            for pos in 0..seq_len {
                let angle = (position + pos) as f32 * freq;
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                let idx_even = h * seq_len * head_dim + pos * head_dim + 2 * i;
                let idx_odd = h * seq_len * head_dim + pos * head_dim + 2 * i + 1;

                let x_even = x.get_flat_f32(idx_even);
                let x_odd = x.get_flat_f32(idx_odd);

                x.set_flat_f32(idx_even, x_even * cos_val - x_odd * sin_val);
                x.set_flat_f32(idx_odd, x_even * sin_val + x_odd * cos_val);
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
    apply_rotary_emb_inplace(q, k, position, head_dim, theta);
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
