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
    pub scaling_factor: f32,
}

impl RoPECache {
    pub fn new(max_seq_len: usize, head_dim: usize, theta: f32) -> Self {
        Self::with_scaling(max_seq_len, head_dim, theta, 1.0)
    }

    pub fn with_scaling(max_seq_len: usize, head_dim: usize, theta: f32, scaling_factor: f32) -> Self {
        let half = head_dim / 2;
        let mut cos = vec![0.0f32; max_seq_len * half];
        let mut sin = vec![0.0f32; max_seq_len * half];
        for pos in 0..max_seq_len {
            for i in 0..half {
                let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
                // Apply scaling: for linear scaling, divide position by factor
                let scaled_pos = pos as f32 / scaling_factor;
                let angle = scaled_pos * freq;
                cos[pos * half + i] = angle.cos();
                sin[pos * half + i] = angle.sin();
            }
        }
        Self { cos, sin, head_dim, max_seq_len, theta, scaling_factor }
    }
}

/// KV cache with a leading batch (slot) dimension.
///
/// Tensors are laid out as `[batch, num_kv_heads, max_seq_len, head_dim]`.
/// Slot 0 of a batch-1 cache matches the previous single-sequence layout for
/// all positions, so single-sequence kernels can index slot 0 unchanged.
pub struct KvCache {
    pub k: Vec<Tensor>,
    pub v: Vec<Tensor>,
    pub batch: usize,
    pub seq_lens: Vec<usize>,
}

impl KvCache {
    pub fn new(
        num_layers: usize,
        batch: usize,
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let k = (0..num_layers)
            .map(|_| {
                Tensor::zeros(
                    &[batch, num_kv_heads, max_seq_len, head_dim],
                    DType::F32,
                )
            })
            .collect();
        let v = (0..num_layers)
            .map(|_| {
                Tensor::zeros(
                    &[batch, num_kv_heads, max_seq_len, head_dim],
                    DType::F32,
                )
            })
            .collect();
        Self {
            k,
            v,
            batch,
            seq_lens: vec![0; batch],
        }
    }

    /// Copy a contiguous `[num_heads, seq_len, head_dim]` block into the cache
    /// slot at `position`. `num_heads` must equal the number of KV heads.
    pub fn update(&mut self, layer_idx: usize, slot: usize, new_k: &Tensor, new_v: &Tensor, position: usize) {
        let num_heads = new_k.shape()[0];
        let seq_len = new_k.shape()[1];
        let head_dim = new_k.shape()[2];
        let cache_num_kv_heads = self.k[layer_idx].shape()[1];
        let cache_seq_len = self.k[layer_idx].shape()[2];

        let k_data = new_k.as_f32_slice();
        let v_data = new_v.as_f32_slice();
        let cache_k = self.k[layer_idx].as_f32_slice_mut();
        let cache_v = self.v[layer_idx].as_f32_slice_mut();

        for h in 0..num_heads {
            for pos in 0..seq_len {
                let src_base = h * seq_len * head_dim + pos * head_dim;
                let dst_base = (slot * cache_num_kv_heads + h) * cache_seq_len * head_dim
                    + (position + pos) * head_dim;
                cache_k[dst_base..dst_base + head_dim]
                    .copy_from_slice(&k_data[src_base..src_base + head_dim]);
                cache_v[dst_base..dst_base + head_dim]
                    .copy_from_slice(&v_data[src_base..src_base + head_dim]);
            }
        }
    }

    /// Copy a batched block into the cache. `new_k`/`new_v` are laid out as
    /// `[num_heads, batch, head_dim]` (one current token per batch slot) and
    /// `positions` holds each slot's absolute position.
    pub fn update_batch(
        &mut self,
        layer_idx: usize,
        new_k: &Tensor,
        new_v: &Tensor,
        positions: &[usize],
    ) {
        let num_heads = new_k.shape()[0];
        let batch = new_k.shape()[1];
        let head_dim = new_k.shape()[2];
        let cache_num_kv_heads = self.k[layer_idx].shape()[1];
        let cache_seq_len = self.k[layer_idx].shape()[2];

        let k_data = new_k.as_f32_slice();
        let v_data = new_v.as_f32_slice();
        let cache_k = self.k[layer_idx].as_f32_slice_mut();
        let cache_v = self.v[layer_idx].as_f32_slice_mut();

        for (b, &position) in positions.iter().enumerate() {
            for h in 0..num_heads {
                let src_base = (h * batch + b) * head_dim;
                let dst_base = (b * cache_num_kv_heads + h) * cache_seq_len * head_dim
                    + position * head_dim;
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
    /// positions actually populated for the given slot. This avoids copying
    /// the entire cache on every attention call.
    pub fn get_kv_used(&self, layer_idx: usize, slot: usize) -> (&Tensor, &Tensor, usize) {
        let kv_len = self.seq_lens[slot].max(1);
        (&self.k[layer_idx], &self.v[layer_idx], kv_len)
    }

    /// Number of positions populated per slot.
    pub fn kv_lens(&self) -> &[usize] {
        &self.seq_lens
    }

    pub fn seq_len(&self, slot: usize) -> usize {
        self.seq_lens[slot]
    }

    /// Mark `len` positions starting at `position` as populated for a slot.
    /// Call before running a forward so attention sees the full written range.
    pub fn reserve(&mut self, slot: usize, position: usize, len: usize) {
        self.seq_lens[slot] = position + len;
    }

    pub fn clear(&mut self) {
        for s in self.seq_lens.iter_mut() {
            *s = 0;
        }
    }

    /// Reset a single slot's populated length to zero so the slot can be
    /// reused by a new sequence (continuous batching).
    pub fn clear_slot(&mut self, slot: usize) {
        self.seq_lens[slot] = 0;
    }

    /// Roll a slot's populated length back to `len` (no larger than its
    /// current length). Used by speculative decoding to discard rejected
    /// draft tokens; the stale KV rows are simply ignored and overwritten.
    pub fn truncate(&mut self, slot: usize, len: usize) {
        self.seq_lens[slot] = len.min(self.seq_lens[slot]);
    }
}

pub struct Attention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    /// Gemma-style per-head Q norm (RMSNorm over each head, applied before RoPE).
    pub q_norm: Option<crate::layers::RmsNorm>,
    /// Gemma-style per-head K norm.
    pub k_norm: Option<crate::layers::RmsNorm>,
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
        Self::new_with_qk_norm(q_proj, k_proj, v_proj, o_proj, None, None, config)
    }

    /// Like [`Attention::new`], with optional per-head Q/K norms (Gemma).
    pub fn new_with_qk_norm(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        q_norm: Option<crate::layers::RmsNorm>,
        k_norm: Option<crate::layers::RmsNorm>,
        config: ModelConfig,
    ) -> Self {
        let max_seq_len = config.max_seq_len;
        let head_dim = config.head_dim();
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
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
        self.forward_gpu_with_rope_cache(input, cache, layer_idx, 0, position, gpu, None)
    }

    /// Batched decode: `input` is `[batch, hidden_size]` (one current token per
    /// batch slot). Each row is RoPE'd at its own absolute `positions[b]`, its
    /// K/V is written into its cache slot, and it attends only to its own
    /// slot's populated positions.
    pub fn forward_batch_gpu(
        &self,
        input: &Tensor,
        mut cache: Option<&mut KvCache>,
        layer_idx: usize,
        positions: &[usize],
        gpu: Option<&GpuContext>,
        rope_cache: Option<&RoPECache>,
    ) -> Tensor {
        let batch = input.shape()[0];
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads();
        let head_dim = self.config.head_dim();
        let theta = self.config.rope_theta;

        let q = self.q_proj.forward_gpu(input, gpu);
        let k = self.k_proj.forward_gpu(input, gpu);
        let v = self.v_proj.forward_gpu(input, gpu);

        let mut q_reshaped = reshape_for_attention(&q, num_heads, head_dim);
        let mut k_reshaped = reshape_for_attention(&k, num_kv_heads, head_dim);
        let v_reshaped = reshape_for_attention(&v, num_kv_heads, head_dim);

        if let Some(ref q_norm) = self.q_norm {
            apply_qk_norm(&mut q_reshaped, q_norm, num_heads, head_dim);
        }
        if let Some(ref k_norm) = self.k_norm {
            apply_qk_norm(&mut k_reshaped, k_norm, num_kv_heads, head_dim);
        }

        if self.config.use_rope {
            apply_rotary_emb_batch(&mut q_reshaped, positions, head_dim, theta, rope_cache);
            apply_rotary_emb_batch(&mut k_reshaped, positions, head_dim, theta, rope_cache);
        }

        let max_seq_len = self.config.max_seq_len;
        let window = self.config.sliding_window;
        let mut scores_buf = self.scores.borrow_mut();
        let mut acc_buf = self.acc.borrow_mut();
        if scores_buf.len() < max_seq_len {
            scores_buf.resize(max_seq_len, 0.0);
        }
        if acc_buf.len() < head_dim {
            acc_buf.resize(head_dim, 0.0);
        }

        let output = match cache.as_mut() {
            Some(c) => {
                c.update_batch(layer_idx, &k_reshaped, &v_reshaped, positions);
                let kv_lens = c.kv_lens();
                scaled_dot_product_attention_batched(
                    &q_reshaped,
                    &c.k[layer_idx],
                    &c.v[layer_idx],
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    batch,
                    kv_lens,
                    window,
                    &mut scores_buf[..max_seq_len],
                    &mut acc_buf[..head_dim],
                    self.config.attn_logit_scale(),
                    self.config.attn_logit_softcap(),
                )
            }
            None => {
                let ones: Vec<usize> = vec![1; batch];
                scaled_dot_product_attention_batched(
                    &q_reshaped,
                    &k_reshaped,
                    &v_reshaped,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    batch,
                    &ones,
                    window,
                    &mut scores_buf[..max_seq_len],
                    &mut acc_buf[..head_dim],
                    self.config.attn_logit_scale(),
                    self.config.attn_logit_softcap(),
                )
            }
        };

        let reshaped = sdp_batched_output_to_hidden(&output, batch, num_heads, head_dim);
        self.o_proj.forward_gpu(&reshaped, gpu)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_gpu_with_rope_cache(
        &self,
        input: &Tensor,
        mut cache: Option<&mut KvCache>,
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

        let q = self.q_proj.forward_gpu(input, gpu);
        let k = self.k_proj.forward_gpu(input, gpu);
        let v = self.v_proj.forward_gpu(input, gpu);

        let mut q_reshaped = reshape_for_attention(&q, num_heads, head_dim);
        let mut k_reshaped = reshape_for_attention(&k, num_kv_heads, head_dim);
        let v_reshaped = reshape_for_attention(&v, num_kv_heads, head_dim);

        if let Some(ref q_norm) = self.q_norm {
            apply_qk_norm(&mut q_reshaped, q_norm, num_heads, head_dim);
        }
        if let Some(ref k_norm) = self.k_norm {
            apply_qk_norm(&mut k_reshaped, k_norm, num_kv_heads, head_dim);
        }

        if self.config.use_rope {
            gpu_rope(
                &mut q_reshaped,
                &mut k_reshaped,
                position,
                head_dim,
                self.config.rope_theta,
                gpu,
                rope_cache,
            );
        }

        if let Some(c) = cache.as_mut() {
            c.update(layer_idx, slot, &k_reshaped, &v_reshaped, position);
        }
        let (k_ref, v_ref, kv_seq_len) = match cache.as_ref() {
            Some(c) => c.get_kv_used(layer_idx, slot),
            None => (&k_reshaped, &v_reshaped, k_reshaped.shape()[1]),
        };

        let window = self.config.sliding_window;
        let kv_start = window.map_or(0, |w| kv_seq_len.saturating_sub(w));

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
            kv_start,
            slot,
            position,
            &mut scores_buf[..kv_seq_len],
            &mut acc_buf[..head_dim],
            self.config.attn_logit_scale(),
            self.config.attn_logit_softcap(),
        );

        let reshaped = sdp_output_to_hidden(&output, seq_len, num_heads, head_dim);
        self.o_proj
            .forward_gpu(&reshaped, gpu)
    }
}

pub(crate) fn reshape_for_attention(tensor: &Tensor, num_heads: usize, head_dim: usize) -> Tensor {
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

/// Apply per-head RMSNorm to a reshaped `[num_heads, seq_len, head_dim]`
/// tensor (Gemma QK-norm). `norm.weight` is `[num_heads, head_dim]`.
pub(crate) fn apply_qk_norm(x: &mut Tensor, norm: &crate::layers::RmsNorm, num_heads: usize, head_dim: usize) {
    let seq_len = x.shape()[1];
    let w = norm.weight.as_f32_slice();
    let eps = norm.eps;
    for h in 0..num_heads {
        for pos in 0..seq_len {
            let base = h * seq_len * head_dim + pos * head_dim;
            let row: Vec<f32> = x
                .as_f32_slice()
                .iter()
                .skip(base)
                .take(head_dim)
                .copied()
                .collect();
            let mut sum_sq = 0.0f64;
            for &v in &row {
                sum_sq += (v as f64) * (v as f64);
            }
            let inv_rms = 1.0 / ((sum_sq / head_dim as f64) as f32 + eps).sqrt();
            let w_row = &w[h * head_dim..(h + 1) * head_dim];
            let gain = if norm.one_centered {
                |w: f32| 1.0 + w
            } else {
                |w: f32| w
            };
            for j in 0..head_dim {
                x.as_f32_slice_mut()[base + j] = row[j] * inv_rms * gain(w_row[j]);
            }
        }
    }
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
    kv_start: usize,
    slot: usize,
    position: usize,
    scores: &mut [f32],
    acc: &mut [f32],
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
    let window_len = kv_seq_len - kv_start;

    let mut output = Tensor::zeros(&[num_heads, seq_len, head_dim], DType::F32);

    let q_slice = q.as_f32_slice();
    let k_slice = k.as_f32_slice();
    let v_slice = v.as_f32_slice();
    let out_slice = output.as_f32_slice_mut();

    for h in 0..num_heads {
        let kv_h = h / kv_groups;
        let head_base = if batch_layout {
            (slot * num_kv_heads + kv_h) * kv_stride * head_dim
        } else {
            kv_h * kv_stride * head_dim
        };
        for pos_q in 0..seq_len {
            let q_row = &q_slice[h * seq_len * head_dim + pos_q * head_dim..][..head_dim];

            // Causal mask: a query may only attend to keys at or before its own
            // position. Cache keys are absolute; bare keys are block-relative.
            let max_k = if batch_layout {
                position + pos_q
            } else {
                pos_q
            };
            let attn_len = window_len.min((max_k + 1).saturating_sub(kv_start));

            let mut max_val: f32 = f32::NEG_INFINITY;

            for t in 0..attn_len {
                let pos_k = kv_start + t;
                let k_row = &k_slice[head_base + pos_k * head_dim..][..head_dim];
                let dot = simd::f32_dot(q_row, k_row);
                let score = scale_softcap(dot * attn_scale, softcap);
                scores[t] = score;
                if score > max_val {
                    max_val = score;
                }
            }

            simd::f32_add_scalar_inplace(&mut scores[..attn_len], -max_val);
            simd::f32_exp_inplace(&mut scores[..attn_len]);
            let sum_exp: f32 = simd::f32_sum(&scores[..attn_len]);
            simd::f32_scale_inplace(&mut scores[..attn_len], 1.0 / sum_exp);

            acc[..head_dim].fill(0.0);
            for (t, score) in scores[..attn_len].iter().enumerate() {
                let pos_k = kv_start + t;
                let v_row = &v_slice[head_base + pos_k * head_dim..][..head_dim];
                simd::f32_axpy(v_row, *score, &mut acc[..head_dim]);
            }
            let out_row = &mut out_slice[h * seq_len * head_dim + pos_q * head_dim..][..head_dim];
            out_row.copy_from_slice(&acc[..head_dim]);
        }
    }

    output
}

/// Apply logit soft-capping `cap * tanh(score / cap)` when `softcap > 0`.
#[inline]
fn scale_softcap(score: f32, softcap: f32) -> f32 {
    if softcap > 0.0 {
        softcap * (score / softcap).tanh()
    } else {
        score
    }
}

/// Reused by the QAT activation recorder (`crate::record`) which needs the
/// bare (non-batched) attention output without owning private buffers.
///
/// Delegates to the scratch-buffer [`scaled_dot_product_attention`] used by
/// inference so recorded activations are bit-identical to the teacher's.
pub(crate) fn scaled_dot_product_attention_owned(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    kv_seq_len: usize,
    kv_start: usize,
    position: usize,
    attn_scale: f32,
    softcap: f32,
) -> Tensor {
    let mut scores = vec![0.0f32; kv_seq_len];
    let mut acc = vec![0.0f32; head_dim];
    scaled_dot_product_attention(
        q,
        k,
        v,
        num_heads,
        num_kv_heads,
        head_dim,
        seq_len,
        kv_seq_len,
        kv_start,
        0,
        position,
        &mut scores,
        &mut acc,
        attn_scale,
        softcap,
    )
}

/// Batched decode SDPA. `q` is `[num_heads, batch, head_dim]`. `k`/`v` are
/// either the cache tensors (`[batch, num_kv_heads, max_seq_len, head_dim]`)
/// with one `kv_lens[b]` per slot, or bare `[num_kv_heads, batch, head_dim]`
/// blocks where each row attends only to itself.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scaled_dot_product_attention_batched(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    batch: usize,
    kv_lens: &[usize],
    window: Option<usize>,
    scores: &mut [f32],
    acc: &mut [f32],
    attn_scale: f32,
    softcap: f32,
) -> Tensor {
    let cache_layout = k.shape().len() == 4;
    let kv_stride = if cache_layout {
        k.shape()[2]
    } else {
        k.shape()[1]
    };
    let kv_groups = num_heads / num_kv_heads;

    let mut output = Tensor::zeros(&[num_heads, batch, head_dim], DType::F32);

    let q_slice = q.as_f32_slice();
    let k_slice = k.as_f32_slice();
    let v_slice = v.as_f32_slice();
    let out_slice = output.as_f32_slice_mut();

    for h in 0..num_heads {
        let kv_h = h / kv_groups;
        for b in 0..batch {
            let kv_len = if cache_layout {
                kv_lens[b].min(kv_stride).max(1)
            } else {
                1
            };
            let kv_start = window.map_or(0, |w| kv_len.saturating_sub(w));
            let window_len = kv_len - kv_start;
            let q_row = &q_slice[(h * batch + b) * head_dim..][..head_dim];
            let head_base = if cache_layout {
                (b * num_kv_heads + kv_h) * kv_stride * head_dim
            } else {
                (kv_h * batch + b) * head_dim
            };

            let mut max_val: f32 = f32::NEG_INFINITY;
            for t in 0..window_len {
                let pos_k = kv_start + t;
                let k_row = &k_slice[head_base + pos_k * head_dim..][..head_dim];
                let dot = simd::f32_dot(q_row, k_row);
                let score = scale_softcap(dot * attn_scale, softcap);
                scores[t] = score;
                if score > max_val {
                    max_val = score;
                }
            }

            simd::f32_add_scalar_inplace(&mut scores[..window_len], -max_val);
            simd::f32_exp_inplace(&mut scores[..window_len]);
            let sum_exp: f32 = simd::f32_sum(&scores[..window_len]);
            simd::f32_scale_inplace(&mut scores[..window_len], 1.0 / sum_exp);

            acc[..head_dim].fill(0.0);
            for (t, score) in scores[..window_len].iter().enumerate() {
                let pos_k = kv_start + t;
                let v_row = &v_slice[head_base + pos_k * head_dim..][..head_dim];
                simd::f32_axpy(v_row, *score, &mut acc[..head_dim]);
            }
            let out_row = &mut out_slice[(h * batch + b) * head_dim..][..head_dim];
            out_row.copy_from_slice(&acc[..head_dim]);
        }
    }

    output
}

/// Permute a `[outer, inner, dim]` attention output into `[inner, outer * dim]`,
/// interleaving so that row `i` holds every head's vector at index `i`. A plain
/// row-major reshape of `[num_heads, seq_len, head_dim]` to `[seq_len, hidden]`
/// would instead group whole heads, scrambling the projections for `seq_len > 1`.
pub(crate) fn permute_outer_inner(output: &Tensor, outer: usize, inner: usize, dim: usize) -> Tensor {
    let hidden = outer * dim;
    let mut result = Tensor::zeros(&[inner, hidden], DType::F32);
    let src = output.as_f32_slice();
    let dst = result.as_f32_slice_mut();
    for o in 0..outer {
        for i in 0..inner {
            let src_base = o * inner * dim + i * dim;
            let dst_base = i * hidden + o * dim;
            dst[dst_base..dst_base + dim].copy_from_slice(&src[src_base..src_base + dim]);
        }
    }
    result
}

/// Apply [`permute_outer_inner`] to the SDPA output so it can feed `o_proj`.
pub(crate) fn sdp_output_to_hidden(
    output: &Tensor,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Tensor {
    permute_outer_inner(output, num_heads, seq_len, head_dim)
}

/// Apply [`permute_outer_inner`] to the batched decode SDPA output
/// (`[num_heads, batch, head_dim]` -> `[batch, hidden]`).
pub(crate) fn sdp_batched_output_to_hidden(
    output: &Tensor,
    batch: usize,
    num_heads: usize,
    head_dim: usize,
) -> Tensor {
    permute_outer_inner(output, num_heads, batch, head_dim)
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

                // Non-interleaved (half-split) pairing: dim `i` is rotated with
                // dim `i + half` (Llama/Gemma/Qwen/SmolLM convention).
                let idx_lo = base + i;
                let idx_hi = base + i + half;

                let x_lo = x_slice[idx_lo];
                let x_hi = x_slice[idx_hi];

                out_slice[idx_lo] = x_lo * cos_val - x_hi * sin_val;
                out_slice[idx_hi] = x_lo * sin_val + x_hi * cos_val;
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

/// Apply RoPE to a batched `[num_heads, batch, head_dim]` tensor, using a
/// per-row absolute position.
pub fn apply_rotary_emb_batch(
    x: &mut Tensor,
    positions: &[usize],
    head_dim: usize,
    theta: f32,
    cache: Option<&RoPECache>,
) {
    let num_heads = x.shape()[0];
    let batch = x.shape()[1];
    let half = head_dim / 2;

    let (cos_table, sin_table) = cache
        .filter(|c| c.head_dim == head_dim && c.theta == theta)
        .map(|c| (&c.cos[..], &c.sin[..]))
        .unwrap_or((&[], &[]));

    let x_slice = x.as_f32_slice_mut();

    for h in 0..num_heads {
        for (b, &position) in positions.iter().enumerate() {
            let base = (h * batch + b) * head_dim;
            for i in 0..half {
                let (cos_val, sin_val) = if !cos_table.is_empty() {
                    let idx = position * half + i;
                    (cos_table[idx], sin_table[idx])
                } else {
                    let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
                    let angle = position as f32 * freq;
                    (angle.cos(), angle.sin())
                };

                // Non-interleaved (half-split) pairing (Llama/Gemma/Qwen/SmolLM).
                let idx_lo = base + i;
                let idx_hi = base + i + half;

                let x_lo = x_slice[idx_lo];
                let x_hi = x_slice[idx_hi];

                x_slice[idx_lo] = x_lo * cos_val - x_hi * sin_val;
                x_slice[idx_hi] = x_lo * sin_val + x_hi * cos_val;
            }
        }
    }
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

                // Non-interleaved (half-split) pairing (Llama/Gemma/Qwen/SmolLM).
                let idx_lo = base + i;
                let idx_hi = base + i + half;

                let x_lo = x_slice[idx_lo];
                let x_hi = x_slice[idx_hi];

                x_slice[idx_lo] = x_lo * cos_val - x_hi * sin_val;
                x_slice[idx_hi] = x_lo * sin_val + x_hi * cos_val;
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
    fn test_attention_softcap_bounds_logits() {
        let num_heads = 1;
        let num_kv_heads = 1;
        let head_dim = 2;

        let q = Tensor::from_slice(&[1.0, 1.0], &[num_heads, 1, head_dim]);
        let k = Tensor::from_slice(&[1.0, 1.0, 2.0, 2.0], &[1, num_kv_heads, 2, head_dim]);
        let v = Tensor::from_slice(&[10.0, 20.0, 30.0, 40.0], &[1, num_kv_heads, 2, head_dim]);

        let no_cap = super::scaled_dot_product_attention_owned(
            &q, &k, &v, num_heads, num_kv_heads, head_dim, 1, 2, 0, 1, 1.0, 0.0,
        );
        let capped = super::scaled_dot_product_attention_owned(
            &q, &k, &v, num_heads, num_kv_heads, head_dim, 1, 2, 0, 1, 1.0, 1.0,
        );

        let no = no_cap.as_f32_slice();
        let cap = capped.as_f32_slice();
        assert!(
            no[1] > 35.0,
            "uncapped attention is dominated by the high-dot key (got {})",
            no[1]
        );
        assert!(
            cap[1] < 32.0,
            "softcap=1.0 flattens the distribution toward uniform (got {})",
            cap[1]
        );
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
            1,
            config.max_seq_len,
            config.num_kv_heads(),
            head_dim,
        );

        let output = attention.forward(&input, Some(&mut cache), 0, 0);
        assert_eq!(output.shape(), &[1, hidden]);

        let output2 = attention.forward(&input, Some(&mut cache), 0, 1);
        assert_eq!(output2.shape(), &[1, hidden]);

        assert_eq!(
            cache.seq_len(0),
            0,
            "Attention does not advance cache - Model does"
        );
    }

    #[test]
    fn test_batched_attention_isolates_slots() {
        let config = ModelConfig::tiny_test();
        let hidden = config.hidden_size;
        let head_dim = config.head_dim();
        let num_kv_heads = config.num_kv_heads();

        let attention = Attention::new(
            Linear::new(Tensor::random(&[hidden, hidden], DType::F32), None),
            Linear::new(
                Tensor::random(&[hidden, num_kv_heads * head_dim], DType::F32),
                None,
            ),
            Linear::new(
                Tensor::random(&[hidden, num_kv_heads * head_dim], DType::F32),
                None,
            ),
            Linear::new(Tensor::random(&[hidden, hidden], DType::F32), None),
            config.clone(),
        );

        let batch = 3;
        let mut cache = KvCache::new(
            config.num_layers,
            batch,
            config.max_seq_len,
            num_kv_heads,
            head_dim,
        );

        // Each slot writes one token at its own position.
        for b in 0..batch {
            let input = Tensor::random(&[1, hidden], DType::F32);
            let positions = [b];
            attention.forward_batch_gpu(&input, Some(&mut cache), 0, &positions, None, None);
            cache.reserve(b, b, 1);
        }

        // Slots must have independent lengths.
        for b in 0..batch {
            assert_eq!(cache.seq_len(b), b + 1);
        }

        let next = Tensor::random(&[batch, hidden], DType::F32);
        let positions: Vec<usize> = (0..batch).map(|b| cache.seq_len(b)).collect();
        let out = attention.forward_batch_gpu(&next, Some(&mut cache), 0, &positions, None, None);
        assert_eq!(out.shape(), &[batch, hidden]);
    }
}
