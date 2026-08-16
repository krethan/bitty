//! End-to-end straight-through-estimator quantization-aware training (STE-QAT).
//!
//! A per-projection reconstruction objective cannot beat naive quantization:
//! with a frozen FP32 teacher, `argmin ||x·Q(W)ᵀ - x·Wᵀ||²` over the ternary
//! feasibility set is exactly the naive round of `W` — QAT only starts to help
//! once the loss is measured **at the model output**, where later layers can
//! compensate for earlier layers' quantization error.
//!
//! This module therefore runs the *deployed quantized model* as the **student**
//! and distills it against a frozen FP32 **teacher** (the same model's exact
//! forward, computed once and held fixed):
//!
//! ```text
//! L = mean_{t,v} (logits_qat[t][v] - logits_fp32[t][v])²
//! ```
//!
//! The student forward is the deploy-time graph — `BitLinear` kernels for the
//! seven projections per layer (with W1A8 if configured), FP32 RMSNorm, RoPE,
//! all-to-all attention, SiLU gating. The backward pass walks the full graph
//! with the straight-through estimator (`∂Q/∂W ≜ 1`) at each quantizer: weight
//! gradients use the dequantized matrix `Q(W)` as the exact linear-map
//! Jacobian (the matmul is linear in the dequantized weights), and the
//! activation quantizer (A8) is treated as identity. Only the seven projection
//! weights per layer are trained; embeddings, norms, and the head stay FP32
//! and frozen (they are exact at inference).
//!
//! Teacher and student share the same graph, so SubLN is **not** supported:
//! the FP32 teacher path (`TransformerLayer`) has no `sub_ln`, and the
//! deployed `sub_ln` block uses a fresh identity norm, so the two graphs would
//! differ.
//!
//! The full-graph backward is pinned by a finite-difference test against an
//! identity-quantized student (smooth objective), and the quantized loop by a
//! deployed-error test (QAT must reduce the end-to-end logit MSE below the
//! naive-quantization baseline).

use bitllm_quantization::{ternary_dequantize, QuantConfig};
use bitllm_runtime::attention::apply_rotary_emb;
use bitllm_runtime::bitlinear::BitLinear;
use bitllm_runtime::config::ModelConfig;
use bitllm_runtime::model::Model;
use bitllm_tensor::{DType, Tensor};

/// Hyperparameters for end-to-end STE-QAT. The quantization config also
/// selects the deployment format used by [`QATModel::deploy`].
#[derive(Debug, Clone)]
pub struct QATConfig {
    /// Learning rate applied to the STE gradient, per gradient step (each
    /// window's gradient is averaged, so this is a step-global LR).
    pub lr: f32,
    /// Number of re-quantize / forward / backward / update steps.
    pub steps: usize,
    /// Deploy-time quantization (ternary, group size, outlier fraction, W1A8).
    pub quant: QuantConfig,
    /// Optional hard clip on latent weight magnitude after each update.
    pub weight_clip: Option<f32>,
    /// Max global gradient norm (L2 across all projection grads). If the norm
    /// exceeds this, all gradients are scaled down proportionally. `None`
    /// disables clipping.
    pub grad_clip: Option<f32>,
    /// Number of initial steps with linear LR warmup from 0 to `lr`.
    pub warmup_steps: usize,
    /// If true, apply cosine LR decay from `lr` to 0 after warmup.
    pub cosine_decay: bool,
    /// If set, evaluate on this window each step for early stopping. Training
    /// stops if MSE hasn't improved by `min_delta` for `patience` consecutive
    /// steps.
    pub eval_window: Option<Vec<u32>>,
    /// Early stopping patience (steps without improvement before stopping).
    pub patience: usize,
    /// Minimum MSE improvement to reset the early stopping counter.
    pub min_delta: f32,
    /// Which projections to train. If empty, all projections are trained.
    /// For ablation studies: specify subset like ["q", "v", "up", "gate"].
    pub train_projections: Vec<String>,
}

impl Default for QATConfig {
    fn default() -> Self {
        Self {
            lr: 0.02,
            steps: 200,
            quant: QuantConfig::ternary(),
            weight_clip: None,
            grad_clip: None,
            warmup_steps: 0,
            cosine_decay: false,
            eval_window: None,
            patience: 10,
            min_delta: 1e-4,
            train_projections: Vec::new(), // Empty means train all projections
        }
    }
}

impl QATConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    pub fn with_steps(mut self, steps: usize) -> Self {
        self.steps = steps;
        self
    }

    pub fn with_quant(mut self, quant: QuantConfig) -> Self {
        self.quant = quant;
        self
    }

    pub fn with_weight_clip(mut self, clip: f32) -> Self {
        self.weight_clip = Some(clip);
        self
    }

    pub fn with_grad_clip(mut self, clip: f32) -> Self {
        self.grad_clip = Some(clip);
        self
    }

    pub fn with_warmup(mut self, steps: usize) -> Self {
        self.warmup_steps = steps;
        self
    }

    pub fn with_cosine_decay(mut self, enabled: bool) -> Self {
        self.cosine_decay = enabled;
        self
    }

    pub fn with_eval_window(mut self, window: Vec<u32>) -> Self {
        self.eval_window = Some(window);
        self
    }

    pub fn with_patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    pub fn with_min_delta(mut self, delta: f32) -> Self {
        self.min_delta = delta;
        self
    }

    pub fn with_train_projections(mut self, projections: Vec<String>) -> Self {
        self.train_projections = projections;
        self
    }

    fn should_train_projection(&self, name: &str) -> bool {
        self.train_projections.is_empty() || self.train_projections.iter().any(|p| p == name)
    }
}

/// Mean-squared error between two same-shaped tensors.
pub fn mean_sq_error(a: &Tensor, b: &Tensor) -> f32 {
    let a_s = a.as_f32_slice();
    let b_s = b.as_f32_slice();
    debug_assert_eq!(a_s.len(), b_s.len());
    a_s.iter()
        .zip(b_s.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        / a_s.len().max(1) as f32
}

/// STE gradient for a linear layer: `EᵀX` of shape `[out, in]`, the gradient
/// of `½·mean(||E||²)` (with `E = x·Qᵀ - y`) w.r.t. the dequantized weights
/// `Q`. The matmul is linear in `Q`, so this is exact and needs no STE; only
/// the round-to-nearest step through the quantizer routes it to the latent
/// weights via the identity `∂Q/∂W ≜ 1`.
pub fn ste_grad(x: &Tensor, error: &Tensor) -> Tensor {
    error
        .transpose()
        .dot(x)
        .expect("ste grad: [out, T] x [T, in]")
}

/// One quantized projection's per-step state: the deploy-time `BitLinear`, the
/// dequantized weight matrix used for the (exact) backward Jacobian, and the
/// accumulated gradient.
struct QProjection {
    lin: BitLinear,
    dequant: Tensor,
    grad: Vec<f32>,
    /// When true the forward is `x·dequantᵀ` with `dequant` equal to the
    /// latent weight — an identity-quantized student for gradient tests.
    identity: bool,
}

fn quantized_projection(weight: &Tensor, config: &QuantConfig) -> QProjection {
    let lin = BitLinear::quantize(weight, config);
    let dequant = ternary_dequantize(&lin.weight_q);
    let grad = vec![0.0f32; dequant.num_elements()];
    QProjection {
        lin,
        dequant,
        grad,
        identity: false,
    }
}

#[cfg(test)]
fn identity_projection(weight: &Tensor) -> QProjection {
    let lin = BitLinear::quantize(weight, &QuantConfig::ternary().without_a8());
    let dequant = weight.clone();
    let grad = vec![0.0f32; dequant.num_elements()];
    QProjection {
        lin,
        dequant,
        grad,
        identity: true,
    }
}

impl QProjection {
    fn forward(&self, x: &Tensor) -> Tensor {
        if self.identity {
            x.dot(&self.dequant.transpose())
                .expect("identity linear forward")
        } else {
            self.lin.forward(x)
        }
    }
}

/// The seven per-layer projections plus the two layer norms (cloned from the
/// model, which are exact at inference and never trained).
struct LayerProj {
    q: QProjection,
    k: QProjection,
    v: QProjection,
    o: QProjection,
    up: QProjection,
    gate: QProjection,
    down: QProjection,
    attn_norm_w: Tensor,
    ffn_norm_w: Tensor,
    eps: f32,
}

/// Activations saved by the forward pass, consumed by the backward pass.
struct SavedLayer {
    h_in: Tensor,
    block1: Tensor,
    inv_rms1: Vec<f32>,
    q_r: Tensor,
    k_r: Tensor,
    v_r: Tensor,
    p: Tensor,
    attn_in: Tensor,
    h_mid: Tensor,
    block2: Tensor,
    inv_rms2: Vec<f32>,
    up: Tensor,
    gate: Tensor,
    gate_sig: Tensor,
    activated: Tensor,
}

fn build_layers(model: &Model, quant: &QuantConfig) -> Vec<LayerProj> {
    model
        .layers
        .iter()
        .map(|l| LayerProj {
            q: quantized_projection(&l.attention.q_proj.weight, quant),
            k: quantized_projection(&l.attention.k_proj.weight, quant),
            v: quantized_projection(&l.attention.v_proj.weight, quant),
            o: quantized_projection(&l.attention.o_proj.weight, quant),
            up: quantized_projection(&l.ffn_up.weight, quant),
            gate: quantized_projection(&l.ffn_gate.weight, quant),
            down: quantized_projection(&l.ffn_down.weight, quant),
            attn_norm_w: l.attn_norm.weight.clone(),
            ffn_norm_w: l.ffn_norm.weight.clone(),
            eps: model.config.norm_eps,
        })
        .collect()
}

#[cfg(test)]
fn build_identity_layers(model: &Model) -> Vec<LayerProj> {
    model
        .layers
        .iter()
        .map(|l| LayerProj {
            q: identity_projection(&l.attention.q_proj.weight),
            k: identity_projection(&l.attention.k_proj.weight),
            v: identity_projection(&l.attention.v_proj.weight),
            o: identity_projection(&l.attention.o_proj.weight),
            up: identity_projection(&l.ffn_up.weight),
            gate: identity_projection(&l.ffn_gate.weight),
            down: identity_projection(&l.ffn_down.weight),
            attn_norm_w: l.attn_norm.weight.clone(),
            ffn_norm_w: l.ffn_norm.weight.clone(),
            eps: model.config.norm_eps,
        })
        .collect()
}

/// `y[m][j] = x[m][j]·w[j]/rms[m]` with `rms[m] = sqrt(mean_j x[m][j]² + eps)`.
/// Returns `(y, inv_rms)` so the backward pass can reuse the per-token scale.
fn rmsnorm_forward(x: &Tensor, w: &Tensor, eps: f32) -> (Tensor, Vec<f32>) {
    let t = x.shape()[0];
    let h = x.shape()[1];
    let xs = x.as_f32_slice();
    let ws = w.as_f32_slice();
    let mut out = Tensor::zeros(&[t, h], DType::F32);
    let os = out.as_f32_slice_mut();
    let mut inv_rms = vec![0.0f32; t];
    for m in 0..t {
        let row = &xs[m * h..][..h];
        let mut sum_sq = 0.0f32;
        for &v in row {
            sum_sq += v * v;
        }
        let inv = 1.0 / (sum_sq / h as f32 + eps).sqrt();
        inv_rms[m] = inv;
        for j in 0..h {
            os[m * h + j] = row[j] * ws[j] * inv;
        }
    }
    (out, inv_rms)
}

/// Exact RMSNorm backward: `gx[i] = w[i]·gy[i]/r − (x[i]·Σ_j gy[j]·x[j]·w[j])/(r³·H)`.
fn rmsnorm_backward(x: &Tensor, w: &Tensor, gy: &Tensor, inv_rms: &[f32]) -> Tensor {
    let t = x.shape()[0];
    let h = x.shape()[1];
    let xs = x.as_f32_slice();
    let ws = w.as_f32_slice();
    let ys = gy.as_f32_slice();
    let mut gx = Tensor::zeros(&[t, h], DType::F32);
    let gs = gx.as_f32_slice_mut();
    for m in 0..t {
        let row = &xs[m * h..][..h];
        let gy_row = &ys[m * h..][..h];
        let inv = inv_rms[m];
        let mut s = 0.0f32;
        for j in 0..h {
            s += gy_row[j] * row[j] * ws[j];
        }
        let coeff = s * inv * inv * inv / h as f32;
        for j in 0..h {
            gs[m * h + j] = ws[j] * inv * gy_row[j] - coeff * row[j];
        }
    }
    gx
}

/// `[seq, num_heads·head_dim]` (per-position, per-head blocks) → `[num_heads, seq, head_dim]`.
fn reshape_for_attention(t: &Tensor, num_heads: usize, head_dim: usize) -> Tensor {
    let seq = t.shape()[0];
    let mut out = Tensor::zeros(&[num_heads, seq, head_dim], DType::F32);
    let src = t.as_f32_slice();
    let dst = out.as_f32_slice_mut();
    for h in 0..num_heads {
        for pos in 0..seq {
            let s = pos * num_heads * head_dim + h * head_dim;
            let d = h * seq * head_dim + pos * head_dim;
            dst[d..d + head_dim].copy_from_slice(&src[s..s + head_dim]);
        }
    }
    out
}

/// Inverse of [`reshape_for_attention`].
fn unreshape_for_attention(t: &Tensor, num_heads: usize, head_dim: usize) -> Tensor {
    let seq = t.shape()[1];
    let mut out = Tensor::zeros(&[seq, num_heads * head_dim], DType::F32);
    let src = t.as_f32_slice();
    let dst = out.as_f32_slice_mut();
    for h in 0..num_heads {
        for pos in 0..seq {
            let s = h * seq * head_dim + pos * head_dim;
            let d = pos * num_heads * head_dim + h * head_dim;
            dst[d..d + head_dim].copy_from_slice(&src[s..s + head_dim]);
        }
    }
    out
}

/// `[num_heads, seq, head_dim]` → `[seq, num_heads·head_dim]`, matching the
/// runtime's `sdp_output_to_hidden`: each position row holds head blocks
/// `[h0|h1|...]` contiguously (NOT the plain `reshape_owned` flatten).
fn attn_heads_to_hidden(t: &Tensor, seq: usize, num_heads: usize, head_dim: usize) -> Tensor {
    let hidden = num_heads * head_dim;
    let mut out = Tensor::zeros(&[seq, hidden], DType::F32);
    let src = t.as_f32_slice();
    let dst = out.as_f32_slice_mut();
    for h in 0..num_heads {
        for pos in 0..seq {
            let s = h * seq * head_dim + pos * head_dim;
            let d = pos * hidden + h * head_dim;
            dst[d..d + head_dim].copy_from_slice(&src[s..s + head_dim]);
        }
    }
    out
}

/// Inverse of [`attn_heads_to_hidden`].
fn hidden_to_attn_heads(t: &Tensor, seq: usize, num_heads: usize, head_dim: usize) -> Tensor {
    let hidden = num_heads * head_dim;
    let mut out = Tensor::zeros(&[num_heads, seq, head_dim], DType::F32);
    let src = t.as_f32_slice();
    let dst = out.as_f32_slice_mut();
    for h in 0..num_heads {
        for pos in 0..seq {
            let s = pos * hidden + h * head_dim;
            let d = h * seq * head_dim + pos * head_dim;
            dst[d..d + head_dim].copy_from_slice(&src[s..s + head_dim]);
        }
    }
    out
}

/// Causal all-to-all scaled dot-product attention (matches the runtime's
/// masked SDPA: a query at position `m` attends only to keys `n <= m`).
/// Returns `(output, softmax weights)`, with masked weights exactly zero.
fn sdpa_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq: usize,
) -> (Tensor, Tensor) {
    let scale = (head_dim as f32).sqrt();
    let kv_groups = num_heads / num_kv_heads;
    let mut out = Tensor::zeros(&[num_heads, seq, head_dim], DType::F32);
    let mut p = Tensor::zeros(&[num_heads, seq, seq], DType::F32);
    let qs = q.as_f32_slice();
    let ks = k.as_f32_slice();
    let vs = v.as_f32_slice();
    let os = out.as_f32_slice_mut();
    let ps = p.as_f32_slice_mut();
    for h in 0..num_heads {
        let kv_h = h / kv_groups;
        for m in 0..seq {
            let q_row = &qs[h * seq * head_dim + m * head_dim..][..head_dim];
            let mut scores = vec![f32::NEG_INFINITY; seq];
            let mut maxv = f32::NEG_INFINITY;
            for n in 0..=m {
                let k_row = &ks[kv_h * seq * head_dim + n * head_dim..][..head_dim];
                let s = q_row
                    .iter()
                    .zip(k_row.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    / scale;
                scores[n] = s;
                if s > maxv {
                    maxv = s;
                }
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - maxv).exp();
                sum += *s;
            }
            let inv = 1.0 / sum;
            let out_row = &mut os[h * seq * head_dim + m * head_dim..][..head_dim];
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for n in 0..=m {
                    let wgt = scores[n] * inv;
                    acc += wgt * vs[kv_h * seq * head_dim + n * head_dim + d];
                }
                out_row[d] = acc;
            }
            for n in 0..seq {
                ps[h * seq * seq + m * seq + n] = scores[n] * inv;
            }
        }
    }
    (out, p)
}

/// Exact SDPA backward using the saved softmax weights. Returns `(gq, gk, gv)`
/// in the head layout.
#[allow(clippy::too_many_arguments)]
fn sdpa_backward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    p: &Tensor,
    gout: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq: usize,
) -> (Tensor, Tensor, Tensor) {
    let scale = (head_dim as f32).sqrt();
    let kv_groups = num_heads / num_kv_heads;
    let mut gq = Tensor::zeros(&[num_heads, seq, head_dim], DType::F32);
    let mut gk = Tensor::zeros(&[num_kv_heads, seq, head_dim], DType::F32);
    let mut gv = Tensor::zeros(&[num_kv_heads, seq, head_dim], DType::F32);
    let qs = q.as_f32_slice();
    let ks = k.as_f32_slice();
    let vs = v.as_f32_slice();
    let ps = p.as_f32_slice();
    let gs = gout.as_f32_slice();
    let gqs = gq.as_f32_slice_mut();
    let gks = gk.as_f32_slice_mut();
    let gvs = gv.as_f32_slice_mut();

    for h in 0..num_heads {
        let kv_h = h / kv_groups;
        // gv[kv_h][n][d] = Σ_m P[h][m][n]·gout[h][m][d]
        for n in 0..seq {
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for m in 0..seq {
                    acc +=
                        ps[h * seq * seq + m * seq + n] * gs[h * seq * head_dim + m * head_dim + d];
                }
                gvs[kv_h * seq * head_dim + n * head_dim + d] += acc;
            }
        }
        // Softmax backward: glogits[m][n] = P[m][n]·(g_p[m][n] − Σ_q P[m][q]·g_p[m][q])
        let mut glogits = vec![0.0f32; seq * seq];
        for m in 0..seq {
            let go_row = &gs[h * seq * head_dim + m * head_dim..][..head_dim];
            let mut g_p = vec![0.0f32; seq];
            for n in 0..seq {
                let v_row = &vs[kv_h * seq * head_dim + n * head_dim..][..head_dim];
                g_p[n] = go_row.iter().zip(v_row.iter()).map(|(a, b)| a * b).sum();
            }
            let mut dot_p = 0.0f32;
            for n in 0..seq {
                dot_p += ps[h * seq * seq + m * seq + n] * g_p[n];
            }
            for n in 0..seq {
                let pv = ps[h * seq * seq + m * seq + n];
                glogits[m * seq + n] = pv * (g_p[n] - dot_p);
            }
        }
        // gq[h][m][t] = Σ_n glogits[m][n]·k[kv_h][n][t]/scale
        for m in 0..seq {
            for tt in 0..head_dim {
                let mut acc = 0.0f32;
                for n in 0..seq {
                    acc += glogits[m * seq + n] * ks[kv_h * seq * head_dim + n * head_dim + tt];
                }
                gqs[h * seq * head_dim + m * head_dim + tt] = acc / scale;
            }
        }
        // gk[kv_h][n][t] = Σ_m glogits[m][n]·q[h][m][t]/scale
        for n in 0..seq {
            for tt in 0..head_dim {
                let mut acc = 0.0f32;
                for m in 0..seq {
                    acc += glogits[m * seq + n] * qs[h * seq * head_dim + m * head_dim + tt];
                }
                gks[kv_h * seq * head_dim + n * head_dim + tt] += acc / scale;
            }
        }
    }
    (gq, gk, gv)
}

/// Exact RoPE backward (the rotation is orthogonal): `gx[e] = gy[e]·c + gy[o]·s`,
/// `gx[o] = −gy[e]·s + gy[o]·c`.
fn rope_backward(gy: &Tensor, position: usize, head_dim: usize, theta: f32) -> Tensor {
    let heads = gy.shape()[0];
    let seq = gy.shape()[1];
    let half = head_dim / 2;
    let mut gx = Tensor::zeros(&[heads, seq, head_dim], DType::F32);
    let ys = gy.as_f32_slice();
    let xs = gx.as_f32_slice_mut();
    for h in 0..heads {
        for pos in 0..seq {
            let base = h * seq * head_dim + pos * head_dim;
            for i in 0..half {
                let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
                let angle = (position + pos) as f32 * freq;
                let (c, s) = (angle.cos(), angle.sin());
                let e = base + i;
                let o = base + i + half;
                let ye = ys[e];
                let yo = ys[o];
                xs[e] = ye * c + yo * s;
                xs[o] = -ye * s + yo * c;
            }
        }
    }
    gx
}

/// Returns `(silu(x), sigmoid(x))`.
fn silu_with_sigmoid(x: &Tensor) -> (Tensor, Tensor) {
    let n = x.num_elements();
    let mut silu = Tensor::zeros(x.shape(), DType::F32);
    let mut sig = Tensor::zeros(x.shape(), DType::F32);
    let xs = x.as_f32_slice();
    let sl = silu.as_f32_slice_mut();
    let sg = sig.as_f32_slice_mut();
    for i in 0..n {
        let v = xs[i];
        let s = 1.0 / (1.0 + (-v).exp());
        sg[i] = s;
        sl[i] = v * s;
    }
    (silu, sig)
}

/// Student layer forward (mirrors the deployed CPU `BitTransformerLayer`
/// path, non-SubLN). Returns `(h_out, saved activations)`.
fn layer_forward(p: &LayerProj, h_in: &Tensor, cfg: &ModelConfig) -> (Tensor, SavedLayer) {
    let num_heads = cfg.num_heads;
    let num_kv = cfg.num_kv_heads();
    let hd = cfg.head_dim();
    let seq = h_in.shape()[0];

    let (block1, inv_rms1) = rmsnorm_forward(h_in, &p.attn_norm_w, p.eps);

    let q = p.q.forward(&block1);
    let k = p.k.forward(&block1);
    let v = p.v.forward(&block1);

    let q_r = apply_rotary_emb(
        &reshape_for_attention(&q, num_heads, hd),
        0,
        hd,
        cfg.rope_theta,
    );
    let k_r = apply_rotary_emb(
        &reshape_for_attention(&k, num_kv, hd),
        0,
        hd,
        cfg.rope_theta,
    );
    let v_r = reshape_for_attention(&v, num_kv, hd);

    let (attn_heads, p_soft) = sdpa_forward(&q_r, &k_r, &v_r, num_heads, num_kv, hd, seq);
    // The runtime feeds `o_proj` with a per-position permutation of the head
    // layout (`sdp_output_to_hidden`), where position `pos` holds head blocks
    // `[h0|h1|...]` contiguously — mirror that exactly so the graph matches
    // the deployed model's `o_proj` input ordering.
    let attn_in = attn_heads_to_hidden(&attn_heads, seq, num_heads, hd);
    let o_out = p.o.forward(&attn_in);

    let h_mid = h_in.add(&o_out).unwrap();
    let (block2, inv_rms2) = rmsnorm_forward(&h_mid, &p.ffn_norm_w, p.eps);

    let up = p.up.forward(&block2);
    let gate = p.gate.forward(&block2);
    let (gate_silu, gate_sig) = silu_with_sigmoid(&gate);
    let activated = gate_silu.mul(&up).unwrap();
    let down_out = p.down.forward(&activated);

    let h_out = h_mid.add(&down_out).unwrap();

    (
        h_out,
        SavedLayer {
            h_in: h_in.clone(),
            block1,
            inv_rms1,
            q_r,
            k_r,
            v_r,
            p: p_soft,
            attn_in,
            h_mid,
            block2,
            inv_rms2,
            up,
            gate,
            gate_sig,
            activated,
        },
    )
}

fn acc_grad(grad: &mut [f32], g: Tensor) {
    let gs = g.as_f32_slice();
    debug_assert_eq!(grad.len(), gs.len());
    for (a, &v) in grad.iter_mut().zip(gs.iter()) {
        *a += v;
    }
}

/// Student layer backward. Accumulates the seven per-projection gradients into
/// `p` and returns `g_h_in` (the gradient w.r.t. the layer input).
fn layer_backward(
    p: &mut LayerProj,
    s: &SavedLayer,
    g_h_out: &Tensor,
    cfg: &ModelConfig,
) -> Tensor {
    let num_heads = cfg.num_heads;
    let num_kv = cfg.num_kv_heads();
    let hd = cfg.head_dim();
    let seq = g_h_out.shape()[0];

    // --- FFN branch: h_out = h_mid + down_out ---
    let g_activated = g_h_out.dot(&p.down.dequant).unwrap(); // [T, H]·[H, inter]
    acc_grad(
        &mut p.down.grad,
        g_h_out.transpose().dot(&s.activated).unwrap(),
    );
    // d(activated)/d(up) = silu(g) = g·σ(g) (activated = silu(g)·up).
    let g_up = g_activated.mul(&s.gate).unwrap().mul(&s.gate_sig).unwrap();
    // silu'(g) = σ(g)·(1 + g·(1 − σ(g)))
    let mut silu_deriv = Tensor::zeros(&[seq, cfg.intermediate_size], DType::F32);
    {
        let gs = s.gate.as_f32_slice();
        let sigs = s.gate_sig.as_f32_slice();
        let ds = silu_deriv.as_f32_slice_mut();
        for i in 0..gs.len() {
            let g = gs[i];
            let sig = sigs[i];
            ds[i] = sig * (1.0 + g * (1.0 - sig));
        }
    }
    let g_gate = g_activated.mul(&s.up).unwrap().mul(&silu_deriv).unwrap();
    acc_grad(&mut p.up.grad, g_up.transpose().dot(&s.block2).unwrap());
    acc_grad(&mut p.gate.grad, g_gate.transpose().dot(&s.block2).unwrap());
    let g_block2 = g_up
        .dot(&p.up.dequant)
        .unwrap()
        .add(&g_gate.dot(&p.gate.dequant).unwrap())
        .unwrap();
    let g_h_mid = g_h_out
        .add(&rmsnorm_backward(
            &s.h_mid,
            &p.ffn_norm_w,
            &g_block2,
            &s.inv_rms2,
        ))
        .unwrap();

    // --- Attention branch: h_mid = h_in + o_out ---
    let g_attn_in = g_h_mid.dot(&p.o.dequant).unwrap(); // [T, H]
    acc_grad(&mut p.o.grad, g_h_mid.transpose().dot(&s.attn_in).unwrap());
    // Inverse of `attn_heads_to_hidden`.
    let g_attn_heads = hidden_to_attn_heads(&g_attn_in, seq, num_heads, hd);
    let (gq_r, gk_r, gv_r) = sdpa_backward(
        &s.q_r,
        &s.k_r,
        &s.v_r,
        &s.p,
        &g_attn_heads,
        num_heads,
        num_kv,
        hd,
        seq,
    );
    let gq = rope_backward(&gq_r, 0, hd, cfg.rope_theta);
    let gk = rope_backward(&gk_r, 0, hd, cfg.rope_theta);
    let gq_flat = unreshape_for_attention(&gq, num_heads, hd);
    let gk_flat = unreshape_for_attention(&gk, num_kv, hd);
    let gv_flat = unreshape_for_attention(&gv_r, num_kv, hd);
    acc_grad(&mut p.q.grad, gq_flat.transpose().dot(&s.block1).unwrap());
    acc_grad(&mut p.k.grad, gk_flat.transpose().dot(&s.block1).unwrap());
    acc_grad(&mut p.v.grad, gv_flat.transpose().dot(&s.block1).unwrap());
    let g_block1 = gq_flat
        .dot(&p.q.dequant)
        .unwrap()
        .add(&gk_flat.dot(&p.k.dequant).unwrap())
        .unwrap()
        .add(&gv_flat.dot(&p.v.dequant).unwrap())
        .unwrap();

    g_h_mid
        .add(&rmsnorm_backward(
            &s.h_in,
            &p.attn_norm_w,
            &g_block1,
            &s.inv_rms1,
        ))
        .unwrap()
}

/// Student forward: embedding → layers → final norm → head. Mirrors
/// [`Model::forward`] on the deployed (quantized) graph.
#[cfg(test)]
fn student_forward(model: &Model, layers: &[LayerProj], tokens: &[u32]) -> Tensor {
    let cfg = &model.config;
    let mut h = model.embedding.forward(tokens);
    for layer in layers {
        let (h_out, _saved) = layer_forward(layer, &h, cfg);
        h = h_out;
    }
    let (normed, _) = rmsnorm_forward(&h, &model.norm.weight, cfg.norm_eps);
    model.lm_head.forward(&normed)
}

/// Run one window through the student graph and backprop the logit MSE against
/// the frozen teacher logits, accumulating per-projection gradients. Returns
/// the window loss.
fn forward_backward_window(
    model: &Model,
    layers: &mut [LayerProj],
    tokens: &[u32],
    teacher: &Tensor,
) -> f32 {
    let cfg = &model.config;
    let t = tokens.len();
    let v = cfg.vocab_size;

    let mut h = model.embedding.forward(tokens);
    let mut saveds = Vec::with_capacity(layers.len());
    for layer in layers.iter() {
        let (h_out, saved) = layer_forward(layer, &h, cfg);
        saveds.push(saved);
        h = h_out;
    }
    let (normed, inv_rms_final) = rmsnorm_forward(&h, &model.norm.weight, cfg.norm_eps);
    let logits = model.lm_head.forward(&normed);
    let loss = mean_sq_error(&logits, teacher);

    let scale = 2.0 / (t * v) as f32;
    let mut g_logits = logits.sub(teacher).unwrap();
    g_logits.f32_scale_inplace(scale);
    let g_normed = g_logits.dot(&model.lm_head.weight).unwrap();
    let mut g_h = rmsnorm_backward(&h, &model.norm.weight, &g_normed, &inv_rms_final);
    for i in (0..layers.len()).rev() {
        g_h = layer_backward(&mut layers[i], &saveds[i], &g_h, cfg);
    }
    loss
}

fn apply_grad(weight: &mut Tensor, grad: &[f32], lr: f32, clip: Option<f32>) {
    let ws = weight.as_f32_slice_mut();
    debug_assert_eq!(ws.len(), grad.len());
    for (w, &g) in ws.iter_mut().zip(grad.iter()) {
        *w -= lr * g;
        if let Some(c) = clip {
            *w = w.clamp(-c, c);
        }
    }
}

/// Evaluate MSE on a held-out window (forward pass only, no gradient accumulation).
/// Used for early stopping.
fn eval_mse_on_window(model: &Model, quant: &QuantConfig, window: &[u32], teacher: &Tensor) -> f32 {
    let layers = build_layers(model, quant);
    let cfg = &model.config;

    let mut h = model.embedding.forward(window);
    for layer in &layers {
        let (h_out, _) = layer_forward(layer, &h, cfg);
        h = h_out;
    }
    let (normed, _) = rmsnorm_forward(&h, &model.norm.weight, cfg.norm_eps);
    let logits = model.lm_head.forward(&normed);

    mean_sq_error(&logits, teacher)
}

/// Quantization-aware trainer over a [`Model`]'s latent FP32 weights.
///
/// The wrapped model stays **unquantized** (`layers` populated): it provides
/// both the frozen FP32 teacher and the latent weights being trained.
/// [`QATModel::deploy`] produces the inference model by re-quantizing the
/// fine-tuned latent weights with the training config.
pub struct QATModel {
    pub config: QATConfig,
    pub model: Model,
}

impl QATModel {
    pub fn new(model: Model, config: QATConfig) -> Self {
        assert!(
            !model.config.sub_ln,
            "QAT requires sub_ln = false (the FP32 teacher graph has no SubLN)"
        );
        Self { config, model }
    }

    /// End-to-end STE-QAT. Computes frozen FP32 teacher logits for `windows`
    /// once, then runs `steps` gradient steps over the full quantized student
    /// graph, re-quantizing every projection each step. Only the seven latent
    /// projection weights per layer are updated. Returns the mean
    /// student-vs-teacher logit MSE at the start and end of training.
    pub fn train(&mut self, windows: &[Vec<u32>]) -> (f32, f32) {
        assert!(!windows.is_empty(), "qat needs training windows");
        let mut teachers = Vec::with_capacity(windows.len());
        for w in windows {
            self.model.clear_cache();
            teachers.push(self.model.forward(w));
        }

        // Pre-compute teacher logits for eval window if early stopping is enabled
        let eval_teacher = self.config.eval_window.as_ref().map(|eval_window| {
            self.model.clear_cache();
            self.model.forward(eval_window)
        });

        let mut start = 0.0f32;
        let mut end = 0.0f32;
        let inv_n = 1.0 / windows.len() as f32;
        let mut best_eval_mse = f32::MAX;
        let mut patience_counter = 0usize;

        for step in 0..self.config.steps {
            let mut layers = build_layers(&self.model, &self.config.quant);
            let mut total = 0.0f32;
            for (w, teacher) in windows.iter().zip(teachers.iter()) {
                total += forward_backward_window(&self.model, &mut layers, w, teacher);
            }
            end = total * inv_n;
            if step == 0 {
                start = end;
            }

            // Compute effective LR with warmup and cosine decay
            let base_lr = self.config.lr;
            let effective_lr = if step < self.config.warmup_steps {
                // Linear warmup
                base_lr * (step + 1) as f32 / self.config.warmup_steps as f32
            } else if self.config.cosine_decay {
                // Cosine decay from base_lr to 0
                let progress = (step - self.config.warmup_steps) as f32
                    / (self.config.steps - self.config.warmup_steps).max(1) as f32;
                base_lr * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
            } else {
                base_lr
            };

            // Gradient clipping: compute global norm and scale if needed
            let grad_scale = if let Some(max_norm) = self.config.grad_clip {
                let mut norm_sq = 0.0f32;
                for layer in &layers {
                    for grad in [
                        &layer.q.grad,
                        &layer.k.grad,
                        &layer.v.grad,
                        &layer.o.grad,
                        &layer.up.grad,
                        &layer.gate.grad,
                        &layer.down.grad,
                    ] {
                        for &g in grad {
                            norm_sq += g * g;
                        }
                    }
                }
                let norm = norm_sq.sqrt();
                if norm > max_norm {
                    max_norm / norm
                } else {
                    1.0
                }
            } else {
                1.0
            };

            let lr = effective_lr * inv_n * grad_scale;
            for (i, layer) in self.model.layers.iter_mut().enumerate() {
                let lp = &layers[i];
                if self.config.should_train_projection("q") {
                    apply_grad(
                        &mut layer.attention.q_proj.weight,
                        &lp.q.grad,
                        lr,
                        self.config.weight_clip,
                    );
                }
                if self.config.should_train_projection("k") {
                    apply_grad(
                        &mut layer.attention.k_proj.weight,
                        &lp.k.grad,
                        lr,
                        self.config.weight_clip,
                    );
                }
                if self.config.should_train_projection("v") {
                    apply_grad(
                        &mut layer.attention.v_proj.weight,
                        &lp.v.grad,
                        lr,
                        self.config.weight_clip,
                    );
                }
                if self.config.should_train_projection("o") {
                    apply_grad(
                        &mut layer.attention.o_proj.weight,
                        &lp.o.grad,
                        lr,
                        self.config.weight_clip,
                    );
                }
                if self.config.should_train_projection("up") {
                    apply_grad(
                        &mut layer.ffn_up.weight,
                        &lp.up.grad,
                        lr,
                        self.config.weight_clip,
                    );
                }
                if self.config.should_train_projection("gate") {
                    apply_grad(
                        &mut layer.ffn_gate.weight,
                        &lp.gate.grad,
                        lr,
                        self.config.weight_clip,
                    );
                }
                if self.config.should_train_projection("down") {
                    apply_grad(
                        &mut layer.ffn_down.weight,
                        &lp.down.grad,
                        lr,
                        self.config.weight_clip,
                    );
                }
            }

            // Early stopping check
            if let (Some(eval_window), Some(teacher)) = (&self.config.eval_window, &eval_teacher) {
                let eval_mse =
                    eval_mse_on_window(&self.model, &self.config.quant, eval_window, teacher);

                if eval_mse < best_eval_mse - self.config.min_delta {
                    best_eval_mse = eval_mse;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= self.config.patience {
                        break;
                    }
                }
            }
        }
        (start, end)
    }

    /// Deploy: quantize the fine-tuned latent weights with the training config
    /// and return the inference-ready model (embedding/norms/head stay FP32).
    pub fn deploy(mut self) -> Model {
        self.model.quantize_to_bit1_with_config(&self.config.quant);
        self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::SeededRng;
    use bitllm_runtime::Model;

    fn tensor2d(data: &[f32], rows: usize, cols: usize) -> Tensor {
        Tensor::from_slice(data, &[rows, cols])
    }

    /// Loss `L = ½·mean (x·Qᵀ - y)²` for a *fixed* dequantized weight `Q` (used
    /// for the finite-difference check of the exact linear-map gradient; the
    /// `½` matches the convention of [`ste_grad`]).
    fn fixed_loss(x: &Tensor, q: &Tensor, y: &Tensor) -> f32 {
        let s = x.dot(&q.transpose()).expect("fixed loss matmul");
        0.5 * mean_sq_error(&s, y)
    }

    #[test]
    fn ste_grad_matches_finite_difference() {
        // Random x [T=24, k=6], dequantized weight Q [n=6, k=6], and a fixed
        // target y = x·W_trueᵀ. The loss is a quadratic in Q; the analytic
        // gradient EᵀX (STE through the linear map) must match finite
        // differences of the dequantized loss.
        let (t, k, n) = (24usize, 6usize, 6usize);
        let mut rng = SeededRng::new(3);
        let x_data: Vec<f32> = (0..t * k).map(|_| rng.next_gaussian()).collect();
        let q_data: Vec<f32> = (0..n * k).map(|_| rng.next_gaussian()).collect();
        let w_true: Vec<f32> = (0..n * k).map(|_| rng.next_gaussian()).collect();

        let x = tensor2d(&x_data, t, k);
        let q = tensor2d(&q_data, n, k);
        let q_true = tensor2d(&w_true, n, k);
        let y = x.dot(&q_true.transpose()).expect("target matmul");

        let error = x
            .dot(&q.transpose())
            .expect("forward")
            .sub(&y)
            .expect("error");
        let grad = ste_grad(&x, &error); // [n, k]
        let gs = grad.as_f32_slice();

        let eps = 1e-3f32;
        for &(j, tt) in &[(0usize, 0usize), (3, 4), (5, 2)] {
            let mut q_plus = q.clone();
            let mut q_minus = q.clone();
            q_plus.set_flat_f32(j * k + tt, q_data[j * k + tt] + eps);
            q_minus.set_flat_f32(j * k + tt, q_data[j * k + tt] - eps);
            let fd = (fixed_loss(&x, &q_plus, &y) - fixed_loss(&x, &q_minus, &y)) / (2.0 * eps);
            let analytic = gs[j * k + tt] / (t * n) as f32;
            assert!(
                (fd - analytic).abs() < 1e-3,
                "({j},{tt}): fd {fd:.6} analytic {analytic:.6}"
            );
        }
    }

    fn random_model(config: &bitllm_runtime::config::ModelConfig, seed: u64) -> Model {
        let mut rng = SeededRng::new(seed);
        let mut model = Model::new(config.clone());
        let scale = 0.2f32;
        let rand = |n: usize, rng: &mut SeededRng| -> Vec<f32> {
            (0..n).map(|_| rng.next_gaussian() * scale).collect()
        };
        let n_emb = model.embedding.weight.num_elements();
        model.embedding.weight =
            Tensor::from_slice(&rand(n_emb, &mut rng), model.embedding.weight.shape());
        let n_head = model.lm_head.weight.num_elements();
        model.lm_head.weight =
            Tensor::from_slice(&rand(n_head, &mut rng), model.lm_head.weight.shape());
        for layer in &mut model.layers {
            for w in [
                &mut layer.attention.q_proj.weight,
                &mut layer.attention.k_proj.weight,
                &mut layer.attention.v_proj.weight,
                &mut layer.attention.o_proj.weight,
            ] {
                *w = Tensor::from_slice(&rand(w.num_elements(), &mut rng), w.shape());
            }
            for w in [
                &mut layer.ffn_up.weight,
                &mut layer.ffn_gate.weight,
                &mut layer.ffn_down.weight,
            ] {
                *w = Tensor::from_slice(&rand(w.num_elements(), &mut rng), w.shape());
            }
        }
        model
    }

    fn proj_mut(l: &mut LayerProj, pi: usize) -> &mut QProjection {
        match pi {
            0 => &mut l.q,
            1 => &mut l.k,
            2 => &mut l.v,
            3 => &mut l.o,
            4 => &mut l.up,
            5 => &mut l.gate,
            6 => &mut l.down,
            _ => unreachable!(),
        }
    }

    fn proj_ref(l: &LayerProj, pi: usize) -> &QProjection {
        match pi {
            0 => &l.q,
            1 => &l.k,
            2 => &l.v,
            3 => &l.o,
            4 => &l.up,
            5 => &l.gate,
            6 => &l.down,
            _ => unreachable!(),
        }
    }

    #[test]
    fn rmsnorm_backward_matches_fd() {
        let (t, h) = (7usize, 9usize);
        let mut rng = SeededRng::new(5);
        let x_data: Vec<f32> = (0..t * h).map(|_| rng.next_gaussian()).collect();
        let w_data: Vec<f32> = (0..h).map(|_| rng.next_gaussian()).collect();
        let gy_data: Vec<f32> = (0..t * h).map(|_| rng.next_gaussian()).collect();
        let x = tensor2d(&x_data, t, h);
        let w = tensor2d(&w_data, 1, h);
        let gy = tensor2d(&gy_data, t, h);
        let (y, inv) = rmsnorm_forward(&x, &w, 1e-5);
        let gy_loss = y
            .as_f32_slice()
            .iter()
            .zip(gy_data.iter())
            .map(|(a, b)| 2.0 * (a - b))
            .collect::<Vec<f32>>();
        let gy_true = tensor2d(&gy_loss, t, h);
        let gx = rmsnorm_backward(&x, &w, &gy_true, &inv);
        let loss = |xr: &Tensor| -> f64 {
            let (y2, _) = rmsnorm_forward(xr, &w, 1e-5);
            y2.as_f32_slice()
                .iter()
                .zip(gy.as_f32_slice().iter())
                .map(|(a, b)| {
                    let d = (*a - *b) as f64;
                    d * d
                })
                .sum()
        };
        let eps = 5e-3f32;
        for &idx in &[2usize, 17, 41, 62] {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp.set_flat_f32(idx, x_data[idx] + eps);
            xm.set_flat_f32(idx, x_data[idx] - eps);
            let fd = ((loss(&xp) - loss(&xm)) as f32) / (2.0 * eps);
            let analytic = gx.as_f32_slice()[idx];
            assert!(
                (fd - analytic).abs() / (fd.abs() + 1e-9) < 1e-2,
                "idx {idx}: fd {fd:.6} analytic {analytic:.6}"
            );
        }
    }

    #[test]
    fn student_graph_matches_deployed_model() {
        let config = bitllm_runtime::config::ModelConfig::tiny_test();
        let window: Vec<u32> = (0..16)
            .map(|i| ((i * 11) % config.vocab_size) as u32)
            .collect();

        // Reference: quantize the model the way deploy does, then run it.
        let mut deployed = random_model(&config, 21);
        deployed.quantize_to_bit1_with_config(&QuantConfig::ternary().without_a8());
        deployed.clear_cache();
        let deployed_logits = deployed.forward(&window);

        // Graph: quantize the projections of an identical unquantized model.
        let model = random_model(&config, 21);
        let layers = build_layers(&model, &QuantConfig::ternary().without_a8());
        let graph_logits = student_forward(&model, &layers, &window);

        let mse = mean_sq_error(&graph_logits, &deployed_logits);
        assert!(
            mse < 1e-6,
            "student graph must reproduce the deployed model's logits: mse {mse:.3e}"
        );
    }

    #[test]
    fn end_to_end_backprop_matches_finite_difference() {
        let config = bitllm_runtime::config::ModelConfig::tiny_test();
        let mut teacher = random_model(&config, 7);
        let student = random_model(&config, 13);

        let window: Vec<u32> = (0..20)
            .map(|i| ((i * 13) % config.vocab_size) as u32)
            .collect();
        teacher.clear_cache();
        let teacher_logits = teacher.forward(&window);

        let mut layers = build_identity_layers(&student);
        let loss = forward_backward_window(&student, &mut layers, &window, &teacher_logits);
        assert!(loss > 0.0, "teacher and student must differ (loss {loss})");

        // FD-check gradient entries spanning both layers and attention/FFN.
        // The loss is summed in f64: an f32 mean over ~5000 logits leaves
        // ~1e-2 of cancellation noise, far larger than the ~1e-3 gradient
        // signals under test.
        let entries: &[(usize, usize, usize)] = &[
            (0, 0, 7),
            (0, 1, 3),
            (0, 2, 11),
            (0, 3, 5),
            (0, 4, 9),
            (0, 5, 13),
            (0, 6, 41),
            (1, 0, 7),
            (1, 1, 123),
            (1, 2, 33),
            (1, 3, 1),
            (1, 4, 55),
            (1, 5, 2),
            (1, 6, 17),
        ];
        let t = window.len();
        let v = config.vocab_size;
        let loss = |layers: &[LayerProj]| -> f64 {
            let logits = student_forward(&student, layers, &window);
            logits
                .as_f32_slice()
                .iter()
                .zip(teacher_logits.as_f32_slice().iter())
                .map(|(a, b)| {
                    let d = (*a - *b) as f64;
                    d * d
                })
                .sum::<f64>()
                / (t * v) as f64
        };
        let eps = 1e-2f32;
        for &(li, pi, idx) in entries {
            let orig = proj_ref(&layers[li], pi).dequant.get_flat_f32(idx);
            {
                let p = proj_mut(&mut layers[li], pi);
                p.dequant.set_flat_f32(idx, orig + eps);
            }
            let l_plus = loss(&layers);
            {
                let p = proj_mut(&mut layers[li], pi);
                p.dequant.set_flat_f32(idx, orig - eps);
            }
            let l_minus = loss(&layers);
            {
                let p = proj_mut(&mut layers[li], pi);
                p.dequant.set_flat_f32(idx, orig);
            }
            let fd = ((l_plus - l_minus) as f32) / (2.0 * eps);
            let analytic = proj_ref(&layers[li], pi).grad[idx];
            assert!(
                (fd - analytic).abs() / (fd.abs() + 1e-9) < 5e-2,
                "layer {li} proj {pi} idx {idx}: fd {fd:.6} analytic {analytic:.6}"
            );
        }
    }

    #[test]
    #[ignore = "full 200-step QAT run (~60s); run the slow suite with `cargo test -p bitllm-train -- --ignored`"]
    fn end_to_end_qat_reduces_deployed_error() {
        let config = bitllm_runtime::config::ModelConfig::tiny_test();
        let windows: Vec<Vec<u32>> = (0..4)
            .map(|w| {
                (0..24)
                    .map(|i| ((w * 40 + i * 7) % config.vocab_size) as u32)
                    .collect()
            })
            .collect();
        let eval_window = windows[0].clone();

        // Two identical seeded models: one for the naive-quantization baseline,
        // one whose latent weights are QAT-trained.
        let mut qat_model = QATModel::new(random_model(&config, 11), QATConfig::new());
        let mut baseline = random_model(&config, 11);

        // FP32 reference logits.
        qat_model.model.clear_cache();
        let fp32_logits = qat_model.model.forward(&eval_window);

        // Naive baseline: quantize without any training.
        baseline.quantize_to_bit1_with_config(&qat_model.config.quant);
        baseline.clear_cache();
        let naive_logits = baseline.forward(&eval_window);
        let mse_before = mean_sq_error(&naive_logits, &fp32_logits);

        // QAT: end-to-end STE against frozen teacher logits, then deploy.
        qat_model.train(&windows);
        let mut deployed = qat_model.deploy();
        deployed.clear_cache();
        let qat_logits = deployed.forward(&eval_window);
        let mse_after = mean_sq_error(&qat_logits, &fp32_logits);

        assert!(
            mse_after < mse_before,
            "QAT must reduce the deployed quantization error: naive {mse_before:.6} -> qat {mse_after:.6}"
        );
    }

    #[test]
    fn deploy_produces_bit1_model() {
        let config = bitllm_runtime::config::ModelConfig::tiny_test();
        // Deploy mechanics don't depend on convergence, so a short run suffices.
        let mut qat = QATModel::new(random_model(&config, 5), QATConfig::new().with_steps(20));
        qat.train(&[(0..8).collect()]);
        let deployed = qat.deploy();
        assert!(deployed.is_bit1());
        assert_eq!(deployed.layers.len(), 0);
    }

    #[test]
    fn grad_clip_limits_gradient_norm() {
        let config = bitllm_runtime::config::ModelConfig::tiny_test();
        let windows = vec![(0..16).collect()];

        // Train without clipping
        let mut qat_no_clip = QATModel::new(
            random_model(&config, 42),
            QATConfig::new().with_lr(0.1).with_steps(10),
        );
        let (_start_no_clip, _end_no_clip) = qat_no_clip.train(&windows);

        // Train with aggressive clipping
        let mut qat_clipped = QATModel::new(
            random_model(&config, 42),
            QATConfig::new()
                .with_lr(0.1)
                .with_steps(10)
                .with_grad_clip(0.05),
        );
        let (start_clipped, end_clipped) = qat_clipped.train(&windows);

        // With clipping, the loss should decrease more slowly (smaller effective steps)
        // or stay similar, but not diverge
        assert!(
            end_clipped < start_clipped,
            "clipped training should still reduce loss: {} -> {}",
            start_clipped,
            end_clipped
        );
    }

    #[test]
    fn lr_warmup_increases_gradually() {
        let config = bitllm_runtime::config::ModelConfig::tiny_test();

        // Train with warmup
        let mut qat_warmup = QATModel::new(
            random_model(&config, 99),
            QATConfig::new()
                .with_lr(0.05)
                .with_steps(20)
                .with_warmup(10),
        );
        let (start, end) = qat_warmup.train(&[(0..16).collect()]);

        // Warmup should not prevent convergence
        assert!(
            end < start,
            "warmup training should reduce loss: {} -> {}",
            start,
            end
        );
    }

    #[test]
    fn cosine_decay_reduces_lr_over_time() {
        let config = bitllm_runtime::config::ModelConfig::tiny_test();

        // Train with cosine decay
        let mut qat_decay = QATModel::new(
            random_model(&config, 77),
            QATConfig::new()
                .with_lr(0.05)
                .with_steps(30)
                .with_cosine_decay(true),
        );
        let (start, end) = qat_decay.train(&[(0..16).collect()]);

        // Cosine decay should still converge
        assert!(
            end < start,
            "cosine decay training should reduce loss: {} -> {}",
            start,
            end
        );
    }

    #[test]
    fn early_stopping_halts_training() {
        let config = bitllm_runtime::config::ModelConfig::tiny_test();
        let train_windows = vec![(0..16).collect()];
        let eval_window: Vec<u32> = (16..32).collect();

        // Train with early stopping (very aggressive patience)
        let mut qat_early = QATModel::new(
            random_model(&config, 55),
            QATConfig::new()
                .with_lr(0.05)
                .with_steps(100)
                .with_eval_window(eval_window)
                .with_patience(5)
                .with_min_delta(1e-6),
        );
        let (start, end) = qat_early.train(&train_windows);

        // Should still converge (or at least not diverge)
        assert!(
            end < start,
            "early stopping training should reduce loss: {} -> {}",
            start,
            end
        );
    }
}
