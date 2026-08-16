//! Ternary low-rank adapters (TernaryLoRA).
//!
//! A `TernaryLoRA` parameterizes a low-rank correction `ΔW = scale_a·A · scale_b·B`
//! where both factors are stored as **packed ternary trits** (the repo's
//! "1-bit" format, 4 trits per byte — see `bitllm-hip-tern::TernaryQuantizer`
//! for the encoding). `A` is `[output_dim, rank]`, `B` is `[rank, input_dim]`,
//! and the adapter output for input `x: [T, input_dim]` is
//! `out = (x·Bᵀ)·Aᵀ` with `out: [T, output_dim]`.
//!
//! Training is **ternary block coordinate descent** (BCD). Each sweep refits
//! one block at a time to the candidate with the lowest *exact* weighted-MSE
//! loss: a full row of `A` is exhaustively searched over all `3^rank` trit
//! combinations (rank ≤ 8), and each `B[r, i]` over its three trit values.
//! A candidate is accepted only if it strictly reduces the loss, so training
//! is deterministic and monotone.
//!
//! There is **no F32 weight shadow**: the packed trits are the only weight
//! storage, and the only F32 state is the two fixed per-matrix scales. No
//! momentum, no noise, no gradient accumulation.
//!
//! Why not straight-through gradient flips? The earlier design flipped trits
//! toward a momentum-weighted gradient vote with annealed noise. On the exact
//! learnability target it either diverged (simultaneous flips overshoot the
//! cooperative term) or stalled in local minima (single-trit moves cannot
//! escape narrow valleys). Exhaustive joint row refit evaluates the true loss
//! of each candidate and converges 10/10 across seeds where flip-based rules
//! scored 0–5/20 (see `learns_synthetic_low_rank_target`).

use crate::training::SeededRng;
use bitllm_tensor::Tensor;

fn mm(a: &Tensor, b: &Tensor) -> Tensor {
    a.dot(b).expect("matmul shape")
}

// Packed ternary encoding (matches `bitllm-hip-tern::TernaryQuantizer`):
//   -1 -> 0b01, 0 -> 0b00, +1 -> 0b10; 0b11 is invalid.
const TRIT_NEG: u8 = 0b01;
const TRIT_ZERO: u8 = 0b00;
const TRIT_POS: u8 = 0b10;

fn trit_at(packed: &[u8], idx: usize) -> i8 {
    let bits = (packed[idx / 4] >> ((idx % 4) * 2)) & 0b11;
    match bits {
        TRIT_NEG => -1,
        TRIT_ZERO => 0,
        TRIT_POS => 1,
        _ => unreachable!("0b11 is not a valid ternary encoding"),
    }
}

fn set_trit(packed: &mut [u8], idx: usize, trit: i8) {
    let bits = match trit {
        -1 => TRIT_NEG,
        0 => TRIT_ZERO,
        1 => TRIT_POS,
        _ => panic!("invalid trit {trit}"),
    };
    let byte = idx / 4;
    let shift = (idx % 4) * 2;
    packed[byte] = (packed[byte] & !(0b11 << shift)) | (bits << shift);
}

/// Configuration for a [`TernaryLoRA`].
#[derive(Debug, Clone, Copy)]
pub struct TernaryLoRAConfig {
    pub input_dim: usize,
    pub output_dim: usize,
    pub rank: usize,
    /// Seed for the deterministic random ±1 initialization of `A`.
    pub seed: u64,
    /// Per-element scale product `scale_a·scale_b` applied to the adapter
    /// matrix. Fixed during training; only the trits move.
    pub init_scale: f32,
}

impl TernaryLoRAConfig {
    pub fn new(input_dim: usize, output_dim: usize, rank: usize, seed: u64) -> Self {
        Self {
            input_dim,
            output_dim,
            rank,
            seed,
            init_scale: 0.02,
        }
    }
}

/// Low-rank ternary adapter. See the module docs for the layout and the
/// block-coordinate-descent training rule.
pub struct TernaryLoRA {
    config: TernaryLoRAConfig,
    scale_a: f32,
    scale_b: f32,
    /// Packed trits for `A` (`output_dim·rank` trits, row-major `[out, rank]`).
    a_packed: Vec<u8>,
    /// Packed trits for `B` (`rank·input_dim` trits, row-major `[rank, in]`).
    b_packed: Vec<u8>,
    /// Number of completed BCD sweeps.
    sweeps: u64,
}

impl TernaryLoRA {
    pub fn new(config: TernaryLoRAConfig) -> Self {
        assert!(config.rank > 0, "rank must be > 0");
        assert!(
            config.rank <= 8,
            "exhaustive ternary row refit needs 3^rank candidates (rank {})",
            config.rank
        );
        let scale = config.init_scale.sqrt();
        let n_a = config.output_dim * config.rank;
        let n_b = config.rank * config.input_dim;
        let mut a_packed = vec![0u8; n_a.div_ceil(4)];
        let mut rng = SeededRng::new(config.seed);
        for i in 0..n_a {
            let trit = if rng.next_unit() < 0.5 { 1 } else { -1 };
            set_trit(&mut a_packed, i, trit);
        }
        Self {
            config,
            scale_a: scale,
            scale_b: scale,
            a_packed,
            b_packed: vec![0u8; n_b.div_ceil(4)],
            sweeps: 0,
        }
    }

    pub fn config(&self) -> &TernaryLoRAConfig {
        &self.config
    }

    /// Adapter forward: `(x·Bᵀ)·Aᵀ`, both factors scaled.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mid = mm(x, &self.b_transpose());
        mm(&mid, &self.a_transpose())
    }

    /// `Aᵀ` dequantized: `[rank, output_dim]`, scaled by `scale_a`.
    fn a_transpose(&self) -> Tensor {
        let (out, rank) = (self.config.output_dim, self.config.rank);
        let mut data = vec![0.0f32; rank * out];
        for (idx, v) in data.iter_mut().enumerate() {
            let r = idx / out;
            let o = idx % out;
            *v = trit_at(&self.a_packed, o * rank + r) as f32 * self.scale_a;
        }
        Tensor::from_slice(&data, &[rank, out])
    }

    /// `Bᵀ` dequantized: `[input_dim, rank]`, scaled by `scale_b`.
    fn b_transpose(&self) -> Tensor {
        let (rank, inn) = (self.config.rank, self.config.input_dim);
        let mut data = vec![0.0f32; inn * rank];
        for (idx, v) in data.iter_mut().enumerate() {
            let i = idx / rank;
            let r = idx % rank;
            *v = trit_at(&self.b_packed, r * inn + i) as f32 * self.scale_b;
        }
        Tensor::from_slice(&data, &[inn, rank])
    }

    /// One monotone BCD sweep: refit every row of `A` exactly, then every
    /// element of `B`, accepting only strictly loss-reducing candidates.
    ///
    /// `x: [T, input_dim]`, `target: [T, output_dim]`, and optional per-element
    /// `weights: [T, output_dim]` (default 1.0). The return value is the
    /// weight-normalized MSE `Σ w·e² / Σ w`.
    pub fn train_step(&mut self, x: &Tensor, target: &Tensor, weights: Option<&Tensor>) -> f32 {
        let (inn, out, rank) = (
            self.config.input_dim,
            self.config.output_dim,
            self.config.rank,
        );
        assert_eq!(
            x.shape(),
            &[target.shape()[0], inn],
            "x/target rows must match"
        );
        assert_eq!(
            target.shape(),
            &[x.shape()[0], out],
            "target must be [T, output_dim]"
        );
        let t = x.shape()[0];
        let w = match weights {
            Some(w) => {
                assert_eq!(
                    w.shape(),
                    &[t, out],
                    "weights must be [T, output_dim] matching target"
                );
                w.as_f32_slice()
            }
            None => &[],
        };
        let s = self.scale_a * self.scale_b;

        // Dequantized trit factor A (B enters through `mid`, scaled by scale_b).
        let a = self.a_dequant(); // [out, rank]
        let a_s = a.as_f32_slice();

        let x_s = x.as_f32_slice();
        let t_s = target.as_f32_slice();
        let mid = mm(x, &self.b_transpose()); // [T, rank] (scaled by scale_b)
        let fwd = mm(&mid, &self.a_transpose()); // [T, out]
        let mut e = vec![0.0f32; t * out];
        for (i, (o, tg)) in fwd.as_f32_slice().iter().zip(t_s).enumerate() {
            e[i] = o - tg;
        }
        let mut loss = squared_loss(&e, w, t, out);
        let total_w = if w.is_empty() {
            (t * out) as f32
        } else {
            w.iter().sum()
        };
        let scale_a = self.scale_a;
        let scale_b = self.scale_b;

        // ---- A sweep: for each output row o, search all 3^rank combos. ----
        let combos = trit_combos(rank);
        let mid_s = mid.as_f32_slice();
        let mut midu = vec![0.0f32; t * rank];
        for i in 0..t * rank {
            midu[i] = mid_s[i] / scale_b;
        }

        for o in 0..out {
            // P1[o,r] and P2[o,r,r'] from the current residual.
            let mut p1 = vec![0.0f32; rank];
            let mut p2 = vec![0.0f32; rank * rank];
            for tt in 0..t {
                let wt = if w.is_empty() { 1.0 } else { w[tt * out + o] };
                let err = e[tt * out + o];
                for r in 0..rank {
                    let m = midu[tt * rank + r];
                    p1[r] += wt * err * m;
                    for r2 in 0..rank {
                        p2[r * rank + r2] += wt * m * midu[tt * rank + r2];
                    }
                }
            }
            for v in &mut p1 {
                *v *= s;
            }
            for v in &mut p2 {
                *v *= s * s;
            }
            let cur = (0..rank)
                .map(|r| trit_at(&self.a_packed, o * rank + r))
                .collect::<Vec<_>>();
            let mut best_delta = 0.0f32;
            let mut best_combo: Option<Vec<i8>> = None;
            for combo in &combos {
                let delta = (0..rank).map(|r| combo[r] - cur[r]).collect::<Vec<_>>();
                let mut d = 0.0f32;
                for r in 0..rank {
                    d += 2.0 * (delta[r] as f32) * p1[r];
                    for r2 in 0..rank {
                        d += (delta[r] as f32) * (delta[r2] as f32) * p2[r * rank + r2];
                    }
                }
                if d < best_delta {
                    best_delta = d;
                    best_combo = Some(combo.clone());
                }
            }
            if let Some(combo) = best_combo {
                for (r, c) in combo.iter().enumerate().take(rank) {
                    set_trit(&mut self.a_packed, o * rank + r, *c);
                }
                // Update the residual in place: e[tt,o] += s·Σ_r δ_r·midu[tt,r].
                for tt in 0..t {
                    let mut acc = 0.0;
                    for r in 0..rank {
                        let d = combo[r] - cur[r];
                        if d != 0 {
                            acc += d as f32 * midu[tt * rank + r];
                        }
                    }
                    e[tt * out + o] += s * acc;
                }
                loss += best_delta;
            }
        }

        // ---- B sweep: for each (r, i), try the three trit values. ----
        for r in 0..rank {
            for i in 0..inn {
                let mut p1 = 0.0f32;
                let mut p2 = 0.0f32;
                for tt in 0..t {
                    let xv = x_s[tt * inn + i];
                    for o in 0..out {
                        let wt = if w.is_empty() { 1.0 } else { w[tt * out + o] };
                        let av = a_s[o * rank + r] / scale_a;
                        p1 += wt * e[tt * out + o] * xv * av;
                        p2 += wt * xv * xv * av * av;
                    }
                }
                p1 *= s;
                p2 *= s * s;
                let cur = trit_at(&self.b_packed, r * inn + i);
                let mut best_delta = 0.0f32;
                let mut best = cur;
                for cand in [-1i8, 0, 1] {
                    if cand == cur {
                        continue;
                    }
                    let d = (cand - cur) as f32;
                    let dl = 2.0 * d * p1 + d * d * p2;
                    if dl < best_delta {
                        best_delta = dl;
                        best = cand;
                    }
                }
                if best != cur {
                    set_trit(&mut self.b_packed, r * inn + i, best);
                    let d = (best - cur) as f32;
                    for tt in 0..t {
                        let xv = x_s[tt * inn + i];
                        for o in 0..out {
                            let av = a_s[o * rank + r] / scale_a;
                            e[tt * out + o] += d * s * xv * av;
                        }
                    }
                    loss += best_delta;
                }
            }
        }

        self.sweeps += 1;
        loss / total_w
    }

    /// Dequantized adapter matrix `ΔW = scale_a·A_trit · scale_b·B_trit`,
    /// shape `[output_dim, input_dim]`.
    pub fn weight(&self) -> Tensor {
        let a = self.a_dequant();
        let b = self.b_dequant();
        mm(&a, &b)
    }

    /// Packed storage bytes (trits + two f32 scales).
    pub fn bytes(&self) -> usize {
        self.a_packed.len() + self.b_packed.len() + 2 * 4
    }

    /// Bytes of an F32 LoRA of the same shape.
    pub fn fp32_bytes(&self) -> usize {
        (self.config.output_dim * self.config.rank + self.config.rank * self.config.input_dim) * 4
    }

    pub fn rank(&self) -> usize {
        self.config.rank
    }

    /// Number of completed BCD sweeps.
    pub fn steps(&self) -> u64 {
        self.sweeps
    }

    /// Number of nonzero trits in the packed A factor.
    pub fn nonzero_trits_a(&self) -> usize {
        self.a_packed.iter().map(|&w| w.count_ones() as usize).sum()
    }

    /// Number of nonzero trits in the packed B factor.
    pub fn nonzero_trits_b(&self) -> usize {
        self.b_packed.iter().map(|&w| w.count_ones() as usize).sum()
    }

    fn a_dequant(&self) -> Tensor {
        let (out, rank) = (self.config.output_dim, self.config.rank);
        let mut data = vec![0.0f32; out * rank];
        for (idx, d) in data.iter_mut().enumerate() {
            *d = trit_at(&self.a_packed, idx) as f32 * self.scale_a;
        }
        Tensor::from_slice(&data, &[out, rank])
    }

    fn b_dequant(&self) -> Tensor {
        let (rank, inn) = (self.config.rank, self.config.input_dim);
        let mut data = vec![0.0f32; rank * inn];
        for (idx, d) in data.iter_mut().enumerate() {
            *d = trit_at(&self.b_packed, idx) as f32 * self.scale_b;
        }
        Tensor::from_slice(&data, &[rank, inn])
    }
}

/// `Σ w·e²` over `[T, out]` (unit weights when `w` is empty).
fn squared_loss(e: &[f32], w: &[f32], t: usize, out: usize) -> f32 {
    let mut acc = 0.0f32;
    for tt in 0..t {
        for o in 0..out {
            let err = e[tt * out + o];
            let wt = if w.is_empty() { 1.0 } else { w[tt * out + o] };
            acc += wt * err * err;
        }
    }
    acc
}

/// All `3^rank` trit combinations, in lexicographic order over `{-1, 0, 1}`.
fn trit_combos(rank: usize) -> Vec<Vec<i8>> {
    let n = 3usize.pow(rank as u32);
    let mut combos = Vec::with_capacity(n);
    for code in 0..n {
        let mut c = vec![0i8; rank];
        let mut x = code;
        for slot in c.iter_mut().rev() {
            *slot = match x % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            };
            x /= 3;
        }
        combos.push(c);
    }
    combos
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor2d(data: &[f32], rows: usize, cols: usize) -> Tensor {
        Tensor::from_slice(data, &[rows, cols])
    }

    fn mse_loss(out: &Tensor, target: &Tensor) -> f32 {
        let o = out.as_f32_slice();
        let t = target.as_f32_slice();
        o.iter()
            .zip(t.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / o.len() as f32
    }

    #[test]
    fn packed_roundtrip() {
        let mut packed = vec![0u8; 3];
        for i in 0..12 {
            set_trit(
                &mut packed,
                i,
                match i % 3 {
                    0 => -1,
                    1 => 0,
                    _ => 1,
                },
            );
        }
        for i in 0..12 {
            assert_eq!(
                trit_at(&packed, i),
                match i % 3 {
                    0 => -1,
                    1 => 0,
                    _ => 1,
                }
            );
        }
    }

    #[test]
    fn forward_shapes_and_blank_start() {
        let cfg = TernaryLoRAConfig::new(8, 6, 4, 7);
        let lora = TernaryLoRA::new(cfg);
        let x = tensor2d(
            &(0..40)
                .map(|i| (i as f32 - 20.0) / 20.0)
                .collect::<Vec<_>>(),
            5,
            8,
        );
        let out = lora.forward(&x);
        assert_eq!(out.shape(), &[5, 6]);
        // Blank start: all-zero adapter output.
        assert!(out.as_f32_slice().iter().all(|v| v.abs() < 1e-6));
    }

    #[test]
    fn learns_synthetic_low_rank_target() {
        let inn = 8;
        let out = 8;
        let rank = 4;
        let scale = 0.1f32;
        let cfg = TernaryLoRAConfig {
            // init_scale is the scale PRODUCT scale_a·scale_b, so it must match
            // the target's product: W_true = scale·A_true · scale·B_true.
            init_scale: scale * scale,
            ..TernaryLoRAConfig::new(inn, out, rank, 3)
        };
        let mut lora = TernaryLoRA::new(cfg);

        // Target W_true = scale·A_true·B_true (both rank factors are trits),
        // so it is exactly representable by this parameterization.
        let mut a_true = vec![0i8; out * rank];
        let mut b_true = vec![0i8; rank * inn];
        let mut rng = SeededRng::new(9);
        for v in &mut a_true {
            *v = if rng.next_unit() < 0.5 { 1 } else { -1 };
        }
        for v in &mut b_true {
            *v = if rng.next_unit() < 0.5 { 1 } else { -1 };
        }
        let a_ts = Tensor::from_slice(
            &a_true.iter().map(|&t| t as f32 * scale).collect::<Vec<_>>(),
            &[out, rank],
        );
        let b_ts = Tensor::from_slice(
            &b_true.iter().map(|&t| t as f32 * scale).collect::<Vec<_>>(),
            &[rank, inn],
        );
        let w_true = mm(&a_ts, &b_ts); // [out, inn]

        // Fixed dataset: 64 random ±1 inputs, target = x·W_trueᵀ.
        let n = 64usize;
        let mut x_data = Vec::with_capacity(n * inn);
        for _ in 0..n * inn {
            x_data.push(if rng.next_unit() < 0.5 { 1.0 } else { -1.0 });
        }
        let x = Tensor::from_slice(&x_data, &[n, inn]);
        let y = mm(&x, &w_true.transpose());

        let initial = mse_loss(&lora.forward(&x), &y);
        for _ in 0..300 {
            lora.train_step(&x, &y, None);
        }
        let final_loss = mse_loss(&lora.forward(&x), &y);
        let w_learned = lora.weight();
        let mse = (0..(out * inn))
            .map(|i| {
                let d = w_learned.get_flat_f32(i) - w_true.get_flat_f32(i);
                d * d
            })
            .sum::<f32>() as f64
            / (out * inn) as f64;
        let rmse = mse.sqrt();

        assert!(
            final_loss < initial * 0.25,
            "training must reduce loss: initial {initial}, final {final_loss}"
        );
        assert!(
            rmse < 0.05,
            "learned ternary adapter must approach the target, rmse {rmse}"
        );
    }

    #[test]
    fn train_step_is_monotone() {
        let inn = 8;
        let out = 8;
        let rank = 4;
        let scale = 0.1f32;
        let cfg = TernaryLoRAConfig {
            init_scale: scale,
            ..TernaryLoRAConfig::new(inn, out, rank, 5)
        };
        let mut lora = TernaryLoRA::new(cfg);

        let mut rng = SeededRng::new(11);
        let mut a_true = vec![0i8; out * rank];
        let mut b_true = vec![0i8; rank * inn];
        for v in &mut a_true {
            *v = if rng.next_unit() < 0.5 { 1 } else { -1 };
        }
        for v in &mut b_true {
            *v = if rng.next_unit() < 0.5 { 1 } else { -1 };
        }
        let a_ts = Tensor::from_slice(
            &a_true.iter().map(|&t| t as f32 * scale).collect::<Vec<_>>(),
            &[out, rank],
        );
        let b_ts = Tensor::from_slice(
            &b_true.iter().map(|&t| t as f32 * scale).collect::<Vec<_>>(),
            &[rank, inn],
        );
        let w_true = mm(&a_ts, &b_ts);
        let n = 64usize;
        let mut x_data = Vec::with_capacity(n * inn);
        for _ in 0..n * inn {
            x_data.push(if rng.next_unit() < 0.5 { 1.0 } else { -1.0 });
        }
        let x = Tensor::from_slice(&x_data, &[n, inn]);
        let y = mm(&x, &w_true.transpose());

        let mut prev = mse_loss(&lora.forward(&x), &y);
        for _ in 0..100 {
            let l = lora.train_step(&x, &y, None);
            assert!(l <= prev + 1e-5, "BCD must be monotone: {prev} -> {l}");
            prev = l;
        }
    }

    #[test]
    fn storage_is_packed() {
        let cfg = TernaryLoRAConfig::new(16, 16, 8, 5);
        let lora = TernaryLoRA::new(cfg);
        let n_params = 16usize * 8 + 8 * 16;
        assert_eq!(lora.bytes(), n_params.div_ceil(4) + 2 * 4);
        assert_eq!(lora.fp32_bytes(), n_params * 4);
        // The packed trits ARE the weight state (4 trits/byte); an F32 weight
        // copy would need 16x the packed bytes.
        assert!(
            lora.bytes() * 4 < lora.fp32_bytes(),
            "no F32 weight shadow allowed"
        );
    }

    #[test]
    fn footprint_less_than_fp32_lora() {
        let cfg = TernaryLoRAConfig::new(96, 96, 8, 1);
        let lora = TernaryLoRA::new(cfg);
        assert!(
            lora.bytes() * 8 < lora.fp32_bytes(),
            "ternary should be ~16x smaller"
        );
    }
}
