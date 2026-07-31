use std::hint::black_box;
use std::sync::OnceLock;

use bitllm_quantization::qmatmul::fused_bit1_matmul;
use bitllm_quantization::scheme::QuantizedTensor;
use bitllm_quantization::ternary::{ternary_dequantize, ternary_quantize};
use bitllm_tensor::Tensor;
use rayon::prelude::*;

use crate::export::PrecisionRow;
use crate::helpers::time_iters;

// ── Constants ─────────────────────────────────────────────────────────
const MATRIX_SIZE: usize = 1024;
const LLAMA_LAYERS: usize = 32;
const MATMULS_PER_LAYER: usize = 7;
const QKV_FLOPS: f64 = 4096.0 * 4096.0;
const MLP_FLOPS: f64 = 4096.0 * 11008.0;
const BASELINE_FLOPS: f64 = 1024.0 * 1024.0;
const N_TRIALS: usize = 5;

// ── Metrics ──────────────────────────────────────────────────────────

/// Computes the cosine similarity between two slices of `f32` values.
/// Returns `0.0` if either slice is empty or has a norm of `0.0`.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| *x as f64 * *y as f64).sum();
    let norm_a: f64 = a.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|y| *y as f64 * *y as f64).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        (dot / (norm_a * norm_b)) as f32
    }
}

/// Computes the relative RMSE between two slices of `f32` values.
/// Returns `0.0` if either slice is empty or the reference norm is `0.0`.
fn relative_rmse(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let ref_norm: f64 = a.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
    if ref_norm == 0.0 {
        return 0.0;
    }
    let rmse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
        .sum::<f64>()
        .sqrt()
        / a.len() as f64;
    (rmse / ref_norm) as f32
}

/// Computes the maximum absolute error between two slices of `f32` values.
/// Returns `0.0` if either slice is empty.
fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Computes the signal-to-noise ratio (SNR) in dB between two slices of `f32` values.
/// Returns `f32::INFINITY` if the noise power is `0.0` or if either slice is empty.
fn signal_to_noise_ratio(original: &[f32], reconstructed: &[f32]) -> f32 {
    if original.is_empty() || reconstructed.is_empty() {
        return 0.0;
    }
    let signal_power: f64 = original.iter().map(|x| *x as f64 * *x as f64).sum();
    let noise_power: f64 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
        .sum();
    if noise_power == 0.0 {
        return f32::INFINITY;
    }
    (10.0 * (signal_power / noise_power).log10()) as f32
}

// ── Deterministic RNG ────────────────────────────────────────────────

/// A simple deterministic random number generator for reproducibility.
struct Rng {
    state: u64,
}

impl Rng {
    /// Creates a new `Rng` with the given seed. If `seed` is `0`, it defaults to `1`.
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Generates the next `u64` random number.
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        self.state.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Generates the next `f32` random number in the range `[-1.0, 1.0]`.
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32
    }

    /// Generates the next `f32` random number from a Gaussian distribution.
    fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().abs().max(1e-10);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// Generates a random tensor with Gaussian-distributed values.
fn random_tensor(rng: &mut Rng, rows: usize, cols: usize) -> Tensor {
    let data: Vec<f32> = (0..rows * cols).map(|_| rng.next_gaussian()).collect();
    Tensor::from_slice(&data, &[rows, cols])
}

// ── Quantize/dequantize wrappers ─────────────────────────────────────

/// Quantizes a tensor to Ternary and returns the dequantized tensor and its size in bytes.
fn quantize_ternary(w: &Tensor) -> (Tensor, usize) {
    let q = ternary_quantize(w);
    let bytes = q.data.len() + q.scales.len() * 4;
    (ternary_dequantize(&q), bytes)
}

// ── Correctness measurement ──────────────────────────────────────────

/// Holds the quality metrics for a quantization scheme.
#[derive(Debug, Clone)]
struct QualityResult {
    cos_sim: f64,
    rel_rmse_pct: f64,
    max_err: f64,
    snr_db: f64,
}

/// Measures the quality of a quantization scheme over `N_TRIALS` trials.
/// Returns the average quality metrics and the size of the quantized weights in bytes.
fn measure_quality(
    w: &Tensor,
    quant_fn: &(dyn Fn(&Tensor) -> (Tensor, usize) + Sync),
) -> (QualityResult, usize) {
    let size = w.shape()[0];

    let results: Vec<_> = (0..N_TRIALS)
        .into_par_iter()
        .map(|trial| {
            let mut rng = Rng::new(42 + trial as u64);
            let w_trial = random_tensor(&mut rng, size, size);
            let orig = w_trial.to_f32();
            let orig_slice = orig.as_f32_slice();
            let (recon, bytes) = quant_fn(&w_trial);
            let recon_slice = recon.as_f32_slice();
            (
                cosine_similarity(orig_slice, recon_slice) as f64,
                relative_rmse(orig_slice, recon_slice) as f64 * 100.0,
                max_abs_error(orig_slice, recon_slice) as f64,
                signal_to_noise_ratio(orig_slice, recon_slice) as f64,
                bytes,
            )
        })
        .collect();

    let mut cos_sims = Vec::with_capacity(N_TRIALS);
    let mut rel_rmses = Vec::with_capacity(N_TRIALS);
    let mut max_errs = Vec::with_capacity(N_TRIALS);
    let mut snrs = Vec::with_capacity(N_TRIALS);
    let mut weight_bytes = 0;
    for (i, (cs, rm, me, snr, bytes)) in results.into_iter().enumerate() {
        cos_sims.push(cs);
        rel_rmses.push(rm);
        max_errs.push(me);
        snrs.push(snr);
        if i == 0 {
            weight_bytes = bytes;
        }
    }

    let n = N_TRIALS as f64;
    (
        QualityResult {
            cos_sim: cos_sims.iter().sum::<f64>() / n,
            rel_rmse_pct: rel_rmses.iter().sum::<f64>() / n,
            max_err: max_errs.iter().sum::<f64>() / n,
            snr_db: snrs.iter().sum::<f64>() / n,
        },
        weight_bytes,
    )
}

// ── Matmul timing ────────────────────────────────────────────────────

/// Measures the average time (in seconds) for a matrix multiplication operation.
/// Uses `black_box` to prevent compiler optimizations from skewing results.
fn measure_matmul<F, Q, R>(input: &Tensor, w_t: &Q, matmul_fn: F, iters: usize) -> f64
where
    F: Fn(&Tensor, &Q) -> R,
{
    time_iters(iters, || {
        let _ = black_box(matmul_fn(black_box(input), black_box(w_t)));
    })
    .mean
}

/// Measures the time for FP32 matrix multiplication.
fn measure_matmul_fp32(input: &Tensor, w_t: &Tensor) -> f64 {
    measure_matmul(input, w_t, |a, b| a.dot(b), 3)
}

/// Measures the time for Ternary quantized matrix multiplication.
fn measure_matmul_ternary(w: &Tensor, input: &Tensor) -> f64 {
    static CACHE: OnceLock<QuantizedTensor> = OnceLock::new();
    let q = CACHE.get_or_init(|| ternary_quantize(w));
    let m = input.shape()[0];
    let k = input.shape()[1];
    let n = w.shape()[0];
    let mut out = vec![0.0f32; m * n];
    let input_slice = input.as_f32_slice();
    time_iters(10, || {
        fused_bit1_matmul(input_slice, q, &mut out, m, k, n);
        black_box(&out);
    })
    .mean
}

// ── Tok/s projection ─────────────────────────────────────────────────

/// Projects the matrix multiplication time to tokens per second for Llama-7B.
fn project_toks_per_sec(matmul_ms: f64) -> f64 {
    let avg_per_matmul_flops = (QKV_FLOPS * 4.0 + MLP_FLOPS * 3.0) / MATMULS_PER_LAYER as f64;
    let scale = avg_per_matmul_flops / BASELINE_FLOPS;
    let avg_per_matmul_ms = matmul_ms * scale;
    let per_token_ms = avg_per_matmul_ms * LLAMA_LAYERS as f64;
    1000.0 / per_token_ms
}

/// Computes the effective memory bandwidth in GB/s.
/// Returns `0.0` if `matmul_ms` is `0.0`.
fn effective_bandwidth_gbps(weight_bytes: usize, matmul_ms: f64) -> f64 {
    if matmul_ms == 0.0 {
        return 0.0;
    }
    let bytes = weight_bytes as f64;
    let seconds = matmul_ms / 1000.0;
    bytes / seconds / 1e9
}

// ── Main benchmark ───────────────────────────────────────────────────

/// Runs a benchmark comparing the precision, performance, and memory usage of different quantization schemes.
/// Returns a vector of `PrecisionRow` structs, each representing the results for a scheme.
pub fn bench_precision_comparison() -> Vec<PrecisionRow> {
    println!("\n=== Bitty Precision Comparison ===\n");

    let size = MATRIX_SIZE;
    let fp32_bytes = size * size * 4;

    let mut rng = Rng::new(42);
    let w = random_tensor(&mut rng, size, size);
    let input = random_tensor(&mut rng, 1, size);
    let w_t = w.transpose();

    println!("  Model: Llama-7B projection (derived from {size}x{size} kernels)\n");

    // FP32 baseline
    let fp32_ms = measure_matmul_fp32(&input, &w_t);
    let fp32_tps = project_toks_per_sec(fp32_ms);
    let mut rows = vec![PrecisionRow {
        name: "FP32".to_string(),
        weight_bytes: fp32_bytes,
        compression_ratio: 1.0,
        cos_sim: 1.0,
        rel_rmse_pct: 0.0,
        max_err: 0.0,
        matmul_ms: fp32_ms * 1000.0,
        tok_per_sec: fp32_tps,
    }];

    // Ternary (2-bit)
    let (q_tern, tern_bytes) = measure_quality(&w, &|w| quantize_ternary(w));
    let tern_ms = measure_matmul_ternary(&w, &input);
    let tern_tps = project_toks_per_sec(tern_ms);
    rows.push(PrecisionRow {
        name: "Ternary".to_string(),
        weight_bytes: tern_bytes,
        compression_ratio: fp32_bytes as f64 / tern_bytes as f64,
        cos_sim: q_tern.cos_sim,
        rel_rmse_pct: q_tern.rel_rmse_pct,
        max_err: q_tern.max_err,
        matmul_ms: tern_ms * 1000.0,
        tok_per_sec: tern_tps,
    });

    // Print precision table
    println!(
        "  {:<12} {:>10} {:>10} {:>12} {:>12} {:>10} {:>8}",
        "Precision", "Weight(KB)", "Compress", "CosSim", "RelRMSE(%)", "Matmul(ms)", "tok/s"
    );
    println!(
        "  {:─<12} {:─>10} {:─>10} {:─>12} {:─>12} {:─>10} {:─>8}",
        "", "", "", "", "", "", ""
    );
    for row in &rows {
        println!(
            "  {:<12} {:>10.1} {:>9.1}x {:>12.6} {:>11.4}% {:>10.1} {:>8.2}",
            row.name,
            row.weight_bytes as f64 / 1024.0,
            row.compression_ratio,
            row.cos_sim,
            row.rel_rmse_pct,
            row.matmul_ms,
            row.tok_per_sec,
        );
    }
    println!();

    // Print SNR table
    println!("  {:<12} {:>10} {:>10}", "Precision", "SNR (dB)", "Rating");
    println!("  {:─<12} {:─>10} {:─>10}", "", "", "");
    for (i, row) in rows.iter().enumerate() {
        let snr = match i {
            0 => f64::INFINITY,
            1 => q_tern.snr_db,
            _ => 0.0,
        };
        let rating = match snr {
            f64::INFINITY => "ref",
            s if s > 40.0 => "excellent",
            s if s > 30.0 => "good",
            s if s > 20.0 => "fair",
            _ => "lossy",
        };
        let snr_str = if snr == f64::INFINITY {
            "  inf".to_string()
        } else {
            format!("{:>8.1}", snr)
        };
        println!("  {:<12} {:>10} {:>10}", row.name, snr_str, rating);
    }
    println!();

    // Print effective bandwidth table
    println!(
        "  {:<12} {:>10} {:>12} {:>12}",
        "Precision", "Weight(KB)", "GB/s eff.", "vs FP32"
    );
    println!("  {:─<12} {:─>10} {:─>12} {:─>12}", "", "", "", "");
    let fp32_bw = effective_bandwidth_gbps(fp32_bytes, fp32_ms);
    for row in &rows {
        let bw = effective_bandwidth_gbps(row.weight_bytes, row.matmul_ms / 1000.0);
        let vs_fp32 = if fp32_bw > 0.0 { bw / fp32_bw } else { 0.0 };
        println!(
            "  {:<12} {:>10.1} {:>11.2} {:>11.2}x",
            row.name,
            row.weight_bytes as f64 / 1024.0,
            bw,
            vs_fp32,
        );
    }
    println!();
    println!(
        "  Derived from {size}x{size} kernels, projected to Llama-7B ({LLAMA_LAYERS} layers, {MATMULS_PER_LAYER} matmuls/layer)."
    );
    println!("  CosSim/RelRMSE/SNR averaged over {N_TRIALS} trials with Gaussian random weights.");
    println!(
        "  Effective bandwidth = weight_bytes / matmul_time (memory bandwidth utilization).\n"
    );

    rows
}
