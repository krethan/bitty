use std::hint::black_box;

use bitllm_quantization::absmax::{absmax_dequantize, absmax_quantize};
use bitllm_quantization::group::GroupQuantizer;
use bitllm_quantization::quantized_matmul;
use bitllm_quantization::scheme::QuantConfig;
use bitllm_quantization::ternary::{ternary_dequantize, ternary_quantize};
use bitllm_tensor::{BinaryTensor, Tensor};

use crate::export::PrecisionRow;
use crate::helpers::time_iters;

// ── Metrics ──────────────────────────────────────────────────────────

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

fn relative_rmse(a: &[f32], b: &[f32]) -> f32 {
    let ref_norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    if ref_norm == 0.0 {
        return 0.0;
    }
    let rmse: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
        / a.len() as f32;
    rmse / ref_norm
}

fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Signal-to-noise ratio in dB. Higher = better reconstruction.
fn signal_to_noise_ratio(original: &[f32], reconstructed: &[f32]) -> f32 {
    let signal_power: f32 = original.iter().map(|x| x * x).sum();
    let noise_power: f32 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum();
    if noise_power == 0.0 {
        return f32::INFINITY;
    }
    10.0 * (signal_power / noise_power).log10()
}

// ── Deterministic RNG ────────────────────────────────────────────────

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        self.state.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32
    }

    fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().abs().max(1e-10);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

fn random_tensor(rng: &mut Rng, rows: usize, cols: usize) -> Tensor {
    let data: Vec<f32> = (0..rows * cols).map(|_| rng.next_gaussian()).collect();
    Tensor::from_slice(&data, &[rows, cols])
}

// ── Quantize/dequantize wrappers ─────────────────────────────────────

fn quantize_int8(w: &Tensor) -> (Tensor, usize) {
    let q = absmax_quantize(w, &QuantConfig::int8());
    let bytes = q.data.len() + q.scales.len() * 4;
    (absmax_dequantize(&q), bytes)
}

fn quantize_int4(w: &Tensor) -> (Tensor, usize) {
    let qg = GroupQuantizer::new(128);
    let q = qg.quantize_int4(w);
    let bytes = q.data.len()
        + q.scales.len() * 4
        + q.zeros.as_ref().map_or(0, |z| z.len() * 4);
    (qg.dequantize_int4(&q), bytes)
}

fn quantize_ternary(w: &Tensor) -> (Tensor, usize) {
    let q = ternary_quantize(w);
    let bytes = q.data.len() + q.scales.len() * 4;
    (ternary_dequantize(&q), bytes)
}

fn quantize_binary(w: &Tensor) -> (Tensor, usize) {
    let bt = BinaryTensor::from_tensor(w);
    let bytes = bt.nbytes() + bt.scales.len() * std::mem::size_of::<f32>();
    (bt.dequantize(), bytes)
}

// ── Correctness measurement ──────────────────────────────────────────

struct QualityResult {
    cos_sim: f64,
    rel_rmse_pct: f64,
    max_err: f64,
    snr_db: f64,
}

fn measure_quality(
    w: &Tensor,
    quant_fn: &dyn Fn(&Tensor) -> (Tensor, usize),
    n_trials: usize,
) -> (QualityResult, usize) {
    let size = w.shape()[0];
    let mut cos_sims = Vec::new();
    let mut rel_rmses = Vec::new();
    let mut max_errs = Vec::new();
    let mut snrs = Vec::new();
    let mut weight_bytes = 0;

    for trial in 0..n_trials {
        let mut rng = Rng::new(42 + trial as u64);
        let w_trial = random_tensor(&mut rng, size, size);
        let orig = w_trial.to_f32();
        let orig_slice = orig.as_f32_slice();

        let (recon, bytes) = quant_fn(&w_trial);
        let recon_slice = recon.as_f32_slice();

        cos_sims.push(cosine_similarity(orig_slice, recon_slice) as f64);
        rel_rmses.push(relative_rmse(orig_slice, recon_slice) as f64 * 100.0);
        max_errs.push(max_abs_error(orig_slice, recon_slice) as f64);
        snrs.push(signal_to_noise_ratio(orig_slice, recon_slice) as f64);

        if trial == 0 {
            weight_bytes = bytes;
        }
    }

    let n = cos_sims.len() as f64;
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

fn measure_matmul_fp32(input: &Tensor, w_t: &Tensor) -> f64 {
    let iters = 3;
    let result = time_iters(iters, || {
        let _ = black_box(input.dot(black_box(w_t)));
    });
    result.mean
}

fn measure_matmul_int8(w: &Tensor, input: &Tensor) -> f64 {
    let q = absmax_quantize(w, &QuantConfig::int8());
    let iters = 3;
    let result = time_iters(iters, || {
        let _ = black_box(quantized_matmul(black_box(input), black_box(&q)));
    });
    result.mean
}

fn measure_matmul_int4(w: &Tensor, input: &Tensor) -> f64 {
    let qg = GroupQuantizer::new(128);
    let q = qg.quantize_int4(w);
    let iters = 3;
    let result = time_iters(iters, || {
        let _ = black_box(quantized_matmul(black_box(input), black_box(&q)));
    });
    result.mean
}

fn measure_matmul_ternary(w: &Tensor, input: &Tensor) -> f64 {
    let q = ternary_quantize(w);
    let iters = 10;
    let result = time_iters(iters, || {
        let _ = black_box(quantized_matmul(black_box(input), black_box(&q)));
    });
    result.mean
}

fn measure_matmul_binary(w: &Tensor, input: &Tensor) -> f64 {
    let bt = BinaryTensor::from_tensor(w);
    let iters = 10;
    let result = time_iters(iters, || {
        let _ = black_box(bt.matmul(black_box(input)));
    });
    result.mean
}

// ── Tok/s projection ─────────────────────────────────────────────────

fn project_toks_per_sec(matmul_ms: f64) -> f64 {
    // Llama-7B: 32 layers, 7 matmuls per layer
    // Per layer (4096 hidden):
    //   Q,K,V,O projections: 4 x (4096 x 4096)
    //   MLP gate+up: 2 x (4096 x 11008)
    //   MLP down: 1 x (11008 x 4096)
    // Total FLOPs per layer:
    //   4 * 4096^2 + 2 * 4096*11008 + 11008*4096 = 4*4096^2 + 3*4096*11008
    let qkv_flops = 4096.0 * 4096.0;
    let mlp_flops = 4096.0 * 11008.0;
    let avg_per_matmul_flops = (qkv_flops * 4.0 + mlp_flops * 3.0) / 7.0;
    let baseline_flops = 1024.0 * 1024.0;
    let scale = avg_per_matmul_flops / baseline_flops;
    let avg_per_matmul_ms = matmul_ms * scale;
    let per_token_ms = avg_per_matmul_ms * 32.0; // 32 layers
    1000.0 / per_token_ms
}

/// Effective memory bandwidth: weight_bytes / matmul_time.
/// This is the metric that explains why binary wins on GPUs.
fn effective_bandwidth_gbps(weight_bytes: usize, matmul_ms: f64) -> f64 {
    if matmul_ms == 0.0 {
        return 0.0;
    }
    // For a single matmul: we read weight_bytes worth of data
    let bytes = weight_bytes as f64;
    let seconds = matmul_ms / 1000.0;
    bytes / seconds / 1e9
}

// ── Main benchmark ───────────────────────────────────────────────────

pub fn bench_precision_comparison() -> Vec<PrecisionRow> {
    println!("\n=== Bitty Precision Comparison ===\n");

    let size = 1024;
    let n_trials = 5;
    let fp32_bytes = size * size * 4;

    let mut rng = Rng::new(42);
    let w = random_tensor(&mut rng, size, size);
    let input = random_tensor(&mut rng, 1, size);
    let w_t = w.transpose();

    println!("  Model: Llama-7B projection (derived from 1024x1024 kernels)\n");

    let mut rows = Vec::new();

    // FP32 baseline
    let fp32_ms = measure_matmul_fp32(&input, &w_t);
    let fp32_tps = project_toks_per_sec(fp32_ms);
    rows.push(PrecisionRow {
        name: "FP32".to_string(),
        weight_bytes: fp32_bytes,
        compression_ratio: 1.0,
        cos_sim: 1.0,
        rel_rmse_pct: 0.0,
        max_err: 0.0,
        matmul_ms: fp32_ms * 1000.0,
        tok_per_sec: fp32_tps,
    });

    // INT8
    let (q_int8, int8_bytes) = measure_quality(&w, &|w| quantize_int8(w), n_trials);
    let int8_ms = measure_matmul_int8(&w, &input);
    let int8_tps = project_toks_per_sec(int8_ms);
    rows.push(PrecisionRow {
        name: "INT8".to_string(),
        weight_bytes: int8_bytes,
        compression_ratio: fp32_bytes as f64 / int8_bytes as f64,
        cos_sim: q_int8.cos_sim,
        rel_rmse_pct: q_int8.rel_rmse_pct,
        max_err: q_int8.max_err,
        matmul_ms: int8_ms * 1000.0,
        tok_per_sec: int8_tps,
    });

    // INT4
    let (q_int4, int4_bytes) = measure_quality(&w, &|w| quantize_int4(w), n_trials);
    let int4_ms = measure_matmul_int4(&w, &input);
    let int4_tps = project_toks_per_sec(int4_ms);
    rows.push(PrecisionRow {
        name: "INT4".to_string(),
        weight_bytes: int4_bytes,
        compression_ratio: fp32_bytes as f64 / int4_bytes as f64,
        cos_sim: q_int4.cos_sim,
        rel_rmse_pct: q_int4.rel_rmse_pct,
        max_err: q_int4.max_err,
        matmul_ms: int4_ms * 1000.0,
        tok_per_sec: int4_tps,
    });

    // Ternary (2-bit)
    let (q_tern, tern_bytes) = measure_quality(&w, &|w| quantize_ternary(w), n_trials);
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

    // Binary (1-bit) — uses BinaryTensor::matmul (XNOR+popcount path)
    let (q_bin, bin_bytes) = measure_quality(&w, &|w| quantize_binary(w), n_trials);
    let bin_ms = measure_matmul_binary(&w, &input);
    let bin_tps = project_toks_per_sec(bin_ms);
    rows.push(PrecisionRow {
        name: "Binary".to_string(),
        weight_bytes: bin_bytes,
        compression_ratio: fp32_bytes as f64 / bin_bytes as f64,
        cos_sim: q_bin.cos_sim,
        rel_rmse_pct: q_bin.rel_rmse_pct,
        max_err: q_bin.max_err,
        matmul_ms: bin_ms * 1000.0,
        tok_per_sec: bin_tps,
    });

    // Print precision table
    println!(
        "  {:<12} {:>10} {:>10} {:>12} {:>12} {:>10} {:>8}",
        "Precision", "Weight(KB)", "Compress", "CosSim", "RelRMSE(%)", "Matmul(ms)", "tok/s"
    );
    println!("  {:─<12} {:─>10} {:─>10} {:─>12} {:─>12} {:─>10} {:─>8}", "", "", "", "", "", "", "");
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
        let snr = if i == 0 {
            f64::INFINITY
        } else {
            match row.name.as_str() {
                "INT8" => q_int8.snr_db,
                "INT4" => q_int4.snr_db,
                "Ternary" => q_tern.snr_db,
                "Binary" => q_bin.snr_db,
                _ => 0.0,
            }
        };
        let rating = if snr == f64::INFINITY {
            "ref".to_string()
        } else if snr > 40.0 {
            "excellent".to_string()
        } else if snr > 30.0 {
            "good".to_string()
        } else if snr > 20.0 {
            "fair".to_string()
        } else {
            "lossy".to_string()
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
    println!("  {:<12} {:>10} {:>12} {:>12}", "Precision", "Weight(KB)", "GB/s eff.", "vs FP32");
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
    println!("  Derived from 1024x1024 kernels, projected to Llama-7B (32 layers, 7 matmuls/layer).");
    println!("  CosSim/RelRMSE/SNR averaged over {} trials with Gaussian random weights.", n_trials);
    println!("  Effective bandwidth = weight_bytes / matmul_time (memory bandwidth utilization).\n");

    rows
}
