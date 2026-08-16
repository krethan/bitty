use std::collections::HashMap;

use bitllm_quantization::ternary::{ternary_dequantize, ternary_quantize};
use bitllm_tensor::Tensor;

type QuantScheme = (&'static str, Box<dyn Fn(&Tensor) -> Tensor>);

// --- Metric helpers ---

fn mse(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>() / a.len() as f32
}

fn rmse(a: &[f32], b: &[f32]) -> f32 {
    mse(a, b).sqrt()
}

fn relative_rmse(a: &[f32], b: &[f32]) -> f32 {
    let ref_norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    if ref_norm == 0.0 {
        return 0.0;
    }
    rmse(a, b) / ref_norm
}

fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

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

struct Metrics {
    mse: f32,
    rmse: f32,
    rrmse: f32,
    max_err: f32,
    cos_sim: f32,
}

fn compute_metrics(orig: &[f32], recon: &[f32]) -> Metrics {
    Metrics {
        mse: mse(orig, recon),
        rmse: rmse(orig, recon),
        rrmse: relative_rmse(orig, recon),
        max_err: max_abs_error(orig, recon),
        cos_sim: cosine_similarity(orig, recon),
    }
}

// --- Deterministic RNG (xorshift64*) ---

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        // Ensure non-zero state
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
        // Uniform in [-1, 1] — use f64 intermediate to avoid u64→f32 precision loss
        ((self.next_u64() as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32
    }

    fn next_gaussian(&mut self) -> f32 {
        // Box-Muller transform
        let u1 = self.next_f32().abs().max(1e-10);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

fn random_tensor_gaussian(rng: &mut Rng, rows: usize, cols: usize) -> Tensor {
    let data: Vec<f32> = (0..rows * cols).map(|_| rng.next_gaussian()).collect();
    Tensor::from_slice(&data, &[rows, cols])
}

// --- Quantize + dequantize helpers ---

fn quantize_ternary(w: &Tensor) -> Tensor {
    let q = ternary_quantize(w);
    ternary_dequantize(&q)
}

// --- Aggregation ---

fn aggregate_metrics(all: &[Metrics]) -> (Metrics, Metrics) {
    let n = all.len() as f32;
    let mean = Metrics {
        mse: all.iter().map(|m| m.mse).sum::<f32>() / n,
        rmse: all.iter().map(|m| m.rmse).sum::<f32>() / n,
        rrmse: all.iter().map(|m| m.rrmse).sum::<f32>() / n,
        max_err: all.iter().map(|m| m.max_err).sum::<f32>() / n,
        cos_sim: all.iter().map(|m| m.cos_sim).sum::<f32>() / n,
    };
    let std = Metrics {
        mse: (all.iter().map(|m| (m.mse - mean.mse).powi(2)).sum::<f32>() / n).sqrt(),
        rmse: (all.iter().map(|m| (m.rmse - mean.rmse).powi(2)).sum::<f32>() / n).sqrt(),
        rrmse: (all.iter().map(|m| (m.rrmse - mean.rrmse).powi(2)).sum::<f32>() / n).sqrt(),
        max_err: (all.iter().map(|m| (m.max_err - mean.max_err).powi(2)).sum::<f32>() / n).sqrt(),
        cos_sim: (all.iter().map(|m| (m.cos_sim - mean.cos_sim).powi(2)).sum::<f32>() / n).sqrt(),
    };
    (mean, std)
}

fn print_agg(label: &str, mean: &Metrics, std: &Metrics) {
    println!(
        "    {:18} CosSim={:.6}+/-{:.4}  RelRMSE={:.2}%+/-{:.3}%  MaxErr={:.4}",
        label,
        mean.cos_sim,
        std.cos_sim,
        mean.rrmse * 100.0,
        std.rrmse * 100.0,
        mean.max_err,
    );
}

// --- Benchmark sections ---

fn bench_weight_reconstruction(size: usize, n_trials: usize) {
    println!("  {}x{} weight reconstruction ({} trials, seeded RNG):", size, size, n_trials);

    let schemes: Vec<QuantScheme> = vec![
        ("Ternary (2-bit)", Box::new(quantize_ternary)),
    ];

    let mut all_metrics: HashMap<&str, Vec<Metrics>> = HashMap::new();
    for &(name, _) in &schemes {
        all_metrics.insert(name, Vec::new());
    }

    for trial in 0..n_trials {
        let mut rng = Rng::new(42 + trial as u64);
        let w = random_tensor_gaussian(&mut rng, size, size);
        let orig = w.to_f32();
        let orig_slice = orig.as_f32_slice();

        for &(name, ref quant_fn) in &schemes {
            let recon = quant_fn(&w);
            let m = compute_metrics(orig_slice, recon.as_f32_slice());
            all_metrics.get_mut(name).unwrap().push(m);
        }
    }

    for &(name, _) in &schemes {
        let trials = &all_metrics[name];
        let (mean, std) = aggregate_metrics(trials);
        print_agg(name, &mean, &std);
    }
    println!();
}

fn bench_inference_output(size: usize, n_trials: usize) {
    println!("  {}x{} inference output comparison ({} trials):", size, size, n_trials);
    println!("  (output = input @ quantized_weight^T, compared to input @ weight^T)\n");

    let schemes: Vec<QuantScheme> = vec![
        ("Ternary (2-bit)", Box::new(quantize_ternary)),
    ];

    let mut all_metrics: HashMap<&str, Vec<Metrics>> = HashMap::new();
    for &(name, _) in &schemes {
        all_metrics.insert(name, Vec::new());
    }

    for trial in 0..n_trials {
        let mut rng = Rng::new(1000 + trial as u64);
        let w = random_tensor_gaussian(&mut rng, size, size);
        let input = random_tensor_gaussian(&mut rng, 1, size);

        // FP32 reference output
        let y_fp32 = input.dot(&w).unwrap();
        let y_fp32_slice = y_fp32.as_f32_slice();

        for &(name, ref quant_fn) in &schemes {
            let w_recon = quant_fn(&w);
            let y_quant = input.dot(&w_recon).unwrap();
            let m = compute_metrics(y_fp32_slice, y_quant.as_f32_slice());
            all_metrics.get_mut(name).unwrap().push(m);
        }
    }

    for &(name, _) in &schemes {
        let trials = &all_metrics[name];
        let (mean, std) = aggregate_metrics(trials);
        print_agg(name, &mean, &std);
    }
    println!();
}

pub fn bench_correctness() {
    println!("\n=== Quantization Correctness Verification ===\n");

    println!("  Metrics: CosSim (higher=better), RelRMSE (lower=better), MaxErr (lower=better)\n");

    // Weight reconstruction at multiple sizes
    bench_weight_reconstruction(64, 10);
    bench_weight_reconstruction(256, 10);
    bench_weight_reconstruction(1024, 5);

    // Inference output comparison (most important for real use)
    println!("  --- Inference Output Comparison ---\n");
    bench_inference_output(64, 10);
    bench_inference_output(256, 10);
    bench_inference_output(1024, 5);

    println!("  Note: Ternary weights are inherently lossy.");
    println!("  Their strength is in memory reduction and throughput, not weight fidelity.");
    println!("  The inference output comparison shows the actual impact on computation.\n");
}
