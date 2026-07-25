use std::time::Instant;

// ── Benchmark result with statistics ──────────────────────────────────

#[allow(dead_code)]
pub struct BenchmarkResult {
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub stddev: f64,
    pub iters: usize,
}

#[allow(dead_code)]
impl BenchmarkResult {
    pub fn summary(&self) -> String {
        format!(
            "{:.3} ms  [{:.3}–{:.3}, σ={:.3}]",
            self.mean * 1000.0,
            self.min * 1000.0,
            self.max * 1000.0,
            self.stddev * 1000.0,
        )
    }
}

// ── Benchmark kind ────────────────────────────────────────────────────

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub enum BenchmarkKind {
    Memory,
    Compute,
}

// ── Core timing ───────────────────────────────────────────────────────

const WARMUP_ITERS: usize = 5;

pub fn time_iters<F: FnMut()>(iterations: usize, mut f: F) -> BenchmarkResult {
    // Warm-up: let caches, branch predictors, and CPU frequency stabilize
    for _ in 0..WARMUP_ITERS {
        f();
    }

    // Timed runs — collect per-iteration samples for statistics
    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        std::hint::black_box(f());
        times.push(start.elapsed().as_secs_f64());
    }

    let mean = times.iter().sum::<f64>() / iterations as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / iterations as f64;
    let stddev = variance.sqrt();

    BenchmarkResult {
        mean,
        min,
        max,
        stddev,
        iters: iterations,
    }
}

/// Auto-scale iteration count to target ~200 ms of total benchmark time.
/// Returns (iterations, pre_measured_avg) — reuse pre_measured_avg to avoid
/// re-running the first benchmark.
pub fn auto_iters<F: FnMut()>(mut f: F) -> (usize, f64) {
    let warmup = 3;
    for _ in 0..warmup {
        f();
    }

    // Single iteration to estimate runtime
    let start = Instant::now();
    std::hint::black_box(f());
    let single = start.elapsed().as_secs_f64();

    let target = 0.2; // 200 ms
    let iters = if single <= 0.0 {
        100
    } else if single >= target {
        3.min(100)
    } else {
        ((target / single).ceil() as usize).clamp(3, 500)
    };

    // Use the single measurement as the pre-measured result
    (iters, single)
}

/// Simple iteration count heuristic (when auto_iters is not needed).
#[allow(dead_code)]
pub fn iter_count(size: usize) -> usize {
    if size <= 128 {
        20
    } else if size <= 512 {
        10
    } else if size <= 2048 {
        3
    } else {
        1
    }
}

// ── Throughput helpers ────────────────────────────────────────────────

fn throughput_gbps(bytes: usize, secs: f64) -> f64 {
    bytes as f64 / secs / 1e9
}

fn bitops(bytes: usize, secs: f64) -> f64 {
    bytes as f64 * 8.0 / secs / 1e12
}

fn gflops(mnk: (usize, usize, usize), secs: f64) -> f64 {
    let flops = 2.0 * mnk.0 as f64 * mnk.1 as f64 * mnk.2 as f64;
    flops / secs / 1e9
}

// ── Print functions ───────────────────────────────────────────────────

#[allow(dead_code)]
pub fn print_throughput(label: &str, bytes: usize, result: &BenchmarkResult) {
    println!(
        "  {:44} {:37} ({:6.2} GB/s)",
        label,
        result.summary(),
        throughput_gbps(bytes, result.mean),
    );
}

pub fn print_throughput_raw(label: &str, bytes: usize, avg_secs: f64) {
    println!(
        "  {:44} {:8.3} ms  ({:6.2} GB/s)  ({:6.1} Tbit/s)",
        label,
        avg_secs * 1000.0,
        throughput_gbps(bytes, avg_secs),
        bitops(bytes, avg_secs),
    );
}

pub fn print_throughput_full(label: &str, input_bytes: usize, output_bytes: usize, avg_secs: f64) {
    let compression = input_bytes as f64 / output_bytes.max(1) as f64;
    println!(
        "  {:44} {:8.3} ms  {:>7.1}x  ({:6.2} GB/s read, {:6.2} GB/s write)",
        label,
        avg_secs * 1000.0,
        compression,
        throughput_gbps(input_bytes, avg_secs),
        throughput_gbps(output_bytes, avg_secs),
    );
}

#[allow(dead_code)]
pub fn print_compute(label: &str, mnk: (usize, usize, usize), result: &BenchmarkResult) {
    let bytes = mnk.0 * mnk.2 * 4 + mnk.1 * mnk.2 * 4 + mnk.0 * mnk.1 * 4; // read + write estimate
    println!(
        "  {:44} {:37} {:6.1} GFLOPS  ({:6.2} GB/s)",
        label,
        result.summary(),
        gflops(mnk, result.mean),
        throughput_gbps(bytes, result.mean),
    );
}

pub fn print_compute_raw(label: &str, mnk: (usize, usize, usize), avg_secs: f64) {
    let bytes = mnk.0 * mnk.2 * 4 + mnk.1 * mnk.2 * 4 + mnk.0 * mnk.1 * 4;
    println!(
        "  {:44} {:8.3} ms  {:6.1} GFLOPS  ({:6.2} GB/s)",
        label,
        avg_secs * 1000.0,
        gflops(mnk, avg_secs),
        throughput_gbps(bytes, avg_secs),
    );
}
