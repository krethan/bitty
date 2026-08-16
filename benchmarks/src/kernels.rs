use bitllm_tensor::simd;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::export::KernelBench;
use crate::helpers::{print_throughput_raw, time_iters};

fn bench_xnor_popcount_1bit(size: usize, iterations: usize) -> f64 {
    let n_bytes = size.div_ceil(8);
    let mut rng = StdRng::seed_from_u64(42);
    let a: Vec<u8> = (0..n_bytes).map(|_| rng.gen()).collect();
    let b: Vec<u8> = (0..n_bytes).map(|_| rng.gen()).collect();
    let mut popcounts = vec![0u32; n_bytes];
    time_iters(iterations, || {
        simd::xnor_popcount_1bit(&a, &b, &mut popcounts, size);
    })
    .mean
}

fn bench_xnor_popcount_2bit(size: usize, iterations: usize) -> f64 {
    let n_packed = size.div_ceil(4);
    let mut rng = StdRng::seed_from_u64(42);
    let a: Vec<u8> = (0..n_packed).map(|_| rng.gen()).collect();
    let b: Vec<u8> = (0..n_packed).map(|_| rng.gen()).collect();
    let mut out = vec![0u8; n_packed];
    time_iters(iterations, || {
        simd::xnor_popcount_2bit(&a, &b, &mut out, size);
    })
    .mean
}

pub fn bench_kernels() {
    println!("\n=== XNOR+Popcount Kernels ===\n");

    println!("  --- 1-bit Binary ---");
    println!("  (maximum compression: 8 elements per byte)\n");
    for &size in &[64, 256, 1024, 4096] {
        let iters = if size <= 256 { 100 } else { 20 };
        let avg = bench_xnor_popcount_1bit(size, iters);
        let n_bytes = size.div_ceil(8);
        print_throughput_raw(&format!("xnor_popcount_1bit ({})", size), n_bytes * 2, avg);
    }

    println!();
    println!("  --- 2-bit Ternary ---");
    println!("  (higher accuracy mode: 4 elements per byte)\n");
    for &size in &[64, 256, 1024, 4096] {
        let iters = if size <= 256 { 100 } else { 20 };
        let avg = bench_xnor_popcount_2bit(size, iters);
        let n_packed = size.div_ceil(4);
        print_throughput_raw(&format!("xnor_popcount_2bit ({})", size), n_packed * 2, avg);
    }
    println!();
}

pub fn collect_kernel_results() -> Vec<KernelBench> {
    let mut results = Vec::new();

    for &size in &[64, 256, 1024, 4096] {
        let iters = if size <= 256 { 100 } else { 20 };
        let avg = bench_xnor_popcount_1bit(size, iters);
        let n_bytes = size.div_ceil(8);
        let bytes_touched = n_bytes * 2;
        let gbps = bytes_touched as f64 / avg / 1e9;
        let tbit = bytes_touched as f64 * 8.0 / avg / 1e12;
        results.push(KernelBench {
            name: "xnor_popcount_1bit".to_string(),
            size,
            time_ms: avg * 1000.0,
            gbps,
            tbit_per_sec: tbit,
        });
    }

    for &size in &[64, 256, 1024, 4096] {
        let iters = if size <= 256 { 100 } else { 20 };
        let avg = bench_xnor_popcount_2bit(size, iters);
        let n_packed = size.div_ceil(4);
        let bytes_touched = n_packed * 2;
        let gbps = bytes_touched as f64 / avg / 1e9;
        let tbit = bytes_touched as f64 * 8.0 / avg / 1e12;
        results.push(KernelBench {
            name: "xnor_popcount_2bit".to_string(),
            size,
            time_ms: avg * 1000.0,
            gbps,
            tbit_per_sec: tbit,
        });
    }

    results
}
