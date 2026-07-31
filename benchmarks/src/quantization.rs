use std::hint::black_box;
use bitllm_quantization::scheme::QuantizedTensor;
use bitllm_quantization::ternary::{ternary_dequantize, ternary_quantize};
use bitllm_tensor::{DType, Tensor};

use crate::helpers::{print_throughput_full, time_iters};

// ── Constants ──────────────────────────────────────────────────────────
const BENCH_SIZES: [usize; 4] = [128, 256, 512, 1024];

// ── Helpers ────────────────────────────────────────────────────────────

fn ternary_output_bytes(q: &QuantizedTensor) -> usize {
    q.data.len() + q.scales.len() * 4
}

// ── Individual benchmarks ─────────────────────────────────────────────

fn bench_ternary_quantize(size: usize, iterations: usize) -> f64 {
    let t = Tensor::random(&[size, size], DType::F32);
    time_iters(iterations, || {
        black_box(ternary_quantize(black_box(&t)));
    })
    .mean
}

fn bench_ternary_dequantize(size: usize, iterations: usize) -> f64 {
    let t = Tensor::random(&[size, size], DType::F32);
    let qt = ternary_quantize(&t);
    time_iters(iterations, || {
        black_box(ternary_dequantize(black_box(&qt)));
    })
    .mean
}

// ── Main benchmark function ───────────────────────────────────────────

pub fn bench_quantization_throughput() {
    println!("\n=== Quantization Throughput ===");
    println!("  (memory-bound: higher GB/s = better)\n");

    for &size in &BENCH_SIZES {
        let n = size * size;
        let fp32_bytes = n * 4;
        let iters = if size <= 256 { 50 } else { 10 };

        println!("  {size}x{size} ({} KB):", fp32_bytes / 1024);

        // Ternary quantize + dequantize
        let t_tr = Tensor::random(&[size, size], DType::F32);
        let q_tr = ternary_quantize(&t_tr);
        let tern_out = ternary_output_bytes(&q_tr);

        let avg = bench_ternary_quantize(size, iters);
        print_throughput_full("Ternary quantize + allocate", fp32_bytes, tern_out, avg);

        let avg = bench_ternary_dequantize(size, iters);
        print_throughput_full("Ternary dequantize + allocate", tern_out, fp32_bytes, avg);

        println!();
    }
}


