use bitllm_tensor::Tensor;
use bitllm_quantization::absmax::{absmax_quantize, absmax_dequantize};
use bitllm_quantization::scheme::QuantConfig;
use bitllm_quantization::group::GroupQuantizer;
use std::time::Instant;

fn bench_tensor_creation(size: usize, iterations: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        let _t = Tensor::zeros(&[size, size], bitllm_tensor::DType::F32);
    }
    start.elapsed().as_secs_f64() / iterations as f64
}

fn bench_matmul(size: usize, iterations: usize) -> f64 {
    let a = Tensor::random(&[size, size], bitllm_tensor::DType::F32);
    let b = Tensor::random(&[size, size], bitllm_tensor::DType::F32);
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = a.dot(&b).unwrap();
    }
    start.elapsed().as_secs_f64() / iterations as f64
}

fn bench_quantize_int8(size: usize, iterations: usize) -> f64 {
    let t = Tensor::random(&[size, size], bitllm_tensor::DType::F32);
    let config = QuantConfig::int8();
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = absmax_quantize(&t, &config);
    }
    start.elapsed().as_secs_f64() / iterations as f64
}

fn bench_quantize_int4(size: usize, iterations: usize) -> f64 {
    let t = Tensor::random(&[size, size], bitllm_tensor::DType::F32);
    let q = GroupQuantizer::new(128);
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = q.quantize_int4(&t);
    }
    start.elapsed().as_secs_f64() / iterations as f64
}

fn bench_dequantize_int8(size: usize, iterations: usize) -> f64 {
    let t = Tensor::random(&[size, size], bitllm_tensor::DType::F32);
    let qt = absmax_quantize(&t, &QuantConfig::int8());
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = absmax_dequantize(&qt);
    }
    start.elapsed().as_secs_f64() / iterations as f64
}

fn bench_quantized_matmul_int8(size: usize, iterations: usize) -> f64 {
    let a = Tensor::random(&[size, size], bitllm_tensor::DType::F32);
    let b = Tensor::random(&[size, size], bitllm_tensor::DType::F32);
    let b_q = absmax_quantize(&b, &QuantConfig::int8());
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = bitllm_quantization::qmatmul::quantized_matmul(&a, &b_q).unwrap();
    }
    start.elapsed().as_secs_f64() / iterations as f64
}

fn run_benchmarks() {
    println!("=== BitLLM Benchmarks ===\n");

    println!("--- Tensor Operations ---");
    for &size in &[64, 128, 256, 512] {
        let iters = if size <= 256 { 100 } else { 10 };
        let avg = bench_tensor_creation(size, iters);
        println!("  Tensor create {:4}x{:<4}: {:8.4}ms", size, size, avg * 1000.0);
    }

    println!("\n--- Matrix Multiplication ---");
    for &size in &[64, 128, 256, 512] {
        let iters = if size <= 128 { 20 } else { 5 };
        let avg = bench_matmul(size, iters);
        println!("  MatMul {:4}x{:<4}:  {:8.4}ms", size, size, avg * 1000.0);
    }

    println!("\n--- Quantization ---");
    for &size in &[128, 256, 512, 1024] {
        let iters = if size <= 256 { 50 } else { 10 };
        let avg8 = bench_quantize_int8(size, iters);
        let avg4 = bench_quantize_int4(size, iters);
        println!("  INT8 quantize {:4}x{:<4}: {:8.4}ms", size, size, avg8 * 1000.0);
        println!("  INT4 quantize {:4}x{:<4}: {:8.4}ms", size, size, avg4 * 1000.0);
    }

    println!("\n--- Dequantization ---");
    for &size in &[128, 256, 512, 1024] {
        let iters = if size <= 256 { 50 } else { 10 };
        let avg = bench_dequantize_int8(size, iters);
        println!("  INT8 dequant  {:4}x{:<4}: {:8.4}ms", size, size, avg * 1000.0);
    }

    println!("\n--- Quantized MatMul (INT8) ---");
    for &size in &[64, 128, 256, 512] {
        let iters = if size <= 128 { 20 } else { 5 };
        let avg = bench_quantized_matmul_int8(size, iters);
        println!("  QMatMul INT8  {:4}x{:<4}: {:8.4}ms", size, size, avg * 1000.0);
    }

    println!("\n--- Memory Comparison ---");
    for &size in &[128, 256, 512, 1024] {
        let n = size * size;
        let fp32_bytes = n * 4;
        let int8_bytes = n;
        let int4_bytes = n / 2;
        let bit1_bytes = n / 8;
        println!("  {:4}x{:<4} matrix:", size, size);
        println!("    FP32: {:8} bytes ({:.1} KB)", fp32_bytes, fp32_bytes as f64 / 1024.0);
        println!("    INT8: {:8} bytes ({:.1} KB) [{}x]", int8_bytes, int8_bytes as f64 / 1024.0, fp32_bytes / int8_bytes);
        println!("    INT4: {:8} bytes ({:.1} KB) [{}x]", int4_bytes, int4_bytes as f64 / 1024.0, fp32_bytes / int4_bytes);
        println!("    BIT1: {:8} bytes ({:.1} KB) [{}x]", bit1_bytes, bit1_bytes as f64 / 1024.0, fp32_bytes / bit1_bytes);
    }
}

fn main() {
    run_benchmarks();
}
