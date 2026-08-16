use bitllm_quantization::qmatmul::fused_bit1_matmul;
use bitllm_quantization::scheme::QuantConfig;
use bitllm_quantization::ternary::ternary_quantize;
use bitllm_runtime::BitLinear;
use bitllm_tensor::simd;
use bitllm_tensor::{DType, Tensor};

#[cfg(feature = "rocm")]
use bitllm_rocm::{GpuBuffer, GpuOps};

use crate::helpers::{
    auto_iters, print_compute_raw, print_throughput_raw, time_iters, BenchmarkResult,
};

fn bench_f32_matmul(size: usize, iterations: usize) -> BenchmarkResult {
    let a = Tensor::random(&[size, size], DType::F32);
    let b = Tensor::random(&[size, size], DType::F32);
    time_iters(iterations, || {
        let _ = a.dot(&b).unwrap();
    })
}

fn bench_f32_matmul_raw(size: usize, iterations: usize) -> BenchmarkResult {
    let a = Tensor::random(&[size, size], DType::F32);
    let b = Tensor::random(&[size, size], DType::F32);
    let b_t = b.transpose();
    let a_slice = a.as_f32_slice();
    let b_slice = b_t.as_f32_slice();
    let mut out = vec![0.0f32; size * size];
    time_iters(iterations, || {
        simd::f32_matmul(a_slice, b_slice, &mut out, size, size, size);
    })
}

fn bench_ternary_matmul(size: usize, iterations: usize) -> BenchmarkResult {
    let a = Tensor::random(&[size, size], DType::F32);
    let b = Tensor::random(&[size, size], DType::F32);
    let b_q = ternary_quantize(&b);
    let mut out = vec![0.0f32; size * size];
    let input_slice = a.as_f32_slice();
    time_iters(iterations, || {
        fused_bit1_matmul(input_slice, &b_q, &mut out, size, size, size);
    })
}

fn bench_bitlinear_forward(size: usize, iterations: usize) -> BenchmarkResult {
    let w = Tensor::random(&[size, size], DType::F32);
    let bl = BitLinear::quantize(&w, &QuantConfig::ternary());
    let input = Tensor::from_slice(&vec![0.5f32; size], &[1, size]);
    time_iters(iterations, || {
        let _ = bl.forward(&input);
    })
}

#[cfg(feature = "rocm")]
fn bench_gpu_bit1_matmul(size: usize, iterations: usize) -> BenchmarkResult {
    let a = Tensor::random(&[size, size], DType::F32);
    let b = Tensor::random(&[size, size], DType::F32);
    let b_q = ternary_quantize(&b);
    let a_slice = a.as_f32_slice();
    let w_slice = b_q.data.as_slice();

    let a_bytes = unsafe {
        std::slice::from_raw_parts(
            a_slice.as_ptr() as *const u8,
            a_slice.len() * std::mem::size_of::<f32>(),
        )
    };
    let w_bytes = unsafe {
        std::slice::from_raw_parts(
            w_slice.as_ptr() as *const u8,
            w_slice.len() * std::mem::size_of::<u8>(),
        )
    };
    let scales_bytes = unsafe {
        std::slice::from_raw_parts(
            b_q.scales.as_ptr() as *const u8,
            b_q.scales.len() * std::mem::size_of::<f32>(),
        )
    };

    let a_gpu = GpuBuffer::from_host(a_bytes).expect("GPU alloc for input");
    let w_gpu = GpuBuffer::from_host(w_bytes).expect("GPU alloc for weight");
    let scales_gpu = GpuBuffer::new(b_q.scales.len() * 4).expect("GPU alloc for scales");
    scales_gpu.copy_from_host(scales_bytes).expect("H2D scales");
    let out_gpu = GpuBuffer::new(size * size * 4).expect("GPU alloc for output");

    // Warmup
    GpuOps::bit1_matmul(
        &a_gpu,
        &w_gpu,
        &scales_gpu,
        &out_gpu,
        size,
        size,
        size,
        0,
        None,
        None,
    )
    .expect("GPU matmul warmup");

    time_iters(iterations, || {
        GpuOps::bit1_matmul(
            &a_gpu,
            &w_gpu,
            &scales_gpu,
            &out_gpu,
            size,
            size,
            size,
            0,
            None,
            None,
        )
        .expect("GPU matmul");
    })
}

pub fn bench_matmul_suite() {
    println!("\n=== MatMul Benchmark (transformer-relevant sizes) ===");
    println!("  (compute-bound: lower ms = better, GFLOPS = 2*M*N*K / time)\n");

    // Full comparison at moderate sizes — auto-scale iterations
    for &size in &[512, 1024] {
        let n = size * size;
        println!("  {}x{}:", size, size);

        let (iters, _) = auto_iters(|| {
            let _ = bench_f32_matmul(size, 1);
        });

        let avg = bench_f32_matmul(size, iters).mean;
        print_compute_raw(
            &format!("FP32 Tensor::dot ({})", size),
            (size, size, size),
            avg,
        );

        let avg = bench_f32_matmul_raw(size, iters).mean;
        print_compute_raw(
            &format!("FP32 SIMD raw ({})", size),
            (size, size, size),
            avg,
        );

        let avg = bench_ternary_matmul(size, iters).mean;
        print_throughput_raw(
            &format!("Ternary fused_bit1_matmul ({})", size),
            n / 8 + 4,
            avg,
        );

        let avg = bench_bitlinear_forward(size, iters).mean;
        print_throughput_raw(
            &format!("BitLinear::forward fused ({})", size),
            n / 8 + 4,
            avg,
        );

        #[cfg(feature = "rocm")]
        {
            let avg = bench_gpu_bit1_matmul(size, iters).mean;
            print_throughput_raw(&format!("GPU bit1_matmul ({})", size), n / 8 + 4, avg);
        }

        println!();
    }

    // Transformer-scale: only practical methods
    println!("  --- Transformer-scale (fused methods only) ---\n");
    for &size in &[2048, 4096, 11008] {
        let n = size * size;
        println!("  {}x{}:", size, size);

        let (iters, _) = auto_iters(|| {
            let _ = bench_bitlinear_forward(size, 1);
        });

        if size <= 4096 {
            let avg = bench_f32_matmul(size, 1).mean;
            print_compute_raw(
                &format!("FP32 Tensor::dot ({})", size),
                (size, size, size),
                avg,
            );
        }

        let avg = bench_bitlinear_forward(size, iters).mean;
        print_throughput_raw(
            &format!("BitLinear::forward fused ({})", size),
            n / 8 + 4,
            avg,
        );

        #[cfg(feature = "rocm")]
        {
            let avg = bench_gpu_bit1_matmul(size, iters).mean;
            print_throughput_raw(&format!("GPU bit1_matmul ({})", size), n / 8 + 4, avg);
        }

        println!();
    }
}
