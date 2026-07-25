use bitllm_quantization::scheme::QuantConfig;
use bitllm_runtime::BitLinear;
use bitllm_tensor::{simd, DType, Tensor};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_bitlinear(c: &mut Criterion) {
    let input_size = 64usize;
    let output_size = 256usize;

    let w_fp32 = Tensor::from_slice(
        &vec![0.01f32; output_size * input_size],
        &[output_size, input_size],
    );

    let input = Tensor::from_slice(&vec![0.5f32; input_size], &[1, input_size]);

    let w_int8 = bitllm_quantization::absmax_quantize(&w_fp32, &QuantConfig::int8());
    let bl_ternary = BitLinear::quantize(&w_fp32, &QuantConfig::ternary());

    let w_t = w_fp32.transpose();

    c.bench_function("fp32_matmul_256x64", |b| {
        b.iter(|| {
            let _ = black_box(input.dot(black_box(&w_t)));
        })
    });

    c.bench_function("fp32_matmul_raw_256x64", |b| {
        let i_s = input.as_f32_slice();
        let w_s = w_t.as_f32_slice();
        let mut out = vec![0.0f32; output_size];
        b.iter(|| {
            simd::f32_matmul_row(
                black_box(i_s),
                black_box(w_s),
                &mut out,
                input_size,
                output_size,
            );
        })
    });

    c.bench_function("int8_matmul_256x64", |b| {
        b.iter(|| {
            let _ = black_box(bitllm_quantization::quantized_matmul(
                black_box(&input),
                black_box(&w_int8),
            ));
        })
    });

    c.bench_function("ternary_fused_forward_256x64", |b| {
        b.iter(|| {
            let _ = black_box(bl_ternary.forward(black_box(&input)));
        })
    });
}

fn bench_xnor_kernel(c: &mut Criterion) {
    for &size in &[64, 256, 1024] {
        let n_packed = (size + 3) / 4;
        let a: Vec<u8> = (0..n_packed).map(|i| (i * 7 + 3) as u8).collect();
        let b: Vec<u8> = (0..n_packed).map(|i| (i * 11 + 5) as u8).collect();
        let mut out = vec![0u8; n_packed];

        c.bench_function(&format!("xnor_popcount_2bit_{size}"), |bench| {
            bench.iter(|| {
                simd::xnor_popcount_2bit(black_box(&a), black_box(&b), &mut out, size);
            })
        });
    }
}

fn bench_memory_footprint(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![128, 256, 512, 1024];

    for &size in &sizes {
        let w_fp32 = Tensor::random(&[size, size], DType::F32);
        let w_int8 = bitllm_quantization::absmax_quantize(&w_fp32, &QuantConfig::int8());
        let bl = BitLinear::quantize(&w_fp32, &QuantConfig::ternary());

        let fp32_bytes = size * size * 4;
        let int8_bytes = size * size;
        let ternary_bytes = (size * size + 3) / 4;
        let input = Tensor::from_slice(&vec![0.5f32; size], &[1, size]);
        let w_t = w_fp32.transpose();

        println!(
            "\n  {:4}x{:<4} weight memory: FP32={:.1}KB INT8={:.1}KB Ternary={:.1}KB",
            size,
            size,
            fp32_bytes as f64 / 1024.0,
            int8_bytes as f64 / 1024.0,
            ternary_bytes as f64 / 1024.0,
        );

        c.bench_function(&format!("fp32_matmul_{size}x{size}"), |b| {
            b.iter(|| {
                let _ = black_box(input.dot(black_box(&w_t)));
            })
        });

        c.bench_function(&format!("int8_matmul_{size}x{size}"), |b| {
            b.iter(|| {
                let _ = black_box(bitllm_quantization::quantized_matmul(
                    black_box(&input),
                    black_box(&w_int8),
                ));
            })
        });

        c.bench_function(&format!("ternary_fused_{size}x{size}"), |b| {
            b.iter(|| {
                let _ = black_box(bl.forward(black_box(&input)));
            })
        });
    }
}

criterion_group!(
    benches,
    bench_bitlinear,
    bench_xnor_kernel,
    bench_memory_footprint
);
criterion_main!(benches);
