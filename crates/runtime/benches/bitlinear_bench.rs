use bitllm_quantization::scheme::QuantConfig;
use bitllm_runtime::BitLinear;
use bitllm_tensor::Tensor;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_bitlinear(c: &mut Criterion) {
    // Tiny model: 64 input, 256 output
    let input_size = 64usize;
    let output_size = 256usize;

    // Full-precision weight matrix (output_size x input_size)
    let w_fp32 = Tensor::from_slice(
        &vec![0.01f32; output_size * input_size],
        &[output_size, input_size],
    );

    // 1-bit quantized layer
    let bl = BitLinear::quantize(&w_fp32, &QuantConfig::ternary());

    // Dummy input (batch 1, input_size)
    let input = Tensor::from_slice(&vec![0.5f32; input_size], &[1, input_size]);

    // Baseline FP32 matmul (using existing quantized_matmul for consistency)
    let w_fp32_q = bitllm_quantization::absmax_quantize(&w_fp32, &QuantConfig::int8());
    c.bench_function("fp32_matmul", |b| {
        b.iter(|| {
            let _ = black_box(bitllm_quantization::quantized_matmul(
                black_box(&input),
                black_box(&w_fp32_q),
            ));
        })
    });

    // 1-bit fused forward (no dequant)
    c.bench_function("bit1_fused_forward", |b| {
        b.iter(|| {
            let _ = black_box(bl.forward(black_box(&input)));
        })
    });
}

criterion_group!(benches, bench_bitlinear);
criterion_main!(benches);
