use criterion::{black_box, criterion_group, criterion_main, Criterion};
use bitllm_runtime::{BitLinear, ModelConfig};
use bitllm_quantization::scheme::QuantConfig;
use bitllm_tensor::{Tensor, DType};

fn bench_bitlinear(c: &mut Criterion) {
    // Tiny model: 256 vocab, 64 hidden, 2 layers equivalent
    let hidden = 64;
    let vocab = 256;

    // Full-precision weight
    let w_fp32 = Tensor::from_slice(
        &vec![0.01f32; vocab * hidden],
        &[vocab, hidden],
    );

    // 1-bit quantized
    let bl = BitLinear::quantize(&w_fp32, &QuantConfig::ternary());

    // Input tensor
    let input = Tensor::from_slice(&vec![0.5f32; hidden], &[1, hidden]);

    // Benchmark FP32 matmul baseline
    let w_fp32_copy = w_fp32.clone();
    c.bench_function("fp32_matmul", |b| {
        b.iter(|| {
            let _ = black_box(bitllm_quantization::qmatmul::quantized_matmul(
                black_box(&input),
                black_box(&input),
                black_box(&bitllm_quantization::absmax_quantize(&w_fp32_copy, &QuantConfig::int8())), 
            ));
        })
    });

    // Benchmark 1-bit fused forward
    c.bench_function("bit1_fused_forward", |b| {
        b.iter(|| {
            let _ = black_box(bl.forward(black_box(&input)));
        })
    });
}

fn absmax_quantize(t: &Tensor, c: &QuantConfig) -> bitllm_quantization::QuantizedTensor {
    bitllm_quantization::absmax_quantize(t, c)
}

criterion_group!(benches, bench_bitlinear);
criterion_main!(benches);