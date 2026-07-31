use bitllm_quantization::fused_bit1_matmul;
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

    let bl_ternary = BitLinear::quantize(&w_fp32, &QuantConfig::ternary());
    let w_ternary = bitllm_quantization::ternary_quantize(&w_fp32);

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

    c.bench_function("bit1_fused_matmul_256x64", |b| {
        let i_s = input.as_f32_slice();
        let mut out = vec![0.0f32; output_size];
        b.iter(|| {
            fused_bit1_matmul(
                black_box(i_s),
                black_box(&w_ternary),
                &mut out,
                1,
                input_size,
                output_size,
            );
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
        let w_ternary = bitllm_quantization::ternary_quantize(&w_fp32);
        let bl = BitLinear::quantize(&w_fp32, &QuantConfig::ternary());

        let fp32_bytes = size * size * 4;
        let ternary_bytes = (size * size + 3) / 4;
        let input = Tensor::from_slice(&vec![0.5f32; size], &[1, size]);
        let w_t = w_fp32.transpose();

        println!(
            "\n  {:4}x{:<4} weight memory: FP32={:.1}KB Ternary={:.1}KB",
            size,
            size,
            fp32_bytes as f64 / 1024.0,
            ternary_bytes as f64 / 1024.0,
        );

        c.bench_function(&format!("fp32_matmul_{size}x{size}"), |b| {
            b.iter(|| {
                let _ = black_box(input.dot(black_box(&w_t)));
            })
        });

        c.bench_function(&format!("bit1_fused_matmul_{size}x{size}"), |b| {
            let i_s = input.as_f32_slice();
            let mut out = vec![0.0f32; size];
            b.iter(|| {
                fused_bit1_matmul(
                    black_box(i_s),
                    black_box(&w_ternary),
                    &mut out,
                    1,
                    size,
                    size,
                );
            })
        });

        c.bench_function(&format!("ternary_fused_{size}x{size}"), |b| {
            b.iter(|| {
                let _ = black_box(bl.forward(black_box(&input)));
            })
        });
    }
}

/// Compare synchronous (layer-by-layer) vs event-driven scheduler execution.
fn bench_scheduler_throughput(c: &mut Criterion) {
    use bitllm_runtime::scheduler::*;
    use bitllm_tensor::pnword::{PNActivation256, PNWeight256};

    let chain_lengths: Vec<usize> = vec![4, 8, 16];

    for &layers in &chain_lengths {
        let w_vals: Vec<i8> = (0..128).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let weight = PNWeight256::pack(&w_vals, 1.0);

        let mut a_vals = [0i8; 128];
        for i in 0..32 {
            a_vals[i] = 1;
        }
        let activation = PNActivation256::pack(&a_vals);

        // Synchronous: sequential chain of dot products
        let bench_name = format!("synchronous_chain_{}_layers", layers);
        c.bench_function(&bench_name, |b| {
            b.iter(|| {
                for _ in 0..layers {
                    black_box(activation.dot(black_box(&weight)));
                }
            })
        });

        // Event-driven: build fresh graph each iteration
        let bench_name = format!("event_scheduler_chain_{}_layers", layers);
        c.bench_function(&bench_name, |b| {
            b.iter(|| {
                let mut graph = Graph::new();
                let sink_id = graph.add_node(Box::new(SinkNode::new("sink")));
                let mut node_ids = sink_id;
                for j in 0..layers {
                    let id = graph.add_node(Box::new(MatMulNode::new(
                        &format!("matmul_{}", j),
                        weight,
                        vec![node_ids],
                    )));
                    node_ids = id;
                }
                let mut scheduler = Scheduler::new(graph);
                let packet = Packet::new(activation, node_ids);
                scheduler.enqueue(packet);
                scheduler.run(200);
                black_box(scheduler.stats().packets_processed);
            })
        });
    }
}

criterion_group!(
    benches,
    bench_bitlinear,
    bench_xnor_kernel,
    bench_memory_footprint,
    bench_scheduler_throughput,
);
criterion_main!(benches);
