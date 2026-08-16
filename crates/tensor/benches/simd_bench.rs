use bitllm_tensor::pnword::{PNActivation256, PNActivation512, PNWeight256};
use bitllm_tensor::simd;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_f32_add(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![64, 256, 1024, 4096];
    for &n in &sizes {
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let mut out = vec![0.0f32; n];
        c.bench_function(&format!("f32_add_n={}", n), |bench| {
            bench.iter(|| {
                simd::f32_add(black_box(&a), black_box(&b), black_box(&mut out));
            });
        });
    }
}

fn bench_f32_dot(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![64, 256, 1024, 4096];
    for &n in &sizes {
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        c.bench_function(&format!("f32_dot_n={}", n), |bench| {
            bench.iter(|| {
                black_box(simd::f32_dot(black_box(&a), black_box(&b)));
            });
        });
    }
}

fn bench_f32_matmul(c: &mut Criterion) {
    let dims: Vec<(usize, usize, usize)> = vec![(32, 32, 32), (64, 64, 64), (128, 128, 128)];
    for &(m, k, n) in &dims {
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32).collect();
        let mut b_t = vec![0.0f32; k * n];
        for i in 0..k {
            for j in 0..n {
                b_t[j * k + i] = b[i * n + j];
            }
        }
        let mut out = vec![0.0f32; m * n];
        c.bench_function(&format!("f32_matmul_{}x{}x{}", m, k, n), |bench| {
            bench.iter(|| {
                simd::f32_matmul(black_box(&a), black_box(&b_t), black_box(&mut out), m, k, n);
            });
        });
    }
}

fn bench_f32_exp(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![64, 256, 1024, 4096];
    for &n in &sizes {
        let a: Vec<f32> = (0..n)
            .map(|i| (i as f32 - n as f32 / 2.0) / 100.0)
            .collect();
        let mut out = vec![0.0f32; n];
        c.bench_function(&format!("f32_exp_n={}", n), |bench| {
            bench.iter(|| {
                simd::f32_exp(black_box(&a), black_box(&mut out));
            });
        });
    }
}

fn bench_f32_sum(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![64, 256, 1024, 4096];
    for &n in &sizes {
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        c.bench_function(&format!("f32_sum_n={}", n), |bench| {
            bench.iter(|| {
                black_box(simd::f32_sum(black_box(&a)));
            });
        });
    }
}

fn bench_f32_max(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![64, 256, 1024, 4096];
    for &n in &sizes {
        let a: Vec<f32> = (0..n)
            .map(|i| (i as f32 - n as f32 / 2.0) / 100.0)
            .collect();
        c.bench_function(&format!("f32_max_n={}", n), |bench| {
            bench.iter(|| {
                black_box(simd::f32_max(black_box(&a)));
            });
        });
    }
}

fn bench_i8_dot(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![64, 256, 1024, 4096];
    for &n in &sizes {
        let a: Vec<u8> = (0..n).map(|i| (i as i8 * 3) as u8).collect();
        let b: Vec<u8> = (0..n).map(|i| (i as i8 * 2) as u8).collect();
        c.bench_function(&format!("i8_dot_n={}", n), |bench| {
            bench.iter(|| {
                black_box(simd::i8_dot_product(black_box(&a), black_box(&b), n));
            });
        });
    }
}

fn bench_pnword256_pack(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![16, 64, 128];
    for &n in &sizes {
        let dense: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) / 100.0).collect();
        let ternary: Vec<i8> = dense.iter().map(|&v| if v > 0.0 { 1 } else if v < 0.0 { -1 } else { 0 }).collect();
        c.bench_function(&format!("pnword256_pack_n={}", n), |bench| {
            bench.iter(|| {
                let _ = black_box(PNActivation256::pack(black_box(&ternary)));
            });
        });
    }
}

fn bench_pnword256_dot(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![16, 64, 128];
    for &n in &sizes {
        let a_vals: Vec<i8> = (0..n).map(|i| if i % 3 == 0 { 1 } else if i % 3 == 1 { -1 } else { 0 }).collect();
        let w_vals: Vec<i8> = (0..n).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let a = PNActivation256::pack(&a_vals);
        let w = PNWeight256::pack(&w_vals, 1.0);
        c.bench_function(&format!("pnword256_dot_n={}", n), |bench| {
            bench.iter(|| {
                black_box(a.dot(black_box(&w)));
            });
        });
    }
}

fn bench_pnword256_xor(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![16, 64, 128];
    for &n in &sizes {
        let a_vals: Vec<i8> = (0..n).map(|i| if i % 3 == 0 { 1 } else if i % 3 == 1 { -1 } else { 0 }).collect();
        let b_vals: Vec<i8> = (0..n).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let a = PNActivation256::pack(&a_vals);
        let b = PNActivation256::pack(&b_vals);
        c.bench_function(&format!("pnword256_xor_n={}", n), |bench| {
            bench.iter(|| {
                black_box(a.xor(black_box(&b)));
            });
        });
    }
}

fn bench_pnword256_and(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![16, 64, 128];
    for &n in &sizes {
        let a_vals: Vec<i8> = (0..n).map(|i| if i % 3 == 0 { 1 } else if i % 3 == 1 { -1 } else { 0 }).collect();
        let b_vals: Vec<i8> = (0..n).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let a = PNActivation256::pack(&a_vals);
        let b = PNActivation256::pack(&b_vals);
        c.bench_function(&format!("pnword256_and_n={}", n), |bench| {
            bench.iter(|| {
                black_box(a.and(black_box(&b)));
            });
        });
    }
}

fn bench_pnword256_popcount(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![16, 64, 128];
    for &n in &sizes {
        let a_vals: Vec<i8> = (0..n).map(|i| if i % 3 == 0 { 1 } else if i % 3 == 1 { -1 } else { 0 }).collect();
        let a = PNActivation256::pack(&a_vals);
        c.bench_function(&format!("pnword256_popcount_n={}", n), |bench| {
            bench.iter(|| {
                black_box(a.popcount());
            });
        });
    }
}

fn bench_pnword512_memory_bandwidth(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![256, 1024, 4096];
    for &n in &sizes {
        let a_vals: Vec<i8> = (0..n).map(|i| if i % 3 == 0 { 1 } else if i % 3 == 1 { -1 } else { 0 }).collect();
        let _b_vals: Vec<i8> = (0..n).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();

        let a = PNActivation512::pack(&a_vals);

        c.bench_function(&format!("pnword512_memory_bandwidth_n={}", n), |bench| {
            bench.iter(|| {
                let result = a.xor(black_box(&a));
                black_box(result.popcount());
            });
        });

        c.bench_function(&format!("dense_f32_memory_read_n={}", n), |bench| {
            let dense_a: Vec<f32> = a_vals.iter().map(|&v| v as f32).collect();
            bench.iter(|| {
                let mut sum = 0.0f32;
                for &v in dense_a.iter() {
                    sum += black_box(v);
                }
                black_box(sum);
            });
        });
    }
}

criterion_group!(
    benches,
    bench_f32_add,
    bench_f32_dot,
    bench_f32_matmul,
    bench_f32_exp,
    bench_f32_sum,
    bench_f32_max,
    bench_i8_dot,
    bench_pnword256_pack,
    bench_pnword256_dot,
    bench_pnword256_xor,
    bench_pnword256_and,
    bench_pnword256_popcount,
    bench_pnword512_memory_bandwidth,
);
criterion_main!(benches);
