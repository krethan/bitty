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

criterion_group!(
    benches,
    bench_f32_add,
    bench_f32_dot,
    bench_f32_matmul,
    bench_f32_exp,
    bench_f32_sum,
    bench_f32_max,
    bench_i8_dot,
);
criterion_main!(benches);
