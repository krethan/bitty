use bitllm_cognition::{BitHNSW, HyperVector, SparseAssociativeMemory};
use bitllm_tensor::simd::f32_dot;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

const DIMS: usize = 1024; // 128 bytes per HyperVector key

/// Generate M keys clustered around a shared base vector so queries are
/// realistic: HNSW graphs need locality to navigate.
fn clustered_keys(n: usize, base: &HyperVector) -> Vec<HyperVector> {
    (0..n)
        .map(|i| {
            let mut k = base.clone();
            // Perturb a fixed 40-bit window — keeps neighbors nearby while
            // still producing distinct keys.
            for b in 0..40 {
                if ((i.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> b) & 1) == 1 {
                    k.flip_bit(b);
                }
            }
            k
        })
        .collect()
}

fn build_hnsw(n: usize) -> BitHNSW<usize> {
    let base = HyperVector::random(DIMS);
    let keys = clustered_keys(n, &base);
    let mut h = BitHNSW::with_params(BitHNSW::new(DIMS), 16, 16, 32);
    for (i, k) in keys.into_iter().enumerate() {
        h.insert(k, i);
    }
    h.prune();
    h
}

fn build_linear(n: usize) -> SparseAssociativeMemory<usize> {
    let base = HyperVector::random(DIMS);
    let keys = clustered_keys(n, &base);
    keys.into_iter().enumerate().map(|(i, k)| (k, i)).collect()
}

/// The dense KV-cache analog: `n` cached positions of `head_dim` f32, queried
/// by an attention-style dot-product scan of every cached key.
fn build_kv_cache(n: usize, head_dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    use rand::Rng;
    let mut k = Vec::with_capacity(n * head_dim);
    for _ in 0..n * head_dim {
        k.push(rng.gen_range(-1.0f32..1.0));
    }
    k
}

fn kv_scan_score(q: &[f32], k_cache: &[f32], head_dim: usize) -> f32 {
    let scale = (head_dim as f32).sqrt();
    let mut best = f32::NEG_INFINITY;
    for pos in 0..(k_cache.len() / head_dim) {
        let k_row = &k_cache[pos * head_dim..][..head_dim];
        let score = f32_dot(q, k_row) / scale;
        if score > best {
            best = score;
        }
    }
    best
}

fn bench_retrieval(c: &mut Criterion) {
    let mut group = c.benchmark_group("associative_memory");
    group.sample_size(50);

    for n in [256usize, 1024, 4096, 16384] {
        // NOTE: queries come from a *second, independent* cluster so search
        // must actually navigate across the graph. Querying key(0) (the entry
        // point) with one bit flipped measures the trivial case — greedy
        // descent never leaves the entry's basin, which is why earlier runs
        // looked ~330x faster than linear scan. See the KNOWN LIMITATION note
        // on `BitHNSW`; navigation quality must be validated separately.
        // --- BitHNSW ---
        let mut h = build_hnsw(n);
        let far_base = HyperVector::random(DIMS);
        let far_keys = clustered_keys(256, &far_base);
        let q = far_keys[0].clone();
        group.bench_function(format!("hnsw_search_far_n={n}"), |b| {
            b.iter_batched(|| q.clone(), |q| h.search(&q, 1), BatchSize::SmallInput)
        });

        // --- Linear scan (SparseAssociativeMemory) ---
        let mem = build_linear(n);
        let q = far_keys[1].clone();
        group.bench_function(format!("linear_recall_n={n}"), |b| {
            b.iter_batched(|| q.clone(), |q| mem.recall(&q), BatchSize::SmallInput)
        });

        // --- Dense KV-cache attention scan (head_dim=64 f32) ---
        let head_dim = 64;
        let cache = build_kv_cache(n, head_dim);
        let mut q_row = vec![0.0f32; head_dim];
        q_row[0] = 1.0;
        q_row[1] = 0.5;
        group.bench_function(format!("attention_kv_scan_n={n}"), |b| {
            b.iter(|| kv_scan_score(&q_row, &cache, head_dim))
        });
    }

    group.finish();
}

fn bench_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("associative_memory_insert");
    group.sample_size(50);

    for n in [256usize, 1024, 4096] {
        // BitHNSW insert: greedy search + link
        let base = HyperVector::random(DIMS);
        let keys = clustered_keys(n, &base);
        group.bench_function(format!("hnsw_insert_n={n}"), |b| {
            b.iter_batched(
                || keys.clone(),
                |keys| {
                    let mut h = BitHNSW::with_params(BitHNSW::new(DIMS), 16, 16, 32);
                    for (i, k) in keys.into_iter().enumerate() {
                        h.insert(k, i);
                    }
                    h
                },
                BatchSize::SmallInput,
            )
        });

        // Linear store: plain push (lower bound for the HNSW overhead)
        let keys2 = clustered_keys(n, &base);
        group.bench_function(format!("linear_store_n={n}"), |b| {
            b.iter_batched(
                || keys2.clone(),
                |keys| {
                    let mut mem: SparseAssociativeMemory<usize> =
                        SparseAssociativeMemory::with_capacity(n);
                    for (i, k) in keys.into_iter().enumerate() {
                        mem.store(k, i);
                    }
                    mem
                },
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

criterion_group!(benches, bench_retrieval, bench_insertion);
criterion_main!(benches);
