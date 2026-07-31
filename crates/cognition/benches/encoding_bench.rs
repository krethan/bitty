use bitllm_cognition::{encode_activation_direct, BitHNSW, RandomIndexCodebook, SparseAssociativeMemory};
use bitllm_tensor::pnword::PNActivation256;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};

const DIMS: usize = 1024; // 128 bytes per HyperVector key

fn sparse_packet(nonzero: usize, seed: u64) -> PNActivation256 {
    let mut values = [0i8; 128];
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..nonzero {
        let i = rng.gen_range(0..128);
        values[i] = if rng.gen_bool(0.5) { 1 } else { -1 };
    }
    PNActivation256::pack(&values)
}

/// Probe sharing ~75% of a packet's activations plus new noise.
fn noisy_probe(p: &PNActivation256, keep_frac: f32, seed: u64) -> PNActivation256 {
    let mut values = [0i8; 128];
    let mut rng = StdRng::seed_from_u64(seed);
    p.unpack(&mut values);
    for v in values.iter_mut() {
        if *v != 0 && rng.gen_bool(keep_frac as f64) {
            continue; // keep
        }
        *v = if rng.gen_bool(0.5) { 1 } else { -1 };
    }
    PNActivation256::pack(&values)
}

fn bench_encode_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("packet_encode");
    group.sample_size(50);

    let cb = RandomIndexCodebook::new(DIMS);
    let p8 = sparse_packet(8, 1);
    let p32 = sparse_packet(32, 2);
    let p64 = sparse_packet(64, 3);

    group.bench_function("direct_trit2bit", |b| b.iter(|| encode_activation_direct(&p8)));

    for (name, p) in [("random_index_nnz=8", &p8), ("random_index_nnz=32", &p32), ("random_index_nnz=64", &p64)] {
        group.bench_function(name, |b| b.iter(|| cb.encode(p)));
    }

    group.finish();
}

/// End-to-end packet retrieval: keys are codebook-encoded activation packets;
/// probes are noisy versions of a stored packet. Compares HNSW navigation
/// against linear recall of the same keys.
fn bench_retrieval(c: &mut Criterion) {
    let mut group = c.benchmark_group("packet_retrieval");
    group.sample_size(50);

    for n in [1024usize, 16384] {
        let cb = RandomIndexCodebook::new(DIMS);
        let packets: Vec<PNActivation256> = (0..n).map(|i| sparse_packet(16, i as u64 + 1)).collect();
        let probes: Vec<PNActivation256> = packets.iter().map(|p| noisy_probe(p, 0.75, 0xDEAD_BEEF)).collect();

        let mut h = BitHNSW::with_params(BitHNSW::new(DIMS), 16, 16, 32);
        for (i, p) in packets.iter().enumerate() {
            h.insert(cb.encode(p), i);
        }
        h.prune();
        let mut keys = Vec::with_capacity(n);
        for p in &probes {
            keys.push(cb.encode(p));
        }

        let mut probe_idx = 0usize;
        group.bench_function(format!("hnsw_n={n}"), |b| {
            b.iter_batched(
                || {
                    let q = &keys[probe_idx % n];
                    probe_idx += 1;
                    q.clone()
                },
                |q| h.search(&q, 1),
                BatchSize::SmallInput,
            )
        });

        let mut mem: SparseAssociativeMemory<usize> = SparseAssociativeMemory::with_capacity(n);
        for (i, p) in packets.iter().enumerate() {
            mem.store(cb.encode(p), i);
        }
        let mut probe_idx = 0usize;
        group.bench_function(format!("linear_n={n}"), |b| {
            b.iter_batched(
                || {
                    let q = &keys[probe_idx % n];
                    probe_idx += 1;
                    q.clone()
                },
                |q| mem.recall(&q),
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

criterion_group!(benches, bench_encode_throughput, bench_retrieval);
criterion_main!(benches);
