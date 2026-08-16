use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use bitllm_cognition::{bundle, HyperVector, SparseAssociativeMemory};

const SEED: u64 = 42;
const MAX_ITEMS: usize = 8192;
const PROBES: usize = 400;
const WINDOW: usize = 8;

struct Row {
    n: usize,
    m: usize,
    items: usize,
    bytes_per_item: f64,
    recall_clean: f32,
    recall_noise: f32,
    recall_window: f32,
    margin_clean: f32,
}

/// Build `m` records, each the bundle of `n` disjoint random keys, and measure
/// how well a probe (an exact key, a 10%-noisy key, or a window-bundle of keys
/// from the same record) recovers the record that contains it.
fn measure(dims: usize, n: usize, max_items: usize, probes: usize, seed: u64) -> Row {
    let config_seed = seed
        ^ (dims as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (n as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let mut rng = StdRng::seed_from_u64(config_seed);

    let m = (max_items / n).max(1);
    let mut mem: SparseAssociativeMemory<usize> = SparseAssociativeMemory::with_capacity(m);
    let mut keysets: Vec<Vec<HyperVector>> = Vec::with_capacity(m);
    for rec in 0..m {
        let keys: Vec<HyperVector> = (0..n)
            .map(|_| HyperVector::random_with(dims, &mut rng))
            .collect();
        let refs: Vec<&HyperVector> = keys.iter().collect();
        mem.store(bundle(&refs), rec);
        keysets.push(keys);
    }

    let total = m * n;
    let probe_count = total.min(probes);
    let mut samples: Vec<usize> = (0..total).collect();
    for i in 0..probe_count {
        let j = rng.gen_range(i..total);
        samples.swap(i, j);
    }
    samples.truncate(probe_count);

    let mut hits = [0usize; 3];
    let mut margin_sum = 0.0f32;
    for &item in &samples {
        let rec = item / n;
        let key = &keysets[rec][item % n];

        let (hit, sim_correct, best_wrong) = evaluate(&mem, key, rec);
        hits[0] += hit as usize;
        margin_sum += sim_correct - best_wrong;

        let noisy = noisy_probe(key, 0.10, &mut rng);
        hits[1] += evaluate(&mem, &noisy, rec).0 as usize;

        let window = window_probe(&keysets[rec], WINDOW, &mut rng);
        hits[2] += evaluate(&mem, &window, rec).0 as usize;
    }

    let pc = probe_count as f32;
    Row {
        n,
        m,
        items: total,
        bytes_per_item: (dims as f64 / 8.0) / n as f64,
        recall_clean: hits[0] as f32 / pc,
        recall_noise: hits[1] as f32 / pc,
        recall_window: hits[2] as f32 / pc,
        margin_clean: margin_sum / pc,
    }
}

/// Recall the record containing `correct` and return whether the correct
/// record won, plus (similarity to correct, best similarity to any other).
fn evaluate(
    mem: &SparseAssociativeMemory<usize>,
    query: &HyperVector,
    correct: usize,
) -> (bool, f32, f32) {
    let sim_correct = query.similarity(mem.get_key(correct).unwrap());
    let mut best = f32::NEG_INFINITY;
    let mut best_idx = usize::MAX;
    let mut second = f32::NEG_INFINITY;
    for (i, k) in mem.keys().iter().enumerate() {
        let s = query.similarity(k);
        if s > best {
            second = best;
            best = s;
            best_idx = i;
        } else if s > second {
            second = s;
        }
    }
    let best_wrong = if best_idx == correct { second } else { best };
    (best_idx == correct, sim_correct, best_wrong)
}

fn noisy_probe(key: &HyperVector, flip_frac: f64, rng: &mut StdRng) -> HyperVector {
    let mut probe = key.clone();
    for i in 0..key.dims() {
        if rng.gen_bool(flip_frac) {
            probe.flip_bit(i);
        }
    }
    probe
}

/// Clean bundle of `win` keys sampled without replacement from a record.
fn window_probe(keys: &[HyperVector], win: usize, rng: &mut StdRng) -> HyperVector {
    let n = keys.len();
    let win = win.min(n);
    let mut idx: Vec<usize> = (0..n).collect();
    for i in 0..win {
        let j = rng.gen_range(i..n);
        idx.swap(i, j);
    }
    let refs: Vec<&HyperVector> = idx[..win].iter().map(|&i| &keys[i]).collect();
    bundle(&refs)
}

fn print_table(rows: &[Row]) {
    println!(
        "{:>4} {:>6} {:>8} {:>9} | {:>9} {:>9} {:>9} {:>9}",
        "n", "m", "items", "B/item", "recall@1", "noise10%", "win8", "margin"
    );
    for r in rows {
        println!(
            "{:>4} {:>6} {:>8} {:>9.1} | {:>9.3} {:>9.3} {:>9.3} {:>9.4}",
            r.n,
            r.m,
            r.items,
            r.bytes_per_item,
            r.recall_clean,
            r.recall_noise,
            r.recall_window,
            r.margin_clean
        );
    }
}

fn main() {
    const DIMS: usize = 1024;
    println!("=== Phase 3 spike: hypervector superposition capacity ===");
    println!(
        "dims={DIMS} ({} bytes/key)  max items/row={MAX_ITEMS}  probes/row={PROBES}",
        DIMS / 8
    );
    println!("query = one of the bundled keys | noise10% = same key, 10% bits flipped | win8 = clean bundle of {WINDOW} keys from the record");
    println!();

    let ns: [usize; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];
    let rows: Vec<Row> = ns
        .iter()
        .map(|&n| measure(DIMS, n, MAX_ITEMS, PROBES, SEED))
        .collect();
    print_table(&rows);

    let baseline = rows[0].recall_clean;
    let n_cap = rows
        .iter()
        .filter(|r| r.recall_clean >= 0.95)
        .map(|r| r.n)
        .max()
        .unwrap_or(0);
    let n_cut = rows
        .iter()
        .filter(|r| r.recall_clean >= baseline - 0.02)
        .map(|r| r.n)
        .max()
        .unwrap_or(0);

    println!();
    println!("=== Findings (dims={DIMS}, {MAX_ITEMS} items) ===");
    println!(
        "n* (recall cap, recall@1 >= 0.95):          n = {n_cap}  -> {:>4} bytes/item  (vs {:.1} bytes/item for N=1 dense)",
        DIMS as f64 / 8.0 / n_cap as f64,
        DIMS as f64 / 8.0
    );
    println!(
        "n (dense-window cutover, within 2pts of N=1 recall): n = {n_cut}  -> {:>4} bytes/item",
        DIMS as f64 / 8.0 / n_cut as f64
    );

    println!();
    println!("=== Capacity vs dims (recall cap n* where recall@1 >= 0.95, max items/row=4096, 200 probes) ===");
    println!("{:>8} {:>10}", "dims", "n*");
    for &dims in &[256usize, 512, 1024, 2048] {
        let mut best = 0usize;
        for &n in &[1usize, 2, 4, 8, 16, 32, 64, 128] {
            let row = measure(dims, n, 4096, 200, SEED);
            if row.recall_clean >= 0.95 {
                best = n;
            }
        }
        println!("{dims:>8} {best:>10}");
    }
}
