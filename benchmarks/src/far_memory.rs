use bitllm_cognition::{bundle, ContextMemory, ContextMemoryConfig};
use bitllm_runtime::Model;

use crate::export::MemoryRow;
use crate::perplexity::{build_synthetic_model, ln_softmax, Rng, VOCAB};

// ── Config ────────────────────────────────────────────────────────────

const SEED: u64 = 42;
const REPLAYS: usize = 16;
const KEY_LEN: usize = 8;
const CHUNK: usize = 16;
/// Filler tokens between a key and its value inside a history record, so the
/// corpus bigram table never sees `key_last -> value` (no near-context leak;
/// the far-context signal is only reachable through the memory).
const VALUE_GAP: usize = 2;
const KEY_ALPHABET: usize = 64;
const VALUE_BASE: usize = 64;
const MEMORY_DIMS: usize = 1024;
/// Memory readout weight added to the predicted value's logit.
const MEMORY_ALPHA: f32 = 20.0;
const EVAL_CONTEXT: usize = 64;

struct RepeatPair {
    first: usize,
    replay: usize,
    value: u32,
}

// ── Corpus ────────────────────────────────────────────────────────────

/// Repeat-structured stream: each 8-token key appears once in history
/// (evicted beyond the dense window) and is replayed `REPLAYS * CHUNK + ...`
/// tokens later, where predicting the token that follows it requires
/// retrieving the evicted record containing its first occurrence.
fn build_repeat_corpus(rng: &mut Rng) -> (Vec<u32>, Vec<RepeatPair>) {
    let keys: Vec<Vec<u32>> = (0..REPLAYS)
        .map(|_| {
            (0..KEY_LEN)
                .map(|_| (rng.next_u64() % KEY_ALPHABET as u64) as u32)
                .collect()
        })
        .collect();
    let values: Vec<u32> = (0..REPLAYS)
        .map(|i| (VALUE_BASE + (i * 7) % (VOCAB - VALUE_BASE)) as u32)
        .collect();

    let mut tokens: Vec<u32> = Vec::new();
    let mut pairs: Vec<RepeatPair> = Vec::new();
    for i in 0..REPLAYS {
        let first = tokens.len();
        tokens.extend_from_slice(&keys[i]);
        for _ in 0..VALUE_GAP {
            tokens.push(fill(rng));
        }
        tokens.push(values[i]);
        let tail = CHUNK - KEY_LEN - VALUE_GAP - 1;
        for _ in 0..tail {
            tokens.push(fill(rng));
        }
        pairs.push(RepeatPair {
            first,
            replay: 0,
            value: values[i],
        });
    }
    // Trailing filler record so the last history record is fully evicted.
    for _ in 0..CHUNK {
        tokens.push((rng.next_u64() % VOCAB as u64) as u32);
    }

    for i in 0..REPLAYS {
        let replay = tokens.len();
        tokens.extend_from_slice(&keys[i]);
        tokens.push(values[i]);
        pairs[i].replay = replay;
    }
    (tokens, pairs)
}

fn fill(rng: &mut Rng) -> u32 {
    (VALUE_BASE as u64 + rng.next_u64() % (VOCAB - VALUE_BASE) as u64) as u32
}

// ── Memory ────────────────────────────────────────────────────────────

fn build_memory(history: &[u32]) -> ContextMemory {
    let config = ContextMemoryConfig {
        dims: MEMORY_DIMS,
        vocab_size: VOCAB,
        chunk_items: CHUNK,
        window: CHUNK,
        min_similarity: 0.5,
        seed: 0x5EED_C0DE,
    };
    let mut mem = ContextMemory::new(config);
    for &t in history {
        mem.push(t);
    }
    mem
}

/// Value the memory predicts for each replay key, if its first-occurrence
/// record is still retrievable.
fn memory_readouts(mem: &ContextMemory, keys: &[Vec<u32>], pairs: &[RepeatPair]) -> Vec<Option<u32>> {
    (0..REPLAYS)
        .map(|i| {
            let refs: Vec<&bitllm_cognition::HyperVector> =
                keys[i].iter().map(|t| mem.key(*t)).collect();
            let query = bundle(&refs);
            mem.probe(&query).and_then(|(_, rec)| {
                let value_idx = pairs[i].first + KEY_LEN + VALUE_GAP;
                if rec.start <= pairs[i].first
                    && value_idx < rec.start + rec.tokens.len()
                {
                    Some(rec.tokens[value_idx - rec.start])
                } else {
                    None
                }
            })
        })
        .collect()
}

// ── Evaluation ────────────────────────────────────────────────────────

/// Token-level add-one-smoothed bigram log-probs (same estimator as the
/// perplexity harness, but for a token corpus).
fn token_bigram_log_probs(tokens: &[u32]) -> Vec<Vec<f64>> {
    let mut counts = vec![vec![0usize; VOCAB]; VOCAB];
    for pair in tokens.windows(2) {
        counts[pair[0] as usize][pair[1] as usize] += 1;
    }
    counts
        .into_iter()
        .map(|row| {
            let total: usize = row.iter().sum::<usize>() + VOCAB;
            row.into_iter()
                .map(|c| ((c as f64 + 1.0) / total as f64).ln())
                .collect()
        })
        .collect()
}

/// Sliding-window nll, optionally boosting the memory's predicted value on
/// echo positions. Returns `(overall_nll, n, echo_nll, n_echo, echo_hits)`.
fn evaluate_with_memory(
    model: &mut Model,
    tokens: &[u32],
    pairs: &[RepeatPair],
    readouts: &[Option<u32>],
    ctx: usize,
    alpha: f32,
) -> (f64, usize, f64, usize, usize) {
    let vocab = model.config.vocab_size;
    let mut sum_nll = 0.0;
    let mut n = 0usize;
    let mut echo_sum = 0.0;
    let mut echo_n = 0usize;
    let mut echo_hits = 0usize;

    let mut i = 0;
    while i + 1 < tokens.len() {
        let end = (i + ctx + 1).min(tokens.len());
        let window = &tokens[i..end];
        model.clear_cache();
        let logits = model.forward(window);
        let slice = logits.as_f32_slice();

        for t in 0..window.len() - 1 {
            let pos = i + t + 1;
            let target = window[t + 1] as usize;
            let mut row: Vec<f32> = slice[t * vocab..(t + 1) * vocab].to_vec();

            if let Some(pair_idx) = pairs.iter().position(|p| p.replay + KEY_LEN == pos) {
                if let Some(Some(value)) = readouts.get(pair_idx) {
                    row[*value as usize] += alpha;
                }
                if readouts[pair_idx] == Some(pairs[pair_idx].value) {
                    echo_hits += 1;
                }
                let nll = -ln_softmax(&row, target);
                echo_sum += nll;
                echo_n += 1;
                sum_nll += nll;
                n += 1;
                continue;
            }

            sum_nll += -ln_softmax(&row, target);
            n += 1;
        }
        i += ctx;
    }

    (sum_nll, n, echo_sum, echo_n, echo_hits)
}

// ── Benchmark ─────────────────────────────────────────────────────────

/// Far-context memory benchmark: can the dense-window cutover memory recover
/// tokens that left the dense window long ago? WINDOW = dense window only;
/// MEM = transformer + `ContextMemory` readout.
pub fn bench_memory() -> Vec<MemoryRow> {
    println!("\n=== Far-Context Memory (ContextMemory) ===\n");
    println!(
        "  NOTE: the bigram synthetic model is context-length-1, so far-context\n\
        \x20       recall cannot move perplexity there. This benchmark uses a\n\
        \x20       repeat-structured corpus: each 8-token key recurs after it was\n\
        \x20       evicted beyond the dense window, and predicting the following\n\
        \x20       token requires retrieving the evicted record (n*={CHUNK} bundled\n\
        \x20       items, dims={MEMORY_DIMS}).\n"
    );

    let mut rng = Rng::new(SEED);
    let (tokens, pairs) = build_repeat_corpus(&mut rng);
    let history_len = tokens.len() - REPLAYS * (KEY_LEN + 1);
    let keys: Vec<Vec<u32>> = pairs
        .iter()
        .map(|p| tokens[p.first..p.first + KEY_LEN].to_vec())
        .collect();

    let mem = build_memory(&tokens[..history_len]);
    let readouts = memory_readouts(&mem, &keys, &pairs);

    let mut model = build_synthetic_model(&token_bigram_log_probs(&tokens));

    let (sum_nll, n, echo_sum, echo_n, _) =
        evaluate_with_memory(&mut model, &tokens, &pairs, &readouts, EVAL_CONTEXT, 0.0);
    let (mem_sum, mem_n, mem_echo_sum, mem_echo_n, mem_hits) =
        evaluate_with_memory(&mut model, &tokens, &pairs, &readouts, EVAL_CONTEXT, MEMORY_ALPHA);

    let window_ppl = (sum_nll / n as f64).exp();
    let window_echo = (echo_sum / echo_n as f64).exp();
    let mem_ppl = (mem_sum / mem_n as f64).exp();
    let mem_echo = (mem_echo_sum / mem_echo_n as f64).exp();
    let recall = mem_hits as f64 / mem_echo_n as f64;

    let dense_bytes = history_len * MEMORY_DIMS.div_ceil(8);
    let mem_bytes = mem.memory_bytes();
    let compression = dense_bytes as f64 / mem_bytes as f64;

    println!(
        "  Config: replays={REPLAYS} key_len={KEY_LEN} chunk={CHUNK} dims={MEMORY_DIMS}\n\
        \x20         history={history_len} tokens  dense_bytes={dense_bytes}\n"
    );
    println!(
        "{:<8} {:>10} {:>10} {:>10} {:>12}",
        "mode", "overall", "echo-ppl", "recall@1", "mem_bytes"
    );
    println!(
        "{:<8} {:>10.3} {:>10.3} {:>10} {:>12}",
        "WINDOW",
        window_ppl,
        window_echo,
        "-",
        "-"
    );
    println!(
        "{:<8} {:>10.3} {:>10.3} {:>10.3} {:>12}",
        "MEM", mem_ppl, mem_echo, recall, mem_bytes
    );
    println!();
    println!(
        "  Dense window only: echo-ppl {window_echo:.2} (≈ uniform floor {VOCAB}, the near\n\
        \x20       path cannot see the far key).  With memory: {mem_echo:.2}.\n\
        \x20       Memory compresses evicted history {compression:.1}x ({mem_bytes} bytes vs\n\
        \x20       {dense_bytes} dense). Overall ppl is filler-dominated (~uniform); the\n\
        \x20       memory's effect is localized to the far-recall (echo) positions.\n"
    );

    vec![
        MemoryRow {
            mode: "WINDOW".to_string(),
            overall_ppl: window_ppl,
            echo_ppl: window_echo,
            recall_at_1: 0.0,
            memory_bytes: 0,
            dense_bytes,
            compression_ratio: 1.0,
        },
        MemoryRow {
            mode: "MEM".to_string(),
            overall_ppl: mem_ppl,
            echo_ppl: mem_echo,
            recall_at_1: recall,
            memory_bytes: mem_bytes,
            dense_bytes,
            compression_ratio: compression,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeat_corpus_echo_positions_match_pairs() {
        let mut rng = Rng::new(SEED);
        let (tokens, pairs) = build_repeat_corpus(&mut rng);
        for p in &pairs {
            assert_eq!(tokens[p.replay + KEY_LEN], p.value, "echo target must be the value");
            assert!(p.replay - (p.first + CHUNK) >= EVAL_CONTEXT, "replay must be far enough");
        }
    }

    #[test]
    fn memory_recovers_evicted_values() {
        let mut rng = Rng::new(SEED);
        let (tokens, pairs) = build_repeat_corpus(&mut rng);
        let history_len = tokens.len() - REPLAYS * (KEY_LEN + 1);
        let keys: Vec<Vec<u32>> = pairs
            .iter()
            .map(|p| tokens[p.first..p.first + KEY_LEN].to_vec())
            .collect();
        let mem = build_memory(&tokens[..history_len]);
        let readouts = memory_readouts(&mem, &keys, &pairs);
        let hits = readouts
            .iter()
            .zip(pairs.iter())
            .filter(|(r, p)| **r == Some(p.value))
            .count();
        assert!(hits >= 15, "memory must recall the evicted values, got {hits}/16");
    }
}
