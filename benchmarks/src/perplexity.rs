use bitllm_runtime::Model;
use bitllm_tensor::Tensor;

use crate::export::PerplexityRow;

// ── Config ────────────────────────────────────────────────────────────

pub(crate) const SEED: u64 = 42;
pub(crate) const CORPUS_CHARS: usize = 8_000;
pub(crate) const EVAL_CONTEXT: usize = 64;
/// Scale of the random transformer projections. Tuned (0.1) so the FP32 model
/// scores well below the uniform floor while bit1 quantization produces a
/// clear, reproducible degradation. The exact gap is deterministic for a
/// fixed seed; what matters is its stability across runs as later phases
/// change the matmul/quantization path.
const TRANSFORMER_SCALE: f32 = 0.1;

// ── Deterministic RNG ────────────────────────────────────────────────

pub(crate) struct Rng {
    state: u64,
}

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        self.state.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform in `[0.0, 1.0)`.
    pub(crate) fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Gaussian via Box-Muller.
    fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_unit().max(1e-10);
        let u2 = self.next_unit();
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

// ── Synthetic corpus ──────────────────────────────────────────────────

/// Builds a bigram transition table from a sample of English-like text.
fn build_bigram(text: &str) -> Vec<(char, Vec<(char, f64)>)> {
    let mut counts: std::collections::HashMap<char, std::collections::HashMap<char, usize>> =
        Default::default();
    let mut order: Vec<char> = Vec::new();
    for pair in text.chars().zip(text.chars().skip(1)) {
        counts.entry(pair.0).or_insert_with(|| {
            order.push(pair.0);
            Default::default()
        });
        *counts.get_mut(&pair.0).unwrap().entry(pair.1).or_insert(0) += 1;
    }

    let mut table = Vec::with_capacity(order.len());
    for c in order {
        let entry = counts.get(&c).expect("entry exists");
        let total: usize = entry.values().sum();
        let mut next: Vec<(char, f64)> = entry
            .iter()
            .map(|(&nc, &n)| (nc, n as f64 / total as f64))
            .collect();
        // Sort for determinism so ties sample in a stable order.
        next.sort_by_key(|(c, _)| *c);
        table.push((c, next));
    }
    table
}

/// Samples `target` characters from the bigram table.
fn sample_corpus(rng: &mut Rng, table: &[(char, Vec<(char, f64)>)], target: usize) -> String {
    let mut out = String::with_capacity(target);
    let start = rng.next_u64() % table.len() as u64;
    let mut cur = table[start as usize].0;
    out.push(cur);

    while out.len() < target {
        let entry = table.iter().find(|(c, _)| *c == cur).expect("in table");
        let r = rng.next_unit();
        let mut acc = 0.0;
        let mut next_char = cur;
        for (nc, p) in &entry.1 {
            acc += p;
            if r < acc {
                next_char = *nc;
                break;
            }
        }
        out.push(next_char);
        cur = next_char;
    }
    out
}

pub(crate) fn synthetic_corpus(rng: &mut Rng) -> String {
    let seed_text = "the quick brown fox jumps over the lazy dog and runs into the forest \
        where the tall trees whisper in the wind while the birds sing a soft song \
        about the quiet lake and the gentle hills beyond the old stone bridge";
    let table = build_bigram(seed_text);
    sample_corpus(rng, &table, CORPUS_CHARS)
}

// ── Char tokenizer for the synthetic vocab ────────────────────────────

/// Printable ASCII character tokenizer; ids are `char - ' '` in `0..=94`.
pub(crate) fn char_tokenize(text: &str) -> Vec<u32> {
    text.chars()
        .map(|c| {
            let id = c as u32;
            debug_assert!(
                c.is_ascii_graphic() || c == ' ',
                "corpus char {:?} out of range",
                c
            );
            id - ' ' as u32
        })
        .collect()
}

// ── Synthetic model ───────────────────────────────────────────────────

pub(crate) const VOCAB: usize = 96;

pub(crate) fn synthetic_config() -> bitllm_runtime::ModelConfig {
    bitllm_runtime::ModelConfig {
        vocab_size: VOCAB,
        hidden_size: 96,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: Some(4),
        intermediate_size: 96,
        norm_eps: 1e-5,
        max_seq_len: 128,
        rope_theta: 10000.0,
        tie_word_embeddings: false,
        sub_ln: false,
        rope_scaling: None,
        architecture: bitllm_runtime::Architecture::Llama,
        activation: bitllm_runtime::Activation::SiluGated,
        norm_type: bitllm_runtime::NormType::RmsNorm,
        use_rope: true,
        position_embeddings: None,
        qk_norm: false,
        sliding_window: None,
        head_dim: None,
        post_ffn_norm: false,
        one_centered_norm: false,
        attn_logit_softcap: None,
        final_logit_softcap: None,
        query_pre_attn_scalar: None,
    }
}

fn fill_gaussian(tensor: &mut Tensor, rng: &mut Rng, scale: f32) {
    let slice = tensor.as_f32_slice_mut();
    for x in slice {
        *x = rng.next_gaussian() * scale;
    }
}

/// Add-one smoothed bigram log-probabilities from the corpus:
/// `logp[prev][target]`.
pub(crate) fn bigram_log_probs(corpus: &str) -> Vec<Vec<f64>> {
    let mut counts = vec![vec![0usize; VOCAB]; VOCAB];
    let tokens = char_tokenize(corpus);
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

/// Builds a deterministic synthetic model whose embedding → lm_head path
/// realizes the corpus bigram language model: logits ≈ log P(target | prev).
///
/// One-hot embeddings keep the F32 lm_head readout exact under RMSNorm (the
/// norm erases embedding magnitude, so any dense row has per-coordinate signal
/// ~1 and is swamped by the random projections' mixing); attention/FFN
/// projections are random (scaled down) and get bit1-packed.
pub(crate) fn build_synthetic_model(bigram: &[Vec<f64>]) -> Model {
    let config = synthetic_config();
    let mut model = Model::new(config.clone());
    let mut rng = Rng::new(SEED);

    // One-hot embeddings (rows 0..=94; row 95 unused).
    for t in 0..VOCAB {
        model.embedding.weight.set_flat_f32(t * VOCAB + t, 4.0);
    }

    for layer in &mut model.layers {
        fill_gaussian(
            &mut layer.attention.q_proj.weight,
            &mut rng,
            TRANSFORMER_SCALE,
        );
        fill_gaussian(
            &mut layer.attention.k_proj.weight,
            &mut rng,
            TRANSFORMER_SCALE,
        );
        fill_gaussian(
            &mut layer.attention.v_proj.weight,
            &mut rng,
            TRANSFORMER_SCALE,
        );
        fill_gaussian(
            &mut layer.attention.o_proj.weight,
            &mut rng,
            TRANSFORMER_SCALE,
        );
        fill_gaussian(&mut layer.ffn_up.weight, &mut rng, TRANSFORMER_SCALE);
        fill_gaussian(&mut layer.ffn_gate.weight, &mut rng, TRANSFORMER_SCALE);
        fill_gaussian(&mut layer.ffn_down.weight, &mut rng, TRANSFORMER_SCALE);
    }

    // lm_head: logits = lm_head @ (sqrt(hidden) * onehot(prev)) = logP[prev].
    let scale = (config.hidden_size as f64).sqrt();
    let slice = model.lm_head.weight.as_f32_slice_mut();
    for prev in 0..VOCAB {
        for target in 0..VOCAB {
            slice[target * VOCAB + prev] = (bigram[prev][target] / scale) as f32;
        }
    }

    model
}

// ── Perplexity evaluation ─────────────────────────────────────────────

/// Numerically stable log-softmax value for `target`.
pub(crate) fn ln_softmax(logits: &[f32], target: usize) -> f64 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|&x| ((x - max) as f64).exp()).sum();
    let lse = max as f64 + sum.ln();
    logits[target] as f64 - lse
}

/// Sliding-window negative log-likelihood. Returns `(sum_nll, n_tokens)`.
fn evaluate(model: &mut Model, tokens: &[u32], ctx: usize) -> (f64, usize) {
    let vocab = model.config.vocab_size;
    let mut sum_nll = 0.0;
    let mut n = 0usize;

    let mut i = 0;
    while i + 1 < tokens.len() {
        let end = (i + ctx + 1).min(tokens.len());
        let window = &tokens[i..end];

        model.clear_cache();
        let logits = model.forward(window);
        let slice = logits.as_f32_slice();

        for t in 0..window.len() - 1 {
            let target = window[t + 1] as usize;
            let row = &slice[t * vocab..(t + 1) * vocab];
            sum_nll += -ln_softmax(row, target);
            n += 1;
        }
        i += ctx;
    }

    (sum_nll, n)
}

// ── Main benchmark ────────────────────────────────────────────────────

/// Fraction of weights kept exact as outlier channels (Phase 2).
const OUTLIER_FRAC: f64 = 0.01;

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Fp32,
    Bit1,
    Bit1A8,
    Bit1Ol,
    Bit1OlA8,
}

/// Builds a fresh deterministic model, optionally quantizes it, and evaluates
/// perplexity. Each mode runs on an identical model so results are comparable.
fn run_mode(mode: Mode, bigram: &[Vec<f64>], tokens: &[u32]) -> PerplexityRow {
    let mut model = build_synthetic_model(bigram);
    let name = match mode {
        Mode::Fp32 => "FP32",
        Mode::Bit1 => "BIT1",
        Mode::Bit1A8 => "BIT1-A8",
        Mode::Bit1Ol => "BIT1-OL",
        Mode::Bit1OlA8 => "BIT1-OL-A8",
    };
    match mode {
        Mode::Fp32 => {}
        // W1A8 is the default quantized path; these two modes explicitly run
        // the exact f32-activation kernel to keep the A8 comparison meaningful.
        Mode::Bit1 => model.quantize_to_bit1_with_config(
            &bitllm_quantization::QuantConfig::ternary().without_a8(),
        ),
        Mode::Bit1A8 => model.quantize_to_bit1_a8(),
        Mode::Bit1Ol => model.quantize_to_bit1_with_config(
            &bitllm_quantization::QuantConfig::ternary_with_outliers(OUTLIER_FRAC).without_a8(),
        ),
        Mode::Bit1OlA8 => model.quantize_to_bit1_outliers_a8(OUTLIER_FRAC),
    }

    let (sum_nll, n) = evaluate(&mut model, tokens, EVAL_CONTEXT);
    let ppl = (sum_nll / n as f64).exp();
    println!(
        "  {:<10} ppl={:>8.3}  bits/tok={:>5.3}  (nll={:.2}, tokens={})",
        name,
        ppl,
        ppl.log2(),
        sum_nll,
        n,
    );
    PerplexityRow {
        name: name.to_string(),
        perplexity: ppl,
        bits_per_token: ppl.log2(),
        nll_sum: sum_nll,
        n_tokens: n,
        ctx_len: EVAL_CONTEXT,
    }
}

/// Runs a perplexity comparison across FP32, packed bit1, bit1 with int8
/// activations, and the Phase 2 outlier-channel variants, on a synthetic
/// deterministic model. Returns one row per mode.
pub fn bench_perplexity() -> Vec<PerplexityRow> {
    println!("\n=== Bitty Perplexity (Synthetic) ===\n");

    println!(
        "  NOTE: no model files in `models/`; using a deterministic synthetic\n\
        \x20       model + corpus so FP32 vs bit1 results are relative, not absolute.\n"
    );

    let config = synthetic_config();
    let corpus = {
        let mut rng = Rng::new(SEED);
        synthetic_corpus(&mut rng)
    };
    let tokens = char_tokenize(&corpus);
    let bigram = bigram_log_probs(&corpus);
    println!(
        "  Config: vocab={} hidden={} layers={} ctx={} corpus={} tokens={}\n",
        config.vocab_size,
        config.hidden_size,
        config.num_layers,
        EVAL_CONTEXT,
        corpus.len(),
        tokens.len(),
    );

    let fp32_row = run_mode(Mode::Fp32, &bigram, &tokens);
    let bit1_row = run_mode(Mode::Bit1, &bigram, &tokens);
    let bit1_a8_row = run_mode(Mode::Bit1A8, &bigram, &tokens);
    let bit1_ol_row = run_mode(Mode::Bit1Ol, &bigram, &tokens);
    let bit1_ol_a8_row = run_mode(Mode::Bit1OlA8, &bigram, &tokens);

    let delta = bit1_row.perplexity / fp32_row.perplexity;
    let delta_a8 = bit1_a8_row.perplexity / fp32_row.perplexity;
    let delta_ol = bit1_ol_row.perplexity / fp32_row.perplexity;
    let delta_ol_a8 = bit1_ol_a8_row.perplexity / fp32_row.perplexity;
    println!("\n  bit1/FP32 perplexity ratio:     {:.4}x", delta);
    println!("  bit1-a8/FP32 perplexity ratio:   {:.4}x", delta_a8);
    println!(
        "  bit1-ol/FP32 perplexity ratio:   {:.4}x  (outlier frac {:.0}%)",
        delta_ol,
        OUTLIER_FRAC * 100.0
    );
    println!("  bit1-ol-a8/FP32 perplexity ratio: {:.4}x", delta_ol_a8);
    println!();
    println!("  Reference floor: an untrained model scores ppl = vocab_size ({VOCAB}, uniform).");
    println!(
        "  NOTE: the synthetic projections are random noise, so weight-outlier\n\
        \x20       channels mainly matter at the matmul level (see qmatmul tests).\n"
    );
    println!();

    vec![fp32_row, bit1_row, bit1_a8_row, bit1_ol_row, bit1_ol_a8_row]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_softmax_uniform_equals_log_vocab() {
        let logits = vec![0.0f32; VOCAB];
        let nll = -ln_softmax(&logits, 10);
        assert!((nll - (VOCAB as f64).ln()).abs() < 1e-6);
    }

    #[test]
    fn uniform_model_scores_vocab_size() {
        let config = synthetic_config();
        let mut model = Model::new(config);
        let tokens: Vec<u32> = (0..50).map(|i| (i % 90) as u32).collect();
        let (sum_nll, n) = evaluate(&mut model, &tokens, EVAL_CONTEXT);
        assert!(n > 0);
        let ppl = (sum_nll / n as f64).exp();
        assert!((ppl - VOCAB as f64).abs() < 1e-3);
    }

    #[test]
    fn corpus_roundtrips_through_tokenizer() {
        let mut rng = Rng::new(SEED);
        let corpus = synthetic_corpus(&mut rng);
        assert_eq!(corpus.len(), CORPUS_CHARS);
        let tokens = char_tokenize(&corpus);
        assert_eq!(tokens.len(), CORPUS_CHARS);
        assert!(tokens.iter().all(|&t| (t as usize) < VOCAB));
    }

    #[test]
    fn synthetic_model_below_uniform_floor() {
        let mut rng = Rng::new(SEED);
        let corpus = synthetic_corpus(&mut rng);
        let tokens = char_tokenize(&corpus);
        let bigram = bigram_log_probs(&corpus);
        let mut model = build_synthetic_model(&bigram);
        let (sum_nll, n) = evaluate(&mut model, &tokens, EVAL_CONTEXT);
        let ppl = (sum_nll / n as f64).exp();
        assert!(
            ppl < VOCAB as f64,
            "synthetic model must beat the uniform floor, got ppl={ppl}"
        );
    }
}
