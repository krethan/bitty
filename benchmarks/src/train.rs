//! Phase 4: training a ternary LoRA readout on frozen bigram hidden states.
//!
//! The synthetic model's hidden state at each position is the one-hot embedding
//! of the previous token (see `perplexity::build_synthetic_model`), normalized
//! by RMSNorm to `√hidden` magnitude before `lm_head`. The readout is therefore
//! a `VOCAB × VOCAB` matrix: `out[p][o] = √hidden · W[o][p]`, and the target
//! logits are the bigram log-probs. We train a [`TernaryLoRA`] on that target
//! with the corpus transition counts as weights, then evaluate both readouts on
//! the model's *real* hidden states (post-RMSNorm), so the exact-FP32 reference
//! reproduces the Phase 1 floor (~15.9 ppl) and the ternary result is directly
//! comparable.
//!
//! **Measured result (a structural negative):** the bigram `logp` table is
//! effectively full-rank, so no rank ≤ 8 ternary adapter can represent it — the
//! trit-grid either cannot span the ~9.7-logit range (small scale) or destroys
//! the common-transition logits (large scale). All trained readouts therefore
//! score *worse* than the uniform floor (`ppl > VOCAB`), and tighter MSE fits
//! make it worse (over-confident, clipped logits). The trainer itself is
//! correct; the phase-4 low-rank learnability test passes. This benchmark
//! documents when ternary readout replacement fails and why.

use bitllm_runtime::Model;
use bitllm_train::lora::{TernaryLoRA, TernaryLoRAConfig};
use bitllm_tensor::{simd, Tensor};

use crate::export::TrainRow;
use crate::perplexity::{
    bigram_log_probs, build_synthetic_model, char_tokenize, ln_softmax, synthetic_corpus, Rng,
    SEED, EVAL_CONTEXT, VOCAB,
};

fn build_target(corpus: &str) -> (Tensor, Tensor, Tensor) {
    let tokens = char_tokenize(corpus);
    let mut counts = vec![0usize; VOCAB * VOCAB];
    for pair in tokens.windows(2) {
        counts[pair[0] as usize * VOCAB + pair[1] as usize] += 1;
    }
    let bigram = bigram_log_probs(corpus);

    // x: one-hot rows scaled by √hidden — the model's RMSNorm normalizes the
    // 4.0·onehot embedding to √hidden·onehot before lm_head (see model.rs
    // forward: hidden -> norm -> lm_head), so the readout input is that.
    let hidden_scale = (VOCAB as f32).sqrt();
    let mut x = vec![0.0f32; VOCAB * VOCAB];
    for p in 0..VOCAB {
        x[p * VOCAB + p] = hidden_scale;
    }
    let x_t = Tensor::from_slice(&x, &[VOCAB, VOCAB]);

    // The readout output for prev token p is √hidden·W[:,p]; the desired
    // logits are the bigram log-probs themselves, so target = bigram[p][o].
    let mut target = vec![0.0f32; VOCAB * VOCAB];
    for p in 0..VOCAB {
        for o in 0..VOCAB {
            target[p * VOCAB + o] = bigram[p][o] as f32;
        }
    }
    let target_t = Tensor::from_slice(&target, &[VOCAB, VOCAB]);

    let weights: Vec<f32> = counts.iter().map(|&c| c as f32).collect();
    let weights_t = Tensor::from_slice(&weights, &[VOCAB, VOCAB]);

    (x_t, target_t, weights_t)
}

/// Corpus perplexity of a fixed readout `W` (layout `[o*VOCAB+p]` = ΔW[o][p]),
/// evaluated on the model's real hidden states: logits = hidden·Wᵀ. Sliding
/// windows mirror `perplexity::evaluate` so the exact-FP32 reference reproduces
/// the Phase 1 floor.
fn eval_readout(model: &mut Model, w: &Tensor, tokens: &[u32]) -> f64 {
    let hidden = model.config.hidden_size;
    let vocab = model.config.vocab_size;
    let w_slice = w.as_f32_slice();
    let mut sum_nll = 0.0;
    let mut n = 0usize;
    let mut i = 0;
    while i + 1 < tokens.len() {
        let end = (i + EVAL_CONTEXT + 1).min(tokens.len());
        let window = &tokens[i..end];
        model.clear_cache();
        let h = model.forward_hidden(window, 0, None);
        let h_slice = h.as_f32_slice();
        let m = window.len();
        let mut logits = vec![0.0f32; m * vocab];
        simd::f32_matmul(h_slice, w_slice, &mut logits, m, hidden, vocab);
        for t in 0..window.len() - 1 {
            let target = window[t + 1] as usize;
            sum_nll += -ln_softmax(&logits[t * vocab..(t + 1) * vocab], target);
            n += 1;
        }
        i += EVAL_CONTEXT;
    }
    (sum_nll / n as f64).exp()
}

struct TrainResult {
    ppl: f64,
    train_mse: f64,
    sweeps: u64,
    weight_bytes: usize,
    fp32_bytes: usize,
}

fn train_readout(
    rank: usize,
    init_scale: f32,
    sweeps: usize,
    x: &Tensor,
    target: &Tensor,
    weights: &Tensor,
    model: &mut Model,
    tokens: &[u32],
) -> TrainResult {
    let cfg = TernaryLoRAConfig::new(VOCAB, VOCAB, rank, 1);
    let cfg = TernaryLoRAConfig {
        init_scale,
        ..cfg
    };
    let mut lora = TernaryLoRA::new(cfg);
    let mut mse = 0.0f32;
    for s in 0..sweeps {
        mse = lora.train_step(x, target, Some(weights));
        if sweeps > 200 && s % (sweeps / 4) == 0 {
            println!(
                "    [rank {rank}, scale {init_scale:.3}] sweep {s:>4} mse {mse:.6}"
            );
        }
    }
    let ppl = eval_readout(model, &lora.weight(), tokens);
    TrainResult {
        ppl,
        train_mse: mse as f64,
        sweeps: lora.steps(),
        weight_bytes: lora.bytes(),
        fp32_bytes: lora.fp32_bytes(),
    }
}

/// Trains a ternary LoRA readout on the frozen bigram hidden states and reports
/// corpus perplexity vs the exact FP32 readout and the uniform floor.
pub fn bench_train() -> Vec<TrainRow> {
    println!("\n=== Bitty Train: Ternary LoRA Readout ===\n");

    let corpus = {
        let mut rng = Rng::new(SEED);
        synthetic_corpus(&mut rng)
    };
    let tokens = char_tokenize(&corpus);
    let bigram = bigram_log_probs(&corpus);
    let (x, target, weights) = build_target(&corpus);

    // The exact FP32 readout (identical to the ppl harness FP32 run):
    // W[o][p] = bigram_logp[p][o] / √hidden, evaluated on real hidden states.
    let mut exact = vec![0.0f32; VOCAB * VOCAB];
    let scale = 1.0 / (VOCAB as f64).sqrt();
    for p in 0..VOCAB {
        for o in 0..VOCAB {
            exact[o * VOCAB + p] = (bigram[p][o] * scale) as f32;
        }
    }
    let exact_t = Tensor::from_slice(&exact, &[VOCAB, VOCAB]);
    let mut model = build_synthetic_model(&bigram);
    let ppl_fp32 = eval_readout(&mut model, &exact_t, &tokens);
    println!(
        "  exact FP32 readout ppl: {ppl_fp32:.3}   (uniform floor {VOCAB})",
    );
    println!(
        "  training target: bigram_logp on the 96×96 prev→next table,\n\
         \x20 read out via √hidden·W; weighted by corpus transition counts;\n\
         \x20 ternary LoRA via block coordinate descent. Eval on real hidden states.\n"
    );

    let configs: &[(usize, f32, usize)] = &[
        (4, 0.1, 400),
        (4, 0.05, 400),
        (8, 0.05, 300),
        (8, 0.1, 300),
    ];

    let mut rows = Vec::with_capacity(configs.len());
    for &(rank, init_scale, sweeps) in configs {
        let r = train_readout(
            rank,
            init_scale,
            sweeps,
            &x,
            &target,
            &weights,
            &mut model,
            &tokens,
        );
        let compression = r.fp32_bytes as f64 / r.weight_bytes.max(1) as f64;
        println!(
            "  TERNARY-R{rank}-S{init_scale:.2}  ppl={:>7.3}  mse={:.5}  sweeps={}  bytes={} ({}x vs FP32)",
            r.ppl,
            r.train_mse,
            r.sweeps,
            r.weight_bytes,
            compression
        );
        rows.push(TrainRow {
            name: format!("TERNARY-R{rank}-S{init_scale:.2}"),
            ppl: r.ppl,
            ppl_fp32,
            rank,
            init_scale: init_scale as f64,
            sweeps: r.sweeps,
            train_mse: r.train_mse as f64,
            weight_bytes: r.weight_bytes,
            fp32_bytes: r.fp32_bytes,
            compression_ratio: compression,
        });
    }

    println!();
    for row in &rows {
        let gap = (row.ppl / ppl_fp32).log2();
        println!(
            "  {}: ppl {:.2}  vs FP32 {:.2}  ({:+.2} bits/tok)  vs uniform {VOCAB}",
            row.name, row.ppl, ppl_fp32, gap
        );
    }
    println!(
        "  NOTE: the bigram logp table is full-rank; rank-≤8 ternary adapters cannot\n\
         \x20       represent it, so every trained readout lands above the uniform floor.\n\
         \x20       See the module doc for the structural reason."
    );
    println!();
    rows
}
