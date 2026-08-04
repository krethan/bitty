//! Phase 5: full-graph STE quantization-aware training (QAT) vs naive quantization.
//!
//! QAT measures the distillation loss **at the model output**: the deployed
//! quantized graph is the student, a frozen FP32 model is the teacher, and
//! `L = mean (logits_qat − logits_fp32)²`. A per-projection reconstruction
//! objective is provably equivalent to a naive round of `W`, so it cannot beat
//! naive quantization; only when later layers can compensate for earlier
//! layers' quantization error does QAT help (see `crates/train/src/qat.rs`).
//!
//! This harness reproduces the four deployment formats from the perplexity
//! suite (BIT1 / BIT1-A8 / BIT1-OL / BIT1-OL-A8) on identical seeded models and
//! compares, per format:
//!
//! - **naive** — quantize the FP32 model, no training;
//! - **QAT** — end-to-end STE against frozen FP32 teacher logits on a set of
//!   training windows, then deploy with the same config.
//!
//! Both are scored on a **held-out** corpus window (logit MSE vs the FP32
//! teacher) and on the full-corpus sliding-window perplexity. The QAT/naive
//! MSE ratio < 1 demonstrates that the weight corrections learned on the
//! training windows transfer to unseen positions.

use bitllm_quantization::QuantConfig;
use bitllm_runtime::Model;
use bitllm_tensor::Tensor;
use bitllm_train::{mean_sq_error, QATConfig, QATModel};

use crate::export::QatRow;
use crate::perplexity::{
    bigram_log_probs, build_synthetic_model, char_tokenize, ln_softmax, synthetic_corpus, Rng,
    EVAL_CONTEXT, SEED, VOCAB,
};

/// Outlier fraction for the BIT1-OL modes (matches the perplexity harness).
const OUTLIER_FRAC: f64 = 0.01;
/// Number of disjoint training windows drawn from the corpus.
const TRAIN_WINDOWS: usize = 4;
/// Stride between consecutive training-window start offsets.
const WINDOW_STRIDE: usize = 1000;
/// Start offset of the held-out evaluation window (no overlap with training).
const EVAL_OFFSET: usize = 5000;

/// Logit MSE of `model` against the frozen teacher on one window.
fn logit_mse(model: &mut Model, teacher: &Tensor, window: &[u32]) -> f32 {
    model.clear_cache();
    mean_sq_error(&model.forward(window), teacher)
}

/// Sliding-window perplexity on the full corpus (mirrors
/// `perplexity::evaluate`).
fn corpus_ppl(model: &mut Model, tokens: &[u32], ctx: usize) -> f64 {
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
    (sum_nll / n as f64).exp()
}

/// Naive-quantization vs QAT-trained deployed error for one deployment format.
pub fn bench_qat() -> Vec<QatRow> {
    println!("\n=== Bitty QAT: Full-Graph STE vs Naive Quantization ===\n");

    let corpus = {
        let mut rng = Rng::new(SEED);
        synthetic_corpus(&mut rng)
    };
    let tokens = char_tokenize(&corpus);
    let bigram = bigram_log_probs(&corpus);

    // Deterministic training windows + a held-out eval window, all drawn from
    // the corpus (window length = EVAL_CONTEXT, the benchmark's decode window).
    let windows: Vec<Vec<u32>> = (0..TRAIN_WINDOWS)
        .map(|i| tokens[i * WINDOW_STRIDE..i * WINDOW_STRIDE + EVAL_CONTEXT].to_vec())
        .collect();
    let eval_window: Vec<u32> = tokens[EVAL_OFFSET..EVAL_OFFSET + EVAL_CONTEXT].to_vec();

    // Frozen FP32 teacher logits for the held-out window. All modes build the
    // identical seeded model, so this reference is shared.
    let mut teacher_model = build_synthetic_model(&bigram);
    teacher_model.clear_cache();
    let teacher_logits = teacher_model.forward(&eval_window);

    println!(
        "  model: vocab={VOCAB} hidden=96 layers=1  windows={}x{EVAL_CONTEXT}  eval@{EVAL_OFFSET}  teacher=FP32\n",
        TRAIN_WINDOWS
    );

    let configs: &[(&str, QuantConfig)] = &[
        ("BIT1", QuantConfig::ternary().without_a8()),
        ("BIT1-A8", QuantConfig::ternary()),
        (
            "BIT1-OL",
            QuantConfig::ternary_with_outliers(OUTLIER_FRAC).without_a8(),
        ),
        (
            "BIT1-OL-A8",
            QuantConfig::ternary_with_outliers(OUTLIER_FRAC),
        ),
    ];

    let mut rows = Vec::with_capacity(configs.len());
    for (name, quant) in configs {
        // Naive baseline: quantize without any training.
        let mut naive = build_synthetic_model(&bigram);
        naive.quantize_to_bit1_with_config(quant);
        let naive_mse = logit_mse(&mut naive, &teacher_logits, &eval_window);
        let ppl_naive = corpus_ppl(&mut naive, &tokens, EVAL_CONTEXT);

        // QAT: end-to-end STE against the frozen teacher, then deploy with the
        // same quantization config.
        let qat_cfg = QATConfig::new().with_quant(quant.clone());
        let mut qat_model = QATModel::new(build_synthetic_model(&bigram), qat_cfg.clone());
        let (train_mse_start, train_mse_end) = qat_model.train(&windows);
        let mut deployed = qat_model.deploy();
        let qat_mse = logit_mse(&mut deployed, &teacher_logits, &eval_window);
        let ppl_qat = corpus_ppl(&mut deployed, &tokens, EVAL_CONTEXT);

        let ratio = qat_mse / naive_mse.max(f32::MIN_POSITIVE);
        println!(
            "  {name:<10}  eval logit-MSE  naive {naive_mse:>10.5}  ->  qat {qat_mse:>10.5}  ({ratio:>6.3}x)\n\
             \x20             train-MSE {train_mse_start:.5} -> {train_mse_end:.5}   ppl naive {ppl_naive:.2} / qat {ppl_qat:.2}",
        );
        rows.push(QatRow {
            name: name.to_string(),
            naive_mse: naive_mse as f64,
            qat_mse: qat_mse as f64,
            mse_ratio: ratio as f64,
            ppl_naive,
            ppl_qat,
            steps: qat_cfg.steps,
        });
    }

    println!();
    for row in &rows {
        let arrow = if row.mse_ratio < 1.0 { "down" } else { "up" };
        println!(
            "  {}: QAT/naive logit-MSE {:.3}x ({})  ppl {:.2} -> {:.2}",
            row.name, row.mse_ratio, arrow, row.ppl_naive, row.ppl_qat
        );
    }
    println!(
        "  NOTE: QAT walks the full deployed graph with STE at each quantizer\n\
         \x20       (∂Q/∂W ≜ 1) against frozen FP32 logits; only the seven\n\
         \x20       projections per layer train. MSE is measured on a held-out\n\
         \x20       corpus window, so a ratio < 1 shows the learned weight\n\
         \x20       corrections transfer to unseen positions.\n"
    );
    rows
}
