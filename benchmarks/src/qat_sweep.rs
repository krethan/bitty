//! Phase 6: QAT hyperparameter sweep — find optimal lr and steps for each format.
//!
//! Runs QAT with different learning rates and training steps for each deployment
//! format (BIT1 / BIT1-A8 / BIT1-OL / BIT1-OL-A8), measuring held-out logit MSE
//! and full-corpus perplexity. Identifies the best config per format.

use bitllm_quantization::QuantConfig;
use bitllm_runtime::Model;
use bitllm_tensor::Tensor;
use bitllm_train::{mean_sq_error, QATConfig, QATModel};

use crate::export::QatSweepRow;
use crate::perplexity::{
    bigram_log_probs, build_synthetic_model, char_tokenize, ln_softmax, synthetic_corpus, Rng,
    EVAL_CONTEXT, SEED, VOCAB,
};

const OUTLIER_FRAC: f64 = 0.01;
const TRAIN_WINDOWS: usize = 4;
const WINDOW_STRIDE: usize = 1000;
const EVAL_OFFSET: usize = 5000;

fn logit_mse(model: &mut Model, teacher: &Tensor, window: &[u32]) -> f32 {
    model.clear_cache();
    mean_sq_error(&model.forward(window), teacher)
}

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

pub fn bench_qat_sweep() -> Vec<QatSweepRow> {
    println!("\n=== QAT Hyperparameter Sweep ===\n");

    let corpus = {
        let mut rng = Rng::new(SEED);
        synthetic_corpus(&mut rng)
    };
    let tokens = char_tokenize(&corpus);
    let bigram = bigram_log_probs(&corpus);

    let windows: Vec<Vec<u32>> = (0..TRAIN_WINDOWS)
        .map(|i| tokens[i * WINDOW_STRIDE..i * WINDOW_STRIDE + EVAL_CONTEXT].to_vec())
        .collect();
    let eval_window: Vec<u32> = tokens[EVAL_OFFSET..EVAL_OFFSET + EVAL_CONTEXT].to_vec();

    let mut teacher_model = build_synthetic_model(&bigram);
    teacher_model.clear_cache();
    let teacher_logits = teacher_model.forward(&eval_window);

    println!(
        "  model: vocab={VOCAB} hidden=96 layers=1  windows={}x{EVAL_CONTEXT}  eval@{EVAL_OFFSET}\n",
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

    let lrs = [0.02, 0.05, 0.1];
    let steps_list = [100, 200, 400];

    let mut rows = Vec::new();
    let mut best_configs: std::collections::HashMap<String, (f32, usize, f64)> =
        std::collections::HashMap::new();

    for (format_name, quant) in configs {
        println!("  Format: {}", format_name);

        let mut naive = build_synthetic_model(&bigram);
        naive.quantize_to_bit1_with_config(quant);
        let naive_mse = logit_mse(&mut naive, &teacher_logits, &eval_window);
        let ppl_naive = corpus_ppl(&mut naive, &tokens, EVAL_CONTEXT);

        let mut best_mse_ratio = f64::MAX;
        let mut best_lr = 0.0;
        let mut best_steps = 0;

        for &lr in &lrs {
            for &steps in &steps_list {
                let qat_cfg = QATConfig::new()
                    .with_lr(lr)
                    .with_steps(steps)
                    .with_quant(quant.clone());
                let mut qat_model = QATModel::new(build_synthetic_model(&bigram), qat_cfg.clone());
                let (train_mse_start, train_mse_end) = qat_model.train(&windows);
                let mut deployed = qat_model.deploy();
                let qat_mse = logit_mse(&mut deployed, &teacher_logits, &eval_window);
                let ppl_qat = corpus_ppl(&mut deployed, &tokens, EVAL_CONTEXT);

                let ratio = qat_mse / naive_mse.max(f32::MIN_POSITIVE);

                rows.push(QatSweepRow {
                    format: format_name.to_string(),
                    lr: lr as f64,
                    steps,
                    naive_mse: naive_mse as f64,
                    qat_mse: qat_mse as f64,
                    mse_ratio: ratio as f64,
                    ppl_naive,
                    ppl_qat,
                    train_mse_start: train_mse_start as f64,
                    train_mse_end: train_mse_end as f64,
                });

                if (ratio as f64) < best_mse_ratio {
                    best_mse_ratio = ratio as f64;
                    best_lr = lr;
                    best_steps = steps;
                }

                print!(
                    "    lr={:.3} steps={:>3}  MSE ratio={:.4}  ppl={:.2}  train={:.5}->{:.5}\n",
                    lr, steps, ratio, ppl_qat, train_mse_start, train_mse_end
                );
            }
        }

        best_configs.insert(
            format_name.to_string(),
            (best_lr, best_steps, best_mse_ratio),
        );
        println!(
            "  Best for {}: lr={:.3} steps={} ratio={:.4}\n",
            format_name, best_lr, best_steps, best_mse_ratio
        );
    }

    println!("=== Best Configs Per Format ===\n");
    for (format, (lr, steps, ratio)) in &best_configs {
        println!(
            "  {:<10}  lr={:.3}  steps={:>3}  MSE ratio={:.4}",
            format, lr, steps, ratio
        );
    }
    println!();

    // Test stability features
    println!("=== Testing Stability Features ===\n");
    test_stability_features(&bigram, &windows, &eval_window, &teacher_logits, &tokens);

    rows
}

fn test_stability_features(
    bigram: &[Vec<f64>],
    windows: &[Vec<u32>],
    eval_window: &[u32],
    teacher_logits: &Tensor,
    tokens: &[u32],
) {
    let quant = QuantConfig::ternary();
    let baseline_cfg = QATConfig::new()
        .with_lr(0.05)
        .with_steps(200)
        .with_quant(quant.clone());

    // Baseline
    let mut baseline_model = QATModel::new(build_synthetic_model(bigram), baseline_cfg.clone());
    let (start, end) = baseline_model.train(windows);
    let mut deployed = baseline_model.deploy();
    let baseline_mse = logit_mse(&mut deployed, teacher_logits, eval_window);
    let baseline_ppl = corpus_ppl(&mut deployed, tokens, EVAL_CONTEXT);
    println!(
        "  Baseline (lr=0.05, steps=200): MSE={:.4}, ppl={:.2}, train={:.5}->{:.5}",
        baseline_mse, baseline_ppl, start, end
    );

    // With gradient clipping
    let clip_cfg = QATConfig::new()
        .with_lr(0.05)
        .with_steps(200)
        .with_quant(quant.clone())
        .with_grad_clip(1.0);
    let mut clip_model = QATModel::new(build_synthetic_model(bigram), clip_cfg);
    let (start, end) = clip_model.train(windows);
    let mut deployed = clip_model.deploy();
    let clip_mse = logit_mse(&mut deployed, teacher_logits, eval_window);
    let clip_ppl = corpus_ppl(&mut deployed, tokens, EVAL_CONTEXT);
    println!(
        "  + grad_clip(1.0):                MSE={:.4}, ppl={:.2}, train={:.5}->{:.5}",
        clip_mse, clip_ppl, start, end
    );

    // With warmup
    let warmup_cfg = QATConfig::new()
        .with_lr(0.05)
        .with_steps(200)
        .with_quant(quant.clone())
        .with_warmup(50);
    let mut warmup_model = QATModel::new(build_synthetic_model(bigram), warmup_cfg);
    let (start, end) = warmup_model.train(windows);
    let mut deployed = warmup_model.deploy();
    let warmup_mse = logit_mse(&mut deployed, teacher_logits, eval_window);
    let warmup_ppl = corpus_ppl(&mut deployed, tokens, EVAL_CONTEXT);
    println!(
        "  + warmup(50):                    MSE={:.4}, ppl={:.2}, train={:.5}->{:.5}",
        warmup_mse, warmup_ppl, start, end
    );

    // With cosine decay
    let decay_cfg = QATConfig::new()
        .with_lr(0.05)
        .with_steps(200)
        .with_quant(quant.clone())
        .with_cosine_decay(true);
    let mut decay_model = QATModel::new(build_synthetic_model(bigram), decay_cfg);
    let (start, end) = decay_model.train(windows);
    let mut deployed = decay_model.deploy();
    let decay_mse = logit_mse(&mut deployed, teacher_logits, eval_window);
    let decay_ppl = corpus_ppl(&mut deployed, tokens, EVAL_CONTEXT);
    println!(
        "  + cosine_decay:                  MSE={:.4}, ppl={:.2}, train={:.5}->{:.5}",
        decay_mse, decay_ppl, start, end
    );

    // With early stopping
    let early_cfg = QATConfig::new()
        .with_lr(0.05)
        .with_steps(400)
        .with_quant(quant.clone())
        .with_eval_window(eval_window.to_vec())
        .with_patience(20);
    let mut early_model = QATModel::new(build_synthetic_model(bigram), early_cfg);
    let (start, end) = early_model.train(windows);
    let mut deployed = early_model.deploy();
    let early_mse = logit_mse(&mut deployed, teacher_logits, eval_window);
    let early_ppl = corpus_ppl(&mut deployed, tokens, EVAL_CONTEXT);
    println!(
        "  + early_stop(patience=20):       MSE={:.4}, ppl={:.2}, train={:.5}->{:.5}",
        early_mse, early_ppl, start, end
    );

    // Combined: warmup + decay + clip
    let combined_cfg = QATConfig::new()
        .with_lr(0.05)
        .with_steps(200)
        .with_quant(quant)
        .with_warmup(50)
        .with_cosine_decay(true)
        .with_grad_clip(1.0);
    let mut combined_model = QATModel::new(build_synthetic_model(bigram), combined_cfg);
    let (start, end) = combined_model.train(windows);
    let mut deployed = combined_model.deploy();
    let combined_mse = logit_mse(&mut deployed, teacher_logits, eval_window);
    let combined_ppl = corpus_ppl(&mut deployed, tokens, EVAL_CONTEXT);
    println!(
        "  Combined (warmup+decay+clip):    MSE={:.4}, ppl={:.2}, train={:.5}->{:.5}",
        combined_mse, combined_ppl, start, end
    );
    println!();
}
