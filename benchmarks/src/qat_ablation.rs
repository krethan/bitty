//! Per-projection ablation studies for QAT.
//!
//! Tests which projections benefit most from QAT training by training subsets
//! of projections and measuring the impact on held-out logit MSE and perplexity.

use bitllm_quantization::QuantConfig;
use bitllm_runtime::Model;
use bitllm_tensor::Tensor;
use bitllm_train::{mean_sq_error, QATConfig, QATModel};

use crate::export::QatAblationRow;
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

pub fn bench_qat_ablation() -> Vec<QatAblationRow> {
    println!("\n=== QAT Per-Projection Ablation Study ===\n");

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

    // Define ablation subsets to test (reduced set for faster execution)
    let ablation_configs: &[(&str, Vec<&str>)] = &[
        ("all", vec![]), // Empty means train all
        ("attn_only", vec!["q", "k", "v", "o"]),
        ("ffn_only", vec!["up", "gate", "down"]),
        ("q_only", vec!["q"]),
        ("v_only", vec!["v"]),
        ("o_only", vec!["o"]),
        ("up_only", vec!["up"]),
        ("gate_only", vec!["gate"]),
        ("down_only", vec!["down"]),
    ];

    let mut rows = Vec::new();

    for (format_name, quant) in configs {
        println!("  Format: {}", format_name);

        // Baseline: naive quantization without training
        let mut naive = build_synthetic_model(&bigram);
        naive.quantize_to_bit1_with_config(quant);
        let naive_mse = logit_mse(&mut naive, &teacher_logits, &eval_window);
        let ppl_naive = corpus_ppl(&mut naive, &tokens, EVAL_CONTEXT);

        for (ablation_name, projections) in ablation_configs {
            let train_projs: Vec<String> = projections.iter().map(|s| s.to_string()).collect();
            let qat_cfg = QATConfig::new()
                .with_lr(0.02)
                .with_steps(200)
                .with_quant(quant.clone())
                .with_train_projections(train_projs);

            let mut qat_model = QATModel::new(build_synthetic_model(&bigram), qat_cfg.clone());
            let (train_mse_start, train_mse_end) = qat_model.train(&windows);
            let mut deployed = qat_model.deploy();
            let qat_mse = logit_mse(&mut deployed, &teacher_logits, &eval_window);
            let ppl_qat = corpus_ppl(&mut deployed, &tokens, EVAL_CONTEXT);

            let ratio = qat_mse / naive_mse.max(f32::MIN_POSITIVE);
            let improvement = if naive_mse > 0.0 {
                (naive_mse - qat_mse) / naive_mse * 100.0
            } else {
                0.0
            };

            rows.push(QatAblationRow {
                format: format_name.to_string(),
                ablation: ablation_name.to_string(),
                projections: projections.join(","),
                naive_mse: naive_mse as f64,
                qat_mse: qat_mse as f64,
                mse_ratio: ratio as f64,
                mse_improvement_pct: improvement as f64,
                ppl_naive,
                ppl_qat,
                train_mse_start: train_mse_start as f64,
                train_mse_end: train_mse_end as f64,
            });

            println!(
                "    {:<15}  MSE ratio={:.4}  improvement={:6.2}%  ppl={:.2}  train={:.5}->{:.5}",
                ablation_name, ratio, improvement, ppl_qat, train_mse_start, train_mse_end
            );
        }
        println!();
    }

    // Summary: best ablation per format
    println!("=== Best Ablation Per Format ===\n");
    for (format_name, _) in configs {
        let format_rows: Vec<_> = rows.iter().filter(|r| r.format == *format_name).collect();
        let best = format_rows
            .iter()
            .min_by(|a, b| a.mse_ratio.partial_cmp(&b.mse_ratio).unwrap())
            .unwrap();
        println!(
            "  {:<10}  {:<15}  MSE ratio={:.4}  improvement={:.2}%",
            format_name, best.ablation, best.mse_ratio, best.mse_improvement_pct
        );
    }
    println!();

    rows
}
