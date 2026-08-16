//! Sliding-window cross-entropy / perplexity evaluation on a real corpus
//! (e.g. WikiText-2), comparable to published benchmarks.
//!
//! Usage:
//!   corpus_ppl <model.gguf | model_dir> <tokenizer.json> <corpus.txt>
//!              [context] [stride] [max_tokens] [threads] [start]
//!
//! The corpus is tokenized once into a single token stream, then evaluated in
//! sliding windows of `context` tokens stepping by `stride` (the classic
//! GPT-style recipe). For overlapping windows only the trailing `stride`
//! tokens are scored so every token contributes exactly once. `max_tokens`
//! (0 = unlimited) caps how much of the corpus is evaluated.
//!
//! Windows are evaluated in parallel across `rayon`'s pool; each worker thread
//! builds one copy of the model from the checkpoint so the whole window stream
//! is scanned exactly once.

use bitllm_runtime::gguf::GgufLoader;
use bitllm_runtime::{load_safetensors_weights, Model, ModelConfig, SafeTensorsLoader};
use bitllm_server::loader::load_gguf_weights;
use bitllm_tokenizer::BpeTokenizer;
use rayon::prelude::*;
use std::path::PathBuf;

fn ln_softmax(logits: &[f32], target: usize) -> f64 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|&x| ((x - max) as f64).exp()).sum();
    let lse = max as f64 + sum.ln();
    logits[target] as f64 - lse
}

enum Source {
    Gguf { path: String, config: ModelConfig },
    Safetensors { path: String, config: ModelConfig },
}

fn build_model(source: &Source) -> Model {
    match source {
        Source::Gguf { path, config } => {
            let loader = GgufLoader::load(path).expect("load gguf");
            let mut model = Model::new(config.clone());
            load_gguf_weights(&mut model, &loader, bitllm_tensor::Device::Cpu, config);
            model
        }
        Source::Safetensors { path, config } => {
            let loader = SafeTensorsLoader::load(path).expect("load safetensors");
            let mut model = Model::new(config.clone());
            load_safetensors_weights(&mut model, &loader, config, None);
            model
        }
    }
}

fn resolve_source(model_arg: &str) -> Source {
    if model_arg.ends_with(".gguf") {
        let loader = GgufLoader::load(model_arg).expect("load gguf metadata");
        let config = loader.config_from_metadata().expect("gguf config");
        Source::Gguf {
            path: model_arg.to_string(),
            config,
        }
    } else {
        let dir = PathBuf::from(model_arg);
        let json = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
        let config = ModelConfig::from_huggingface_json(&json).expect("parse config");
        Source::Safetensors {
            path: dir.join("model.safetensors").display().to_string(),
            config,
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model_arg = args.next().expect("usage: corpus_ppl <model.gguf | model_dir> <tokenizer.json> <corpus.txt> [context] [stride] [max_tokens]");
    let tok_path = PathBuf::from(args.next().expect("missing tokenizer.json"));
    let corpus_path = PathBuf::from(args.next().expect("missing corpus.txt"));
    let context: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(512);
    let stride: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(context);
    let max_tokens: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let threads: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4).max(1);
    let start_token: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);

    let source = resolve_source(&model_arg);
    let max_seq_len = match &source {
        Source::Gguf { config, .. } | Source::Safetensors { config, .. } => config.max_seq_len,
    };
    let vocab = match &source {
        Source::Gguf { config, .. } | Source::Safetensors { config, .. } => config.vocab_size,
    };

    let tokenizer = BpeTokenizer::load(&tok_path).expect("load tokenizer");
    let corpus = std::fs::read_to_string(&corpus_path).expect("read corpus");
    let t0 = std::time::Instant::now();
    eprintln!("[stage] tokenizing {} bytes...", corpus.len());
    let all_tokens = tokenizer.encode(&corpus);
    eprintln!(
        "[stage] tokenized {} tokens in {:.1}s",
        all_tokens.len(),
        t0.elapsed().as_secs_f32()
    );
    let tokens: &[u32] = if max_tokens > 0 || start_token > 0 {
        let end = if max_tokens > 0 {
            (start_token + max_tokens).min(all_tokens.len())
        } else {
            all_tokens.len()
        };
        &all_tokens[start_token..end]
    } else {
        &all_tokens[..]
    };
    eprintln!("[stage] tokenized {} tokens", tokens.len());

    let context = context.min(max_seq_len).max(2);
    let stride = stride.min(context);

    // Build the window list once: (start, end, first_scored_offset).
    let mut windows: Vec<(usize, usize, usize)> = Vec::new();
    let mut i = 0usize;
    while i + 1 < tokens.len() {
        let end = (i + context + 1).min(tokens.len());
        let start = if i == 0 {
            0
        } else {
            context.saturating_sub(stride)
        };
        windows.push((i, end, start));
        if end == tokens.len() {
            break;
        }
        i += stride;
    }
    eprintln!(
        "[stage] {} windows, context={} stride={}",
        windows.len(),
        context,
        stride
    );

    let started = std::time::Instant::now();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("build thread pool");
    let nan_bad = std::sync::atomic::AtomicBool::new(false);
    // Split windows across threads; each thread owns one model copy.
    let results: Vec<(f64, usize)> = pool.install(|| {
        windows
            .par_chunks(windows.len().div_ceil(threads).max(1))
            .map(|chunk| {
                let mut model = build_model(&source);
                let mut nll = 0.0f64;
                let mut n = 0usize;
                for &(w_start, w_end, score_start) in chunk {
                    let window = &tokens[w_start..w_end];
                    model.clear_cache();
                    let logits = model.forward(window);
                    let slice = logits.as_f32_slice();
                    for t in score_start..window.len() - 1 {
                        let target = window[t + 1] as usize;
                        let row = &slice[t * vocab..(t + 1) * vocab];
                        let contrib = -ln_softmax(row, target);
                        if !contrib.is_finite() && !nan_bad.swap(true, std::sync::atomic::Ordering::Relaxed) {
                            eprintln!(
                                "[nan] window={} corpus_tok={} local_t={} target={} logits_max={} contrib={}",
                                w_start,
                                start_token + w_start + t + 1,
                                t,
                                target,
                                row.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                                contrib
                            );
                        }
                        nll += contrib;
                        n += 1;
                    }
                }
                (nll, n)
            })
            .collect()
    });

    let sum_nll: f64 = results.iter().map(|r| r.0).sum();
    let n: usize = results.iter().map(|r| r.1).sum();
    let ppl = (sum_nll / n as f64).exp();
    println!(
        "tokens_scored={} nll={:.2} ppl={:.3} bits/tok={:.3} windows={} elapsed={:.1}s",
        n,
        sum_nll,
        ppl,
        ppl.log2(),
        windows.len(),
        started.elapsed().as_secs_f32()
    );
    println!("  uniform floor would be ppl={} (vocab)", vocab);
}
