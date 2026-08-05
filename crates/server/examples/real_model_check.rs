//! Load a real HuggingFace model directory through the server's exact
//! `load_model` path and run greedy generation.
//!
//! Usage:
//!   cargo run -p bitllm-server --example real_model_check -- /path/to/model/dir [--quantize ternary] [--prompt "hello"]
//!
//! The directory must contain `config.json` and `model.safetensors`; a
//! `tokenizer.json` is used when present (otherwise a byte fallback).

use bitllm_server::loader::{load_model, ModelLoadOptions};
use bitllm_runtime::Sampler;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().expect("usage: real_model_check <model_dir> [--quantize ternary] [--prompt \"...\"]"));
    let mut quantize = None;
    let mut prompt = "The quick brown fox jumps over the lazy dog".to_string();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--quantize" => quantize = args.next(),
            "--prompt" => prompt = args.next().unwrap_or(prompt),
            other => anyhow::bail!("unknown arg: {other}"),
        }
    }

    let safetensors = dir.join("model.safetensors");
    let config_json = dir.join("config.json");
    anyhow::ensure!(safetensors.exists(), "missing {}", safetensors.display());
    anyhow::ensure!(config_json.exists(), "missing {}", config_json.display());

    let opts = ModelLoadOptions {
        gguf: None,
        safetensors: Some(safetensors.display().to_string()),
        config_json: Some(config_json.display().to_string()),
        config: "tiny".to_string(),
        quantize: quantize.clone(),
        device: bitllm_tensor::Device::Cpu,
    };

    let loaded = load_model(&opts)?;
    let mut model = loaded.model;
    let cfg = &model.config;

    println!("\n=== {}({}) ===", loaded.name, loaded.source);
    println!(
        "architecture={:?}  activation={:?}  norm={:?}  use_rope={}  qk_norm={}",
        cfg.architecture, cfg.activation, cfg.norm_type, cfg.use_rope, cfg.qk_norm
    );
    println!(
        "hidden={} layers={} heads={} kv_heads={} head_dim={} ff_dim={} max_seq={} sliding_window={:?} pos_emb={:?}",
        cfg.hidden_size,
        cfg.num_layers,
        cfg.num_heads,
        cfg.num_kv_heads(),
        cfg.head_dim(),
        cfg.ff_dim(),
        cfg.max_seq_len,
        cfg.sliding_window,
        cfg.position_embeddings,
    );

    // Forward-pass sanity check.
    let logits = model.forward(&[1u32, 2, 3, 4]);
    let slice = logits.as_f32_slice();
    let is_finite = slice.iter().all(|x| x.is_finite());
    println!("forward logits shape={:?} finite={}", logits.shape(), is_finite);
    anyhow::ensure!(
        is_finite,
        "non-finite logits — forward pass broken for this architecture"
    );

    // Tokenize + generate (mirrors the server worker path).
    let tokenizer_path = dir.join("tokenizer.json");
    let tokenizer = if tokenizer_path.exists() {
        println!("tokenizer: {}", tokenizer_path.display());
        bitllm_tokenizer::BpeTokenizer::load(&tokenizer_path)?
    } else {
        println!("tokenizer: byte-level fallback (no tokenizer.json)");
        let mut vocab = std::collections::HashMap::new();
        for i in 0u32..256 {
            let ch = (i as u8) as char;
            vocab.insert(ch.to_string(), i);
        }
        bitllm_tokenizer::BpeTokenizer::from_vocab_and_merges(vocab, vec![])
    };

    let sampler = Sampler::greedy();
    let prompt_tokens = tokenizer.encode_with_special(&prompt, true, false);
    println!("\nprompt: {:?}", prompt);
    println!("prompt tokens ({}): {:?}", prompt_tokens.len(), prompt_tokens);
    let generated = model.generate(&prompt_tokens, 24, &sampler);
    let text = tokenizer.decode(&generated).unwrap_or_default();
    println!("generated {} tokens: {:?}", generated.len(), generated);
    println!("decoded: {:?}", text);

    let mut tokens = prompt_tokens.clone();
    tokens.extend_from_slice(&generated);
    let full = tokenizer.decode(&tokens).unwrap_or_default();
    println!("full text: {:?}", full);

    if let Some(q) = quantize {
        println!("quantized: {q}");
    }
    println!("OK\n");
    Ok(())
}
