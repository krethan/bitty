//! Load a real HuggingFace model directory through the server's exact
//! `load_model` path and run greedy generation.
//!
//! Usage:
//!   cargo run -p bitllm-server --example real_model_check -- /path/to/model/dir [--quantize ternary] [--prompt "hello"]
//!   cargo run -p bitllm-server --example real_model_check -- --gguf /path/to/model.gguf [--tokenizer /path/to/tokenizer.json] [--quantize ternary] [--prompt "hello"]
//!
//! The directory must contain `config.json` and `model.safetensors`; a
//! `tokenizer.json` is used when present (otherwise a byte fallback).

use bitllm_runtime::Sampler;
use bitllm_server::loader::{load_model, ModelLoadOptions};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut args = std::env::args().skip(1);
    let mut quantize = None;
    let mut prompt = "The quick brown fox jumps over the lazy dog".to_string();
    let mut gguf: Option<PathBuf> = None;
    let mut tokenizer_path: Option<PathBuf> = None;
    let mut dir: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--quantize" => quantize = args.next(),
            "--prompt" => prompt = args.next().unwrap_or(prompt),
            "--gguf" => gguf = args.next().map(PathBuf::from),
            "--tokenizer" => tokenizer_path = args.next().map(PathBuf::from),
            other if dir.is_none() => dir = Some(PathBuf::from(other)),
            other => anyhow::bail!("unknown arg: {other}"),
        }
    }

    let opts = if let Some(gguf_path) = gguf {
        anyhow::ensure!(gguf_path.exists(), "missing {}", gguf_path.display());
        ModelLoadOptions {
            gguf: Some(gguf_path.display().to_string()),
            safetensors: None,
            config_json: None,
            config: "tiny".to_string(),
            quantize: quantize.clone(),
            device: bitllm_tensor::Device::Cpu,
        }
    } else {
        let dir = dir.ok_or_else(|| {
            anyhow::anyhow!("usage: real_model_check <model_dir> or --gguf <file>")
        })?;
        let safetensors = dir.join("model.safetensors");
        let config_json = dir.join("config.json");
        anyhow::ensure!(safetensors.exists(), "missing {}", safetensors.display());
        anyhow::ensure!(config_json.exists(), "missing {}", config_json.display());
        if tokenizer_path.is_none() {
            let tok = dir.join("tokenizer.json");
            if tok.exists() {
                tokenizer_path = Some(tok);
            }
        }
        ModelLoadOptions {
            gguf: None,
            safetensors: Some(safetensors.display().to_string()),
            config_json: Some(config_json.display().to_string()),
            config: "tiny".to_string(),
            quantize: quantize.clone(),
            device: bitllm_tensor::Device::Cpu,
        }
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
    println!(
        "forward logits shape={:?} finite={}",
        logits.shape(),
        is_finite
    );
    anyhow::ensure!(
        is_finite,
        "non-finite logits — forward pass broken for this architecture"
    );

    // Tokenize + generate (mirrors the server worker path).
    let tokenizer = if let Some(tok) = tokenizer_path {
        println!("tokenizer: {}", tok.display());
        bitllm_tokenizer::BpeTokenizer::load(&tok)?
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
    println!(
        "prompt tokens ({}): {:?}",
        prompt_tokens.len(),
        prompt_tokens
    );
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
