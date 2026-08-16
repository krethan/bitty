//! Ablation: score text using ONLY embedding -> final norm -> lm_head (no
//! transformer layers). If this is already ~uniform, the embedding/lm_head
//! layout or final-norm is wrong. If it's reasonable, the fault is in the
//! transformer layers (attention/RoPE/FFN).

use bitllm_server::loader::{load_model, ModelLoadOptions};
use bitllm_tokenizer::BpeTokenizer;
use std::path::PathBuf;

const TEXT: &str = "The quick brown fox jumps over the lazy dog. The United Kingdom is a country in Europe that includes England, Scotland, Wales, and Northern Ireland. Machine learning is the study of computer algorithms that improve automatically through experience.";

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let first = args
        .next()
        .expect("usage: real_ppl_ablate <model.gguf | model_dir> <tokenizer.json>");
    let tok_path = PathBuf::from(args.next().expect("missing tokenizer.json"));

    let opts = if first.ends_with(".gguf") {
        ModelLoadOptions {
            gguf: Some(first),
            safetensors: None,
            config_json: None,
            config: "tiny".to_string(),
            quantize: None,
            device: bitllm_tensor::Device::Cpu,
        }
    } else {
        let dir = PathBuf::from(first);
        ModelLoadOptions {
            gguf: None,
            safetensors: Some(dir.join("model.safetensors").display().to_string()),
            config_json: Some(dir.join("config.json").display().to_string()),
            config: "tiny".to_string(),
            quantize: None,
            device: bitllm_tensor::Device::Cpu,
        }
    };
    let loaded = load_model(&opts)?;
    let model = &loaded.model;
    let tokenizer = BpeTokenizer::load(&tok_path)?;

    let tokens = tokenizer.encode_with_special(TEXT, true, false);
    let v = model.config.vocab_size;

    // embedding -> final norm -> lm_head only
    let hidden = model.embedding.forward(&tokens);
    let normed = model.norm.forward(&hidden);
    let logits = model.lm_head.forward(&normed);

    let s = logits.as_f32_slice();
    let mut nll = 0.0f64;
    let mut n = 0usize;
    for i in 1..tokens.len() {
        let row = &s[(i - 1) * v..i * v];
        let max = row.iter().cloned().fold(f32::MIN, f32::max);
        let sum: f64 = row.iter().map(|x| ((x - max) as f64).exp()).sum();
        let lse = (max as f64) + sum.ln();
        let lp = (row[tokens[i] as usize] as f64) - lse;
        nll += -lp;
        n += 1;
    }
    println!(
        "no-layer tokens={} ppl={:.2} (uniform would be {})",
        n,
        (nll / n as f64).exp(),
        v
    );
    Ok(())
}
