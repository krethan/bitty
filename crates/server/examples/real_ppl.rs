//! Quick real-model correctness probe: report cross-entropy / perplexity of a
//! fixed natural-language paragraph. Broken weight layouts blow up ppl to the
//! vocab size (uniform); a correctly-loaded model lands far below that.

use bitllm_server::loader::{load_model, ModelLoadOptions};
use bitllm_tokenizer::BpeTokenizer;
use std::path::PathBuf;

const TEXT: &str = "The quick brown fox jumps over the lazy dog. The United Kingdom is a country in Europe that includes England, Scotland, Wales, and Northern Ireland. Machine learning is the study of computer algorithms that improve automatically through experience.";

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let first = args.next().expect("usage: real_ppl <model.gguf | model_dir> <tokenizer.json>");
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
    let mut model = loaded.model;
    let tokenizer = BpeTokenizer::load(&tok_path)?;

    let tokens = tokenizer.encode_with_special(TEXT, true, false);
    let logits = model.forward(&tokens);
    let v = model.config.vocab_size;
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
    println!("tokens={} ppl={:.2} (uniform would be {})", n, (nll / n as f64).exp(), v);
    Ok(())
}
