use bitllm_server::loader::{load_model, ModelLoadOptions};
use bitllm_tensor::Tensor;
use std::path::PathBuf;

fn dump(name: &str, t: &Tensor, dim: usize, rows: usize) {
    let s = t.as_f32_slice();
    for r in 0..rows.min(t.shape()[0]) {
        let row = &s[r * dim..(r + 1) * dim];
        let maxa = row.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let first = &row[..8.min(dim)];
        println!("{}-row{} max_abs={:.6} first8={:?}", name, r, maxa, first);
    }
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let first = args.next().expect("usage: probe_weights <model_dir | model.gguf>");
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
    let h = model.config.hidden_size;
    let hd = model.config.head_dim();

    println!("CONFIG hidden={} layers={} heads={} kv={} inter={} rope_theta={} max_seq={} head_dim={}",
        model.config.hidden_size, model.config.num_layers, model.config.num_heads,
        model.config.num_kv_heads(), model.config.intermediate_size,
        model.config.rope_theta, model.config.max_seq_len, hd);

    let tokens = [1u32, 2, 3, 4];
    let hidden0 = model.embedding.forward(&tokens);
    dump("emb", &hidden0, h, 1);

    let layer = &model.layers[0];
    let normed = layer.attn_norm.forward(&hidden0);
    let attn_out = layer.attention.forward(&normed, None, 0, 0);
    dump("attn-out", &attn_out, h, 4);

    let layer_out = model.layers[0].forward(&hidden0, None, 0, 0);
    dump("layer0-out", &layer_out, h, 4);

    let mut l1in = hidden0.clone();
    for (i, l) in model.layers.iter().enumerate() {
        l1in = l.forward(&l1in, None, i, 0);
        if i == 1 {
            dump("layer1-out", &l1in, h, 1);
        }
        if i == 29 {
            dump("layer29-out", &l1in, h, 1);
        }
    }
    let final_normed = model.norm.forward(&l1in);
    dump("final_normed", &final_normed, h, 4);
    let manual_logits = model.lm_head.forward(&final_normed);
    let ms = manual_logits.as_f32_slice();
    let mmax = ms.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    println!("manual logits max_abs={}", mmax);

    let logits = model.forward(&tokens);
    let s = logits.as_f32_slice();
    let lmax = s.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    println!("full logits max_abs={}", lmax);
    let v = model.config.vocab_size;
    for r in 0..tokens.len() {
        let row = &ms[r * v..(r + 1) * v];
        let mx = row.iter().cloned().fold(f32::MIN, f32::max);
        let mut top: Vec<(f32, usize)> = row.iter().enumerate().map(|(i, x)| (*x, i)).collect();
        top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        println!("manual row{} max={:.4} top3={:?}", r, mx, &top[..3]);
    }
    let v = model.config.vocab_size;
    let mut top: Vec<(f32, usize)> = (0..v).map(|i| (s[3 * v + i], i)).collect();
    top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("row3 top5 logits: {:?}", &top[..5]);
    Ok(())
}
