//! Model loading shared by the `serve` CLI path and the hot-swap endpoint.

use anyhow::Context;
use bitllm_runtime::model::Model;
use bitllm_runtime::{load_safetensors_weights, ModelConfig};

/// Sources and options for building a model.
pub struct ModelLoadOptions {
    pub gguf: Option<String>,
    pub safetensors: Option<String>,
    pub config_json: Option<String>,
    pub config: String,
    pub quantize: Option<String>,
    pub device: bitllm_tensor::Device,
}

/// A loaded model plus the id/name resolved from its metadata.
pub struct LoadedModel {
    pub model: Model,
    pub name: String,
    pub source: String,
}

/// Load a model from a GGUF file, a SafeTensors file, or a built-in config.
pub fn load_model(opts: &ModelLoadOptions) -> anyhow::Result<LoadedModel> {
    if let Some(gguf_path) = &opts.gguf {
        log::info!("Loading GGUF model from {}", gguf_path);
        let loader = bitllm_runtime::gguf::GgufLoader::load(gguf_path)
            .with_context(|| format!("Failed to load GGUF {}", gguf_path))?;
        let model_config = loader.config_from_metadata().ok_or_else(|| {
            anyhow::anyhow!("Could not extract config from GGUF metadata")
        })?;
        let name = loader
            .metadata_str("general.name")
            .map(|s| s.to_string())
            .unwrap_or_else(|| "bitllm-model".to_string());
        log::info!(
            "GGUF config: {} layers, {} hidden, {} heads",
            model_config.num_layers,
            model_config.hidden_size,
            model_config.num_heads
        );
        let mut model = Model::new(model_config.clone());
        load_gguf_weights(&mut model, &loader, opts.device, &model_config);
        Ok(LoadedModel {
            model,
            name,
            source: format!("gguf:{}", gguf_path),
        })
    } else if let Some(st_path) = &opts.safetensors {
        log::info!("Loading SafeTensors model from {}", st_path);
        let loader = bitllm_runtime::SafeTensorsLoader::load(st_path)
            .with_context(|| format!("Failed to load SafeTensors {}", st_path))?;

        let model_config = if let Some(cj_path) = &opts.config_json {
            let json = std::fs::read_to_string(cj_path)
                .with_context(|| format!("Failed to read {}", cj_path))?;
            ModelConfig::from_huggingface_json(&json)
                .map_err(|e| anyhow::anyhow!("Failed to parse config {}: {}", cj_path, e))?
        } else {
            let config_path = std::path::Path::new(st_path)
                .parent()
                .unwrap_or(std::path::Path::new("."))
                .join("config.json");
            if config_path.exists() {
                let json = std::fs::read_to_string(&config_path)
                    .with_context(|| format!("Failed to read {}", config_path.display()))?;
                ModelConfig::from_huggingface_json(&json)
                    .map_err(|e| anyhow::anyhow!("Failed to parse config {}: {}", config_path.display(), e))?
            } else {
                log::warn!("No config.json found, using tiny_test defaults");
                ModelConfig::tiny_test()
            }
        };

        let name = loader
            .metadata()
            .get("model_name")
            .cloned()
            .unwrap_or_else(|| "bitllm-model".to_string());

        log::info!(
            "SafeTensors config: {} layers, {} hidden, {} heads",
            model_config.num_layers,
            model_config.hidden_size,
            model_config.num_heads
        );

        let mut model = Model::new(model_config.clone());
        let stats = load_safetensors_weights(
            &mut model,
            &loader,
            &model_config,
            opts.quantize.as_deref(),
        );
        log::info!(
            "Loaded {} tensors, skipped {}",
            stats.loaded,
            stats.skipped.len()
        );
        if !stats.skipped.is_empty() {
            log::debug!("Skipped tensors: {:?}", stats.skipped);
        }
        Ok(LoadedModel {
            model,
            name,
            source: format!("safetensors:{}", st_path),
        })
    } else {
        let model_config = match opts.config.as_str() {
            "small" => ModelConfig::llama_small(),
            _ => ModelConfig::tiny_test(),
        };
        Ok(LoadedModel {
            model: Model::new(model_config),
            name: "bitllm-model".to_string(),
            source: format!("config:{}", opts.config),
        })
    }
}

pub fn load_gguf_weights(
    model: &mut Model,
    loader: &bitllm_runtime::gguf::GgufLoader,
    device: bitllm_tensor::Device,
    config: &ModelConfig,
) {
    use bitllm_runtime::gguf::{to_torch_layout, GgufWeightMapper};

    let mut lm_head_loaded = false;
    for name in loader.tensor_names() {
        let target = GgufWeightMapper::map_weight(name);
        if matches!(target, bitllm_runtime::WeightTarget::LmHead) {
            lm_head_loaded = true;
        }
        let tensor = match loader.load_tensor(name) {
            Ok(t) => t,
            Err(e) => {
                log::warn!("Failed to load tensor '{}': {}", name, e);
                continue;
            }
        };

        let tensor = to_torch_layout(tensor);
        let tensor = if device != bitllm_tensor::Device::Cpu {
            tag_tensor_device(tensor, device)
        } else {
            tensor
        };

        if !bitllm_runtime::apply_weight_target(model, &target, tensor) {
            log::debug!("Skipping tensor '{}'", name);
        }
    }

    // Handle tied word embeddings: if the config says to tie embeddings and
    // the model doesn't have a separate output weight, copy embedding to lm_head.
    if config.tie_word_embeddings || !lm_head_loaded {
        model.tie_embeddings();
    }
}

fn tag_tensor_device(
    tensor: bitllm_tensor::Tensor,
    device: bitllm_tensor::Device,
) -> bitllm_tensor::Tensor {
    use bitllm_tensor::Tensor;
    let mut t = Tensor::on_device(tensor.shape(), tensor.dtype(), device);
    t.data_mut().copy_from_slice(tensor.data());
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitllm_runtime::config::Architecture;
    use bitllm_runtime::gguf::GgufLoader;

    // Regression guard for the GGUF q/k layout bug: official Qwen2.5 (and
    // other non-interleaved-RoPE) GGUFs store `attn_q`/`attn_k` in plain torch
    // order. A mistaken RoPE-half "de-interleave" of these rows scrambled the
    // projections (ppl 5.77 -> 31K-63K). This test loads a synthetic Qwen-style
    // GGUF through the real server loader path and asserts q/k/v keep the exact
    // plain torch layout that llama.cpp converters emit.

    // --- minimal GGUF v3 writer (self-contained; mirrors runtime test helpers) ---
    fn w_u32(buf: &mut Vec<u8>, v: u32) {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    fn w_u64(buf: &mut Vec<u8>, v: u64) {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    fn w_str(buf: &mut Vec<u8>, s: &str) {
        w_u64(buf, s.len() as u64);
        buf.extend_from_slice(s.as_bytes());
    }

    fn w_kv_string(buf: &mut Vec<u8>, key: &str, val: &str) {
        w_str(buf, key);
        w_u32(buf, 8);
        w_str(buf, val);
    }

    fn w_kv_u64(buf: &mut Vec<u8>, key: &str, val: u64) {
        w_str(buf, key);
        w_u32(buf, 10);
        w_u64(buf, val);
    }

    fn w_kv_f32(buf: &mut Vec<u8>, key: &str, val: f32) {
        w_str(buf, key);
        w_u32(buf, 6);
        buf.extend_from_slice(&val.to_le_bytes());
    }

    fn w_kv_string_array(buf: &mut Vec<u8>, key: &str, items: &[&str]) {
        w_str(buf, key);
        w_u32(buf, 9); // array
        w_u32(buf, 8); // element type: string
        w_u64(buf, items.len() as u64);
        for s in items {
            w_str(buf, s);
        }
    }

    /// Build a Qwen-shaped GGUF v3, F32. `tensors` are
    /// `(name, ggml_dims, torch_row_major_data)` where `ggml_dims` is the
    /// declared GGUF dims (reverse of the logical torch shape).
    fn build_qwen_gguf(tensors: &[(&str, &[u64], &[f32])]) -> Vec<u8> {
        let mut buf = Vec::new();
        w_u32(&mut buf, 0x4655_4747); // GGUF magic
        w_u32(&mut buf, 3); // version

        w_u64(&mut buf, tensors.len() as u64);
        w_u64(&mut buf, 11); // metadata kv count

        w_kv_string(&mut buf, "general.architecture", "qwen2");
        w_kv_u64(&mut buf, "general.alignment", 32);
        w_kv_u64(&mut buf, "qwen2.block_count", 1);
        w_kv_u64(&mut buf, "qwen2.embedding_length", 8);
        w_kv_u64(&mut buf, "qwen2.attention.head_count", 2);
        w_kv_u64(&mut buf, "qwen2.attention.head_count_kv", 1);
        w_kv_u64(&mut buf, "qwen2.attention.key_length", 4);
        w_kv_u64(&mut buf, "qwen2.context_length", 32);
        w_kv_f32(&mut buf, "qwen2.layer_norm_rms_epsilon", 1e-5);
        w_kv_f32(&mut buf, "qwen2.rope.freq_base", 1e6);
        w_kv_string_array(
            &mut buf,
            "tokenizer.ggml.tokens",
            &["<unk>", "<s>", "</s>", "a", "b", "c", "d", "e"],
        );

        let mut data_offset = 0u64;
        for (name, dims, data) in tensors.iter().copied() {
            w_str(&mut buf, name);
            w_u32(&mut buf, dims.len() as u32);
            for &d in dims {
                w_u64(&mut buf, d);
            }
            w_u32(&mut buf, 0); // GgmlType::F32
            w_u64(&mut buf, data_offset);
            data_offset += (data.len() * 4) as u64;
        }
        let header_end = buf.len() as u64;
        let data_start = (header_end + 31) & !31;
        buf.extend(std::iter::repeat_n(0u8, (data_start - header_end) as usize));

        for (_name, _dims, data) in tensors.iter().copied() {
            for &val in data {
                buf.extend_from_slice(&val.to_le_bytes());
            }
        }
        buf
    }

    /// Deterministic row-distinct data so any row reorder is caught.
    fn seq(n: usize, base: f32) -> Vec<f32> {
        (0..n).map(|i| base + (i as f32) * 0.5).collect()
    }

    fn load_test_model(
        gguf_bytes: Vec<u8>,
    ) -> (Model, ModelConfig) {
        let dir = std::env::temp_dir().join("bitllm_server_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("qwen_regress_{:?}.gguf", std::thread::current().id()));
        std::fs::write(&path, &gguf_bytes).unwrap();
        let loader = GgufLoader::load(&path).unwrap();
        let config = loader
            .config_from_metadata()
            .expect("config from test GGUF metadata");
        assert_eq!(config.architecture, Architecture::Qwen2);
        assert_eq!(config.head_dim(), 4);
        let mut model = Model::new(config.clone());
        load_gguf_weights(&mut model, &loader, bitllm_tensor::Device::Cpu, &config);
        std::fs::remove_file(&path).ok();
        (model, config)
    }

    #[test]
    fn qwen_gguf_attn_projections_keep_plain_torch_layout() {
        // torch [out, in]: q/o = [8, 8] (2 heads x head_dim 4), k/v = [4, 8].
        // GGUF declares dims reversed (ggml ne order); data is torch row-major.
        let q = seq(8 * 8, 100.0);
        let k = seq(4 * 8, 300.0);
        let v = seq(4 * 8, 500.0);
        let o = seq(8 * 8, 700.0);
        let q_bias = seq(8, 900.0);
        let k_bias = seq(4, 950.0);
        let v_bias = seq(4, 980.0);
        let ffn_gate = seq(8 * 8, 2000.0);
        let ffn_down = seq(8 * 8, 3000.0);
        let ffn_up = seq(8 * 8, 4000.0);
        let attn_norm = seq(8, 5000.0);
        let ffn_norm = seq(8, 6000.0);
        let output_norm = seq(8, 7000.0);

        let gguf = build_qwen_gguf(&[
            ("blk.0.attn_q.weight", &[8, 8], &q),
            ("blk.0.attn_k.weight", &[8, 4], &k),
            ("blk.0.attn_v.weight", &[8, 4], &v),
            ("blk.0.attn_output.weight", &[8, 8], &o),
            ("blk.0.attn_q.bias", &[8], &q_bias),
            ("blk.0.attn_k.bias", &[4], &k_bias),
            ("blk.0.attn_v.bias", &[4], &v_bias),
            ("blk.0.ffn_gate.weight", &[8, 8], &ffn_gate),
            ("blk.0.ffn_down.weight", &[8, 8], &ffn_down),
            ("blk.0.ffn_up.weight", &[8, 8], &ffn_up),
            ("blk.0.attn_norm.weight", &[8], &attn_norm),
            ("blk.0.ffn_norm.weight", &[8], &ffn_norm),
            ("output_norm.weight", &[8], &output_norm),
        ]);

        let (model, _config) = load_test_model(gguf);
        let att = &model.layers[0].attention;

        assert_eq!(att.q_proj.weight.shape(), &[8, 8]);
        assert_eq!(att.q_proj.weight.as_f32_slice(), &q[..]);
        assert_eq!(att.k_proj.weight.shape(), &[4, 8]);
        assert_eq!(att.k_proj.weight.as_f32_slice(), &k[..]);
        assert_eq!(att.v_proj.weight.shape(), &[4, 8]);
        assert_eq!(att.v_proj.weight.as_f32_slice(), &v[..]);
        assert_eq!(att.o_proj.weight.shape(), &[8, 8]);
        assert_eq!(att.o_proj.weight.as_f32_slice(), &o[..]);

        assert_eq!(att.q_proj.bias.as_ref().unwrap().as_f32_slice(), &q_bias[..]);
        assert_eq!(att.k_proj.bias.as_ref().unwrap().as_f32_slice(), &k_bias[..]);
        assert_eq!(att.v_proj.bias.as_ref().unwrap().as_f32_slice(), &v_bias[..]);
    }

    #[test]
    fn qwen_gguf_round_trip_rejects_unneeded_qk_repack() {
        // Reproduce the interleaved q/k layout the old de-interleave expected
        // and assert the loader does NOT try to un-do it (real GGUFs are plain).
        let mut q_interleaved = seq(8 * 8, 100.0);
        let hd = 4usize;
        let n_heads = 2usize;
        let inner = 8usize;
        // torch row r -> interleaved row r_g, then restore torch order manually.
        let perm: Vec<usize> = (0..n_heads * hd)
            .map(|r| {
                let d_lo = r % (hd / 2);
                let d_hi = (r / (hd / 2)) % 2;
                hd * (r / hd) + 2 * d_lo + d_hi
            })
            .collect();
        let src = q_interleaved.clone();
        for r in 0..n_heads * hd {
            for c in 0..inner {
                q_interleaved[perm[r] * inner + c] = src[r * inner + c];
            }
        }

        let gguf = build_qwen_gguf(&[
            ("blk.0.attn_q.weight", &[8, 8], &q_interleaved),
            ("blk.0.attn_k.weight", &[8, 4], &seq(4 * 8, 300.0)),
            ("blk.0.attn_v.weight", &[8, 4], &seq(4 * 8, 500.0)),
            ("blk.0.attn_output.weight", &[8, 8], &seq(8 * 8, 700.0)),
        ]);

        let (model, _config) = load_test_model(gguf);
        // Whatever the file contains, it is stored verbatim (plain pass-through).
        assert_eq!(
            model.layers[0].attention.q_proj.weight.as_f32_slice(),
            &q_interleaved[..]
        );
    }

    #[test]
    fn real_gemma2_checkpoint_loads_zero_skips() {
        let dir = "/tmp/opencode/models/gemma2";
        if !std::path::Path::new(dir).join("model.safetensors").exists() {
            eprintln!("skipping: {} not present", dir);
            return;
        }
        let json = std::fs::read_to_string(format!("{}/config.json", dir)).unwrap();
        let config =
            bitllm_runtime::ModelConfig::from_huggingface_json(&json).unwrap();
        let loader =
            bitllm_runtime::SafeTensorsLoader::load(format!("{}/model.safetensors", dir))
                .unwrap();
        let mut model = Model::new(config.clone());
        let stats =
            bitllm_runtime::load_safetensors_weights(&mut model, &loader, &config, None);

        assert!(stats.skipped.is_empty(), "skipped: {:?}", stats.skipped);
        assert_eq!(stats.loaded, loader.tensor_names().len());

        assert!(config.post_ffn_norm, "gemma2 must use post-FFN norm");
        assert!(config.one_centered_norm, "gemma2 must use one-centered RMSNorm");
        assert_eq!(config.attn_logit_softcap, Some(50.0));
        assert_eq!(config.final_logit_softcap, Some(30.0));

        let logits = model.forward_slot(&[3, 4, 5], 0, None);
        assert_eq!(logits.shape(), &[3, config.vocab_size]);
        for &l in logits.as_f32_slice().iter().take(1000) {
            assert!(l.is_finite(), "non-finite logit {l}");
        }
    }

    #[test]
    fn real_phi_checkpoint_loads_zero_skips() {
        let dir = "/tmp/opencode/models/phi";
        if !std::path::Path::new(dir).join("model.safetensors").exists() {
            eprintln!("skipping: {} not present", dir);
            return;
        }
        let json = std::fs::read_to_string(format!("{}/config.json", dir)).unwrap();
        let config =
            bitllm_runtime::ModelConfig::from_huggingface_json(&json).unwrap();
        let loader =
            bitllm_runtime::SafeTensorsLoader::load(format!("{}/model.safetensors", dir))
                .unwrap();
        let mut model = Model::new(config.clone());
        let stats =
            bitllm_runtime::load_safetensors_weights(&mut model, &loader, &config, None);

        assert!(stats.skipped.is_empty(), "skipped: {:?}", stats.skipped);

        let logits = model.forward_slot(&[3, 4, 5], 0, None);
        assert_eq!(logits.shape(), &[3, config.vocab_size]);
        for &l in logits.as_f32_slice().iter().take(1000) {
            assert!(l.is_finite(), "non-finite logit {l}");
        }
    }

    #[test]
    fn real_llama_checkpoint_loads_zero_skips() {
        let dir = "/tmp/opencode/models/llama";
        if !std::path::Path::new(dir).join("model.safetensors").exists() {
            eprintln!("skipping: {} not present", dir);
            return;
        }
        let json = std::fs::read_to_string(format!("{}/config.json", dir)).unwrap();
        let config =
            bitllm_runtime::ModelConfig::from_huggingface_json(&json).unwrap();
        let loader =
            bitllm_runtime::SafeTensorsLoader::load(format!("{}/model.safetensors", dir))
                .unwrap();
        let mut model = Model::new(config.clone());
        let stats =
            bitllm_runtime::load_safetensors_weights(&mut model, &loader, &config, None);

        assert!(stats.skipped.is_empty(), "skipped: {:?}", stats.skipped);

        let logits = model.forward_slot(&[3, 4, 5], 0, None);
        assert_eq!(logits.shape(), &[3, config.vocab_size]);
        for &l in logits.as_f32_slice().iter().take(1000) {
            assert!(l.is_finite(), "non-finite logit {l}");
        }
    }

    #[test]
    fn real_smollm2_checkpoint_loads_zero_skips() {
        let dir = "/tmp/opencode/models/smollm2";
        if !std::path::Path::new(dir).join("model.safetensors").exists() {
            eprintln!("skipping: {} not present", dir);
            return;
        }
        let json = std::fs::read_to_string(format!("{}/config.json", dir)).unwrap();
        let config =
            bitllm_runtime::ModelConfig::from_huggingface_json(&json).unwrap();
        let loader =
            bitllm_runtime::SafeTensorsLoader::load(format!("{}/model.safetensors", dir))
                .unwrap();
        let mut model = Model::new(config.clone());
        let stats =
            bitllm_runtime::load_safetensors_weights(&mut model, &loader, &config, None);

        assert!(stats.skipped.is_empty(), "skipped: {:?}", stats.skipped);

        let logits = model.forward_slot(&[3, 4, 5], 0, None);
        assert_eq!(logits.shape(), &[3, config.vocab_size]);
        for &l in logits.as_f32_slice().iter().take(1000) {
            assert!(l.is_finite(), "non-finite logit {l}");
        }
    }

    fn load_real_gguf(path: &str) -> Option<(Model, ModelConfig, usize)> {
        let loader = GgufLoader::load(path).ok()?;
        let config = loader.config_from_metadata()?;
        let mut model = Model::new(config.clone());
        load_gguf_weights(&mut model, &loader, bitllm_tensor::Device::Cpu, &config);
        let n = loader.tensor_names().len();
        Some((model, config, n))
    }

    #[test]
    fn real_smollm2_gguf_loads_finite_logits() {
        let path = "/tmp/opencode/models/smollm2/SmolLM2-135M-Instruct-F16.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("skipping: {} not present", path);
            return;
        }
        let (mut model, config, _n) = load_real_gguf(path).unwrap();
        let logits = model.forward_slot(&[3, 4, 5], 0, None);
        assert_eq!(logits.shape(), &[3, config.vocab_size]);
        for &l in logits.as_f32_slice().iter().take(1000) {
            assert!(l.is_finite(), "non-finite logit {l}");
        }
    }

    #[test]
    fn real_qwen_gguf_loads_finite_logits() {
        let path = "/tmp/opencode/models/qwen25/qwen2.5-0.5b-instruct-fp16.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("skipping: {} not present", path);
            return;
        }
        let (mut model, config, _n) = load_real_gguf(path).unwrap();
        let logits = model.forward_slot(&[3, 4, 5], 0, None);
        assert_eq!(logits.shape(), &[3, config.vocab_size]);
        for &l in logits.as_f32_slice().iter().take(1000) {
            assert!(l.is_finite(), "non-finite logit {l}");
        }
    }
}
