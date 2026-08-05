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

fn load_gguf_weights(
    model: &mut Model,
    loader: &bitllm_runtime::gguf::GgufLoader,
    device: bitllm_tensor::Device,
    config: &ModelConfig,
) {
    use bitllm_runtime::gguf::{to_torch_layout, uninterleave_rope_heads, GgufWeightMapper};

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
        // llama.cpp stores q/k rows RoPE-half interleaved for RoPE models; undo
        // it so the projections match torch (safetensors) layout. Non-RoPE
        // architectures (GPT-2, Phi) have plain q/k in GGUF.
        let tensor = match target {
            bitllm_runtime::WeightTarget::AttentionQ { .. }
            | bitllm_runtime::WeightTarget::AttentionK { .. }
                if config.use_rope =>
            {
                uninterleave_rope_heads(&tensor, config.head_dim())
            }
            _ => tensor,
        };
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
