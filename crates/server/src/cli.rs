use clap::{Parser, Subcommand};
use std::sync::Arc;

use crate::server::{create_router, AppState};
use crate::worker::InferenceWorker;
use bitllm_runtime::{Model, ModelConfig};

#[derive(Parser)]
#[command(
    name = "bitllm",
    version,
    about = "BitLLM - Optimized 1-bit LLM inference engine"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    Serve {
        #[arg(short, long, default_value = "127.0.0.1")]
        host: String,

        #[arg(short, long, default_value = "8080")]
        port: u16,

        #[arg(short, long)]
        model_name: Option<String>,

        #[arg(short, long)]
        tokenizer: Option<String>,

        #[arg(long, default_value = "tiny")]
        config: String,

        #[arg(long)]
        gguf: Option<String>,

        #[arg(long, default_value_t = 0)]
        gpu: i32,
    },
    Bench {
        #[arg(short, long, default_value = "tiny")]
        model: String,

        #[arg(short, long, default_value = "10")]
        iterations: usize,
    },
}

pub async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            host,
            port,
            model_name,
            tokenizer,
            config,
            gguf,
            gpu,
        } => {
            let device = if gpu >= 0 {
                if bitllm_rocm::is_available() {
                    log::info!("Using ROCm GPU device {}", gpu);
                    bitllm_tensor::Device::Gpu { device_id: gpu }
                } else {
                    log::warn!(
                        "GPU {} requested but ROCm not available, falling back to CPU",
                        gpu
                    );
                    bitllm_tensor::Device::Cpu
                }
            } else {
                bitllm_tensor::Device::Cpu
            };

            let (model, model_name_resolved) = if let Some(gguf_path) = &gguf {
                log::info!("Loading GGUF model from {}", gguf_path);
                let loader = bitllm_runtime::gguf::GgufLoader::load(gguf_path)
                    .map_err(|e| anyhow::anyhow!("Failed to load GGUF: {}", e))?;
                let model_config = loader.config_from_metadata().ok_or_else(|| {
                    anyhow::anyhow!("Could not extract config from GGUF metadata")
                })?;
                let name = model_name
                    .or_else(|| loader.metadata_str("general.name").map(|s| s.to_string()))
                    .unwrap_or_else(|| "bitllm-model".to_string());
                log::info!(
                    "GGUF config: {} layers, {} hidden, {} heads",
                    model_config.num_layers,
                    model_config.hidden_size,
                    model_config.num_heads
                );
                let mut model = Model::new(model_config);
                load_gguf_weights(&mut model, &loader, device);
                (model, name)
            } else {
                let model_config = match config.as_str() {
                    "small" => ModelConfig::llama_small(),
                    _ => ModelConfig::tiny_test(),
                };
                let name = model_name.unwrap_or_else(|| "bitllm-model".to_string());
                (Model::new(model_config), name)
            };

            let tok = match tokenizer {
                Some(path) => {
                    log::info!("Loading tokenizer from {}", path);
                    bitllm_tokenizer::BpeTokenizer::load(&path)?
                }
                None => {
                    log::info!("No tokenizer specified, using byte-level fallback");
                    create_byte_tokenizer()
                }
            };

            let state = Arc::new(AppState {
                worker: InferenceWorker::new(model),
                tokenizer: Arc::new(tok),
                model_name: model_name_resolved,
            });

            let router = create_router(state);
            let addr = format!("{}:{}", host, port);
            log::info!("BitLLM server starting on {}", addr);

            let listener = tokio::net::TcpListener::bind(&addr).await?;
            axum::serve(listener, router).await?;
        }
        Commands::Bench { model, iterations } => {
            println!(
                "Running benchmark: model={}, iterations={}",
                model, iterations
            );
            let config = match model.as_str() {
                "small" => ModelConfig::llama_small(),
                _ => ModelConfig::tiny_test(),
            };

            let mut m = Model::new(config);
            let sampler = bitllm_runtime::Sampler::greedy();
            let tokens = vec![0u32; 8];

            let start = std::time::Instant::now();
            for _ in 0..iterations {
                let _ = m.generate(&tokens, 32, &sampler);
            }
            let elapsed = start.elapsed();

            println!("Completed {} iterations in {:.2?}", iterations, elapsed);
            println!("Average: {:.2?} per iteration", elapsed / iterations as u32);
        }
    }

    Ok(())
}

fn create_byte_tokenizer() -> bitllm_tokenizer::BpeTokenizer {
    let mut vocab = std::collections::HashMap::new();
    for i in 0u32..256 {
        let ch = (i as u8) as char;
        vocab.insert(ch.to_string(), i);
    }
    bitllm_tokenizer::BpeTokenizer::from_vocab_and_merges(vocab, vec![])
}

fn load_gguf_weights(
    model: &mut bitllm_runtime::Model,
    loader: &bitllm_runtime::gguf::GgufLoader,
    device: bitllm_tensor::Device,
) {
    use bitllm_runtime::gguf::GgufWeightMapper;
    use bitllm_runtime::loader::WeightTarget;

    for name in loader.tensor_names() {
        let target = GgufWeightMapper::map_weight(name);
        let tensor = match loader.load_tensor(name) {
            Ok(t) => t,
            Err(e) => {
                log::warn!("Failed to load tensor '{}': {}", name, e);
                continue;
            }
        };

        let tensor = if device != bitllm_tensor::Device::Cpu {
            tag_tensor_device(tensor, device)
        } else {
            tensor
        };

        match target {
            WeightTarget::Embedding => {
                model.embedding.weight = tensor;
            }
            WeightTarget::FinalNorm => {
                model.norm.weight = tensor;
            }
            WeightTarget::LmHead => {
                model.lm_head.weight = tensor;
            }
            WeightTarget::AttentionQ { layer_idx } => {
                if let Some(layer) = model.layers.get_mut(layer_idx) {
                    layer.attention.q_proj.weight = tensor;
                }
            }
            WeightTarget::AttentionK { layer_idx } => {
                if let Some(layer) = model.layers.get_mut(layer_idx) {
                    layer.attention.k_proj.weight = tensor;
                }
            }
            WeightTarget::AttentionV { layer_idx } => {
                if let Some(layer) = model.layers.get_mut(layer_idx) {
                    layer.attention.v_proj.weight = tensor;
                }
            }
            WeightTarget::AttentionO { layer_idx } => {
                if let Some(layer) = model.layers.get_mut(layer_idx) {
                    layer.attention.o_proj.weight = tensor;
                }
            }
            WeightTarget::FfnGate { layer_idx } => {
                if let Some(layer) = model.layers.get_mut(layer_idx) {
                    layer.ffn_gate.weight = tensor;
                }
            }
            WeightTarget::FfnDown { layer_idx } => {
                if let Some(layer) = model.layers.get_mut(layer_idx) {
                    layer.ffn_down.weight = tensor;
                }
            }
            WeightTarget::FfnUp { layer_idx } => {
                if let Some(layer) = model.layers.get_mut(layer_idx) {
                    layer.ffn_up.weight = tensor;
                }
            }
            WeightTarget::AttnNorm { layer_idx } => {
                if let Some(layer) = model.layers.get_mut(layer_idx) {
                    layer.attn_norm.weight = tensor;
                }
            }
            WeightTarget::FfnNorm { layer_idx } => {
                if let Some(layer) = model.layers.get_mut(layer_idx) {
                    layer.ffn_norm.weight = tensor;
                }
            }
            WeightTarget::Unknown(unknown_name) => {
                log::debug!("Skipping unknown tensor: {}", unknown_name);
            }
        }
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
