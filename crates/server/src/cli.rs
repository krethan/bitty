use clap::{Parser, Subcommand};
use std::sync::Arc;

use crate::loader::{load_model, ModelLoadOptions};
use crate::metrics::Metrics;
use crate::server::{create_router, AppState};
use crate::worker::{InferenceWorker, DEFAULT_QUEUE_CAPACITY};
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

        #[arg(long)]
        safetensors: Option<String>,

        #[arg(long)]
        config_json: Option<String>,

        /// Optional on-load weight quantization: "ternary" (1-bit)
        #[arg(long)]
        quantize: Option<String>,

        #[arg(long, default_value_t = 0)]
        gpu: i32,

        /// Bounded inference queue depth; requests beyond this get a 503.
        #[arg(long, default_value_t = DEFAULT_QUEUE_CAPACITY)]
        queue_depth: usize,
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
            safetensors,
            config_json,
            quantize,
            gpu,
            queue_depth,
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

            let opts = ModelLoadOptions {
                gguf,
                safetensors,
                config_json,
                config,
                quantize,
                device,
            };
            let loaded = load_model(&opts)?;
            let model_name_resolved = model_name.unwrap_or(loaded.name);
            let source = loaded.source;

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

            let metrics = Arc::new(Metrics::new());
            let state = Arc::new(AppState {
                worker: InferenceWorker::with_capacity(
                    loaded.model,
                    Arc::clone(&metrics),
                    queue_depth,
                ),
                tokenizer: Arc::new(tok),
                metrics,
                model_name: Arc::new(tokio::sync::RwLock::new(model_name_resolved)),
                model_source: Arc::new(tokio::sync::RwLock::new(source)),
            });

            let router = create_router(state);
            let addr = format!("{}:{}", host, port);
            log::info!("BitLLM server starting on {}", addr);

            let listener = tokio::net::TcpListener::bind(&addr).await?;
            axum::serve(listener, router)
                .with_graceful_shutdown(shutdown_signal())
                .await?;
            log::info!("BitLLM server stopped");
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

/// Wait for SIGINT or SIGTERM, then let in-flight requests drain.
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => log::info!("SIGINT received, draining in-flight requests"),
        _ = terminate => log::info!("SIGTERM received, draining in-flight requests"),
    }
}

fn create_byte_tokenizer() -> bitllm_tokenizer::BpeTokenizer {
    let mut vocab = std::collections::HashMap::new();
    for i in 0u32..256 {
        let ch = (i as u8) as char;
        vocab.insert(ch.to_string(), i);
    }
    bitllm_tokenizer::BpeTokenizer::from_vocab_and_merges(vocab, vec![])
}
