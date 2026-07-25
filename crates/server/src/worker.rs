use bitllm_runtime::Model;
use bitllm_runtime::Sampler;
use tokio::sync::{mpsc, oneshot};

pub enum InferenceRequest {
    Generate {
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        sampler: Sampler,
        response_tx: oneshot::Sender<Vec<u32>>,
    },
    GenerateStreaming {
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        sampler: Sampler,
        token_tx: mpsc::Sender<u32>,
    },
}

#[derive(Clone)]
pub struct InferenceWorker {
    request_tx: mpsc::Sender<InferenceRequest>,
}

impl InferenceWorker {
    pub fn new(model: Model) -> Self {
        let (tx, mut rx) = mpsc::channel::<InferenceRequest>(64);

        tokio::spawn(async move {
            let mut model = model;

            while let Some(req) = rx.recv().await {
                match req {
                    InferenceRequest::Generate {
                        prompt_tokens,
                        max_tokens,
                        sampler,
                        response_tx,
                    } => {
                        let result = model.generate(&prompt_tokens, max_tokens, &sampler);
                        let _ = response_tx.send(result);
                    }
                    InferenceRequest::GenerateStreaming {
                        prompt_tokens,
                        max_tokens,
                        sampler,
                        token_tx,
                    } => {
                        model.generate_streaming(&prompt_tokens, max_tokens, &sampler, token_tx);
                    }
                }
            }

            log::info!("Inference worker shutting down (all senders dropped)");
        });

        Self { request_tx: tx }
    }

    pub async fn generate(
        &self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        sampler: Sampler,
    ) -> Vec<u32> {
        let (response_tx, response_rx) = oneshot::channel();

        let _ = self
            .request_tx
            .send(InferenceRequest::Generate {
                prompt_tokens,
                max_tokens,
                sampler,
                response_tx,
            })
            .await;

        response_rx.await.unwrap_or_default()
    }

    pub async fn generate_streaming(
        &self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        sampler: Sampler,
    ) -> mpsc::Receiver<u32> {
        let (token_tx, token_rx) = mpsc::channel(64);

        let _ = self
            .request_tx
            .send(InferenceRequest::GenerateStreaming {
                prompt_tokens,
                max_tokens,
                sampler,
                token_tx,
            })
            .await;

        token_rx
    }

    pub fn queue_depth(&self) -> usize {
        self.request_tx.max_capacity() - self.request_tx.capacity()
    }
}
