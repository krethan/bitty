//! Single-inference-thread worker with a bounded request queue.
//!
//! The worker owns the [`Model`] and processes requests serially (generation is
//! a synchronous, memory-bound loop). Requests are enqueued on a bounded
//! channel; when the queue is full the sender uses `try_send` and the caller
//! surfaces a 503 — explicit backpressure instead of unbounded queuing.
//! A [`InferenceRequest::Swap`] hot-swaps the model between requests.

use crate::metrics::Metrics;
use bitllm_runtime::Model;
use bitllm_runtime::Sampler;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

/// Default bound on the request queue.
pub const DEFAULT_QUEUE_CAPACITY: usize = 64;

#[derive(Debug, thiserror::Error, Clone, Copy)]
pub enum WorkerError {
    #[error("inference queue is full (max {0} pending)")]
    QueueFull(usize),
}

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
    Swap {
        model: Box<Model>,
        ack: oneshot::Sender<()>,
    },
}

#[derive(Clone)]
pub struct InferenceWorker {
    request_tx: mpsc::Sender<InferenceRequest>,
    metrics: Arc<Metrics>,
    queue_capacity: usize,
}

impl InferenceWorker {
    pub fn new(model: Model, metrics: Arc<Metrics>) -> Self {
        Self::with_capacity(model, metrics, DEFAULT_QUEUE_CAPACITY)
    }

    /// Create a worker with a custom queue bound.
    pub fn with_capacity(model: Model, metrics: Arc<Metrics>, capacity: usize) -> Self {
        let capacity = capacity.max(1);
        let (tx, mut rx) = mpsc::channel::<InferenceRequest>(capacity);
        metrics.set_queue_capacity(capacity);
        let task_metrics = Arc::clone(&metrics);

        tokio::spawn(async move {
            let mut model = model;

            while let Some(req) = rx.recv().await {
                task_metrics.set_queue_depth(rx.capacity());
                match req {
                    InferenceRequest::Generate {
                        prompt_tokens,
                        max_tokens,
                        sampler,
                        response_tx,
                    } => {
                        let generated = model.generate(&prompt_tokens, max_tokens, &sampler);
                        task_metrics.inc_tokens(generated.len() as u64);
                        let _ = response_tx.send(generated);
                    }
                    InferenceRequest::GenerateStreaming {
                        prompt_tokens,
                        max_tokens,
                        sampler,
                        token_tx,
                    } => {
                        let mut count = 0u64;
                        model.generate_streaming(
                            &prompt_tokens,
                            max_tokens,
                            &sampler,
                            token_tx,
                            &mut count,
                        );
                        task_metrics.inc_tokens(count);
                    }
                    InferenceRequest::Swap { model: new_model, ack } => {
                        model = *new_model;
                        task_metrics.inc_swaps();
                        log::info!("Inference worker: model hot-swapped");
                        let _ = ack.send(());
                    }
                }
            }

            task_metrics.set_queue_depth(0);
            log::info!("Inference worker shutting down (all senders dropped)");
        });

        Self {
            request_tx: tx,
            metrics,
            queue_capacity: capacity,
        }
    }

    fn enqueue(&self, req: InferenceRequest) -> Result<(), WorkerError> {
        match self.request_tx.try_send(req) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.metrics.inc_rejected();
                Err(WorkerError::QueueFull(self.queue_capacity))
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                Err(WorkerError::QueueFull(self.queue_capacity))
            }
        }
    }

    pub async fn generate(
        &self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        sampler: Sampler,
    ) -> Result<Vec<u32>, WorkerError> {
        let (response_tx, response_rx) = oneshot::channel();
        self.enqueue(InferenceRequest::Generate {
            prompt_tokens,
            max_tokens,
            sampler,
            response_tx,
        })?;
        Ok(response_rx.await.unwrap_or_default())
    }

    pub async fn generate_streaming(
        &self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        sampler: Sampler,
    ) -> Result<mpsc::Receiver<u32>, WorkerError> {
        let (token_tx, token_rx) = mpsc::channel(64);
        self.enqueue(InferenceRequest::GenerateStreaming {
            prompt_tokens,
            max_tokens,
            sampler,
            token_tx,
        })?;
        Ok(token_rx)
    }

    pub fn queue_depth(&self) -> usize {
        self.request_tx.max_capacity() - self.request_tx.capacity()
    }

    /// Hot-swap the inference model. Blocks until the worker has swapped it
    /// (i.e. after any in-flight request finishes) or the queue is full.
    pub async fn swap(&self, model: Model) -> Result<(), WorkerError> {
        let (ack_tx, ack_rx) = oneshot::channel();
        self.enqueue(InferenceRequest::Swap {
            model: Box::new(model),
            ack: ack_tx,
        })?;
        let _ = ack_rx.await;
        Ok(())
    }
}
