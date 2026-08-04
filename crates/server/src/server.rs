//! Axum router and request handlers for the BitLLM inference server.

use crate::api::*;
use crate::loader::{load_model, ModelLoadOptions};
use crate::metrics::Metrics;
use crate::worker::{InferenceWorker, WorkerError};
use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::State,
    http::{header, StatusCode},
    response::sse::{Event, KeepAlive, Sse},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use bitllm_tokenizer::BpeTokenizer;
use bitllm_tensor::Device;
use futures::stream::Stream;
use futures::stream::StreamExt;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;
use tokio_stream::wrappers::ReceiverStream;

pub struct AppState {
    pub worker: InferenceWorker,
    pub tokenizer: Arc<BpeTokenizer>,
    pub metrics: Arc<Metrics>,
    pub model_name: Arc<tokio::sync::RwLock<String>>,
    pub model_source: Arc<tokio::sync::RwLock<String>>,
}

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(handle_chat_completion))
        .route("/v1/completions", post(handle_completion))
        .route("/v1/models", get(handle_list_models))
        .route("/v1/model", get(handle_get_model).post(handle_swap_model))
        .route("/v1/ws", get(handle_ws))
        .route("/health", get(handle_health))
        .route("/metrics", get(handle_metrics))
        .with_state(state)
}

/// Records request count, in-flight gauge, and latency histogram. Dropping the
/// guard (end of request, or end of the wrapped stream for SSE) records the
/// duration.
struct RequestGuard {
    metrics: Arc<Metrics>,
    start: Instant,
}

impl RequestGuard {
    fn new(metrics: &Arc<Metrics>, endpoint: &str) -> Self {
        metrics.inc_requests(endpoint);
        metrics.in_flight_inc();
        Self {
            metrics: Arc::clone(metrics),
            start: Instant::now(),
        }
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        self.metrics.in_flight_dec();
        self.metrics.observe_duration(self.start.elapsed().as_secs_f64());
    }
}

/// Wraps an SSE stream so the latency guard lives until the response finishes.
struct GuardedStream {
    inner: futures::stream::BoxStream<'static, Result<Event, Infallible>>,
    _guard: RequestGuard,
}

impl Stream for GuardedStream {
    type Item = Result<Event, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

fn error_response(message: &str) -> ErrorResponse {
    ErrorResponse {
        error: ErrorDetail {
            message: message.to_string(),
            error_type: "server_error".to_string(),
        },
    }
}

fn service_unavailable(e: WorkerError) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(error_response(&e.to_string())),
    )
}

async fn handle_chat_completion(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let prompt = messages_to_prompt(&request.messages);
    let max_tokens = request.max_tokens.unwrap_or(256);
    let temperature = request.temperature.unwrap_or(0.7);
    let top_k = request.top_k;
    let stream = request.stream.unwrap_or(false);

    let sampler = make_sampler(top_k, temperature);
    let tokens = state.tokenizer.encode_with_special(&prompt, true, false);

    if stream {
        let guard = RequestGuard::new(&state.metrics, "/v1/chat/completions");
        let token_rx = state
            .worker
            .generate_streaming(tokens, max_tokens, sampler)
            .await
            .map_err(service_unavailable)?;

        let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let model_name = state.model_name.read().await.clone();
        let done_model_name = model_name.clone();

        let stream = ReceiverStream::new(token_rx);
        let id_clone = id.clone();
        let tokenizer = Arc::clone(&state.tokenizer);
        let event_stream = stream.filter_map(move |token| {
            let id = id.clone();
            let model_name = model_name.clone();
            let tok = Arc::clone(&tokenizer);
            async move {
                let text = tok.decode(&[token]).unwrap_or_default();
                let chunk = ChatCompletionChunk {
                    id,
                    object: "chat.completion.chunk".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    model: model_name,
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: None,
                            content: Some(text),
                        },
                        finish_reason: None,
                    }],
                };
                let data = serde_json::to_string(&chunk).ok()?;
                Some(Ok::<_, Infallible>(Event::default().data(data)))
            }
        });

        let done_stream = futures::stream::once(async move {
            let chunk = ChatCompletionChunk {
                id: id_clone,
                object: "chat.completion.chunk".to_string(),
                created: chrono::Utc::now().timestamp(),
                model: done_model_name,
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some("stop".to_string()),
                }],
            };
            let data = serde_json::to_string(&chunk).unwrap_or_default();
            Ok::<_, Infallible>(Event::default().data(data))
        });

        let full_stream: futures::stream::BoxStream<'static, Result<Event, Infallible>> =
            Box::pin(event_stream.chain(done_stream));
        Ok(Sse::new(GuardedStream {
            inner: full_stream,
            _guard: guard,
        })
        .keep_alive(KeepAlive::default())
        .into_response())
    } else {
        let _guard = RequestGuard::new(&state.metrics, "/v1/chat/completions");
        let generated = state
            .worker
            .generate(tokens, max_tokens, sampler)
            .await
            .map_err(service_unavailable)?;

        let text = state.tokenizer.decode(&generated).unwrap_or_default();
        let prompt_tokens = state.tokenizer.encode(&prompt).len();

        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: state.model_name.read().await.clone(),
            choices: vec![Choice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: text,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens: generated.len(),
                total_tokens: prompt_tokens + generated.len(),
            },
        };

        Ok(Json(response).into_response())
    }
}

async fn handle_completion(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CompletionRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let max_tokens = request.max_tokens.unwrap_or(256);
    let temperature = request.temperature.unwrap_or(0.7);
    let top_k = request.top_k;
    let stream = request.stream.unwrap_or(false);

    let sampler = make_sampler(top_k, temperature);
    let tokens = state
        .tokenizer
        .encode_with_special(&request.prompt, true, false);

    if stream {
        let guard = RequestGuard::new(&state.metrics, "/v1/completions");
        let token_rx = state
            .worker
            .generate_streaming(tokens, max_tokens, sampler)
            .await
            .map_err(service_unavailable)?;

        let id = format!("cmpl-{}", uuid::Uuid::new_v4());
        let model_name = state.model_name.read().await.clone();
        let done_model_name = model_name.clone();

        let stream = ReceiverStream::new(token_rx);
        let id_clone = id.clone();
        let tokenizer = Arc::clone(&state.tokenizer);
        let event_stream = stream.filter_map(move |token| {
            let id = id.clone();
            let model_name = model_name.clone();
            let tok = Arc::clone(&tokenizer);
            async move {
                let text = tok.decode(&[token]).unwrap_or_default();
                let chunk = CompletionChunk {
                    id,
                    object: "text_completion".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    model: model_name,
                    choices: vec![CompletionChunkChoice {
                        index: 0,
                        text,
                        finish_reason: None,
                    }],
                };
                let data = serde_json::to_string(&chunk).ok()?;
                Some(Ok::<_, Infallible>(Event::default().data(data)))
            }
        });

        let done_stream = futures::stream::once(async move {
            let chunk = CompletionChunk {
                id: id_clone,
                object: "text_completion".to_string(),
                created: chrono::Utc::now().timestamp(),
                model: done_model_name,
                choices: vec![CompletionChunkChoice {
                    index: 0,
                    text: String::new(),
                    finish_reason: Some("stop".to_string()),
                }],
            };
            let data = serde_json::to_string(&chunk).unwrap_or_default();
            Ok::<_, Infallible>(Event::default().data(data))
        });

        let full_stream: futures::stream::BoxStream<'static, Result<Event, Infallible>> =
            Box::pin(event_stream.chain(done_stream));
        Ok(Sse::new(GuardedStream {
            inner: full_stream,
            _guard: guard,
        })
        .keep_alive(KeepAlive::default())
        .into_response())
    } else {
        let _guard = RequestGuard::new(&state.metrics, "/v1/completions");
        let generated = state
            .worker
            .generate(tokens, max_tokens, sampler)
            .await
            .map_err(service_unavailable)?;

        let text = state.tokenizer.decode(&generated).unwrap_or_default();
        let prompt_tokens = state.tokenizer.encode(&request.prompt).len();

        let response = CompletionResponse {
            id: format!("cmpl-{}", uuid::Uuid::new_v4()),
            object: "text_completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: state.model_name.read().await.clone(),
            choices: vec![CompletionChoice {
                index: 0,
                text,
                finish_reason: Some("stop".to_string()),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens: generated.len(),
                total_tokens: prompt_tokens + generated.len(),
            },
        };

        Ok(Json(response).into_response())
    }
}

async fn handle_list_models(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.model_name.read().await.clone(),
            "object": "model",
            "owned_by": "bitllm"
        }]
    }))
}

async fn handle_get_model(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "object": "model",
        "id": state.model_name.read().await.clone(),
        "source": state.model_source.read().await.clone(),
        "owned_by": "bitllm"
    }))
}

/// Hot-swap the inference model. The new weights are loaded off the async
/// runtime, then the worker replaces its model between requests and acks.
async fn handle_swap_model(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ModelSwapRequest>,
) -> Result<Json<Value>, (StatusCode, Json<ErrorResponse>)> {
    let has_source =
        request.gguf.is_some() || request.safetensors.is_some() || request.config.is_some();
    if !has_source {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(error_response("model swap requires gguf, safetensors, or config")),
        ));
    }

    let opts = ModelLoadOptions {
        gguf: request.gguf.clone(),
        safetensors: request.safetensors.clone(),
        config_json: request.config_json.clone(),
        config: request.config.clone().unwrap_or_else(|| "tiny".to_string()),
        quantize: request.quantize.clone(),
        device: Device::Cpu,
    };

    let (model, name, source) = tokio::task::spawn_blocking(move || {
        let loaded = load_model(&opts)?;
        anyhow::Ok((loaded.model, loaded.name, loaded.source))
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(error_response(&format!("model swap task failed: {}", e))),
        )
    })?
    .map_err(|e: anyhow::Error| {
        (
            StatusCode::BAD_REQUEST,
            Json(error_response(&format!("model load failed: {}", e))),
        )
    })?;

    state
        .worker
        .swap(model)
        .await
        .map_err(service_unavailable)?;

    let final_name = request.name.unwrap_or_else(|| name.clone());
    *state.model_name.write().await = final_name.clone();
    *state.model_source.write().await = source.clone();

    log::info!("Model hot-swapped: {} ({})", final_name, source);
    Ok(Json(json!({
        "status": "ok",
        "model": final_name,
        "source": source,
    })))
}

/// WebSocket streaming endpoint. The client sends one JSON frame
/// (`WsRequest`) and receives `{"type":"token","text":...}` frames followed by
/// a `{"type":"done",...}` or `{"type":"error",...}` frame.
async fn handle_ws(ws: WebSocketUpgrade, State(state): State<Arc<AppState>>) -> Response {
    ws.on_upgrade(move |socket| handle_ws_session(socket, state))
}

async fn handle_ws_session(mut socket: WebSocket, state: Arc<AppState>) {
    let _guard = RequestGuard::new(&state.metrics, "/v1/ws");

    let request: WsRequest = loop {
        match socket.recv().await {
            Some(Ok(Message::Text(txt))) => match serde_json::from_str(&txt) {
                Ok(req) => break req,
                Err(e) => {
                    let _ = send_ws_frame(&mut socket, "error", &format!("invalid request: {}", e)).await;
                    return;
                }
            },
            Some(Ok(Message::Close(_))) => return,
            Some(Ok(_)) => continue,
            Some(Err(e)) => {
                log::warn!("WebSocket receive error: {}", e);
                return;
            }
            None => return,
        }
    };

    let prompt = match (&request.prompt, &request.messages) {
        (Some(p), _) => p.clone(),
        (None, Some(messages)) => messages_to_prompt(messages),
        (None, None) => {
            let _ = send_ws_frame(&mut socket, "error", "expected `prompt` or `messages`").await;
            return;
        }
    };
    let max_tokens = request.max_tokens.unwrap_or(256);
    let temperature = request.temperature.unwrap_or(0.7);
    let sampler = make_sampler(request.top_k, temperature);
    let tokens = state.tokenizer.encode_with_special(&prompt, true, false);

    let token_rx = match state
        .worker
        .generate_streaming(tokens.clone(), max_tokens, sampler)
        .await
    {
        Ok(rx) => rx,
        Err(e) => {
            let _ = send_ws_frame(&mut socket, "error", &e.to_string()).await;
            return;
        }
    };

    let tokenizer = Arc::clone(&state.tokenizer);
    let mut emitted = 0usize;
    let mut token_rx = token_rx;
    while let Some(token) = token_rx.recv().await {
        emitted += 1;
        let text = tokenizer.decode(&[token]).unwrap_or_default();
        if send_ws_frame(&mut socket, "token", &text).await.is_err() {
            return;
        }
    }

    let done = json!({
        "type": "done",
        "finish_reason": "stop",
        "usage": {
            "prompt_tokens": tokens.len(),
            "completion_tokens": emitted,
        }
    });
    let _ = socket.send(Message::Text(done.to_string())).await;
}

async fn send_ws_frame(socket: &mut WebSocket, kind: &str, text: &str) -> Result<(), ()> {
    let frame = json!({ "type": kind, "text": text });
    socket
        .send(Message::Text(frame.to_string()))
        .await
        .map_err(|_| ())
}

async fn handle_health(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "queue_depth": state.worker.queue_depth(),
    }))
}

async fn handle_metrics(State(state): State<Arc<AppState>>) -> Response {
    let body = state.metrics.render();
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        body,
    )
        .into_response()
}

fn messages_to_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&msg.role);
        prompt.push_str("\n");
        prompt.push_str(&msg.content);
        prompt.push_str("\n");
    }
    prompt
}

fn make_sampler(top_k: Option<usize>, temperature: f32) -> bitllm_runtime::Sampler {
    if let Some(k) = top_k {
        bitllm_runtime::Sampler::top_k(k, temperature)
    } else if temperature <= 0.0 {
        bitllm_runtime::Sampler::greedy()
    } else {
        bitllm_runtime::Sampler::temperature(temperature)
    }
}
