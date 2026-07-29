use crate::api::*;
use crate::worker::InferenceWorker;
use axum::{
    extract::State,
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use bitllm_tokenizer::BpeTokenizer;
use futures::stream::StreamExt;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

pub struct AppState {
    pub worker: InferenceWorker,
    pub tokenizer: Arc<BpeTokenizer>,
    pub model_name: String,
}

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(handle_chat_completion))
        .route("/v1/completions", post(handle_completion))
        .route("/v1/models", get(handle_list_models))
        .route("/health", get(handle_health))
        .with_state(state)
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
        let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let model_name = state.model_name.clone();
        let done_model_name = model_name.clone();

        let token_rx = state
            .worker
            .generate_streaming(tokens, max_tokens, sampler)
            .await;

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

        let full_stream = event_stream.chain(done_stream);
        Ok(Sse::new(full_stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        let generated = state.worker.generate(tokens, max_tokens, sampler).await;

        let text = state.tokenizer.decode(&generated).unwrap_or_default();
        let prompt_tokens = state.tokenizer.encode(&prompt).len();

        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: state.model_name.clone(),
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
        let id = format!("cmpl-{}", uuid::Uuid::new_v4());
        let model_name = state.model_name.clone();
        let done_model_name = model_name.clone();

        let token_rx = state
            .worker
            .generate_streaming(tokens, max_tokens, sampler)
            .await;

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

        let full_stream = event_stream.chain(done_stream);
        Ok(Sse::new(full_stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        let generated = state.worker.generate(tokens, max_tokens, sampler).await;

        let text = state.tokenizer.decode(&generated).unwrap_or_default();
        let prompt_tokens = state.tokenizer.encode(&request.prompt).len();

        let response = CompletionResponse {
            id: format!("cmpl-{}", uuid::Uuid::new_v4()),
            object: "text_completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: state.model_name.clone(),
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
            "id": state.model_name,
            "object": "model",
            "owned_by": "bitllm"
        }]
    }))
}

async fn handle_health(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "queue_depth": state.worker.queue_depth(),
    }))
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
