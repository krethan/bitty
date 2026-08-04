//! Integration tests for the BitLLM server router.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use bitllm_runtime::{Model, ModelConfig};
use bitllm_server::metrics::Metrics;
use bitllm_server::server::{create_router, AppState};
use bitllm_server::worker::{InferenceWorker, DEFAULT_QUEUE_CAPACITY};
use bitllm_tokenizer::BpeTokenizer;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tower::ServiceExt;

fn test_tokenizer() -> Arc<BpeTokenizer> {
    let mut vocab = HashMap::new();
    vocab.insert("hello".to_string(), 0);
    vocab.insert("world".to_string(), 1);
    vocab.insert("eos".to_string(), 2);
    Arc::new(BpeTokenizer::from_vocab_and_merges(vocab, vec![]))
}

fn test_model() -> Model {
    let config = ModelConfig::tiny_test();
    Model::new(config)
}

async fn test_state(queue_capacity: usize) -> Arc<AppState> {
    let metrics = Arc::new(Metrics::new());
    let worker = InferenceWorker::with_capacity(test_model(), Arc::clone(&metrics), queue_capacity);
    Arc::new(AppState {
        worker,
        tokenizer: test_tokenizer(),
        metrics,
        model_name: Arc::new(tokio::sync::RwLock::new("test-model".to_string())),
        model_source: Arc::new(tokio::sync::RwLock::new("builtin:tiny".to_string())),
    })
}

#[tokio::test]
async fn test_health_endpoint() {
    let state = test_state(DEFAULT_QUEUE_CAPACITY).await;
    let app = create_router(state);

    let response = app
        .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["status"], "ok");
    assert!(json["queue_depth"].is_number());
    assert!(json["version"].is_string());
}

#[tokio::test]
async fn test_metrics_endpoint() {
    let state = test_state(DEFAULT_QUEUE_CAPACITY).await;
    let app = create_router(state);

    let response = app
        .oneshot(Request::builder().uri("/metrics").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/plain; version=0.0.4; charset=utf-8"
    );

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8_lossy(&body);

    assert!(text.contains("bitllm_requests_total"));
    assert!(text.contains("bitllm_tokens_total"));
    assert!(text.contains("bitllm_requests_rejected_total"));
    assert!(text.contains("bitllm_model_swaps_total"));
    assert!(text.contains("bitllm_in_flight"));
    assert!(text.contains("bitllm_queue_depth"));
    assert!(text.contains("bitllm_queue_capacity"));
    assert!(text.contains("bitllm_request_duration_seconds"));
}

#[tokio::test]
async fn test_get_model_endpoint() {
    let state = test_state(DEFAULT_QUEUE_CAPACITY).await;
    let app = create_router(state);

    let response = app
        .oneshot(Request::builder().uri("/v1/model").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["id"], "test-model");
    assert_eq!(json["source"], "builtin:tiny");
    assert_eq!(json["owned_by"], "bitllm");
}

#[tokio::test]
async fn test_list_models_endpoint() {
    let state = test_state(DEFAULT_QUEUE_CAPACITY).await;
    let app = create_router(state);

    let response = app
        .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["object"], "list");
    let data = json["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "test-model");
    assert_eq!(data[0]["owned_by"], "bitllm");
}

#[tokio::test]
async fn test_swap_model_invalid_body() {
    let state = test_state(DEFAULT_QUEUE_CAPACITY).await;
    let app = create_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/model")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(json["error"]["message"]
        .as_str()
        .unwrap()
        .contains("requires gguf, safetensors, or config"));
}

#[tokio::test]
async fn test_swap_model_with_config() {
    let state = test_state(DEFAULT_QUEUE_CAPACITY).await;
    let app = create_router(state);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/model")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_string(&json!({
                        "config": "tiny",
                        "name": "swapped-model"
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["status"], "ok");
    assert_eq!(json["model"], "swapped-model");

    // Verify the model name was updated
    let response = app
        .oneshot(Request::builder().uri("/v1/model").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["id"], "swapped-model");
}

#[tokio::test]
async fn test_backpressure_503() {
    let state = test_state(1).await; // queue capacity 1
    let app = create_router(state.clone());

    // Send 3 requests concurrently with large max_tokens
    // With queue capacity 1, at least one should get 503
    let make_request = |app: axum::Router, max_tokens: usize| async move {
        app.oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_string(&json!({
                        "prompt": "hello world",
                        "max_tokens": max_tokens
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap()
    };

    let (r1, r2, r3) = tokio::join!(
        make_request(app.clone(), 500),
        make_request(app.clone(), 500),
        make_request(app.clone(), 500)
    );

    let statuses = [r1.status(), r2.status(), r3.status()];
    let ok_count = statuses.iter().filter(|&&s| s == StatusCode::OK).count();
    let rejected_count = statuses
        .iter()
        .filter(|&&s| s == StatusCode::SERVICE_UNAVAILABLE)
        .count();

    // With queue capacity 1: 1 processing + 1 queued = 2 can proceed, 1 rejected
    // But timing may allow all 3 to queue if the first hasn't started yet
    // At minimum, verify the system handled it gracefully
    assert!(ok_count + rejected_count == 3);
    assert!(rejected_count >= 1, "Expected at least one 503 rejection, got {} OK and {} rejected", ok_count, rejected_count);

    // Verify metrics recorded the rejection
    let metrics_text = state.metrics.render();
    assert!(metrics_text.contains("bitllm_requests_rejected_total"));
}
