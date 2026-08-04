# BitLLM Roadmap

## Improvement Plan (current)

Quality-first work layered on the original roadmap below. Each phase lands with
numbers from the perplexity harness (`run_benchmarks perplexity`) and must not
break `cargo test --workspace`.

- [x] **Phase 1 — W1A8 (ternary weights + int8 activations)**: `fused_bit1_int8_matmul` (per-token absmax int8 activation quant, i32 accumulation, XNOR+LUT), opt-in via `BitLinear.a8` / `Model::quantize_to_bit1_a8()`. Fixed a pre-existing `BitLinear` double-scale bug. Baseline: FP32 ppl 15.92, BIT1 ppl 102.68 (6.45x), BIT1-A8 ≈ BIT1 (one-hot-dominated activations quantize near-losslessly; A8 fidelity pinned at matmul level by Gaussian reference tests). `benchmarks/results/bitty_2026-07-31_20-48-16.{csv,json}`.
- [x] **Phase 2 — Top-k outlier channels**: `OutlierMap` on `QuantizedTensor` keeps the top `ceil(frac·n)` weights (ranked by |w|, default 1%) exact; both matmul kernels subtract the packed ternary sign and add the exact value back. Ternary scale is computed **excluding** outliers (otherwise one huge weight collapses the bulk to `±scale` — shown empirically to make outliers useless). Matmul-level fat-tailed test (>3x relative-RMSE improvement vs plain ternary at 100x tail); end-to-end BIT1-OL ppl 93.80 (5.89x) vs BIT1 102.68 (6.45x).
- [x] **Phase 3 — Cognition primitives**: `ContextMemory` (`bitllm-cognition`) implements the dense-window cutover — recent tokens stay dense, older tokens are evicted into ≤n*-item bundled records. Spike (`examples/spike_bundle_recall.rs`): superposition of 16-item records compresses ~16x at dims=1024 with recall@1 ≥ 0.95; cap scales ~linearly with dims. Integrated harness (`run_benchmarks perplexity`, "Far-Context Memory"): echo-ppl 70.05 (dense window only) → **1.000 with recall@1 16/16**, evicted history compressed 8.5x on a 272-token history. `benchmarks/results/bitty_2026-07-31_20-48-16.{csv,json}`.
- [x] **Phase 4 — TernaryLoRA training**: `bitllm-train` crate with a monotone ternary **block-coordinate-descent** trainer (`train_step`, exhaustive `3^rank` per-row search, no F32 shadow weights, no momentum/noise) and 2-bit packed storage (`{-1,0,+1}` trits + two fixed scales, ~15.4-15.7x vs an F32 LoRA of equal shape). `StochasticFlip`/`TrainingConfig` moved out of `hip_tern`; low-rank learnability + monotonicity unit tests pass. Integrated benchmark (`run_benchmarks train`, exports as `train` rows): the trained readout is evaluated on the model's *real* hidden states against the exact FP32 readout (ppl 15.919, matching the Phase 1 floor). **Structural negative result**: the bigram `logp` table is full-rank, so rank ≤ 8 ternary adapters cannot represent it — all trained readouts land above the uniform floor (best 261.4 at rank-4/scale-0.05; R4/R8 sweep 261-445, 15.36-15.67x compression). Documents when low-rank ternary readout replacement fails and why; trainer correctness is pinned by the low-rank unit test.
- [x] **Phase 5 — Group-wise ternary scales**: `QuantConfig.group_size` (multiple of 8, `0` = legacy global scale), `quantize_grouped_with_outliers` (per-group absmax along `k`, scales shared across rows, `scales.len() = ceil(k/gs)`), grouped variants of both fused matmul kernels (per-chunk scale inline for f32, per-group i32 flush for int8), outlier correction uses the outlier's group scale. Runtime: `Model::quantize_to_bit1_grouped(gs)` / `_a8(gs)`. Tests: grouped matches manual-unpacked reference (f32 + int8, 7 size pairs), grouped quantize roundtrip, fat-tailed groups (100x ratio) grouped ≤ global, homogeneous grouped ≈ global. All 168 workspace tests pass.
- [x] **Phase 5 — Quantization-aware training**: full-graph **STE-QAT** in `bitllm-train` (`crates/train/src/qat.rs`). The deployed quantized graph is the student, a frozen FP32 model the teacher; `L = mean (logits_qat − logits_fp32)²` at the model output (a per-projection reconstruction objective is provably identical to naive rounding, so it cannot help). Backward walks the full graph with STE (`∂Q/∂W ≜ 1`) — exact RMSNorm, SDPA, RoPE, SiLU backwards, pinned by finite-difference tests; only the seven projections per layer train. Benchmark (`run_benchmarks qat`, exports as `qat` rows): held-out eval logit-MSE drops for all four formats — BIT1 47.63→26.60 (0.56x), BIT1-A8 47.82→25.90 (0.54x), BIT1-OL 45.72→16.46 (0.36x), BIT1-OL-A8 45.81→17.91 (0.39x); ppl improves 3/4 formats (BIT1 102.7→52.0, BIT1-A8 102.7→63.9, BIT1-OL-A8 93.8→47.5), while BIT1-OL degrades 93.8→109.4 — a sharp-metric artifact (ppl only scores the target logit; QAT calibrates mean logit fidelity). All 17 `bitllm-train` tests pass; full workspace green.
## Phase 0: Correctness & Quality Harness

- [x] Fix `MercurialModel::init_weights` quantize-on-clone bug (weights were packed unquantized)
- [x] Fix `WeightStreamer` buffer sizing (packed 2-bit data needs 2 bits/weight, not 1)
- [x] Round-trip unit tests for the exact 2-bit packed ternary representation
- [x] Perplexity benchmark harness (`benchmarks/src/perplexity.rs`, synthetic bigram model + corpus, FP32 vs bit1, exports to `benchmarks/results/`)
- [x] README crate table update (9 crates, honest experimental status for rocm/hip_tern/cognition)

## Phase 1: Core Engine (Completed)

- [x] Tensor library (F32 activations + packed BIT1 weights)
- [x] Ternary (1-bit) quantization with absmax scaling
- [x] Fused 1-bit matmul (XNOR + LUT kernel)
- [x] BPE tokenizer
- [x] LLaMA transformer architecture
- [x] KV-cache for autoregressive generation
- [x] Multiple sampling strategies
- [x] OpenAI-compatible API server
- [x] Tensor parallelism primitives
- [x] Comprehensive test suite

## Phase 2: Model Loading & Real Inference

- [x] SafeTensors/GGUF weight loading (F16/BF16/INT8/INT4 → F32 at load)
- [x] JSON model config parsing
- [x] Weight quantization on load (`--quantize ternary`)
- [x] Streaming token generation (SSE `chat.completion.chunk`)
- [x] Tokenizer file loading (BPE merges via `tokenizer.json`)
- [x] Batch inference support (`Model::generate_batch`, batched decode KV cache)

## Phase 3: Performance Optimization

- [x] SIMD-accelerated quantization kernels (AVX2/NEON)
- [x] Parallel matmul with rayon thread pool
- [x] Memory-mapped weight loading
- [x] Prefill/decode separation
- [x] Continuous batching
- [x] Speculative decoding

## Phase 4: GPU Acceleration

- [ ] AMD ROCm backend
- [ ] GPU tensor operations
- [ ] GPU quantized matmul kernels
- [ ] Unified CPU/GPU memory management
- [ ] Multi-GPU tensor parallelism with NCCL

## Phase 5: Advanced 1-bit Techniques

- [x] Group-wise 1-bit quantization (finer-grained scales)
- [x] BitNet-style activation quantization (W1A8) — now the default quantized path (`QuantConfig.a8`, serde-default `true`); exact f32-activation kernel retained via `.without_a8()`
- [x] Mixed-precision residual paths — SubLN (`ModelConfig.sub_ln`): block input `x - RMSNorm(x)`, residual add kept in f32
- [x] Quantization-aware training support — full-graph STE-QAT (see the improvement plan bullet above)

## Phase 6: Production Features

- [x] Prometheus metrics endpoint (`/metrics`, text exposition v0.0.4: per-endpoint request counter, token counter, rejection counter, swap counter, in-flight gauge, queue-depth/capacity gauges, latency histogram)
- [x] Request queuing and backpressure (bounded `mpsc` queue, `try_send` → 503 on full queue)
- [x] WebSocket streaming (`/v1/ws`, JSON frame protocol with token/done/error messages)
- [x] Model hot-swapping (`POST /v1/model` loads weights off-runtime via `spawn_blocking`, worker `Swap` with ack)
- [x] Graceful shutdown (SIGINT/SIGTERM drain via `axum::serve(...).with_graceful_shutdown`)
- [x] Docker deployment (`Dockerfile` multi-stage, `.dockerignore`, `docker-compose.yml` with optional Prometheus)

## Phase 6.5: QAT Hardening

- [x] Hyperparameter sweep (lr × steps grid per format; identified optimal configs)
- [x] Improved default QATConfig (lr 0.05 → 0.02; BIT1-OL ppl regression fixed: 109.39 → 62.33)
- [x] Gradient clipping (global L2 norm clipping to prevent gradient explosion)
- [x] LR warmup (linear warmup from 0 to lr over configurable steps)
- [x] Cosine LR decay (smooth LR decay after warmup for fine-tuning)
- [x] Early stopping (eval on held-out window, stop when MSE plateaus)
- [x] Per-projection ablation studies (attention projections benefit most: attn_only gives 64.13% MSE improvement vs all 60.73%)

## Phase 7: Model Support

- [x] Tied word embeddings (LLaMA-2 7B, Gemma, T5)
- [x] RoPE scaling for extended context (LLaMA-2/3 linear/dynamic)
- [x] SentencePiece tokenizer support (.model file parsing)
- [ ] Architecture enum (Llama, Mistral, Gpt2, Phi, Qwen)
- [ ] End-to-end test with real model fixture
- [ ] LLaMA 2/3 full support
- [ ] Mistral/Mixtral support
- [ ] Phi support
- [ ] Gemma support
- [ ] Qwen support
- [ ] Mistral/Mixtral
- [ ] Phi-2
- [ ] GPT-2
- [ ] Qwen
- [ ] Custom model support
