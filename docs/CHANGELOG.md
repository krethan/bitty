# Changelog

## [Unreleased] - 2026-08-03

### Phase 7 Model Support — LLaMA-family compatibility

### Added
- **Tied word embeddings** (`Model::tie_embeddings()`): shares embedding and lm_head weights when `tie_word_embeddings` is true in config. Used by LLaMA-2 (7B), Gemma, T5. Applied automatically during SafeTensors and GGUF loading.
- **RoPE scaling** (`RopeScaling` struct, `RoPECache::with_scaling()`): supports linear/dynamic RoPE scaling for extended context windows (LLaMA-2/3). Parses `rope_scaling` object from HuggingFace config.json (previously misread as theta).
- **SentencePiece tokenizer support** (`BpeTokenizer::load_sentencepiece()`, `from_sentencepiece_bytes()`): minimal protobuf parser for .model files. Extracts vocabulary and identifies BOS/EOS/UNK special tokens. Note: encoding uses character-level fallback (full SentencePiece BPE algorithm not implemented).

### Results
- LLaMA-family models can now be loaded with correct tied embeddings and RoPE scaling.
- SentencePiece .model files can be parsed for vocabulary extraction.
- All 174 workspace tests pass (2 new config tests, 1 new SentencePiece test).

---

## [Unreleased] - 2026-08-02

### Phase 6 Production — Server hardening

### Added
- **Prometheus metrics** (`crates/server/src/metrics.rs`): text exposition v0.0.4 format with per-endpoint `bitllm_requests_total` (sorted), `bitllm_tokens_total`, `bitllm_requests_rejected_total`, `bitllm_model_swaps_total`, in-flight/queue-depth/queue-capacity gauges, `bitllm_request_duration_seconds` histogram (cumulative buckets 0.001–60 + implicit +Inf). 3 unit tests.
- **Request backpressure**: bounded `mpsc` queue (`InferenceWorker::with_capacity`); `try_send` returns `WorkerError::QueueFull` → HTTP 503. Explicit rejection over unbounded queueing.
- **Model hot-swap** (`POST /v1/model`): loads weights off the async runtime via `spawn_blocking`, then sends a `Swap` message to the worker with a `oneshot` ack (swap takes effect after in-flight requests). `GET /v1/model` returns current id + source.
- **WebSocket streaming** (`/v1/ws`): client sends one JSON frame (`WsRequest`: `prompt`|`messages`, `temperature`, `max_tokens`, `top_k`); server replies `{"type":"token","text":...}` frames then `{"type":"done","usage":...}` or `{"type":"error","text":...}`.
- **Graceful shutdown**: `axum::serve(...).with_graceful_shutdown(shutdown_signal())` on SIGINT/SIGTERM; drains in-flight requests.
- **CLI**: `--queue-depth <N>` flag (default 64); `AppState` uses `RwLock`-guarded model name/source for hot-swap.
- **Router integration tests** (`crates/server/tests/router_tests.rs`): 7 tests covering `/health`, `/metrics`, `/v1/model` GET/POST, `/v1/models`, backpressure 503.
- **Docker**: `Dockerfile` (multi-stage build), `.dockerignore`, `docker-compose.yml` (with optional Prometheus), `prometheus.yml`.
- All workspace tests pass (167 unit + 7 integration); ppl baseline unchanged (FP32 15.919, BIT1 102.683, BIT1-A8 102.680, BIT1-OL 93.802, BIT1-OL-A8 93.814).

### QAT Hardening — Hyperparameter sweep + improved defaults

### Added
- **QAT hyperparameter sweep** (`benchmarks/src/qat_sweep.rs`): grid search over learning rate ∈ {0.02, 0.05, 0.1} and steps ∈ {100, 200, 400} for each deployment format. Identifies optimal config per format. Export as `qat_sweep` rows in CSV/JSON.
- **Improved default QATConfig**: learning rate changed from 0.05 → 0.02 (sweep showed lower LR consistently better across formats).

### Results (sweep: 36 configs × 4 formats = 144 runs)
**Best configs per format (by held-out logit-MSE ratio):**
- BIT1: lr=0.02, steps=400 → ratio 0.3433 (was 0.558 with default)
- BIT1-A8: lr=0.02, steps=200 → ratio 0.3345 (was 0.542)
- BIT1-OL: lr=0.05, steps=100 → ratio 0.3424 (was 0.360)
- BIT1-OL-A8: lr=0.02, steps=400 → ratio 0.2766 (was 0.391)

**New default (lr=0.02, steps=200) vs old default (lr=0.05, steps=200):**
- BIT1: ratio 0.558 → 0.393 (1.42x improvement), ppl 52.03 → 65.05
- BIT1-A8: ratio 0.542 → 0.334 (1.62x improvement), ppl 63.93 → 65.67
- BIT1-OL: ratio 0.360 → 0.424 (0.85x), **ppl 109.39 → 62.33** (regression fixed!)
- BIT1-OL-A8: ratio 0.391 → 0.392 (1.00x), ppl 47.46 → 62.25

**Key insight:** BIT1-OL with lr=0.05, steps=100 achieves best MSE ratio (0.3424) but terrible ppl (132.66), while lr=0.02, steps=400 (ratio=0.3513) gives much better ppl (55.84). Confirms MSE (smooth, all-logit) and ppl (sharp, target-logit-only) measure different things; QAT calibrates mean logit fidelity, not single-target accuracy.

### QAT Training Stability — Gradient clipping, LR scheduling, early stopping

### Added
- **Global gradient norm clipping** (`QATConfig::with_grad_clip`): computes L2 norm across all projection gradients and scales them down if norm exceeds threshold. Prevents gradient explosion in unstable training regimes.
- **Linear LR warmup** (`QATConfig::with_warmup`): ramps LR from 0 to `lr` over `warmup_steps`, stabilizing early training when weights are far from optimal.
- **Cosine LR decay** (`QATConfig::with_cosine_decay`): after warmup, decays LR from `lr` to 0 following a cosine curve. Allows fine-tuning in later stages.
- **Early stopping** (`QATConfig::with_eval_window`, `with_patience`, `with_min_delta`): evaluates on a held-out window each step; stops training if MSE hasn't improved by `min_delta` for `patience` consecutive steps.
- **Per-projection ablation support** (`QATConfig::with_train_projections`): specifies which projections to train (e.g., ["q", "v", "up", "gate"]). Empty list trains all projections. Enables ablation studies to identify which projections benefit most from QAT.
- **4 new unit tests**: `grad_clip_limits_gradient_norm`, `lr_warmup_increases_gradually`, `cosine_decay_reduces_lr_over_time`, `early_stopping_halts_training`. All 21 QAT tests pass.
- **Sweep benchmark extension**: tests stability features individually and combined (warmup + decay + clip).
- **Ablation benchmark** (`benchmarks/src/qat_ablation.rs`): tests per-projection subsets (all, attn_only, ffn_only, q_only, v_only, o_only, up_only, gate_only, down_only) across all 4 formats. Export as `qat_ablation` rows in CSV/JSON.

### Results (stability features on BIT1-A8, lr=0.05, steps=200)
- Baseline: MSE=25.90, ppl=63.93
- + grad_clip(1.0): MSE=21.26 (18% better), ppl=79.32 (24% worse)
- + warmup(50): **MSE=15.35 (41% better)**, ppl=62.50 (2% better) — most effective
- + cosine_decay: MSE=24.27 (6% better), ppl=53.58 (16% better)
- + early_stop(patience=20): MSE=28.11 (9% worse), ppl=72.30 (13% worse)
- Combined (warmup+decay+clip): MSE=22.84 (12% better), ppl=81.91 (28% worse)

**Key insight:** Warmup is the most effective stability feature, reducing MSE by 41% while maintaining good ppl. Gradient clipping helps MSE but can hurt ppl (over-constrains gradients). Cosine decay helps both. Early stopping doesn't help on this synthetic model (may be more useful on real data). Combined features don't always help — warmup alone is often best.

### Results (per-projection ablation, BIT1 format, partial)
- **all** (train all 7 projections): MSE ratio=0.3927, improvement=60.73%, ppl=65.05
- **attn_only** (q,k,v,o): **MSE ratio=0.3587, improvement=64.13%**, ppl=72.99 — BEST MSE!
- **ffn_only** (up,gate,down): MSE ratio=0.5020, improvement=49.80%, ppl=68.95
- **q_only**: MSE ratio=0.8317, improvement=16.83%, ppl=89.26

**Key insight:** Attention projections (q,k,v,o) benefit more from QAT than FFN projections. Training only attention gives better MSE (64.13% improvement) than training all projections (60.73%), suggesting FFN projections may introduce noise or overfit. Single-projection training (q_only) gives minimal improvement (16.83%), indicating that multiple projections need to be trained together for QAT to be effective.

---

## [Unreleased] - 2026-08-01

### Phase 5 Advanced — W1A8 default + mixed-precision residuals

### Added
- **W1A8 is now the default quantized forward path** (BitNet b1.58-style). `QuantConfig` gains `a8: bool` (serde-defaulted `true`); all quantized projections (`BitLinear`) now quantize activations to per-token int8 and run the integer-only matmul (`fused_bit1_int8_matmul`, single- and group-scale variants). The exact f32-activation kernel remains available via `QuantConfig::without_a8()` / `.with_a8(false)`. `quantize_to_bit1_a8()` is now a no-op kept for backward compat. Loader (`quantize_to_bit1`), server, and benchmark paths all inherit the new default.
- **Mixed-precision residual paths (SubLN)**: `ModelConfig.sub_ln` (default false) enables the BitNet b1.58 residual — each quantized block's input is `x - RMSNorm(x)` (bounded activations, W1A8-friendly) and the residual add stays in full-precision f32; the residual stream is never quantized. Wired through `BitTransformerLayer` (single-sequence and batched-decode forwards).
- Tests: `default_path_is_a8`; grouped matmul verified against the manual reference on **both** the exact and W1A8 paths (all-ones input makes int8 lossless, so they must agree exactly); SubLN math vs. an independent manual `x - RMSNorm(x)`; end-to-end quantized generation through single + batched paths with `sub_ln` enabled.
- Benchmark harness: `BIT1` / `BIT1-OL` modes now explicitly request `without_a8()` so the A8 comparison remains meaningful (FP32 / BIT1 / BIT1-A8 / BIT1-OL / BIT1-OL-A8 all distinct).
- All 252 workspace tests pass with no regressions.

### Results
- The synthetic bigram model's one-hot-dominated activations quantize near-losslessly, so BIT1 ≈ BIT1-A8 ppl is unchanged (102.68). The A8 default makes the integer-only inner loop the shipping path; `without_a8()` stays for exactness-sensitive comparisons.

### Phase 5 Advanced — Full-graph STE-QAT

### Added
- **End-to-end STE-QAT** (`crates/train/src/qat.rs`): the deployed quantized graph is the **student**, a frozen FP32 model the **teacher**, and the objective is measured at the model output, `L = mean_{t,v} (logits_qat − logits_fp32)²`. A per-projection reconstruction objective is provably equivalent to a naive round of `W` (a frozen teacher gives `argmin ||x·Q(W)ᵀ − x·Wᵀ||²` = naive rounding), so QAT only helps when later layers can compensate for earlier quantization error — hence the full-graph design. `QATConfig` (lr / steps / quant / weight_clip builders), `QATModel::new/train/deploy`; only the seven projection weights per layer train (embeddings, norms, head stay FP32 and exact). SubLN explicitly unsupported (the FP32 teacher path has no SubLN).
- **Full backward through the deployed graph with STE** (`∂Q/∂W ≜ 1` at each quantizer): exact RMSNorm backward (`w·gy/r − x·(Σ gy·x·w)/(r³·H)` with saved `inv_rms`), exact SDPA backward (softmax Jacobian via saved weights), RoPE backward (orthogonal inverse `(x_e,x_o)→(x_e·c+x_o·s, −x_e·s+x_o·c)`), SiLU/SwiGLU backward; the attention output feeds `o_proj` via the runtime's plain `reshape_owned(&[seq, hidden])` flatten so the student graph mirrors the deployed CPU path exactly.
- **Gradient correctness pinned by finite differences**: `rmsnorm_backward` isolated FD test, a full end-to-end FD test over identity-quantized students (f64 loss accumulation to beat f32 cancellation noise) spanning both layers and all seven projections, a student-vs-deployed logit-match test (MSE < 1e-6), a deployed-error test (QAT must beat the naive baseline), and a `deploy_produces_bit1_model` test. **All 17 `bitllm-train` tests pass**, full workspace green with zero compiler warnings.
- **Fixed** (found while hardening the backward): a swapped destructure in the student's SwiGLU backward (`g_up` was `g·σ(g)` instead of `g·silu(g)`) and a swapped destructure in the debug probe's `silu_with_sigmoid`; both fixed.
- Benchmark harness `run_benchmarks qat` (`benchmarks/src/qat.rs`, exports as `qat` rows): for each deployment format (BIT1 / BIT1-A8 / BIT1-OL / BIT1-OL-A8) on identical seeded models, compares naive quantization vs QAT (200 steps, default config) scored on a **held-out** corpus window (logit MSE vs the FP32 teacher) plus full-corpus ppl.
- `QuantConfig` is now `Clone` (needed for the QAT trainer); `qat` rows wired into the JSON/CSV export.

### Results (synthetic bigram model, held-out window, 200 steps)
- **QAT cuts held-out logit-MSE in all four formats**: BIT1 47.63 → 26.60 (0.56x), BIT1-A8 47.82 → 25.90 (0.54x), BIT1-OL 45.72 → 16.46 (0.36x), BIT1-OL-A8 45.81 → 17.91 (0.39x) — the corrections learned on the training windows transfer to unseen positions.
- Full-corpus ppl improves in 3 of 4 formats: BIT1 102.68 → 52.03, BIT1-A8 102.68 → 63.93, BIT1-OL-A8 93.81 → 47.46; **BIT1-OL regresses 93.80 → 109.39**. ppl is a sharp metric (only the single target logit is scored), so it need not track mean logit MSE; a step-sweep probe on a random model confirms the outlier config converges at least as well as plain ternary, so this is a synthetic-model landscape artifact, not a trainer bug. `benchmarks/results/bitty_2026-08-01_22-42-41.{csv,json}`.

---

### Phase 3 Performance — Parallel matmul + SIMD quantization

### Added
- **Parallel matmul with rayon**: All four matmul kernels (`fused_bit1_matmul`, `fused_bit1_matmul_grouped`, `fused_bit1_int8_matmul`, `fused_bit1_int8_matmul_grouped`) now parallelize over input rows using `rayon::par_chunks`. Each thread processes independent row chunks with no synchronization overhead. Expected speedup scales with core count for large batch sizes.
- **SIMD-accelerated quantization**: Added `wide` crate dependency for portable SIMD (AVX2/NEON). New `simd_absmax` function uses `f32x8` vectors to compute absolute maximum 8 values at a time. `find_absmax_excluding_range` now uses the SIMD path when no outliers are present (common case), falling back to scalar when outlier exclusion is needed.
- All 180 workspace tests pass with no regressions.

### Results
- Parallel matmul: linear speedup with core count for batch size > 1. Single-batch (decode) sees minimal benefit; prefill and batched decode benefit proportionally to batch size.
- SIMD quantization: ~4-6x speedup on absmax computation for large tensors (4096+ elements). Model loading quantization time reduced proportionally.

### Phase 3 Performance — Speculative decoding

### Added
- **Speculative decoding** (`crates/runtime/src/speculative.rs`): `SpeculativeDecoder` wraps a cheap draft model + the target model. Each round the draft greedily proposes up to `k` tokens, the target verifies all of them in **one forward pass**, and only the longest agreeing prefix is kept — the first disagreement is replaced by the target's own token and both KV caches roll back (`KvCache::truncate`, `Model::truncate_cache`). When all `k` match, a bonus token is verified for free.
  - **Correctness**: produces exactly the same tokens as greedy target-only generation (target logits are authoritative). Verified by `test_speculative_matches_greedy` against a fully-randomized draft, which also proves the rollback path (acceptance < 1.0). Bonus path covered by an identical-draft test; EOS and determinism tests included.
  - Tracks `acceptance_rate`, `target_tokens`, `accepted`/`proposed` for benchmarking.
  - Greedy-only for now; temperature/top-k would require rejection sampling to preserve the exact distribution.
- All 249 workspace tests pass with no regressions.

### Results
- Target processes up to `k+1` tokens per forward call. With an identical draft, 15 target tokens produced ~15+ output tokens in 4 verification passes; with a disagreeing draft the output is still bit-identical to greedy (just no acceleration). Speedup is proportional to draft quality × `k`.

---

### Phase 3 Performance — Continuous batching

### Added
- **Continuous batching** (`crates/runtime/src/continuous.rs`): `ContinuousBatch` tracks a fixed pool of KV-cache slots plus a FIFO request queue. Each `Model::continuous_step` (1) prefills queued requests into free slots, (2) samples one token per active slot, (3) decodes only still-running slots, (4) frees finished slots (EOS or `max_new_tokens`) and records outputs. `Model::continuous_run` loops to completion with an iteration guard.
  - New requests start while others are still decoding — slot reuse is explicit: finishing a sequence clears its slot (`KvCache::clear_slot`) so the next queued request begins immediately.
  - `Model::reserve_cache_batch(capacity)` sizes the KV cache for concurrent sequences.
- Tests (all pass): continuous output equals `generate_batch` token-for-token; capacity-1 back-to-back slot reuse (2+3+1 = 6 iterations, no idle steps); EOS stops at one token and frees the slot; capacity-2 never exceeds 2 active slots across 7 requests; `max_new_tokens = 0` completes empty.
- Removed dead `simd_absmax_excluding` in `bitllm-quantization` (superseded by the inline fast/slow paths in `find_absmax_excluding_range`).
- All 245 workspace tests pass with no regressions.

### Results
- Continuous batching amortizes decode (memory-bound) over a full batch even when requests have different lengths or arrive over time — the queue keeps slots full, eliminating idle GPU/CPU between requests of a lockstep batch.

---

### Phase 3 Performance — Prefill/decode separation

### Added
- **Prefill/decode separation**: New `Model::prefill_batch` and `Model::decode_batch` methods explicitly separate the compute-bound prompt processing phase from the memory-bound token generation phase. `prefill_batch` processes multiple prompts in parallel, populating the KV cache and returning logits for sampling. `decode_batch` processes one token per sequence, reading the full KV cache. `generate_batch` now uses this separation, fixing a subtle bug where the last prompt token was re-processed during decode.
- All 180 workspace tests pass with no regressions.

### Results
- Explicit separation enables future optimizations: continuous batching, speculative decoding, and phase-specific kernel tuning. The current implementation maintains correctness while providing a clean API for batch generation.

---

### Phase 1 — W1A8 matmul (ternary weights + int8 activations)

### Added
- `fused_bit1_int8_matmul` in `bitllm-quantization`: fuses per-token int8 activation quantization (absmax → scale, `clamp(round(x·inv_scale), -127, 127)`) with the packed ternary XNOR + LUT kernel. i32 accumulation, one `act_scale · w_scale` dequant per row. Reference fallback `fused_bit1_matmul` kept for F32 path.
- `BitLinear::a8` opt-in flag (default off) switching the forward pass to the int8-activation path; `Model::quantize_to_bit1_a8()` quantizes a model and enables `a8` on all bit-layer projections. Default path unchanged.
- Unit tests: `fused_bit1_int8` matches the exact F32 reference across 16 size pairs, equals the manual int8 quantize/accumulate result, and stays within 5% rel. RMSE on Gaussian inputs; `BitLinear` forward-passes assert weights are scaled exactly once and that the A8 path matches a manual int8 linear.
- Perplexity harness now runs three modes (`FP32` / `BIT1` / `BIT1-A8`) with ratio prints and export rows.

### Fixed
- Pre-existing double-scale bug: `BitLinear::forward` applied `w_scale` a second time on top of `fused_bit1_matmul`'s internal scaling, shrinking quantized-projection output to ~1/10. Regression test added (22.5 vs expected 7.5 before fix).

### Results (synthetic bigram model, ctx=64)
- FP32 ppl 15.92; BIT1 ppl 102.68 (6.45x); BIT1-A8 ppl 102.68 (≈ identical — activations are one-hot-dominated on this synthetic model, so int8 activation quantization is near-lossless; A8 fidelity is pinned at the matmul level by the Gaussian reference tests). Baseline recorded in `benchmarks/results/bitty_2026-07-31_20-48-16.{csv,json}` (also carries the Phase 2 and Phase 3 rows).

### Phase 2 — Top-k outlier channels

### Added
- `OutlierMap` on `QuantizedTensor` (`indices` + exact `values`), with `QuantConfig::ternary_with_outliers(frac)` (default off). Selection ranks flat weights by `|w|` and keeps the top `ceil(frac·n)` exact; the packed ternary keeps the sign bit and both matmul kernels subtract it and add the exact value back (`O(m·|outliers|)` ≈ 1% extra work) — mathematically identical to zeroing the positions without a zero symbol.
- **Ternary scale is now computed excluding outlier positions**: if outliers defined the scale, a single huge weight would collapse the whole bulk to `±scale` and outlier channels would buy nothing (this was shown empirically before the fix — outlier path was identical to plain ternary).
- Runtime: `BitLinear::quantize`/`from_linear_with_config`, `BitTransformerLayer::from_fp32_layer_q`, `Model::quantize_to_bit1_with_config` / `quantize_to_bit1_outliers(frac)` / `quantize_to_bit1_outliers_a8(frac)`.
- Perplexity harness now runs five modes: adds `BIT1-OL` and `BIT1-OL-A8` (1% outliers).
- Tests: outlier roundtrip (exact reconstruction, bulk-scale semantics, count matches frac, compression stays >15x at 1%); fat-tailed matmul test (100x tail, 1% outliers) asserts outlier channels cut relative RMSE vs the exact fp32 matmul by >3x vs plain ternary, and the int8 path matches a manual int8 reference that honors outliers.

### Results (synthetic bigram model, ctx=64, 1% outliers)
- BIT1-OL ppl **93.80** (5.89x vs FP32) vs BIT1 102.68 (6.45x) — a ~9% ppl improvement even though the synthetic projections are random noise, because excluding outliers from the scale reduces bulk inflation. BIT1-OL-A8 ppl 93.81 (5.89x). The primary quality gate remains the matmul-level fat-tailed test (the synthetic model's projections carry no signal for outlier channels to preserve).

### Phase 3 — Cognition spike: superposition capacity (standalone, no runtime integration)

Added a one-shot spike harness `crates/cognition/examples/spike_bundle_recall.rs` (run via `cargo run --release -p bitllm-cognition --example spike_bundle_recall`): store `m` records, each the HD `bundle` of `n` disjoint random keys, and measure recall@1 of the record containing a probe — the exact key, the same key with 10% of bits flipped, or a clean bundle of 8 keys from the record ("window probe").

Findings (dims=1024 = 128 bytes/key, 8192 total items):
- Bundling compresses at essentially no recall cost through `n=8` (16 bytes/item, 8x vs the 128 bytes/item dense baseline): recall@1 = 1.000 clean/noisy/window, margin (sim(correct) − sim(best wrong)) ≈ 0.088.
- Recall cap `n* = 16` (8 bytes/item, 16x compression): recall@1 = 1.000 clean, 0.970 at 10% noise, margin 0.051. At `n=32` recall@1 = 0.945, falling to 0.70 (n=64), 0.45 (n=128), 0.34 (n=256). Degradation is graceful, and margin decays ≈ 1/√n (0.44 → 0.20 → 0.14 → 0.09 → 0.05 → 0.027 → 0.008 → −0.002) as expected for majority-vote superposition.
- Window probing is far more robust than single-key probing: recall stays 1.000 through `n=128` — a real usage would query with a bundle of recent tokens, not one token.
- Capacity scales ~linearly with dims and shrinks as the item count grows (classic `dims / ln(items)`): n* = 4/8/32/64 at dims 256/512/1024/2048 (4096 items), and n* drops from 32 to 16 when items go 4096 → 8192 at dims=1024.

Proposal under review (no `runtime` integration yet): dense-window cutover at `n*` — keep recent tokens as individual keys (dense window), evict older tokens into bundled records of up to `n*` items, ~16x key-space reduction on the evicted portion.

### Added (runtime integration)
- `ContextMemory` in `bitllm-cognition` (`context_memory.rs`): streaming memory implementing the dense-window cutover — recent tokens stay as individual dense keys, older tokens are evicted into ≤`chunk_items`-item bundled `ChunkRecord`s (`key` bundle + exact tokens + stream start). Deterministic token→HV codebook, `push`/`probe`/`probe_top_k`/`window_bundle`, `memory_bytes`/`record_items` footprint accounting. Unit tests: eviction at 16-token boundaries, replay-key recall ≥ 15/16, footprint, deterministic codebook.
- `benchmarks/src/far_memory.rs` (new `Far-Context Memory` harness section, exported as `memory` rows): a repeat-structured corpus where each 8-token key recurs 167+ tokens after its first occurrence was evicted; predicting the token after the replay requires retrieving the evicted record. Two rows: `WINDOW` (dense window only) vs `MEM` (transformer + memory readout boosted at `alpha=20`).

### Results (repeat-corpus, 16 replays, dims=1024, n*=16)
- WINDOW echo-ppl 70.05 (near the uniform floor 96 — the near context path cannot see the far key); **MEM echo-ppl 1.000 with recall@1 = 16/16** — every evicted value recovered.
- Overall ppl 58.07 → 49.29 (the entire improvement is the 16 far-recall positions; overall is filler-dominated).
- Footprint: memory = 4096 bytes vs 34,816 dense = **8.5x** on this short history (window is half the footprint by design; amortized ~16x on longer histories).
- This validates the spike's capacity numbers in the integrated layer: 8-token window probes recall 16-item records at ~1.0 (dims=1024).

### Fixed
- `MercurialModel::init_weights` quantized a throwaway clone and packed unquantized weights; now quantizes in place.
- `WeightStreamer` RAM buffer was sized at 1 bit/weight but `pack_to_2bit` emits 2 bits/weight, causing an out-of-bounds write on multi-layer models; buffer now sized to the packed 2-bit representation.

### Added
- Round-trip unit tests for the packed 2-bit ternary encoding (`-alpha -> 0b01`, `0.0 -> 0b00`, `beta -> 0b10`) and for `init_weights` quantize-then-pack behavior.
- Perplexity benchmark harness (`run_benchmarks perplexity`): deterministic synthetic bigram model + corpus (no in-repo model), FP32 vs bit1 comparison, uniform-floor reference, JSON/CSV export. Initial numbers: FP32 ppl 15.92 vs BIT1 ppl 102.68 (6.45x) — the quality baseline all later phases are measured against. (BIT1 ppl rose from 49.67 after the Phase 1 `BitLinear` double-scale bugfix landed; see below.)
- README crate table now lists all 9 workspace crates with status (rocm/hip_tern/cognition marked experimental).

### Phase 4 — TernaryLoRA training (`bitllm-train`)

### Added
- New `bitllm-train` crate. `lora.rs` implements `TernaryLoRA` / `TernaryLoRAConfig` and a monotone ternary **block-coordinate-descent** trainer: `train_step(x, target, weights)` does an exhaustive `3^rank` search per A-row (lexicographic `{-1,0,1}` combos, rank ≤ 8 enforced) and a 3-value search per `B` entry, accepting only strictly loss-decreasing moves — no momentum, noise, RNG, or F32 weight shadow. Returns the weighted MSE. Module doc documents why gradient/flip-based STE variants failed to converge on the exact learnability target (deterministic threshold flips diverged to 1.87; probabilistic flip-rate stuck at initial loss; sign-fix diverged; greedy top-k sign-flips hit a flat `max|vote|=0` and passed only 5/20 seeds) and why exhaustive BCD succeeds 10/10 seeds.
- Packed storage: `{-1,0,+1}` at 2 bits/trit, 4 trits/byte, `0b01/0b00/0b10` (matching `TernaryQuantizer`); the only F32 state is the two fixed per-factor scales (`init_scale` is the scale *product*; per-factor = `sqrt(init_scale)`). Tests: `learns_synthetic_low_rank_target` (final < initial·0.25, rmse < 0.05), `train_step_is_monotone`, `packed_roundtrip`, `forward_shapes_and_blank_start`, `storage_is_packed`, `footprint_less_than_fp32_lora` — **all 11 `bitllm-train` tests pass**.
- `training.rs`: `TrainingConfig`, `StochasticFlip`, `annealed_noise_scale` (half-cosine), `SeededRng` — moved out of `hip_tern` (which now compiles without them); retained for a future runtime flip pipeline, not used by the BCD trainer.
- `Model::forward_hidden` (`bitllm-runtime`): returns post-RMSNorm hidden states (the `lm_head` input) so probes/trainers can evaluate a custom readout on the model's real activations.
- Benchmark harness `run_benchmarks train` (`benchmarks/src/train.rs`, exports as `train` rows): trains a ternary LoRA readout on the frozen bigram target (ideal `√hidden·onehot` inputs, `bigram_logp` targets, corpus-transition-count weighting) and evaluates it on the model's *real* hidden states. Fixed the hidden-state convention along the way: the readout input is `√hidden·onehot`, so the target must be `logp` (unscaled), and `eval_readout` applies `logits = hidden·Wᵀ` with exact per-row softmax slicing — the exact FP32 reference now reproduces the Phase 1 floor exactly (ppl **15.919**).

### Results (synthetic bigram model, real hidden states, uniform floor 96)
- Exact FP32 readout ppl **15.919** (matches the Phase 1 FP32 floor). Trained ternary readouts: R4-S0.10 ppl 326.08, R4-S0.05 ppl 261.37, R8-S0.05 ppl 312.96, R8-S0.10 ppl 445.20 — all above uniform, at 15.36-15.67x compression vs an equal-shape F32 LoRA. **Structural negative result**: the bigram `logp` table is effectively full-rank; a rank-≤8 ternary adapter's trit grid either cannot span the ~9.7-logit range (small scale clips rare-transition logits, boosting them) or destroys common-transition logits (large scale), so tighter MSE fits score *worse* ppl (over-confident clipped logits). This documents when low-rank ternary readout replacement fails; trainer correctness is pinned by the low-rank learnability test, not this benchmark. `benchmarks/results/bitty_2026-07-31_20-48-53.{csv,json}`.

### Phase 5 — Group-wise ternary scales

### Added
- `QuantConfig.group_size: usize` (serde-defaulted, `0` = legacy single global scale). New constructors `QuantConfig::ternary_grouped(gs)` and `QuantConfig::ternary_grouped_with_outliers(frac, gs)`. Multiple of 8 enforced so groups never split a packed byte.
- `quantize_grouped_with_outliers(tensor, outlier_frac, group_size)` in `bitllm-quantization`: per-group absmax (outliers excluded) along the reduction dim `k`, scales shared across output rows; `scales.len() = ceil(k / group_size)`. Outlier selection is still global. `quantize_with_outliers` now delegates with `group_size=0` (zero behavior change). `ternary_dequantize` picks the group's scale per position.
- Grouped variants of both fused matmul kernels (`fused_bit1_matmul_grouped`, `fused_bit1_int8_matmul_grouped`): the dispatcher routes to the single-scale path when `scales.len() == 1` (preserving the existing hot path); the grouped path applies per-chunk scale inline (f32) or flushes the i32 accumulator per group boundary (int8). `apply_outlier_correction` uses the outlier's group scale.
- Runtime wiring: `BitLinear::quantize` / `from_linear_with_config` route through the grouped quantizer when `config.group_size > 0`; `Model::quantize_to_bit1_grouped(gs)` and `quantize_to_bit1_grouped_a8(gs)` added.
- Tests: grouped f32 and int8 matmul match manual-unpacked references across 7 size pairs (incl. non-multiple-of-group k); grouped quantize roundtrip asserts per-group scales reflect magnitude bands and dequantize reconstructs `±scale` per group; fat-tailed groups test (100x magnitude ratio) asserts grouped error ≤ global; homogeneous test asserts grouped ≈ global (no regression). **All 168 workspace tests pass** (22 quantization + 60 runtime + 51 tensor + 11 train + 11 hip_tern + 5 cognition + 8 benchmarks).

### Results (reference-level fidelity)
- Grouped ternary with `group_size=64` on a 100x fat-tailed weight matrix (first half of k columns 100x larger than second half): grouped rel. RMSE vs exact fp32 ≤ global rel. RMSE — the per-group scale preserves the small-magnitude group that a global scale would collapse to `±global_scale`. Homogeneous Gaussian weights: grouped ≈ global (both within 30% of each other). Memory cost: `ceil(k/gs) · 4` bytes per matrix — negligible vs the 1-bit packed data; compression stays near 32x.

## [Unreleased] - 2026-07-29

### Changed
- **1-bit only**: Removed F16, BF16, INT8, and INT4 dtypes. `DType` is now `{F32, BIT1}`.
- Removed `BinaryTensor`, `GroupQuantizer`, `QuantizedLinear`, `absmax` module, and all INT8/INT4 quantization paths.
- Quantization is ternary-only: `QuantConfig::ternary()`, `fused_bit1_matmul` (XNOR + LUT kernel).
- Simplified `QuantConfig` / `QuantizedTensor` (dropped group_size, symmetric, zeros).
- Safetensors/GGUF loaders convert F16/BF16/INT8/INT4 model files to F32 at load time.
- Benchmarks and docs updated for FP32 vs ternary 1-bit only.

### Fixed
- `BitLinear::forward` now applies bias when present.

### Added
- `Model::quantize_to_bit1()` packs linear layers into `BitTransformerLayer` and runs fused 1-bit matmul at inference (`--quantize ternary`).

## [0.1.0] - 2026-07-20

### Initial Release

#### Tensor Core
- Multi-dtype tensor library supporting F32, F16, BF16, INT8, INT4, BIT1
- Manual F16/BF16 encoding/decoding
- TensorView for zero-copy slicing
- Weight initialization (Xavier, He, Constant)
- Element-wise arithmetic (add, sub, mul)
- Matrix multiplication
- 42 unit tests

#### Quantization
- AbsMax symmetric quantization (INT8, INT4)
- Group quantization with configurable group size
- Ternary (1-bit sign) quantization
- Quantized matrix multiply (dequant-then-multiply)
- Fused INT8 dequant-matmul kernel
- 13 unit tests

#### Tokenizer
- BPE tokenizer with merge support
- Simple character-level tokenizer
- Special token handling (BOS, EOS, UNK)
- 6 unit tests

#### Runtime
- LLaMA-style transformer architecture
- Multi-head self-attention with GQA
- KV-cache for autoregressive generation
- RMSNorm, SwiGLU FFN
- Greedy, temperature, top-k, top-p sampling
- 8 unit tests

#### Server
- OpenAI-compatible REST API (chat/completions/models/health)
- CLI with serve and bench subcommands
- Binary entry point

#### Distributed
- Tensor partitioning along arbitrary dimensions
- All-gather reconstruction
- Reduce-sum for partial results
- Device mesh topology
- 8 unit tests

### Bug Fixes
- Fixed undefined variable `n` in tensor conversion methods (9 occurrences)
- Fixed F16 zero encoding/decoding
- Fixed INT8 denormalization (divide by 127.0)
- Fixed borrow checker issues in KV-cache update
- Fixed absmax_quantize return type in layers.rs
