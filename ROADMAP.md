# BitLLM Roadmap

## Status

- [x] Phase 7: Multi-architecture model support (GPT-2, Phi, Gemma, Mistral, Qwen, Custom)
- [x] Fix Phi/Gemma real-checkpoint loading: bias broadcast, auto-tie embeddings, Gemma2 mappers
- [x] Fix Qwen safetensors bias loading (LlamaWeightMapper/GgufWeightMapper missing bias tensors)
- [x] Fix GPT-2 byte-level tokenizer pre-tokenization (spaces → Ġ)
- [x] Remove GGUF q/k RoPE-half de-interleave: official Qwen2.5 (and other non-interleaved-RoPE) GGUFs store `attn_q`/`attn_k` in plain torch order; applying the transform reordered the projection rows and broke the model (ppl went from 5.77 to 31K–63K). Weight-equivalence analysis confirmed raw GGUF q/k correlate 0.90 with safetensors, de-interleaved -0.0052.
- [x] Full Gemma2 support: post-FFN RMSNorm, one-centered RMSNorm (`(1+w)`), attention logit softcap (`tanh(x/cap)*cap`), final logit softcap, and `query_pre_attn_scalar` logit scale threaded through config, HF JSON, GGUF metadata, the CPU SDPA kernels, and the bit-transformer kernels.
- [x] `PostFfnNorm` weight target mapped from `model.layers.N.post_feedforward_layernorm.weight` (HF) and `blk.N.ffn_post_norm.weight` (GGUF).
- [x] Real Gemma2/Phi checkpoint loading verified: 0 skipped tensors, finite logits (regression tests guard `/tmp/opencode/models/{gemma2,phi}`).

## Verified Models

| Model | Format | PPL (text) | Notes |
|-------|--------|-----------|-------|
| SmolLM2-135M-Instruct | safetensors (BF16) | 12.35 | Working |
| SmolLM2-135M-Instruct | GGUF F16 | 12.35 | Matches safetensors |
| SmolLM2-135M-Instruct | GGUF Q8_0 | 12.43 | Matches F16 |
| Qwen2.5-0.5B-Instruct | safetensors (BF16) | 5.81 | Working (bias fix) |
| Qwen2.5-0.5B-Instruct | GGUF Q8_0 | 5.77 | Matches safetensors |
| Qwen2.5-0.5B-Instruct | GGUF F16 | 5.77 | Matches safetensors |
| Gemma2 (tiny, untrained) | safetensors | 255884 | ≈ uniform floor (256000); 0 skipped, finite logits, one-centered norm + softcaps verified |
| Phi (tiny, untrained) | safetensors | 1013 | ≈ uniform floor (1024); 0 skipped, finite logits, `lm_head.bias` loaded |
| Llama (tiny-random) | safetensors | 32648 | ≈ uniform floor (32000); 0 skipped, finite logits |

PPL values above are the cross-entropy of a fixed natural-language paragraph (see
`crates/server/examples/real_ppl.rs`), reproduced exactly on every run. Tiny
untrained checkpoints score at the uniform floor by construction — the number only
confirms finite, non-degenerate loading, not quality.

## WikiText-2 (sliding-window eval, `crates/server/examples/corpus_ppl.rs`)

Every token scored exactly once (context 512, stride 512, no BOS/EOS).

| Model | Split | Tokens | PPL | bits/tok |
|-------|-------|--------|-----|----------|
| SmolLM2-135M-Instruct | valid | 273,867 | 21.545 | 4.429 |
| SmolLM2-135M-Instruct | test | 312,143 | 20.368 | 4.348 |

Note: this eval runs on CPU (8 workers, ~27 min per split on a Ryzen 7 3700X).

## Known Issues

- The tokenizer byte-level pre-tokenization changed SmolLM2 ppl from 38.30 to 12.35 (improvement). Any downstream consumers relying on the old broken tokenization will see different results.

## Performance & Numerical Fixes

- Tokenizer: the BPE merge-priority map was rebuilt for every word (~50k merge lookups per word). Hoisted out of the word loop → encoding a 1.1 MB corpus went from ~30 min to 0.3 s (same output, verified token-for-token).
- AVX2 `fast_exp256`: the integer exponent in `fast_pow2_256` left the biased range for inputs beyond ~±87, producing garbage (NaN / huge wrong signs) instead of saturating to 0/+inf. This corrupted a full softmax row at a specific position and made the WikiText-2 valid-set PPL NaN (SmolLM2). Fixed by clamping the exp input to the f32 range before the polynomial; regression test `test_simd_f32_exp_overflow_range` guards it.

## Next Steps

- [x] Verify Gemma2/Phi with real checkpoints (0-skip + finite-logit regression tests)
- [x] Add QAT activation recorder validation for Qwen
- [x] Add a GGUF-vs-safetensors weight-equivalence unit test to prevent q/k layout regressions
- [x] GPU backend testing: hardware-gated GPU-vs-CPU parity tests for `GpuContext` ops (matmul/add/sub/mul/scale/softmax/rope/rms-norm) and end-to-end model forward incl. the KV-cache decode path. Skip cleanly without a device; run on a ROCm host via `cargo test --features bitllm-runtime/gpu,bitllm-rocm/rocm`.
