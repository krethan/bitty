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
| Gemma2 (tiny) | safetensors | — | 0 skipped, finite logits; one-centered norm + softcaps verified |
| Phi (tiny) | safetensors | — | 0 skipped, finite logits; `lm_head.bias` loaded |

## Known Issues

- The tokenizer byte-level pre-tokenization changed SmolLM2 ppl from 38.30 to 12.35 (improvement). Any downstream consumers relying on the old broken tokenization will see different results.

## Next Steps

- [x] Verify Gemma2/Phi with real checkpoints (0-skip + finite-logit regression tests)
- [x] Add QAT activation recorder validation for Qwen
- [x] Add a GGUF-vs-safetensors weight-equivalence unit test to prevent q/k layout regressions
- [ ] GPU backend testing
