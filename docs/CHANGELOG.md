# Changelog

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
