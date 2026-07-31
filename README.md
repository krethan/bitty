# BitLLM

A highly optimized, open-source 1-bit LLM inference engine written in Rust.

BitLLM dramatically reduces memory requirements for running large language models by supporting aggressive quantization down to 1-bit precision, while maintaining inference quality through carefully designed quantization schemes.

## Features

- **1-bit quantization**: Ternary weights ({-1, 0, +1}) with per-tensor absmax scaling
- **Fused 1-bit matrix multiplication**: XNOR + LUT kernel operating directly on packed bits
- **Transformer inference**: Full LLaMA/GPT-2 architecture support with KV-cache
- **OpenAI-compatible API server**: Drop-in replacement for `/v1/chat/completions`
- **Tensor parallelism**: Multi-device model partitioning
- **BPE tokenizer**: Built-in tokenization with special token support
- **Multiple sampling strategies**: Greedy, temperature, top-k, top-p
- **Broad model format support**: Loads safetensors and GGUF (F32/F16/BF16/INT8/INT4 weights are converted to F32 at load time)

## Quick Start

```bash
# Build the project
cargo build --release

# Run the inference server
cargo run --release --bin bitllm -- serve --host 0.0.0.0 --port 8080

# Run benchmarks
cargo run --release --bin bitllm -- bench --model tiny --iterations 100

# Run tests
cargo test --workspace
```

## Architecture

BitLLM is organized as a Rust workspace with 6 crates:

| Crate | Purpose |
|---|---|
| `bitllm-tensor` | Core tensor operations (F32 activations, BIT1 packed weights) |
| `bitllm-quantization` | Ternary quantization and fused 1-bit matmul kernels |
| `bitllm-tokenizer` | BPE tokenizer and encoding/decoding |
| `bitllm-runtime` | Transformer model, attention, KV-cache, sampling |
| `bitllm-server` | OpenAI-compatible REST API server |
| `bitllm-distributed` | Tensor parallelism and multi-device support |

## Quantization

| Scheme | Bits | Compression |
|---|---|---|
| Ternary (absmax-scaled) | 1 | ~32x |

Load with packed 1-bit weights (fused XNOR+LUT matmul at inference):

```bash
cargo run --release --bin bitllm -- serve \
  --safetensors model.safetensors \
  --config-json config.json \
  --quantize ternary
```

Embeddings, norms, and lm_head stay F32; attention/FFN projections are packed BIT1.

## API

BitLLM exposes an OpenAI-compatible API:

```bash
# Chat completions
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128,
    "temperature": 0.7
  }'

# Text completions
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The capital of France is",
    "max_tokens": 64
  }'

# Health check
curl http://localhost:8080/health
```

## License

MIT
