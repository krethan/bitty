# BitLLM

A highly optimized, open-source 1-bit LLM inference engine written in Rust.

BitLLM dramatically reduces memory requirements for running large language models by supporting aggressive quantization down to 1-bit precision, while maintaining inference quality through carefully designed quantization schemes.

## Features

- **Multi-precision quantization**: F32, F16, BF16, INT8, INT4, and 1-bit (BIT1)
- **Per-channel and group quantization**: AbsMax, symmetric, and asymmetric schemes
- **Quantized matrix multiplication**: Fused dequantize-multiply kernels
- **Transformer inference**: Full LLaMA/GPT-2 architecture support with KV-cache
- **OpenAI-compatible API server**: Drop-in replacement for `/v1/chat/completions`
- **Tensor parallelism**: Multi-device model partitioning
- **BPE tokenizer**: Built-in tokenization with special token support
- **Multiple sampling strategies**: Greedy, temperature, top-k, top-p

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
| `bitllm-tensor` | Core tensor operations with multi-dtype support |
| `bitllm-quantization` | Quantization algorithms (absmax, group, ternary) |
| `bitllm-tokenizer` | BPE tokenizer and encoding/decoding |
| `bitllm-runtime` | Transformer model, attention, KV-cache, sampling |
| `bitllm-server` | OpenAI-compatible REST API server |
| `bitllm-distributed` | Tensor parallelism and multi-device support |

## Quantization Schemes

| Scheme | Bits | Compression | Quality |
|---|---|---|---|
| INT8 AbsMax | 8 | 4x | Near-lossless |
| INT4 Group (g=128) | 4 | 8x | High quality |
| 1-bit Sign | 1 | 32x | Experimental |

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
