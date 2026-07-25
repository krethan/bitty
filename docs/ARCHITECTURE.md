# BitLLM Architecture

## System Overview

BitLLM implements a layered architecture for quantized LLM inference:

```
┌─────────────────────────────────────────────┐
│                Server Layer                  │
│   (REST API, OpenAI compat, CLI)            │
├─────────────────────────────────────────────┤
│              Runtime Layer                   │
│   (Transformer, Attention, KV-cache)        │
├─────────────────────────────────────────────┤
│            Quantization Layer                │
│   (INT8/INT4/1-bit, group quant, GEMM)     │
├─────────────────────────────────────────────┤
│             Tensor Layer                     │
│   (Multi-dtype, arithmetic, matmul)         │
├─────────────────────────────────────────────┤
│           Tokenizer Layer                    │
│   (BPE, encoding, decoding)                 │
├─────────────────────────────────────────────┤
│          Distributed Layer                   │
│   (Tensor parallelism, device mesh)         │
└─────────────────────────────────────────────┘
```

## Tensor Crate (`bitllm-tensor`)

The foundation of BitLLM. Provides a `Tensor` struct backed by raw `Vec<u8>` storage with automatic dtype-aware encoding/decoding.

### Supported DTypes

| Type | Bits | Storage | Use Case |
|------|------|---------|----------|
| F32 | 32 | 4 bytes/elem | Default, training |
| F16 | 16 | 2 bytes/elem | Mixed precision |
| BF16 | 16 | 2 bytes/elem | Stable gradients |
| INT8 | 8 | 1 byte/elem | Quantized inference |
| INT4 | 4 | 2 elem/byte | Aggressive quant |
| BIT1 | 1 | 8 elem/byte | 1-bit inference |

### Key Operations

- **Element access**: `get_flat_f32()`, `set_flat_f32()` with automatic dtype decoding
- **Shape ops**: `reshape()`, `transpose()`, `flatten()`
- **Arithmetic**: `add()`, `sub()`, `mul()`, `dot()` (matmul)
- **Conversion**: `to_dtype()`, `to_f32()`, `to_f16()`, etc.

## Quantization Crate (`bitllm-quantization`)

### AbsMax Quantization

Symmetric per-block quantization. Divides tensor into blocks of 256 elements, computes per-block scale = max(|x|) / 127.

### Group Quantization

Per-group quantization with configurable group size (default 128). Each group gets independent scale factors, providing better accuracy for INT4.

### Ternary (1-bit) Quantization

Sign-only quantization: each value becomes ±absmax. Achieves 32x compression ratio.

### Quantized Matrix Multiply

Two modes:
1. **Dequantize-then-multiply**: Full dequantization then standard matmul
2. **Fused dequant-matmul**: Inline dequantization during multiply (INT8)

## Runtime Crate (`bitllm-runtime`)

### Model Architecture

LLaMA-style transformer with:
- Token + position embeddings
- Multi-head self-attention with GQA support
- SwiGLU feed-forward network
- RMSNorm normalization
- KV-cache for autoregressive generation

### Layers

- **Attention**: Multi-head with configurable KV-head grouping
- **Linear**: Dense layer with optional quantization
- **RMSNorm**: Root mean square normalization
- **Embedding**: Token lookup table

### Sampling

- Greedy (argmax)
- Temperature sampling
- Top-k filtering
- Top-p (nucleus) sampling

## Distributed Crate (`bitllm-distributed`)

### Tensor Parallelism

- **Partition**: Split tensors along arbitrary dimensions
- **All-gather**: Reconstruct full tensor from partitions
- **Reduce-sum**: Combine partial results

### Device Mesh

Logical topology for multi-device execution. Currently supports CPU-based partitioning; ROCm GPU support planned.

## Server Crate (`bitllm-server`)

Axum-based HTTP server providing:
- `POST /v1/chat/completions` - Chat interface
- `POST /v1/completions` - Text completion
- `GET /v1/models` - List available models
- `GET /health` - Health check

## Data Flow

```
Input Tokens
    ↓
Tokenizer (encode)
    ↓
Embedding Lookup
    ↓
Transformer Layers × N
  ├── Attention (with KV-cache)
  │     ├── Q/K/V projections
  │     ├── Scaled dot-product attention
  │     └── Output projection
  └── FFN
        ├── Gate projection (SwiGLU)
        ├── Up projection
        └── Down projection
    ↓
Final Norm
    ↓
LM Head (logits)
    ↓
Sampler → Output Token
```
