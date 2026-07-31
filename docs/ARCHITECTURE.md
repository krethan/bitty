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
│   (1-bit ternary, fused XNOR+LUT GEMM)      │
├─────────────────────────────────────────────┤
│             Tensor Layer                     │
│   (F32 + BIT1 dtypes, arithmetic, matmul)   │
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
| F32 | 32 | 4 bytes/elem | Activations, computation |
| BIT1 | 1 | 8 elem/byte | Packed 1-bit weights |

Model files stored as F16/BF16/INT8/INT4 (safetensors, GGUF) are converted to F32 at load time.

### Key Operations

- **Element access**: `get_flat_f32()`, `set_flat_f32()` with automatic dtype decoding
- **Shape ops**: `reshape()`, `transpose()`, `flatten()`
- **Arithmetic**: `add()`, `sub()`, `mul()`, `dot()` (matmul)
- **Conversion**: `to_dtype()`, `to_f32()`, `to_bit1()`

## Quantization Crate (`bitllm-quantization`)

### Ternary (1-bit) Quantization

Weights are quantized to {-1, +1} scaled by the per-tensor absmax: `w_q = sign(w) * max(|w|)`. Achieves ~32x compression ratio over F32.

### Fused 1-bit Matrix Multiply

`fused_bit1_matmul` operates directly on packed BIT1 weights using an XNOR + LUT inner loop: for each group of 8 input elements a 256-entry LUT maps sign-match masks to magnitude sums, so each packed byte contributes its dot-product term in O(1).

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
