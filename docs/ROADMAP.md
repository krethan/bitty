# BitLLM Roadmap

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
- [ ] Streaming token generation
- [ ] Tokenizer file loading (BPE merges)
- [ ] Batch inference support

## Phase 3: Performance Optimization

- [ ] SIMD-accelerated quantization kernels (AVX2/NEON)
- [ ] Parallel matmul with rayon thread pool
- [ ] Memory-mapped weight loading
- [ ] Prefill/decode separation
- [ ] Continuous batching
- [ ] Speculative decoding

## Phase 4: GPU Acceleration

- [ ] AMD ROCm backend
- [ ] GPU tensor operations
- [ ] GPU quantized matmul kernels
- [ ] Unified CPU/GPU memory management
- [ ] Multi-GPU tensor parallelism with NCCL

## Phase 5: Advanced 1-bit Techniques

- [ ] Group-wise 1-bit quantization (finer-grained scales)
- [ ] BitNet-style activation quantization (W1A8)
- [ ] Mixed-precision residual paths
- [ ] Quantization-aware training support

## Phase 6: Production Features

- [ ] Prometheus metrics endpoint
- [ ] Request queuing and backpressure
- [ ] WebSocket streaming
- [ ] Model hot-swapping
- [ ] Graceful shutdown
- [ ] Docker deployment

## Phase 7: Model Support

- [ ] LLaMA 2/3
- [ ] Mistral/Mixtral
- [ ] Phi-2
- [ ] GPT-2
- [ ] Qwen
- [ ] Custom model support
