# BitLLM Roadmap

## Phase 1: Core Engine (Completed)

- [x] Tensor library with 6 dtypes (F32/F16/BF16/INT8/INT4/BIT1)
- [x] AbsMax quantization (INT8/INT4)
- [x] Group quantization with configurable group size
- [x] Ternary (1-bit) quantization
- [x] Quantized matrix multiplication
- [x] BPE tokenizer
- [x] LLaMA transformer architecture
- [x] KV-cache for autoregressive generation
- [x] Multiple sampling strategies
- [x] OpenAI-compatible API server
- [x] Tensor parallelism primitives
- [x] Comprehensive test suite (77 tests)

## Phase 2: Model Loading & Real Inference

- [ ] SafeTensors/GGUF weight loading
- [ ] JSON model config parsing
- [ ] Weight quantization on load
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

## Phase 5: Advanced Quantization

- [ ] GPTQ weight quantization
- [ ] AWQ activation-aware quantization
- [ ] SmoothQuant for INT8 inference
- [ ] Mixed-precision quantization per layer
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
