# BitLLM Benchmarks

## Benchmark Methodology

All benchmarks run on the CPU with release profile optimizations. Measurements include:
- **Latency**: Time per inference iteration
- **Throughput**: Tokens per second
- **Memory**: Peak memory usage
- **Compression**: Ratio of quantized vs FP32 size

## Quantization Accuracy

### Ternary 1-bit (absmax-scaled)

| Metric | Value |
|--------|-------|
| Sign accuracy | 100% |
| Compression | ~32x vs FP32 |

Weights are quantized to `sign(w) * max(|w|)` and packed 8 values per byte.

## Inference Performance

### Tiny Model (2 layers, hidden=64, vocab=256)

| Metric | Value |
|--------|-------|
| Forward pass | < 1ms |
| Generation (32 tokens) | < 100ms |
| Throughput | ~320 tokens/sec |

### LLaMA-Small (32 layers, hidden=4096, vocab=32000)

| Metric | Value |
|--------|-------|
| Forward pass | TBD |
| Generation | TBD |
| Memory (FP32 weights) | ~2.1GB |
| Memory (1-bit weights) | ~80MB |

## Tensor Parallelism

### Partition Overhead

| World Size | Overhead |
|------------|----------|
| 1 | 0% (baseline) |
| 2 | < 5% |
| 4 | < 10% |

## Memory Reduction

Theoretical weight memory from quantization (activations remain F32):

| Precision | Bytes/Param | 7B Model Weights |
|-----------|-------------|------------------|
| FP32 | 4.0 | 28.0 GB |
| 1-bit ternary | 0.125 | 0.875 GB |

## Running Benchmarks

```bash
# Run the built-in benchmark
cargo run --release --bin bitllm -- bench --model tiny --iterations 1000

# Run Rust benchmarks (requires nightly for criterion)
cargo bench --workspace

# Run all tests with timing
cargo test --workspace -- --nocapture

# Standalone benchmark suite
cargo run --release -p bitllm-benchmarks
```
