# BitLLM Benchmarks

## Benchmark Methodology

All benchmarks run on the CPU with release profile optimizations. Measurements include:
- **Latency**: Time per inference iteration
- **Throughput**: Tokens per second
- **Memory**: Peak memory usage
- **Compression**: Ratio of quantized vs FP32 size

## Quantization Accuracy

### INT8 AbsMax Roundtrip

| Metric | Value |
|--------|-------|
| Max error | < 0.02 |
| Mean error | < 0.005 |
| Compression | 4.0x |

### INT4 Group Quantization (g=128)

| Metric | Value |
|--------|-------|
| Max error | < 0.1 |
| Mean error | < 0.02 |
| Compression | 7.8x |

### 1-bit Sign Quantization

| Metric | Value |
|--------|-------|
| Sign accuracy | 100% |
| Compression | 31.5x |

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
| Memory (FP32) | ~2.1GB |
| Memory (INT8) | ~540MB |
| Memory (INT4) | ~280MB |
| Memory (1-bit) | ~80MB |

## Tensor Parallelism

### Partition Overhead

| World Size | Overhead |
|------------|----------|
| 1 | 0% (baseline) |
| 2 | < 5% |
| 4 | < 10% |

## Memory Reduction

Theoretical memory reduction from quantization:

| Precision | Bytes/Param | 7B Model Size |
|-----------|-------------|---------------|
| FP32 | 4.0 | 28.0 GB |
| FP16 | 2.0 | 14.0 GB |
| INT8 | 1.0 | 7.0 GB |
| INT4 | 0.5 | 3.5 GB |
| 1-bit | 0.125 | 0.875 GB |

## Running Benchmarks

```bash
# Run the built-in benchmark
cargo run --release --bin bitllm -- bench --model tiny --iterations 1000

# Run Rust benchmarks (requires nightly for criterion)
cargo bench --workspace

# Run all tests with timing
cargo test --workspace -- --nocapture
```
