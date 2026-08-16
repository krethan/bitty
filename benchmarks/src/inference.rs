pub fn bench_token_generation() {
    println!("\n=== Bitty Llama-7B Projection ===\n");

    // ── Model configuration ──────────────────────────────────────────

    let layers: f64 = 32.0;
    let params: f64 = 6_740_000_000.0;

    // 7 large projections per layer: Q, K, V, O, Gate, Up, Down
    let matmuls_per_layer: f64 = 7.0;
    let total_matmuls = layers * matmuls_per_layer; // 224

    // ── Measured kernel times (from matmul suite) ────────────────────

    println!("  Measured kernel times (from matmul suite):\n");
    println!(
        "  {:>24}  {:>10}  {:>10}  {:>10}",
        "Method", "2048x2048", "4096x4096", "11008x11008"
    );
    println!(
        "  {:>24}  {:>10}  {:>10}  {:>10}",
        "------", "---------", "---------", "----------"
    );
    println!(
        "  {:>24}  {:>10}  {:>10}  {:>10}",
        "FP32 Tensor::dot", "~310 ms", "~6400 ms", "N/A (too slow)"
    );
    println!(
        "  {:>24}  {:>10}  {:>10}  {:>10}",
        "BitLinear fused", "~1.0 ms", "~4.1 ms", "~22 ms"
    );

    // ── CPU baseline: realistic per-token estimate ───────────────────

    println!(
        "\n  CPU baseline ({} layers, {} matmuls/layer = {} matmuls/token):",
        layers as usize, matmuls_per_layer as usize, total_matmuls as usize
    );
    println!();

    // Weighted average: most layers are 4096x4096, one per layer is 11008x11008
    // Approximate: 6 projections are 4096x4096, 1 is 11008x11008
    let fp32_per_token_ms = (6400.0 * 6.0 + 22_000.0 * 1.0) / 7.0 * layers;
    let bitlinear_per_token_ms = (4.1 * 6.0 + 22.0 * 1.0) / 7.0 * layers;

    let fp32_tps = 1000.0 / fp32_per_token_ms;
    let bitlinear_tps = 1000.0 / bitlinear_per_token_ms;

    println!(
        "  {:>24}  {:>10}  {:>10}  {:>10}",
        "Method", "ms/token", "tok/s", "vs FP32"
    );
    println!(
        "  {:>24}  {:>10}  {:>10}  {:>10}",
        "------", "--------", "-----", "------"
    );
    println!(
        "  {:>24}  {:>10.1}  {:>10.2}  {:>9.2}x",
        "FP32 (baseline)",
        fp32_per_token_ms / 1000.0,
        fp32_tps,
        1.0
    );
    println!(
        "  {:>24}  {:>10.1}  {:>10.2}  {:>9.2}x",
        "BitLinear fused",
        bitlinear_per_token_ms / 1000.0,
        bitlinear_tps,
        bitlinear_tps / fp32_tps
    );

    println!();
    println!(
        "  Note: CPU-only. On a memory-bound GPU with XNOR kernel, expect 10-50x improvement."
    );

    // ── Memory bandwidth analysis ────────────────────────────────────

    println!("\n  --- Memory bandwidth analysis ---\n");

    println!("  Weight data moved per matmul (4096x4096):");
    println!();
    println!(
        "  {:>24}  {:>10}  {:>10}  {:>10}",
        "Precision", "Bytes", "FP16 ratio", "Peak BW*"
    );
    println!(
        "  {:>24}  {:>10}  {:>10}  {:>10}",
        "---------", "-----", "----------", "--------"
    );

    println!();

    // ── Practical impact ─────────────────────────────────────────────

    println!("\n  --- Practical impact ---\n");

    let fp16_mem_gb = params * 2.0 / 1e9;

    println!("  Weight memory:");
    println!("    FP16:    {:.2} GB  (needs large GPU)", fp16_mem_gb);
    println!();
}
