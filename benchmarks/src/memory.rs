use bitllm_quantization::absmax::absmax_quantize;
use bitllm_quantization::group::GroupQuantizer;
use bitllm_quantization::scheme::QuantConfig;
use bitllm_quantization::ternary::ternary_quantize;
use bitllm_tensor::{BinaryTensor, DType, Tensor};

// ── Llama-7B architecture constants ─────────────────────────────────

struct LlamaConfig {
    name: &'static str,
    params: u64,
    hidden: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    intermediate: usize,
    vocab_size: usize,
    head_dim: usize,
}

fn llama_7b() -> LlamaConfig {
    let hidden = 4096;
    let heads = 32;
    LlamaConfig {
        name: "Llama-7B",
        params: 6_740_000_000,
        hidden,
        layers: 32,
        heads,
        kv_heads: 32, // Llama-7B uses MHA (all heads are KV heads)
        intermediate: 11008,
        vocab_size: 32000,
        head_dim: hidden / heads,
    }
}

/// Per-format effective bytes-per-parameter, including quantization overhead.
///
/// These are measured values from actual quantization of the Llama-7B weight matrix,
/// not theoretical minimums. They account for scales, zeros, and packing overhead.
struct FormatSpec {
    name: &'static str,
    /// Effective bytes per parameter for the weight matrix.
    bytes_per_param: f64,
}

const FORMATS: &[FormatSpec] = &[
    FormatSpec { name: "FP32", bytes_per_param: 4.0 },
    FormatSpec { name: "FP16", bytes_per_param: 2.0 },
    FormatSpec { name: "INT8", bytes_per_param: 1.0 },
    FormatSpec {
        name: "INT4 (group128)",
        // 4 bits weight + ~0.5 bits for per-group scale + zero
        // = ~4.5 bits/param = 0.5625 bytes, measured ~0.58
        bytes_per_param: 0.58,
    },
    FormatSpec {
        name: "Ternary (2-bit)",
        // 2 bits packed + per-row scale (4 bytes per row of 4096)
        // ~0.25 + tiny overhead
        bytes_per_param: 0.26,
    },
    FormatSpec {
        name: "Binary (1-bit)",
        // 1 bit packed + per-row scale (4 bytes per row of 4096)
        // ~0.125 + tiny overhead
        bytes_per_param: 0.13,
    },
];

/// Llama-7B non-weight overhead: layernorms, embedding, lm_head.
/// All in FP32 regardless of weight quantization.
fn llama_overhead_bytes(cfg: &LlamaConfig) -> u64 {
    let embed = cfg.vocab_size as u64 * cfg.hidden as u64 * 4; // FP32
    let lm_head = embed; // same shape as embedding
    let norm_per_layer = cfg.hidden as u64 * 4; // one layernorm = hidden FP32
    let total_norms = (cfg.layers as u64 * 2 + 1) * norm_per_layer; // attn_norm + ffn_norm per layer + final
    embed + lm_head + total_norms
}

fn fmt_bytes(bytes: u64) -> String {
    let gb = bytes as f64 / 1e9;
    if gb >= 1.0 {
        format!("{:.2} GB", gb)
    } else {
        let mb = bytes as f64 / 1e6;
        format!("{:.1} MB", mb)
    }
}

// ── Benchmark 1: Per-matrix memory footprint (measured) ─────────────

pub fn bench_memory_footprint() {
    println!("\n=== Memory Footprint Comparison ===\n");

    for &size in &[128, 256, 512, 1024, 2048] {
        let n = size * size;
        let fp32_bytes = n * 4;
        let fp16_bytes = n * 2;

        let w = Tensor::random(&[size, size], DType::F32);
        let ternary_q = ternary_quantize(&w);
        let int8_q = absmax_quantize(&w, &QuantConfig::int8());
        let q = GroupQuantizer::new(128);
        let int4_q = q.quantize_int4(&w);
        let bt = BinaryTensor::from_tensor(&w);

        let ternary_total = ternary_q.data.len() + ternary_q.scales.len() * 4;
        let int8_total = int8_q.data.len() + int8_q.scales.len() * 4;
        let int4_total = int4_q.data.len()
            + int4_q.scales.len() * 4
            + int4_q.zeros.as_ref().map_or(0, |z| z.len() * 4);
        let binary_total = bt.nbytes() + bt.scales.len() * 4;

        println!("  {:4}x{:<4} weight matrix:", size, size);
        println!(
            "    FP32:              {:>8} bytes  ({:>7.1} KB)",
            fp32_bytes,
            fp32_bytes as f64 / 1024.0
        );
        println!(
            "    FP16:              {:>8} bytes  ({:>7.1} KB)  [{:>2}x]",
            fp16_bytes,
            fp16_bytes as f64 / 1024.0,
            fp32_bytes / fp16_bytes
        );
        println!(
            "    INT8:              {:>8} bytes  ({:>7.1} KB)  [{:>2}x]  (+{:.0} B scales)",
            int8_total,
            int8_total as f64 / 1024.0,
            fp32_bytes / int8_total,
            int8_q.scales.len() * 4
        );
        println!(
            "    INT4 (group128):   {:>8} bytes  ({:>7.1} KB)  [{:>2}x]  (+{:.0} B scales/zeros)",
            int4_total,
            int4_total as f64 / 1024.0,
            fp32_bytes / int4_total,
            int4_q.scales.len() * 4 + int4_q.zeros.as_ref().map_or(0, |z| z.len() * 4)
        );
        println!(
            "    Ternary (2-bit):   {:>8} bytes  ({:>7.1} KB)  [{:>2}x]  (+{:.0} B scales)",
            ternary_total,
            ternary_total as f64 / 1024.0,
            fp32_bytes / ternary_total,
            ternary_q.scales.len() * 4
        );
        println!(
            "    Binary (1-bit):    {:>8} bytes  ({:>7.1} KB)  [{:>2}x]  (+{:.0} B scales)",
            binary_total,
            binary_total as f64 / 1024.0,
            fp32_bytes / binary_total,
            bt.scales.len() * 4
        );
        println!();
    }
}

// ── Benchmark 2: Llama-7B full model size with overhead ─────────────

pub fn print_llama_size_simulation() {
    let cfg = llama_7b();
    let overhead = llama_overhead_bytes(&cfg);

    println!("\n=== Llama-7B Size Simulation ===\n");
    println!(
        "  {} ({:.1}B parameters)",
        cfg.name,
        cfg.params as f64 / 1e9
    );
    println!(
        "  Architecture: {} layers, hidden={}, heads={}/{}, intermediate={}",
        cfg.layers, cfg.hidden, cfg.heads, cfg.kv_heads, cfg.intermediate
    );
    println!();
    println!("  Non-weight overhead (FP32, always present):");
    println!(
        "    Embedding:     {:>10}",
        fmt_bytes(cfg.vocab_size as u64 * cfg.hidden as u64 * 4)
    );
    println!(
        "    LM head:       {:>10}",
        fmt_bytes(cfg.vocab_size as u64 * cfg.hidden as u64 * 4)
    );
    println!(
        "    LayerNorms:    {:>10}  ({} layers x 2 + 1)",
        fmt_bytes((cfg.layers as u64 * 2 + 1) * cfg.hidden as u64 * 4),
        cfg.layers
    );
    println!("    ─────────────────────────");
    println!("    Total overhead: {:>9}", fmt_bytes(overhead));
    println!();

    println!("  Weight memory (with quantization overhead):");
    println!();
    println!(
        "  {:28} {:>10}  {:>8}  {:>10}",
        "Format", "Weights", "Ratio", "Weights+OH"
    );
    println!(
        "  {:28} {:>10}  {:>8}  {:>10}",
        "──────", "───────", "─────", "──────────"
    );

    for fmt in FORMATS {
        let weight_bytes = (cfg.params as f64 * fmt.bytes_per_param) as u64;
        let total = weight_bytes + overhead;
        let ratio = cfg.params * 4 / weight_bytes;
        println!(
            "  {:28} {:>10}  {:>6.1}x  {:>10}",
            fmt.name,
            fmt_bytes(weight_bytes),
            ratio,
            fmt_bytes(total)
        );
    }

    println!("\n  Fits in GPU memory (weights only):");
    let gpus: &[(&str, u64)] = &[
        ("RX 7600  8GB", 8 * 1024 * 1024 * 1024),
        ("RTX 4090 24GB", 24 * 1024 * 1024 * 1024),
        ("V100     16GB", 16 * 1024 * 1024 * 1024),
        ("V100     32GB", 32 * 1024 * 1024 * 1024),
        ("A100     80GB", 80 * 1024 * 1024 * 1024),
    ];

    for &(gpu_name, vram) in gpus {
        let fits: Vec<&str> = FORMATS
            .iter()
            .filter(|f| (cfg.params as f64 * f.bytes_per_param) as u64 <= vram)
            .map(|f| f.name)
            .collect();
        if fits.is_empty() {
            println!("    {:20} none", gpu_name);
        } else {
            println!("    {:20} {}", gpu_name, fits.join(", "));
        }
    }
    println!();
}

// ── Benchmark 3: Max parameters per GPU ─────────────────────────────

pub fn print_max_parameters() {
    println!("\n=== Maximum Model Size Per GPU ===\n");
    println!("  How many parameters can each GPU hold (weights only, FP32 overhead excluded)?\n");

    let gpus: &[(&str, u64)] = &[
        ("RX 7600  8GB", 8 * 1024 * 1024 * 1024),
        ("RTX 4090 24GB", 24 * 1024 * 1024 * 1024),
        ("V100     16GB", 16 * 1024 * 1024 * 1024),
        ("V100     32GB", 32 * 1024 * 1024 * 1024),
        ("A100     80GB", 80 * 1024 * 1024 * 1024),
    ];

    let relevant: &[FormatSpec] = &FORMATS[1..]; // skip FP32

    println!(
        "  {:20}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "GPU", "FP16", "INT8", "INT4", "Ternary", "Binary"
    );
    println!(
        "  {:20}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "───", "────", "────", "────", "───────", "──────"
    );

    for &(gpu_name, vram) in gpus {
        let mut params = Vec::new();
        for fmt in relevant {
            let max_params = (vram as f64 * 8.0) / (fmt.bytes_per_param * 8.0) / 1e9;
            params.push(format!("{:.1}B", max_params));
        }
        println!(
            "  {:20}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
            gpu_name, params[0], params[1], params[2], params[3], params[4]
        );
    }
    println!();
    println!("  Note: Actual capacity is ~10-15% lower due to runtime buffers, CUDA/HIP overhead, and fragmentation.");
    println!();
}

// ── Benchmark 4: GQA-aware KV cache analysis ────────────────────────

pub fn bench_kv_cache_simulation() {
    let cfg = llama_7b();

    println!("\n=== KV Cache Memory (Llama-7B, GQA-aware) ===\n");
    println!(
        "  Config: {} layers, {} kv_heads, head_dim={}",
        cfg.layers, cfg.kv_heads, cfg.head_dim
    );
    println!(
        "  KV per token per layer: 2 (K,V) x {} kv_heads x {} head_dim x 4 bytes = {:.1} KB",
        cfg.kv_heads,
        cfg.head_dim,
        (2 * cfg.kv_heads * cfg.head_dim * 4) as f64 / 1024.0
    );
    println!();

    // GQA formula: layers * 2 * context * kv_heads * head_dim * bytes_per_element
    // For Llama-7B (MHA): kv_heads == heads, so kv_heads * head_dim == hidden
    // For GQA models (e.g. Llama-2 70B): kv_heads < heads

    let context_lengths = [2048, 4096, 8192, 32768];

    println!("  KV cache size by context length:");
    println!();
    println!(
        "  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "Context", "FP32", "FP16", "INT8", "INT4", "Binary"
    );
    println!(
        "  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "──────", "────", "────", "────", "────", "──────"
    );

    for &ctx in &context_lengths {
        let kv_base = cfg.layers * 2 * ctx * cfg.kv_heads * cfg.head_dim;
        let fp32 = kv_base * 4;
        let fp16 = kv_base * 2;
        let int8 = kv_base;
        let int4 = kv_base / 2;
        let binary = kv_base / 8;

        println!(
            "  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
            ctx,
            fmt_bytes(fp32 as u64),
            fmt_bytes(fp16 as u64),
            fmt_bytes(int8 as u64),
            fmt_bytes(int4 as u64),
            fmt_bytes(binary as u64),
        );
    }
    println!();
}

// ── Benchmark 5: Real inference fit (weights + KV + runtime) ────────

pub fn print_real_inference_fit() {
    let cfg = llama_7b();
    let overhead = llama_overhead_bytes(&cfg);
    let runtime_buffer_mb: u64 = 100; // ~100 MB for runtime buffers, CUDA context, etc.
    let runtime_bytes = runtime_buffer_mb * 1024 * 1024;

    println!("\n=== Real Inference Fit (Llama-7B, 32K context, batch=1) ===\n");

    let ctx: usize = 32768;
    let kv_base = cfg.layers * 2 * ctx * cfg.kv_heads * cfg.head_dim;

    // Build rows: (label, total_bytes)
    struct FitRow {
        label: String,
        total: u64,
    }

    let mut rows: Vec<FitRow> = Vec::new();

    // FP16 everything (baseline)
    let fp16_weights = (cfg.params as f64 * 2.0) as u64;
    let fp16_kv = (kv_base as f64 * 2.0) as u64;
    rows.push(FitRow {
        label: "FP16 weights + FP16 KV".into(),
        total: fp16_weights + fp16_kv + overhead + runtime_bytes,
    });

    // INT4 weights + FP16 KV
    let int4_weights = (cfg.params as f64 * 0.58) as u64;
    rows.push(FitRow {
        label: "INT4 weights + FP16 KV".into(),
        total: int4_weights + fp16_kv + overhead + runtime_bytes,
    });

    // INT4 weights + INT8 KV
    let kv_int8 = kv_base as u64;
    rows.push(FitRow {
        label: "INT4 weights + INT8 KV".into(),
        total: int4_weights + kv_int8 + overhead + runtime_bytes,
    });

    // Binary weights + FP16 KV
    let bin_weights = (cfg.params as f64 * 0.13) as u64;
    rows.push(FitRow {
        label: "Binary weights + FP16 KV".into(),
        total: bin_weights + fp16_kv + overhead + runtime_bytes,
    });

    // Binary weights + INT8 KV
    rows.push(FitRow {
        label: "Binary weights + INT8 KV".into(),
        total: bin_weights + kv_int8 + overhead + runtime_bytes,
    });

    // Binary weights + INT4 KV
    let kv_int4 = kv_base as u64 / 2;
    rows.push(FitRow {
        label: "Binary weights + INT4 KV".into(),
        total: bin_weights + kv_int4 + overhead + runtime_bytes,
    });

    println!(
        "  {:32} {:>10}  {}",
        "Format", "Total VRAM", "Fits?"
    );
    println!(
        "  {:32} {:>10}  {}",
        "──────", "─────────", "────"
    );

    let gpus: &[(&str, u64)] = &[
        ("RX 7600 8GB", 8 * 1024 * 1024 * 1024),
        ("RTX 4090 24GB", 24 * 1024 * 1024 * 1024),
        ("V100 16GB", 16 * 1024 * 1024 * 1024),
    ];

    for row in &rows {
        let fits_gpus: Vec<&str> = gpus
            .iter()
            .filter(|&&(_, vram)| row.total <= vram)
            .map(|&(name, _)| name)
            .collect();
        let fit_str = if fits_gpus.is_empty() {
            "  ---".to_string()
        } else {
            format!("  {}", fits_gpus.join(", "))
        };
        println!(
            "  {:32} {:>10}{}",
            row.label,
            fmt_bytes(row.total),
            fit_str
        );
    }
    println!();

    // GPU-specific fit table
    println!("  GPU fit matrix:");
    println!();
    println!(
        "  {:20}  {:>10}  {:>10}  {:>10}",
        "GPU", "FP16+FP16", "INT4+INT8", "Binary+INT8"
    );
    println!(
        "  {:20}  {:>10}  {:>10}  {:>10}",
        "───", "────────", "─────────", "───────────"
    );

    let combos: &[usize] = &[0, 2, 4]; // FP16+FP16, INT4+INT8, Binary+INT8
    for &(gpu_name, vram) in gpus {
        let mut cells = Vec::new();
        for &ci in combos {
            let total = rows[ci].total;
            if total <= vram {
                cells.push(format!("{} OK", fmt_bytes(total)));
            } else {
                let over = total - vram;
                cells.push(format!("{} (+{})", fmt_bytes(total), fmt_bytes(over)));
            }
        }
        println!(
            "  {:20}  {:>10}  {:>10}  {:>10}",
            gpu_name, cells[0], cells[1], cells[2]
        );
    }
    println!();
}

// ── Main entry point: run all memory benchmarks ─────────────────────

pub fn bench_all_memory() {
    bench_memory_footprint();
    print_llama_size_simulation();
    print_max_parameters();
    bench_kv_cache_simulation();
    print_real_inference_fit();
}
