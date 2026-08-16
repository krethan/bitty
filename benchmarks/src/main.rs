mod correctness;
mod export;
mod far_memory;
mod helpers;
mod inference;
mod kernels;
mod matmul;
mod memory;
mod perplexity;
mod precision;
mod qat;
mod qat_ablation;
mod qat_sweep;
mod quantization;
mod train;

use bitllm_tensor::simd;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "bitty-bench", about = "Bitty LLM Inference Benchmark Suite")]
struct Cli {
    #[command(subcommand)]
    command: Option<BenchCmd>,
}

#[derive(Subcommand)]
enum BenchCmd {
    /// Run all benchmarks (default)
    Full,
    /// XNOR+popcount kernel throughput only
    Kernels,
    /// Precision comparison table only
    Precision,
    /// Quantization throughput only
    Quantization,
    /// Correctness verification only
    Correctness,
    /// Matmul benchmarks only
    Matmul,
    /// Memory footprint analysis only
    Memory,
    /// Token generation projection only
    Inference,
    /// Perplexity (FP32 vs bit1) on a synthetic model
    Perplexity,
    /// Ternary LoRA readout trained on frozen bigram hidden states
    Train,
    /// Full-graph STE-QAT vs naive quantization (logit MSE)
    Qat,
    /// QAT hyperparameter sweep (lr x steps grid per format)
    QatSweep,
    /// QAT per-projection ablation study
    QatAblation,
}

fn detect_cpu_features() {
    println!("\n=== CPU Feature Detection ===\n");
    println!("  SIMD: {}", simd::detect_simd_info());
    println!();
}

fn print_summary(precision_rows: &[export::PrecisionRow], export_path: Option<&str>) {
    println!("=== Bitty Summary ===\n");

    if let Some(best) = precision_rows.iter().max_by(|a, b| {
        a.compression_ratio
            .partial_cmp(&b.compression_ratio)
            .unwrap()
    }) {
        println!(
            "  Best compression:     {:<10} {:.1}x reduction",
            best.name, best.compression_ratio
        );
    }

    if let Some(best) = precision_rows
        .iter()
        .filter(|r| r.name != "FP32")
        .max_by(|a, b| a.cos_sim.partial_cmp(&b.cos_sim).unwrap())
    {
        println!(
            "  Best accuracy:        {:<10} {:.4}% cosine similarity",
            best.name,
            best.cos_sim * 100.0
        );
    }

    if let Some(best) = precision_rows
        .iter()
        .filter(|r| r.name != "FP32")
        .max_by(|a, b| {
            let a_speedup = precision_rows[0].matmul_ms / a.matmul_ms;
            let b_speedup = precision_rows[0].matmul_ms / b.matmul_ms;
            a_speedup.partial_cmp(&b_speedup).unwrap()
        })
    {
        let speedup = precision_rows[0].matmul_ms / best.matmul_ms;
        println!(
            "  Fastest matmul:       {:<10} {:.0}x vs FP32",
            best.name, speedup
        );
    }

    if let Some(best) = precision_rows
        .iter()
        .filter(|r| r.name != "FP32")
        .max_by(|a, b| a.tok_per_sec.partial_cmp(&b.tok_per_sec).unwrap())
    {
        println!(
            "  Fastest tok/s:        {:<10} {:.2} tok/s (Llama-7B projected)",
            best.name, best.tok_per_sec
        );
    }

    let llama_7b_params: f64 = 6_740_000_000.0;
    let matrix_elements: f64 = (1024 * 1024) as f64; // precision bench uses 1024x1024
    if let Some(best) = precision_rows
        .iter()
        .filter(|r| r.name != "FP32")
        .min_by(|a, b| a.weight_bytes.cmp(&b.weight_bytes))
    {
        let bytes_per_param = best.weight_bytes as f64 / matrix_elements;
        let gb = bytes_per_param * llama_7b_params / 1e9;
        println!(
            "  Smallest Llama-7B:    {:<10} {:.2} GB weights",
            best.name, gb
        );
    }

    if let Some(path) = export_path {
        println!("\n  Export: {}", path);
    }
    println!();
}

fn main() {
    let cli = Cli::parse();
    let run_all = cli.command.is_none() || matches!(cli.command, Some(BenchCmd::Full));

    println!("=== Bitty LLM Inference Benchmark Suite ===");
    println!("Build with: cargo run --release -p bitllm-benchmarks\n");

    detect_cpu_features();

    let cmd = cli.command.unwrap_or(BenchCmd::Full);

    match &cmd {
        BenchCmd::Kernels => kernels::bench_kernels(),
        BenchCmd::Precision => {}
        BenchCmd::Quantization => quantization::bench_quantization_throughput(),
        BenchCmd::Correctness => correctness::bench_correctness(),
        BenchCmd::Matmul => matmul::bench_matmul_suite(),
        BenchCmd::Memory => memory::bench_all_memory(),
        BenchCmd::Inference => inference::bench_token_generation(),
        BenchCmd::Perplexity => {}
        BenchCmd::Train => {}
        BenchCmd::Qat => {}
        BenchCmd::QatSweep => {}
        BenchCmd::QatAblation => {}
        BenchCmd::Full => {
            quantization::bench_quantization_throughput();
            correctness::bench_correctness();
            matmul::bench_matmul_suite();
            kernels::bench_kernels();
            memory::bench_all_memory();
            inference::bench_token_generation();
        }
    }

    // Always run precision + perplexity (unless user wants a single isolated benchmark)
    let precision_rows = if run_all || matches!(cmd, BenchCmd::Precision) {
        precision::bench_precision_comparison()
    } else {
        Vec::new()
    };

    let perplexity_rows = if run_all || matches!(cmd, BenchCmd::Perplexity) {
        perplexity::bench_perplexity()
    } else {
        Vec::new()
    };

    let memory_rows = if run_all || matches!(cmd, BenchCmd::Perplexity) {
        far_memory::bench_memory()
    } else {
        Vec::new()
    };

    let train_rows = if run_all || matches!(cmd, BenchCmd::Train) {
        train::bench_train()
    } else {
        Vec::new()
    };

    let qat_rows = if run_all || matches!(cmd, BenchCmd::Qat) {
        qat::bench_qat()
    } else {
        Vec::new()
    };

    let qat_sweep_rows = if matches!(cmd, BenchCmd::QatSweep) {
        qat_sweep::bench_qat_sweep()
    } else {
        Vec::new()
    };

    let qat_ablation_rows = if matches!(cmd, BenchCmd::QatAblation) {
        qat_ablation::bench_qat_ablation()
    } else {
        Vec::new()
    };

    // Export
    let results_dir = std::path::Path::new("benchmarks/results");
    let mut export_path: Option<String> = None;

    if !precision_rows.is_empty()
        || !perplexity_rows.is_empty()
        || !memory_rows.is_empty()
        || !train_rows.is_empty()
        || !qat_rows.is_empty()
        || !qat_sweep_rows.is_empty()
        || !qat_ablation_rows.is_empty()
    {
        let machine = export::collect_machine_info();
        let kernel_results: Vec<export::KernelBench> = kernels::collect_kernel_results();
        let export_data = export::build_export(
            machine,
            chrono::Local::now().format("%Y-%m-%d_%H:%M:%S").to_string(),
            kernel_results,
            precision_rows.clone(),
            perplexity_rows,
            memory_rows,
            train_rows,
            qat_rows,
            qat_sweep_rows,
            qat_ablation_rows,
        );

        match export::write_json(&export_data, results_dir) {
            Ok(p) => {
                println!("  JSON: {}", p);
                export_path = Some(p);
            }
            Err(e) => eprintln!("  JSON export failed: {}", e),
        }
        match export::write_csv(&export_data, results_dir) {
            Ok(p) => println!("  CSV:  {}", p),
            Err(e) => eprintln!("  CSV export failed: {}", e),
        }
    }

    // Summary
    if run_all || matches!(cmd, BenchCmd::Precision) {
        print_summary(&precision_rows, export_path.as_deref());
    }

    println!("=== Done ===");
}
