use serde::Serialize;
use std::fs;
use std::io;
use std::path::Path;

const BENCHMARK_VERSION: &str = "1.0.0";

#[derive(Serialize)]
pub struct BenchmarkExport {
    pub benchmark_version: String,
    pub git_commit: Option<String>,
    pub machine: MachineInfo,
    pub timestamp: String,
    pub kernels: Vec<KernelBench>,
    pub precision: Vec<PrecisionRow>,
    pub perplexity: Vec<PerplexityRow>,
    pub memory: Vec<MemoryRow>,
    pub train: Vec<TrainRow>,
    pub qat: Vec<QatRow>,
    pub qat_sweep: Vec<QatSweepRow>,
    pub qat_ablation: Vec<QatAblationRow>,
}

#[derive(Serialize)]
pub struct MachineInfo {
    pub os: String,
    pub cpu_model: String,
    pub ram_gb: f64,
}

#[derive(Serialize)]
pub struct KernelBench {
    pub name: String,
    pub size: usize,
    pub time_ms: f64,
    pub gbps: f64,
    pub tbit_per_sec: f64,
}

#[derive(Clone, Serialize)]
pub struct PrecisionRow {
    pub name: String,
    pub weight_bytes: usize,
    pub compression_ratio: f64,
    pub cos_sim: f64,
    pub rel_rmse_pct: f64,
    pub max_err: f64,
    pub matmul_ms: f64,
    pub tok_per_sec: f64,
}

#[derive(Clone, Serialize)]
pub struct PerplexityRow {
    pub name: String,
    pub perplexity: f64,
    pub bits_per_token: f64,
    pub nll_sum: f64,
    pub n_tokens: usize,
    pub ctx_len: usize,
}

#[derive(Clone, Serialize)]
pub struct MemoryRow {
    pub mode: String,
    pub overall_ppl: f64,
    pub echo_ppl: f64,
    pub recall_at_1: f64,
    pub memory_bytes: usize,
    pub dense_bytes: usize,
    pub compression_ratio: f64,
}

#[derive(Clone, Serialize)]
pub struct TrainRow {
    pub name: String,
    pub ppl: f64,
    pub ppl_fp32: f64,
    pub rank: usize,
    pub init_scale: f64,
    pub sweeps: u64,
    pub train_mse: f64,
    pub weight_bytes: usize,
    pub fp32_bytes: usize,
    pub compression_ratio: f64,
}

#[derive(Clone, Serialize)]
pub struct QatRow {
    pub name: String,
    pub naive_mse: f64,
    pub qat_mse: f64,
    pub mse_ratio: f64,
    pub ppl_naive: f64,
    pub ppl_qat: f64,
    pub steps: usize,
}

#[derive(Clone, Serialize)]
pub struct QatSweepRow {
    pub format: String,
    pub lr: f64,
    pub steps: usize,
    pub naive_mse: f64,
    pub qat_mse: f64,
    pub mse_ratio: f64,
    pub ppl_naive: f64,
    pub ppl_qat: f64,
    pub train_mse_start: f64,
    pub train_mse_end: f64,
}

#[derive(Clone, Serialize)]
pub struct QatAblationRow {
    pub format: String,
    pub ablation: String,
    pub projections: String,
    pub naive_mse: f64,
    pub qat_mse: f64,
    pub mse_ratio: f64,
    pub mse_improvement_pct: f64,
    pub ppl_naive: f64,
    pub ppl_qat: f64,
    pub train_mse_start: f64,
    pub train_mse_end: f64,
}

pub fn collect_machine_info() -> MachineInfo {
    let os = std::env::var("OSTYPE").unwrap_or_else(|_| std::env::consts::OS.to_string());
    let cpu_model = fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string());

    let ram_gb = fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemTotal"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|s| s.parse::<f64>().ok())
                .map(|kb| kb / 1024.0 / 1024.0)
        })
        .unwrap_or(0.0);

    MachineInfo {
        os,
        cpu_model,
        ram_gb,
    }
}

pub fn git_commit_hash() -> Option<String> {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok().map(|s| s.trim().to_string())
            } else {
                None
            }
        })
}

pub fn build_export(
    machine: MachineInfo,
    timestamp: String,
    kernels: Vec<KernelBench>,
    precision: Vec<PrecisionRow>,
    perplexity: Vec<PerplexityRow>,
    memory: Vec<MemoryRow>,
    train: Vec<TrainRow>,
    qat: Vec<QatRow>,
    qat_sweep: Vec<QatSweepRow>,
    qat_ablation: Vec<QatAblationRow>,
) -> BenchmarkExport {
    BenchmarkExport {
        benchmark_version: BENCHMARK_VERSION.to_string(),
        git_commit: git_commit_hash(),
        machine,
        timestamp,
        kernels,
        precision,
        perplexity,
        memory,
        train,
        qat,
        qat_sweep,
        qat_ablation,
    }
}

pub fn write_json(export: &BenchmarkExport, dir: &Path) -> io::Result<String> {
    fs::create_dir_all(dir)?;
    let ts = export.timestamp.replace([':', ' '], "-");
    let filename = format!("bitty_{}.json", ts);
    let path = dir.join(&filename);
    let json = serde_json::to_string_pretty(export)?;
    fs::write(&path, &json)?;
    Ok(path.display().to_string())
}

pub fn write_csv(export: &BenchmarkExport, dir: &Path) -> io::Result<String> {
    fs::create_dir_all(dir)?;
    let ts = export.timestamp.replace([':', ' '], "-");
    let filename = format!("bitty_{}.csv", ts);
    let path = dir.join(&filename);

    let mut w = String::new();
    w.push_str("name,weight_bytes,compression_ratio,cos_sim,rel_rmse_pct,max_err,matmul_ms,tok_per_sec\n");
    for row in &export.precision {
        w.push_str(&format!(
            "{},{},{:.2},{:.6},{:.4},{:.6},{:.3},{:.2}\n",
            row.name,
            row.weight_bytes,
            row.compression_ratio,
            row.cos_sim,
            row.rel_rmse_pct,
            row.max_err,
            row.matmul_ms,
            row.tok_per_sec,
        ));
    }

    w.push_str("\nname,perplexity,bits_per_token,nll_sum,n_tokens,ctx_len\n");
    for row in &export.perplexity {
        w.push_str(&format!(
            "{},{:.4},{:.4},{:.4},{},{}\n",
            row.name, row.perplexity, row.bits_per_token, row.nll_sum, row.n_tokens, row.ctx_len,
        ));
    }

    w.push_str("\nmode,overall_ppl,echo_ppl,recall_at_1,memory_bytes,dense_bytes,compression_ratio\n");
    for row in &export.memory {
        w.push_str(&format!(
            "{},{:.4},{:.4},{:.4},{},{},{:.2}\n",
            row.mode,
            row.overall_ppl,
            row.echo_ppl,
            row.recall_at_1,
            row.memory_bytes,
            row.dense_bytes,
            row.compression_ratio,
        ));
    }

    w.push_str("\nname,ppl,ppl_fp32,rank,init_scale,sweeps,train_mse,weight_bytes,fp32_bytes,compression_ratio\n");
    for row in &export.train {
        w.push_str(&format!(
            "{},{:.4},{:.4},{},{:.3},{},{:.6},{},{},{:.2}\n",
            row.name,
            row.ppl,
            row.ppl_fp32,
            row.rank,
            row.init_scale,
            row.sweeps,
            row.train_mse,
            row.weight_bytes,
            row.fp32_bytes,
            row.compression_ratio,
        ));
    }

    w.push_str("\nname,naive_mse,qat_mse,mse_ratio,ppl_naive,ppl_qat,steps\n");
    for row in &export.qat {
        w.push_str(&format!(
            "{},{:.6},{:.6},{:.4},{:.2},{:.2},{}\n",
            row.name, row.naive_mse, row.qat_mse, row.mse_ratio, row.ppl_naive, row.ppl_qat,
            row.steps,
        ));
    }

    w.push_str("\nformat,lr,steps,naive_mse,qat_mse,mse_ratio,ppl_naive,ppl_qat,train_mse_start,train_mse_end\n");
    for row in &export.qat_sweep {
        w.push_str(&format!(
            "{},{:.6},{},{:.6},{:.6},{:.4},{:.2},{:.2},{:.6},{:.6}\n",
            row.format, row.lr, row.steps, row.naive_mse, row.qat_mse, row.mse_ratio,
            row.ppl_naive, row.ppl_qat, row.train_mse_start, row.train_mse_end,
        ));
    }

    w.push_str("\nformat,ablation,projections,naive_mse,qat_mse,mse_ratio,mse_improvement_pct,ppl_naive,ppl_qat,train_mse_start,train_mse_end\n");
    for row in &export.qat_ablation {
        w.push_str(&format!(
            "{},{},{},{:.6},{:.6},{:.4},{:.2},{:.2},{:.2},{:.6},{:.6}\n",
            row.format, row.ablation, row.projections, row.naive_mse, row.qat_mse, row.mse_ratio,
            row.mse_improvement_pct, row.ppl_naive, row.ppl_qat, row.train_mse_start, row.train_mse_end,
        ));
    }

    fs::write(&path, &w)?;
    Ok(path.display().to_string())
}
