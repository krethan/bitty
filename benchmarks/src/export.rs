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
) -> BenchmarkExport {
    BenchmarkExport {
        benchmark_version: BENCHMARK_VERSION.to_string(),
        git_commit: git_commit_hash(),
        machine,
        timestamp,
        kernels,
        precision,
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

    fs::write(&path, &w)?;
    Ok(path.display().to_string())
}
