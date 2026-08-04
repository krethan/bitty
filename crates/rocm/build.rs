fn is_wsl() -> bool {
    std::fs::read_to_string("/proc/version")
        .map(|v| v.to_lowercase().contains("microsoft"))
        .unwrap_or(false)
}

fn detect_rocm_path() -> Option<String> {
    if let Ok(path) = std::env::var("ROCM_PATH") {
        return Some(path);
    }
    if is_wsl() {
        let candidates = [
            "/opt/rocm".to_string(),
            "/usr/local/rocm".to_string(),
        ];
        for c in &candidates {
            if std::path::Path::new(&format!("{}/bin/hipcc", c)).exists() {
                return Some(c.clone());
            }
        }
    }
    None
}

fn main() {
    #[cfg(feature = "rocm")]
    {
        let rocm_path = detect_rocm_path().unwrap_or_else(|| "/opt/rocm".to_string());

        let hipcc = format!("{}/bin/hipcc", rocm_path);

        if !std::path::Path::new(&hipcc).exists() {
            println!(
                "cargo:warning=hipcc not found at {}, GPU kernels will not be compiled",
                hipcc
            );
            if is_wsl() {
                println!("cargo:warning=WSL2 detected: ensure AMD ROCm is installed on the Windows host");
                println!("cargo:warning=and the AMD GPU driver is enabled in WSL2 (dxgkrnl).");
                println!("cargo:warning=Set ROCM_PATH env var if ROCm is installed in a non-standard location.");
            }
            return;
        }

        println!("cargo:rerun-if-changed=src/kernels/");

        let kernels = ["element_wise", "matmul", "softmax", "rope"];

        let mut build = cc::Build::new();
        build
            .compiler(&hipcc)
            .cpp(true)
            .flag("--offload-arch=gfx900")
            .flag("--offload-arch=gfx90a")
            .flag("--offload-arch=gfx940")
            .flag("--offload-arch=gfx1100")
            .flag("--offload-arch=gfx1102")
            .flag("-std=c++17")
            .include(format!("{}/include", rocm_path));

        for kernel in &kernels {
            let src = format!("src/kernels/{}.hip", kernel);
            if std::path::Path::new(&src).exists() {
                build.file(&src);
            }
        }

        build.compile("bitllm_kernels");

        println!("cargo:rustc-link-lib=dylib=amdhip64");
        println!("cargo:rustc-link-search=native={}/lib", rocm_path);
    }
}
