fn main() {
    #[cfg(feature = "rocm")]
    {
        let rocm_path = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());

        let hipcc = format!("{}/bin/hipcc", rocm_path);

        if !std::path::Path::new(&hipcc).exists() {
            println!(
                "cargo:warning=hipcc not found at {}, GPU kernels will not be compiled",
                hipcc
            );
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
