fn is_wsl() -> bool {
    std::fs::read_to_string("/proc/version")
        .map(|v| v.to_lowercase().contains("microsoft"))
        .unwrap_or(false)
}

fn main() {
    let has_hipcc = std::process::Command::new("hipcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if has_hipcc {
        let kernel_dir = if is_wsl() {
            "../../kernels/hip_tern"
        } else {
            "kernels/hip_tern"
        };

        let arch_flags = if is_wsl() {
            vec!["--offload-arch=gfx1102"]
        } else {
            vec![
                "--offload-arch=gfx900",
                "--offload-arch=gfx90a",
                "--offload-arch=gfx940",
                "--offload-arch=gfx1100",
                "--offload-arch=gfx1102",
            ]
        };

        let mut cmd = std::process::Command::new("hipcc");
        cmd.args(["-c", &format!("{}/hip_tern_kernel.h", kernel_dir)]);
        for arch in &arch_flags {
            cmd.arg(arch);
        }
        cmd.args(["-o", "hip_tern_kernel.o", "-std=c++17"]);
        let status = cmd
            .current_dir("src")
            .status()
            .expect("Failed to invoke hipcc for kernel compilation");
        assert!(status.success(), "hipcc compilation failed");

        let mut link_cmd = std::process::Command::new("hipcc");
        link_cmd.args(["-shared", "hip_tern_kernel.o", "-o", "hip_tern_32.hsaco"]);
        for arch in &arch_flags {
            link_cmd.arg(arch);
        }
        let status = link_cmd
            .current_dir("src")
            .status()
            .expect("Failed to link .hsaco");
        assert!(status.success(), "hipcc linking failed");

        println!("cargo:warning=HIP kernel compiled successfully for gfx1102 (RX 7600)");
    } else {
        println!("cargo:warning=HIP kernel compilation skipped (hipcc not available)");
        println!("cargo:warning=In a WSL2 + RX 7600 environment, ensure ROCm is installed on the Windows host");
        println!("cargo:warning=and the AMD GPU driver is enabled in WSL2 (dxgkrnl).");
        println!("cargo:warning=Compile with:");
        println!("cargo:warning=hipcc -c kernels/hip_tern/hip_tern_kernel.h -o hip_tern_kernel.o --offload-arch=gfx1102");
        println!("cargo:warning=hipcc -shared hip_tern_kernel.o -o hip_tern_32.hsaco --offload-arch=gfx1102");
    }
}
