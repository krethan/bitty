fn main() {
    let has_hipcc = std::process::Command::new("hipcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if has_hipcc {
        let status = std::process::Command::new("hipcc")
            .args([
                "-c",
                "kernels/hip_tern/hip_tern_kernel.h",
                "-o",
                "hip_tern_kernel.o",
            ])
            .current_dir("src")
            .status()
            .expect("Failed to invoke hipcc for kernel compilation");
        assert!(status.success(), "hipcc compilation failed");
        let status = std::process::Command::new("hipcc")
            .args(["-shared", "hip_tern_kernel.o", "-o", "hip_tern_32.hsaco"])
            .current_dir("src")
            .status()
            .expect("Failed to link .hsaco");
        assert!(status.success(), "hipcc linking failed");

        println!("cargo:warning=HIP kernel compiled successfully");
    } else {
        println!("cargo:warning=HIP kernel compilation skipped (hipcc not available)");
        println!("cargo:warning=In a real AMD ROCm environment, compile with:");
        println!("cargo:warning=hipcc -c kernels/hip_tern/hip_tern_kernel.h -o hip_tern_kernel.o");
        println!("cargo:warning=hipcc -shared hip_tern_kernel.o -o hip_tern_32.hsaco");
    }
}
