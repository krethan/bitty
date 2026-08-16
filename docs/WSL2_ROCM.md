# WSL2 + AMD RX 7600 ROCm Setup

This guide covers setting up AMD ROCm for the RX 7600 (gfx1102, Navi 33) in WSL2 on Windows.

## Prerequisites

- Windows 11 (build 22H2 or later)
- WSL2 enabled (`wsl --install`)
- AMD RX 7600 GPU with latest Adrenalin drivers on Windows
- At least 16 GB RAM (8 GB for GPU VRAM + system)

## Windows Host Setup

1. Install AMD ROCm on Windows:
   - Download from https://www.amd.com/en/developer/rocm/install.html
   - ROCm 6.2+ is required for RDNA3 (gfx1102) support
   - Install the full ROCm stack including HIP SDK

2. Enable WSL2 GPU support:
   - Windows 11 22H2+ has built-in WSL2 GPU support via dxgkrnl
   - No additional driver needed beyond the standard AMD Adrenalin driver

3. Verify GPU passthrough:
   ```powershell
   wsl lspci | grep -i amd
   ```
   You should see the RX 7600 listed.

## WSL2 Linux Setup

1. In WSL2, install ROCm runtime packages:
   ```bash
   sudo apt update
   sudo apt install rocm-hip-sdk rocm-opencl rocm-utils
   ```

2. Add your user to the render and video groups:
   ```bash
   sudo usermod -aG render,video $USER
   ```

3. Reboot WSL2:
   ```powershell
   wsl --shutdown
   ```

4. Verify GPU access:
   ```bash
   ls /dev/kfd /dev/dri/card0
   rocm-smi
   ```

## Building with ROCm Support

The `bitllm-rocm` crate is a CPU stub by default; the runtime pulls it in via the
`gpu` feature, and the real HIP kernels only compile when the `rocm` feature of
`bitllm-rocm` is enabled on a host with ROCm installed.

```bash
# Runtime with the GPU backend wired in (CPU-stub kernels unless ROCm is present)
cargo build -p bitllm-runtime --features gpu

# hip_tern kernel for RX 7600 (gfx1102) — decoupled from the workspace,
# so it must be built via its own manifest (auto-detects WSL2 and targets gfx1102)
cargo build --manifest-path crates/hip_tern/Cargo.toml --features hip
```

## Running

```bash
# Set ROCM_PATH if ROCm is in a non-standard location
export ROCM_PATH=/opt/rocm

# Run the server with GPU acceleration (falls back to CPU if ROCm is unavailable)
cargo run -p bitllm-server -- --gpu
```

## Troubleshooting

### `hipcc not found`
- ROCm is not installed in WSL2 or `ROCM_PATH` is wrong
- Set `ROCM_PATH` to the ROCm installation directory
- Run `which hipcc` to find the correct path

### `No ROCm devices found in WSL2`
- AMD GPU driver is not enabled in WSL2
- Ensure the Windows Adrenalin driver is up to date
- Check that `dxgkrnl` is loaded: `wslcat /proc/driver/amdgpu/version`
- Restart WSL2: `wsl --shutdown` then reopen

### gfx1102 not supported by installed ROCm version
- ROCm 5.7+ is required for RDNA3 consumer GPU support
- Upgrade ROCm on the Windows host
- As a workaround, set `HSA_OVERRIDE_GFX_VERSION=11.0.0` in WSL2:
  ```bash
  export HSA_OVERRIDE_GFX_VERSION=11.0.0
  ```

### Kernel compilation fails
- Ensure `hipcc` is available in PATH
- Check that the HIP SDK is installed: `hipcc --version`
- Verify `/opt/rocm/include/hip/hip_runtime.h` exists