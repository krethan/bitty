//! AMD RDNA3 HIP-based Ternary-Exponential Attention Kernel for Project Mercurial
//!
//! This crate provides the core components for running 1.005-bit LLMs on AMD RX 7600 GPUs.
//!
//! ## Features
//!
//! - `hip`: Enable HIP-specific functionality for AMD GPUs. Requires `hip-runtime-sys` on Linux.

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

mod mercurial;

pub use mercurial::{
    DeltaBinaryKVCache, MercurialBuilder, MercurialConfig, MercurialModel, StochasticFlip,
    TernaryQuantizer, TrainingConfig,
};

#[cfg(feature = "hip")]
use hip_runtime_sys as hip;

#[cfg(not(feature = "hip"))]
mod hip {
    pub type hipModule_t = usize;
    pub type hipFunction_t = usize;
    pub type hipStream_t = usize;
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
pub struct HipTernConfig {
    pub q_ptr: u64,
    pub k_ptr: u64,
    pub v_ptr: u64,
    pub out_ptr: u64,
    pub tokens: i32,
    pub seq_len: i32,
    pub d64: i32,
    pub wavefront_size: i32,
}

impl HipTernConfig {
    pub fn new(d64: i32) -> Self {
        Self {
            q_ptr: 0,
            k_ptr: 0,
            v_ptr: 0,
            out_ptr: 0,
            tokens: 0,
            seq_len: 0,
            d64,
            wavefront_size: 32,
        }
    }
}

pub struct HipTernKernel {
    module: hip::hipModule_t,
    func: hip::hipFunction_t,
    stream: hip::hipStream_t,
    config: std::sync::Arc<HipTernConfig>,
}

impl HipTernKernel {
    pub fn module(&self) -> hip::hipModule_t {
        self.module
    }

    pub fn func(&self) -> hip::hipFunction_t {
        self.func
    }

    pub fn config(&self) -> &HipTernConfig {
        &self.config
    }

    pub fn stream(&self) -> hip::hipStream_t {
        self.stream
    }

    pub fn new(d64: usize) -> Result<Self, String> {
        let config = std::sync::Arc::new(HipTernConfig::new(d64 as i32));

        #[cfg(feature = "hip")]
        {
            unsafe {
                let mut module: hip::hipModule_t = 0;
                let module_name = format!("hip_tern_{}.hsaco", d64);
                let c_module_name = std::ffi::CString::new(module_name.clone())
                    .map_err(|_| format!("Failed to create CString for {}", module_name))?;

                let status = hip::hipModuleLoad(&mut module, c_module_name.as_ptr());
                if status != hip::hipSuccess {
                    return Err(format!("Failed to load HIP module: {:?}", status));
                }

                let func_name = b"hip_tern_kernel\0";
                let mut func: hip::hipFunction_t = 0;
                let c_func_name = std::ffi::CString::new(func_name.to_vec())
                    .map_err(|_| format!("Failed to create CString for {}", func_name))?;

                let status = hip::hipModuleGetFunction(&mut func, module, c_func_name.as_ptr());
                if status != hip::hipSuccess {
                    return Err(format!("Failed to get HIP function: {:?}", status));
                }

                let mut stream: hip::hipStream_t = 0;
                let status = hip::hipStreamCreate(&mut stream);
                if status != hip::hipSuccess {
                    return Err(format!("Failed to create HIP stream: {:?}", status));
                }

                Ok(Self {
                    module,
                    func,
                    stream,
                    config,
                })
            }
        }
        #[cfg(not(feature = "hip"))]
        {
            Ok(Self {
                module: 0,
                func: 0,
                stream: 0,
                config,
            })
        }
    }

    /// Launch the HIP ternary attention kernel.
    ///
    /// # Safety
    /// All pointer arguments must point to valid GPU-accessible memory of the correct size.
    /// The caller must ensure the kernel config matches the actual tensor dimensions.
    pub unsafe fn launch(
        &self,
        _q_ptr: *const u64,
        _k_ptr: *const u64,
        _v_ptr: *const half::f16,
        _out_ptr: *mut half::f16,
        _tokens: usize,
        _seq_len: usize,
    ) -> Result<(), String> {
        #[cfg(feature = "hip")]
        {
            let config_value = *self.config;
            let mut config_ptr: *mut std::ffi::c_void = &mut config_value as *const _ as *mut _;

            let status = hip::hipLaunchKernel(
                self.func,
                0,
                0,
                0,
                config_ptr as *mut std::ffi::c_void,
                self.stream,
                0,
                std::ptr::null(),
            );

            if status != hip::hipSuccess {
                return Err(format!("HIP kernel launch failed: {:?}", status));
            }

            let status = hip::hipStreamSynchronize(self.stream);
            if status != hip::hipSuccess {
                return Err(format!("HIP stream sync failed: {:?}", status));
            }

            Ok(())
        }
        #[cfg(not(feature = "hip"))]
        {
            Err("HIP not available (enable `hip` feature)".into())
        }
    }
}

impl Drop for HipTernKernel {
    fn drop(&mut self) {
        #[cfg(feature = "hip")]
        unsafe {
            hip::hipStreamDestroy(self.stream);
            hip::hipModuleUnload(self.module);
        }
    }
}

pub struct WeightStreamer {
    pub weights_ram: Vec<u8>,
    pub weights_vram: Vec<u8>,
    pub current_buffer: usize,
    pub pcie_width: usize,
}

impl WeightStreamer {
    pub fn new(total_bits: usize) -> Self {
        let ram_size = total_bits.div_ceil(8);
        let vram_size = ram_size * 2;
        Self {
            weights_ram: vec![0; ram_size],
            weights_vram: vec![0; vram_size],
            current_buffer: 0,
            pcie_width: 8,
        }
    }

    /// Stream a layer's weights from RAM to VRAM via PCIe.
    ///
    /// # Safety
    /// The caller must ensure `layer_idx` is within bounds and `hip_stream` is a valid stream.
    pub unsafe fn stream_layer(
        &mut self,
        layer_idx: usize,
        hip_stream: hip::hipStream_t,
    ) -> Result<(), String> {
        let src = self.weights_ram.as_ptr().add(layer_idx * self.pcie_width);
        let dst = self
            .weights_vram
            .as_mut_ptr()
            .add(self.current_buffer * self.pcie_width);
        let size = self.pcie_width;

        #[cfg(feature = "hip")]
        {
            let status = hip::hipMemcpyAsync(
                dst as *mut std::ffi::c_void,
                src as *const std::ffi::c_void,
                size,
                hip::hipMemcpyHostToDevice,
                hip_stream,
            );

            if status != hip::hipSuccess {
                return Err(format!("PCIe streaming failed: {:?}", status));
            }

            self.current_buffer = 1 - self.current_buffer;
            Ok(())
        }
        #[cfg(not(feature = "hip"))]
        {
            let _ = (src, dst, size, hip_stream);
            self.current_buffer = 1 - self.current_buffer;
            Ok(())
        }
    }
}
