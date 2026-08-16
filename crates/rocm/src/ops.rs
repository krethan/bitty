use crate::memory::GpuBuffer;
use crate::{Result, RocmError};

pub struct GpuOps;

impl GpuOps {
    pub fn f32_add(a: &GpuBuffer, b: &GpuBuffer, out: &GpuBuffer, n: usize) -> Result<()> {
        Self::launch_binary_kernel("bitllm_f32_add", a, b, out, n)
    }

    pub fn f32_sub(a: &GpuBuffer, b: &GpuBuffer, out: &GpuBuffer, n: usize) -> Result<()> {
        Self::launch_binary_kernel("bitllm_f32_sub", a, b, out, n)
    }

    pub fn f32_mul(a: &GpuBuffer, b: &GpuBuffer, out: &GpuBuffer, n: usize) -> Result<()> {
        Self::launch_binary_kernel("bitllm_f32_mul", a, b, out, n)
    }

    pub fn f32_scale(a: &GpuBuffer, scale: f32, out: &GpuBuffer, n: usize) -> Result<()> {
        #[cfg(feature = "rocm")]
        {
            unsafe {
                let block_size = 256;
                let grid_size = (n + block_size - 1) / block_size;

                let err = rocm_rs::hip::hipLaunchKernel(
                    Self::get_kernel("bitllm_f32_scale")?,
                    rocm_rs::hip::dim3 {
                        x: grid_size as u32,
                        y: 1,
                        z: 1,
                    },
                    rocm_rs::hip::dim3 {
                        x: block_size as u32,
                        y: 1,
                        z: 1,
                    },
                    &mut [
                        a.ptr() as *mut std::ffi::c_void,
                        &scale as *const f32 as *mut std::ffi::c_void,
                        out.ptr() as *mut std::ffi::c_void,
                        &n as *const usize as *mut std::ffi::c_void,
                    ] as *mut *mut std::ffi::c_void,
                    0,
                    rocm_rs::hip::hipStream_t::std_stream,
                );
                if err != rocm_rs::hip::hipError_t::hipSuccess {
                    return Err(RocmError::KernelLaunchFailed(format!("{:?}", err)));
                }
            }
            Ok(())
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = (a, scale, out, n);
            Err(RocmError::NotAvailable)
        }
    }

    pub fn f32_matmul(
        a: &GpuBuffer,
        b: &GpuBuffer,
        out: &GpuBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        #[cfg(feature = "rocm")]
        {
            unsafe {
                let block_x = 32u32;
                let block_y = 32u32;
                let grid_x = ((n + 31) / 32) as u32;
                let grid_y = ((m + 31) / 32) as u32;

                let err = rocm_rs::hip::hipLaunchKernel(
                    Self::get_kernel("bitllm_f32_matmul")?,
                    rocm_rs::hip::dim3 {
                        x: grid_x,
                        y: grid_y,
                        z: 1,
                    },
                    rocm_rs::hip::dim3 {
                        x: block_x,
                        y: block_y,
                        z: 1,
                    },
                    &mut [
                        a.ptr() as *mut std::ffi::c_void,
                        b.ptr() as *mut std::ffi::c_void,
                        out.ptr() as *mut std::ffi::c_void,
                        &m as *const usize as *mut std::ffi::c_void,
                        &n as *const usize as *mut std::ffi::c_void,
                        &k as *const usize as *mut std::ffi::c_void,
                    ] as *mut *mut std::ffi::c_void,
                    0,
                    rocm_rs::hip::hipStream_t::std_stream,
                );
                if err != rocm_rs::hip::hipError_t::hipSuccess {
                    return Err(RocmError::KernelLaunchFailed(format!("{:?}", err)));
                }
            }
            Ok(())
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = (a, b, out, m, n, k);
            Err(RocmError::NotAvailable)
        }
    }

    pub fn f32_matmul_transb(
        a: &GpuBuffer,
        b: &GpuBuffer,
        out: &GpuBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        #[cfg(feature = "rocm")]
        {
            unsafe {
                let block_x = 32u32;
                let block_y = 32u32;
                let grid_x = ((n + 31) / 32) as u32;
                let grid_y = ((m + 31) / 32) as u32;

                let err = rocm_rs::hip::hipLaunchKernel(
                    Self::get_kernel("bitllm_f32_matmul_transB")?,
                    rocm_rs::hip::dim3 {
                        x: grid_x,
                        y: grid_y,
                        z: 1,
                    },
                    rocm_rs::hip::dim3 {
                        x: block_x,
                        y: block_y,
                        z: 1,
                    },
                    &mut [
                        a.ptr() as *mut std::ffi::c_void,
                        b.ptr() as *mut std::ffi::c_void,
                        out.ptr() as *mut std::ffi::c_void,
                        &m as *const usize as *mut std::ffi::c_void,
                        &n as *const usize as *mut std::ffi::c_void,
                        &k as *const usize as *mut std::ffi::c_void,
                    ] as *mut *mut std::ffi::c_void,
                    0,
                    rocm_rs::hip::hipStream_t::std_stream,
                );
                if err != rocm_rs::hip::hipError_t::hipSuccess {
                    return Err(RocmError::KernelLaunchFailed(format!("{:?}", err)));
                }
            }
            Ok(())
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = (a, b, out, m, n, k);
            Err(RocmError::NotAvailable)
        }
    }

    pub fn f32_softmax(
        input: &GpuBuffer,
        output: &GpuBuffer,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        #[cfg(feature = "rocm")]
        {
            unsafe {
                let block_size = 256u32;
                let shared_mem = block_size as usize * std::mem::size_of::<f32>();

                let err = rocm_rs::hip::hipLaunchKernel(
                    Self::get_kernel("bitllm_softmax")?,
                    rocm_rs::hip::dim3 {
                        x: rows as u32,
                        y: 1,
                        z: 1,
                    },
                    rocm_rs::hip::dim3 {
                        x: block_size,
                        y: 1,
                        z: 1,
                    },
                    &mut [
                        input.ptr() as *mut std::ffi::c_void,
                        output.ptr() as *mut std::ffi::c_void,
                        &rows as *const usize as *mut std::ffi::c_void,
                        &cols as *const usize as *mut std::ffi::c_void,
                    ] as *mut *mut std::ffi::c_void,
                    shared_mem,
                    rocm_rs::hip::hipStream_t::std_stream,
                );
                if err != rocm_rs::hip::hipError_t::hipSuccess {
                    return Err(RocmError::KernelLaunchFailed(format!("{:?}", err)));
                }
            }
            Ok(())
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = (input, output, rows, cols);
            Err(RocmError::NotAvailable)
        }
    }

    pub fn f32_rope(
        q: &GpuBuffer,
        k: &GpuBuffer,
        num_heads: usize,
        head_dim: usize,
        position: usize,
        theta: f32,
    ) -> Result<()> {
        #[cfg(feature = "rocm")]
        {
            unsafe {
                let total = num_heads * head_dim / 2;
                let block_size = 256u32;
                let grid_size = ((total + 255) / 256) as u32;

                let err = rocm_rs::hip::hipLaunchKernel(
                    Self::get_kernel("bitllm_rope")?,
                    rocm_rs::hip::dim3 {
                        x: grid_size,
                        y: 1,
                        z: 1,
                    },
                    rocm_rs::hip::dim3 {
                        x: block_size,
                        y: 1,
                        z: 1,
                    },
                    &mut [
                        q.ptr() as *mut std::ffi::c_void,
                        k.ptr() as *mut std::ffi::c_void,
                        &num_heads as *const usize as *mut std::ffi::c_void,
                        &head_dim as *const usize as *mut std::ffi::c_void,
                        &1usize as *const usize as *mut std::ffi::c_void,
                        &position as *const usize as *mut std::ffi::c_void,
                        &theta as *const f32 as *mut std::ffi::c_void,
                    ] as *mut *mut std::ffi::c_void,
                    0,
                    rocm_rs::hip::hipStream_t::std_stream,
                );
                if err != rocm_rs::hip::hipError_t::hipSuccess {
                    return Err(RocmError::KernelLaunchFailed(format!("{:?}", err)));
                }
            }
            Ok(())
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = (q, k, num_heads, head_dim, position, theta);
            Err(RocmError::NotAvailable)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bit1_matmul(
        a: &GpuBuffer,
        w: &GpuBuffer,
        scales: &GpuBuffer,
        out: &GpuBuffer,
        m: usize,
        n: usize,
        k: usize,
        group_size: i32,
        outlier_mask: Option<&GpuBuffer>,
        outlier_vals: Option<&GpuBuffer>,
    ) -> Result<()> {
        #[cfg(feature = "rocm")]
        {
            unsafe {
                let block_x = 32u32;
                let block_y = 32u32;
                let grid_x = ((n + 31) / 32) as u32;
                let grid_y = ((m + 31) / 32) as u32;

                let kernel_name = if group_size > 0 {
                    "bitllm_bit1_matmul_grouped"
                } else {
                    "bitllm_bit1_matmul"
                };

                let mut args: Vec<*mut std::ffi::c_void> = vec![
                    a.ptr() as *mut std::ffi::c_void,
                    w.ptr() as *mut std::ffi::c_void,
                    scales.ptr() as *mut std::ffi::c_void,
                    out.ptr() as *mut std::ffi::c_void,
                    &m as *const usize as *mut std::ffi::c_void,
                    &n as *const usize as *mut std::ffi::c_void,
                    &k as *const usize as *mut std::ffi::c_void,
                    &group_size as *const i32 as *mut std::ffi::c_void,
                ];

                let (mask_ptr, vals_ptr) = match (outlier_mask, outlier_vals) {
                    (Some(mask), Some(vals)) => {
                        args.push(mask.ptr() as *mut std::ffi::c_void);
                        args.push(vals.ptr() as *mut std::ffi::c_void);
                        (mask.ptr(), vals.ptr())
                    }
                    _ => {
                        args.push(std::ptr::null_mut());
                        args.push(std::ptr::null_mut());
                        (std::ptr::null_mut(), std::ptr::null_mut())
                    }
                };

                let err = rocm_rs::hip::hipLaunchKernel(
                    Self::get_kernel(kernel_name)?,
                    rocm_rs::hip::dim3 {
                        x: grid_x,
                        y: grid_y,
                        z: 1,
                    },
                    rocm_rs::hip::dim3 {
                        x: block_x,
                        y: block_y,
                        z: 1,
                    },
                    args.as_mut_ptr(),
                    args.len() * std::mem::size_of::<*mut std::ffi::c_void>(),
                    rocm_rs::hip::hipStream_t::std_stream,
                );
                if err != rocm_rs::hip::hipError_t::hipSuccess {
                    return Err(RocmError::KernelLaunchFailed(format!("{:?}", err)));
                }
            }
            Ok(())
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = (
                a,
                w,
                scales,
                out,
                m,
                n,
                k,
                group_size,
                outlier_mask,
                outlier_vals,
            );
            Err(RocmError::NotAvailable)
        }
    }

    fn launch_binary_kernel(
        name: &str,
        a: &GpuBuffer,
        b: &GpuBuffer,
        out: &GpuBuffer,
        n: usize,
    ) -> Result<()> {
        #[cfg(feature = "rocm")]
        {
            unsafe {
                let block_size = 256;
                let grid_size = (n + block_size - 1) / block_size;

                let err = rocm_rs::hip::hipLaunchKernel(
                    Self::get_kernel(name)?,
                    rocm_rs::hip::dim3 {
                        x: grid_size as u32,
                        y: 1,
                        z: 1,
                    },
                    rocm_rs::hip::dim3 {
                        x: block_size as u32,
                        y: 1,
                        z: 1,
                    },
                    &mut [
                        a.ptr() as *mut std::ffi::c_void,
                        b.ptr() as *mut std::ffi::c_void,
                        out.ptr() as *mut std::ffi::c_void,
                        &n as *const usize as *mut std::ffi::c_void,
                    ] as *mut *mut std::ffi::c_void,
                    0,
                    rocm_rs::hip::hipStream_t::std_stream,
                );
                if err != rocm_rs::hip::hipError_t::hipSuccess {
                    return Err(RocmError::KernelLaunchFailed(format!(
                        "Kernel '{}' launch failed: {:?}",
                        name, err
                    )));
                }
            }
            Ok(())
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = (name, a, b, out, n);
            Err(RocmError::NotAvailable)
        }
    }

    #[cfg(feature = "rocm")]
    unsafe fn get_kernel(name: &str) -> Result<rocm_rs::hip::hipFunction_t> {
        use std::ffi::CString;
        let c_name = CString::new(name).unwrap();
        let mut func: rocm_rs::hip::hipFunction_t = std::ptr::null_mut();
        let err = rocm_rs::hip::hipModuleGetFunction(
            &mut func,
            rocm_rs::hip::hipModule_t::std_module(),
            c_name.as_ptr(),
        );
        if err != rocm_rs::hip::hipError_t::hipSuccess || func.is_null() {
            return Err(RocmError::KernelLaunchFailed(format!(
                "Failed to get kernel '{}': {:?}",
                name, err
            )));
        }
        Ok(func)
    }
}
