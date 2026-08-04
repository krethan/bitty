use crate::{Result, RocmError};

#[derive(Debug)]
pub struct GpuBuffer {
    ptr: *mut u8,
    size: usize,
    #[cfg(feature = "rocm")]
    device_id: i32,
}

unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

impl GpuBuffer {
    pub fn new(size: usize) -> Result<Self> {
        #[cfg(feature = "rocm")]
        {
            unsafe {
                let mut ptr: *mut u8 = std::ptr::null_mut();
                let err = rocm_rs::hip::hipMalloc(&mut ptr as *mut *mut u8, size);
                if err != rocm_rs::hip::hipError_t::hipSuccess {
                    return Err(RocmError::AllocationFailed(size));
                }
                let mut device_id = 0;
                rocm_rs::hip::hipGetDevice(&mut device_id);
                Ok(Self {
                    ptr,
                    size,
                    device_id,
                })
            }
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = size;
            Err(RocmError::NotAvailable)
        }
    }

    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn from_host(data: &[u8]) -> Result<Self> {
        let buf = Self::new(data.len())?;
        buf.copy_from_host(data)?;
        Ok(buf)
    }

    pub fn copy_from_host(&self, data: &[u8]) -> Result<()> {
        if data.len() > self.size {
            return Err(RocmError::TransferFailed(format!(
                "Source size {} exceeds buffer size {}",
                data.len(),
                self.size
            )));
        }
        #[cfg(feature = "rocm")]
        {
            unsafe {
                let err = rocm_rs::hip::hipMemcpy(
                    self.ptr as *mut std::ffi::c_void,
                    data.as_ptr() as *const std::ffi::c_void,
                    data.len(),
                    rocm_rs::hip::hipMemcpyKind::hipMemcpyHostToDevice,
                );
                if err != rocm_rs::hip::hipError_t::hipSuccess {
                    return Err(RocmError::TransferFailed(format!(
                        "hipMemcpy H2D failed: {:?}",
                        err
                    )));
                }
            }
            Ok(())
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = data;
            Err(RocmError::NotAvailable)
        }
    }

    pub fn copy_to_host(&self, buf: &mut [u8]) -> Result<()> {
        if buf.len() > self.size {
            return Err(RocmError::TransferFailed(format!(
                "Dest size {} exceeds buffer size {}",
                buf.len(),
                self.size
            )));
        }
        #[cfg(feature = "rocm")]
        {
            unsafe {
                let err = rocm_rs::hip::hipMemcpy(
                    buf.as_mut_ptr() as *mut std::ffi::c_void,
                    self.ptr as *const std::ffi::c_void,
                    buf.len(),
                    rocm_rs::hip::hipMemcpyKind::hipMemcpyDeviceToHost,
                );
                if err != rocm_rs::hip::hipError_t::hipSuccess {
                    return Err(RocmError::TransferFailed(format!(
                        "hipMemcpy D2H failed: {:?}",
                        err
                    )));
                }
            }
            Ok(())
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = buf;
            Err(RocmError::NotAvailable)
        }
    }

    #[cfg(feature = "rocm")]
    pub fn copy_from_host_async(&self, data: &[u8], stream: rocm_rs::hip::hipStream_t) -> Result<()> {
        if data.len() > self.size {
            return Err(RocmError::TransferFailed(format!(
                "Source size {} exceeds buffer size {}",
                data.len(),
                self.size
            )));
        }
        #[cfg(feature = "rocm")]
        {
            unsafe {
                let err = rocm_rs::hip::hipMemcpyAsync(
                    self.ptr as *mut std::ffi::c_void,
                    data.as_ptr() as *const std::ffi::c_void,
                    data.len(),
                    rocm_rs::hip::hipMemcpyKind::hipMemcpyHostToDevice,
                    stream,
                );
                if err != rocm_rs::hip::hipError_t::hipSuccess {
                    return Err(RocmError::TransferFailed(format!(
                        "hipMemcpyAsync H2D failed: {:?}",
                        err
                    )));
                }
            }
            Ok(())
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = (data, stream);
            Err(RocmError::NotAvailable)
        }
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        #[cfg(feature = "rocm")]
        {
            unsafe {
                rocm_rs::hip::hipFree(self.ptr as *mut std::ffi::c_void);
            }
        }
    }
}

/// Streams model weights from host RAM to GPU VRAM with double-buffering
/// and prefetch support for overlapping compute with data transfer.
#[derive(Debug)]
#[cfg(feature = "rocm")]
pub struct GpuWeightStreamer {
    /// Host-side RAM buffer (pinned for faster PCIe transfer)
    pub host_ram: Vec<u8>,
    /// Double-buffered VRAM regions
    pub vram_buffers: Vec<GpuBuffer>,
    current_buffer: usize,
    /// PCIe transfer width in bytes per layer
    pub pcie_width: usize,
}

#[cfg(feature = "rocm")]
impl GpuWeightStreamer {
    pub fn new(total_bytes: usize, num_buffers: usize) -> Result<Self> {
        let host_ram = vec![0u8; total_bytes];
        let mut vram_buffers = Vec::with_capacity(num_buffers);
        for _ in 0..num_buffers {
            vram_buffers.push(GpuBuffer::new(total_bytes / num_buffers)?);
        }
        Ok(Self {
            host_ram,
            vram_buffers,
            current_buffer: 0,
            pcie_width: 8,
        })
    }

    /// Queue a layer's weights for async transfer to the current VRAM buffer.
    pub fn queue_layer(&mut self, layer_idx: usize, stream: rocm_rs::hip::hipStream_t) -> Result<()> {
        let offset = layer_idx * self.pcie_width;
        let src = &self.host_ram[offset..offset + self.pcie_width];
        let dst = &self.vram_buffers[self.current_buffer];
        dst.copy_from_host_async(src, stream)?;
        Ok(())
    }

    /// Swap to the next VRAM buffer (must be called after the previous buffer's
    /// async transfer has completed via stream synchronization).
    pub fn swap_buffer(&mut self) {
        self.current_buffer = 1 - self.current_buffer;
    }

    /// Get a reference to the current VRAM buffer for kernel access.
    pub fn current_buffer(&self) -> &GpuBuffer {
        &self.vram_buffers[self.current_buffer]
    }
}
