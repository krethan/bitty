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
