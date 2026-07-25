use crate::{Result, RocmError};

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub total_memory: usize,
    pub multi_processor_count: i32,
}

#[derive(Debug)]
pub struct Device {
    #[cfg(feature = "rocm")]
    id: i32,
}

impl Device {
    pub fn new(device_id: i32) -> Result<Self> {
        #[cfg(feature = "rocm")]
        {
            unsafe {
                let mut count = 0;
                let err = rocm_rs::hip::hipGetDeviceCount(&mut count);
                if err != rocm_rs::hip::hipError_t::hipSuccess {
                    return Err(RocmError::HipError(format!(
                        "hipGetDeviceCount failed: {:?}",
                        err
                    )));
                }
                if device_id >= count {
                    return Err(RocmError::HipError(format!(
                        "Device {} not found, {} devices available",
                        device_id, count
                    )));
                }
            }
            Ok(Self { id: device_id })
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = device_id;
            Err(RocmError::NotAvailable)
        }
    }

    pub fn info(&self) -> Result<DeviceInfo> {
        #[cfg(feature = "rocm")]
        {
            unsafe {
                let prev_device = self.set_current()?;

                let mut name_buf = [0u8; 256];
                let err = rocm_rs::hip::hipDeviceGetAttribute(
                    name_buf.as_mut_ptr() as *mut i8,
                    rocm_rs::hip::hipDeviceAttribute_t::hipDeviceAttributePersistingL2CacheMaxSize,
                    self.id,
                );

                let mut props: rocm_rs::hip::hipDeviceProp_t = std::mem::zeroed();
                let err = rocm_rs::hip::hipGetDeviceProperties(&mut props, self.id);
                if err != rocm_rs::hip::hipError_t::hipSuccess {
                    return Err(RocmError::HipError(format!(
                        "hipGetDeviceProperties failed: {:?}",
                        err
                    )));
                }

                let name = std::ffi::CStr::from_ptr(props.name.as_ptr())
                    .to_string_lossy()
                    .to_string();

                let _ = prev_device;

                Ok(DeviceInfo {
                    name,
                    compute_major: props.major as i32,
                    compute_minor: props.minor as i32,
                    total_memory: props.totalGlobalMem,
                    multi_processor_count: props.multiProcessorCount,
                })
            }
        }
        #[cfg(not(feature = "rocm"))]
        {
            Err(RocmError::NotAvailable)
        }
    }

    pub fn set_current(&self) -> Result<Option<Device>> {
        #[cfg(feature = "rocm")]
        {
            unsafe {
                let mut prev = 0;
                let err = rocm_rs::hip::hipGetDevice(&mut prev);
                if err != rocm_rs::hip::hipError_t::hipSuccess {
                    return Err(RocmError::HipError(format!(
                        "hipGetDevice failed: {:?}",
                        err
                    )));
                }
                let err = rocm_rs::hip::hipSetDevice(self.id);
                if err != rocm_rs::hip::hipError_t::hipSuccess {
                    return Err(RocmError::HipError(format!(
                        "hipSetDevice failed: {:?}",
                        err
                    )));
                }
                if prev == self.id {
                    Ok(None)
                } else {
                    Ok(Some(Device { id: prev }))
                }
            }
        }
        #[cfg(not(feature = "rocm"))]
        {
            Err(RocmError::NotAvailable)
        }
    }

    pub fn synchronize(&self) -> Result<()> {
        #[cfg(feature = "rocm")]
        {
            unsafe {
                let err = rocm_rs::hip::hipDeviceSynchronize();
                if err != rocm_rs::hip::hipError_t::hipSuccess {
                    return Err(RocmError::HipError(format!(
                        "hipDeviceSynchronize failed: {:?}",
                        err
                    )));
                }
            }
            Ok(())
        }
        #[cfg(not(feature = "rocm"))]
        {
            Err(RocmError::NotAvailable)
        }
    }
}

pub fn device_count() -> Result<i32> {
    #[cfg(feature = "rocm")]
    {
        unsafe {
            let mut count = 0;
            let err = rocm_rs::hip::hipGetDeviceCount(&mut count);
            if err != rocm_rs::hip::hipError_t::hipSuccess {
                return Err(RocmError::HipError(format!(
                    "hipGetDeviceCount failed: {:?}",
                    err
                )));
            }
            Ok(count)
        }
    }
    #[cfg(not(feature = "rocm"))]
    {
        Err(RocmError::NotAvailable)
    }
}

pub fn detect_devices() -> Result<Vec<DeviceInfo>> {
    let count = device_count()?;
    let mut devices = Vec::new();
    for i in 0..count {
        let device = Device::new(i)?;
        devices.push(device.info()?);
    }
    Ok(devices)
}
