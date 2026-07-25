use bitllm_rocm::{GpuBuffer, GpuOps};
use bitllm_tensor::{DType, Device, Tensor};

#[derive(Clone)]
pub struct GpuContext {
    device_id: i32,
}

impl GpuContext {
    pub fn new(device_id: i32) -> Result<Self, String> {
        let _dev = bitllm_rocm::Device::new(device_id)
            .map_err(|e| format!("Failed to open GPU device {}: {}", device_id, e))?;
        Ok(Self { device_id })
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    pub fn is_available() -> bool {
        bitllm_rocm::is_available()
    }

    fn upload(&self, tensor: &Tensor) -> Result<GpuBuffer, String> {
        let buf =
            GpuBuffer::from_host(tensor.data()).map_err(|e| format!("GPU upload failed: {}", e))?;
        Ok(buf)
    }

    fn download(&self, buf: &GpuBuffer, shape: &[usize], dtype: DType) -> Result<Tensor, String> {
        let nbytes = shape.iter().product::<usize>() * dtype.bit_width() / 8;
        let mut host = vec![0u8; nbytes];
        buf.copy_to_host(&mut host)
            .map_err(|e| format!("GPU download failed: {}", e))?;
        let mut t = Tensor::on_device(
            shape,
            dtype,
            Device::Gpu {
                device_id: self.device_id,
            },
        );
        t.data_mut().copy_from_slice(&host);
        Ok(t)
    }

    pub fn matmul_transposed(
        &self,
        a: &Tensor,
        b_t: &Tensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Tensor, String> {
        let ga = self.upload(a)?;
        let gb = self.upload(b_t)?;
        let gout = GpuBuffer::new(m * n * 4).map_err(|e| format!("GPU alloc failed: {}", e))?;
        GpuOps::f32_matmul(&ga, &gb, &gout, m, n, k)
            .map_err(|e| format!("GPU matmul failed: {}", e))?;
        self.download(&gout, &[m, n], DType::F32)
    }

    pub fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
        let n = a.num_elements();
        let ga = self.upload(a)?;
        let gb = self.upload(b)?;
        let gout = GpuBuffer::new(n * 4).map_err(|e| format!("GPU alloc failed: {}", e))?;
        GpuOps::f32_add(&ga, &gb, &gout, n).map_err(|e| format!("GPU add failed: {}", e))?;
        self.download(&gout, a.shape(), DType::F32)
    }

    pub fn sub(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
        let n = a.num_elements();
        let ga = self.upload(a)?;
        let gb = self.upload(b)?;
        let gout = GpuBuffer::new(n * 4).map_err(|e| format!("GPU alloc failed: {}", e))?;
        GpuOps::f32_sub(&ga, &gb, &gout, n).map_err(|e| format!("GPU sub failed: {}", e))?;
        self.download(&gout, a.shape(), DType::F32)
    }

    pub fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
        let n = a.num_elements();
        let ga = self.upload(a)?;
        let gb = self.upload(b)?;
        let gout = GpuBuffer::new(n * 4).map_err(|e| format!("GPU alloc failed: {}", e))?;
        GpuOps::f32_mul(&ga, &gb, &gout, n).map_err(|e| format!("GPU mul failed: {}", e))?;
        self.download(&gout, a.shape(), DType::F32)
    }

    pub fn scale(&self, a: &Tensor, scale: f32) -> Result<Tensor, String> {
        let n = a.num_elements();
        let ga = self.upload(a)?;
        let gout = GpuBuffer::new(n * 4).map_err(|e| format!("GPU alloc failed: {}", e))?;
        GpuOps::f32_scale(&ga, scale, &gout, n).map_err(|e| format!("GPU scale failed: {}", e))?;
        self.download(&gout, a.shape(), DType::F32)
    }

    pub fn softmax(&self, input: &Tensor, rows: usize, cols: usize) -> Result<Tensor, String> {
        let n = rows * cols;
        let gin = self.upload(input)?;
        let gout = GpuBuffer::new(n * 4).map_err(|e| format!("GPU alloc failed: {}", e))?;
        GpuOps::f32_softmax(&gin, &gout, rows, cols)
            .map_err(|e| format!("GPU softmax failed: {}", e))?;
        self.download(&gout, input.shape(), DType::F32)
    }

    pub fn rope(
        &self,
        q: &Tensor,
        k: &Tensor,
        num_heads: usize,
        head_dim: usize,
        position: usize,
        theta: f32,
    ) -> Result<(Tensor, Tensor), String> {
        let gq = self.upload(q)?;
        let gk = self.upload(k)?;
        GpuOps::f32_rope(&gq, &gk, num_heads, head_dim, position, theta)
            .map_err(|e| format!("GPU rope failed: {}", e))?;
        let q_out = self.download(&gq, q.shape(), DType::F32)?;
        let k_out = self.download(&gk, k.shape(), DType::F32)?;
        Ok((q_out, k_out))
    }

    pub fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor, String> {
        let n = input.num_elements();
        let hidden = *input.shape().last().unwrap_or(&n);
        let rows = n / hidden;

        let gin = self.upload(input)?;
        let gw = self.upload(weight)?;

        let gemul = GpuBuffer::new(n * 4).map_err(|e| format!("GPU alloc failed: {}", e))?;
        GpuOps::f32_mul(&gin, &gw, &gemul, n)
            .map_err(|e| format!("GPU rms_norm mul failed: {}", e))?;

        let _gsum = GpuBuffer::new(rows * 4).map_err(|e| format!("GPU alloc failed: {}", e))?;
        // For rms_norm, we need row-wise sum of squares, then scale.
        // GPU doesn't have a row-sum kernel yet, so compute sum-of-squares on host
        // from the multiplied result, then apply scale on GPU.
        let mul_host = self.download(&gemul, input.shape(), DType::F32)?;
        let mul_slice = mul_host.as_f32_slice();

        let mut scales = Vec::with_capacity(rows);
        for i in 0..rows {
            let row = &mul_slice[i * hidden..][..hidden];
            let sum_sq: f32 = row.iter().map(|v| v * v).sum();
            let rms = (sum_sq / hidden as f32 + eps).sqrt();
            scales.push(1.0 / rms);
        }

        let mut result = Tensor::zeros(input.shape(), DType::F32);
        let out_slice = result.as_f32_slice_mut();
        for i in 0..rows {
            let src = &mul_slice[i * hidden..][..hidden];
            let dst = &mut out_slice[i * hidden..][..hidden];
            let scale = scales[i];
            for j in 0..hidden {
                dst[j] = src[j] * scale;
            }
        }

        Ok(result)
    }
}
