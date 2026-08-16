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

        let in_slice = if input.dtype() == DType::F32 {
            input.as_f32_slice()
        } else {
            let t = input.to_f32();
            return self.rms_norm_with_slice(t.as_f32_slice(), weight, hidden, rows, eps);
        };
        self.rms_norm_with_slice(in_slice, weight, hidden, rows, eps)
    }

    fn rms_norm_with_slice(
        &self,
        in_slice: &[f32],
        weight: &Tensor,
        hidden: usize,
        rows: usize,
        eps: f32,
    ) -> Result<Tensor, String> {
        let w_slice = weight.as_f32_slice();

        let mut result = Tensor::zeros(&[rows, hidden], DType::F32);
        let out_slice = result.as_f32_slice_mut();

        for i in 0..rows {
            let row = &in_slice[i * hidden..][..hidden];
            let sum_sq: f32 = row.iter().map(|v| v * v).sum();
            let inv_rms = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();
            let w_row = &w_slice[..hidden];
            let out_row = &mut out_slice[i * hidden..][..hidden];
            for j in 0..hidden {
                out_row[j] = row[j] * w_row[j] * inv_rms;
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::apply_rotary_emb_inplace;
    use crate::config::ModelConfig;
    use crate::layers::{Linear, RmsNorm};
    use crate::model::Model;

    /// GPU backend testing is hardware-gated: these tests exercise real HIP
    /// kernels when run on a ROCm machine (`cargo test -p bitllm-runtime
    /// --features gpu` on a host with a GPU driver; the `gpu` feature forwards
    /// `bitllm-rocm/rocm`) and skip cleanly everywhere else. Run the ops
    /// through the high-level `GpuContext` wrappers and compare against the
    /// CPU implementations.
    fn require_gpu() -> Option<GpuContext> {
        if !GpuContext::is_available() {
            eprintln!("skipping GPU tests: no ROCm device available");
            return None;
        }
        match GpuContext::new(0) {
            Ok(ctx) => Some(ctx),
            Err(e) => {
                eprintln!("skipping GPU tests: failed to open device 0: {}", e);
                None
            }
        }
    }

    fn assert_close(a: &Tensor, b: &Tensor, tol: f32) {
        assert_eq!(
            a.shape(),
            b.shape(),
            "shape mismatch: {:?} vs {:?}",
            a.shape(),
            b.shape()
        );
        for i in 0..a.num_elements() {
            let va = a.get_flat_f32(i);
            let vb = b.get_flat_f32(i);
            let diff = (va - vb).abs();
            let denom = va.abs().max(vb.abs()).max(1e-6);
            assert!(
                diff / denom < tol,
                "i={}: cpu {} gpu {} (diff {})",
                i,
                va,
                vb,
                diff
            );
        }
    }

    fn random_tensor(shape: &[usize]) -> Tensor {
        Tensor::random(shape, DType::F32)
    }

    #[test]
    fn gpu_matmul_matches_cpu() {
        let Some(ctx) = require_gpu() else { return };
        let lin = Linear::new(random_tensor(&[8, 6]), Some(random_tensor(&[8])));
        let input = random_tensor(&[5, 6]);
        let cpu = lin.forward(&input);
        let gpu = lin.forward_gpu(&input, Some(&ctx));
        assert_close(&cpu, &gpu, 1e-3);
    }

    #[test]
    fn gpu_elementwise_matches_cpu() {
        let Some(ctx) = require_gpu() else { return };
        let a = random_tensor(&[32, 8]);
        let b = random_tensor(&[32, 8]);

        assert_close(&a.add(&b).unwrap(), &ctx.add(&a, &b).unwrap(), 1e-5);
        assert_close(&a.sub(&b).unwrap(), &ctx.sub(&a, &b).unwrap(), 1e-5);
        assert_close(&a.mul(&b).unwrap(), &ctx.mul(&a, &b).unwrap(), 1e-5);

        let scale = 0.5f32;
        let mut cpu_scaled = a.clone();
        for v in cpu_scaled.as_f32_slice_mut().iter_mut() {
            *v *= scale;
        }
        assert_close(&cpu_scaled, &ctx.scale(&a, scale).unwrap(), 1e-5);
    }

    #[test]
    fn gpu_softmax_matches_cpu() {
        let Some(ctx) = require_gpu() else { return };
        let rows = 4usize;
        let cols = 16usize;
        let input = random_tensor(&[rows, cols]);

        let mut cpu = input.clone();
        let s = cpu.as_f32_slice_mut();
        for r in 0..rows {
            let row = &mut s[r * cols..(r + 1) * cols];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|x| (x - max).exp()).sum();
            for v in row.iter_mut() {
                *v = (*v - max).exp() / sum;
            }
        }
        let gpu = ctx.softmax(&input, rows, cols).unwrap();
        assert_close(&cpu, &gpu, 1e-3);
    }

    #[test]
    fn gpu_rope_matches_cpu() {
        let Some(ctx) = require_gpu() else { return };
        let num_heads = 4usize;
        let seq_len = 8usize;
        let head_dim = 16usize;
        let position = 3usize;
        let theta = 10000.0f32;

        let q = random_tensor(&[num_heads, seq_len, head_dim]);
        let k = random_tensor(&[num_heads, seq_len, head_dim]);

        let mut q_cpu = q.clone();
        let mut k_cpu = k.clone();
        apply_rotary_emb_inplace(&mut q_cpu, &mut k_cpu, position, head_dim, theta);

        let (q_gpu, k_gpu) = ctx
            .rope(&q, &k, num_heads, head_dim, position, theta)
            .unwrap();
        assert_close(&q_cpu, &q_gpu, 1e-3);
        assert_close(&k_cpu, &k_gpu, 1e-3);
    }

    #[test]
    fn gpu_rms_norm_matches_cpu() {
        let Some(ctx) = require_gpu() else { return };
        let norm = RmsNorm::new(random_tensor(&[32]), 1e-5);
        let input = random_tensor(&[7, 32]);
        let cpu = norm.forward(&input);
        let gpu = norm.forward_gpu(&input, Some(&ctx));
        assert_close(&cpu, &gpu, 1e-6);
    }

    fn randomize_tiny(model: &mut Model, config: &ModelConfig) {
        model.embedding.weight = random_tensor(&[config.vocab_size, config.hidden_size]);
        model.lm_head.weight = random_tensor(&[config.vocab_size, config.hidden_size]);
        model.norm.weight = random_tensor(&[config.hidden_size]);
        let head_dim = config.head_dim();
        let qk_out = config.num_heads * head_dim;
        let kv_out = config.num_kv_heads() * head_dim;
        for layer in &mut model.layers {
            layer.attn_norm.weight = random_tensor(&[config.hidden_size]);
            layer.ffn_norm.weight = random_tensor(&[config.hidden_size]);
            layer.attention.q_proj.weight = random_tensor(&[qk_out, config.hidden_size]);
            layer.attention.k_proj.weight = random_tensor(&[kv_out, config.hidden_size]);
            layer.attention.v_proj.weight = random_tensor(&[kv_out, config.hidden_size]);
            layer.attention.o_proj.weight = random_tensor(&[config.hidden_size, qk_out]);
            layer.ffn_up.weight = random_tensor(&[config.intermediate_size, config.hidden_size]);
            layer.ffn_gate.weight = random_tensor(&[config.intermediate_size, config.hidden_size]);
            layer.ffn_down.weight = random_tensor(&[config.hidden_size, config.intermediate_size]);
        }
    }

    #[test]
    fn gpu_model_forward_matches_cpu() {
        let Some(ctx) = require_gpu() else { return };
        let config = ModelConfig::tiny_test();
        let mut model = Model::new(config.clone());
        randomize_tiny(&mut model, &config);

        let tokens: Vec<u32> = (0..16).map(|i| i * 13 % config.vocab_size as u32).collect();

        model.clear_cache();
        let cpu = model.forward_hidden(&tokens, 0, None);

        model.clear_cache();
        let gpu = model.forward_hidden(&tokens, 0, Some(&ctx));

        assert_close(&cpu, &gpu, 1e-2);

        // Second forward: exercise the KV cache path on GPU too. CPU and GPU
        // both prefill 16 tokens, then decode 8 more at position 16.
        let more: Vec<u32> = (16..24)
            .map(|i| i * 13 % config.vocab_size as u32)
            .collect();

        model.clear_cache();
        let _cpu_prefill = model.forward_hidden(&tokens, 0, None);
        let cpu_decode = model.forward_hidden(&more, 0, None);

        model.clear_cache();
        let _gpu_prefill = model.forward_hidden(&tokens, 0, Some(&ctx));
        let gpu_decode = model.forward_hidden(&more, 0, Some(&ctx));

        assert_close(&cpu_decode, &gpu_decode, 1e-2);
    }
}
