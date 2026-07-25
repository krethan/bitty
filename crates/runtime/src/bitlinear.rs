use bitllm_quantization::{absmax_quantize, QuantConfig, QuantizedTensor};
use bitllm_tensor::{DType, Tensor};

/// One-bit (ternary -1/+1) Linear layer
/// Weights are stored as 2-bit signs: 01=+1, 10=-1, others 00 (ignored)
pub struct BitLinear {
    pub weight_q: QuantizedTensor, // packed 1-bit weights
    pub scale: Tensor,             // per-output-channel scale (f32)
    pub bias: Option<Tensor>,
}

impl BitLinear {
    /// Convenience constructor: quantize a full-precision weight tensor on the fly
    pub fn quantize(weight: &Tensor, config: &QuantConfig) -> Self {
        assert!(
            config.bits == 1 || config.scheme == bitllm_quantization::scheme::QuantScheme::Ternary
        );
        let weight = absmax_quantize(weight, config); // uses ternary() internally
                                                      // Broadcast the (single) scale to one entry per output row.
        let n = weight.shape[0];
        let scale_val = weight.scales.first().cloned().unwrap_or(1.0);
        let scales: Vec<f32> = vec![scale_val; n];
        let scale_vec = Tensor::from_slice(&scales, &[n]);
        Self {
            weight_q: weight,
            scale: scale_vec,
            bias: None,
        }
    }

    /// Construct from pre-quantized weights
    pub fn from_quantized(weight_q: QuantizedTensor, scale: Tensor, bias: Option<Tensor>) -> Self {
        Self {
            weight_q,
            scale,
            bias,
        }
    }

    /// SIMD/fused dequant + matmul (no full F32 reconstruction of weights)
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let (_m, k) = (input.shape()[0], input.shape()[1]);
        let n = self.scale.shape()[0]; // output dim
        let kp = self.weight_q.shape[1]; // head_dim or group size

        assert_eq!(
            k,
            kp,
            "shapes incompatible input={:?}, weight={:?}",
            input.shape(),
            self.weight_q.shape
        );

        let input_f32 = input.to_f32();
        let mut result = Tensor::zeros(&[input.shape()[0], n], DType::F32);

        // Plain-loop fused kernel: scale + 1-bit sign multiply
        for row in 0..input.shape()[0] {
            for col in 0..n {
                let mut acc = 0.0f32;
                let s = self.scale.get_flat_f32(col);
                for t in 0..k {
                    let a = input_f32.get_flat_f32(row * k + t);
                    // weight_q.data packs two 1-bit weights per u8
                    let byte = self.weight_q.data[t / 2];
                    let bit = if t % 2 == 0 {
                        byte & 0x03
                    } else {
                        (byte >> 4) & 0x03
                    };
                    // Ternary encoding: 01=+1  10=-1  00=0 (ignore)
                    let w = match bit {
                        0x01 => 1.0f32,
                        0x02 => -1.0f32,
                        _ => 0.0f32,
                    };
                    acc += a * w * s;
                }
                result.set_flat_f32(row * n + col, acc);
            }
        }
        if let Some(ref b) = self.bias {
            result = result.add(b).unwrap();
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanity_bitlinear_int8() {
        let w = Tensor::from_slice(&[1.0, -1.0, 2.0, -2.0, 3.0, -3.0], &[3, 2]);
        let in_ = Tensor::from_slice(&[0.5, 0.5], &[1, 2]);
        let bl = BitLinear::quantize(&w, &bitllm_quantization::scheme::QuantConfig::ternary());
        let out = bl.forward(&in_);
        assert_eq!(out.shape(), &[1, 3]);
    }
}
