use crate::layers::Linear;
use bitllm_quantization::QuantConfig;
use bitllm_quantization::QuantizedTensor;
use bitllm_quantization::{
    fused_bit1_int8_matmul, fused_bit1_matmul, quantize_grouped_with_outliers,
    quantize_with_outliers,
};
use bitllm_tensor::{DType, Tensor};

/// One-bit (ternary -1/+1) Linear layer.
/// Weights are stored as packed 1-bit sign values, with per-tensor scale.
pub struct BitLinear {
    pub weight_q: QuantizedTensor,
    pub scale: Tensor,
    pub bias: Option<Tensor>,
    /// When true (the default), activations are quantized to int8 (per-token
    /// absmax) before the fused matmul (`fused_bit1_int8_matmul`). Set to
    /// false to use the exact f32-activation path.
    pub a8: bool,
}

impl BitLinear {
    /// Quantize a full-precision weight tensor to 1-bit ternary, honoring the
    /// config's `outlier_frac` (top-fraction of weights kept exact) and
    /// `group_size` (per-group scale if > 0). Activations default to W1A8.
    pub fn quantize(weight: &Tensor, config: &QuantConfig) -> Self {
        let weight_q = if config.group_size > 0 {
            quantize_grouped_with_outliers(weight, config.outlier_frac, config.group_size)
        } else {
            quantize_with_outliers(weight, config.outlier_frac)
        };
        let n = weight_q.shape[0];
        let scale_vec = if weight_q.scales.len() == 1 {
            Tensor::from_slice(&vec![weight_q.scales[0]; n], &[n])
        } else {
            Tensor::from_slice(&weight_q.scales, &[weight_q.scales.len()])
        };
        Self {
            weight_q,
            scale: scale_vec,
            bias: None,
            a8: config.a8,
        }
    }

    /// Construct from pre-quantized weights.
    pub fn from_quantized(weight_q: QuantizedTensor, scale: Tensor, bias: Option<Tensor>) -> Self {
        Self {
            weight_q,
            scale,
            bias,
            a8: true,
        }
    }

    /// Create a BitLinear from a standard Linear by quantizing its weights.
    pub fn from_linear(linear: &Linear) -> Self {
        Self::from_linear_with_config(linear, &QuantConfig::ternary())
    }

    /// Like [`from_linear`], honoring the config's outlier fraction and
    /// group size.
    pub fn from_linear_with_config(linear: &Linear, config: &QuantConfig) -> Self {
        let weight_q = if config.group_size > 0 {
            quantize_grouped_with_outliers(&linear.weight, config.outlier_frac, config.group_size)
        } else {
            quantize_with_outliers(&linear.weight, config.outlier_frac)
        };
        let n = weight_q.shape[0];
        let scale_vec = if weight_q.scales.len() == 1 {
            Tensor::from_slice(&vec![weight_q.scales[0]; n], &[n])
        } else {
            Tensor::from_slice(&weight_q.scales, &[weight_q.scales.len()])
        };
        Self {
            weight_q,
            scale: scale_vec,
            bias: linear.bias.clone(),
            a8: config.a8,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let m = input.shape()[0];
        let k = input.shape()[1];
        let n = self.weight_q.shape[0];
        assert_eq!(k, self.weight_q.shape[1]);

        let input_slice = input.as_f32_slice();
        let mut result = Tensor::zeros(&[m, n], DType::F32);
        let out_slice = result.as_f32_slice_mut();

        if self.a8 {
            fused_bit1_int8_matmul(input_slice, &self.weight_q, out_slice, m, k, n);
            if let Some(ref bias) = self.bias {
                result.add_assign(bias).unwrap();
            }
            return result;
        }

        fused_bit1_matmul(input_slice, &self.weight_q, out_slice, m, k, n);

        if let Some(ref bias) = self.bias {
            result.add_assign(bias).unwrap();
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference dequantized ternary matmul for a single input row.
    fn ref_bitlinear(x: &[f32], wq: &QuantizedTensor) -> Vec<f32> {
        let n = wq.shape[0];
        let w_scale = wq.scales[0];
        let mut out = vec![0.0f32; n];
        for (j, o) in out.iter_mut().enumerate() {
            let mut sum = 0.0f64;
            for (t, &xt) in x.iter().enumerate() {
                let c = t / 8;
                let b = t % 8;
                let w = if (wq.data[c * n + j] >> b) & 1 == 1 {
                    w_scale as f64
                } else {
                    -w_scale as f64
                };
                sum += xt as f64 * w;
            }
            *o = sum as f32;
        }
        out
    }

    #[test]
    fn forward_scales_weights_exactly_once() {
        let w = Tensor::from_slice(&[2.0, -2.0, 1.0, -1.0, 3.0, -3.0], &[2, 3]);
        let lin = BitLinear::quantize(&w, &QuantConfig::ternary().without_a8());
        let x = Tensor::from_slice(&[1.0, -1.0, 0.5], &[1, 3]);
        let out = lin.forward(&x);

        let expected = ref_bitlinear(x.as_f32_slice(), &lin.weight_q);
        for (j, (&got, &exp)) in out.as_f32_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "j={}: got {} expected {}",
                j,
                got,
                exp
            );
        }
    }

    #[test]
    fn forward_a8_matches_manual_int8() {
        let w = Tensor::from_slice(&[2.0, -2.0, 1.0, -1.0, 3.0, -3.0], &[2, 3]);
        let lin = BitLinear::quantize(&w, &QuantConfig::ternary());
        let x = Tensor::from_slice(&[1.0, -1.0, 0.5], &[1, 3]);
        let out = lin.forward(&x);

        // Manual int8: act_scale = 1.0/127, xq = [127, -127, 64] (rounded).
        // dot vs signs [-1,1,-1]? w row0 signs: [2->+, -2->-, 1->+] -> [+1,-1,+1]
        // dot = 127*+1 + (-127)*(-1) + 64*1 = 318; *w_scale(3) * act_scale(1/127).
        // = 318 * 3 / 127 = 7.5118...
        let got = out.as_f32_slice()[0];
        let expected = 318.0 * 3.0 / 127.0;
        assert!(
            (got - expected).abs() < 1e-3,
            "a8 j=0: got {} expected {}",
            got,
            expected
        );
    }

    #[test]
    fn forward_grouped_matches_reference() {
        // [2, 16] weight with two magnitude bands: cols 0..8 are ~10x larger
        // than cols 8..16. Grouped quantize with gs=8 should give two scales.
        let mut data = Vec::with_capacity(32);
        for _row in 0..2 {
            for col in 0..8 {
                data.push((col as f32 - 4.0) * 1.0);
            }
            for col in 0..8 {
                data.push((col as f32 - 4.0) * 0.1);
            }
        }
        let w = Tensor::from_slice(&data, &[2, 16]);
        let config = QuantConfig::ternary_grouped(8);
        let lin = BitLinear::quantize(&w, &config);
        assert_eq!(lin.weight_q.scales.len(), 2);

        let x = Tensor::from_slice(&[1.0f32; 16], &[1, 16]);

        // Manual grouped reference.
        let n = lin.weight_q.shape[0];
        let gs = 8;
        let mut expected = [0.0f32; 2];
        for (j, exp) in expected.iter_mut().enumerate() {
            let mut sum = 0.0f64;
            for t in 0..16 {
                let c = t / 8;
                let b = t % 8;
                let g = t / gs;
                let scale = lin.weight_q.scales[g];
                let bit = (lin.weight_q.data[c * n + j] >> b) & 1;
                let w = if bit == 1 { scale } else { -scale };
                sum += 1.0f64 * w as f64;
            }
            *exp = sum as f32;
        }

        // Both the exact f32 path and the W1A8 path must reproduce it. With
        // all-ones input the int8 quantization is lossless (127 * 1/127 = 1),
        // so the two paths agree exactly on this input.
        for a8 in [false, true] {
            let lin = BitLinear::quantize(&w, &config.clone().with_a8(a8));
            let out = lin.forward(&x);
            for (j, (&got, &exp)) in out.as_f32_slice().iter().zip(expected.iter()).enumerate() {
                let diff = (got - exp).abs();
                assert!(
                    diff < 1e-3,
                    "a8={}: j={}: got {} expected {}",
                    a8,
                    j,
                    got,
                    exp
                );
            }
        }
    }

    #[test]
    fn default_path_is_a8() {
        // W1A8 is the default: a plain ternary config selects the int8 kernel.
        let w = Tensor::from_slice(&[2.0, -2.0, 1.0, -1.0, 3.0, -3.0], &[2, 3]);
        let lin = BitLinear::quantize(&w, &QuantConfig::ternary());
        assert!(lin.a8, "W1A8 should be the default quantized path");
        let lin2 = BitLinear::quantize(&w, &QuantConfig::ternary().without_a8());
        assert!(!lin2.a8);
    }
}
