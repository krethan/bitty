use bitllm_quantization::{QuantConfig, QuantizedTensor};
use bitllm_tensor::{simd, DType, Tensor};

/// One-bit (binary {-1,+1}) Linear layer.
/// Weights packed 1 bit per element, 8 per byte.
/// Bit set = +1, bit clear = -1. Per-output-channel scale.
pub struct BitLinear {
    pub weight_q: QuantizedTensor,
    pub scale: Tensor,
    pub bias: Option<Tensor>,
}

impl BitLinear {
    pub fn quantize(weight: &Tensor, config: &QuantConfig) -> Self {
        assert!(
            config.bits == 1 || config.scheme == bitllm_quantization::scheme::QuantScheme::Ternary
        );

        let src = weight.to_f32();
        let n = src.shape()[0];
        let k = src.shape()[1];
        let bytes_per_row = k.div_ceil(8);

        let mut packed = vec![0u8; n * bytes_per_row];
        let mut scales = Vec::with_capacity(n);

        for row in 0..n {
            let mut row_max = 0.0f32;
            for col in 0..k {
                let v = src.get_flat_f32(row * k + col).abs();
                if v > row_max {
                    row_max = v;
                }
            }
            let s = if row_max == 0.0 { 1.0 } else { row_max };
            scales.push(s);
            let inv = 1.0 / s;

            for col in 0..k {
                let v = src.get_flat_f32(row * k + col) * inv;
                if v > 0.0 {
                    let byte_idx = row * bytes_per_row + col / 8;
                    let bit_idx = col % 8;
                    packed[byte_idx] |= 1 << bit_idx;
                }
            }
        }

        let scale_tensor = Tensor::from_slice(&scales, &[n]);
        Self {
            weight_q: QuantizedTensor {
                data: packed,
                shape: weight.shape().to_vec(),
                scales,
                zeros: None,
                config: QuantConfig::ternary(),
            },
            scale: scale_tensor,
            bias: None,
        }
    }

    pub fn from_quantized(weight_q: QuantizedTensor, scale: Tensor, bias: Option<Tensor>) -> Self {
        Self {
            weight_q,
            scale,
            bias,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let batch = input.shape()[0];
        let k = input.shape()[1];
        let n = self.scale.shape()[0];

        assert_eq!(k, self.weight_q.shape[1]);

        let input_f32 = input.to_f32();
        let mut result = Tensor::zeros(&[batch, n], DType::F32);
        let bytes_per_row = k.div_ceil(8);

        use rayon::prelude::*;

        for row in 0..batch {
            let in_row_offset = row * k;
            let out_row_offset = row * n;

            let out_slice = result.as_f32_slice_mut();
            let out_row = &mut out_slice[out_row_offset..out_row_offset + n];

            out_row
                .par_iter_mut()
                .enumerate()
                .for_each(|(col, out_elem)| {
                    let s = self.scale.get_flat_f32(col);
                    let in_f32 = input_f32.as_f32_slice();
                    let in_slice = &in_f32[in_row_offset..in_row_offset + k];
                    let packed_row = &self.weight_q.data[col * bytes_per_row..];
                    let acc = ternary_dot_f32(in_slice, packed_row, k);
                    *out_elem = acc * s;
                });
        }

        if let Some(ref b) = self.bias {
            result = result.add(b).unwrap();
        }
        result
    }
}

/// Dot product of f32 input against packed binary weights (+1/-1).
/// Packed format: 1 bit per element, 8 per byte, LSB first.
/// Bit set = +1, bit clear = -1.
fn ternary_dot_f32(input: &[f32], packed: &[u8], k: usize) -> f32 {
    let mut acc = 0.0f32;
    let mut t = 0;

    while t + 8 <= k {
        let byte = packed[t / 8];
        let mut signs = [0.0f32; 8];
        signs[0] = if byte & 0x01 != 0 { 1.0 } else { -1.0 };
        signs[1] = if byte & 0x02 != 0 { 1.0 } else { -1.0 };
        signs[2] = if byte & 0x04 != 0 { 1.0 } else { -1.0 };
        signs[3] = if byte & 0x08 != 0 { 1.0 } else { -1.0 };
        signs[4] = if byte & 0x10 != 0 { 1.0 } else { -1.0 };
        signs[5] = if byte & 0x20 != 0 { 1.0 } else { -1.0 };
        signs[6] = if byte & 0x40 != 0 { 1.0 } else { -1.0 };
        signs[7] = if byte & 0x80 != 0 { 1.0 } else { -1.0 };
        acc += simd::f32_dot(&input[t..t + 8], &signs);
        t += 8;
    }

    while t < k {
        let bit = (packed[t / 8] >> (t % 8)) & 1;
        let w = if bit == 1 { 1.0f32 } else { -1.0f32 };
        acc += input[t] * w;
        t += 1;
    }

    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanity_bitlinear() {
        let w = Tensor::from_slice(&[1.0, -1.0, 2.0, -2.0, 3.0, -3.0], &[3, 2]);
        let input = Tensor::from_slice(&[0.5, 0.5], &[1, 2]);
        let bl = BitLinear::quantize(&w, &QuantConfig::ternary());
        let out = bl.forward(&input);
        assert_eq!(out.shape(), &[1, 3]);
    }

    #[test]
    fn bitlinear_correctness() {
        let w = Tensor::from_slice(&[1.0, -1.0, 2.0, -2.0], &[2, 2]);
        let bl = BitLinear::quantize(&w, &QuantConfig::ternary());
        let input = Tensor::from_slice(&[1.0, 1.0], &[1, 2]);
        let out = bl.forward(&input);

        let scale0 = bl.scale.get_flat_f32(0);
        let scale1 = bl.scale.get_flat_f32(1);

        let out0 = out.get_flat_f32(0);
        let out1 = out.get_flat_f32(1);

        assert!(
            (out0 - scale0 + scale0).abs() < 0.01 || (out0 - (-scale0 + scale0)).abs() < 0.01,
            "row0 output should be +scale or -scale, got {}",
            out0
        );
        assert!(
            (out1 - scale1 + scale1).abs() < 0.01 || (out1 - (-scale1 + scale1)).abs() < 0.01,
            "row1 output should be +scale or -scale, got {}",
            out1
        );
    }

    #[test]
    fn bitlinear_per_channel_scales_differ() {
        let w = Tensor::from_slice(&[0.1, 0.2, 10.0, 20.0], &[2, 2]);
        let bl = BitLinear::quantize(&w, &QuantConfig::ternary());
        let s0 = bl.scale.get_flat_f32(0);
        let s1 = bl.scale.get_flat_f32(1);
        assert!(
            (s0 - 0.2).abs() < 0.01,
            "channel 0 scale should be ~0.2, got {}",
            s0
        );
        assert!(
            (s1 - 20.0).abs() < 0.01,
            "channel 1 scale should be ~20.0, got {}",
            s1
        );
    }

    #[test]
    fn bitlinear_batch() {
        let w = Tensor::from_slice(&[1.0, -1.0, -1.0, 1.0], &[2, 2]);
        let bl = BitLinear::quantize(&w, &QuantConfig::ternary());
        let input = Tensor::from_slice(&[1.0, 0.5, 0.5, 1.0], &[2, 2]);
        let out = bl.forward(&input);
        assert_eq!(out.shape(), &[2, 2]);
    }
}
