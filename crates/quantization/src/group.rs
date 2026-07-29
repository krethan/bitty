use crate::scheme::{QuantConfig, QuantizedTensor};
use bitllm_tensor::Tensor;

pub struct GroupQuantizer {
    pub group_size: usize,
}

impl GroupQuantizer {
    pub fn new(group_size: usize) -> Self {
        assert!(group_size > 0, "group_size must be > 0");
        Self { group_size }
    }

    pub fn quantize_int4(&self, tensor: &Tensor) -> QuantizedTensor {
        let src = tensor.to_f32();
        let n = src.num_elements();
        let num_groups = n.div_ceil(self.group_size);

        let mut scales = Vec::with_capacity(num_groups);
        for g in 0..num_groups {
            let start = g * self.group_size;
            let end = (start + self.group_size).min(n);
            let max_val = find_absmax_range(&src, start, end);
            scales.push(if max_val == 0.0 { 1.0 } else { max_val / 7.0 });
        }

        let packed_len = n.div_ceil(2);
        let mut packed = vec![0u8; packed_len];

        for i in 0..n {
            let group = i / self.group_size;
            let s = scales[group];
            let val = src.get_flat_f32(i);
            let q = (val / s).round().clamp(-8.0, 7.0) as i8;
            let nibble = (q & 0x0f) as u8;
            if i % 2 == 0 {
                packed[i / 2] = nibble;
            } else {
                packed[i / 2] |= nibble << 4;
            }
        }

        QuantizedTensor {
            data: packed,
            shape: tensor.shape().to_vec(),
            scales,
            zeros: None,
            config: QuantConfig::int4_group(self.group_size),
        }
    }

    pub fn dequantize_int4(&self, qtensor: &QuantizedTensor) -> Tensor {
        let n = qtensor.num_elements();
        let mut result = Tensor::new(&qtensor.shape, bitllm_tensor::DType::F32);

        if qtensor.scales.is_empty() {
            return result;
        }

        for i in 0..n {
            let group = i / self.group_size;
            let s = qtensor.scales[group.min(qtensor.scales.len() - 1)];
            let byte = qtensor.data[i / 2];
            let nibble = if i % 2 == 0 {
                byte & 0x0f
            } else {
                (byte >> 4) & 0x0f
            };
            let q = if nibble & 0x08 != 0 {
                nibble as i8 - 16
            } else {
                nibble as i8
            };
            result.set_flat_f32(i, q as f32 * s);
        }

        result
    }
}

fn find_absmax_range(tensor: &Tensor, start: usize, end: usize) -> f32 {
    let mut max_val = 0.0f32;
    for i in start..end {
        let v = tensor.get_flat_f32(i).abs();
        if v > max_val {
            max_val = v;
        }
    }
    max_val
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_int4_roundtrip() {
        let data = [
            0.5f32, -0.3, 0.8, -1.0, 0.1, -0.7, 0.2, 0.9, -0.4, 0.6, -0.1, 0.3, -0.8, 0.5, -0.2,
            0.7,
        ];
        let t = Tensor::from_slice(&data, &[16]);
        let qz = GroupQuantizer::new(8);
        let qt = qz.quantize_int4(&t);
        let reconstructed = qz.dequantize_int4(&qt);

        for i in 0..16 {
            let diff = (reconstructed.get_flat_f32(i) - data[i]).abs();
            assert!(
                diff < 0.4,
                "Group INT4 failed at {}: got {} expected {}",
                i,
                reconstructed.get_flat_f32(i),
                data[i]
            );
        }
    }

    #[test]
    fn test_group_int4_compression() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let t = Tensor::from_slice(&data, &[256]);
        let qz = GroupQuantizer::new(64);
        let qt = qz.quantize_int4(&t);
        assert!(
            qt.compression_ratio() > 3.0,
            "INT4 should compress > 3x, got {}",
            qt.compression_ratio()
        );
    }
}
