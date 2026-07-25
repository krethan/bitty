use crate::scheme::{QuantConfig, QuantizedTensor};
use crate::ternary::{ternary_dequantize, ternary_quantize};
use bitllm_tensor::Tensor;

pub fn absmax_quantize(tensor: &Tensor, config: &QuantConfig) -> QuantizedTensor {
    let f32_data = tensor.to_f32();
    let n = f32_data.num_elements();

    match config.bits {
        1 => {
            let mut qt = ternary_quantize(tensor);
            qt.config = config.clone();
            qt
        }
        8 => quantize_int8(&f32_data, n),
        4 => quantize_int4(&f32_data, n),
        _ => panic!("absmax does not support {} bits", config.bits),
    }
}

fn quantize_int8(f32_data: &Tensor, n: usize) -> QuantizedTensor {
    let mut scales = Vec::new();
    let mut quantized = Vec::with_capacity(n);
    let group_size = 256;

    for i in 0..n {
        if i % group_size == 0 {
            let mut block_max = 0.0f32;
            let block_end = (i + group_size).min(n);
            for j in i..block_end {
                let v = f32_data.get_flat_f32(j).abs();
                if v > block_max {
                    block_max = v;
                }
            }
            let s = if block_max == 0.0 {
                1.0
            } else {
                block_max / 127.0
            };
            scales.push(s);
        }
        let s = scales.last().cloned().unwrap_or(1.0);
        let val = f32_data.get_flat_f32(i);
        let q = (val / s).round().clamp(-127.0, 127.0) as i8;
        quantized.push(q as u8);
    }

    QuantizedTensor {
        data: quantized,
        shape: f32_data.shape().to_vec(),
        scales,
        zeros: None,
        config: QuantConfig::int8(),
    }
}

fn quantize_int4(f32_data: &Tensor, n: usize) -> QuantizedTensor {
    let mut scales = Vec::new();
    let mut packed = vec![0u8; n.div_ceil(2)];
    let group_size = 128;

    for i in 0..n {
        if i % group_size == 0 {
            let mut max_val = 0.0f32;
            let end = (i + group_size).min(n);
            for j in i..end {
                let v = f32_data.get_flat_f32(j).abs();
                if v > max_val {
                    max_val = v;
                }
            }
            let s = if max_val == 0.0 { 1.0 } else { max_val / 7.0 };
            scales.push(s);
        }
        let s = scales.last().cloned().unwrap_or(1.0);
        let val = f32_data.get_flat_f32(i);
        let q = (val / s).round().clamp(-7.0, 7.0) as i8;
        let byte = i / 2;
        let nib = if i % 2 == 0 {
            (q as u8) & 0x0F
        } else {
            ((q as u8) & 0x0F) << 4
        };
        packed[byte] |= nib;
    }

    QuantizedTensor {
        data: packed,
        shape: f32_data.shape().to_vec(),
        scales,
        zeros: None,
        config: QuantConfig::int4(),
    }
}

pub fn absmax_dequantize(qtensor: &QuantizedTensor) -> Tensor {
    match qtensor.config.bits {
        8 => dequantize_int8(qtensor),
        4 => dequantize_int4(qtensor),
        1 => ternary_dequantize(qtensor),
        _ => panic!(
            "absmax dequantize does not support {} bits",
            qtensor.config.bits
        ),
    }
}

fn dequantize_int8(qtensor: &QuantizedTensor) -> Tensor {
    let n = qtensor.num_elements();
    let mut result = Tensor::new(&qtensor.shape, bitllm_tensor::DType::F32);
    let group_size = 256;

    for i in 0..n {
        let group = i / group_size;
        let s = qtensor.scales[group.min(qtensor.scales.len() - 1)];
        let q = qtensor.data[i] as i8;
        result.set_flat_f32(i, q as f32 * s);
    }
    result
}

fn dequantize_int4(qtensor: &QuantizedTensor) -> Tensor {
    let n = qtensor.num_elements();
    let mut result = Tensor::new(&qtensor.shape, bitllm_tensor::DType::F32);
    let group_size = 128;

    for i in 0..n {
        let group = i / group_size;
        let s = qtensor.scales[group.min(qtensor.scales.len() - 1)];
        let byte = qtensor.data[i / 2];
        let nibble = if i % 2 == 0 {
            byte & 0x0F
        } else {
            (byte >> 4) & 0x0F
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int8_roundtrip() {
        let data = [100.0f32, -100.0, 50.0, -50.0, 25.0, -25.0];
        let t = Tensor::from_slice(&data, &[6]);
        let qt = absmax_quantize(&t, &QuantConfig::int8());
        let reconstructed = absmax_dequantize(&qt);
        for i in 0..6 {
            let diff = (reconstructed.get_flat_f32(i) - data[i]).abs();
            let rel_err = diff / data[i].abs().max(1.0);
            assert!(
                rel_err < 0.02,
                "INT8 failed at {}: got {} expected {} (rel_err={})",
                i,
                reconstructed.get_flat_f32(i),
                data[i],
                rel_err
            );
        }
    }

    #[test]
    fn test_int4_roundtrip() {
        let data = [6.0f32, -6.0, 2.0, -2.0, 1.0, -1.0];
        let t = Tensor::from_slice(&data, &[6]);
        let qt = absmax_quantize(&t, &QuantConfig::int4());
        let reconstructed = absmax_dequantize(&qt);
        for i in 0..6 {
            let diff = (reconstructed.get_flat_f32(i) - data[i]).abs();
            let scale = qt.scales[0];
            assert!(
                diff <= scale / 2.0 + 1e-6,
                "INT4 failed at {}: got {} expected {} (diff={}, scale={})",
                i,
                reconstructed.get_flat_f32(i),
                data[i],
                diff,
                scale
            );
        }
    }

    #[test]
    fn test_int8_compression() {
        let data: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) / 512.0).collect();
        let t = Tensor::from_slice(&data, &[1024]);
        let qt = absmax_quantize(&t, &QuantConfig::int8());
        assert!(
            qt.compression_ratio() > 2.0,
            "INT8 should compress > 2x, got {}",
            qt.compression_ratio()
        );
    }
}
