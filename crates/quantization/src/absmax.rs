use bitllm_tensor::{Tensor, DType};
use crate::scheme::{QuantConfig, QuantizedTensor};
use crate::ternary::{ternary_quantize, ternary_dequantize};

/// Quantization utility for absmax/ternary schemes

pub fn absmax_quantize(tensor: &Tensor, config: &QuantConfig) -> QuantizedTensor {
    let f32_data = tensor.to_f32();
    let n = f32_data.num_elements();

    match config.bits {
        1 => {
            let qt = ternary_quantize(tensor);
            qt
        }
        8 => quantize_int8(&f32_data, n),
        4 => quantize_int4(&f32_data, n),
        _ => panic!("Unsupported bits: {}", config.bits),
    }
}

fn quantize_int8(f32_data: &Tensor, n: usize) -> QuantizedTensor {
    let mut scales = Vec::with_capacity(n);
    let mut quantized = Vec::with_capacity(n);
    let group_size = 256;

    for i in 0..n {
        if i % group_size == 0 {
            let mut block_max = 0.0f32;
            let block_end = std::cmp::min(i + group_size, n);

            for j in i..block_end {
                let v = f32_data.get_flat_f32(j).abs();
                if v > block_max {
                    block_max = v;
                }
            }

            let s = if block_max == 0.0 { 1.0 } else { block_max / 127.0 };
            scales.push(s);
        }

        let s = scales[i / group_size];
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
    let mut scales = Vec::with_capacity(n);
    let mut packed = vec![0u8; (n + 1) / 2]; // 4 bits per byte

    for i in 0..n {
        if i % 128 == 0 {
            let mut block_max = 0.0f32;
            let end = std::cmp::min(i + 128, n);

            for j in i..end {
                let v = f32_data.get_flat_f32(j).abs();
                if v > block_max {
                    block_max = v;
                }
            }

            let s = if block_max == 0.0 { 1.0 } else { block_max / 7.0 };
            scales.push(s);
        }

        let s = scales[i / 128];
        let val = f32_data.get_flat_f32(i);
        let q = (val / s).round().clamp(-7.0, 7.0) as i8;

        let byte_idx = i / 2;
        let nibble = if i % 2 == 0 { q as u8 & 0x0F } else { (q as u8 & 0x0F) << 4 };
        packed[byte_idx] |= nibble;
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
        _ => panic!("Unsupported bits for dequantize: {}", qtensor.config.bits),
    }
}

fn dequantize_int8(qtensor: &QuantizedTensor) -> Tensor {
    let n = qtensor.num_elements();
    let mut result = Tensor::new(&qtensor.shape, DType::F32);
    let group_size = 256;

    for i in 0..n {
        let group = i / group_size;
        let s = qtensor.scales[group];
        let q = qtensor.data[i] as i8;
        result.set_flat_f32(i, q as f32 * s);
    }
    result
}

fn dequantize_int4(qtensor: &QuantizedTensor) -> Tensor {
    let n = qtensor.num_elements();
    let mut result = Tensor::new(&qtensor.shape, DType::F32);
    let group_size = 128;

    for i in 0..n {
        let group = i / group_size;
        let s = qtensor.scales[group];
        let byte_idx = i / 2;
        let nibble = if i % 2 == 0 { qtensor.data[byte_idx] & 0x0F } else { (qtensor.data[byte_idx] >> 4) & 0x0F };
        let q = if nibble & 0x08 != 0 { nibble as i8 - 16 } else { nibble as i8 };
        result.set_flat_f32(i, q as f32 * s);
    }
    result
}